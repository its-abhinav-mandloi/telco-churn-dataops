"""Prefect flow orchestrating ingestion -> preprocessing -> monitoring -> EDA -> training."""

from __future__ import annotations

import re
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from prefect import flow, get_run_logger, task
from prefect.artifacts import create_markdown_artifact, create_table_artifact
from prefect.cache_policies import NO_CACHE
from prefect.runtime import deployment as runtime_deployment
from prefect.runtime import flow_run as runtime_flow_run

from telco_churn_dataops import __version__
from telco_churn_dataops.config import Settings, get_settings
from telco_churn_dataops.eda import run_eda
from telco_churn_dataops.ingestion import DataValidationError, IngestionResult, ingest
from telco_churn_dataops.logging_utils import RunEventLog, isoformat, utc_now
from telco_churn_dataops.modeling import (
    build_model_metadata,
    compute_permutation_importance,
    persist_champion,
    plot_confusion_matrices,
    plot_feature_importance,
    train_and_compare,
    write_model_card,
)
from telco_churn_dataops.monitoring import (
    FAIL,
    drift_checks,
    monitoring_report,
    profile,
    quality_checks,
)
from telco_churn_dataops.preprocessing import (
    build_preprocessor,
    categorical_summary,
    clean,
    missing_value_table,
    numeric_summary,
    quality_report,
    split,
    transformed_preview,
)
from telco_churn_dataops.storage import (
    MANIFEST_NAME,
    apply_retention,
    latest_manifest,
    relative_path,
    run_dir,
    run_log_filename,
    write_json,
)

FLOW_NAME = "telco-churn-pipeline"


class PipelineQualityError(RuntimeError):
    """Raised when monitoring reports FAIL; the run is marked failed and no model is promoted."""


def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, requests.ConnectionError | requests.Timeout):
        return True
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        return exc.response.status_code >= 500 or exc.response.status_code == 429
    return False


def _retry_on_network_error(task: Any, task_run: Any, state: Any) -> bool:
    """Prefect retry condition: retry only transient network failures, not validation errors."""
    try:
        state.result()
    except Exception as exc:  # noqa: BLE001 - inspecting the failure type is the purpose
        return _is_transient(exc)
    return False


def new_run_id() -> str:
    """Globally unique, sortable, filesystem-safe run identifier.

    Combines a microsecond UTC timestamp with the full Prefect flow-run UUID
    (or a full UUID4 fallback outside a Prefect run) so runs started in the same
    second or sharing a UUID prefix never collide.
    """
    raw_id = str(runtime_flow_run.id or uuid.uuid4()).strip().lower()
    safe_id = re.sub(r"[^a-z0-9-]+", "-", raw_id).strip("-") or str(uuid.uuid4())
    return f"{utc_now():%Y%m%dT%H%M%S%fZ}-{safe_id}"


@task(
    name="ingest-dataset",
    retries=3,
    retry_delay_seconds=[5, 15, 30],
    retry_condition_fn=_retry_on_network_error,
    cache_policy=NO_CACHE,
)
def ingest_task(
    settings: Settings, run_id: str, events: RunEventLog
) -> tuple[pd.DataFrame, IngestionResult]:
    """Download and validate the public dataset and persist a raw copy with its checksum."""
    events.event("ingestion", "started", source_url=settings.dataset_url)
    df, result = ingest(
        settings.dataset_url,
        settings.raw_dir,
        run_id,
        timeout=settings.download_timeout_seconds,
        min_rows=settings.min_expected_rows,
    )
    events.event("ingestion", "completed", **result.to_dict())
    return df, result


@task(name="preprocess-data", cache_policy=NO_CACHE)
def preprocess_task(
    raw: pd.DataFrame, out_dir: Path, run_id: str, settings: Settings, events: RunEventLog
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean types, produce before/after quality reports and summary statistics."""
    events.event("preprocessing", "started")
    before = quality_report(raw)
    missing_value_table(raw).to_csv(out_dir / f"{run_id}_missing_values_before.csv")
    cleaned = clean(raw)
    after = quality_report(cleaned)
    missing_value_table(cleaned).to_csv(out_dir / f"{run_id}_missing_values_after_cleaning.csv")
    num = numeric_summary(cleaned)
    num.to_csv(out_dir / f"{run_id}_numeric_summary.csv")
    cat = categorical_summary(cleaned)
    cat.to_csv(out_dir / f"{run_id}_categorical_summary.csv")
    parts = split(cleaned, settings.test_size, settings.random_seed)
    preview = transformed_preview(build_preprocessor(), parts.x_train)
    report = {
        "before_cleaning": before,
        "after_cleaning": after,
        "total_charges_blank_converted_to_nan": int(cleaned["TotalCharges"].isna().sum()),
        "imputation_note": (
            "Remaining NaNs are imputed inside the sklearn Pipeline: numeric=median, "
            "categorical=most_frequent, fitted on the training split only."
        ),
        "numeric_summary": num.to_dict(orient="index"),
        "categorical_summary": cat.to_dict(orient="index"),
        "split": {
            "test_size": settings.test_size,
            "random_seed": settings.random_seed,
            "stratified_on": "Churn",
            "n_train": len(parts.x_train),
            "n_test": len(parts.x_test),
        },
        "transformation_evidence": preview,
    }
    path = write_json(out_dir / f"{run_id}_preprocessing_report.json", report)
    events.event(
        "preprocessing",
        "completed",
        rows_before=before["n_rows"],
        rows_after=after["n_rows"],
        duplicates_removed=before["n_rows"] - after["n_rows"],
        missing_before=before["missing_by_column"],
        missing_after_cleaning=after["missing_by_column"],
        report=relative_path(path),
    )
    return cleaned, report


@task(name="monitor-quality-and-drift", cache_policy=NO_CACHE)
def monitor_task(
    raw: pd.DataFrame,
    cleaned: pd.DataFrame,
    out_dir: Path,
    run_id: str,
    settings: Settings,
    events: RunEventLog,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run quality checks and drift checks against the previous successful run."""
    current = profile(cleaned)
    baseline = latest_manifest(settings, status="success")
    report = monitoring_report(
        quality_checks(raw, cleaned, settings.min_expected_rows),
        drift_checks(current, baseline.get("data_profile") if baseline else None),
        baseline.get("run_id") if baseline else None,
    )
    write_json(out_dir / f"{run_id}_monitoring_report.json", report)
    level = {"PASS": "INFO", "WARN": "WARNING", "FAIL": "ERROR"}[report["overall_status"]]
    events.event(
        "monitoring",
        report["overall_status"],
        level=level,
        quality_status=report["quality_status"],
        drift_status=report["drift_status"],
        baseline_run_id=report["baseline_run_id"],
        failing_checks=[
            c["check"]
            for c in report["quality_checks"] + report["drift_checks"]
            if c["status"] != "PASS"
        ],
    )
    return current, report


@task(name="exploratory-data-analysis", cache_policy=NO_CACHE)
def eda_task(cleaned: pd.DataFrame, out_dir: Path, run_id: str, events: RunEventLog) -> dict:
    """Generate EDA plots/tables (univariate, bivariate, correlation, Cramer's V, binning)."""
    events.event("eda", "started")
    summary = run_eda(cleaned, out_dir, run_id)
    events.event(
        "eda",
        "completed",
        n_plots=len(summary["plots"]),
        n_tables=len(summary["tables"]),
        churn_rate=summary["churn_rate"],
    )
    return summary


@task(name="train-evaluate-models", cache_policy=NO_CACHE)
def train_task(
    cleaned: pd.DataFrame,
    ingestion: IngestionResult,
    out_dir: Path,
    run_id: str,
    settings: Settings,
    events: RunEventLog,
) -> dict[str, Any]:
    """Compare candidate models, select the champion by ROC-AUC, persist model and model card."""
    events.event("training", "started", candidates=["logistic_regression", "random_forest"])
    parts = split(cleaned, settings.test_size, settings.random_seed)
    outcome = train_and_compare(parts, settings.random_seed)
    champion = outcome["champion"]
    model = outcome["fitted"][champion]
    fi = compute_permutation_importance(model, parts, settings.random_seed)
    fi.to_csv(out_dir / f"{run_id}_feature_importance.csv", index=False)
    meta = build_model_metadata(
        run_id=run_id,
        champion=champion,
        metrics=outcome["metrics"],
        dataset_sha256=ingestion.sha256,
        seed=settings.random_seed,
        test_size=settings.test_size,
        n_train=len(parts.x_train),
        n_test=len(parts.x_test),
        trained_at=isoformat(),
    )
    meta["feature_importance"] = {
        "method": "permutation importance (mean ROC-AUC drop over 5 repeats, test set)",
        "top": fi.head(10).to_dict(orient="records"),
    }
    paths = persist_champion(model, meta, out_dir / "model", settings.model_dir)
    paths["feature_importance_plot"] = plot_feature_importance(
        fi, out_dir / f"{run_id}_feature_importance.png", champion
    )
    paths["confusion_matrices"] = plot_confusion_matrices(
        outcome["metrics"], out_dir / f"{run_id}_confusion_matrices.png"
    )
    paths["model_card"] = write_model_card(
        out_dir / f"{run_id}_model_card.md", meta, fi["feature"].head(5).tolist()
    )
    write_json(out_dir / f"{run_id}_metrics.json", meta)
    events.event(
        "training",
        "completed",
        champion=champion,
        champion_metrics={
            k: v for k, v in outcome["metrics"][champion].items() if k != "confusion_matrix"
        },
        model_version=meta["model_version"],
    )
    return {"metadata": meta, "artifacts": paths}


def _publish_prefect_artifacts(manifest: dict[str, Any]) -> None:
    """Publish a run summary to the Prefect UI / Prefect Cloud *Artifacts* tab."""
    metrics = manifest.get("model", {}).get("metrics", {})
    if metrics:
        create_table_artifact(
            key="telco-churn-model-metrics",
            table=[
                {"model": name, **{k: v for k, v in m.items() if k != "confusion_matrix"}}
                for name, m in metrics.items()
            ],
            description=f"Model comparison for run {manifest['run_id']}",
        )
    mon = manifest.get("monitoring", {})
    create_markdown_artifact(
        key="telco-churn-run-summary",
        markdown=(
            f"# Telco churn run `{manifest['run_id']}`\n\n"
            f"* Status: **{manifest['status']}**  |  "
            f"Duration: {manifest.get('duration_seconds')} s\n"
            f"* Dataset rows: {manifest.get('ingestion', {}).get('n_rows')}  |  SHA-256: "
            f"`{manifest.get('ingestion', {}).get('sha256', '')[:16]}...`\n"
            f"* Monitoring: overall **{mon.get('overall_status')}** (quality "
            f"{mon.get('quality_status')}, drift {mon.get('drift_status')})\n"
            f"* Champion: **{manifest.get('model', {}).get('champion')}** "
            f"ROC-AUC {metrics.get(manifest.get('model', {}).get('champion'), {}).get('roc_auc')}\n"
        ),
        description="Telco churn DataOps run summary",
    )


@flow(name=FLOW_NAME, log_prints=True)
def telco_churn_pipeline(dataset_url: str | None = None) -> dict[str, Any]:
    """End-to-end churn DataOps pipeline; returns the persisted run manifest."""
    settings = get_settings()
    if dataset_url:
        settings = settings.model_copy(update={"dataset_url": dataset_url})
    logger = get_run_logger()
    run_id = new_run_id()
    out_dir = run_dir(settings, run_id)
    log_path = out_dir / run_log_filename(run_id)
    events = RunEventLog(log_path, run_id, logger)
    started = utc_now()
    t0 = time.perf_counter()
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "app_version": __version__,
        "status": "running",
        "started_at": isoformat(started),
        "finished_at": None,
        "duration_seconds": None,
        "prefect": {
            "flow_name": FLOW_NAME,
            "flow_run_id": runtime_flow_run.id,
            "flow_run_name": runtime_flow_run.name,
            "deployment_id": runtime_deployment.id,
            "scheduled_start_time": str(runtime_flow_run.scheduled_start_time),
        },
        "config": {
            "dataset_url": settings.dataset_url,
            "random_seed": settings.random_seed,
            "test_size": settings.test_size,
        },
        "artifact_dir": relative_path(out_dir),
        "run_log": relative_path(log_path),
        "errors": [],
    }
    manifest_path = out_dir / MANIFEST_NAME
    write_json(manifest_path, manifest)
    events.event("flow", "started", **manifest["prefect"])
    try:
        raw, ingestion = ingest_task(settings, run_id, events)
        manifest["ingestion"] = ingestion.to_dict()
        cleaned, prep = preprocess_task(raw, out_dir, run_id, settings, events)
        manifest["preprocessing"] = {
            "rows_before": prep["before_cleaning"]["n_rows"],
            "rows_after": prep["after_cleaning"]["n_rows"],
            "duplicate_rows": prep["before_cleaning"]["duplicate_rows"],
            "missing_before": prep["before_cleaning"]["missing_by_column"],
            "missing_after_cleaning": prep["after_cleaning"]["missing_by_column"],
            "total_charges_blank_converted_to_nan": prep["total_charges_blank_converted_to_nan"],
            "n_output_features": prep["transformation_evidence"]["n_output_features"],
            "split": prep["split"],
        }
        data_profile, monitoring = monitor_task(raw, cleaned, out_dir, run_id, settings, events)
        manifest["data_profile"] = data_profile
        manifest["monitoring"] = {
            k: monitoring[k]
            for k in ("overall_status", "quality_status", "drift_status", "baseline_run_id")
        }
        manifest["monitoring"]["checks"] = monitoring["quality_checks"] + monitoring["drift_checks"]
        write_json(manifest_path, manifest)
        if monitoring["overall_status"] == FAIL:
            raise PipelineQualityError(
                "Monitoring status FAIL; model not promoted. See monitoring report."
            )
        eda = eda_task(cleaned, out_dir, run_id, events)
        manifest["eda"] = {
            "churn_rate": eda["churn_rate"],
            "top_categorical_associations": eda["top_categorical_associations"],
            "numeric_target_correlation": eda["numeric_target_correlation"],
            "plots": eda["plots"],
            "tables": eda["tables"],
        }
        training = train_task(cleaned, ingestion, out_dir, run_id, settings, events)
        meta = training["metadata"]
        manifest["model"] = {
            "champion": meta["champion"],
            "model_version": meta["model_version"],
            "selection_metric": meta["selection_metric"],
            "metrics": meta["metrics"],
            "feature_importance": meta["feature_importance"],
            "artifacts": training["artifacts"],
        }
        manifest["status"] = "success"
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["errors"].append(
            {
                "type": type(exc).__name__,
                "message": str(exc)[:500],
                "traceback_tail": traceback.format_exc().splitlines()[-3:],
            }
        )
        events.event("flow", "failed", level="ERROR", error=f"{type(exc).__name__}: {exc}"[:500])
        raise
    finally:
        manifest["finished_at"] = isoformat()
        manifest["duration_seconds"] = round(time.perf_counter() - t0, 2)
        write_json(manifest_path, manifest)
        events.event(
            "flow",
            "finished",
            final_status=manifest["status"],
            duration_seconds=manifest["duration_seconds"],
            manifest=relative_path(manifest_path),
        )
        removed = apply_retention(settings)
        if removed:
            events.event("retention", "pruned", removed_runs=removed)
        try:
            _publish_prefect_artifacts(manifest)
        except Exception as art_exc:  # noqa: BLE001 - artifact publishing must not mask results
            logger.warning("Could not publish Prefect artifacts: %s", art_exc)
    return manifest


__all__ = ["DataValidationError", "PipelineQualityError", "telco_churn_pipeline"]


if __name__ == "__main__":
    result = telco_churn_pipeline()
    print(f"Run {result['run_id']} finished with status {result['status']}")
