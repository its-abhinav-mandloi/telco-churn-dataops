"""Unit tests for ingestion, preprocessing, EDA, monitoring and modeling (synthetic data only)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import requests

from telco_churn_dataops.eda import cramers_v, run_eda
from telco_churn_dataops.ingestion import DataValidationError, ingest, validate_csv_bytes
from telco_churn_dataops.modeling import evaluate, train_and_compare
from telco_churn_dataops.monitoring import FAIL, PASS, WARN, drift_checks, profile, quality_checks
from telco_churn_dataops.preprocessing import (
    FEATURES,
    build_preprocessor,
    clean,
    missing_value_table,
    split,
)
from telco_churn_dataops.storage import apply_retention, latest_manifest, write_json


def _response(content: bytes, status: int = 200, ctype: str = "text/plain") -> MagicMock:
    resp = MagicMock()
    resp.content = content
    resp.status_code = status
    resp.headers = {"Content-Type": f"{ctype}; charset=utf-8"}
    if status >= 400:
        resp.raise_for_status.side_effect = requests.HTTPError(response=resp)
    else:
        resp.raise_for_status.return_value = None
    return resp


def test_ingest_validates_and_records_provenance(synthetic_raw, tmp_path):
    content = synthetic_raw.to_csv(index=False).encode()
    with patch("telco_churn_dataops.ingestion.requests.get", return_value=_response(content)) as g:
        df, result = ingest("https://example.test/t.csv", tmp_path, "RUN1", min_rows=100)
    assert g.call_args.kwargs["timeout"] == 30.0
    assert result.n_rows == len(synthetic_raw) == len(df)
    assert result.n_columns == 21
    assert len(result.sha256) == 64
    assert (tmp_path / "telco_churn_RUN1.csv").read_bytes() == content


def test_ingest_retry_is_idempotent(synthetic_raw, tmp_path):
    content = synthetic_raw.to_csv(index=False).encode()
    with patch("telco_churn_dataops.ingestion.requests.get", return_value=_response(content)):
        ingest("https://x.test/a.csv", tmp_path, "RUN1", min_rows=100)
        ingest("https://x.test/a.csv", tmp_path, "RUN1", min_rows=100)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["telco_churn_RUN1.csv"]


def test_ingest_rejects_wrong_schema_and_http_errors(tmp_path):
    with pytest.raises(DataValidationError):
        validate_csv_bytes(b"a,b\n1,2\n", min_rows=1)
    with pytest.raises(DataValidationError):
        validate_csv_bytes(b"", min_rows=1)
    with (
        patch("telco_churn_dataops.ingestion.requests.get", return_value=_response(b"", 503)),
        pytest.raises(requests.HTTPError),
    ):
        ingest("https://x.test/a.csv", tmp_path, "RUN2")
    html = _response(b"<html></html>", ctype="text/html")
    with (
        patch("telco_churn_dataops.ingestion.requests.get", return_value=html),
        pytest.raises(DataValidationError),
    ):
        ingest("https://x.test/a.csv", tmp_path, "RUN3")


def test_clean_converts_blank_total_charges(synthetic_raw):
    n_blank = int((synthetic_raw["TotalCharges"].str.strip() == "").sum())
    assert n_blank > 0
    assert missing_value_table(synthetic_raw).loc["TotalCharges", "missing_count"] == n_blank
    cleaned = clean(synthetic_raw)
    assert cleaned["TotalCharges"].dtype.kind == "f"
    assert int(cleaned["TotalCharges"].isna().sum()) == n_blank
    assert set(cleaned["Churn"].unique()) <= {0, 1}


def test_split_is_stratified_excludes_id_and_preprocessor_has_no_leakage(synthetic_raw):
    data = split(clean(synthetic_raw), test_size=0.25, seed=42)
    assert "customerID" not in data.x_train.columns
    assert list(data.x_train.columns) == FEATURES
    assert abs(data.y_train.mean() - data.y_test.mean()) < 0.05
    pre = build_preprocessor().fit(data.x_train)
    expected_median = data.x_train["TotalCharges"].median()
    assert pre.named_transformers_["num"]["impute"].statistics_[3] == pytest.approx(expected_median)
    out = pre.transform(data.x_train)
    assert not np.isnan(out).any()
    assert out[:, 0:4].mean(axis=0) == pytest.approx(np.zeros(4), abs=1e-9)


def test_cramers_v_bounds(synthetic_raw):
    cleaned = clean(synthetic_raw)
    v = cramers_v(cleaned["Contract"], cleaned["Churn"])
    assert 0.0 <= v <= 1.0
    assert cramers_v(cleaned["Churn"], cleaned["Churn"]) == pytest.approx(1.0, abs=0.02)


def test_run_eda_writes_named_artifacts(synthetic_raw, tmp_path):
    summary = run_eda(clean(synthetic_raw), tmp_path, "RUNX")
    assert {"target_distribution", "correlation_heatmap", "binning_churn_rates"} <= set(
        summary["plots"]
    )
    assert all(p.name.startswith("RUNX_") for p in tmp_path.iterdir())
    assert (tmp_path / "RUNX_eda_summary.json").exists()
    assert set(summary["tenure_bins"]) == {"0-12m", "13-24m", "25-48m", "49-60m", "61-72m"}


def test_train_and_compare_selects_champion_by_roc_auc(synthetic_raw):
    data = split(clean(synthetic_raw), test_size=0.25, seed=42)
    out = train_and_compare(data, seed=42, cv_folds=3)
    assert set(out["metrics"]) == {"logistic_regression", "random_forest"}
    best = max(out["metrics"], key=lambda k: out["metrics"][k]["roc_auc"])
    assert out["champion"] == best
    metrics = evaluate(out["fitted"][best], data.x_test, data.y_test)
    assert {"roc_auc", "f1", "precision", "recall", "accuracy", "confusion_matrix"} <= set(metrics)


def test_monitoring_grades_drift_and_quality(synthetic_raw):
    cleaned = clean(synthetic_raw)
    current = profile(cleaned)
    assert drift_checks(current, None)[0]["status"] == PASS
    assert all(c["status"] == PASS for c in drift_checks(current, current))
    previous = {**current, "n_rows": int(current["n_rows"] * 1.3)}
    statuses = {c["check"]: c["status"] for c in drift_checks(current, previous)}
    assert statuses["row_count_change_pct"] == WARN
    previous = {**current, "n_rows": current["n_rows"] * 3}
    statuses = {c["check"]: c["status"] for c in drift_checks(current, previous)}
    assert statuses["row_count_change_pct"] == FAIL
    dup = cleaned.copy()
    dup.loc[1, "customerID"] = dup.loc[0, "customerID"]
    q = {c["check"]: c["status"] for c in quality_checks(synthetic_raw, dup, min_rows=100)}
    assert q["duplicate_customer_ids"] == FAIL


def test_manifest_discovery_and_retention(tmp_settings):
    for i, status in enumerate(["success", "failed", "success"]):
        run_id = f"2026010{i}T000000Z-abc"
        write_json(
            tmp_settings.runs_dir / run_id / "manifest.json",
            {"run_id": run_id, "status": status, "started_at": f"2026-01-0{i}"},
        )
    assert latest_manifest(tmp_settings)["run_id"] == "20260102T000000Z-abc"
    assert latest_manifest(tmp_settings, status="failed")["run_id"] == "20260101T000000Z-abc"
    removed = apply_retention(tmp_settings, keep=1)
    assert len(removed) == 2
