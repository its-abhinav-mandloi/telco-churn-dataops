"""Data-quality and drift monitoring against the previous successful run."""

from __future__ import annotations

from typing import Any

import pandas as pd

from telco_churn_dataops.ingestion import EXPECTED_COLUMNS
from telco_churn_dataops.preprocessing import ID_COLUMN, NUMERIC_FEATURES, TARGET_COLUMN

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
_SEVERITY = {PASS: 0, WARN: 1, FAIL: 2}

# Documented thresholds (also rendered in README / dashboard).
THRESHOLDS: dict[str, dict[str, float]] = {
    "row_count_change_pct": {"warn": 10.0, "fail": 50.0},
    "missing_pct_change_pp": {"warn": 1.0, "fail": 5.0},
    "numeric_mean_shift_std": {"warn": 0.10, "fail": 0.25},
    "churn_rate_change_pp": {"warn": 5.0, "fail": 10.0},
    "max_missing_pct_per_column": {"warn": 1.0, "fail": 5.0},
}


def _grade(value: float, key: str) -> str:
    t = THRESHOLDS[key]
    if value > t["fail"]:
        return FAIL
    if value > t["warn"]:
        return WARN
    return PASS


def worst(statuses: list[str]) -> str:
    """Return the most severe status in ``statuses`` (PASS if empty)."""
    return max(statuses, key=_SEVERITY.__getitem__, default=PASS)


def profile(df: pd.DataFrame) -> dict[str, Any]:
    """Compact statistical profile of a cleaned frame, stored in the manifest as a baseline."""
    missing_pct = (df.isna().mean() * 100).round(4)
    return {
        "n_rows": int(len(df)),
        "missing_pct_total": round(float(df.isna().to_numpy().mean() * 100), 4),
        "missing_pct_by_column": {k: float(v) for k, v in missing_pct.items() if v > 0},
        "numeric": {
            c: {
                "mean": round(float(df[c].mean()), 4),
                "std": round(float(df[c].std()), 4),
                "min": round(float(df[c].min()), 4),
                "max": round(float(df[c].max()), 4),
            }
            for c in NUMERIC_FEATURES
        },
        "churn_rate": round(float(df[TARGET_COLUMN].mean()), 4),
    }


def quality_checks(raw: pd.DataFrame, clean_df: pd.DataFrame, min_rows: int) -> list[dict]:
    """Run absolute data-quality checks on the current batch."""
    checks: list[dict[str, Any]] = []
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in raw.columns]
    checks.append(
        {
            "check": "schema_columns_present",
            "status": FAIL if missing_cols else PASS,
            "value": missing_cols,
            "threshold": "all expected columns",
        }
    )
    checks.append(
        {
            "check": "min_row_count",
            "status": FAIL if len(clean_df) < min_rows else PASS,
            "value": int(len(clean_df)),
            "threshold": f">= {min_rows}",
        }
    )
    dup_ids = int(clean_df[ID_COLUMN].duplicated().sum())
    checks.append(
        {
            "check": "duplicate_customer_ids",
            "status": FAIL if dup_ids else PASS,
            "value": dup_ids,
            "threshold": "0",
        }
    )
    max_missing = float((clean_df.isna().mean() * 100).max())
    checks.append(
        {
            "check": "max_missing_pct_per_column",
            "status": _grade(max_missing, "max_missing_pct_per_column"),
            "value": round(max_missing, 4),
            "threshold": THRESHOLDS["max_missing_pct_per_column"],
        }
    )
    target_values = sorted(int(v) for v in clean_df[TARGET_COLUMN].unique())
    checks.append(
        {
            "check": "binary_target_both_classes",
            "status": PASS if target_values == [0, 1] else FAIL,
            "value": target_values,
            "threshold": "[0, 1]",
        }
    )
    return checks


def drift_checks(current: dict[str, Any], previous: dict[str, Any] | None) -> list[dict]:
    """Compare the current profile to the previous successful run's profile."""
    if not previous:
        return [
            {
                "check": "baseline_available",
                "status": PASS,
                "value": "no previous successful run; this run becomes the baseline",
                "threshold": None,
            }
        ]
    checks: list[dict[str, Any]] = []
    prev_rows = max(previous["n_rows"], 1)
    row_change = abs(current["n_rows"] - previous["n_rows"]) / prev_rows * 100
    checks.append(
        {
            "check": "row_count_change_pct",
            "status": _grade(row_change, "row_count_change_pct"),
            "value": round(row_change, 4),
            "previous": previous["n_rows"],
            "current": current["n_rows"],
            "threshold": THRESHOLDS["row_count_change_pct"],
        }
    )
    miss_change = abs(current["missing_pct_total"] - previous["missing_pct_total"])
    checks.append(
        {
            "check": "missing_pct_change_pp",
            "status": _grade(miss_change, "missing_pct_change_pp"),
            "value": round(miss_change, 4),
            "previous": previous["missing_pct_total"],
            "current": current["missing_pct_total"],
            "threshold": THRESHOLDS["missing_pct_change_pp"],
        }
    )
    for col, stats in current["numeric"].items():
        prev = previous.get("numeric", {}).get(col)
        if not prev:
            continue
        shift = abs(stats["mean"] - prev["mean"]) / (prev["std"] or 1.0)
        checks.append(
            {
                "check": f"numeric_mean_shift_std:{col}",
                "status": _grade(shift, "numeric_mean_shift_std"),
                "value": round(shift, 4),
                "previous": prev["mean"],
                "current": stats["mean"],
                "threshold": THRESHOLDS["numeric_mean_shift_std"],
            }
        )
    churn_change = abs(current["churn_rate"] - previous["churn_rate"]) * 100
    checks.append(
        {
            "check": "churn_rate_change_pp",
            "status": _grade(churn_change, "churn_rate_change_pp"),
            "value": round(churn_change, 4),
            "previous": previous["churn_rate"],
            "current": current["churn_rate"],
            "threshold": THRESHOLDS["churn_rate_change_pp"],
        }
    )
    return checks


def monitoring_report(
    quality: list[dict[str, Any]],
    drift: list[dict[str, Any]],
    baseline_run_id: str | None,
) -> dict[str, Any]:
    """Combine quality and drift checks into one report with an overall status."""
    return {
        "overall_status": worst([c["status"] for c in quality + drift]),
        "quality_status": worst([c["status"] for c in quality]),
        "drift_status": worst([c["status"] for c in drift]),
        "baseline_run_id": baseline_run_id,
        "quality_checks": quality,
        "drift_checks": drift,
        "thresholds": THRESHOLDS,
    }
