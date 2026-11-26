"""End-to-end Prefect flow tests (synthetic data, temporary Prefect test server, no network)."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from prefect.testing.utilities import prefect_test_harness

from telco_churn_dataops import flows
from telco_churn_dataops.ingestion import DataValidationError
from telco_churn_dataops.storage import latest_manifest, read_json


@pytest.fixture(scope="module", autouse=True)
def prefect_harness():
    with prefect_test_harness(server_startup_timeout=120):
        yield


def _fake_fetch(content: bytes):
    return lambda url, timeout: (content, 200, "text/plain")


def test_flow_success_writes_manifest_log_and_model(tmp_settings, synthetic_raw):
    content = synthetic_raw.to_csv(index=False).encode()
    with (
        patch.object(flows, "get_settings", return_value=tmp_settings),
        patch("telco_churn_dataops.ingestion.fetch_bytes", _fake_fetch(content)),
    ):
        manifest = flows.telco_churn_pipeline()
        second = flows.telco_churn_pipeline()
    assert manifest["status"] == "success"
    assert manifest["monitoring"]["overall_status"] == "PASS"
    assert manifest["model"]["champion"] in {"logistic_regression", "random_forest"}
    assert (tmp_settings.model_dir / "champion.joblib").exists()
    run_dir = tmp_settings.runs_dir / manifest["run_id"]
    assert read_json(run_dir / "manifest.json")["status"] == "success"
    assert "ingestion" in (run_dir / "run_log.jsonl").read_text()
    assert second["monitoring"]["baseline_run_id"] == manifest["run_id"]


def test_flow_failure_is_recorded(tmp_settings):
    with (
        patch.object(flows, "get_settings", return_value=tmp_settings),
        patch("telco_churn_dataops.ingestion.fetch_bytes", _fake_fetch(b"a,b\n1,2\n")),
        pytest.raises(DataValidationError),
    ):
        flows.telco_churn_pipeline()
    manifest = latest_manifest(tmp_settings)
    assert manifest["status"] == "failed"
    assert manifest["errors"][0]["type"] == "DataValidationError"


def test_transient_error_classification():
    import requests

    assert flows._is_transient(requests.ConnectionError())
    assert flows._is_transient(requests.Timeout())
    assert not flows._is_transient(DataValidationError("bad"))
