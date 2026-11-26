"""End-to-end Prefect flow tests (synthetic data, temporary Prefect test server, no network)."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from prefect.testing.utilities import prefect_test_harness

from telco_churn_dataops import flows
from telco_churn_dataops.ingestion import DataValidationError
from telco_churn_dataops.logging_utils import read_jsonl
from telco_churn_dataops.storage import latest_manifest, read_json, run_dir


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
    run_dir_path = tmp_settings.runs_dir / manifest["run_id"]
    assert read_json(run_dir_path / "manifest.json")["status"] == "success"
    assert "ingestion" in (run_dir_path / f"{manifest['run_id']}_run_log.jsonl").read_text()
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


def test_new_run_id_unique_for_same_timestamp_and_prefix(tmp_settings):
    fixed_time = datetime(2026, 11, 26, 16, 30, 0, 123456, tzinfo=UTC)
    flow_run_ids = [
        "12345678-aaaa-4111-8111-000000000001",
        "12345678-bbbb-4222-8222-000000000002",
        "12345678-cccc-4333-8333-000000000003",
    ]
    generated: list[str] = []
    dirs = []
    with patch.object(flows, "utc_now", return_value=fixed_time):
        for fr_id in flow_run_ids:
            with patch.object(flows, "runtime_flow_run", SimpleNamespace(id=fr_id)):
                rid = flows.new_run_id()
            assert rid.endswith(fr_id)
            assert rid.startswith("20261126T163000123456Z-")
            generated.append(rid)
            dirs.append(run_dir(tmp_settings, rid))
    assert len(set(generated)) == len(flow_run_ids)
    assert len({d.resolve() for d in dirs}) == len(flow_run_ids)


def test_new_run_id_fallback_uuid_unique_at_same_timestamp():
    fixed_time = datetime(2026, 11, 26, 16, 30, 0, 0, tzinfo=UTC)
    with (
        patch.object(flows, "utc_now", return_value=fixed_time),
        patch.object(flows, "runtime_flow_run", SimpleNamespace(id=None)),
    ):
        ids = {flows.new_run_id() for _ in range(5)}
    assert len(ids) == 5


def test_run_artifacts_and_metadata_match_individual_run(tmp_settings, synthetic_raw):
    fixed_time = datetime(2026, 11, 26, 16, 35, 0, 0, tzinfo=UTC)
    content = synthetic_raw.to_csv(index=False).encode()
    with (
        patch.object(flows, "get_settings", return_value=tmp_settings),
        patch.object(flows, "utc_now", return_value=fixed_time),
        patch("telco_churn_dataops.ingestion.fetch_bytes", _fake_fetch(content)),
    ):
        m1 = flows.telco_churn_pipeline()
        m2 = flows.telco_churn_pipeline()

    assert m1["run_id"] != m2["run_id"]
    assert m1["prefect"]["flow_run_id"] != m2["prefect"]["flow_run_id"]
    for manifest in (m1, m2):
        rid = manifest["run_id"]
        fr_id = manifest["prefect"]["flow_run_id"]
        assert fr_id is not None and rid.endswith(fr_id.lower())
        rdir = tmp_settings.runs_dir / rid
        assert rdir.is_dir()
        assert manifest["artifact_dir"].endswith(rid)
        assert manifest["ingestion"]["raw_path"].endswith(f"telco_churn_{rid}.csv")
        assert (tmp_settings.raw_dir / f"telco_churn_{rid}.csv").exists()
        assert manifest["run_log"].endswith(f"{rid}_run_log.jsonl")
        events = read_jsonl(rdir / f"{rid}_run_log.jsonl")
        assert events and all(e["run_id"] == rid for e in events)
        meta = read_json(rdir / "model" / "champion_metadata.json")
        assert meta["run_id"] == rid
        assert meta["model_version"] == f"{meta['champion']}-{rid}"
