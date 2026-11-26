"""FastAPI tests with a mocked Prefect client and isolated (synthetic) artifacts."""

from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Any
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from telco_churn_dataops.api.main import create_app, get_prefect_provider
from telco_churn_dataops.config import get_settings
from telco_churn_dataops.modeling import build_model_metadata, persist_champion, train_and_compare
from telco_churn_dataops.prefect_adapter import PrefectUnavailableError
from telco_churn_dataops.preprocessing import clean, split
from telco_churn_dataops.storage import write_json

DEPLOYMENT_ID = UUID("11111111-1111-1111-1111-111111111111")
FLOW_ID = UUID("22222222-2222-2222-2222-222222222222")

VALID_CUSTOMER: dict[str, Any] = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 2,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 95.7,
    "TotalCharges": 191.4,
}

FAKE_DETAILS = {
    "retrieved_at": "2026-11-26T16:00:00+00:00",
    "prefect_api": {"url": "http://prefect.test/api", "is_cloud": False},
    "flow": {"source": "PrefectClient.read_flow", "id": "f-1", "name": "telco-churn-pipeline"},
    "deployment": {
        "source": "PrefectClient.read_deployment_by_name",
        "id": "d-1",
        "name": "telco-churn-every-2-min",
        "schedules": [{"interval_seconds": 120.0, "active": True}],
    },
    "latest_flow_run": {"source": "PrefectClient.read_flow_runs", "state_name": "Completed"},
    "work_pool": {"source": "PrefectClient.read_work_pool", "name": "telco-process-pool"},
}


class FakeProvider:
    def __init__(self, error: Exception | None = None):
        self.error = error

    async def get_application_details(self) -> dict[str, Any]:
        if self.error:
            raise self.error
        return FAKE_DETAILS


@pytest.fixture()
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture()
def client_factory(tmp_settings):
    def _make(provider: Any = None) -> TestClient:
        app = create_app(tmp_settings)
        app.dependency_overrides[get_settings] = lambda: tmp_settings
        if provider is not None:
            app.dependency_overrides[get_prefect_provider] = lambda: provider
        return TestClient(app)

    return _make


@pytest.fixture()
def trained(tmp_settings, synthetic_raw):
    data = split(clean(synthetic_raw), test_size=0.25, seed=42)
    out = train_and_compare(data, seed=42, cv_folds=3)
    champ = out["champion"]
    meta = build_model_metadata("RUN-T", champ, out["metrics"], "abc", 42, 0.25, 300, 100, "now")
    persist_champion(
        out["fitted"][champ],
        meta,
        tmp_settings.runs_dir / "RUN-T" / "model",
        tmp_settings.model_dir,
    )
    write_json(
        tmp_settings.runs_dir / "RUN-T" / "manifest.json",
        {
            "run_id": "RUN-T",
            "status": "success",
            "started_at": "2026-11-26T16:00:00+00:00",
            "model": {
                "champion": champ,
                "model_version": meta["model_version"],
                "selection_metric": "test roc_auc",
                "metrics": out["metrics"],
                "feature_importance": {"method": "permutation", "top": []},
            },
            "eda": {
                "churn_rate": 0.3,
                "top_categorical_associations": {"Contract": 0.3},
                "numeric_target_correlation": {"tenure": -0.3},
                "plots": {},
                "tables": {},
            },
        },
    )
    return meta


def test_health(client_factory):
    resp = client_factory().get("/api/v1/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok" and body["version"] == "1.0.0"
    assert body["model_loaded"] is False


def test_latest_endpoints_404_before_any_run(client_factory):
    client = client_factory()
    for path in ("/api/v1/runs/latest", "/api/v1/metrics/latest", "/api/v1/eda/latest"):
        resp = client.get(path)
        assert resp.status_code == 404
        assert "run the pipeline" in resp.json()["detail"].lower()


def test_prediction_503_before_model(client_factory):
    resp = client_factory().post("/api/v1/predictions", json=VALID_CUSTOMER)
    assert resp.status_code == 503


def test_prediction_success(client_factory, trained):
    resp = client_factory().post("/api/v1/predictions", json=VALID_CUSTOMER)
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["predicted_label"] in {"Churn", "No churn"}
    assert body["model_run_id"] == "RUN-T"
    assert body["model_version"] == trained["model_version"]


@pytest.mark.parametrize(
    "patch",
    [
        {"Contract": "Weekly"},
        {"tenure": -1},
        {"MonthlyCharges": "lots"},
        {"unexpected_field": 1},
    ],
)
def test_prediction_validation_422(client_factory, patch):
    resp = client_factory().post("/api/v1/predictions", json={**VALID_CUSTOMER, **patch})
    assert resp.status_code == 422


def test_prediction_missing_field_422(client_factory):
    payload = {k: v for k, v in VALID_CUSTOMER.items() if k != "Contract"}
    assert client_factory().post("/api/v1/predictions", json=payload).status_code == 422


def test_latest_endpoints_after_run(client_factory, trained):
    client = client_factory()
    assert client.get("/api/v1/runs/latest").json()["run_id"] == "RUN-T"
    metrics = client.get("/api/v1/metrics/latest")
    assert metrics.status_code == 200 and metrics.json()["champion"] == trained["champion"]
    assert client.get("/api/v1/eda/latest").json()["churn_rate"] == 0.3
    assert client.get("/api/v1/runs").json()[0]["status"] == "success"


def test_application_details_success(client_factory):
    resp = client_factory(FakeProvider()).get("/api/v1/application-details")
    assert resp.status_code == 200
    body = resp.json()
    assert body["flow"]["name"] == "telco-churn-pipeline"
    assert body["deployment"]["schedules"][0]["interval_seconds"] == 120.0
    assert body["latest_flow_run"]["state_name"] == "Completed"
    assert body["work_pool"]["name"] == "telco-process-pool"


def test_application_details_unavailable(client_factory):
    err = PrefectUnavailableError("boom", reason="unreachable", api_url="http://prefect.test/api")
    resp = client_factory(FakeProvider(err)).get("/api/v1/application-details")
    assert resp.status_code == 503
    body = resp.json()
    assert body["reason"] == "unreachable" and "hint" in body


def test_application_details_not_configured_uses_real_adapter(client_factory):
    resp = client_factory().get("/api/v1/application-details")
    assert resp.status_code == 503
    assert resp.json()["reason"] == "not_configured"


class _FakePrefectClient:
    """Stands in for prefect.client.orchestration.PrefectClient."""

    def __init__(self, healthy: bool = True):
        self.healthy = healthy

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def api_healthcheck(self):
        return None if self.healthy else ConnectionError("refused")

    async def read_deployment_by_name(self, name):
        assert name == "telco-churn-pipeline/telco-churn-every-2-min"
        sched = SimpleNamespace(
            interval=timedelta(seconds=120),
            anchor_date=None,
            timezone="UTC",
        )
        return SimpleNamespace(
            id=DEPLOYMENT_ID,
            flow_id=FLOW_ID,
            name="telco-churn-every-2-min",
            entrypoint="src/telco_churn_dataops/flows.py:telco_churn_pipeline",
            paused=False,
            status="READY",
            work_pool_name="telco-process-pool",
            work_queue_name="default",
            schedules=[SimpleNamespace(id="s-1", active=True, schedule=sched)],
            updated=None,
        )

    async def read_flow(self, flow_id):
        return SimpleNamespace(id=flow_id, name="telco-churn-pipeline", created=None, tags=[])

    async def read_flow_runs(self, **kwargs):
        return [
            SimpleNamespace(
                id="r-2",
                name="scheduled-run",
                state_type=None,
                state_name="Scheduled",
                expected_start_time=None,
                start_time=None,
                end_time=None,
                total_run_time=timedelta(0),
            ),
            SimpleNamespace(
                id="r-1",
                name="happy-run",
                state_type=SimpleNamespace(value="COMPLETED"),
                state_name="Completed",
                expected_start_time=None,
                start_time=datetime(2026, 1, 1),
                end_time=None,
                total_run_time=timedelta(seconds=7),
            ),
        ]

    async def read_work_pool(self, name):
        return SimpleNamespace(
            id="wp-1",
            name=name,
            type="process",
            status="READY",
            is_paused=False,
            concurrency_limit=None,
        )

    async def read_workers_for_work_pool(self, name, limit=None):
        return [SimpleNamespace(name="worker-1", status="ONLINE", last_heartbeat_time=None)]


@pytest.mark.anyio
async def test_adapter_maps_prefect_objects(tmp_settings):
    from telco_churn_dataops.prefect_adapter import PrefectApiAdapter

    settings = tmp_settings.model_copy(update={"prefect_api_url": "http://prefect.test/api"})
    adapter = PrefectApiAdapter(settings, client_factory=lambda url, key: _FakePrefectClient())
    details = await adapter.get_application_details()
    assert details["deployment"]["schedules"][0]["interval_seconds"] == 120.0
    assert details["latest_flow_run"]["name"] == "happy-run"
    assert details["next_scheduled_run"]["name"] == "scheduled-run"
    assert details["work_pool"]["workers"][0]["name"] == "worker-1"
    assert "GET /deployments/name" in details["deployment"]["source"]


@pytest.mark.anyio
async def test_adapter_unhealthy_raises(tmp_settings):
    from telco_churn_dataops.prefect_adapter import PrefectApiAdapter

    settings = tmp_settings.model_copy(update={"prefect_api_url": "http://prefect.test/api"})
    adapter = PrefectApiAdapter(settings, client_factory=lambda u, k: _FakePrefectClient(False))
    with pytest.raises(PrefectUnavailableError) as info:
        await adapter.get_application_details()
    assert info.value.reason == "unreachable"
