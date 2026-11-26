# API Testing & Documentation

This document provides reproducible `curl` commands, expected HTTP status codes, real trimmed responses captured against the local runtime, and screenshot references for **Sub-Objective 2 (API Access)**.

An importable Postman collection is also included at [postman/Telco-Churn-DataOps.postman_collection.json](../postman/Telco-Churn-DataOps.postman_collection.json). Interactive OpenAPI documentation is available at `http://127.0.0.1:8000/docs`.

---

## 1. Service Health — `GET /api/v1/health` (`200 OK`)

```bash
curl -i http://127.0.0.1:8000/api/v1/health
```

**Observed response (`HTTP 200`, local runtime):**
```json
{
  "status": "ok",
  "service": "telco-churn-dataops-api",
  "version": "1.0.0",
  "model_loaded": true,
  "latest_run_status": "success",
  "timestamp": "2026-11-26T16:25:55.416349+00:00"
}
```

---

## 2. Prefect Application Details — `GET /api/v1/application-details` (`200 OK`)

Requires a running Prefect server (`make prefect-server`, `make deploy-local`, `make worker`) or Prefect Cloud credentials (`PREFECT_API_URL`, `PREFECT_API_KEY`).

```bash
curl -i http://127.0.0.1:8000/api/v1/application-details
```

**Observed response (`HTTP 200`, local Prefect server at `http://127.0.0.1:4200/api`):**
```json
{
  "retrieved_at": "2026-11-26T16:28:02.535135+00:00",
  "prefect_api": {
    "url": "http://127.0.0.1:4200/api",
    "is_cloud": false
  },
  "flow": {
    "source": "PrefectClient.read_flow -> GET /flows/{id}",
    "id": "4631b8ca-1f32-4550-a24b-02b707f6925b",
    "name": "telco-churn-pipeline",
    "created": "2026-11-26T16:13:00.554097+00:00",
    "tags": []
  },
  "deployment": {
    "source": "PrefectClient.read_deployment_by_name -> GET /deployments/name/{flow_name}/{deployment_name}",
    "id": "71490689-7c8b-48d9-ab9d-a39c7d20f834",
    "name": "telco-churn-every-2-min",
    "entrypoint": "src/telco_churn_dataops/flows.py:telco_churn_pipeline",
    "paused": false,
    "status": "READY",
    "work_pool_name": "telco-process-pool",
    "work_queue_name": "default",
    "schedules": [
      {
        "id": "2a83d995-a1da-4029-9fac-ab2123ed4660",
        "active": true,
        "type": "IntervalSchedule",
        "interval_seconds": 120.0,
        "cron": null,
        "timezone": "UTC"
      }
    ]
  },
  "latest_flow_run": {
    "source": "PrefectClient.read_flow_runs -> POST /flow_runs/filter",
    "id": "01a218ac-6a09-7d9d-a88e-c5549c0fc5d9",
    "name": "refined-gibbon",
    "state_type": "COMPLETED",
    "state_name": "Completed",
    "start_time": "2026-11-26T16:26:38.532553+00:00",
    "end_time": "2026-11-26T16:26:42.644710+00:00",
    "total_run_time_seconds": 4.112157
  },
  "work_pool": {
    "source": "PrefectClient.read_work_pool -> GET /work_pools/{name}; read_workers_for_work_pool -> POST /work_pools/{name}/workers/filter",
    "id": "64ba749a-70a7-4614-bec2-3af34a919500",
    "name": "telco-process-pool",
    "type": "process",
    "status": "READY",
    "is_paused": false,
    "workers": [
      {
        "name": "ProcessWorker ebcc1677-c745-45d2-a5aa-d16126378ad8",
        "status": "ONLINE",
        "last_heartbeat_time": "2026-11-26T16:27:54.507819+00:00"
      }
    ]
  },
  "next_scheduled_run": {
    "name": "burrowing-dormouse",
    "expected_start_time": "2026-11-26T16:28:38.820968+00:00"
  }
}
```

---

## 3. Customer Churn Prediction — `POST /api/v1/predictions` (`200 OK`)

This service uses **`HTTP 200 OK`** for synchronous single-customer scoring (no server-side resource is created).

```bash
curl -i -X POST http://127.0.0.1:8000/api/v1/predictions \
  -H "Content-Type: application/json" \
  -d '{
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
    "TotalCharges": 191.4
  }'
```

**Observed response (`HTTP 200`, local runtime):**
```json
{
  "churn_probability": 0.8952,
  "predicted_label": "Churn",
  "decision_threshold": 0.5,
  "model_run_id": "20261126T162546Z-3d3ba4d1",
  "model_version": "logistic_regression-20261126T162546Z-3d3ba4d1",
  "model_name": "logistic_regression"
}
```

---

## 4. Invalid Prediction Payload — `POST /api/v1/predictions` (`422 Unprocessable Entity`)

```bash
curl -i -X POST http://127.0.0.1:8000/api/v1/predictions \
  -H "Content-Type: application/json" \
  -d '{"gender": "Female", "tenure": -5, "Contract": "Weekly"}'
```

**Observed response (`HTTP 422`, trimmed):**
```json
{
  "detail": [
    {
      "type": "greater_than_equal",
      "loc": ["body", "tenure"],
      "msg": "Input should be greater than or equal to 0",
      "input": -5
    },
    {
      "type": "literal_error",
      "loc": ["body", "Contract"],
      "msg": "Input should be 'Month-to-month', 'One year' or 'Two year'",
      "input": "Weekly"
    },
    {
      "type": "missing",
      "loc": ["body", "MonthlyCharges"],
      "msg": "Field required"
    }
  ]
}
```

---

## 5. Pipeline Manifest Unavailable — `GET /api/v1/runs/latest` (`404 Not Found`)

Before `make run-once` has executed (or when pointing `ARTIFACTS_DIR` to an empty folder):

```bash
curl -i http://127.0.0.1:8000/api/v1/runs/latest
```

**Expected response (`HTTP 404`, verified in `tests/test_api.py::test_latest_endpoints_404_before_any_run`):**
```json
{
  "detail": "No pipeline run manifest available yet. Run the pipeline first (`make run-once`) or wait for the scheduled Prefect deployment."
}
```

---

## 6. Prefect Unavailable Diagnostic — `GET /api/v1/application-details` (`503 Service Unavailable`)

When the Prefect server is stopped or `PREFECT_API_URL` is unreachable, the API never fabricates values and returns a transparent `503`:

```bash
curl -i http://127.0.0.1:8000/api/v1/application-details
```

**Observed response (`HTTP 503`, local Prefect server stopped):**
```json
{
  "detail": "Prefect API health check failed: ConnectError",
  "reason": "unreachable",
  "prefect_api_url": "http://127.0.0.1:4200/api",
  "checked_at": "2026-11-26T16:28:49.096342+00:00",
  "hint": "Start the Prefect server (`make prefect-server`) and create the deployment (`make deploy-local`), or set PREFECT_API_URL/PREFECT_API_KEY for Prefect Cloud."
}
```

---

## 7. Pipeline Metrics & EDA Endpoints (`200 OK`)

```bash
curl -s http://127.0.0.1:8000/api/v1/metrics/latest
curl -s http://127.0.0.1:8000/api/v1/eda/latest
```

---

## 8. Screenshot Locations

Screenshots are stored locally under `docs/screenshots/` (gitignored by `.gitignore` so binary captures do not bloat Git history):

| File | Runtime Mode | What It Shows |
|---|---|---|
| `docs/screenshots/prefect_deployment_local.png` | **Local Prefect Server** (`127.0.0.1:4200`) | Deployment `telco-churn-every-2-min`, 120s interval schedule, and completed runs |
| `docs/screenshots/dashboard_local.png` | **Local Streamlit** (`127.0.0.1:8501`) | Latest run status, duration, monitoring badge, data quality, and Prefect UI link |
| `docs/screenshots/swagger_api_local.png` | **Local FastAPI** (`127.0.0.1:8000/docs`) | OpenAPI / Swagger UI listing all `/api/v1` endpoints |
| `docs/screenshots/api_application_details_local.png` | **Local FastAPI + Local Prefect** | Live `200 OK` JSON from `GET /api/v1/application-details` |
| `docs/screenshots/api_error_503_local.png` | **Local FastAPI (Prefect stopped)** | Live `503 Service Unavailable` diagnostic response |

> **Note for submission:** Local screenshots are explicitly labeled as **Local Mode**. After connecting to **Prefect Cloud** ([docs/DEPLOYMENT.md](DEPLOYMENT.md)), capture your cloud workspace deployment/run logs and `GET /api/v1/application-details` (`"is_cloud": true`) for the cloud section of your report.
