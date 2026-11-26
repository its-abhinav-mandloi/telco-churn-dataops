# Group Demo Script (5–7 Minutes)

> **Important:** Record your demo video **only after verifying** that the Prefect server/Cloud workspace, worker, FastAPI service, and Streamlit dashboard are all running cleanly and at least two 120-second runs have completed.

## Pre-Flight Checklist (Before Recording)
- [ ] `make lint` and `make test` pass.
- [ ] Prefect Cloud (or local Prefect server) is running, `telco-churn-every-2-min` is deployed with a 120s interval, and `prefect worker start --pool telco-process-pool` is polling.
- [ ] FastAPI (`http://127.0.0.1:8000/docs`) and Streamlit (`http://127.0.0.1:8501`) are open in browser tabs.
- [ ] No API keys or `.env` files are visible on screen.

---

## Demo Sequence

### 1. Business Problem & Architecture (0:00 – 1:00)
* **Show:** [README.md](../README.md) architecture diagram.
* **Explain:** Telecom churn prediction on the public IBM Telco dataset (`7,043` customers, `26.54%` churn) so retention teams can prioritize outreach. Walk through the end-to-end flow: Ingestion -> Preprocessing -> Monitoring -> EDA -> Model Training -> Prefect -> FastAPI + Streamlit.

### 2. Automated 2-Minute DataOps Workflow in Prefect (1:00 – 2:30)
* **Show:** Prefect Cloud (or local Prefect UI) **Deployments** page for `telco-churn-pipeline/telco-churn-every-2-min`.
* **Expected observation:** Active `120s` interval schedule, `telco-process-pool` work pool `Ready` with an online worker, and completed runs spaced 2 minutes apart.
* **Click into a completed run:** Show structured JSON task logs (ingestion SHA-256, `11` blank `TotalCharges` converted to `NaN`, monitoring `PASS`, champion selection) and the **Artifacts** tab (`telco-churn-model-metrics` table and run summary).

### 3. Custom Monitoring & EDA Dashboard (2:30 – 4:00)
* **Show:** Streamlit dashboard (`http://127.0.0.1:8501`).
* **Expected observation:**
  * Top metrics showing latest run status, duration, monitoring status (`PASS`), and Prefect flow-run name.
  * **Data quality** tab: 7,043 rows, 21 columns, SHA-256 provenance, before/after missingness.
  * **Monitoring & drift** tab: quality + drift checks against the previous successful run with documented thresholds.
  * **Model metrics** tab: `LogisticRegression` vs `RandomForestClassifier` comparison table, confusion matrices, and permutation feature importance (`tenure`, `InternetService`, `Contract` at top).
  * **EDA** tab: correlation heatmap, Cramer's V, tenure-bin churn rates, and contract x tenure heatmap.

### 4. Built-in Prefect API & FastAPI Endpoints (4:00 – 5:45)
* **Show:** Swagger UI (`http://127.0.0.1:8000/docs`) or Postman collection.
* **Execute & verify:**
  1. `GET /api/v1/health` -> `200 OK` (`model_loaded: true`).
  2. `GET /api/v1/application-details` -> `200 OK` returning live Prefect `flow`, `deployment` (with `interval_seconds: 120.0`), `latest_flow_run`, and `work_pool` / `workers`, each annotated with its Prefect REST endpoint and `retrieved_at`.
  3. `POST /api/v1/predictions` with a valid high-risk customer -> `200 OK` with `churn_probability`, `predicted_label`, `model_run_id`, and `model_version`.
  4. `POST /api/v1/predictions` with invalid values (`tenure: -5`, `Contract: "Weekly"`) -> `422 Unprocessable Entity`.

### 5. Automated Tests & Wrap-Up (5:45 – 6:30)
* **Show:** Terminal running `make lint && make test` (28 passing tests, zero Ruff errors) and briefly note the GitHub Actions CI workflow (`.github/workflows/ci.yml`).
* **Wrap up:** Summarize how both Sub-Objective 1 (10 marks) and Sub-Objective 2 (5 marks) are satisfied end-to-end.
