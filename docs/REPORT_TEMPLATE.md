# AIMLCZG549 Assignment I — Project Report Template

> Export or copy this outline into `<groupid>.docx` / `<groupid>.pdf` for portal submission. Replace all placeholder fields with your actual group details and insert verified screenshots.

---

## Cover Page
* **Course:** AIMLCZG549 — API-driven Cloud Native Solutions
* **Assignment:** Assignment I (EC-1, 15 Marks)
* **Project Title:** Telco Churn DataOps
* **Group ID:** `<Enter Group ID>`
* **Submission Date:** `<Enter Date>`
* **Repository URL:** `https://github.com/its-abhinav-mandloi/telco-churn-dataops`
* **Demo Video Link:** `<Enter Google Drive URL after recording>`

---

## 1. Group Member Contributions

> Fill in actual student names, BITS IDs, and verified contributions. Do not leave placeholders.

| # | Student Name | BITS ID | Specific Responsibilities & Deliverables |
|---|---|---|---|
| 1 | `<Member 1 Name>` | `<ID>` | `<e.g., Data ingestion, preprocessing, EDA, model training>` |
| 2 | `<Member 2 Name>` | `<ID>` | `<e.g., Prefect 3 orchestration, 120s deployment, Cloud setup>` |
| 3 | `<Member 3 Name>` | `<ID>` | `<e.g., FastAPI service, Prefect API adapter, Postman & API tests>` |
| 4 | `<Member 4 Name>` | `<ID>` | `<e.g., Streamlit dashboard, monitoring/drift checks, report & video>` |

---

## 2. Business Understanding & Dataset (Sub-Objective 1.1 & 1.2)
* **Problem statement:** Predict telecom subscriber churn to guide targeted retention campaigns.
* **Dataset:** IBM Telco Customer Churn (`7,043` records, `21` columns, `26.54%` churn rate).
* **Ingestion controls:** Timeout, HTTP status & content-type checks, schema & row-count validation, SHA-256 provenance, atomic timestamped raw storage.
* **[Screenshot Slot 1]:** Ingestion log / Data quality tab showing row count, SHA-256, and source URL.

---

## 3. Data Pre-processing (Sub-Objective 1.3)
* Summary statistics, dtypes, duplicate detection (`0` duplicates), missing-value detection (`11` blank `TotalCharges` strings converted to `NaN`/float).
* Leakage-free `ColumnTransformer` fitted on the stratified training split (`80/20`, `seed=42`, `customerID` excluded): median imputation + `StandardScaler` for numeric features, most-frequent imputation + `OneHotEncoder` for categorical features.
* **[Screenshot Slot 2]:** Before/after quality report & numeric summary table.

---

## 4. Exploratory Data Analysis & Modeling (Sub-Objective 1.4)
* Univariate & bivariate charts: target distribution, numeric histograms/boxplots, categorical churn rates, Pearson correlation heatmap, bias-corrected Cramer's V, tenure bins (`0-12m`..`61-72m`), monthly-charge quartiles, and contract x tenure heatmap.
* Model comparison under the same preprocessing pipeline:
  * `LogisticRegression`: test ROC-AUC `0.8413`, 5-fold CV `0.8459 ± 0.0124`, Recall `0.7834` (Champion)
  * `RandomForestClassifier`: test ROC-AUC `0.8399`, 5-fold CV `0.8453 ± 0.0095`, Accuracy `0.7615`
* Feature importance via test-set permutation importance (ROC-AUC drop): top drivers are `tenure`, `InternetService`, and `Contract`.
* **[Screenshot Slot 3]:** EDA plots (correlation heatmap, Cramer's V, tenure-bin churn chart).
* **[Screenshot Slot 4]:** Model comparison table, confusion matrices, and permutation feature importance plot.

---

## 5. DataOps Automation, Scheduling & Monitoring (Sub-Objective 1.5)
* Named Prefect flow (`telco-churn-pipeline`) and deployment (`telco-churn-every-2-min`) with a `120`-second `IntervalSchedule`.
* Quality + drift monitoring against the prior successful run (`PASS` / `WARN` / `FAIL`) with documented thresholds; structured JSONL run logs and persisted manifests.
* **[Screenshot Slot 5]:** Prefect Cloud (or local Prefect UI — label runtime mode explicitly) showing the 120-second deployment, work pool, and consecutive scheduled flow runs.
* **[Screenshot Slot 6]:** Prefect flow-run logs and Prefect Artifacts (`telco-churn-model-metrics`).
* **[Screenshot Slot 7]:** Streamlit dashboard showing run history, drift/monitoring status, and Prefect UI link.

---

## 6. API Access & Built-in Orchestration APIs (Sub-Objective 2: 3.1, 3.2, 3.3)
* FastAPI `/api/v1` service integrating with Prefect's built-in `PrefectClient` (`GET /flows/{id}`, `GET /deployments/name/{flow}/{dep}`, `POST /flow_runs/filter`, `GET /work_pools/{name}`, `POST /work_pools/{name}/workers/filter`).
* At least four live details presented via `GET /api/v1/application-details`: flow metadata, deployment & 120s schedule, latest flow run state/timestamps, and work pool/workers.
* **[Screenshot Slot 8]:** FastAPI Swagger UI (`/docs`).
* **[Screenshot Slot 9]:** `GET /api/v1/application-details` (`200 OK`) showing the four+ Prefect details.
* **[Screenshot Slot 10]:** `POST /api/v1/predictions` (`200 OK`) and invalid payload (`422 Unprocessable Entity`).
* **[Screenshot Slot 11]:** Error handling (`503 Service Unavailable` when Prefect is unreachable or `404` before first run).

---

## 7. Verification, Limitations & Ethics
* Automated tests: `31` passing pytest tests + Ruff formatting/linting + GitHub Actions CI.
* Limitations & ethics: single static sample dataset (~7k rows), no temporal validation, no fairness audit across demographic attributes (`gender`, `SeniorCitizen`, `Partner`, `Dependents`) — decision-support only.

---

## 8. Submission Checklist

Verify every item below before uploading `<groupid>.docx` / `<groupid>.pdf` and sharing the demo video:

- [ ] **Group details:** Actual group ID, student names, BITS IDs, and individual contributions filled in (no placeholders).
- [ ] **Prefect Cloud deployment screenshot:** Real Prefect Cloud UI showing `telco-churn-pipeline/telco-churn-every-2-min` with the active **120-second schedule** and concurrency limit `1` (never label a local server screenshot as Prefect Cloud).
- [ ] **Consecutive scheduled runs:** Screenshot showing at least **two consecutive completed scheduled runs** (~2 minutes apart) in Prefect Cloud.
- [ ] **Prefect logs & artifacts:** Screenshot of structured task logs and Prefect artifacts (`telco-churn-model-metrics`, `telco-churn-run-summary`).
- [ ] **Streamlit dashboard:** Screenshot showing latest run state, data quality, monitoring/drift status, and model metrics.
- [ ] **Swagger UI:** Screenshot of FastAPI `/docs` listing `/api/v1` endpoints.
- [ ] **Built-in Prefect API (`200 OK`, Cloud mode):** Screenshot/output of `GET /api/v1/application-details` returning `200` with `"is_cloud": true` and flow, deployment/schedule, latest flow run, and work-pool/worker details.
- [ ] **API status-code coverage:** Documented request/response evidence for prediction `200`, invalid payload validation `422`, and Prefect-unavailable diagnostic `503`.
- [ ] **Final deliverables:** Exported `<groupid>.pdf` or `<groupid>.docx` uploaded to the portal and verified Google Drive demo-video link included.
