# Deployment Guide (Prefect Cloud & Local)

No Prefect Cloud credentials, workspace URLs, or API keys are committed to this repository. Supply your own credentials via environment variables or a gitignored `.env` file.

---

## Part A: Prefect Cloud Setup (Cloud Mode)

### 1. Create a Prefect Cloud Account & API Key
1. Sign in at [https://app.prefect.cloud](https://app.prefect.cloud) and create a workspace (e.g., `aiml-assignment-1`).
2. Go to **Account Settings -> API Keys -> Create API Key**, name it (e.g., `telco-churn-local-worker`), and copy the key once.
3. Note your workspace API URL:
   `https://api.prefect.cloud/api/accounts/<ACCOUNT_ID>/workspaces/<WORKSPACE_ID>`

### 2. Configure Secure Environment Variables
Either log in via CLI:
```bash
.venv/bin/prefect cloud login --key "<YOUR_PREFECT_API_KEY>"
```
Or copy `.env.example` to `.env` (which is gitignored) and set:
```dotenv
PREFECT_API_URL=https://api.prefect.cloud/api/accounts/<ACCOUNT_ID>/workspaces/<WORKSPACE_ID>
PREFECT_API_KEY=<YOUR_PREFECT_API_KEY>
PREFECT_WORK_POOL=telco-process-pool
```
Never paste API keys into `prefect.yaml`, `docker-compose.yml`, screenshots, or Git commits.

### 3. Create the Work Pool and Deploy the 2-Minute Schedule
```bash
set -a && source .env && set +a

# Create a process work pool in Prefect Cloud
.venv/bin/prefect work-pool create --type process "$PREFECT_WORK_POOL" --overwrite

# Register the 120-second deployment from prefect.yaml
.venv/bin/prefect --no-prompt deploy --all

# Verify deployment registration
.venv/bin/prefect deployment ls
```

### 4. Start the Worker
```bash
set -a && source .env && set +a
.venv/bin/prefect worker start --pool "$PREFECT_WORK_POOL"
```
Keep this terminal open while demonstrating the scheduled runs.

### 5. Verify the 2-Minute Schedule, Concurrency Limit & Cloud Dashboard
1. Open [https://app.prefect.cloud](https://app.prefect.cloud) -> **Deployments** -> `telco-churn-every-2-min`.
2. Confirm the **Schedule** shows `Every 2 minutes` (`120` seconds, active), **Concurrency Limit** is `1` (`ENQUEUE`), and **Work Pool** shows `telco-process-pool` with status `Ready`.
   - Because the concurrency limit is `1`, only one pipeline run executes at once. If the worker is stopped or offline for several minutes, pending scheduled runs are queued and processed sequentially once the worker reconnects (each run also gets a globally unique `run_id` containing the full Prefect flow-run UUID so artifacts never overwrite each other).
3. Open **Flow Runs** and observe two consecutive scheduled runs completing ~120 seconds apart.
4. Click a completed run to inspect **Logs** (structured JSON lines for ingestion, preprocessing, monitoring, EDA, training) and **Artifacts** (`telco-churn-model-metrics` table and `telco-churn-run-summary` markdown).
5. Start FastAPI with the same `.env` (`make run-api`) and call:
   ```bash
   curl -s http://127.0.0.1:8000/api/v1/application-details
   ```
   Confirm `"is_cloud": true` and that flow, deployment/schedule, concurrency limit, latest flow run, and work-pool/worker details match Prefect Cloud.

### 6. Stop the Demo (Avoid Unnecessary Runs)
After recording your video and screenshots:
1. Pause the deployment schedule so Prefect Cloud stops queuing 2-minute runs:
   ```bash
   set -a && source .env && set +a
   # Find the schedule ID (or read it from GET /api/v1/application-details -> deployment.schedules[0].id)
   .venv/bin/prefect deployment schedule ls telco-churn-pipeline/telco-churn-every-2-min
   .venv/bin/prefect deployment schedule pause telco-churn-pipeline/telco-churn-every-2-min <SCHEDULE_ID>
   # Or pause all schedules in the workspace:
   # .venv/bin/prefect deployment schedule pause --all
   ```
2. Stop the worker with `Ctrl+C`.
   *(To resume later for a live re-demo, run `.venv/bin/prefect deployment schedule resume telco-churn-pipeline/telco-churn-every-2-min <SCHEDULE_ID>`.)*

---

## Part B: Local Server Mode (No Cloud Account Required)

```bash
# Terminal 1
make prefect-server

# Terminal 2
make deploy-local
make worker

# Terminal 3
make run-api

# Terminal 4
make run-dashboard
```
Open:
* Prefect UI: `http://127.0.0.1:4200`
* FastAPI Swagger UI: `http://127.0.0.1:8000/docs`
* Streamlit Dashboard: `http://127.0.0.1:8501`
