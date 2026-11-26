.PHONY: install format lint test run-once run-api run-dashboard prefect-server deploy-local worker docker-up docker-down

PYTHON := .venv/bin/python
PREFECT := .venv/bin/prefect
UVICORN := .venv/bin/uvicorn
STREAMLIT := .venv/bin/streamlit
RUFF := .venv/bin/ruff
PYTEST := .venv/bin/pytest

API_HOST ?= 127.0.0.1
API_PORT ?= 8000
DASHBOARD_PORT ?= 8501
PREFECT_WORK_POOL ?= telco-process-pool
PREFECT_API_URL ?= http://127.0.0.1:4200/api

install:
	uv sync

format:
	$(RUFF) format src tests

lint:
	$(RUFF) format --check src tests
	$(RUFF) check src tests

test:
	$(PYTEST)

run-once:
	$(PYTHON) -m telco_churn_dataops.flows

run-api:
	PREFECT_API_URL=$(PREFECT_API_URL) $(UVICORN) telco_churn_dataops.api.main:app --host $(API_HOST) --port $(API_PORT)

run-dashboard:
	PREFECT_API_URL=$(PREFECT_API_URL) $(STREAMLIT) run src/telco_churn_dataops/dashboard/app.py --server.port $(DASHBOARD_PORT) --server.headless true

prefect-server:
	$(PREFECT) server start --host 127.0.0.1 --port 4200

deploy-local:
	PREFECT_API_URL=$(PREFECT_API_URL) $(PREFECT) work-pool create --type process $(PREFECT_WORK_POOL) --overwrite
	PREFECT_API_URL=$(PREFECT_API_URL) $(PREFECT) --no-prompt deploy --all

worker:
	PREFECT_API_URL=$(PREFECT_API_URL) $(PREFECT) worker start --pool $(PREFECT_WORK_POOL)

docker-up:
	docker compose up --build

docker-down:
	docker compose down
