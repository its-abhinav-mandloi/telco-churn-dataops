"""FastAPI service exposing predictions, pipeline results and Prefect application details."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from telco_churn_dataops import __version__
from telco_churn_dataops.api.schemas import (
    ApplicationDetailsResponse,
    CustomerFeatures,
    EdaResponse,
    ErrorResponse,
    HealthResponse,
    MetricsResponse,
    PredictionResponse,
    PrefectUnavailableResponse,
    RunManifestResponse,
    RunSummary,
)
from telco_churn_dataops.config import Settings, get_settings
from telco_churn_dataops.logging_utils import get_logger, isoformat
from telco_churn_dataops.modeling import METADATA_FILENAME, load_serving_model
from telco_churn_dataops.prefect_adapter import (
    PrefectApiAdapter,
    PrefectDetailsProvider,
    PrefectUnavailableError,
)
from telco_churn_dataops.preprocessing import FEATURES
from telco_churn_dataops.storage import latest_manifest, list_manifests

API_PREFIX = "/api/v1"
log = get_logger("telco_churn_dataops.api")


class ModelRegistry:
    """Lazily loads the promoted champion and reloads it when a newer model is promoted."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._mtime: float | None = None
        self._loaded: tuple[Any, dict[str, Any]] | None = None

    def get(self, model_dir: Path) -> tuple[Any, dict[str, Any]] | None:
        """Return ``(pipeline, metadata)`` or ``None`` when no model has been trained yet."""
        meta_path = model_dir / METADATA_FILENAME
        if not meta_path.exists():
            return None
        mtime = meta_path.stat().st_mtime
        with self._lock:
            if self._loaded is None or mtime != self._mtime:
                self._loaded = load_serving_model(model_dir)
                self._mtime = mtime
            return self._loaded


registry = ModelRegistry()


def get_prefect_provider(
    settings: Annotated[Settings, Depends(get_settings)],
) -> PrefectDetailsProvider:
    """Dependency returning the Prefect API adapter (overridden with a mock in tests)."""
    return PrefectApiAdapter(settings)


SettingsDep = Annotated[Settings, Depends(get_settings)]
router = APIRouter(prefix=API_PREFIX)
NOT_FOUND = {404: {"model": ErrorResponse, "description": "No pipeline run available yet"}}


def _not_found(what: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"No {what} available yet. Run the pipeline first (`make run-once`) "
        "or wait for the scheduled Prefect deployment.",
    )


@router.get("/health", response_model=HealthResponse, tags=["service"])
def health(settings: SettingsDep) -> HealthResponse:
    """Liveness/health information and service version."""
    latest = latest_manifest(settings)
    return HealthResponse(
        status="ok",
        service="telco-churn-dataops-api",
        version=__version__,
        model_loaded=(settings.model_dir / METADATA_FILENAME).exists(),
        latest_run_status=latest.get("status") if latest else None,
        timestamp=isoformat(),
    )


@router.post(
    "/predictions",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["model"],
    responses={
        422: {"description": "Invalid customer payload"},
        503: {"model": ErrorResponse, "description": "No trained model available yet"},
    },
)
def predict(customer: CustomerFeatures, settings: SettingsDep) -> PredictionResponse:
    """Score one customer with the promoted champion model."""
    loaded = registry.get(settings.model_dir)
    if loaded is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No trained model available yet. Run `make run-once` to train one.",
        )
    model, meta = loaded
    frame = pd.DataFrame([customer.model_dump()])[FEATURES]
    probability = float(model.predict_proba(frame)[0, 1])
    threshold = float(meta.get("decision_threshold", 0.5))
    log.info(
        "prediction served",
        extra={"fields": {"model_version": meta["model_version"], "probability": probability}},
    )
    return PredictionResponse(
        churn_probability=round(probability, 4),
        predicted_label="Churn" if probability >= threshold else "No churn",
        decision_threshold=threshold,
        model_run_id=meta["run_id"],
        model_version=meta["model_version"],
        model_name=meta["champion"],
    )


@router.get(
    "/runs/latest", response_model=RunManifestResponse, tags=["pipeline"], responses=NOT_FOUND
)
def runs_latest(settings: SettingsDep) -> dict[str, Any]:
    """Return the most recent pipeline-run manifest (any status, including failures)."""
    manifest = latest_manifest(settings)
    if manifest is None:
        raise _not_found("pipeline run manifest")
    return manifest


@router.get("/runs", response_model=list[RunSummary], tags=["pipeline"])
def runs_history(
    settings: SettingsDep, limit: Annotated[int, Query(ge=1, le=100)] = 20
) -> list[RunSummary]:
    """Return a compact history of recent runs, newest first."""
    out = []
    for m in list_manifests(settings)[:limit]:
        model = m.get("model") or {}
        champ = model.get("champion")
        out.append(
            RunSummary(
                run_id=m["run_id"],
                status=m.get("status", "unknown"),
                started_at=m.get("started_at", ""),
                duration_seconds=m.get("duration_seconds"),
                monitoring_status=(m.get("monitoring") or {}).get("overall_status"),
                champion=champ,
                roc_auc=model.get("metrics", {}).get(champ, {}).get("roc_auc") if champ else None,
            )
        )
    return out


@router.get(
    "/metrics/latest", response_model=MetricsResponse, tags=["pipeline"], responses=NOT_FOUND
)
def metrics_latest(settings: SettingsDep) -> MetricsResponse:
    """Return model-comparison metrics from the latest successful run."""
    manifest = latest_manifest(settings, status="success")
    if manifest is None or "model" not in manifest:
        raise _not_found("model metrics")
    model = manifest["model"]
    return MetricsResponse(
        run_id=manifest["run_id"],
        champion=model["champion"],
        model_version=model["model_version"],
        selection_metric=model["selection_metric"],
        metrics=model["metrics"],
        feature_importance=model["feature_importance"],
    )


@router.get("/eda/latest", response_model=EdaResponse, tags=["pipeline"], responses=NOT_FOUND)
def eda_latest(settings: SettingsDep) -> EdaResponse:
    """Return the EDA summary (correlations, Cramer's V, artifact paths) of the latest run."""
    manifest = latest_manifest(settings, status="success")
    if manifest is None or "eda" not in manifest:
        raise _not_found("EDA summary")
    return EdaResponse(run_id=manifest["run_id"], **manifest["eda"])


@router.get(
    "/application-details",
    response_model=ApplicationDetailsResponse,
    tags=["prefect"],
    responses={503: {"model": PrefectUnavailableResponse, "description": "Prefect unavailable"}},
)
async def application_details(
    provider: Annotated[PrefectDetailsProvider, Depends(get_prefect_provider)],
) -> Any:
    """Flow, deployment/schedule, latest flow run and work pool details from the Prefect API."""
    try:
        return await provider.get_application_details()
    except PrefectUnavailableError as exc:
        log.warning(
            "prefect api unavailable", extra={"fields": {"reason": exc.reason, "error": str(exc)}}
        )
        body = PrefectUnavailableResponse(
            detail=str(exc),
            reason=exc.reason,
            prefect_api_url=exc.api_url,
            checked_at=isoformat(),
            hint=(
                "Start the Prefect server (`make prefect-server`) and create the deployment "
                "(`make deploy-local`), or set PREFECT_API_URL/PREFECT_API_KEY for Prefect Cloud."
            ),
        )
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content=body.model_dump()
        )


def create_app(settings: Settings | None = None) -> FastAPI:
    """Application factory."""
    settings = settings or get_settings()
    app = FastAPI(
        title="Telco Churn DataOps API",
        version=__version__,
        description=(
            "Churn predictions, pipeline run results and Prefect application details "
            "(flow, deployment, flow runs, work pool) retrieved via Prefect's built-in API."
        ),
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
    app.include_router(router)

    @app.get("/", include_in_schema=False)
    def root() -> dict[str, str]:
        return {"service": "telco-churn-dataops-api", "docs": "/docs", "api": API_PREFIX}

    return app


app = create_app()
