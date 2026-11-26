"""Pydantic request/response models for the FastAPI service."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

YesNo = Literal["Yes", "No"]
YesNoNoPhone = Literal["Yes", "No", "No phone service"]
YesNoNoInternet = Literal["Yes", "No", "No internet service"]


class CustomerFeatures(BaseModel):
    """One Telco customer, using the dataset's original column names and categories."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
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
        },
    )

    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=120, description="Months with the company")
    PhoneService: YesNo
    MultipleLines: YesNoNoPhone
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: YesNoNoInternet
    OnlineBackup: YesNoNoInternet
    DeviceProtection: YesNoNoInternet
    TechSupport: YesNoNoInternet
    StreamingTV: YesNoNoInternet
    StreamingMovies: YesNoNoInternet
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0, le=1000)
    TotalCharges: float | None = Field(
        default=None, ge=0, description="May be null for brand-new customers (median-imputed)"
    )


class PredictionResponse(BaseModel):
    """Churn prediction for one customer."""

    churn_probability: float = Field(ge=0, le=1)
    predicted_label: Literal["Churn", "No churn"]
    decision_threshold: float
    model_run_id: str
    model_version: str
    model_name: str


class HealthResponse(BaseModel):
    """Service health."""

    status: Literal["ok"]
    service: str
    version: str
    model_loaded: bool
    latest_run_status: str | None
    timestamp: str


class ErrorResponse(BaseModel):
    """Error envelope used for 404 / 503 responses."""

    detail: str
    hint: str | None = None


class PrefectUnavailableResponse(BaseModel):
    """Transparent diagnostic returned when the Prefect API cannot be used."""

    detail: str
    reason: str
    prefect_api_url: str | None
    checked_at: str
    hint: str


class FlexibleModel(BaseModel):
    """Base for documents that carry nested, run-specific structures."""

    model_config = ConfigDict(extra="allow")


class RunManifestResponse(FlexibleModel):
    """Persisted pipeline-run manifest."""

    run_id: str
    status: str
    started_at: str
    finished_at: str | None = None
    duration_seconds: float | None = None
    prefect: dict[str, Any] = {}
    ingestion: dict[str, Any] | None = None
    monitoring: dict[str, Any] | None = None
    errors: list[dict[str, Any]] = []


class RunSummary(BaseModel):
    """One row of run history."""

    run_id: str
    status: str
    started_at: str
    duration_seconds: float | None = None
    monitoring_status: str | None = None
    champion: str | None = None
    roc_auc: float | None = None


class MetricsResponse(BaseModel):
    """Latest successful model metrics."""

    run_id: str
    champion: str
    model_version: str
    selection_metric: str
    metrics: dict[str, dict[str, Any]]
    feature_importance: dict[str, Any]


class EdaResponse(BaseModel):
    """Latest EDA summary."""

    run_id: str
    churn_rate: float
    top_categorical_associations: dict[str, float]
    numeric_target_correlation: dict[str, float]
    plots: dict[str, str]
    tables: dict[str, str]


class ApplicationDetailsResponse(FlexibleModel):
    """Application details retrieved from the Prefect built-in API."""

    retrieved_at: str
    prefect_api: dict[str, Any]
    flow: dict[str, Any]
    deployment: dict[str, Any]
    latest_flow_run: dict[str, Any]
    work_pool: dict[str, Any]
