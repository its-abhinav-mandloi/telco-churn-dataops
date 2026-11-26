"""Central configuration loaded from environment variables (and an optional ``.env`` file)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DATASET_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)


class Settings(BaseSettings):
    """Runtime settings. Every field can be overridden with an environment variable."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    dataset_url: str = Field(default=DEFAULT_DATASET_URL)
    data_dir: Path = Field(default=PROJECT_ROOT / "data")
    artifacts_dir: Path = Field(default=PROJECT_ROOT / "artifacts")
    random_seed: int = 42
    test_size: float = 0.2
    download_timeout_seconds: float = 30.0
    min_expected_rows: int = 1000
    run_retention: int = 30

    prefect_api_url: str | None = None
    prefect_api_key: SecretStr | None = None
    prefect_work_pool: str = "telco-process-pool"
    prefect_flow_name: str = "telco-churn-pipeline"
    prefect_deployment_name: str = "telco-churn-every-2-min"
    prefect_ui_url: str | None = None

    api_host: str = "127.0.0.1"
    api_port: int = 8000
    dashboard_port: int = 8501
    cors_origins: str = "http://localhost:8501,http://127.0.0.1:8501"

    @property
    def raw_dir(self) -> Path:
        """Directory holding timestamped raw downloads (gitignored)."""
        return self.data_dir / "raw"

    @property
    def runs_dir(self) -> Path:
        """Directory holding one sub-directory per pipeline run."""
        return self.artifacts_dir / "runs"

    @property
    def model_dir(self) -> Path:
        """Directory holding the currently promoted champion model."""
        return self.artifacts_dir / "model"

    @property
    def cors_origin_list(self) -> list[str]:
        """CORS origins parsed from the comma-separated ``CORS_ORIGINS`` variable."""
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def prefect_dashboard_url(self) -> str | None:
        """Best-effort human-facing Prefect UI URL derived from configuration."""
        if self.prefect_ui_url:
            return self.prefect_ui_url
        if not self.prefect_api_url:
            return None
        url = self.prefect_api_url.rstrip("/")
        if "api.prefect.cloud" in url:
            return url.replace("api.prefect.cloud/api", "app.prefect.cloud")
        return url[: -len("/api")] if url.endswith("/api") else url


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance."""
    return Settings()
