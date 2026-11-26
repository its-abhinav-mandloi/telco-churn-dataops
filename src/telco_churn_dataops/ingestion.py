"""Data ingestion: download, validate, checksum and record provenance of the public CSV."""

from __future__ import annotations

import hashlib
import io
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from telco_churn_dataops.logging_utils import isoformat
from telco_churn_dataops.storage import relative_path

EXPECTED_COLUMNS: tuple[str, ...] = (
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
)

ACCEPTED_CONTENT_TYPES = ("text/csv", "text/plain", "application/octet-stream", "application/csv")


class DataValidationError(ValueError):
    """Raised when downloaded content is not the expected Telco CSV."""


@dataclass(frozen=True)
class IngestionResult:
    """Provenance record of one ingestion."""

    source_url: str
    raw_path: str
    sha256: str
    n_rows: int
    n_columns: int
    bytes: int
    content_type: str
    http_status: int
    downloaded_at: str

    def to_dict(self) -> dict[str, Any]:
        """Return the record as a plain dictionary."""
        return asdict(self)


def sha256_bytes(content: bytes) -> str:
    """Return the hex SHA-256 digest of ``content``."""
    return hashlib.sha256(content).hexdigest()


def validate_csv_bytes(content: bytes, min_rows: int) -> pd.DataFrame:
    """Parse ``content`` as CSV and validate the Telco schema and minimum row count."""
    if not content.strip():
        raise DataValidationError("Downloaded file is empty")
    try:
        df = pd.read_csv(io.BytesIO(content))
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise DataValidationError(f"Content is not parseable CSV: {exc}") from exc
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(f"CSV is missing expected columns: {missing}")
    if len(df) < min_rows:
        raise DataValidationError(f"CSV has {len(df)} rows; at least {min_rows} are required")
    return df


def fetch_bytes(url: str, timeout: float) -> tuple[bytes, int, str]:
    """Download ``url`` with a timeout. Raises ``requests.HTTPError`` on non-2xx status."""
    response = requests.get(url, timeout=timeout, headers={"User-Agent": "telco-churn-dataops"})
    response.raise_for_status()
    content_type = response.headers.get("Content-Type", "").split(";")[0].strip().lower()
    if content_type and content_type not in ACCEPTED_CONTENT_TYPES:
        raise DataValidationError(f"Unexpected content type: {content_type!r}")
    return response.content, response.status_code, content_type


def save_raw_copy(content: bytes, raw_dir: Path, run_id: str) -> Path:
    """Atomically persist the raw bytes as ``telco_churn_<run_id>.csv``.

    Writing via a temp file + ``os.replace`` makes retries idempotent: a retried
    download overwrites the same run-scoped file and never leaves a partial file.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / f"telco_churn_{run_id}.csv"
    fd, tmp = tempfile.mkstemp(dir=raw_dir, prefix=".download.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(content)
        os.replace(tmp, target)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return target


def ingest(
    url: str, raw_dir: Path, run_id: str, timeout: float = 30.0, min_rows: int = 1000
) -> tuple[pd.DataFrame, IngestionResult]:
    """Download, validate and store the dataset, returning the frame and its provenance."""
    content, status, content_type = fetch_bytes(url, timeout)
    df = validate_csv_bytes(content, min_rows)
    path = save_raw_copy(content, raw_dir, run_id)
    result = IngestionResult(
        source_url=url,
        raw_path=relative_path(path),
        sha256=sha256_bytes(content),
        n_rows=int(df.shape[0]),
        n_columns=int(df.shape[1]),
        bytes=len(content),
        content_type=content_type,
        http_status=status,
        downloaded_at=isoformat(),
    )
    return df, result
