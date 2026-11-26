"""Structured (JSON) logging helpers and the per-run JSONL event log."""

from __future__ import annotations

import json
import logging
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_LOCK = threading.Lock()


def utc_now() -> datetime:
    """Return the current timezone-aware UTC time."""
    return datetime.now(UTC)


def isoformat(dt: datetime | None = None) -> str:
    """Serialise a datetime (default: now) as an ISO-8601 UTC string."""
    return (dt or utc_now()).isoformat()


class JsonFormatter(logging.Formatter):
    """Format standard-library log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: D102
        payload: dict[str, Any] = {
            "ts": isoformat(datetime.fromtimestamp(record.created, UTC)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extra = getattr(record, "fields", None)
        if isinstance(extra, dict):
            payload.update(extra)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def get_logger(name: str) -> logging.Logger:
    """Return a logger that emits JSON lines to stderr (configured once)."""
    logger = logging.getLogger(name)
    if not any(isinstance(h.formatter, JsonFormatter) for h in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


class RunEventLog:
    """Append-only JSONL event log for a single pipeline run.

    Each event is also forwarded to an optional logger so that it appears in the
    Prefect run logs (and therefore in the Prefect UI / Prefect Cloud dashboard).
    """

    def __init__(self, path: Path, run_id: str, logger: logging.Logger | Any | None = None):
        self.path = path
        self.run_id = run_id
        self.logger = logger
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def event(self, step: str, status: str, level: str = "INFO", **fields: Any) -> dict[str, Any]:
        """Record one structured event and return it."""
        record = {
            "ts": isoformat(),
            "run_id": self.run_id,
            "step": step,
            "status": status,
            "level": level,
            **fields,
        }
        line = json.dumps(record, default=str)
        with _LOCK, self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        if self.logger is not None:
            log_fn = getattr(self.logger, level.lower(), self.logger.info)
            log_fn(line)
        return record


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file, skipping malformed lines."""
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events
