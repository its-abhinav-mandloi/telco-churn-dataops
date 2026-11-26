"""Filesystem helpers: atomic JSON writes, run directories, manifest discovery, retention."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from telco_churn_dataops.config import PROJECT_ROOT, Settings

MANIFEST_NAME = "manifest.json"
RUN_LOG_NAME = "run_log.jsonl"


def atomic_write_text(path: Path, text: str) -> None:
    """Write text to ``path`` atomically (temp file + rename) so readers never see partial files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def write_json(path: Path, data: Any) -> Path:
    """Atomically write ``data`` as pretty JSON and return the path."""
    atomic_write_text(path, json.dumps(data, indent=2, default=str))
    return path


def read_json(path: Path) -> Any:
    """Read a JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def relative_path(path: Path | str) -> str:
    """Return ``path`` relative to the project root (never leak absolute machine paths)."""
    p = Path(path).resolve()
    try:
        return p.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return p.name


def run_dir(settings: Settings, run_id: str) -> Path:
    """Return (and create) the artifact directory for ``run_id``."""
    path = settings.runs_dir / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_manifests(settings: Settings) -> list[dict[str, Any]]:
    """Return all readable run manifests, newest first."""
    manifests: list[dict[str, Any]] = []
    if not settings.runs_dir.exists():
        return manifests
    for path in settings.runs_dir.glob(f"*/{MANIFEST_NAME}"):
        try:
            manifests.append(read_json(path))
        except (OSError, json.JSONDecodeError):
            continue
    manifests.sort(key=lambda m: m.get("started_at", ""), reverse=True)
    return manifests


def latest_manifest(settings: Settings, status: str | None = None) -> dict[str, Any] | None:
    """Return the newest manifest, optionally filtered by final ``status``."""
    for manifest in list_manifests(settings):
        if status is None or manifest.get("status") == status:
            return manifest
    return None


def apply_retention(settings: Settings, keep: int | None = None) -> list[str]:
    """Delete the oldest run directories and raw files beyond the retention limit."""
    keep = keep if keep is not None else settings.run_retention
    removed: list[str] = []
    if settings.runs_dir.exists():
        dirs = sorted((p for p in settings.runs_dir.iterdir() if p.is_dir()), reverse=True)
        for old in dirs[keep:]:
            shutil.rmtree(old, ignore_errors=True)
            removed.append(old.name)
    if settings.raw_dir.exists():
        raws = sorted(settings.raw_dir.glob("telco_churn_*.csv"), reverse=True)
        for old in raws[keep:]:
            old.unlink(missing_ok=True)
    return removed
