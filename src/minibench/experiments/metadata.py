"""Credential-free experiment metadata and timestamps."""
from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _git_revision(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_is_dirty(path: Path) -> bool | None:
    try:
        return bool(
            subprocess.check_output(
                ["git", "-C", str(path), "status", "--porcelain"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return None


def _model_request_metadata(provider: dict[str, Any]) -> dict[str, Any]:
    """Preserve request identity without capturing credentials or environment values."""
    return {
        key: provider[key]
        for key in (
            "name",
            "model",
            "base_url",
            "api_version",
            "json_mode",
            "extra_body",
        )
        if key in provider
    }


def _plain_config(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _plain_config(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_config(item) for item in value]
    return value


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()
