from __future__ import annotations

from pathlib import Path
from typing import Any

from minibench.factory.config import load_experiment_config, validate_experiment_config
from minibench.factory.experiments import run_family_experiment


def run_config(path: str | Path) -> dict[str, Any]:
    config = load_experiment_config(path)
    return run_experiment(config)


def run_experiment(config: dict[str, Any]) -> dict[str, Any]:
    normalized = validate_experiment_config(config)
    run_dir, summary = run_family_experiment(normalized)
    return {"run_dir": str(run_dir), **summary}
