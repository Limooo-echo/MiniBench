"""Run frozen smoke rosters with the formal protocol; --prepare-only calls no model."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from minibench.datasets.xiangqi.schema import load_records
from minibench.datasets.xiangqi.engines.pikafish import pikafish_fingerprint, resolve_pikafish_executable
from minibench.datasets.xiangqi.runtime import PROTOCOL_VERSION
from minibench.evaluate import run_experiment
from minibench.factory.config import load_experiment_config, validate_experiment_config

FAMILIES = {
    key: (family, ROOT / f"config/experiments/xiangqi_smoke/{key}.yaml",
          ROOT / f"data/xiangqi/{folder}/tasks.jsonl",
          ROOT / f"data/xiangqi/evaluation_samples/smoke/{key}.jsonl")
    for key, family, folder in (
        ("d3", "xiangqi-mate-in-one", "mate_in_one"), ("h2", "xiangqi-history", "history"),
        ("c2", "xiangqi-rule-variants", "rule_variants"), ("m2", "xiangqi-multimodal", "multimodal"),
    )
}


def _parse_tasks(value: str) -> list[str]:
    names = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not names or set(names) - set(FAMILIES):
        raise ValueError("expected a nonempty subset of d3,h2,c2,m2")
    return list(dict.fromkeys(names))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (ROOT / candidate).resolve()


def prepare_suite(suite_dir: Path, tasks: list[str], *, model: str | None = None,
                  api_key_env: str | None = None, pikafish_path: Path | None = None):
    """Save all execution inputs before credentials are checked or agents exist."""
    suite_dir = suite_dir.resolve()
    suite_dir.mkdir(parents=True, exist_ok=False)
    (suite_dir / "samples").mkdir()
    (suite_dir / "configs").mkdir()
    engine = _resolve(resolve_pikafish_executable(pikafish_path, start_dir=ROOT)) if set(tasks) & {"d3", "h2", "m2"} else None
    plan = {
        "suite": "xiangqi-frozen-smoke-v2", "protocol_version": PROTOCOL_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(), "selection_policy": "frozen-smoke-roster-no-resampling",
        "depths": {"h2_opponent": 16, "d3_m2_diagnostics": 8, "c2_search": 3},
        "engine": pikafish_fingerprint(engine) if engine else None,
        "note": "Only credential environment variable names are saved, never their values.", "tasks": {},
    }
    prepared = {}
    for key in tasks:
        family, config_path, data_path, frozen_path = FAMILIES[key]
        if not config_path.is_file() or not frozen_path.is_file():
            raise FileNotFoundError("Missing frozen smoke config/roster; run scripts/freeze_xiangqi_evaluation.py after release validation")
        config = deepcopy(load_experiment_config(config_path))
        task = config["task"]
        selection = task.get("selection") or {}
        if _resolve(task["path"]) != data_path.resolve():
            raise ValueError(f"{key}: task.path must retain the full source dataset")
        if _resolve(selection.get("path", "")) != frozen_path.resolve():
            raise ValueError(f"{key}: config must reference its frozen smoke roster")
        frozen_hash = _sha256(frozen_path)
        if selection.get("sha256") != frozen_hash:
            raise ValueError(f"{key}: frozen smoke selection hash mismatch")
        if task.get("task_ids") or task.get("limit") is not None or (task.get("sampling") or {}).get("enabled"):
            raise ValueError(f"{key}: frozen selection cannot be combined with resampling, task_ids or limit")
        selected = load_records(frozen_path, expected_family=family)
        originals = {r["id"]: r for r in load_records(data_path, expected_family=family)}
        for record in selected:
            if originals.get(record["id"]) != record:
                raise ValueError(f"{key}: frozen record differs from source dataset: {record['id']}")
        sample_path = suite_dir / "samples" / f"{key}.jsonl"
        shutil.copyfile(frozen_path, sample_path)
        task.update(path=str(data_path.resolve()), selection={"path": str(sample_path), "sha256": frozen_hash}, prompt_version=PROTOCOL_VERSION)
        if model is not None:
            config["provider"]["model"] = model
        if api_key_env is not None:
            config["provider"]["api_key_env"] = api_key_env
        config["run"].update(output_dir=str(suite_dir), run_name=key, on_existing="error")
        evaluation = config.setdefault("evaluation", {})
        if key in {"d3", "h2", "m2"}:
            evaluation.update(pikafish_path=str(engine), pikafish_depth=16 if key == "h2" else 8)
        if key == "h2":
            evaluation["history_mode"] = "paired"
        elif key == "c2":
            evaluation["oracle_depth"] = 3
        elif key == "m2":
            evaluation.update(max_plies=1, verify_with_pikafish=True, step_dir=str(suite_dir / "m2-input-images"))
        config = validate_experiment_config(config)
        saved_config = suite_dir / "configs" / f"{key}.json"
        _write_json(saved_config, config)
        plan["tasks"][key] = {
            "family": family, "source_path": str(data_path.resolve()), "source_sha256": _sha256(data_path),
            "frozen_selection_path": str(frozen_path.resolve()), "selection_sha256": frozen_hash,
            "sample_path": f"samples/{key}.jsonl", "sample_sha256": _sha256(sample_path),
            "config_path": f"configs/{key}.json", "config_sha256": _sha256(saved_config),
            "selected_record_count": len(selected), "task_ids": [r["id"] for r in selected],
        }
        prepared[key] = config
    _write_json(suite_dir / "smoke_plan.json", plan)
    _write_json(suite_dir / "suite_results.json", {"status": "prepared", "suite_dir": str(suite_dir), "tasks": {}, "failures": {}})
    return plan, prepared


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default="d3,h2,c2,m2")
    parser.add_argument("--model", help="Optional model override; never changes the frozen roster")
    parser.add_argument("--api-key-env", help="Optional provider credential variable name")
    parser.add_argument("--pikafish-path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--prepare-only", action="store_true", help="Save inputs without a model call or engine process")
    args = parser.parse_args(argv)
    tasks = _parse_tasks(args.tasks)
    suite_dir = (args.output_dir or ROOT / "runs" / f"xiangqi-smoke-{datetime.now().strftime('%Y%m%d-%H%M%S')}").resolve()
    _plan, prepared = prepare_suite(suite_dir, tasks, model=args.model, api_key_env=args.api_key_env, pikafish_path=args.pikafish_path)
    if args.prepare_only:
        print(json.dumps({"status": "prepared", "suite_dir": str(suite_dir), "tasks": tasks, "model_calls": 0}, ensure_ascii=False))
        return 0
    outcomes, failures = {}, {}
    report = {"suite_dir": str(suite_dir), "status": "running", "tasks": outcomes, "failures": failures,
              "note": "Engineering smoke only; use minibench score-suite for capability scores."}
    try:
        for config in prepared.values():
            variable = config.get("provider", {}).get("api_key_env")
            if variable and not os.environ.get(variable):
                raise RuntimeError(f"missing ${variable}; export it before executing this suite")
        _write_json(suite_dir / "suite_results.json", report)
        for key in tasks:
            print(f"Starting frozen {key.upper()} smoke", flush=True)
            try:
                outcomes[key] = run_experiment(prepared[key])
            except Exception as exc:
                failures[key] = f"{type(exc).__name__}: {exc}"
            _write_json(suite_dir / "suite_results.json", report)
        report["status"] = "completed" if not failures else "completed_with_failures"
    except KeyboardInterrupt:
        report["status"] = "interrupted"
    except Exception as exc:
        report["status"] = "failed"
        failures["suite"] = f"{type(exc).__name__}: {exc}"
    _write_json(suite_dir / "suite_results.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
