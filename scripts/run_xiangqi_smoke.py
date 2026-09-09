"""Run a reproducible, small end-to-end Xiangqi acceptance suite.

The selected records are materialised before any model request.  This makes a
failed or interrupted run auditable and lets the browser test driver reuse the
exact same positions later.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from minibench.datasets.xiangqi.schema import load_records, sample_records
from minibench.datasets.xiangqi.engines.pikafish import (
    pikafish_fingerprint,
    resolve_pikafish_executable,
)
from minibench.evaluate import run_experiment
from minibench.factory.config import load_experiment_config


FAMILIES = {
    "d3": (
        "xiangqi-mate-in-one",
        ROOT / "config/experiments/xiangqi_mate_in_one.yaml",
        ROOT / "data/xiangqi/mate_in_one/tasks.jsonl",
        6,
    ),
    "h2": (
        "xiangqi-history",
        ROOT / "config/experiments/xiangqi_history.yaml",
        ROOT / "data/xiangqi/history/tasks.jsonl",
        6,
    ),
    "c2": (
        "xiangqi-rule-variants",
        ROOT / "config/experiments/xiangqi_rule_variants.yaml",
        ROOT / "data/xiangqi/rule_variants/tasks.jsonl",
        3,
    ),
    "m2": (
        "xiangqi-multimodal",
        ROOT / "config/experiments/xiangqi_multimodal.yaml",
        ROOT / "data/xiangqi/multimodal/tasks.jsonl",
        6,
    ),
}


def _parse_tasks(value: str) -> list[str]:
    tasks = [item.strip().lower() for item in value.split(",") if item.strip()]
    unknown = sorted(set(tasks) - set(FAMILIES))
    if unknown:
        raise ValueError(f"unknown task(s): {', '.join(unknown)}")
    return list(dict.fromkeys(tasks))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def _count_for(args: argparse.Namespace, task: str) -> int:
    configured = getattr(args, f"{task}_count")
    return int(configured if configured is not None else FAMILIES[task][3])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run sampled D3/H2/C2/M2 with Direct Qwen and thinking disabled."
    )
    parser.add_argument("--tasks", default="d3,h2,c2,m2")
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--d3-count", type=int)
    parser.add_argument("--h2-count", type=int)
    parser.add_argument(
        "--c2-count", type=int,
        help="Number of paired scenarios; each scenario expands to four rulesets.",
    )
    parser.add_argument("--m2-count", type=int)
    parser.add_argument("--model", default="qwen3.8-max")
    parser.add_argument("--api-key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument(
        "--pikafish-depth", type=int, default=8,
        help="Engine depth for this quick acceptance run (formal H2 defaults to 16).",
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)

    tasks = _parse_tasks(args.tasks)
    if not os.environ.get(args.api_key_env):
        raise SystemExit(
            f"missing ${args.api_key_env}; export it only in this shell before running"
        )
    created = datetime.now().strftime("%Y%m%d-%H%M%S")
    suite_dir = args.output_dir or ROOT / "runs" / f"xiangqi-smoke-{created}"
    suite_dir = suite_dir.resolve()
    suite_dir.mkdir(parents=True, exist_ok=False)
    samples_dir = suite_dir / "samples"
    samples_dir.mkdir()
    pikafish_path = None
    if set(tasks) & {"d3", "h2", "m2"}:
        pikafish_path = resolve_pikafish_executable(None, start_dir=ROOT)

    plan: dict[str, Any] = {
        "suite": "xiangqi-end-to-end-smoke-v1",
        "created_local": created,
        "seed": args.seed,
        "agent": "direct",
        "provider": "qwen",
        "model": args.model,
        "thinking_enabled": False,
        "pikafish_depth": args.pikafish_depth,
        "api_key_env": args.api_key_env,
        "note": "No API key or environment value is persisted.",
        "pikafish": (
            pikafish_fingerprint(pikafish_path) if pikafish_path else None
        ),
        "tasks": {},
    }
    prepared: dict[str, tuple[dict[str, Any], Path]] = {}
    for short_name in tasks:
        family, config_path, data_path, _default_count = FAMILIES[short_name]
        config = deepcopy(load_experiment_config(config_path))
        count = _count_for(args, short_name)
        strategy = str(config["task"]["sampling"]["strategy"])
        records = load_records(data_path, expected_family=family)
        selected = sample_records(
            records, count=count, seed=args.seed, strategy=strategy
        )
        sample_path = samples_dir / f"{short_name}.jsonl"
        _write_jsonl(sample_path, selected)
        plan["tasks"][short_name] = {
            "family": family,
            "requested_count": count,
            "selected_record_count": len(selected),
            "sampling_strategy": strategy,
            "source_path": str(data_path.relative_to(ROOT)),
            "source_sha256": _sha256(data_path),
            "sample_path": str(sample_path.relative_to(suite_dir)),
            "sample_sha256": _sha256(sample_path),
            "task_ids": [record["id"] for record in selected],
        }
        config["task"]["path"] = str(sample_path)
        config["task"]["sampling"]["enabled"] = False
        config["agent"]["name"] = "direct"
        config["provider"].update(
            name="qwen",
            model=args.model,
            api_key_env=args.api_key_env,
            json_mode=True,
            extra_body={"enable_thinking": False},
        )
        config["run"].update(
            output_dir=str(suite_dir),
            run_name=short_name,
            on_existing="error",
        )
        if short_name == "h2":
            config["evaluation"]["history_mode"] = "paired"
        if short_name in {"d3", "h2", "m2"}:
            config["evaluation"]["pikafish_path"] = str(pikafish_path)
            config["evaluation"]["pikafish_depth"] = args.pikafish_depth
        if short_name == "m2":
            config["evaluation"].update(
                max_plies=1,
                verify_with_pikafish=True,
                step_dir=str(suite_dir / "m2-input-images"),
            )
        prepared[short_name] = (config, sample_path)

    _write_json(suite_dir / "smoke_plan.json", plan)

    outcomes: dict[str, Any] = {}
    failures: dict[str, str] = {}
    for short_name in tasks:
        config, _sample_path = prepared[short_name]
        print(f"\n=== {short_name.upper()} ===", flush=True)
        try:
            outcomes[short_name] = run_experiment(config)
        except Exception as exc:  # preserve other task diagnostics in an acceptance run
            failures[short_name] = f"{type(exc).__name__}: {exc}"
            print(f"[{short_name}] FAILED: {failures[short_name]}", file=sys.stderr)

    report = {
        "suite_dir": str(suite_dir),
        "status": "completed" if not failures else "completed_with_failures",
        "tasks": outcomes,
        "failures": failures,
        "note": (
            "This is an engineering acceptance report. Produce the official "
            "four-dimensional scores with `minibench score-suite`."
        ),
    }
    _write_json(suite_dir / "suite_results.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
