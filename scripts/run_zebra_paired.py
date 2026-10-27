"""Isolated, resumable local Zebra experiment; no other experiment is changed."""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
import getpass
import json
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import yaml

from build_zebra_paired import verify_paired_release
from minibench.agents.direct import DirectAgent
from minibench.core.agent import ReasoningConfig
from minibench.core.checkpoint import (
    RunLock, atomic_write_json, atomic_write_jsonl, fingerprint_payload,
    read_json_object, read_jsonl_objects, sha256_file,
)
from minibench.core.runtime import StrictJSONObjectError, parse_single_json_object
from minibench.datasets.zebra.dataset import zebra_task_from_dict
from minibench.datasets.zebra.evaluation import expected_solution_table, score_zebra_output
from minibench.datasets.zebra.prompting import build_zebra_prompt
from minibench.datasets.zebra.qwen_client import ZebraQwenClient

MODES = ("direct", "codebook", "counterfactual")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    fixed = {
        "model": "qwen3.8-flash", "agent": "direct", "prompt_version": "v2",
        "temperature": 0.0, "thinking_tokens": 7168, "answer_tokens": 1024,
        "max_format_repairs": 0, "max_retries": 0,
    }
    for key, expected in fixed.items():
        if config.get(key) != expected:
            raise ValueError(f"This protocol requires {key}={expected!r}")
    for key in ("workers", "requests_per_minute", "timeout", "validation_timeout_ms"):
        if type(config.get(key)) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if not str(config.get("base_url", "")).startswith("https://"):
        raise ValueError("base_url must use HTTPS")
    return config


def source_hashes() -> dict[str, str]:
    files = [
        Path(__file__).resolve(), ROOT / "scripts/build_zebra_paired.py",
        *sorted((ROOT / "src/minibench/datasets/zebra").glob("*.py")),
        *sorted((ROOT / "src/minibench/agents").glob("*.py")),
        *sorted((ROOT / "src/minibench/core").glob("*.py")),
        ROOT / "src/minibench/factory/providers.py",
    ]
    return {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in files}


def work_plan(dataset: Path) -> list[dict[str, Any]]:
    groups = {mode: read_jsonl_objects(dataset / f"{mode}.jsonl") for mode in MODES}
    ids = [[r["source_id"] for r in groups[mode]] for mode in MODES]
    if len(ids[0]) != 100 or len(set(ids[0])) != 100 or not all(x == ids[0] for x in ids):
        raise ValueError("All three conditions must share the same ordered 100 sources")
    plan = []
    for index in range(100):
        # Rotate condition order to avoid always giving one condition the first slot.
        for mode in MODES[index % 3:] + MODES[:index % 3]:
            record = groups[mode][index]
            key = fingerprint_payload({"mode": mode, "source_id": record["source_id"]})[:24]
            plan.append({"key": key, "mode": mode, "record": record})
    return plan


def answer_diagnostics(task: Any, raw: str) -> dict[str, Any]:
    try:
        payload = parse_single_json_object(raw)
    except ValueError:
        return {"strict_json": False, "schema_complete": False, "error": "invalid_json"}
    solution = payload.get("solution")
    expected = expected_solution_table(task)
    if not isinstance(solution, dict):
        return {"strict_json": True, "schema_complete": False, "error": "missing_solution"}
    missing = [house for house in expected if house not in solution]
    extra = [house for house in solution if house not in expected]
    columns_ok = all(
        isinstance(solution.get(house), dict)
        and set(solution[house]) == set(columns)
        and all(isinstance(value, str) and value.strip() and value != "___"
                for value in solution[house].values())
        for house, columns in expected.items()
    )
    complete = (not missing and not extra and columns_ok
                and isinstance(payload.get("reasoning"), str)
                and set(payload) == {"reasoning", "solution"})
    return {"strict_json": True, "schema_complete": complete,
            "missing_houses": missing, "extra_houses": extra, "columns_valid": columns_ok}


class RequestGate:
    def __init__(self, requests_per_minute: int):
        self.interval = 60.0 / requests_per_minute
        self.lock = threading.Lock()
        self.next_at = 0.0

    def __call__(self) -> None:
        with self.lock:
            delay = self.next_at - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            self.next_at = time.monotonic() + self.interval


def execute_item(item: dict[str, Any], config: dict[str, Any], run_dir: Path,
                 fingerprint: str, gate: RequestGate) -> dict[str, Any]:
    task = zebra_task_from_dict(item["record"])
    job_dir = run_dir / "jobs" / item["key"]
    client = ZebraQwenClient(
        journal_dir=job_dir / "calls", model=config["model"],
        base_url=config["base_url"], api_key_env=config["api_key_env"],
        timeout=config["timeout"], before_request=gate,
    )
    agent = DirectAgent(client, ReasoningConfig(
        prompt_version="v2", final_temperature=0.0, final_max_tokens=8192,
        max_format_repairs=0, trace="full",
    ))
    started = time.monotonic()
    strict_error = False
    try:
        raw = agent.generate(build_zebra_prompt(task), task)
    except StrictJSONObjectError as exc:
        if exc.raw_output is None:
            raise
        raw = exc.raw_output
        strict_error = True
    result = {
        "key": item["key"], "mode": item["mode"], "task_id": task.id,
        "source_id": task.source_id, "size": task.size, "difficulty": task.difficulty,
        "run_fingerprint": fingerprint, "task_sha256": fingerprint_payload(item["record"]),
        "completed_at": now(), "elapsed_seconds": time.monotonic() - started,
        "raw_output": raw, "result": score_zebra_output(task, raw),
        "diagnostics": {**answer_diagnostics(task, raw), "strict_json_error": strict_error},
        "metrics": client.metrics_snapshot(),
        "trace": agent.last_run.to_dict() if agent.last_run else None,
    }
    atomic_write_json(job_dir / "result.json", result)
    return result


def restore_results(plan: list[dict[str, Any]], run_dir: Path,
                    fingerprint: str) -> dict[str, dict[str, Any]]:
    results = {}
    for item in plan:
        path = run_dir / "jobs" / item["key"] / "result.json"
        if not path.exists():
            continue
        saved = read_json_object(path)
        record = item["record"]
        required = {"key": item["key"], "mode": item["mode"],
                    "source_id": record["source_id"], "task_id": record["id"],
                    "run_fingerprint": fingerprint, "task_sha256": fingerprint_payload(record)}
        if any(saved.get(key) != value for key, value in required.items()):
            raise ValueError(f"Saved result identity changed: {item['key']}")
        if saved["result"] != score_zebra_output(zebra_task_from_dict(record), saved["raw_output"]):
            raise ValueError(f"Saved result score mismatch: {item['key']}")
        results[item["key"]] = saved
    return results


def save_summary(run_dir: Path, plan: list[dict[str, Any]],
                 results: dict[str, dict[str, Any]], status: str,
                 errors: list[dict[str, str]] | None = None) -> dict[str, Any]:
    ordered = [results[item["key"]] for item in plan if item["key"] in results]
    groups = {}
    for mode in MODES:
        rows = [row for row in ordered if row["mode"] == mode]
        cells = sum(row["result"]["total_cells"] for row in rows)
        success = sum(row["result"]["success"] for row in rows)
        groups[mode] = {
            "completed": len(rows), "planned": 100, "success": success,
            "accuracy": success / len(rows) if rows else None,
            "parsed": sum(row["result"]["parsed"] for row in rows),
            "schema_complete": sum(row["diagnostics"]["schema_complete"] for row in rows),
            "parsed_zero_cells": sum(row["result"]["parsed"] and
                                     row["result"]["correct_cells"] == 0 for row in rows),
            "cell_accuracy": sum(row["result"]["correct_cells"] for row in rows) / cells if cells else None,
        }
    paired = {}
    direct = {row["source_id"]: row for row in ordered if row["mode"] == "direct"}
    for mode in MODES[1:]:
        other = {row["source_id"]: row for row in ordered if row["mode"] == mode}
        counts = {"both_correct": 0, "direct_only": 0, "variant_only": 0, "both_wrong": 0}
        for source_id in direct.keys() & other.keys():
            a, b = direct[source_id]["result"]["success"], other[source_id]["result"]["success"]
            counts["both_correct" if a and b else "direct_only" if a else "variant_only" if b else "both_wrong"] += 1
        paired[mode] = counts
    summary = {"status": status, "updated_at": now(), "completed": len(ordered),
               "planned": len(plan), "conditions": groups, "paired_against_direct": paired,
               "errors": errors or []}
    atomic_write_jsonl(run_dir / "per-task-results.jsonl", ordered)
    atomic_write_json(run_dir / "summary.json", summary)
    atomic_write_json(run_dir / "status.json", {k: summary[k] for k in
                      ("status", "updated_at", "completed", "planned", "errors")})
    return summary


def run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    dataset, run_dir = resolve_path(config["dataset"]), resolve_path(config["output_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    with RunLock(run_dir):
        print("Verifying all 300 puzzles with Z3: existence, uniqueness, and gold...", flush=True)
        verification = verify_paired_release(dataset, fresh=True,
                                             timeout_ms=config["validation_timeout_ms"])
        plan = work_plan(dataset)
        identity = {"config": config, "dataset_manifest_sha256": sha256_file(dataset / "manifest.json"),
                    "code_sha256": source_hashes(),
                    "work_plan": [{"key": x["key"], "task_sha256": fingerprint_payload(x["record"])}
                                  for x in plan]}
        fingerprint = fingerprint_payload(identity)
        manifest_path = run_dir / "manifest.json"
        if manifest_path.exists():
            if read_json_object(manifest_path).get("fingerprint") != fingerprint:
                raise ValueError("Run config/data/code changed; use a new output_dir")
        else:
            atomic_write_json(manifest_path, {"fingerprint": fingerprint, "created_at": now(), **identity})
        results = restore_results(plan, run_dir, fingerprint)
        atomic_write_json(run_dir / "preflight.json", {"verified_at": now(),
                          "verification": verification, "planned": len(plan), "fingerprint": fingerprint})
        if args.preflight:
            save_summary(run_dir, plan, results, "completed" if len(results) == 300 else "ready")
            print(f"PASS: 100 paired sources, 300 verified puzzles. Run: {run_dir}", flush=True)
            return 0
        if len(results) == len(plan):
            save_summary(run_dir, plan, results, "completed")
            print("All 300 results already complete; no model calls.", flush=True)
            return 0
        if args.ask_key:
            save_summary(run_dir, plan, results, "waiting_for_key")
            if not sys.stdin.isatty():
                raise RuntimeError("--ask-key requires an interactive terminal")
            key = getpass.getpass("Enter your DashScope API key (hidden, process memory only): ")
            if not key.strip():
                raise ValueError("No API key entered")
            os.environ[config["api_key_env"]] = key.strip()
            del key
        elif not os.environ.get(config["api_key_env"]):
            save_summary(run_dir, plan, results, "waiting_for_key")
            raise RuntimeError("API key missing. Run the WSL launcher and enter it privately.")
        print(f"Starting Direct / Qwen3.8-Flash: {len(results)}/300 restored, {config['workers']} workers.", flush=True)
        gate = RequestGate(config["requests_per_minute"])
        pending = iter(item for item in plan if item["key"] not in results)
        failures: list[dict[str, str]] = []
        save_summary(run_dir, plan, results, "running")
        with ThreadPoolExecutor(max_workers=config["workers"]) as pool:
            active = {}
            def fill() -> None:
                while len(active) < config["workers"] and not failures:
                    item = next(pending, None)
                    if item is None:
                        break
                    active[pool.submit(execute_item, item, config, run_dir, fingerprint, gate)] = item
            fill()
            while active:
                done, _ = wait(active, return_when=FIRST_COMPLETED)
                for future in done:
                    item = active.pop(future)
                    try:
                        result = future.result()
                        results[item["key"]] = result
                        print(f"[{len(results):03d}/300] {item['mode']}: "
                              f"{'correct' if result['result']['success'] else 'incorrect'}", flush=True)
                    except Exception as exc:
                        message = str(exc)
                        secret = os.environ.get(config["api_key_env"])
                        if secret:
                            message = message.replace(secret, "[redacted]")
                        failures.append({"key": item["key"], "mode": item["mode"],
                                         "error": f"{type(exc).__name__}: {message}"})
                        print(f"Infrastructure error in {item['mode']}; stopping new work. Resume via the same launcher.", flush=True)
                save_summary(run_dir, plan, results, "stopping_after_error" if failures else "running", failures)
                fill()
        summary = save_summary(run_dir, plan, results, "failed" if failures else "completed", failures)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/experiments/zebra_paired_qwen38.yaml")
    parser.add_argument("--preflight", action="store_true", help="Offline verification only; no API key or calls")
    parser.add_argument("--ask-key", action="store_true", help="Privately prompt in the user's terminal")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
