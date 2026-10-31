"""Run the authorized Direct Zebra history on/off pair from an interactive TTY."""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from datetime import datetime
import getpass
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import traceback
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml
from minibench.core.checkpoint import atomic_write_json
from minibench.datasets.zebra.dataset import load_zebra_tasks
from minibench.datasets.zebra.state_validation import PrefixStateValidator
from minibench.experiments.runner import run_family_experiment
from minibench.experiments.zebra import _zebra_code_sha256
from minibench.factory.agents import make_agent_from_config
from minibench.factory.config import load_experiment_config


def now() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def load_pair() -> dict[str, dict]:
    configs = {
        mode: load_experiment_config(ROOT / f"config/experiments/zebra_history_{mode}.yaml")
        for mode in ("thinking_off", "thinking_on")
    }
    expected_agent = {"name": "direct", "prompt_version": "v2", "max_format_repairs": 0,
                      "final_temperature": 0.0, "trace": "full"}
    for mode, config in configs.items():
        if config["agent"] != expected_agent:
            raise ValueError(f"{mode}: expected Direct v2, no repair, and full trace")
        provider = config["provider"]
        if provider["model"] != "qwen3.8-flash" or provider["max_retries"] != 0:
            raise ValueError(f"{mode}: unexpected model or automatic retries")
        agent = make_agent_from_config(config["agent"], provider)
        if agent.name != "direct" or agent.config.max_format_repairs != 0:
            raise ValueError("factory did not construct the expected Direct agent")
        if config["task"]["limit"] is not None or config["task"]["task_ids"]:
            raise ValueError("this batch must run the complete frozen 100-task dataset")
    off, on = (deepcopy(configs[mode]) for mode in ("thinking_off", "thinking_on"))
    off["run"].pop("run_name")
    on["run"].pop("run_name")
    off_extra = off["provider"]["extra_body"]
    on_extra = on["provider"]["extra_body"]
    if off_extra.pop("enable_thinking") is not False or on_extra.pop("enable_thinking") is not True:
        raise ValueError("thinking treatments must be explicitly off and on")
    if on_extra.pop("thinking_budget") != 7168 or off != on:
        raise ValueError("on/off configs differ beyond the thinking treatment")
    return configs


def preflight(configs: dict[str, dict]) -> dict:
    config = configs["thinking_off"]
    task_path = ROOT / config["task"]["path"]
    tasks = load_zebra_tasks(task_path)
    if len(tasks) != 100 or sum(len(task.clue_turns) for task in tasks) != 978:
        raise ValueError("the frozen Zebra history roster has changed")
    if config["evaluation"]["memory_modes"] != [
        "deferred_reasoning", "incremental_state", "incremental_state_pro",
    ]:
        raise ValueError("expected the three authorized history groups")
    print("Offline preflight: checking 100 puzzles and all clue prefixes...", flush=True)
    for task in tasks:
        PrefixStateValidator(task, timeout_ms=config["evaluation"]["state_validation_timeout_ms"]).preflight_prefixes()
    return {"tasks_per_treatment": 100, "instances_per_treatment": 300,
            "planned_instances": 600, "planned_model_calls": 6468,
            "data_sha256": hashlib.sha256(task_path.read_bytes()).hexdigest(),
            "code_sha256": _zebra_code_sha256(), "preflight_at": now()}


class Tee:
    def __init__(self, terminal, log, secret: str):
        self.terminal, self.log, self.secret = terminal, log, secret

    def write(self, text: str) -> int:
        clean = text.replace(self.secret, "[REDACTED]") if self.secret else text
        self.terminal.write(clean)
        self.log.write(clean)
        self.flush()
        return len(text)

    def flush(self) -> None:
        self.terminal.flush()
        self.log.flush()


def run_batch(batch_dir: Path, configs: dict[str, dict], evidence: dict) -> int:
    if not sys.stdin.isatty():
        raise RuntimeError("Open this launcher in an interactive terminal for hidden key entry")
    batch_dir.mkdir(parents=True, exist_ok=False)
    snapshots = batch_dir / "configs"
    snapshots.mkdir()
    # Preserve the exact source and task bytes used by this uncommitted protocol.
    for path in (ROOT / "src/minibench").rglob("*.py"):
        target = batch_dir / "source" / path.relative_to(ROOT / "src")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    shutil.copy2(ROOT / configs["thinking_off"]["task"]["path"], batch_dir / "history_eval.jsonl")
    for mode, config in configs.items():
        config["run"] = {"output_dir": str(batch_dir), "run_name": mode, "on_existing": "error"}
        (snapshots / f"{mode}.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    manifest = {"schema_version": 1, "created_at": now(), "model": "qwen3.8-flash",
                "agent": "direct", "order": list(configs), **evidence,
                "config_snapshots": {mode: f"configs/{mode}.yaml" for mode in configs}}
    atomic_write_json(batch_dir / "batch-manifest.json", manifest)
    state = {"status": "awaiting_api_key", "started_at": now(), "completed_treatments": [],
             "current_treatment": None, "batch_dir": str(batch_dir)}

    def save_state() -> None:
        state["updated_at"] = now()
        atomic_write_json(batch_dir / "batch-status.json", state)

    save_state()
    print(f"\nDirect / Qwen3.8-Flash / 100 tasks x 3 modes x 2 thinking treatments")
    print(f"Local results: {batch_dir}")
    print("The key stays in this process environment; it is not saved to files.")
    key = ""
    try:
        key = getpass.getpass("DashScope API key (hidden; paste then press Enter): ").strip()
        if not key:
            raise ValueError("No key entered; no API requests sent")
        os.environ["DASHSCOPE_API_KEY"] = key
        with (batch_dir / "console.log").open("a", encoding="utf-8", buffering=1) as log:
            with redirect_stdout(Tee(sys.stdout, log, key)), redirect_stderr(Tee(sys.stderr, log, key)):
                try:
                    for mode, config in configs.items():
                        if _zebra_code_sha256() != evidence["code_sha256"]:
                            raise RuntimeError("Source changed since preflight; stopping before next treatment")
                        task_path = ROOT / config["task"]["path"]
                        if hashlib.sha256(task_path.read_bytes()).hexdigest() != evidence["data_sha256"]:
                            raise RuntimeError("Dataset changed since preflight")
                        state.update(status="running", current_treatment=mode)
                        save_state()
                        print(f"\n[{now()}] Starting {mode}: 300 instances", flush=True)
                        run_dir, summary = run_family_experiment(config)
                        state["completed_treatments"].append(mode)
                        save_state()
                        print(f"\nFinished {mode}: {run_dir}", flush=True)
                        print(json.dumps(summary["by_memory_mode"], ensure_ascii=False), flush=True)
                    state.update(status="completed", current_treatment=None, finished_at=now())
                    save_state()
                    print(f"\nBoth treatments completed. Results: {batch_dir}", flush=True)
                    return 0
                except (Exception, KeyboardInterrupt) as exc:
                    state.update(status="interrupted", error=f"{type(exc).__name__}: {exc}".replace(key, "[REDACTED]"))
                    save_state()
                    traceback.print_exc()
                    return 1
    except (Exception, KeyboardInterrupt) as exc:
        state.update(status="not_started", error=f"{type(exc).__name__}: {exc}".replace(key, "[REDACTED]") if key else str(exc))
        save_state()
        print(state["error"], file=sys.stderr)
        return 1
    finally:
        os.environ.pop("DASHSCOPE_API_KEY", None)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true", help="offline checks only; no key or model calls")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    os.chdir(ROOT)
    configs = load_pair()
    evidence = preflight(configs)
    if args.preflight:
        print(json.dumps(evidence, indent=2))
        return 0
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d-%H%M%S")
    destination = args.output_dir or ROOT / "runs" / f"zebra-history-direct-qwen38flash-{stamp}"
    return run_batch(destination.resolve(), configs, evidence)


if __name__ == "__main__":
    raise SystemExit(main())
