"""Four-process continuation of a frozen Zebra batch, alongside a draining run.

Uses the batch's saved source and configs. A pre-existing sequential treatment
keeps its lock and one worker slot until it finishes; other jobs can proceed.
Only this coordinator writes results for the treatment locks it owns.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import ExitStack
from dataclasses import asdict, replace
from datetime import datetime, timezone
import getpass
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def bootstrap(batch_dir: Path) -> None:
    source = batch_dir / "source"
    if not (source / "minibench").is_dir():
        raise ValueError("batch source snapshot is missing")
    sys.path.insert(0, str(source))
    os.chdir(ROOT)


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def worker_bootstrap(batch_dir: Path) -> None:
    bootstrap(batch_dir)
    # The parent drains submitted work after Ctrl+C instead of losing responses.
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def run_job(task, mode: str, config: dict) -> dict:
    from minibench.datasets.zebra.evaluation import evaluate_zebra_work_item
    from minibench.datasets.zebra.prompting import ZEBRA_SYSTEM_PROMPT
    from minibench.factory.agents import make_agent_from_config
    evaluation = config["evaluation"]
    agent = make_agent_from_config(config["agent"], config["provider"], system_prompt=ZEBRA_SYSTEM_PROMPT)
    result = evaluate_zebra_work_item(
        task, agent, mode,
        state_max_tokens=evaluation["state_max_tokens"],
        ack_max_tokens=evaluation["ack_max_tokens"],
        final_max_tokens=evaluation["final_max_tokens"],
        state_validation_timeout_ms=evaluation["state_validation_timeout_ms"],
    )
    return asdict(result)


class Coordinator:
    def __init__(self, batch_dir: Path, workers: int):
        import yaml
        from minibench.core.checkpoint import read_json_object, sha256_file
        from minibench.datasets.zebra.dataset import load_zebra_tasks
        from minibench.datasets.zebra.evaluation import plan_zebra_work_items, zebra_work_key
        from minibench.experiments.zebra import _zebra_code_sha256, _zebra_fingerprint_inputs
        self.batch = batch_dir
        self.workers = workers
        self.stack = ExitStack()
        self.contexts = {}
        self.futures = {}
        self.failures = []
        self.started = utcnow()
        manifest = read_json_object(batch_dir / "batch-manifest.json")
        if _zebra_code_sha256() != manifest["code_sha256"]:
            raise ValueError("frozen source does not match the original batch fingerprint")
        data = batch_dir / "history_eval.jsonl"
        if sha256_file(data) != manifest["data_sha256"]:
            raise ValueError("frozen task data does not match the original batch")
        self.tasks = load_zebra_tasks(data)
        for treatment in ("thinking_off", "thinking_on"):
            config = yaml.safe_load((batch_dir / "configs" / f"{treatment}.yaml").read_text(encoding="utf-8"))
            if config["agent"]["name"] != "direct" or config["provider"]["model"] != "qwen3.8-flash":
                raise ValueError("unexpected frozen agent/model")
            plan = plan_zebra_work_items(self.tasks, config["evaluation"]["memory_modes"])
            keys = [zebra_work_key(t.id, m) for t, m in plan]
            inputs = _zebra_fingerprint_inputs(config, task_path=data, plan_keys=keys)
            self.contexts[treatment] = {
                "config": config, "plan": plan, "plan_index": {k: i for i, k in enumerate(keys)},
                "inputs": inputs, "owned": False, "results": {}, "submitted": set(),
                "run_dir": batch_dir / treatment,
            }

    def acquire(self, treatment: str) -> bool:
        from minibench.core.checkpoint import RunLock, RunLockedError, atomic_write_json, fingerprint_payload, read_json_object
        from minibench.datasets.zebra.evaluation import zebra_result_key
        from minibench.experiments.zebra import (
            _load_zebra_predictions, _validate_zebra_manifest, _zebra_sidecar_lock_name,
        )
        ctx = self.contexts[treatment]
        if ctx["owned"]:
            return True
        lock = RunLock(self.batch, lock_name=_zebra_sidecar_lock_name(treatment))
        try:
            self.stack.enter_context(lock)
        except RunLockedError:
            return False
        # Own this lock for the full coordinator lifetime, including key entry.
        ctx["owned"] = True
        run_dir = ctx["run_dir"]
        expected = {"schema_version": 1, "family": "zebra", "created_at": utcnow(),
                    "fingerprint_inputs": ctx["inputs"], "fingerprint": fingerprint_payload(ctx["inputs"])}
        if run_dir.exists():
            actual = _validate_zebra_manifest(run_dir, expected)
            rows = _load_zebra_predictions(run_dir, tasks=self.tasks, work_plan=ctx["plan"], plan_index=ctx["plan_index"])
            ctx["created_at"] = actual["created_at"]
            state = read_json_object(run_dir / "run_state.json")
            ctx["resume_count"] = state["resume_count"] + (state["status"] != "completed")
            ctx["results"] = {zebra_result_key(row): row for row in rows}
        else:
            run_dir.mkdir()
            atomic_write_json(run_dir / "manifest.json", expected)
            ctx["created_at"] = expected["created_at"]
            ctx["resume_count"] = 0
        before_recovery = len(ctx["results"])
        self.recover_journal(treatment)
        if len(ctx["results"]) < len(ctx["plan"]) or len(ctx["results"]) != before_recovery:
            self.snapshot(treatment)
        print(f"Acquired {treatment}: reuse {len(ctx['results'])}/{len(ctx['plan'])} completed instances", flush=True)
        return True

    def recover_journal(self, treatment: str) -> None:
        from minibench.core.checkpoint import read_json_object
        from minibench.datasets.zebra.evaluation import score_zebra_output, zebra_result_from_dict, zebra_result_key
        ctx = self.contexts[treatment]
        task_map = {t.id: t for t in self.tasks}
        for path in sorted((self.batch / "parallel-journal" / treatment).glob("*.json")):
            saved = read_json_object(path)
            row = zebra_result_from_dict(saved)
            key = zebra_result_key(row)
            if key not in ctx["plan_index"]:
                raise ValueError("journal contains an unplanned work key")
            task = task_map[row.task_id]
            for name in ("source_id", "variant", "size", "difficulty", "capability", "rule_mode", "tags"):
                if getattr(row, name) != getattr(task, name):
                    raise ValueError(f"journal metadata mismatch: {name}")
            scored = score_zebra_output(task, row.raw_output)
            if any(
                getattr(row, key) != expected
                for key, expected in scored.items()
                if key in saved or key not in {"is_legal", "legality_reasons"}
            ):
                raise ValueError("journal scores do not match output")
            row = replace(row, is_legal=scored["is_legal"],
                          legality_reasons=scored["legality_reasons"])
            if key in ctx["results"] and ctx["results"][key] != row:
                raise ValueError("conflicting duplicate journal result")
            ctx["results"][key] = row

    def snapshot(self, treatment: str, *, interrupted=False) -> None:
        from minibench.datasets.zebra.evaluation import write_zebra_snapshot, zebra_result_key
        from minibench.experiments.zebra import _write_zebra_state
        ctx = self.contexts[treatment]
        rows = sorted(ctx["results"].values(), key=lambda r: ctx["plan_index"][zebra_result_key(r)])
        total = len(ctx["plan"])
        status = "completed" if len(rows) == total else "interrupted" if interrupted else "running"
        write_zebra_snapshot(ctx["run_dir"], rows, planned_total=total, run_status=status)
        _write_zebra_state(ctx["run_dir"], status=status, planned_total=total, completed_total=len(rows),
                           created_at=ctx["created_at"], resume_count=ctx["resume_count"])

    def status(self, state: str) -> dict:
        from minibench.core.checkpoint import atomic_write_json, read_json_object
        external = sum(not c["owned"] for c in self.contexts.values())
        counts = {}
        for name, ctx in self.contexts.items():
            if ctx["owned"]:
                counts[name] = len(ctx["results"])
            else:
                counts[name] = read_json_object(ctx["run_dir"] / "run_state.json")["completed_total"]
        value = {"status": state, "started_at": self.started, "updated_at": utcnow(),
                 "workers": self.workers, "external_workers": external,
                 "parallel_capacity": max(0, self.workers - external),
                 "active_jobs": [dict(treatment=t, task_id=k[0], mode=k[1]) for t, k in self.futures.values()],
                 "completed_by_treatment": counts, "completed_total": sum(counts.values()),
                 "planned_total": sum(len(c["plan"]) for c in self.contexts.values()), "failures": self.failures}
        atomic_write_json(self.batch / "parallel-status.json", value)
        return value

    def pending(self):
        from minibench.datasets.zebra.evaluation import zebra_work_key
        for treatment, ctx in self.contexts.items():
            if not ctx["owned"]:
                continue
            for task, mode in ctx["plan"]:
                key = zebra_work_key(task.id, mode)
                if key not in ctx["results"] and key not in ctx["submitted"]:
                    yield treatment, key, task, mode

    def record_batch_completion(self) -> None:
        from minibench.core.checkpoint import atomic_write_json
        if not all(c["owned"] and len(c["results"]) == len(c["plan"]) for c in self.contexts.values()):
            raise RuntimeError("cannot mark an incomplete batch completed")
        atomic_write_json(self.batch / "batch-status.json", {
            "status": "completed", "completed_treatments": list(self.contexts),
            "current_treatment": None, "updated_at": utcnow(), "workers": self.workers,
            "continued_with": "parallel-status.json", "batch_dir": str(self.batch),
        })

    def run(self) -> int:
        from minibench.core.checkpoint import atomic_write_json
        from minibench.datasets.zebra.evaluation import zebra_result_from_dict, zebra_result_key
        for name in self.contexts:
            self.acquire(name)
        initial = self.status("awaiting_api_key")
        if not any(self.pending()) and not initial["external_workers"]:
            self.status("completed")
            self.record_batch_completion()
            return 0
        if not sys.stdin.isatty():
            raise RuntimeError("use an interactive terminal for hidden API key entry")
        print(f"Total concurrency: {self.workers}; existing run uses {initial['external_workers']} slot(s).")
        print("Completed instances are reused. Existing in-flight requests are not interrupted.")
        key = getpass.getpass("DashScope API key (hidden; paste then Enter): ").strip()
        if not key:
            raise ValueError("no key entered; no new model requests sent")
        os.environ["DASHSCOPE_API_KEY"] = key
        atomic_write_json(self.batch / "parallel-resume-manifest.json", {
            "started_at": self.started, "workers": self.workers, "frozen_source": "source/minibench",
            "initial_completed": initial["completed_by_treatment"], "protocol_changed": False,
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        })
        stopping = False
        with ProcessPoolExecutor(max_workers=self.workers, mp_context=multiprocessing.get_context("spawn"),
                                 initializer=worker_bootstrap, initargs=(self.batch,)) as pool:
            while True:
                try:
                    if not stopping:
                        try:
                            for name in self.contexts:
                                self.acquire(name)
                            external = sum(not c["owned"] for c in self.contexts.values())
                            available = max(0, self.workers - external - len(self.futures))
                            for treatment, work_key, task, mode in self.pending():
                                if not available:
                                    break
                                ctx = self.contexts[treatment]
                                future = pool.submit(run_job, task, mode, ctx["config"])
                                self.futures[future] = treatment, work_key
                                ctx["submitted"].add(work_key)
                                available -= 1
                        except Exception as exc:
                            # Submission/lock errors must not discard other paid requests.
                            self.failures.append({"error": f"{type(exc).__name__}: {exc}".replace(key, "[REDACTED]")})
                            stopping = True
                    external = sum(not c["owned"] for c in self.contexts.values())
                    state = self.status("draining" if stopping else "running")
                    if not self.futures:
                        if stopping or not external:
                            break
                        time.sleep(3)
                        continue
                    ready, _ = wait(self.futures, timeout=3, return_when=FIRST_COMPLETED)
                    for future in ready:
                        treatment, work_key = self.futures.pop(future)
                        ctx = self.contexts[treatment]
                        try:
                            raw = future.result()
                            row = zebra_result_from_dict(raw)
                            if zebra_result_key(row) != work_key:
                                raise ValueError("worker returned wrong work key")
                            filename = hashlib.sha256(json.dumps(work_key).encode()).hexdigest() + ".json"
                            atomic_write_json(self.batch / "parallel-journal" / treatment / filename, raw)
                            ctx["results"][work_key] = row
                            self.snapshot(treatment)
                            print(f"{treatment} {len(ctx['results'])}/300 completed: {work_key[0]}/{work_key[1]}", flush=True)
                        except Exception as exc:
                            self.failures.append({"treatment": treatment, "work_key": work_key,
                                                  "error": f"{type(exc).__name__}: {exc}".replace(key, "[REDACTED]")})
                            stopping = True
                except KeyboardInterrupt:
                    stopping = True
                    self.failures.append({"error": "Interrupted: draining already submitted jobs"})
        for name, ctx in self.contexts.items():
            if ctx["owned"] and len(ctx["results"]) < len(ctx["plan"]):
                self.snapshot(name, interrupted=True)
        final = self.status("interrupted" if stopping else "completed")
        if final["status"] == "completed":
            self.record_batch_completion()
        print(json.dumps(final, indent=2), flush=True)
        return 1 if stopping else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch_dir", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--inspect", action="store_true", help="verify source/configs and show saved progress only")
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError("workers must be from 1 to 4")
    batch_dir = args.batch_dir.resolve()
    bootstrap(batch_dir)
    from minibench.core.checkpoint import RunLock, read_json_object
    coordinator = Coordinator(batch_dir, args.workers)
    if args.inspect:
        for name, ctx in coordinator.contexts.items():
            path = ctx["run_dir"] / "run_state.json"
            print(name, read_json_object(path) if path.exists() else "not started")
        print("Frozen source/config verification passed; no API calls.")
        return 0
    try:
        with RunLock(batch_dir, lock_name=".parallel-coordinator.lock"), coordinator.stack:
            return coordinator.run()
    finally:
        os.environ.pop("DASHSCOPE_API_KEY", None)


if __name__ == "__main__":
    raise SystemExit(main())
