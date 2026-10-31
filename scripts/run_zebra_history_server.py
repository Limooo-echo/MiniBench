"""Run a frozen Zebra history batch with raw HTTP journals and up to 16 workers.

Only successful, checksum-verified calls are replayed. No automatic HTTP retry
is added. A worker failure stops submissions and drains other paid requests.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import getpass
import hashlib
from io import BytesIO
import json
import multiprocessing
import os
from pathlib import Path
import signal
import sys
import time
import urllib.error
import urllib.request
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import resume_zebra_history_parallel as parallel


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def bootstrap(batch: Path) -> None:
    source = batch / "source"
    if not (source / "minibench").is_dir():
        raise ValueError("frozen source snapshot is missing")
    sys.path.insert(0, str(source))
    os.chdir(batch)


def worker_bootstrap(batch: Path) -> None:
    bootstrap(batch)
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def work_hash(task_id: str, mode: str) -> str:
    return hashlib.sha256(json.dumps((task_id, mode)).encode()).hexdigest()


class RawCallJournal:
    """Capture exactly the urllib request body and complete response body.

    Each process runs one synchronous model client. Patching urlopen is scoped
    to that single call; headers and environment values are never serialized.
    """

    def __init__(self, client, root: Path, *, task_id: str, mode: str,
                 treatment: str, planned_calls: int):
        from minibench.core.checkpoint import atomic_write_json
        self.client, self.root = client, root
        self.calls = root / "calls"
        self.calls.mkdir(parents=True, exist_ok=True)
        self.index = 0
        self.planned_calls = planned_calls
        self.original = client._complete_request_once
        self.started = utcnow()
        metadata = {"schema_version": 1, "task_id": task_id, "mode": mode,
                    "treatment": treatment, "planned_calls": planned_calls}
        metadata_path = root / "metadata.json"
        if metadata_path.exists():
            if json.loads(metadata_path.read_text(encoding="utf-8")) != metadata:
                raise ValueError("raw job metadata differs from the resumed job")
        else:
            atomic_write_json(metadata_path, metadata)
        self.progress("starting")

    def progress(self, state: str, **extra) -> None:
        from minibench.core.checkpoint import atomic_write_json
        atomic_write_json(self.root / "progress.json", {
            "status": state, "updated_at": utcnow(), "started_at": self.started,
            "pid": os.getpid(), "completed_calls": self.index,
            "planned_calls": self.planned_calls, **extra,
        })

    def event(self, state: str, **extra) -> None:
        with (self.root / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"at": utcnow(), "event": state, **extra}) + "\n")
            handle.flush()

    def clean(self, value: str) -> str:
        secret = os.environ.get(self.client.api_key_env)
        return value.replace(secret, "[REDACTED]") if secret else value

    def complete_request_once(self, request):
        from minibench.core.agent import CompletionResult
        from minibench.core.checkpoint import atomic_write_json, atomic_write_text, fingerprint_payload
        index = self.index
        if index >= self.planned_calls:
            raise ValueError("model made more calls than the frozen history plan")
        stem = self.calls / f"{index:04d}"
        request_bytes = request.data or b""
        payload = json.loads(request_bytes)
        identity = fingerprint_payload({"endpoint": request.full_url, "payload": payload})
        request_path = stem.with_suffix(".request.json")
        if request_path.exists():
            saved_payload = json.loads(request_path.read_text(encoding="utf-8"))
            if fingerprint_payload({"endpoint": request.full_url, "payload": saved_payload}) != identity:
                raise ValueError(f"resumed request differs at call {index}")
            if stem.with_suffix(".request.raw").read_bytes() != request_bytes:
                raise ValueError(f"resumed request bytes differ at call {index}")
        else:
            atomic_write_text(stem.with_suffix(".request.raw"), request_bytes.decode("utf-8"))
            atomic_write_json(request_path, payload)
        record_path = stem.with_suffix(".record.json")
        self.progress("calling", active_call=index, phase="final" if index + 1 == self.planned_calls else "clue")
        if record_path.exists():
            record = json.loads(record_path.read_text(encoding="utf-8"))
            checksum = record.pop("record_sha256", None)
            if checksum != fingerprint_payload(record) or record.get("status") != "completed":
                raise ValueError(f"invalid saved call record {index}")
            if record.get("request_sha256") != identity:
                raise ValueError(f"saved call fingerprint differs at call {index}")
            raw = stem.with_suffix(".response.raw").read_bytes()
            if hashlib.sha256(raw).hexdigest() != record["response_raw_sha256"]:
                raise ValueError(f"saved response checksum differs at call {index}")
            if json.loads(raw) != json.loads(stem.with_suffix(".response.json").read_text(encoding="utf-8")):
                raise ValueError(f"saved response JSON differs at call {index}")
            result = CompletionResult(**record["response"])
            self.client._record_completion_metrics(result.elapsed_seconds or 0, result.usage)
            if self.client._completion_traces is not None and record.get("completion_trace") is not None:
                self.client._completion_traces.append(deepcopy(record["completion_trace"]))
            self.index += 1
            self.event("replayed", call_index=index, request_sha256=identity)
            self.progress("running")
            return result

        actual_urlopen = urllib.request.urlopen
        response_bytes = None
        reparsed_saved_response = False

        def capture_transport(actual_request, *args, **kwargs):
            nonlocal response_bytes, reparsed_saved_response
            if actual_request is not request:
                raise ValueError("unexpected nested transport request")
            raw_path = stem.with_suffix(".response.raw")
            if raw_path.exists():
                # A complete HTTP response persisted before a process crash can
                # be parsed again without charging for another model request.
                response_bytes = raw_path.read_bytes()
                reparsed_saved_response = True
                try:
                    atomic_write_json(stem.with_suffix(".response.json"), json.loads(response_bytes))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    pass
                self.event("reparse_saved_response", call_index=index)
                return BytesIO(response_bytes)
            try:
                with actual_urlopen(actual_request, *args, **kwargs) as response:
                    response_bytes = response.read()
            except urllib.error.HTTPError as exc:
                body = exc.read()
                atomic_write_json(stem.with_suffix(f".error-{time.time_ns()}.json"), {
                    "at": utcnow(), "status_code": exc.code,
                    "body": self.clean(body.decode("utf-8", errors="replace")),
                })
                raise urllib.error.HTTPError(exc.url, exc.code, exc.msg, exc.headers, BytesIO(body)) from exc
            # Raw response bytes are UTF-8 JSON for this non-streaming frozen
            # protocol. Persist before handing them to the original parser.
            atomic_write_text(raw_path, response_bytes.decode("utf-8"))
            try:
                atomic_write_json(stem.with_suffix(".response.json"), json.loads(response_bytes))
            except (UnicodeDecodeError, json.JSONDecodeError):
                pass  # The original client reports malformed responses.
            return BytesIO(response_bytes)

        trace_count = len(self.client._completion_traces or [])
        try:
            with patch.object(urllib.request, "urlopen", capture_transport):
                result = self.original(request)
            trace = None
            if self.client._completion_traces is not None and len(self.client._completion_traces) > trace_count:
                trace = deepcopy(self.client._completion_traces[-1])
            record = {"schema_version": 1, "status": "completed", "at": utcnow(),
                      "request_sha256": identity, "endpoint": request.full_url,
                      "response_raw_sha256": hashlib.sha256(response_bytes).hexdigest(),
                      "response": asdict(result), "completion_trace": trace,
                      "original_elapsed_unknown": reparsed_saved_response}
            record["record_sha256"] = fingerprint_payload(record)
            atomic_write_json(record_path, record)
            self.index += 1
            self.event("completed", call_index=index, request_sha256=identity)
            self.progress("running")
            return result
        except BaseException as exc:
            detail = self.clean(f"{type(exc).__name__}: {exc}")
            self.progress("failed", active_call=index, error=detail)
            self.event("failed", call_index=index, error=detail)
            raise


def run_job(batch: Path, treatment: str, task, mode: str, config: dict) -> dict:
    from minibench.datasets.zebra.evaluation import evaluate_zebra_work_item
    from minibench.datasets.zebra.prompting import ZEBRA_SYSTEM_PROMPT
    from minibench.factory.agents import make_agent_from_config
    evaluation = config["evaluation"]
    agent = make_agent_from_config(config["agent"], config["provider"], system_prompt=ZEBRA_SYSTEM_PROMPT)
    root = batch / "raw_data" / treatment / work_hash(task.id, mode)
    journal = RawCallJournal(agent.client, root, task_id=task.id, mode=mode,
                             treatment=treatment, planned_calls=len(task.clue_turns) + 1)
    agent.client._complete_request_once = journal.complete_request_once
    try:
        result = evaluate_zebra_work_item(
            task, agent, mode, state_max_tokens=evaluation["state_max_tokens"],
            ack_max_tokens=evaluation["ack_max_tokens"], final_max_tokens=evaluation["final_max_tokens"],
            state_validation_timeout_ms=evaluation["state_validation_timeout_ms"],
        )
        if journal.index != journal.planned_calls:
            raise ValueError("model completed fewer calls than the frozen history plan")
        journal.progress("completed")
        return asdict(result)
    except BaseException as exc:
        journal.progress("failed", error=journal.clean(f"{type(exc).__name__}: {exc}"))
        raise


def load_credentials(files: list[Path]) -> None:
    """Load only recognized credential assignments, without shell evaluation."""
    if os.environ.get("DASHSCOPE_API_KEY"):
        return
    allowed = {"DASHSCOPE_API_KEY", "DASHSCOPE_APIKEY", "QWEN_API_KEY", "API_KEY"}
    for path in files:
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if line.startswith("export "):
                line = line[7:].lstrip()
            name, equals, value = line.partition("=")
            if not equals or name.strip() not in allowed:
                continue
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            else:
                value = value.split(" #", 1)[0].rstrip()
            if value:
                os.environ["DASHSCOPE_API_KEY"] = value
                return


class Coordinator(parallel.Coordinator):
    def __init__(self, batch_dir: Path, workers: int):
        super().__init__(batch_dir, workers)
        for treatment, ctx in self.contexts.items():
            config = ctx["config"]
            if config["agent"] != {"name": "direct", "prompt_version": "v2", "max_format_repairs": 0,
                                   "final_temperature": 0.0, "trace": "full"}:
                raise ValueError("server experiment requires the frozen Direct v2 configuration")
            provider = config["provider"]
            if any(provider.get(key) != expected for key, expected in {
                "name": "qwen", "model": "qwen3.8-flash",
                "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "api_key_env": "DASHSCOPE_API_KEY", "temperature": 0.0,
                "max_tokens": 4096, "timeout": 600, "json_mode": True, "max_retries": 0,
            }.items()):
                raise ValueError("server experiment provider differs from the frozen configuration")
            expected_extra = {"enable_thinking": treatment == "thinking_on", "preserve_thinking": False, "stream": False}
            if treatment == "thinking_on":
                expected_extra["thinking_budget"] = 7168
            if provider["extra_body"] != expected_extra:
                raise ValueError("unexpected thinking or transport treatment")
            ev = config["evaluation"]
            if any(ev.get(key) != expected for key, expected in {
                "memory_modes": ["deferred_reasoning", "incremental_state", "incremental_state_pro"],
                "state_max_tokens": 2048, "ack_max_tokens": 32,
                "final_max_tokens": 4096, "state_validation_timeout_ms": 10000,
            }.items()):
                raise ValueError("server experiment evaluation differs from the frozen configuration")
        off, on = (deepcopy(self.contexts[t]["config"]) for t in ("thinking_off", "thinking_on"))
        for config in (off, on):
            config.pop("run", None)
            config["provider"]["extra_body"].pop("enable_thinking")
        on["provider"]["extra_body"].pop("thinking_budget")
        if off != on:
            raise ValueError("thinking treatments differ beyond their native thinking switches")

    def pending(self):
        from minibench.datasets.zebra.evaluation import zebra_work_key
        length = max(len(ctx["plan"]) for ctx in self.contexts.values())
        for index in range(length):
            for treatment, ctx in self.contexts.items():
                if not ctx["owned"] or index >= len(ctx["plan"]):
                    continue
                task, mode = ctx["plan"][index]
                key = zebra_work_key(task.id, mode)
                if key not in ctx["results"] and key not in ctx["submitted"]:
                    yield treatment, key, task, mode

    def status(self, state: str) -> dict:
        from minibench.core.checkpoint import atomic_write_json
        value = super().status(state)
        for job in value["active_jobs"]:
            path = self.batch / "raw_data" / job["treatment"] / work_hash(job["task_id"], job["mode"]) / "progress.json"
            if path.exists():
                job["call_progress"] = json.loads(path.read_text(encoding="utf-8"))
        atomic_write_json(self.batch / "parallel-status.json", value)
        return value

    def run(self) -> int:
        from minibench.core.checkpoint import atomic_write_json
        from minibench.datasets.zebra.evaluation import zebra_result_from_dict, zebra_result_key
        for treatment in self.contexts:
            if not self.acquire(treatment):
                raise RuntimeError("a treatment is already owned by another runner")
        initial = self.status("initializing")
        if not any(self.pending()):
            self.status("completed")
            self.record_batch_completion()
            return 0
        key = os.environ.get("DASHSCOPE_API_KEY", "")
        if not key:
            self.status("awaiting_api_key")
            if not sys.stdin.isatty():
                raise RuntimeError("DASHSCOPE_API_KEY is missing; enter it through an interactive terminal")
            key = getpass.getpass("DashScope API key (hidden; paste then Enter): ").strip()
            if not key:
                raise ValueError("no API key entered; no requests sent")
            os.environ["DASHSCOPE_API_KEY"] = key
        atomic_write_json(self.batch / "parallel-resume-manifest.json", {
            "started_at": self.started, "workers": self.workers,
            "frozen_source": "source/minibench", "initial_completed": initial["completed_by_treatment"],
            "protocol_changed": False, "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "raw_capture": "actual HTTP request/response bodies; successful calls replay by checksum",
        })
        plan = [{"treatment": t, "task_id": task.id, "mode": mode}
                for index in range(max(len(c["plan"]) for c in self.contexts.values()))
                for t, c in self.contexts.items() if index < len(c["plan"])
                for task, mode in [c["plan"][index]]]
        plan_path = self.batch / "server-plan.json"
        if plan_path.exists() and json.loads(plan_path.read_text(encoding="utf-8")) != plan:
            raise ValueError("saved server work plan differs")
        atomic_write_json(plan_path, plan)
        stopping, last_print = False, 0.0
        with ProcessPoolExecutor(max_workers=self.workers, mp_context=multiprocessing.get_context("spawn"),
                                 initializer=worker_bootstrap, initargs=(self.batch,)) as pool:
            while True:
                try:
                    if not stopping:
                        available = self.workers - len(self.futures)
                        try:
                            for treatment, key_tuple, task, mode in self.pending():
                                if not available:
                                    break
                                future = pool.submit(run_job, self.batch, treatment, task, mode,
                                                     self.contexts[treatment]["config"])
                                self.futures[future] = treatment, key_tuple
                                self.contexts[treatment]["submitted"].add(key_tuple)
                                available -= 1
                        except Exception as exc:
                            self.failures.append({"error": f"{type(exc).__name__}: {exc}".replace(key, "[REDACTED]")})
                            stopping = True
                    current = self.status("draining" if stopping else "running")
                    if time.monotonic() - last_print >= 30:
                        print(f"[{utcnow()}] completed {current['completed_total']}/{current['planned_total']}; "
                              f"active {len(self.futures)}/{self.workers}; failures {len(self.failures)}", flush=True)
                        last_print = time.monotonic()
                    if not self.futures:
                        break
                    ready, _ = wait(self.futures, timeout=3, return_when=FIRST_COMPLETED)
                    for future in ready:
                        treatment, key_tuple = self.futures.pop(future)
                        ctx = self.contexts[treatment]
                        try:
                            raw = future.result()
                            row = zebra_result_from_dict(raw)
                            if zebra_result_key(row) != key_tuple:
                                raise ValueError("worker returned the wrong key")
                            atomic_write_json(self.batch / "parallel-journal" / treatment /
                                              (work_hash(*key_tuple) + ".json"), raw)
                            ctx["results"][key_tuple] = row
                            self.snapshot(treatment)
                            print(f"{treatment} {len(ctx['results'])}/{len(ctx['plan'])} completed: "
                                  f"{key_tuple[0]}/{key_tuple[1]}", flush=True)
                        except Exception as exc:
                            self.failures.append({"treatment": treatment, "work_key": key_tuple,
                                                  "error": f"{type(exc).__name__}: {exc}".replace(key, "[REDACTED]")})
                            stopping = True
                except KeyboardInterrupt:
                    stopping = True
                    self.failures.append({"error": "Interrupted: draining submitted jobs"})
        for treatment, ctx in self.contexts.items():
            if len(ctx["results"]) < len(ctx["plan"]):
                self.snapshot(treatment, interrupted=True)
        complete = all(len(c["results"]) == len(c["plan"]) for c in self.contexts.values())
        self.status("completed" if complete else "interrupted")
        if complete:
            self.record_batch_completion()
        return 0 if complete else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch_dir", type=Path)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--credential-file", type=Path, action="append", default=[])
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error("workers must be from 1 to 16")
    batch = args.batch_dir.resolve()
    bootstrap(batch)
    from minibench.core.checkpoint import RunLock, read_json_object
    coordinator = Coordinator(batch, args.workers)
    if len(coordinator.tasks) != 100 or sum(len(task.clue_turns) for task in coordinator.tasks) != 978:
        raise ValueError("expected the complete frozen 100-task, 978-clue roster")
    if args.inspect:
        for treatment, ctx in coordinator.contexts.items():
            path = ctx["run_dir"] / "run_state.json"
            print(treatment, read_json_object(path) if path.exists() else "not started")
        print(f"Frozen source/data verification passed; {sum(len(c['plan']) for c in coordinator.contexts.values())} "
              f"instances; {args.workers} workers; no API calls.")
        return 0
    load_credentials(args.credential_file)
    try:
        with RunLock(batch, lock_name=".parallel-coordinator.lock"), coordinator.stack:
            return coordinator.run()
    finally:
        os.environ.pop("DASHSCOPE_API_KEY", None)


if __name__ == "__main__":
    raise SystemExit(main())
