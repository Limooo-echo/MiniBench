"""Portable API benchmark scheduler: isolated problems, durable calls and resume.

The native evaluators remain authoritative. A job owns a whole game and its
requested conditions. Worker processes isolate H2 engine replay and rendering.
No model is called by prepare or status.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict, is_dataclass
import json
import math
import multiprocessing as mp
from multiprocessing.managers import BaseManager
import os
from pathlib import Path
import re
import shutil
import signal
import statistics
import subprocess
import threading
import time
from urllib.parse import urlsplit

import yaml

from minibench.core.checkpoint import (
    RunLock, atomic_write_json, atomic_write_jsonl, fingerprint_payload,
    read_json_object, sha256_file,
)
from minibench.core.rate_limit import RateGate
from minibench.core.streaming import JournalClient, append_json, call_attempts, now, replay_h2_engine, safe_error
from minibench.factory.agents import AGENT_NAMES, REASONING_AGENTS
from minibench.agents.passthrough import PassthroughAgent
from minibench.core.agent import ReasoningConfig
from minibench.factory.config import load_experiment_config, validate_experiment_config
from minibench.experiments.dispatch import _evaluate
from minibench.experiments.xiangqi import (
    _sample_xiangqi_tasks,
    _select_frozen_xiangqi_tasks,
)
from minibench.experiments.selection import (
    _select_tasks,
    _select_mahjong_rule_configuration,
)
from minibench.experiments.registry import get_task_family_spec
from minibench.factory.providers import resolve_provider

ROOT = Path(__file__).resolve().parents[2]
VERSION = "minibench-cloud-v1"


class GateManager(BaseManager):
    pass


GateManager.register("RateGate", RateGate)


def task_id(task):
    return str(task["id"] if isinstance(task, dict) else task.id)


def portable_value(value):
    """JSON projection for audit samples; loaders still use canonical inputs."""
    if is_dataclass(value):
        return portable_value(asdict(value))
    if isinstance(value, Path):
        resolved = value.resolve()
        if not resolved.is_relative_to(ROOT):
            raise ValueError("Dataset asset path is outside repository")
        return resolved.relative_to(ROOT).as_posix()
    if isinstance(value, dict):
        return {key: portable_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [portable_value(item) for item in value]
    return value


def input_images(value, source):
    """Resolve assets as the native JSONL loaders do, including Path fields."""
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"image", "image_path"} and isinstance(item, (str, Path)) and item:
                path = Path(item)
                yield path if path.is_absolute() else source.parent / path
            elif key == "image" and isinstance(item, dict) and item.get("path"):
                path = Path(item["path"])
                yield path if path.is_absolute() else source.parent / path
            else:
                yield from input_images(item, source)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from input_images(item, source)


def reject_secrets(value):
    """Accept env-variable names, never credentials in an on-disk config."""
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in {"api_key", "authorization", "headers", "password", "access_token"}:
                raise ValueError(f"Credentials/headers are forbidden in configuration: {key}")
            reject_secrets(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            reject_secrets(item)
    elif isinstance(value, str) and re.search(r"\bsk-[A-Za-z0-9]", value):
        raise ValueError("Possible plaintext API key in configuration")


def code_hash():
    paths = list((ROOT / "src").rglob("*.py"))
    paths += list((ROOT / "scripts").glob("*.py"))
    return fingerprint_payload({str(p.relative_to(ROOT)): sha256_file(p) for p in sorted(paths)})


def select_tasks(config, pilot_count=None):
    family = config["task"]["family"]
    spec = get_task_family_spec(family)
    tc = config["task"]
    path = Path(tc.get("path") or spec.default_path)
    tasks = spec.load_tasks(path)
    if tc.get("selection"):
        tasks = _select_frozen_xiangqi_tasks(tasks, tc, path, family)
    tasks = _select_tasks(tasks, tc.get("task_ids") or [])
    if (tc.get("sampling") or {}).get("enabled"):
        sc = tc["sampling"]
        tasks = _sample_xiangqi_tasks(tasks, family=family, count=sc["count"],
                                      seed=sc["seed"], strategy=sc["strategy"])
    if family == "mahjong_rule_variants":
        tasks = _select_mahjong_rule_configuration(tasks, config.get("evaluation", {}))
    if tc.get("limit") is not None:
        if family == "mahjong_rule_variants":
            sources = list(dict.fromkeys(t.source_task_id for t in tasks))[:tc["limit"]]
            tasks = [t for t in tasks if t.source_task_id in sources]
        else:
            tasks = tasks[:tc["limit"]]
    if pilot_count and len(tasks) > pilot_count:
        if family.startswith("xiangqi-"):
            strategy = "paired-stratified" if family == "xiangqi-rule-variants" else "stratified"
            # C2 sampler counts paired scenarios, not individual condition rows.
            if family == "xiangqi-rule-variants" and pilot_count % 2:
                raise ValueError("C2 pilot_count must be even to retain standard/variant pairs")
            sample_count = pilot_count // 2 if family == "xiangqi-rule-variants" else pilot_count
            tasks = _sample_xiangqi_tasks(tasks, family=family, count=sample_count,
                                         seed=42, strategy=strategy)
        else:
            # A pilot is a frozen throughput probe, not a statistical estimate.
            tasks = tasks[:pilot_count]
    if not tasks or len({task_id(t) for t in tasks}) != len(tasks):
        raise ValueError(f"{family}: empty selection or duplicate task IDs")
    return spec, path, tasks


def freeze_file(source: Path, suite: Path):
    source = source.resolve()
    if not source.is_relative_to(ROOT):
        raise ValueError(f"Portable suite inputs must be inside the repository: {source}")
    digest = sha256_file(source)
    relative = Path("inputs") / source.relative_to(ROOT)
    target = suite / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copyfile(source, target)
    if sha256_file(target) != digest:
        raise ValueError("Frozen file collision")
    return {"path": relative.as_posix(), "sha256": digest}


def prepare(config_path: Path, suite: Path, *, pilot_count: int | None = None):
    os.chdir(ROOT)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    reject_secrets(raw)
    if not isinstance(raw, dict) or raw.get("version") != 1:
        raise ValueError("Cloud configuration version must be 1")
    allowed = {"version", "workers", "accounts", "experiments", "agents", "omit_max_tokens", "pilot_count", "engine_binding"}
    if set(raw) - allowed:
        raise ValueError("Unknown cloud configuration fields")
    accounts = raw.get("accounts")
    experiments = raw.get("experiments")
    if not isinstance(accounts, dict) or not accounts or not isinstance(experiments, list) or not experiments:
        raise ValueError("accounts and experiments must be nonempty")
    agents = raw.get("agents", ["direct"])
    if not isinstance(agents, list) or not agents or len(set(agents)) != len(agents) or any(a not in AGENT_NAMES for a in agents):
        raise ValueError("Invalid or duplicate agent names")
    workers = raw.get("workers", 8)
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 64:
        raise ValueError("workers must be an integer in 1..64")
    count = pilot_count if pilot_count is not None else raw.get("pilot_count")
    if count is not None and (isinstance(count, bool) or not isinstance(count, int) or count < 1):
        raise ValueError("pilot_count must be positive")
    if not isinstance(raw.get("omit_max_tokens", True), bool):
        raise ValueError("omit_max_tokens must be boolean")
    engine_binding = raw.get("engine_binding", "reference")
    if engine_binding not in {"reference", "runtime"}:
        raise ValueError("engine_binding must be reference or runtime")
    providers, groups, fingerprints = {}, {}, {}
    for name, account in accounts.items():
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError("Unsafe account name")
        if set(account) - {"provider", "limits"}:
            raise ValueError("Unknown account fields")
        p = deepcopy(account["provider"])
        model, base, env = resolve_provider(p.get("name", "generic"), model=p.get("model"),
                                           base_url=p.get("base_url"), api_key_env=p.get("api_key_env"))
        url = urlsplit(base)
        if url.scheme not in {"https", "http"} or url.username or url.password or url.query or url.fragment:
            raise ValueError("Invalid API base URL or credentials in URL")
        if url.scheme == "http" and url.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("Remote API must use HTTPS")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", env):
            raise ValueError("Invalid api_key_env")
        if p.get("max_retries", 0) != 0:
            raise ValueError("Cloud transport disables automatic retries to avoid unknown rebilling")
        p.update(model=model, base_url=base, api_key_env=env, max_retries=0)
        limits = deepcopy(account.get("limits", {"max_in_flight": workers}))
        group = fingerprint_payload({"host": url.hostname, "env": env})[:16]
        if group in groups and groups[group] != limits:
            raise ValueError("Same host/env account aliases must share identical limits")
        groups[group] = limits
        providers[name] = {"provider": p, "rate_group": group}
        fingerprints[name] = group
    RateGate(groups)  # validate before creating a new output directory
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = "source-archive"
    suite.mkdir(parents=True, exist_ok=False)
    plan = {"version": VERSION, "created_at": now(), "workers": workers,
            "omit_max_tokens": raw.get("omit_max_tokens", True), "accounts": providers,
            "limits": groups, "agents": agents, "code_sha256": code_hash(),
            "experiments": {}, "jobs": [], "files": [], "pilot_count": count,
            "base_commit": commit, "engine_binding": engine_binding}
    seen, file_paths = set(), set()
    for item in experiments:
        if set(item) - {"id", "config", "account", "agent_overrides", "evaluation_overrides"}:
            raise ValueError("Unknown experiment fields")
        name = item["id"]
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name) or name in seen:
            raise ValueError("Invalid/duplicate experiment ID")
        seen.add(name)
        account = item["account"]
        if account not in providers:
            raise ValueError(f"Unknown account: {account}")
        config = load_experiment_config(ROOT / item["config"])
        config["provider"] = deepcopy(providers[account]["provider"])
        config.setdefault("evaluation", {}).update(item.get("evaluation_overrides", {}))
        validate_experiment_config(config)
        spec, source, tasks = select_tasks(config, count)
        family = config["task"]["family"]
        tree_directory = config.get("evaluation",{}).get("solution_tree_dir")
        if family in {"xiangqi-history","xiangqi-rule-variants"} and tree_directory:
            from minibench.datasets.xiangqi.solution_tree import TreeSession
            from minibench.datasets.xiangqi.tree_certificate import (
                REPORT_FILENAME, require_saved_tree_certificate,
            )
            directory = Path(tree_directory)
            certified = [asdict(task) if is_dataclass(task) else task for task in tasks]
            certified = [record for record in certified if record.get("shortest_proof")]
            if certified:
                proof_path = source.resolve().parent.parent / REPORT_FILENAME
                require_saved_tree_certificate(certified, data_path=source, tree_dir=directory,
                                               report_path=proof_path)
                independent_file = freeze_file(proof_path, suite)
                if independent_file["path"] not in file_paths:
                    plan["files"].append(independent_file)
                    file_paths.add(independent_file["path"])
            for task in tasks:
                raw = asdict(task) if is_dataclass(task) else task
                TreeSession(raw,directory)  # Reject unknown shortest before any paid request.
                tree_file = freeze_file(directory/f"{task_id(task)}.json",suite)
                if tree_file["path"] not in file_paths:
                    plan["files"].append(tree_file)
                    file_paths.add(tree_file["path"])
            config["evaluation"]["solution_tree_dir"] = str(
                Path("inputs")/directory.resolve().relative_to(ROOT))
        if family == "xiangqi-rule-variants":
            candidate = source.resolve().parent / "runtime_certificates.json"
            if candidate.is_file():
                certificate_file = freeze_file(candidate, suite)
                if certificate_file["path"] not in file_paths:
                    plan["files"].append(certificate_file)
                    file_paths.add(certificate_file["path"])
        reference_engine = deepcopy(config.get("evaluation", {}))
        if family.startswith("xiangqi-") and family != "xiangqi-rule-variants" and engine_binding == "runtime":
            from minibench.datasets.xiangqi.engines.pikafish import resolve_pikafish_executable, pikafish_fingerprint
            binary = resolve_pikafish_executable()
            identity = pikafish_fingerprint(binary)
            config["evaluation"].pop("pikafish_path", None)
            config["evaluation"]["pikafish_binary_sha256"] = identity["binary_sha256"]
            config["evaluation"]["pikafish_nnue_sha256"] = identity["eval_file_sha256"]
        configs = {}
        for agent_name in agents:
            c = deepcopy(config)
            c["agent"] = {"name": agent_name}
            if agent_name != "passthrough":
                c["agent"].update(prompt_version="v2", trace="summary")
                c["agent"].update(item.get("agent_overrides", {}))
            configs[agent_name] = validate_experiment_config(c)
        frozen = freeze_file(source, suite)
        if frozen["path"] not in file_paths:
            plan["files"].append(frozen)
            file_paths.add(frozen["path"])
        # Some loaders validate images for every row before selection. Retain
        # all referenced assets beside the frozen JSONL, not just pilot images.
        for t in spec.load_tasks(source):
            for image in input_images(t, source.resolve()):
                image_info = freeze_file(image, suite)
                if image_info["path"] not in file_paths:
                    plan["files"].append(image_info)
                    file_paths.add(image_info["path"])
        selected = [task_id(t) for t in tasks]
        atomic_write_json(suite / "samples" / f"{name}.json", [portable_value(t) for t in tasks])
        plan["experiments"][name] = {
            "family": family, "account": account, "source": frozen["path"],
            "task_ids": selected, "record_count": len(tasks), "configs": configs,
            "reference_config": item["config"],
            "reference_engine_evaluation": reference_engine,
        }
        if family == "xiangqi-history":
            mode = config.get("evaluation", {}).get("history_mode", "paired")
            modes = ["full-state", "move-history-only"] if mode == "paired" else [mode]
        elif family == "xiangqi-multimodal":
            modes = config.get("evaluation", {}).get("input_modes", ["text", "chinese-piece-image"])
        else:
            modes = ["native"]
        condition_counts = {mode: len(tasks) for mode in modes}
        if family == "xiangqi-rule-variants":
            condition_counts = {"all": len(tasks)}
            for t in tasks:
                condition_counts[t["ruleset"]] = condition_counts.get(t["ruleset"], 0) + 1
        plan["experiments"][name]["condition_counts"] = condition_counts
        # Interleave architectures; do not fill all slots with one architecture.
        for t in tasks:
            for agent_name in agents:
                for mode in modes:
                    identity = [name, agent_name, task_id(t), mode]
                    job={"key": fingerprint_payload(identity),
                         "experiment": name, "agent": agent_name, "task_id": task_id(t), "mode": mode}
                    if family == "xiangqi-rule-variants":
                        job["c2_phase"] = "standard" if t["ruleset"]=="standard" else "variant"
                    plan["jobs"].append(job)
    plan["planned_jobs"] = len(plan["jobs"])
    plan["plan_sha256"] = fingerprint_payload(plan)
    atomic_write_json(suite / "plan.json", plan)
    atomic_write_json(suite / "state.json", {
        "status": "prepared", "completed_jobs": 0, "planned_jobs": len(plan["jobs"]), "updated_at": now(),
    })
    return plan



def c2_phase_dependencies(plan):
    """Each architecture finishes its standard baseline before any C2 variant."""
    baselines={}
    for job in plan["jobs"]:
        if job.get("c2_phase")=="standard":
            baselines.setdefault((job["experiment"],job["agent"]),set()).add(job["key"])
    return {job["key"]:baselines.get((job["experiment"],job["agent"]),set())
            for job in plan["jobs"] if job.get("c2_phase")=="variant"}


def build_agent(name, client, config):
    if name == "passthrough":
        return PassthroughAgent(client)
    options = dict(config)
    options.pop("name", None)
    options.pop("predictions", None)
    if "max_tokens" in options:
        options["final_max_tokens"] = options.pop("max_tokens")
    return REASONING_AGENTS[name](client, ReasoningConfig(**options))


def _interrupt_worker(signum, frame):
    raise KeyboardInterrupt("Cloud worker interrupted; partial calls retained")


def _manager_initializer():
    # Only the scheduler handles terminal Ctrl+C. Keep the accounting server
    # alive while workers close their streams and release reservations.
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _worker_initializer():
    # Give the worker and its engines a private group for bounded fault cleanup.
    if hasattr(os, "setsid"):
        os.setsid()
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)


def _shutdown_pool(pool, *, failed, grace_seconds=15):
    """Let idle workers leave through pool sentinels, not abrupt queue exits.

    Active workers handle SIGTERM in execute_job and journal their partial
    response. A watchdog kills only this pool's process groups if cooperative
    cleanup stalls; ordinary successful shutdown has no watchdog.
    """
    processes = list((pool._processes or {}).values())
    forced = []
    def force_stop():
        for process in processes:
            if process.is_alive():
                try:
                    if hasattr(os, "killpg") and os.getpgid(process.pid) == process.pid:
                        os.killpg(process.pid, signal.SIGKILL)
                    else:
                        process.kill()
                    forced.append(process.pid)
                except ProcessLookupError:
                    pass
    watchdog = threading.Timer(grace_seconds, force_stop) if failed else None
    if watchdog is not None:
        watchdog.daemon = True
        watchdog.start()
    try:
        pool.shutdown(wait=True, cancel_futures=True)
    finally:
        if watchdog is not None:
            watchdog.cancel()
            watchdog.join()
    return forced


def execute_job(suite_text, plan, job, gate):
    """Entire episode executes serially in one isolated process."""
    os.chdir(ROOT)
    previous_term = signal.signal(signal.SIGTERM, _interrupt_worker)
    suite = Path(suite_text)
    directory = suite / "jobs" / job["key"]
    directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    experiment = plan["experiments"][job["experiment"]]
    config = deepcopy(experiment["configs"][job["agent"]])
    family = experiment["family"]
    account = plan["accounts"][experiment["account"]]
    provider = config["provider"]
    spec = get_task_family_spec(family)
    try:
        selected = [t for t in spec.load_tasks(suite / experiment["source"]) if task_id(t) == job["task_id"]]
        if len(selected) != 1:
            raise ValueError("Frozen task could not be uniquely loaded")
        def event(value):
            if value["kind"] != "stream_progress":
                append_json(directory / "events.jsonl", value)
            atomic_write_json(directory / "progress.json", {**job, **value})
        client = JournalClient(
            directory / "calls", event, gate=gate, account=account["rate_group"],
            omit_max_tokens=plan["omit_max_tokens"],
            model=provider["model"], base_url=provider["base_url"],
            api_key_env=provider["api_key_env"], timeout=provider.get("timeout", 3600),
            temperature=provider.get("temperature", 0), max_tokens=provider.get("max_tokens", 4096),
            json_mode=provider.get("json_mode", False), extra_body=provider.get("extra_body"),
            default_system_prompt=spec.system_prompt, max_retries=0,
        )
        agent = build_agent(job["agent"], client, config["agent"])
        evaluation = deepcopy(config.get("evaluation", {}))
        if family in {"xiangqi-history","xiangqi-rule-variants"} and evaluation.get("solution_tree_dir"):
            evaluation["solution_tree_dir"] = str(suite/evaluation["solution_tree_dir"])
        if family == "xiangqi-history":
            evaluation["history_mode"] = job["mode"]
        elif family == "xiangqi-multimodal":
            evaluation["input_modes"] = [job["mode"]]
            evaluation["step_dir"] = str(directory / "images")
        context = replay_h2_engine(directory / "engine") if family == "xiangqi-history" else nullcontext()
        # Native callback evidence survives a failure in a later condition.
        def record_partial(result):
            row = asdict(result) if is_dataclass(result) else dict(result)
            append_json(directory / "native_observations.jsonl", row)
        with context:
            results = _evaluate(spec, selected, agent, family, evaluation, on_result=record_partial)
        if not results:
            raise RuntimeError("Native evaluator returned no observations")
        rows = [asdict(r) if is_dataclass(r) else dict(r) for r in results]
        for row in rows:
            if row.get("status") == "error" or row.get("diagnostics"):
                atomic_write_json(directory / "failed_observation.json", row)
                raise RuntimeError("Infrastructure/diagnostic failure; retained but not scored")
            reasons = row.get("reasons") or []
            if any(isinstance(r, str) and r.lower().startswith((
                "llm_error", "pikafish_error", "api_error", "engine_error",
                "provider_error", "agent_request_error", "scorer_error",
            )) for r in reasons):
                atomic_write_json(directory / "failed_observation.json", row)
                raise RuntimeError("Infrastructure failure prefix; not scored")
        type_info = ({"module": type(results[0]).__module__, "name": type(results[0]).__name__}
                     if is_dataclass(results[0]) else None)
        result = {"job": job, "plan_sha256": plan["plan_sha256"], "records": rows,
                  "result_type": type_info, "elapsed_seconds": time.perf_counter() - started,
                  "finished_at": now()}
        result["sha256"] = fingerprint_payload(result)
        # Completion is committed in the worker BEFORE notifying the scheduler.
        atomic_write_json(directory / "result.json", result)
        atomic_write_json(directory / "state.json", {"status": "completed", "updated_at": now()})
        return job["key"]
    except BaseException as exc:
        atomic_write_json(directory / "state.json", {
            "status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "error",
            "updated_at": now(), "error_type": type(exc).__name__,
            "error": safe_error(exc, os.environ.get(provider["api_key_env"])),
        })
        raise RuntimeError(f"Job {job['key']} failed ({type(exc).__name__}); see jobs/{job['key']}/state.json") from None
    finally:
        signal.signal(signal.SIGTERM, previous_term)


def completed_jobs(suite, plan, jobs=None):
    result = {}
    for job in plan["jobs"] if jobs is None else jobs:
        path = suite / "jobs" / job["key"] / "result.json"
        if not path.exists():
            continue
        value = read_json_object(path)
        digest = value.pop("sha256", None)
        if (fingerprint_payload(value) != digest or value.get("job") != job
                or value.get("plan_sha256") != plan["plan_sha256"]):
            raise ValueError("Completed checkpoint identity/integrity mismatch")
        if not value.get("records") or any(row.get("status") == "error" for row in value["records"]):
            raise ValueError("Invalid completed checkpoint")
        value["sha256"] = digest
        result[job["key"]] = value
    return result


def hydrate(rows, info):
    if info is None:
        return rows
    import importlib
    if not info["module"].startswith("minibench.datasets."):
        raise ValueError("Unexpected checkpoint result class")
    cls = getattr(importlib.import_module(info["module"]), info["name"])
    if not is_dataclass(cls):
        raise ValueError("Unexpected non-dataclass result class")
    if info["name"] == "ZebraInstanceResult":
        from minibench.datasets.zebra.evaluation import zebra_result_from_dict
        return [zebra_result_from_dict(row) for row in rows]
    return [cls(**row) for row in rows]


def report(suite, plan, completed):
    # Regenerate aggregations from atomic worker checkpoints. predictions.jsonl
    # is not the resume authority, so a crash cannot corrupt the completed set.
    values, latency, totals = {}, {}, {}
    for job in plan["jobs"]:
        value = completed.get(job["key"])
        if value is None:
            continue
        pair = job["experiment"], job["agent"]
        values.setdefault(pair, []).append(value)
        latency.setdefault(job["experiment"], []).append(value["elapsed_seconds"])
    for (name, agent), batches in values.items():
        info = batches[0]["result_type"]
        if any(b["result_type"] != info for b in batches):
            raise ValueError("Inconsistent result types in experiment")
        records = [row for b in batches for row in b["records"]]
        spec = get_task_family_spec(plan["experiments"][name]["family"])
        directory = suite / "results" / name / agent
        atomic_write_jsonl(directory / "predictions.jsonl", records)
        native_summary = spec.summarize(hydrate(records, info))
        planned = sum(j["experiment"] == name and j["agent"] == agent for j in plan["jobs"])
        status = "completed" if len(batches) == planned else "partial"
        native_summary["cloud_coverage"] = {
            "completed_jobs": len(batches), "planned_jobs": planned, "run_status": status,
            "note": "Native metric means describe saved observations, not missing jobs.",
        }
        if "run_status" in native_summary:
            native_summary["run_status"] = status
        atomic_write_json(directory / "results.json", native_summary)
        c = deepcopy(plan["experiments"][name]["configs"][agent])
        c["task"] = {"family": plan["experiments"][name]["family"],
                     "path": plan["experiments"][name]["source"],
                     "task_ids": plan["experiments"][name]["task_ids"]}
        atomic_write_json(directory / "resolved_config.json", c)
    for name, seconds in latency.items():
        seconds.sort()
        totals[name] = {"completed_jobs": len(seconds), "mean_seconds": statistics.mean(seconds),
                        "median_seconds": statistics.median(seconds),
                        "p95_seconds": seconds[max(0, math.ceil(.95 * len(seconds)) - 1)]}
    # Sum each actually saved API response exactly once; replay does not rebill.
    usages = {"known_success_tokens": 0, "saved_successful_calls": 0,
              "successful_calls_without_usage": 0, "unfinished_attempts": 0,
              "known_failed_tokens": 0, "known_total_tokens": 0,
              "started_attempts": 0, "failed_attempts_with_usage": 0,
              "attempts_without_usage": 0}
    from minibench.datasets.xiangqi.resources import (
        physical_bucket, add_physical_attempt, finish_physical_bucket, assess_attempt_usage,
    )
    resources = {}
    for job in plan["jobs"]:
        calls = suite / "jobs" / job["key"] / "calls"
        for attempt in call_attempts(calls):
            experiment = plan["experiments"][job["experiment"]]
            if experiment["family"].startswith("xiangqi-"):
                bucket = resources.setdefault(job["experiment"],physical_bucket())
                pricing = experiment["configs"][job["agent"]].get("evaluation",{}).get("pricing")
                add_physical_attempt(bucket,attempt,pricing)
            success = attempt["status"] == "success"
            usages["started_attempts"] += 1
            usages["saved_successful_calls" if success else "unfinished_attempts"] += 1
            assessment = assess_attempt_usage(attempt)
            tokens = assessment["total_tokens"]
            if tokens is not None:
                usages["known_success_tokens" if success else "known_failed_tokens"] += int(tokens)
                usages["known_total_tokens"] += int(tokens)
                if not success:
                    usages["failed_attempts_with_usage"] += 1
            else:
                usages["attempts_without_usage"] += 1
                if success:
                    usages["successful_calls_without_usage"] += 1
    usages["usage_complete"] = (usages["started_attempts"] > 0
                                and usages["attempts_without_usage"] == 0)
    usages["total_tokens"] = usages["known_total_tokens"] if usages["usage_complete"] else None
    if usages["started_attempts"] == usages["attempts_without_usage"]:
        for key in ("known_success_tokens", "known_failed_tokens", "known_total_tokens"):
            usages[key] = None
    summary = {"completed_jobs": len(completed), "planned_jobs": len(plan["jobs"]),
               "by_experiment_latency": totals, "usage": usages,
               "by_experiment_resources": {name:finish_physical_bucket(value)
                                           for name,value in resources.items()},
               "note": "Unknown/incomplete request usage is not zero; this is not an account balance."}
    atomic_write_json(suite / "summary.json", summary)
    return summary


def restore_accounting(suite, plan, gate):
    from minibench.datasets.xiangqi.resources import assess_attempt_usage
    for job in plan["jobs"]:
        exp = plan["experiments"][job["experiment"]]
        group = plan["accounts"][exp["account"]]["rate_group"]
        calls = suite / "jobs" / job["key"] / "calls"
        for attempt in call_attempts(calls):
            if attempt.get("started_at"):
                gate.restore_start(group, attempt["started_at"])
            usage = attempt.get("usage") if assess_attempt_usage(attempt)["total_tokens"] is not None else None
            gate.restore(group, usage, attempt.get("at"))


def verify_shortest_trees(suite, plan):
    """Preflight every tree before ANY job can issue a paid request.

    The legacy depth-3 diagnostic bypass never bypasses shortest-tree checks.
    H2 needs the same preflight as C2; checking only inside its worker is late
    when another concurrently scheduled task has already used the account.
    """
    from minibench.datasets.xiangqi.schema import load_records
    from minibench.datasets.xiangqi.solution_tree import TreeSession
    for exp in plan['experiments'].values():
        family=exp['family']
        if family not in {'xiangqi-history','xiangqi-rule-variants'}:
            continue
        trees=next(iter(exp['configs'].values())).get('evaluation',{}).get('solution_tree_dir')
        wanted=set(exp['task_ids'])
        records=[r for r in load_records(suite/exp['source'],expected_family=family)
                 if r['id'] in wanted]
        if {r['id'] for r in records} != wanted:
            raise ValueError('Frozen shortest-tree task selection is incomplete')
        if not trees and any(r.get('shortest_proof') for r in records):
            raise ValueError('Certified shortest-mate tasks require solution_tree_dir')
        if trees:
            from minibench.datasets.xiangqi.tree_certificate import require_saved_tree_certificate
            certified = [record for record in records if record.get("shortest_proof")]
            if certified:
                require_saved_tree_certificate(certified, data_path=suite/exp['source'],
                                               tree_dir=suite/trees)
            for record in records:
                TreeSession(record,suite/trees)


def run(suite, plan, *, workers=None, allow_uncertified_c2=False):
    completed = completed_jobs(suite, plan)
    if len(completed) == len(plan["jobs"]):
        report(suite, plan, completed)
        return 0
    verify_shortest_trees(suite,plan)
    if not allow_uncertified_c2:
        from minibench.datasets.xiangqi.variants.certification import certificate_header_valid, verify_winning_line
        from minibench.datasets.xiangqi.schema import load_records
        for exp in plan["experiments"].values():
            if exp["family"] != "xiangqi-rule-variants":
                continue
            trees = next(iter(exp["configs"].values())).get("evaluation",{}).get("solution_tree_dir")
            if trees:
                # Already checked unconditionally, including H2, above.
                continue
            source = suite / exp["source"]
            path = source.parent / "runtime_certificates.json"
            if not path.is_file():
                raise ValueError("C2 runtime-policy certificate is missing; certification is required before a full experiment.")
            records = load_records(source, expected_family="xiangqi-rule-variants")
            bundle = read_json_object(path)
            if not certificate_header_valid(bundle, records, sha256_file(source)):
                raise ValueError("C2 runtime-policy certificate is incomplete or stale")
            for record in records:
                if record["id"] in exp["task_ids"]:
                    proof = verify_winning_line(record, bundle["tasks"][record["id"]]["moves_uci"])
                    if not proof["valid"]:
                        raise ValueError(f'C2 runtime-policy witness failed: {record["id"]}')
    # Validate engine identity before any paid request (not after another job).
    for e in plan["experiments"].values():
        family = e["family"]
        if family.startswith("xiangqi-") and family != "xiangqi-rule-variants":
            from minibench.datasets.xiangqi.engines.pikafish import resolve_pikafish_executable, pikafish_fingerprint
            from minibench.datasets.xiangqi.runtime import validate_engine_fingerprint
            cfg = next(iter(e["configs"].values()))["evaluation"]
            binary = resolve_pikafish_executable(cfg.get("pikafish_path"))
            validate_engine_fingerprint(binary, binary_sha256=cfg.get("pikafish_binary_sha256"),
                                        nnue_sha256=cfg.get("pikafish_nnue_sha256"))
            actual = pikafish_fingerprint(binary)
            identity_path = suite / "engine_identity.json"
            # Paths differ on servers; binary/network content must match on resume.
            identity = {k: actual.get(k) for k in ("binary_sha256", "eval_file_sha256")}
            if identity_path.exists() and read_json_object(identity_path) != identity:
                raise ValueError("Pikafish/NNUE changed since first run")
            atomic_write_json(identity_path, identity)
    for a in plan["accounts"].values():
        variable = a["provider"]["api_key_env"]
        if not os.environ.get(variable):
            raise ValueError(f"Set transient environment variable {variable}")
    count = plan["workers"] if workers is None else workers
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 64:
        raise ValueError("workers must be in 1..64")
    queue = [j for j in plan["jobs"] if j["key"] not in completed]
    dependencies=c2_phase_dependencies(plan)
    started = time.perf_counter()
    state = {"status": "running", "pid": os.getpid(), "workers": count,
             "completed_jobs": len(completed), "planned_jobs": len(plan["jobs"]),
             "started_at": now(), "run_segments_seconds": 0}
    def save(**updates):
        state.update(updates, updated_at=now(), completed_jobs=len(completed),
                     elapsed_this_invocation_seconds=time.perf_counter() - started)
        atomic_write_json(suite / "state.json", state)
    context = mp.get_context("spawn")
    manager = GateManager(ctx=context)
    manager.start(initializer=_manager_initializer)
    gate = None
    pool = None
    futures = {}
    failure = None
    last_report = time.monotonic() - 30
    previous_term = signal.signal(signal.SIGTERM, _interrupt_worker)
    try:
        gate = manager.RateGate(plan["limits"])
        restore_accounting(suite, plan, gate)
        pool = ProcessPoolExecutor(max_workers=count, mp_context=context,
                                   initializer=_worker_initializer)
        save()
        while queue or futures:
            while queue and len(futures) < count:
                index=next((i for i,j in enumerate(queue)
                            if dependencies.get(j["key"],set()) <= completed.keys()),None)
                if index is None:
                    break
                job = queue.pop(index)
                futures[pool.submit(execute_job, str(suite), plan, job, gate)] = job
            if not futures:
                raise RuntimeError("No runnable job: unresolved C2 phase dependencies")
            done, _ = wait(futures, timeout=1, return_when=FIRST_COMPLETED)
            for future in done:
                job = futures.pop(future)
                future.result()
                completed.update(completed_jobs(suite, plan, [job]))
                print(f"DONE {len(completed)}/{len(plan['jobs'])} {job['experiment']}/{job['agent']}/{job['task_id']}/{job['mode']}", flush=True)
            # Per-job evidence is already durable. Batch derived tables to avoid
            # rereading/rebuilding an entire large suite after every completion.
            if done and time.monotonic() - last_report >= 30:
                report(suite, plan, completed)
                last_report = time.monotonic()
            save(active_jobs=list(futures.values()), accounts=gate.snapshot())
        save(status="completed", active_jobs=[])
    except BaseException as exc:
        failure = exc
        if gate is not None:
            gate.cancel()
        for future in futures:
            future.cancel()
        # Abort outstanding HTTP reads promptly, preserving each worker's finally
        # partial snapshot. Linux API cloud is the supported execution platform.
        processes = list(pool._processes.values()) if pool is not None else []
        for process in processes:
            if process.is_alive():
                os.kill(process.pid, signal.SIGTERM)
        completed = completed_jobs(suite, plan)
        save(status="paused" if isinstance(exc, KeyboardInterrupt) else "error",
             error=str(exc), error_type=type(exc).__name__,
             accounts=gate.snapshot() if gate is not None else {})
    finally:
        if pool is not None:
            forced = _shutdown_pool(pool, failed=failure is not None)
            save(forced_worker_pids=forced)
        if gate is not None:
            # Forced workers may not execute finally. Rebuild from durable
            # physical-attempt evidence, retaining unknown usage reservations.
            final_gate = RateGate(plan["limits"])
            restore_accounting(suite, plan, final_gate)
            save(accounts=final_gate.snapshot(), active_jobs=[])
        manager.shutdown()
        signal.signal(signal.SIGTERM, previous_term)
        completed = completed_jobs(suite, plan)
        summary = report(suite, plan, completed)
        summary["wall_seconds_this_invocation"] = time.perf_counter() - started
        atomic_write_json(suite / "summary.json", summary)
    if failure:
        if isinstance(failure, KeyboardInterrupt):
            return 130
        raise failure
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "run", "status", "plot"])
    parser.add_argument("--config", type=Path, default=Path("config/cloud/xiangqi-pilot.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--pilot-count", type=int)
    parser.add_argument("--allow-uncertified-c2", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    os.chdir(ROOT)
    suite = args.output.resolve()
    if not suite.is_relative_to((ROOT / "runs").resolve()) or suite == (ROOT / "runs").resolve():
        parser.error("--output must be a child of repository runs/")
    if args.command == "prepare":
        plan = prepare(args.config, suite, pilot_count=args.pilot_count)
        print(json.dumps({"status": "prepared", "planned_jobs": plan["planned_jobs"],
                          "agents": plan["agents"], "output": str(suite)}, ensure_ascii=False))
        return 0
    with RunLock(suite):
        plan = verify_plan(suite)
        if args.command in {"status", "plot"}:
            print(json.dumps(report(suite, plan, completed_jobs(suite, plan)), ensure_ascii=False, indent=2))
            if args.command == "plot":
                from minibench.visualization.xiangqi_suite import plot_suite
                print(plot_suite(suite))
            return 0
        return run(suite, plan, workers=args.workers,
                   allow_uncertified_c2=args.allow_uncertified_c2)


def verify_plan(suite):
    plan = read_json_object(suite / "plan.json")
    digest = plan.pop("plan_sha256", None)
    if plan.get("version") != VERSION or fingerprint_payload(plan) != digest:
        raise ValueError("Frozen plan integrity mismatch")
    plan["plan_sha256"] = digest
    if code_hash() != plan["code_sha256"]:
        raise ValueError("Source code changed; create a new suite rather than silently resuming")
    for info in plan["files"]:
        path = (suite / info["path"]).resolve()
        if not path.is_relative_to(suite) or sha256_file(path) != info["sha256"]:
            raise ValueError("Frozen input changed or escaped suite")
        # Native loaders refer to dataset images by repository paths.
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg", ".webp"}:
            original = ROOT / Path(info["path"]).relative_to("inputs")
            if sha256_file(original) != info["sha256"]:
                raise ValueError("Original dataset image changed")
    return plan


if __name__ == "__main__":
    raise SystemExit(main())
