#!/usr/bin/env python3
"""Remove generated M2 PNGs only after a complete four-family cloud suite.

Dry-run by default. Results, frozen inputs, plots, and request journals are
retained. This post-processing tool deliberately does not require the current
evaluation code hash to equal an old finished run's code hash.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from minibench.core.checkpoint import (
    RunLock, RunLockedError, atomic_write_json, fingerprint_payload,
    read_json_object, sha256_file,
)

FAMILIES = {
    "xiangqi-mate-in-one", "xiangqi-history",
    "xiangqi-rule-variants", "xiangqi-multimodal",
}
IMAGE_MODES = {"chinese-piece-image", "latin-piece-image"}
RECEIPT_NAME = "m2-image-cleanup.json"
RECEIPT_VERSION = "minibench-m2-image-cleanup-v1"


def checked_path(suite: Path, relative: str) -> Path:
    """Reject traversal and links, including links in intermediate folders."""
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"Unsafe run-relative path: {relative}")
    path = suite
    for part in relative_path.parts:
        path = path / part
        if path.is_symlink():
            raise ValueError(f"Refusing symlink: {relative}")
    if not path.resolve().is_relative_to(suite):
        raise ValueError(f"Path escapes the selected run: {relative}")
    return path


def read_object(suite: Path, relative: str) -> dict:
    return read_json_object(checked_path(suite, relative))


def require_digest(value: dict, field: str, label: str) -> str:
    value = dict(value)
    digest = value.pop(field, None)
    if not isinstance(digest, str) or fingerprint_payload(value) != digest:
        raise ValueError(f"Integrity mismatch: {label}")
    return digest


def completed_records(suite: Path) -> tuple[dict, dict]:
    plan = read_object(suite, "plan.json")
    require_digest(plan, "plan_sha256", "plan.json")
    if plan.get("version") != "minibench-cloud-v1":
        raise ValueError("Only minibench-cloud-v1 suites are supported")
    jobs, experiments, agents = plan.get("jobs"), plan.get("experiments"), plan.get("agents")
    if not isinstance(jobs, list) or not jobs or plan.get("planned_jobs") != len(jobs):
        raise ValueError("Invalid planned job count")
    if not isinstance(experiments, dict) or not isinstance(agents, list) or not agents:
        raise ValueError("Missing experiments or agents")
    if len(set(agents)) != len(agents):
        raise ValueError("Duplicate agents")
    if not FAMILIES.issubset({e.get("family") for e in experiments.values()}):
        raise ValueError("D3, H2, C2 and M2 must all be included in this suite")
    expected = set()
    for name, experiment in experiments.items():
        ids = experiment.get("task_ids")
        counts = experiment.get("condition_counts")
        if not isinstance(ids, list) or not ids or len(set(ids)) != len(ids):
            raise ValueError(f"Invalid task IDs: {name}")
        if experiment.get("record_count") != len(ids) or set(experiment.get("configs", {})) != set(agents):
            raise ValueError(f"Invalid task/agent coverage: {name}")
        if not isinstance(counts, dict) or not counts:
            raise ValueError(f"Missing condition counts: {name}")
        modes = ["native"] if experiment["family"] == "xiangqi-rule-variants" else list(counts)
        if experiment["family"] == "xiangqi-rule-variants":
            if counts.get("all") != len(ids):
                raise ValueError("Invalid C2 condition count")
        elif any(count != len(ids) for count in counts.values()):
            raise ValueError(f"Invalid condition count: {name}")
        expected.update((name, agent, task_id, mode) for agent in agents for task_id in ids for mode in modes)
    identities = set()
    for job in jobs:
        identity = tuple(job.get(k) for k in ("experiment", "agent", "task_id", "mode"))
        if identity in identities or job.get("key") != fingerprint_payload(list(identity)):
            raise ValueError("Invalid or duplicate job identity")
        identities.add(identity)
    if identities != expected:
        raise ValueError("Jobs do not cover every planned task, agent and condition")
    state = read_object(suite, "state.json")
    if (state.get("status") != "completed" or state.get("active_jobs")
            or state.get("completed_jobs") != len(jobs) or state.get("planned_jobs") != len(jobs)):
        raise ValueError("The entire suite is not completed; no images will be deleted")
    if any(account.get("active_calls", 0) != 0 for account in state.get("accounts", {}).values()):
        raise ValueError("There are still active API calls")
    for info in plan.get("files", []):
        if sha256_file(checked_path(suite, info["path"])) != info["sha256"]:
            raise ValueError("Frozen input integrity mismatch")
    records = {}
    for job in jobs:
        prefix = f"jobs/{job['key']}"
        if read_object(suite, prefix + "/state.json").get("status") != "completed":
            raise ValueError(f"Job is not completed: {job['task_id']}")
        result = read_object(suite, prefix + "/result.json")
        require_digest(result, "sha256", prefix + "/result.json")
        rows = result.get("records")
        if result.get("job") != job or result.get("plan_sha256") != plan["plan_sha256"]:
            raise ValueError("Completed result belongs to a different plan/job")
        if not isinstance(rows, list) or not rows:
            raise ValueError("Missing completed observations")
        if any(not isinstance(row, dict) or row.get("status") in {"error", "missing", "interrupted", "partial"}
               or row.get("diagnostics") for row in rows):
            raise ValueError("An observation still has an evaluation failure")
        if any(isinstance(reason, str) and reason.lower().startswith((
                "llm_error", "pikafish_error", "api_error", "engine_error", "provider_error",
                "agent_request_error", "scorer_error"))
                for row in rows for reason in (row.get("reasons") or [])):
            raise ValueError("An observation still has an infrastructure failure")
        records[job["key"]] = rows
    return plan, records


def cleanup(suite_dir: str | Path, *, apply: bool = False) -> dict:
    suite = Path(suite_dir).resolve(strict=True)
    if not suite.is_dir() or suite in {Path(suite.anchor), ROOT, ROOT / "runs"}:
        raise ValueError("Select one specific cloud run directory")
    checked_path(suite, ".run.lock")
    with RunLock(suite):
        plan, records = completed_records(suite)
        receipt_path = checked_path(suite, RECEIPT_NAME)
        previous = read_json_object(receipt_path) if receipt_path.exists() else None
        old = {}
        if previous is not None:
            if (previous.get("version") != RECEIPT_VERSION
                    or previous.get("plan_sha256") != plan["plan_sha256"]
                    or previous.get("status") not in {"in_progress", "partial", "completed"}):
                raise ValueError("Cleanup receipt belongs to another or invalid run")
            old = {item["path"]: item for item in previous["files"]}
        targets, image_roots = {}, []
        for job in plan["jobs"]:
            if plan["experiments"][job["experiment"]]["family"] != "xiangqi-multimodal":
                continue
            image_roots.append(f"jobs/{job['key']}/images")
            mode = job["mode"]
            if mode == "text":
                continue
            if mode not in IMAGE_MODES or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", job["task_id"]):
                raise ValueError("Unexpected M2 mode or unsafe task ID")
            rows = records[job["key"]]
            if len(rows) != 1 or rows[0].get("task_id") != job["task_id"] or rows[0].get("input_mode") != mode:
                raise ValueError("M2 observation does not match its task/mode")
            steps = rows[0].get("steps", [])
            if len(steps) != 1 or not re.fullmatch(r"[a-f0-9]{64}", steps[0].get("input_image_sha256") or ""):
                raise ValueError("Missing M2 input-image fingerprint")
            relative = f"jobs/{job['key']}/images/{job['task_id']}/{mode}_step00.png"
            targets[relative] = steps[0]["input_image_sha256"]
        if set(old) - set(targets):
            raise ValueError("Unexpected paths in the previous cleanup receipt")
        for relative in image_roots:
            root = checked_path(suite, relative)
            if root.exists() and not root.is_dir():
                raise ValueError("M2 images path is not a directory")
            for parent, dirs, files in os.walk(root, followlinks=False):
                for name in dirs + files:
                    path = checked_path(suite, (Path(parent) / name).relative_to(suite).as_posix())
                    if path.suffix.lower() == ".png" and path.relative_to(suite).as_posix() not in targets:
                        raise ValueError("Unexpected PNG in an M2 job; refusing to guess what to delete")
        entries, present = [], []
        for relative, digest in sorted(targets.items()):
            path = checked_path(suite, relative)
            saved = old.get(relative)
            if saved and saved.get("sha256") != digest:
                raise ValueError("Cleanup receipt image fingerprint mismatch")
            if path.exists():
                if not path.is_file() or sha256_file(path) != digest:
                    raise ValueError(f"Generated image changed: {relative}")
                entry = {"path": relative, "sha256": digest, "bytes": path.stat().st_size, "deleted": False}
                present.append(entry)
            elif saved and isinstance(saved.get("bytes"), int) and saved["bytes"] >= 0:
                entry = {**saved, "deleted": True}
            else:
                raise ValueError(f"Image is missing without a matching cleanup receipt: {relative}")
            entries.append(entry)
        result = {"status": "dry-run", "validated_jobs": len(plan["jobs"]),
                  "candidate_files": len(present), "candidate_bytes": sum(x["bytes"] for x in present),
                  "previously_removed_files": len(entries) - len(present), "deleted_this_run": 0}
        if not apply:
            return result
        receipt = {"version": RECEIPT_VERSION, "plan_sha256": plan["plan_sha256"],
                   "status": "in_progress", "files": entries,
                   "updated_at": datetime.now(timezone.utc).isoformat(),
                   "retained": ["frozen inputs", "results", "plots", "request/response journals with encoded images"]}
        atomic_write_json(receipt_path, receipt)
        try:
            for entry in present:
                path = checked_path(suite, entry["path"])
                if not path.is_file() or sha256_file(path) != entry["sha256"]:
                    raise ValueError("Image changed during cleanup; remaining images retained")
                path.unlink()
                entry["deleted"] = True
                result["deleted_this_run"] += 1
            receipt["status"] = "completed"
        except BaseException:
            receipt["status"] = "partial"
            raise
        finally:
            receipt["updated_at"] = datetime.now(timezone.utc).isoformat()
            receipt["deleted_files"] = sum(bool(x["deleted"]) for x in entries)
            receipt["deleted_bytes"] = sum(x["bytes"] for x in entries if x["deleted"])
            atomic_write_json(receipt_path, receipt)
        return {**result, "status": "completed", "receipt": str(receipt_path)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-dir", type=Path, required=True, help="One completed cloud suite directory")
    parser.add_argument("--apply", action="store_true", help="Delete verified generated PNGs; default is preview only")
    args = parser.parse_args(argv)
    try:
        result = cleanup(args.suite_dir, apply=args.apply)
    except (OSError, ValueError, KeyError, TypeError, RunLockedError) as exc:
        print(f"拒绝清理：{exc}", file=sys.stderr)
        return 2
    print(f"已核验本轮 {result['validated_jobs']} 个条件全部完成（含 D3/H2/C2/M2）。")
    if args.apply:
        print(f"本次删除 M2 棋盘 PNG {result['deleted_this_run']} 张；此前已清理 {result['previously_removed_files']} 张。")
        print(f"清理记录：{result['receipt']}")
    else:
        print(f"预览：可清理 {result['candidate_files']} 张 PNG，约 {result['candidate_bytes'] / 1024**2:.2f} MiB；未删除文件。")
        print("实际清理请添加 --apply。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
