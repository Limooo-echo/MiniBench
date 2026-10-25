"""Independently re-prove the current H2/C2 strategy DAGs, with durable progress.

No model or engine is called. Existing trees are candidate move-order hints,
never proof. Missing, invalid and budget-unknown records cannot pass the gate.
This command verifies the declared locally-shortest/all-optimal (or all-legal)
scope; it does not silently change the experiment's definition of a solution.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from minibench.core.checkpoint import RunLock, atomic_write_json, sha256_file
from minibench.datasets.xiangqi.tree_certificate import (
    FAMILY_FOLDERS, TREE_FOLDERS, build_verification_report, record_sha256,
    verifier_fingerprint, verifier_hashes,
)

ALIASES = {"h2": "xiangqi-history", "c2": "xiangqi-rule-variants"}


def read_records(data_root: Path) -> list[dict]:
    rows = []
    for family, folder in FAMILY_FOLDERS.items():
        path = data_root / folder / "tasks.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if (not isinstance(record, dict) or record.get("family") != family
                    or not isinstance(record.get("id"), str)
                    or re.fullmatch(r"[A-Za-z0-9_-]+", record["id"]) is None):
                raise ValueError(f"Invalid family or unsafe task ID in {path}")
            rows.append(record)
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate task IDs across H2/C2")
    return rows


def verify_one(payload):
    from minibench.datasets.xiangqi.tree_verification import verify_tree

    record, path_text, seconds, budget, code_sha, memory_mb, ordering_mode = payload
    path = Path(path_text)
    try:
        if memory_mb:
            try:
                import resource
            except ImportError as exc:
                raise RuntimeError("memory_limit_unavailable; explicitly use --memory-mb 0") from exc
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            cap = memory_mb * 1024 * 1024
            if hard != resource.RLIM_INFINITY:
                cap = min(cap, hard)
            resource.setrlimit(resource.RLIMIT_AS, (cap, hard))
        raw = path.read_bytes()
        tree_sha = hashlib.sha256(raw).hexdigest()
        tree = json.loads(raw)
        result = verify_tree(record, tree, seconds=seconds, node_budget=budget,
                             ordering_mode=ordering_mode)
    except (OSError, ValueError, TypeError) as exc:
        tree_sha = sha256_file(path) if path.is_file() else None
        result = {"status": "invalid", "valid": False,
                  "reasons": [f"tree_input_invalid:{type(exc).__name__}:{exc}"]}
    except MemoryError:
        tree_sha = sha256_file(path) if path.is_file() else None
        result = {"status": "unknown", "valid": False,
                  "reasons": ["verification_memory_budget_exhausted"]}
    except Exception as exc:
        tree_sha = sha256_file(path) if path.is_file() else None
        result = {"status": "unknown", "valid": False,
                  "reasons": [f"verification_error:{type(exc).__name__}:{exc}"]}
    result.update(record_sha256=record_sha256(record), tree_sha256=tree_sha,
                  verification_source_sha256=code_sha,
                  verification_parameters={"seconds": seconds, "node_budget": budget,
                                           "memory_mb": memory_mb, "hints": True,
                                           "ordering_mode": ordering_mode})
    return record["id"], result


def run_verification(data_root: Path, output: Path, *, families, task_ids=(),
                     seconds=60.0, node_budget=2_000_000, workers=2,
                     resume=False, memory_mb=0, ordering_mode="history") -> dict:
    rows = read_records(data_root)
    wanted = set(task_ids)
    if wanted - {row["id"] for row in rows}:
        raise ValueError("Requested task ID does not exist")
    selected = [row for row in rows if row["family"] in families
                and (not wanted or row["id"] in wanted)]
    if not selected:
        raise ValueError("No tasks selected")
    code_hashes = verifier_hashes()
    code_sha = verifier_fingerprint(code_hashes)
    records = {}
    if output.exists():
        if not resume:
            raise FileExistsError("Report exists; use --resume to retain verified work")
        prior = json.loads(output.read_text(encoding="utf-8"))
        if (prior.get("verifier_hashes") == code_hashes
                and prior.get("verification_source_sha256") == code_sha):
            for row in rows:
                old = prior.get("records", {}).get(row["id"], {})
                raw = old.get("verification", {})
                tree = data_root / "solution_trees" / TREE_FOLDERS[row["family"]] / f"{row['id']}.json"
                if (old.get("status") == "verified" and raw.get("valid") is True
                        and raw.get("verification_source_sha256") == code_sha
                        and raw.get("record_sha256") == record_sha256(row)
                        and tree.is_file() and raw.get("tree_sha256") == sha256_file(tree)):
                    records[row["id"]] = raw

    def save():
        report = build_verification_report(data_root, records)
        report["run"] = {"selected_count": len(selected), "workers": workers,
                         "seconds_per_record": seconds, "node_budget": node_budget,
                         "memory_mb_per_worker": memory_mb,
                         "ordering_mode": ordering_mode,
                         "resume": resume, "calls_external_models": False}
        atomic_write_json(output, report)
        return report

    save()  # Unknown until proved; durable before starting worker processes.
    jobs = [(row, str(data_root / "solution_trees" / TREE_FOLDERS[row["family"]]
                      / f"{row['id']}.json"), seconds, node_budget, code_sha, memory_mb, ordering_mode)
            for row in selected if row["id"] not in records]
    print(json.dumps({"selected": len(selected), "reused": len(selected)-len(jobs),
                      "pending": len(jobs), "report": str(output)}, ensure_ascii=False), flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(verify_one, job): job[0]["id"] for job in jobs}
        for future in as_completed(futures):
            identity = futures[future]
            try:
                identity, result = future.result()
            except Exception as exc:
                row = next(row for row in rows if row["id"] == identity)
                path = data_root / "solution_trees" / TREE_FOLDERS[row["family"]] / f"{identity}.json"
                result = {"status": "unknown", "valid": False,
                          "reasons": [f"worker_failed:{type(exc).__name__}:{exc}"],
                          "record_sha256": record_sha256(row),
                          "tree_sha256": sha256_file(path) if path.is_file() else None,
                          "verification_source_sha256": code_sha}
            records[identity] = result
            report = save()
            print(json.dumps({"task_id": identity, "status": report["records"][identity]["status"],
                              "summary": report["summary"],
                              "reasons": result.get("reasons", [])}, ensure_ascii=False), flush=True)
    if verifier_hashes() != code_hashes:
        # build_verification_report has already marked old-code results unknown.
        print("Verifier code changed during search; results require re-verification.", file=sys.stderr)
    return save()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/xiangqi")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "data/xiangqi/solution_tree_verification.json")
    parser.add_argument("--families", default="h2,c2")
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--node-budget", type=int, default=2_000_000)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--memory-mb", type=int, default=3072,
                        help="Linux address-space cap per worker; 0 explicitly disables it")
    parser.add_argument("--ordering-mode", choices=("history", "checks-history"), default="history",
                        help="Full-width candidate ordering only; never prunes a legal move")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    names = args.families.split(",")
    if not names or len(set(names)) != len(names) or set(names) - set(ALIASES):
        parser.error("--families must be a distinct subset of h2,c2")
    if (not math.isfinite(args.seconds) or args.seconds <= 0
            or args.node_budget < 1 or args.workers < 1 or args.memory_mb < 0):
        parser.error("Time, node and worker budgets must be positive")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    lock_root = Path(tempfile.gettempdir()) / "minibench-tree-audit-locks"
    lock_name = hashlib.sha256(str(output).encode()).hexdigest() + ".lock"
    with RunLock(lock_root, lock_name=lock_name):
        report = run_verification(args.data_root.resolve(), output,
                                  families={ALIASES[name] for name in names}, task_ids=args.task_id,
                                  seconds=args.seconds, node_budget=args.node_budget,
                                  workers=args.workers, resume=args.resume, memory_mb=args.memory_mb,
                                  ordering_mode=args.ordering_mode)
    print(json.dumps(report["summary"], ensure_ascii=False))
    if report["summary"]["invalid"]:
        return 1
    return 0 if report["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
