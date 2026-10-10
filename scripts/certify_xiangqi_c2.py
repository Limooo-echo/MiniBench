"""Offline, resumable certification against C2's registered depth-3 opponent."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from minibench.core.checkpoint import atomic_write_json, sha256_file
from minibench.datasets.xiangqi.schema import load_records
from minibench.datasets.xiangqi.variants.certification import (
    CERTIFICATE_VERSION, POLICY, policy_fingerprint, verify_winning_line, find_winning_line,
)


def check(record, engine_path=None, candidate_line=None):
    reference_line = record["c2_analysis"]["principal_variation_uci"]
    line = candidate_line if candidate_line is not None else reference_line
    result = verify_winning_line(record, line)
    if not result["valid"] and line != reference_line:
        line = reference_line
        result = verify_winning_line(record, line)
    if not result["valid"] and engine_path:
        from minibench.datasets.xiangqi.engines.pikafish import PikafishEngine
        try:
            with PikafishEngine(engine_path, timeout=30) as engine:
                found = find_winning_line(record, engine)
            if found:
                line = found
                result = verify_winning_line(record, line)
        except TimeoutError as exc:
            result["search_status"] = str(exc)
    return record["id"], {**result, "moves_uci": line}


def candidate_lines(paths, expected):
    """A cached witness is only a candidate; workers recheck every ply."""
    candidates = {}
    for path in dict.fromkeys(paths):
        if not path.exists():
            continue
        prior = json.loads(path.read_text(encoding="utf-8"))
        if not all(prior.get(k) == expected[k] for k in (
            "version", "policy", "policy_fingerprint", "dataset_sha256",
        )):
            continue
        for key, value in prior.get("tasks", {}).items():
            if isinstance(value, dict) and value.get("valid") is True and isinstance(value.get("moves_uci"), list):
                candidates[key] = value["moves_uci"]
    return candidates


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=ROOT / "data/xiangqi/rule_variants/tasks.jsonl")
    p.add_argument("--output", type=Path, default=ROOT / "data/xiangqi/rule_variants/runtime_certificates.json")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed-certificates", type=Path,
                   default=ROOT / "data/xiangqi/rule_variants/runtime_certificates.json",
                   help="Matching published witness candidates; every candidate is revalidated")
    p.add_argument("--pikafish-path", type=Path, help="Offline ordering guidance only, never the C2 judge")
    a = p.parse_args()
    records = load_records(a.data, expected_family="xiangqi-rule-variants")
    bundle = {"version": CERTIFICATE_VERSION, "policy": POLICY,
              "policy_fingerprint": policy_fingerprint(), "dataset_sha256": sha256_file(a.data),
              "complete": False, "tasks": {},
              "claim": "Existence of a bounded winning continuation against this deterministic opponent; not a shortest/all-defences proof."}
    candidates = candidate_lines((a.seed_certificates, a.output), bundle)
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        pending = {pool.submit(check, r, a.pikafish_path, candidates.get(r["id"])): r["id"] for r in records}
        for f in as_completed(pending):
            key, result = f.result()
            bundle["tasks"][key] = result
            atomic_write_json(a.output, bundle)
            print(f'{len(bundle["tasks"])}/{len(records)} valid={result["valid"]} {key}', flush=True)
    bundle["complete"] = len(bundle["tasks"]) == len(records) and all(v["valid"] for v in bundle["tasks"].values())
    atomic_write_json(a.output, bundle)
    print(json.dumps({"complete": bundle["complete"], "valid": sum(v["valid"] for v in bundle["tasks"].values()),
                      "total": len(records), "output": str(a.output)}))
    return 0 if bundle["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
