"""Build/resume shortest-mate DAGs offline; unknown is not a certificate."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from minibench.core.checkpoint import atomic_write_json, sha256_file
from minibench.datasets.xiangqi.schema import load_records
from minibench.datasets.xiangqi.solution_tree import build_tree, record_key, VERSION


def check(payload):
    record, cap, seconds, budget = payload
    return record["id"], build_tree(record, max_attacker_moves=cap,
                                    seconds=seconds, node_budget=budget)


def preserve_stronger_proof(prior, result):
    """A failed larger export must not erase an already proved minimum."""
    if prior.get("record_key") != result.get("record_key") or prior.get("version") != VERSION:
        return result
    if prior.get("shortest_certified"):
        if result.get("shortest_certified") and (
                prior["shortest_attacker_moves"] != result["shortest_attacker_moves"]):
            raise ValueError("Conflicting certified shortest distances; stop and investigate")
        if result.get("complete") and result.get("forced_mate") is False:
            raise ValueError("Conflicting positive/negative proof; stop and investigate")
        if not result.get("shortest_certified") or prior.get("complete") and not result.get("complete"):
            return {**prior,"latest_export_attempt":{
                "seconds":result["seconds"],"visits":result["visits"],
                "stop_reason":result.get("stop_reason")}}
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--max-moves", type=int, default=5)
    p.add_argument("--seconds", type=float, default=30)
    p.add_argument("--node-budget", type=int, default=500_000)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--bands", default="short,medium")
    p.add_argument("--retry-unknown", action="store_true")
    a = p.parse_args(argv)
    if a.workers < 1 or a.seconds <= 0 or a.node_budget < 1:
        p.error("worker/time/node budgets must be positive")
    records = [r for r in load_records(a.data) if r["difficulty"] in a.bands.split(",")]
    a.out_dir.mkdir(parents=True, exist_ok=True)
    pending, entries = [], {}
    for record in records:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", record["id"]):
            raise ValueError("unsafe task ID")
        path = a.out_dir/f"{record['id']}.json"
        if path.exists():
            prior = json.loads(path.read_text())
            if prior.get("record_key") == record_key(record) and prior.get("version") == VERSION:
                reusable_complete = prior.get("complete") and (
                    prior.get("forced_mate") or prior.get("requested_max_attacker_moves",0)>=a.max_moves)
                if reusable_complete or (prior.get("shortest_certified") and not a.retry_unknown):
                    entries[record["id"]] = prior
                    continue
        pending.append((record, a.max_moves, a.seconds, a.node_budget))
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures = {pool.submit(check, item): item[0]["id"] for item in pending}
        for future in as_completed(futures):
            identity, result = future.result()
            path=a.out_dir/f"{identity}.json"
            if path.exists():
                result=preserve_stronger_proof(json.loads(path.read_text()),result)
            atomic_write_json(a.out_dir/f"{identity}.json", result)
            entries[identity] = result
            print(f"{len(entries)}/{len(records)} shortest={result['shortest_attacker_moves']} "
                  f"complete={result['complete']} {identity}", flush=True)
    report = {"version": VERSION, "dataset_sha256": sha256_file(a.data),
              "selected_rows": len(records),
              "complete_trees": sum(x["complete"] and bool(x["forced_mate"]) for x in entries.values()),
              "certified_shortest": sum(bool(x["shortest_certified"]) for x in entries.values()),
              "files": {k: {"sha256": sha256_file(a.out_dir/f"{k}.json"),
                            "shortest_attacker_moves": v["shortest_attacker_moves"],
                            "complete": v["complete"]}
                        for k, v in sorted(entries.items())}}
    atomic_write_json(a.out_dir/"manifest.json", report)
    print(json.dumps({k:v for k,v in report.items() if k!="files"}))
    return 0 if report["certified_shortest"] == len(records) else 2


if __name__ == "__main__":
    raise SystemExit(main())
