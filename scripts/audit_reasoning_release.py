"""Audit the 100-task Zebra and one-stroke release without model calls."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import networkx as nx
from PIL import Image
import yaml

from minibench.datasets.one_stroke.dataset import (
    load_one_stroke_tasks,
    one_stroke_edge_ids,
    simulate_one_stroke_history,
)
from minibench.datasets.one_stroke.evaluation import plan_one_stroke_work_items
from minibench.datasets.zebra.dataset import difficulty_for_size, load_zebra_tasks
from minibench.datasets.zebra.evaluation import plan_zebra_work_items

QUOTAS = {"easy": 34, "medium": 33, "hard": 33}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def independent_euler(vertices, edges, start=None, end=None) -> bool:
    """Check both fixed endpoints using NetworkX's own constructed trail."""
    if not edges:
        return start is None or end is None or start == end
    graph = nx.MultiGraph()
    graph.add_nodes_from(vertices)
    graph.add_edges_from(edges)
    graph.remove_nodes_from(list(nx.isolates(graph)))
    if start is not None and start not in graph:
        return False
    if end is not None and end not in graph:
        return False
    chosen_start = start
    if chosen_start is None and end is not None:
        odd = [v for v, degree in graph.degree() if degree % 2]
        if odd:
            if len(odd) != 2 or end not in odd:
                return False
            chosen_start = next(v for v in odd if v != end)
        else:
            chosen_start = end
    if not nx.has_eulerian_path(graph, source=chosen_start):
        return False
    trail = list(nx.eulerian_path(graph, source=chosen_start, keys=True))
    return end is None or trail[-1][1] == end


def audit(root: Path, *, source_parquet: Path | None = None,
          verify_zebra: bool = True) -> dict[str, Any]:
    paths = {
        "zebra_direct": "data/zebra/eval.jsonl",
        "zebra_codebook": "data/zebra/rule_codebook_eval.jsonl",
        "zebra_counterfactual": "data/zebra/rule_counterfactual_eval.jsonl",
        "zebra_history": "data/zebra/history_eval.jsonl",
        "zebra_rules": "data/zebra/rule_eval.jsonl",
        "one_stroke_direct": "data/one_stroke/direct.jsonl",
        "one_stroke_history": "data/one_stroke/history.jsonl",
        "one_stroke_visual": "data/one_stroke/multimodal.jsonl",
    }
    rows = {key: read_rows(root / relative) for key, relative in paths.items()}
    datasets = {}
    for key, records in rows.items():
        count = 200 if key == "zebra_rules" else 100
        require(len(records) == count, f"{key}: expected {count} records")
        require(len({r["id"] for r in records}) == count, f"{key}: duplicate IDs")
        if key != "zebra_rules":
            difficulty = Counter(difficulty_for_size(r["size"]) if key.startswith("zebra")
                                 else r["difficulty"] for r in records)
            require(dict(difficulty) == QUOTAS, f"{key}: wrong difficulty quotas")
        datasets[key] = {"path": paths[key], "count": count,
                         "sha256": hashlib.sha256((root / paths[key]).read_bytes()).hexdigest()}

    direct = {r["id"]: r for r in rows["zebra_direct"]}
    source_ids = set(direct)
    for key in ("zebra_codebook", "zebra_history"):
        require({r["source_id"] for r in rows[key]} == source_ids, f"{key}: source pairing")
        require(len({r["source_id"] for r in rows[key]}) == 100, f"{key}: duplicate sources")
        for record in rows[key]:
            require(record["solution"] == direct[record["source_id"]]["solution"],
                    f"{record['id']}: paired gold changed")
    for record in rows["zebra_codebook"]:
        decoded = record["puzzle"]
        for mapping in record["rule_mapping"]:
            decoded = decoded.replace(mapping["token"], mapping["meaning"])
        context = "\n".join([
            "The bracketed tokens below are temporary textual macros for this puzzle only.",
            "Expand each token to its quoted meaning before applying the clues.",
            "Do not reuse a token meaning from any other puzzle.", "",
            *(f'- {mapping["token"]} = "{mapping["meaning"]}"'
              for mapping in record["rule_mapping"]),
        ])
        require(record["rule_context"] == context, f"{record['id']}: public codebook meanings differ")
        require(decoded == direct[record["source_id"]]["puzzle"],
                f"{record['id']}: codebook is not reversible")
    for record in rows["zebra_history"]:
        original = direct[record["source_id"]]["puzzle"]
        original_clues = original.split("\n\n## Clues:\n", 1)[1].strip().splitlines()
        expected = [line.split(". ", 1)[1] for line in original_clues]
        require(record["clue_turns"] == expected, f"{record['id']}: clue order/content changed")

    cf = rows["zebra_counterfactual"]
    require(len({r["source_id"] for r in cf}) == 100, "counterfactual: repeated sources")
    for record in cf:
        require(record.get("validation_status") == "verified_unique", f"{record['id']}: unverified CF")
        require(record["solution"] != record["original_solution"], f"{record['id']}: unchanged gold")
        require("not-scoreable" not in record.get("tags", []), f"{record['id']}: candidate leaked")
    combined = {r["id"]: r for r in rows["zebra_rules"]}
    expected_rules = {r["id"]: r for r in [*rows["zebra_codebook"], *cf]}
    require(combined == expected_rules, "combined rule file differs from its component sets")
    semantic_checks = 0
    if verify_zebra:
        from minibench.datasets.zebra.validation import verify_record
        for key in ("zebra_direct", "zebra_codebook", "zebra_history", "zebra_counterfactual"):
            for record in rows[key]:
                proof = verify_record(record)
                require(proof.get("status") == "verified_unique",
                        f"{record['id']}: independent Zebra proof failed: {proof}")
                require(proof.get("gold_asserted") is False, f"{record['id']}: gold was asserted")
                semantic_checks += 1
    source_report = None
    if source_parquet is not None:
        import pyarrow.parquet as pq
        official = {r["id"]: r for r in pq.read_table(source_parquet).to_pylist()}
        for record in rows["zebra_direct"]:
            source = official[record["id"]]
            require(all(record[field] == source[field] for field in ("puzzle", "size", "solution")),
                    f"{record['id']}: differs from official source")
        for record in cf:
            source = official[record["source_id"]]
            require(record["puzzle"] == source["puzzle"] and record["size"] == source["size"]
                    and record["original_solution"] == source["solution"],
                    f"{record['id']}: counterfactual origin mismatch")
        source_report = {"records": len(official),
                         "sha256": hashlib.sha256(source_parquet.read_bytes()).hexdigest()}

    loaded = {key: load_one_stroke_tasks(root / paths[key]) for key in
              ("one_stroke_direct", "one_stroke_history", "one_stroke_visual")}
    direct_graphs = {task.id: task for task in loaded["one_stroke_direct"]}
    visual_source_ids = [task.source_task_id for task in loaded["one_stroke_visual"]]
    require(len(set(visual_source_ids)) == 100 and set(visual_source_ids) == set(direct_graphs),
            "visual: source pairing is not one-to-one")
    solvable = Counter((task.difficulty, task.solution_exists) for task in direct_graphs.values())
    require(dict(solvable) == {("easy", True): 24, ("easy", False): 10,
                              ("medium", True): 23, ("medium", False): 10,
                              ("hard", True): 23, ("hard", False): 10}, "direct: wrong solvable ratio")
    graph_checks = 0
    for key, tasks in loaded.items():
        for task in tasks:
            require(independent_euler(task.vertices, task.edges, task.start, task.end)
                    == task.solution_exists, f"{task.id}: independent Euler verdict mismatch")
            graph_checks += 1
            if key == "one_stroke_history":
                state = simulate_one_stroke_history(task)
                unused = set(state.remaining_edge_ids)
                remaining = [edge for edge_id, edge in zip(one_stroke_edge_ids(task.edges), task.edges)
                             if edge_id in unused]
                require(independent_euler(task.vertices, remaining, state.current_vertex, task.end),
                        f"{task.id}: residual graph has no completion")
    require("history-easy-09" not in {t.id for t in loaded["one_stroke_history"]}, "duplicate history survived")
    history_signatures = [json.dumps({k: r.get(k) for k in
                         ("vertices", "edges", "start", "end", "history_events")}, sort_keys=True)
                         for r in rows["one_stroke_history"]]
    require(len(set(history_signatures)) == 100, "history: exact duplicate inputs")
    for visual in loaded["one_stroke_visual"]:
        source = direct_graphs[visual.source_task_id]
        require(all(getattr(visual, name) == getattr(source, name) for name in
                    ("vertices", "edges", "start", "end", "solution_exists", "solution_path", "difficulty")),
                f"{visual.id}: text/image graph mismatch")
        with Image.open(visual.image_path) as image:
            image.verify()

    plans = {
        "zebra_direct": len(plan_zebra_work_items(load_zebra_tasks(root / paths["zebra_direct"]))),
        "zebra_rules": len(plan_zebra_work_items(load_zebra_tasks(root / paths["zebra_rules"]))),
        "zebra_history": len(plan_zebra_work_items(load_zebra_tasks(root / paths["zebra_history"]),
                                                  ("incremental_state", "deferred_reasoning"))),
        "one_stroke_direct": len(plan_one_stroke_work_items(loaded["one_stroke_direct"])),
        "one_stroke_history": len(plan_one_stroke_work_items(loaded["one_stroke_history"])),
        "one_stroke_visual": len(plan_one_stroke_work_items(loaded["one_stroke_visual"])),
    }
    require(plans == {"zebra_direct": 100, "zebra_rules": 200, "zebra_history": 200,
                      "one_stroke_direct": 100, "one_stroke_history": 200, "one_stroke_visual": 100},
            f"wrong work plan: {plans}")
    manifest_path = root / "config/scoring/manifest.example.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    for experiment in manifest["experiments"]:
        if experiment["id"].startswith(("zebra-", "one-stroke-")):
            path = (manifest_path.parent / experiment["dataset"]["path"]).resolve()
            require(hashlib.sha256(path.read_bytes()).hexdigest() == experiment["dataset"]["sha256"],
                    f"{experiment['id']}: scoring hash mismatch")
    return {"status": "passed", "datasets": datasets, "difficulty_quotas": QUOTAS,
            "zebra_semantic_records_verified": semantic_checks, "zebra_source": source_report,
            "counterfactual_paired_sources": len({r["source_id"] for r in cf} & source_ids),
            "counterfactual_supplemental_sources": sorted({r["source_id"] for r in cf} - source_ids),
            "independent_graph_checks": graph_checks, "valid_images": 100,
            "work_plans": plans, "total_instances": sum(plans.values())}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-parquet", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = audit(ROOT, source_parquet=args.source_parquet)
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
