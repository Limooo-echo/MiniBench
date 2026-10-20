"""Build a staged, solver-certified Zebra release from the local public pool."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from import_zebra_tasks import (CLUE_BANDS, DIFFICULTIES, _balanced_targets,
    convert_official_records, difficulty_targets, frozen_clue_bands)
from derive_zebra_variants import (COUNTERFACTUAL_PATTERNS, derive_codebook_record, derive_history_record,
    _base_variant, _counterfactual_context, _variant_tags, load_jsonl, write_jsonl)
from minibench.datasets.zebra.dataset import difficulty_for_size, load_zebra_tasks
from minibench.datasets.zebra.validation import (applicable_operators,
    parse_puzzle, solve_unique, verify_record)

SOURCE_SHA256 = "545cd5cce9c54575c7f74769b8f1250b8bb94f5745519c83b99b8db01f95924d"
ROOT = Path(__file__).resolve().parents[1]
RELEASE_FILES = ("eval.jsonl", "rule_codebook_eval.jsonl", "history_eval.jsonl",
                 "rule_counterfactual_eval.jsonl", "rule_eval.jsonl")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_balanced(pool: list[dict[str, Any]], eligible: set[str], *, bands: dict[str, str],
                    targets: dict[str, int], seed: int,
                    preserve: set[str] = frozenset(), prefer: set[str] = frozenset()) -> list[str]:
    """Deterministic minimum-cost flow with exact difficulty/size/band quotas.

    Retained records are mandatory. Preferred source overlap has higher priority
    than the seeded sampling objective, which only chooses between valid sets.
    """
    import networkx as nx
    rng = random.Random(seed)
    chosen: list[str] = []
    for difficulty in DIFFICULTIES:
        full = [r for r in pool if difficulty_for_size(r["size"]) == difficulty]
        rows = sorted((r for r in full if r["id"] in eligible), key=lambda r: r["id"])
        sizes = sorted({r["size"] for r in full})
        size_quota = _balanced_targets(sizes, targets[difficulty], rng)
        band_quota = _balanced_targets(list(CLUE_BANDS), targets[difficulty], rng)
        forced = [r for r in rows if r["id"] in preserve]
        for row in forced:
            size_quota[row["size"]] -= 1
            band_quota[bands[row["id"]]] -= 1
        if min([*size_quota.values(), *band_quota.values()]) < 0:
            raise ValueError(f"retained records exceed {difficulty} quotas")
        graph = nx.DiGraph()
        for size in sizes:
            graph.add_node(f"size:{size}", demand=-size_quota[size])
        for band in CLUE_BANDS:
            graph.add_node(f"band:{band}", demand=band_quota[band])
        ranks = list(range(1, len(rows) + 1))
        rng.shuffle(ranks)
        preference_weight = (len(rows) + 1) * (targets[difficulty] + 1)
        for row, rank in zip(rows, ranks):
            if row["id"] in preserve:
                continue
            node = f"candidate:{row['id']}"
            graph.add_node(node, demand=0)
            graph.add_edge(f"size:{row['size']}", node, capacity=1,
                           weight=rank - preference_weight * int(row["id"] in prefer))
            graph.add_edge(node, f"band:{bands[row['id']]}", capacity=1, weight=0)
        try:
            _, flow = nx.network_simplex(graph)
        except nx.NetworkXUnfeasible as exc:
            raise ValueError(f"cannot satisfy certified {difficulty} size/band quotas; "
                f"eligible={len(rows)}, sizes={size_quota}, bands={band_quota}") from exc
        chosen.extend(row["id"] for row in rows if row["id"] in preserve or
                      flow[f"size:{row['size']}"].get(f"candidate:{row['id']}", 0))
    return chosen


def make_counterfactual(source: dict[str, Any], operator: str, evidence: dict[str, Any],
                        seed: int) -> dict[str, Any]:
    houses = int(source["size"].split("*")[0])
    context, ordinary, altered = _counterfactual_context(operator, houses)
    context = context.replace(" (pending solution validation)", "")
    return {**_base_variant(source, "counterfactual_semantics", seed),
        "id": f"{source['id']}__counterfactual_semantics__{operator}",
        "puzzle": source["puzzle"], "solution": evidence["solution"],
        "original_solution": source["solution"], "capability": "rule_condition",
        "rule_mode": "counterfactual_semantics", "rule_context": context,
        "counterfactual_rule": {"operator": operator, "original_semantics": ordinary,
            "new_semantics": altered, "affected_clue_indices": evidence["affected_clue_indices"]},
        "clue_turns": [], "validation_status": "verified_unique",
        "validation_evidence": {k: v for k, v in evidence.items() if k != "solution"},
        "tags": _variant_tags(source, "counterfactual-verified",
                              extra=("validation:verified_unique", f"rule-operator:{operator}"))}


def build(args: argparse.Namespace) -> dict[str, Any]:
    import pyarrow.parquet as pq
    import networkx as nx
    stage: Path = args.staging_dir
    stage.mkdir(parents=True, exist_ok=True)
    if digest(args.source_parquet) != SOURCE_SHA256:
        raise ValueError("public pool SHA256 does not match the reviewed source archive")
    pool = convert_official_records(pq.read_table(args.source_parquet).to_pylist())
    if len(pool) != 1000 or len({r["id"] for r in pool}) != len(pool):
        raise ValueError("expected the complete 1,000-record public pool")
    by_id = {r["id"]: r for r in pool}
    bands = frozen_clue_bands(pool)
    old = load_jsonl(args.output_dir / "eval.jsonl")
    smoke = {r["id"] for r in load_jsonl(args.output_dir / "tasks.jsonl")}
    # Rebuilds preserve the original frozen 45 rather than treating an expanded
    # release as mandatory, so rerunning the tool reproduces the same selection.
    legacy = args.output_dir / "legacy" / "20260810" / "eval.jsonl"
    if legacy.is_file():
        old = load_jsonl(legacy)
    for row in old:
        source = by_id[row["id"]]
        if any(row[key] != source[key] for key in ("size", "puzzle", "solution")):
            raise ValueError(f"legacy record differs from source: {row['id']}")
    old_by_id = {r["id"]: r for r in old}
    validation_path = Path(__file__).resolve().parents[1] / "src/minibench/datasets/zebra/validation.py"
    cache_identity = {"source_sha256": SOURCE_SHA256, "validator_sha256": digest(validation_path),
                      "timeout_ms": args.timeout_ms}
    cache_path = stage / "pool_proofs.json"
    cache: dict[str, Any] = {"identity": cache_identity, "records": {}}
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("identity") == cache_identity:
            cache = cached
    proofs = cache["records"]
    for index, source in enumerate(sorted(pool, key=lambda r: r["id"]), 1):
        if source["id"] in proofs:
            continue
        original = verify_record(source, timeout_ms=args.timeout_ms)
        mutations: dict[str, Any] = {}
        if original["status"] == "verified_unique":
            parsed = parse_puzzle(source["puzzle"], expected_size=source["size"])
            for operator in applicable_operators(parsed):
                evidence = solve_unique(parsed, override=operator, timeout_ms=args.timeout_ms)
                if evidence["status"] == "verified_unique":
                    evidence["answer_changed"] = evidence["solution"] != source["solution"]
                mutations[operator] = evidence
        proofs[source["id"]] = {"original": original, "counterfactual": mutations}
        if index % 25 == 0 or index == len(pool):
            cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8", newline="\n")
            print(json.dumps({"validated_sources": index, "pool": len(pool)}), flush=True)
    # Reused proof caches must also use canonical bytes after Windows builds.
    cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8", newline="\n")
    targets = difficulty_targets(args.target_total)
    eligible_direct = {key for key, p in proofs.items() if p["original"]["status"] == "verified_unique"} - smoke
    retained = set(old_by_id) & eligible_direct
    selected_ids = select_balanced(pool, eligible_direct, bands=bands, targets=targets,
                                  seed=args.seed, preserve=retained)
    direct = []
    for key in selected_ids:
        if key in old_by_id:
            direct.append(old_by_id[key])  # Preserve every retained raw record exactly.
        else:
            row = dict(by_id[key])
            row["tags"] = [*row["tags"], "split:test", "benchmark:eval",
                "sampling:balanced-size-clue-band", f"selection-seed:{args.seed}", f"clue-band:{bands[key]}"]
            direct.append(row)
    eligible_cf = {key for key, proof in proofs.items() if key in eligible_direct and any(
        value["status"] == "verified_unique" and value.get("answer_changed")
        for value in proof["counterfactual"].values())}
    cf_ids = select_balanced(pool, eligible_cf, bands=bands, targets=targets,
                            seed=args.seed, prefer=set(selected_ids))
    cf = []
    operator_counts: dict[str, Counter[str]] = {difficulty: Counter() for difficulty in DIFFICULTIES}
    for key in cf_ids:
        source = old_by_id.get(key, by_id[key])
        candidates = proofs[key]["counterfactual"]
        difficulty = difficulty_for_size(source["size"])
        rng = random.Random(f"zebra-counterfactual:{args.seed}:{key}")
        operators = [op for op, value in candidates.items()
                     if value["status"] == "verified_unique" and value.get("answer_changed")]
        rng.shuffle(operators)
        operator = min(operators, key=lambda op: operator_counts[difficulty][op])
        operator_counts[difficulty][operator] += 1
        cf.append(make_counterfactual(source, operator, candidates[operator], args.seed))
    codebook = [derive_codebook_record(r, seed=args.seed) for r in direct]
    history = [derive_history_record(r, seed=args.seed) for r in direct]
    outputs = dict(zip(RELEASE_FILES, (direct, codebook, history, cf, codebook + cf)))
    certificates = []
    for filename, rows in outputs.items():
        if len(rows) != (args.target_total * 2 if filename == "rule_eval.jsonl" else args.target_total):
            raise ValueError(f"unexpected count for {filename}")
        if len({r["id"] for r in rows}) != len(rows):
            raise ValueError(f"duplicate task ID in {filename}")
        write_jsonl(rows, stage / filename, overwrite=True)
        load_zebra_tasks(stage / filename)
        if filename == "rule_eval.jsonl":
            continue
        if Counter(difficulty_for_size(r["size"]) for r in rows) != Counter(targets):
            raise ValueError(f"difficulty quotas failed for {filename}")
        for row in rows:
            proof = verify_record(row, timeout_ms=args.timeout_ms)
            if proof["status"] != "verified_unique":
                raise ValueError(f"release verification failed {row['id']}: {proof}")
            certificates.append({"file": filename, "id": row["id"], "source_id": row.get("source_id", row["id"]),
                                 "clue_band": bands[row.get("source_id", row["id"])], **proof})
    write_jsonl(certificates, stage / "validation.jsonl", overwrite=True)
    report = {"seed": args.seed, "target_total": args.target_total, "difficulty_targets": targets,
        "source_path": str(args.source_parquet.resolve()), "source_sha256": SOURCE_SHA256,
        "validator_sha256": cache_identity["validator_sha256"], "records": {k: len(v) for k, v in outputs.items()},
        "builder_sha256": digest(Path(__file__).resolve()),
        "pool_original_statuses": dict(Counter(p["original"]["status"] for p in proofs.values())),
        "pool_counterfactual_statuses": dict(Counter(v["status"] for p in proofs.values() for v in p["counterfactual"].values())),
        "counterfactual_eligible_sources": len(eligible_cf),
        "original_45_retained": sorted(retained), "original_rejections": sorted(set(old_by_id) - retained),
        "direct_added_ids": sorted(set(selected_ids) - retained),
        "codebook_counterfactual_source_intersection": sorted(set(selected_ids) & set(cf_ids)),
        "counterfactual_fill_ids": sorted(set(cf_ids) - set(selected_ids)),
        "counterfactual_operator_counts": {d: dict(c) for d, c in operator_counts.items()},
        "counterfactual_attempted_operator_counts": {op: sum(op in p["counterfactual"] for p in proofs.values())
                                                   for op in COUNTERFACTUAL_PATTERNS},
        "source_sampling": {"algorithm": "deterministic-minimum-cost-flow-v1", "networkx_version": nx.__version__,
            "clue_band_frozen_before_exclusions": True,
            "selected": [{"id": key, "size": by_id[key]["size"], "clue_band": bands[key]} for key in selected_ids],
            "counterfactual": [{"id": key, "size": by_id[key]["size"], "clue_band": bands[key]} for key in cf_ids]},
        "file_sha256": {filename: digest(stage / filename) for filename in RELEASE_FILES},
        "validation_sha256": digest(stage / "validation.jsonl"),
        "all_release_records_verified": True, "history_modes_share_source_and_clue_order": True}
    (stage / "release_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8", newline="\n")
    if args.publish:
        archive = args.output_dir / "legacy" / "20260810"
        archive.mkdir(parents=True, exist_ok=True)
        # Only these four files existed in the 45-task release. Never backfill
        # newly created formal CF/combined files into the older legacy archive.
        for filename in ("eval.jsonl", "rule_codebook_eval.jsonl", "history_eval.jsonl",
                         "rule_counterfactual_candidates.jsonl"):
            existing = args.output_dir / filename
            if existing.is_file() and not (archive / filename).exists():
                shutil.copy2(existing, archive / filename)
        for filename in (*RELEASE_FILES, "validation.jsonl", "release_manifest.json", "pool_proofs.json"):
            shutil.copy2(stage / filename, args.output_dir / filename)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-parquet", type=Path, default=Path(
        ROOT.parent / "Dataset/ZebraLogic/grid_mode/test-00000-of-00001.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/zebra"))
    parser.add_argument("--staging-dir", type=Path, default=Path("tmp/benchmark-expansion-20261013/zebra"))
    parser.add_argument("--target-total", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--timeout-ms", type=int, default=10000)
    parser.add_argument("--publish", action="store_true", help="Replace formal files only after staged certification.")
    return parser


def main(argv: list[str] | None = None) -> int:
    report = build(build_parser().parse_args(argv))
    print(json.dumps({key: report[key] for key in ("records", "pool_original_statuses",
        "counterfactual_eligible_sources", "counterfactual_operator_counts")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
