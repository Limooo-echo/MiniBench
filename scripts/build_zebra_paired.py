"""Freeze 100 identical Zebra sources under native, codebook and changed rules.

No model calls or downloads are made. The certified counterfactual release
already stores its original puzzle and original gold, so it supplies the common
source set without requiring access to the upstream parquet file. Every output
is solved afresh without asserting its gold, including the reconstructed native
tasks. Existing evaluation datasets are never modified.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from derive_zebra_variants import derive_codebook_record, load_jsonl, write_jsonl
from minibench.datasets.zebra.dataset import difficulty_for_size, load_zebra_tasks
from minibench.datasets.zebra.validation import verify_record


GROUP_FILES = {
    "direct": "direct.jsonl",
    "codebook": "codebook.jsonl",
    "counterfactual": "counterfactual.jsonl",
}
SOURCE_COUNT = 100
QUOTAS = {"easy": 34, "medium": 33, "hard": 33}
VALIDATOR = ROOT / "src/minibench/datasets/zebra/validation.py"
DATASET_ID = "zebra-paired-20261027"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_ids(rows: list[dict[str, Any]]) -> list[str]:
    identifiers = [row.get("source_id", row.get("id")) for row in rows]
    if not all(isinstance(value, str) and value for value in identifiers):
        raise ValueError("every task must have a non-empty source ID")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("duplicate source ID")
    return identifiers


def paired_records(
    counterfactual: list[dict[str, Any]],
    original_direct: list[dict[str, Any]],
    *,
    seed: int = 20261013,
    expected_count: int = SOURCE_COUNT,
) -> dict[str, list[dict[str, Any]]]:
    """Reconstruct native tasks and preserve the CF source order exactly."""
    if len(counterfactual) != expected_count:
        raise ValueError(f"expected {expected_count} counterfactual sources")
    source_ids = _source_ids(counterfactual)
    previous_ids = _source_ids(original_direct)
    previous = dict(zip(previous_ids, original_direct))
    direct = []
    for source_id, changed in zip(source_ids, counterfactual):
        if (changed.get("rule_mode") != "counterfactual_semantics"
                or changed.get("validation_status") != "verified_unique"):
            raise ValueError(f"not a certified counterfactual parent: {source_id}")
        if not isinstance(changed.get("original_solution"), dict):
            raise ValueError(f"missing original solution: {source_id}")
        native = {
            "id": source_id,
            "source_id": source_id,
            "variant": "native",
            "size": changed["size"],
            "puzzle": changed["puzzle"],
            "solution": deepcopy(changed["original_solution"]),
            "capability": "direct",
            "rule_mode": None,
            "rule_context": None,
            "clue_turns": [],
            "tags": ["source:WildEval/ZebraLogic", "split:test", "benchmark:eval",
                     f"paired-release:{DATASET_ID}"],
        }
        if source_id in previous:
            if any(native[key] != previous[source_id][key]
                   for key in ("puzzle", "size", "solution")):
                raise ValueError(f"original parent disagrees with direct release: {source_id}")
        direct.append(native)
    groups = {
        "direct": direct,
        "codebook": [derive_codebook_record(row, seed=seed) for row in direct],
        "counterfactual": deepcopy(counterfactual),
    }
    check_pairing(groups, expected_count=expected_count)
    return groups


def check_pairing(
    groups: dict[str, list[dict[str, Any]]], *, expected_count: int = SOURCE_COUNT,
) -> list[str]:
    """Reject unpaired sources, altered clues, or inconsistent gold contracts."""
    if set(groups) != set(GROUP_FILES):
        raise ValueError("expected exactly direct, codebook and counterfactual groups")
    source_ids = _source_ids(groups["direct"])
    if len(source_ids) != expected_count:
        raise ValueError(f"expected {expected_count} unique sources per group")
    for group, rows in groups.items():
        if _source_ids(rows) != source_ids:
            raise ValueError(f"source set/order differs for {group}")
        if len({row["id"] for row in rows}) != len(rows):
            raise ValueError(f"duplicate task ID in {group}")
    for native, encoded, changed in zip(*(groups[name] for name in GROUP_FILES)):
        source_id = native["source_id"]
        if (native["id"] != source_id or native.get("capability") != "direct"
                or native.get("rule_mode") is not None or native.get("rule_context") is not None
                or native.get("clue_turns")):
            raise ValueError(f"native task contains rule/history changes: {source_id}")
        if encoded.get("rule_mode") != "temporary_codebook":
            raise ValueError(f"incorrect codebook mode: {source_id}")
        if changed.get("rule_mode") != "counterfactual_semantics":
            raise ValueError(f"incorrect counterfactual mode: {source_id}")
        decoded = encoded["puzzle"]
        for item in encoded["rule_mapping"]:
            decoded = decoded.replace(item["token"], item["meaning"])
        if (decoded != native["puzzle"] or changed["puzzle"] != native["puzzle"]
                or encoded["size"] != native["size"] or changed["size"] != native["size"]):
            raise ValueError(f"variant changes source clues or grid: {source_id}")
        if encoded["solution"] != native["solution"]:
            raise ValueError(f"codebook gold differs from native: {source_id}")
        if changed["original_solution"] != native["solution"]:
            raise ValueError(f"counterfactual original gold differs: {source_id}")
        if changed["solution"] == native["solution"]:
            raise ValueError(f"counterfactual answer must change: {source_id}")
    return source_ids


def verify_groups(
    groups: dict[str, list[dict[str, Any]]], *, timeout_ms: int = 10000,
) -> list[dict[str, Any]]:
    """Fresh blind SAT/uniqueness/gold checks; never accepts cached evidence."""
    if timeout_ms < 1:
        raise ValueError("timeout_ms must be positive")
    certificates = []
    for group, rows in groups.items():
        for row in rows:
            proof = verify_record(row, timeout_ms=timeout_ms)
            if (proof.get("status") != "verified_unique"
                    or proof.get("first_check") != "sat"
                    or proof.get("second_check") != "unsat"
                    or proof.get("gold_asserted") is not False
                    or proof.get("gold_matches") is not True
                    or proof.get("integer_clauses_satisfied") is not True):
                raise ValueError(f"fresh verification failed for {group}/{row['id']}: {proof}")
            certificates.append({"group": group, "task_id": row["id"],
                                 "source_id": row.get("source_id", row["id"]),
                                 **{k: v for k, v in proof.items() if k != "solution"}})
    return certificates


def _verification_summary(certificates: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "fresh": True,
        "instances": len(certificates),
        "verified_unique": len(certificates),
        "per_group": dict(Counter(item["group"] for item in certificates)),
        "statuses": dict(Counter(item["status"] for item in certificates)),
        "gold_matches": all(item["gold_matches"] for item in certificates),
        "gold_asserted": False,
        "solver_versions": sorted({item["solver"] for item in certificates}),
    }


def verify_paired_release(
    path: str | Path, *, fresh: bool = True, timeout_ms: int = 10000,
) -> dict[str, Any]:
    """Verify frozen bytes and pairing, then (by default) freshly solve all 300."""
    directory = Path(path)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    expected_files = {*GROUP_FILES.values(), "validation.jsonl"}
    if (manifest.get("schema_version") != 1 or manifest.get("dataset_id") != DATASET_ID
            or set(manifest.get("files", {})) != expected_files):
        raise ValueError("invalid paired release manifest")
    if manifest.get("validator_sha256") != digest(VALIDATOR):
        raise ValueError("validator differs from the frozen release; rebuild before running")
    for filename, entry in manifest["files"].items():
        if digest(directory / filename) != entry["sha256"]:
            raise ValueError(f"frozen file hash mismatch: {filename}")
    groups = {group: load_jsonl(directory / filename) for group, filename in GROUP_FILES.items()}
    for filename in GROUP_FILES.values():
        load_zebra_tasks(directory / filename)
    identifiers = check_pairing(groups)
    if identifiers != manifest.get("source_ids"):
        raise ValueError("source order differs from the manifest")
    if Counter(difficulty_for_size(row["size"]) for row in groups["direct"]) != Counter(QUOTAS):
        raise ValueError("difficulty quotas must be 34/33/33")
    saved = load_jsonl(directory / "validation.jsonl")
    expected_keys = [(group, row["id"], row.get("source_id", row["id"]))
                     for group, rows in groups.items() for row in rows]
    if [(c.get("group"), c.get("task_id"), c.get("source_id")) for c in saved] != expected_keys:
        raise ValueError("validation certificates differ from the frozen task order")
    for filename, entry in manifest["files"].items():
        expected_count = 300 if filename == "validation.jsonl" else SOURCE_COUNT
        if entry.get("records") != expected_count:
            raise ValueError(f"manifest record count mismatch: {filename}")
    certificates = verify_groups(groups, timeout_ms=timeout_ms) if fresh else saved
    if any(c.get("status") != "verified_unique" or c.get("gold_matches") is not True
           for c in certificates):
        raise ValueError("release contains failed verification certificates")
    summary = _verification_summary(certificates)
    summary["fresh"] = fresh
    return {"source_count": len(identifiers), "source_ids": identifiers,
            "manifest_sha256": digest(directory / "manifest.json"), **summary}


def _path_label(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def build_release(
    output_dir: Path, *, counterfactual_path: Path, direct_path: Path,
    seed: int = 20261013, timeout_ms: int = 10000,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"{output_dir} exists; use --verify-only to recheck its frozen files")
    counterfactual = load_jsonl(counterfactual_path)
    previous = load_jsonl(direct_path)
    groups = paired_records(counterfactual, previous, seed=seed)
    identifiers = check_pairing(groups)
    if Counter(difficulty_for_size(r["size"]) for r in groups["direct"]) != Counter(QUOTAS):
        raise ValueError("expected 34 easy, 33 medium and 33 hard shared sources")
    certificates = verify_groups(groups, timeout_ms=timeout_ms)
    # Do not create output files until every solver check succeeds.
    output_dir.mkdir(parents=True)
    files = {}
    for group, filename in GROUP_FILES.items():
        write_jsonl(groups[group], output_dir / filename, overwrite=False)
        load_zebra_tasks(output_dir / filename)
        files[filename] = {"sha256": digest(output_dir / filename), "records": len(groups[group])}
    write_jsonl(certificates, output_dir / "validation.jsonl", overwrite=False)
    files["validation.jsonl"] = {"sha256": digest(output_dir / "validation.jsonl"),
                                "records": len(certificates)}
    old_ids = set(_source_ids(previous))
    manifest = {
        "schema_version": 1,
        "dataset_id": DATASET_ID,
        "selection": "Existing verified counterfactual sources and order, shared by all three groups",
        "source_count": SOURCE_COUNT,
        "source_ids": identifiers,
        "difficulty_counts": QUOTAS,
        "codebook_seed": seed,
        "counterfactual_gold_differs_from_native": True,
        "source_clues_identical_after_codebook_expansion": True,
        "previous_direct_overlap": sum(key in old_ids for key in identifiers),
        "added_to_direct": [key for key in identifiers if key not in old_ids],
        "parents": {
            "counterfactual": {"path": _path_label(counterfactual_path), "sha256": digest(counterfactual_path)},
            "previous_direct": {"path": _path_label(direct_path), "sha256": digest(direct_path)},
        },
        "builder_sha256": digest(Path(__file__)),
        "validator_sha256": digest(VALIDATOR),
        "verification": _verification_summary(certificates),
        "files": files,
        "comparison_limit": "Counterfactual changes constraints and gold; shared sources do not imply equal solve difficulty.",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/zebra/paired-20261027")
    parser.add_argument("--counterfactual", type=Path, default=ROOT / "data/zebra/rule_counterfactual_eval.jsonl")
    parser.add_argument("--previous-direct", type=Path, default=ROOT / "data/zebra/eval.jsonl")
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--timeout-ms", type=int, default=10000)
    parser.add_argument("--verify-only", action="store_true", help="Freshly solve the existing frozen 300 tasks.")
    args = parser.parse_args(argv)
    if args.verify_only:
        report = verify_paired_release(args.output_dir, timeout_ms=args.timeout_ms)
    else:
        manifest = build_release(args.output_dir, counterfactual_path=args.counterfactual,
                                 direct_path=args.previous_direct, seed=args.seed,
                                 timeout_ms=args.timeout_ms)
        report = {"source_count": manifest["source_count"], **manifest["verification"]}
    print(json.dumps({key: value for key, value in report.items() if key != "source_ids"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
