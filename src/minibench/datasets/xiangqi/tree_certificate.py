"""Content-bound, fail-closed evidence for independently verified strategy DAGs.

This module does not prove a tree. It checks that the saved independent search
results refer to exactly the current data, tree bytes, rules and checker code.
Unknown or incomplete search results never become a passing certificate.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

from .runtime import PROTOCOL_VERSION

REPORT_VERSION = "xiangqi-independent-tree-verification-report-v1"
REPORT_FILENAME = "solution_tree_verification.json"
FAMILY_FOLDERS = {"xiangqi-history": "history", "xiangqi-rule-variants": "rule_variants"}
TREE_FOLDERS = {"xiangqi-history": "h2", "xiangqi-rule-variants": "c2"}
ROOT = Path(__file__).resolve().parents[4]
VERIFIER_FILES = (
    "src/minibench/datasets/xiangqi/tree_verification.py",
    "src/minibench/datasets/xiangqi/tree_certificate.py",
    "src/minibench/datasets/xiangqi/variants/board.py",
    "src/minibench/datasets/xiangqi/variants/rules.py",
    "src/minibench/datasets/xiangqi/reference.py",
    "src/minibench/datasets/xiangqi/schema.py",
    "src/minibench/datasets/xiangqi/validation.py",
    "src/minibench/datasets/xiangqi/runtime.py",
    "src/minibench/datasets/xiangqi/solution_tree.py",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record_sha256(record: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(record, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def verifier_hashes() -> dict[str, str]:
    return {name: sha256_file(ROOT / name) for name in VERIFIER_FILES}


def verifier_fingerprint(hashes: dict[str, str] | None = None) -> str:
    return record_sha256(verifier_hashes() if hashes is None else hashes)


def _rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or any(not isinstance(row, dict) or not isinstance(row.get("id"), str) for row in rows):
        raise ValueError("Independent tree evidence requires nonempty object records")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Independent tree evidence requires unique task IDs")
    return rows


def _status(result: dict[str, Any]) -> str:
    value = result.get("status")
    if value in {"valid", "verified"} and result.get("valid") is True:
        return "verified"
    if value == "invalid":
        return "invalid"
    return "unknown"


def _complete_proof(result: dict[str, Any], record: dict[str, Any]) -> bool:
    """A positive status alone is not an independent proof transcript."""
    from .tree_verification import VERIFICATION_VERSION
    checks = result.get("checks")
    first_moves = result.get("optimal_root_moves_uci")
    return (
        result.get("version") == VERIFICATION_VERSION
        and result.get("computed_shortest_attacker_moves") == record.get("shortest_mate_moves")
        and result.get("proof_complete") is True
        and result.get("coverage_verified") is True
        and isinstance(checks, dict)
        and all(checks.get(key) is True for key in
                ("root_shortest", "optimal_root_set", "all_nodes_and_edges"))
        and isinstance(first_moves, list) and bool(first_moves)
        and all(isinstance(move, str) for move in first_moves)
        and len(set(first_moves)) == len(first_moves)
        and isinstance(result.get("nodes_verified"), int)
        and not isinstance(result.get("nodes_verified"), bool)
        and result["nodes_verified"] > 0
    )


def build_verification_report(data_root: Path, results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Build a portable evidence envelope; caller persists it atomically.

    Every current H2/C2 row is included. A missing result becomes unknown, not a
    success. A result must carry the bytes hashes captured when it was searched;
    otherwise it cannot be bound to a later-modified tree or dataset record.
    """
    data_root = Path(data_root)
    inputs, records = {}, {}
    code_hashes = verifier_hashes()
    source_fingerprint = verifier_fingerprint(code_hashes)
    for family, folder in FAMILY_FOLDERS.items():
        path = data_root / folder / "tasks.jsonl"
        rows = _rows(path)
        inputs[family] = {"path": f"data/xiangqi/{folder}/tasks.jsonl",
                          "sha256": sha256_file(path), "record_count": len(rows),
                          "task_ids": [row["id"] for row in rows]}
        for record in rows:
            task_id = record["id"]
            tree_path = data_root / "solution_trees" / TREE_FOLDERS[family] / f"{task_id}.json"
            tree_hash = sha256_file(tree_path) if tree_path.is_file() else None
            raw = dict(results.get(task_id) or {"status": "unknown", "valid": None,
                                               "reasons": ["not_verified"]})
            identity_hash = record_sha256(record)
            status = _status(raw)
            if status == "verified" and not _complete_proof(raw, record):
                status = "unknown"
                raw = {**raw, "status": "unknown", "valid": None,
                       "reasons": [*raw.get("reasons", []), "independent_tree_proof_incomplete_or_inconsistent"]}
            if (raw.get("record_sha256") != identity_hash
                    or raw.get("tree_sha256") != tree_hash or tree_hash is None
                    or raw.get("verification_source_sha256") != source_fingerprint):
                status = "unknown"
                raw = {**raw, "status": "unknown", "valid": None,
                       "reasons": [*raw.get("reasons", []), "verification_input_binding_missing_or_changed"]}
            records[task_id] = {
                "family": family, "record_sha256": identity_hash,
                "tree_path": f"data/xiangqi/solution_trees/{TREE_FOLDERS[family]}/{task_id}.json",
                "tree_sha256": tree_hash, "status": status, "verification": raw,
            }
    unknown_ids = set(results) - set(records)
    if unknown_ids:
        raise ValueError("Verification results contain tasks outside the current datasets")
    counts = Counter(item["status"] for item in records.values())
    complete = bool(records) and counts["verified"] == len(records)
    return {"version": REPORT_VERSION, "protocol_version": PROTOCOL_VERSION,
            "verifier_hashes": code_hashes, "verification_source_sha256": source_fingerprint,
            "inputs": inputs, "records": records,
            "summary": {"total": len(records), "verified": counts["verified"],
                        "invalid": counts["invalid"], "unknown": counts["unknown"]},
            "valid": complete, "complete": complete,
            "claim": "Independent bounded search of shortest distances and all declared strategy edges; not merely positive replay",
            "limitations": ["Only the declared finite-horizon rules are covered; official long-check/chase adjudication is not included."]}


def validate_saved_tree_certificate(
    records: list[dict[str, Any]], *, data_path: Path, tree_dir: Path,
    report_path: Path | None = None,
) -> dict[str, Any]:
    """Check evidence for selected current records without rerunning search.

    Other unselected records may remain unknown. A full-release audit passes all
    100+100 rows, so it cannot pass unless every submitted tree is verified.
    """
    data_path, tree_dir = Path(data_path), Path(tree_dir)
    path = Path(report_path) if report_path is not None else data_path.parent.parent / REPORT_FILENAME
    result: dict[str, Any] = {"valid": False, "reasons": [], "records": {},
                              "record_count": len(records), "report_path": str(path)}
    try:
        report_bytes = path.read_bytes()
        report = json.loads(report_bytes)
        result["report_sha256"] = hashlib.sha256(report_bytes).hexdigest()
        if report.get("version") != REPORT_VERSION or report.get("protocol_version") != PROTOCOL_VERSION:
            raise ValueError("independent_tree_report_version_or_protocol_mismatch")
        if report.get("verifier_hashes") != verifier_hashes():
            raise ValueError("independent_tree_verifier_code_changed")
        source_fingerprint = verifier_fingerprint(report["verifier_hashes"])
        if report.get("verification_source_sha256") != source_fingerprint:
            raise ValueError("independent_tree_verifier_fingerprint_missing_or_changed")
        family = next((f for f, folder in FAMILY_FOLDERS.items() if data_path.parent.name == folder), None)
        if family is None:
            raise ValueError("independent_tree_dataset_family_unknown")
        canonical_rows = _rows(data_path)
        canonical = {row["id"]: row for row in canonical_rows}
        identity = report.get("inputs", {}).get(family, {})
        if (identity.get("path") != f"data/xiangqi/{FAMILY_FOLDERS[family]}/tasks.jsonl"
                or identity.get("sha256") != sha256_file(data_path)
                or identity.get("record_count") != len(canonical_rows)
                or identity.get("task_ids") != [row["id"] for row in canonical_rows]):
            raise ValueError("independent_tree_dataset_changed")
        if not records or len({row["id"] for row in records}) != len(records):
            raise ValueError("independent_tree_selected_records_empty_or_duplicate")
        expected_ids = {task_id for entry in report.get("inputs", {}).values()
                        for task_id in entry.get("task_ids", [])}
        if (set(report.get("inputs", {})) != set(FAMILY_FOLDERS)
                or set(report.get("records", {})) != expected_ids):
            raise ValueError("independent_tree_report_roster_incomplete")
        for supplied in records:
            task_id = supplied["id"]
            current = canonical.get(task_id)
            if current is None:
                raise ValueError(f"independent_tree_record_missing:{task_id}")
            # H2 passes a deliberately small projection; C2 adds board/internal
            # rule objects. Compare the persisted identity fields, never runtime
            # derived fields, against the content-hashed canonical JSONL row.
            for key in ("id", "fen", "family", "ruleset", "agent_color", "max_plies",
                        "oracle", "shortest_proof", "shortest_mate_moves"):
                if key in supplied and supplied[key] != current.get(key):
                    raise ValueError(f"independent_tree_runtime_record_changed:{task_id}:{key}")
            item = report["records"].get(task_id, {})
            proof = current.get("shortest_proof", {})
            tree_path = tree_dir / f"{task_id}.json"
            tree_hash = sha256_file(tree_path)
            expected_path = f"data/xiangqi/solution_trees/{TREE_FOLDERS[family]}/{task_id}.json"
            raw = item.get("verification", {})
            reasons = []
            if (item.get("status") != "verified" or _status(raw) != "verified"
                    or not _complete_proof(raw, current)):
                reasons.append("independent_tree_not_verified")
            if (item.get("family") != family or item.get("record_sha256") != record_sha256(current)
                    or raw.get("record_sha256") != item.get("record_sha256")
                    or raw.get("verification_source_sha256") != source_fingerprint):
                reasons.append("independent_tree_record_binding_changed")
            if (item.get("tree_path") != expected_path or item.get("tree_sha256") != tree_hash
                    or raw.get("tree_sha256") != tree_hash or proof.get("tree_sha256") != tree_hash):
                reasons.append("independent_tree_bytes_changed")
            result["records"][task_id] = {"valid": not reasons, "status": item.get("status", "unknown"),
                                           "reasons": reasons, "tree_sha256": tree_hash}
            result["reasons"].extend(f"{task_id}:{reason}" for reason in reasons)
        # Recheck the evidence and dataset snapshots after validation to prevent
        # a concurrent writer from silently moving the inputs under this gate.
        if (hashlib.sha256(path.read_bytes()).hexdigest() != result["report_sha256"]
                or sha256_file(data_path) != identity["sha256"]):
            result["reasons"].append("independent_tree_inputs_changed_during_gate")
        result["valid"] = not result["reasons"]
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        result["reasons"].append(f"independent_tree_certificate_unavailable_or_invalid:{type(exc).__name__}:{exc}")
    return result


def require_saved_tree_certificate(records: list[dict[str, Any]], *, data_path: Path,
                                   tree_dir: Path, report_path: Path | None = None) -> dict[str, Any]:
    checked = validate_saved_tree_certificate(records, data_path=data_path, tree_dir=tree_dir,
                                               report_path=report_path)
    if not checked["valid"]:
        raise ValueError("Independent shortest-tree verification required: " + "; ".join(checked["reasons"]))
    return checked


def preflight_tree_records(records: list[dict[str, Any]], directory: str | Path | None) -> None:
    """Protect standalone evaluators as well as the parallel cloud scheduler."""
    selected = [record for record in records if record.get("shortest_proof")]
    if not selected:
        return
    if directory is None:
        raise ValueError("Certified shortest tasks require solution_tree_dir and independent evidence")
    tree_dir = Path(directory)
    family = next((f for f, folder in TREE_FOLDERS.items() if tree_dir.name == folder), None)
    if family is None:
        raise ValueError("Unknown shortest-tree family directory")
    data_path = tree_dir.parent.parent / FAMILY_FOLDERS[family] / "tasks.jsonl"
    require_saved_tree_certificate(selected, data_path=data_path, tree_dir=tree_dir)
