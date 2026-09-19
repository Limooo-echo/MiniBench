"""Dataset provenance and exact-position leakage audit for Xiangqi releases."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
import hashlib
from pathlib import Path
from typing import Any

from minibench.datasets.xiangqi.schema import FAMILY_PATHS, load_records

REQUIRED_EXTERNAL_PROVENANCE = (
    "source_repository", "source_revision", "source_license", "generation_command",
    "generator_seed", "model_service_snapshot",
)


def audit_xiangqi_release(manifest_path: str | Path) -> dict[str, Any]:
    manifest_file = Path(manifest_path)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    repository_root = Path(__file__).resolve().parents[4]
    records_by_family = {
        family: load_records(repository_root / path, expected_family=family)
        for family, path in FAMILY_PATHS.items()
    }
    fen_families: dict[str, set[str]] = defaultdict(set)
    per_family: dict[str, Any] = {}
    unexpected_duplicate_total = 0
    for family, records in records_by_family.items():
        counts = Counter(record["fen"] for record in records)
        duplicates = sorted(fen for fen, count in counts.items() if count > 1)
        expected_duplicates: list[str] = []
        unexpected_duplicates = duplicates
        if family == "xiangqi-rule-variants":
            by_fen: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for record in records:
                by_fen[record["fen"]].append(record)
            expected_duplicates = sorted(
                fen for fen, group in by_fen.items()
                if len(group) == 4
                and len({item.get("scenario_id") for item in group}) == 1
                and {item.get("ruleset") for item in group}
                == {"standard", "horse-no-leg-block", "chariot-no-center", "soldier-free-retreat"}
            )
            unexpected_duplicates = sorted(set(duplicates) - set(expected_duplicates))
        unexpected_duplicate_total += len(unexpected_duplicates)
        per_family[family] = {
            "record_count": len(records),
            "unique_fen_count": len(counts),
            "expected_paired_duplicate_fens": expected_duplicates,
            "unexpected_duplicate_fens_within_family": unexpected_duplicates,
        }
        for fen in counts:
            fen_families[fen].add(family)
    cross_family = {
        fen: sorted(families) for fen, families in fen_families.items() if len(families) > 1
    }
    def has_unresolved(value: Any) -> bool:
        if value is None or value == "":
            return True
        if isinstance(value, str):
            return value.startswith("UNRESOLVED")
        if isinstance(value, dict):
            return any(has_unresolved(item) for item in value.values())
        if isinstance(value, list):
            return any(has_unresolved(item) for item in value)
        return False

    unresolved = [
        field for field in REQUIRED_EXTERNAL_PROVENANCE
        if field not in manifest or has_unresolved(manifest[field])
    ]
    expected_cross_family = {
        fen: families for fen, families in cross_family.items()
        if set(families) == {"xiangqi-mate-in-one", "xiangqi-multimodal"}
    }
    unexpected_cross_family = {
        fen: families for fen, families in cross_family.items()
        if fen not in expected_cross_family
    }
    gate_path = manifest_file.parent / "independent_validation.json"
    gate_problems = []
    if gate_path.is_file():
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        if not (gate.get("release_ready") and gate.get("valid") and gate.get("full")):
            gate_problems.append("independent_validation_not_full_or_failed")
        if gate.get("release_id") != manifest.get("release_id"):
            gate_problems.append("independent_validation_release_mismatch")
        for family, path in FAMILY_PATHS.items():
            digest = hashlib.sha256((repository_root / path).read_bytes()).hexdigest()
            if gate.get("inputs", {}).get(family, {}).get("sha256") != digest:
                gate_problems.append(f"independent_validation_dataset_hash_mismatch:{family}")
        for relative, recorded in gate.get("frozen_samples", {}).get("inputs", {}).items():
            sample_path = manifest_file.parent / "evaluation_samples" / relative
            if not sample_path.is_file() or hashlib.sha256(sample_path.read_bytes()).hexdigest() != recorded.get("sha256"):
                gate_problems.append(f"independent_validation_sample_hash_mismatch:{relative}")
        if len(gate.get("frozen_samples", {}).get("inputs", {})) != 9:
            gate_problems.append("independent_validation_frozen_inputs_incomplete")
        source_inputs = gate.get("c2_sources", {}).get("inputs", {})
        for relative, recorded in source_inputs.items():
            source_path = manifest_file.parent / "sources" / "ccpd" / relative
            if not source_path.is_file() or hashlib.sha256(source_path.read_bytes()).hexdigest() != recorded.get("sha256"):
                gate_problems.append(f"independent_validation_source_hash_mismatch:{relative}")
        if not gate.get("c2_sources", {}).get("valid") or not source_inputs:
            gate_problems.append("independent_validation_source_inputs_incomplete")
        for relative, digest in gate.get("validator_hashes", {}).items():
            code_path = repository_root / relative
            if not code_path.is_file() or hashlib.sha256(code_path.read_bytes()).hexdigest() != digest:
                gate_problems.append(f"independent_validation_code_hash_mismatch:{relative}")
        if not gate.get("validator_hashes"):
            gate_problems.append("independent_validation_code_hash_missing")
    else:
        gate_problems.append("independent_validation_missing")
    technical_ready = not gate_problems and unexpected_duplicate_total == 0 and not unexpected_cross_family
    return {
        "manifest": str(manifest_file),
        "per_family": per_family,
        "cross_family_exact_fen_overlap_count": len(cross_family),
        "cross_family_exact_fen_overlaps": cross_family,
        "expected_d3_m2_overlap_count": len(expected_cross_family),
        "unexpected_cross_family_overlap_count": len(unexpected_cross_family),
        "unexpected_cross_family_overlaps": unexpected_cross_family,
        "unresolved_external_provenance": unresolved,
        "independent_validation_report": str(gate_path),
        "independent_validation_problems": gate_problems,
        "technical_ready": technical_ready,
        "release_ready": not unresolved and technical_ready,
        "limitations": [
            "Exact-FEN auditing cannot prove absence of near-duplicate positions.",
            "Training-data contamination cannot be established from local files alone; document source and model-provider policy.",
        ],
    }
