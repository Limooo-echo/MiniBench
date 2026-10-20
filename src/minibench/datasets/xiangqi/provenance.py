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


def require_candidate_outputs(*paths: str | Path | None) -> None:
    """Legacy pool tools may not overwrite the certified release or its reports.

    Keep their imported candidate-building functions for reproducibility, but
    stage CLI output outside data/xiangqi and publish through the tree-aware
    release workflow. Resolve symlinks as well as '..' before checking.
    """
    protected = Path(__file__).resolve().parents[4] / "data" / "xiangqi"
    for path in paths:
        if path is not None and Path(path).resolve().is_relative_to(protected.resolve()):
            raise ValueError(
                "Candidate-only tool cannot write the certified data/xiangqi release; "
                "choose an output staging directory, then use "
                "prepare_xiangqi_shortest_release.py and validate_xiangqi_data.py.")


def audit_xiangqi_release(manifest_path: str | Path, *, validation_report: str | Path | None = None) -> dict[str, Any]:
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
                if len(group) == 2
                and len({item.get("scenario_id") for item in group}) == 1
                and len({item.get("design_stratum") for item in group}) == 1
                and {item.get("ruleset") for item in group}
                == {"standard", group[0].get("design_stratum")}
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
    expected_d3_m2 = {
        fen: families for fen, families in cross_family.items()
        if set(families) == {"xiangqi-mate-in-one", "xiangqi-multimodal"}
    }
    expected_h2_c2 = {
        fen: families for fen, families in cross_family.items()
        if set(families) == {"xiangqi-history", "xiangqi-rule-variants"}
    }
    expected_cross_family = {**expected_d3_m2, **expected_h2_c2}
    unexpected_cross_family = {
        fen: families for fen, families in cross_family.items()
        if fen not in expected_cross_family
    }
    gate_path = Path(validation_report) if validation_report else manifest_file.parent / "independent_validation.json"
    gate_problems = []
    for rows in records_by_family.values():
        for record in rows:
            if "shortest_proof" not in record:
                continue
            proof=record["shortest_proof"]
            tree_path=repository_root/proof["tree_path"]
            if (not tree_path.is_file() or
                    hashlib.sha256(tree_path.read_bytes()).hexdigest()!=proof["tree_sha256"]):
                gate_problems.append(f"shortest_tree_missing_or_changed:{record['id']}")
    submitted = manifest.get("submission")
    if submitted:
        selection_path = manifest_file.parent / "submission_manifest.json"
        try:
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
            if hashlib.sha256(selection_path.read_bytes()).hexdigest() != submitted.get("sha256"):
                gate_problems.append("submitted_selection_manifest_hash_mismatch")
            keys = {"xiangqi-mate-in-one": "d3", "xiangqi-history": "h2",
                    "xiangqi-rule-variants": "c2", "xiangqi-multimodal": "m2"}
            for family, rows in records_by_family.items():
                entry = selection["datasets"][keys[family]]
                actual = hashlib.sha256((repository_root / FAMILY_PATHS[family]).read_bytes()).hexdigest()
                if (len(rows) != 100 or entry["sha256"] != actual
                        or entry["task_ids"] != [r["id"] for r in rows]):
                    gate_problems.append(f"submitted_selection_changed:{family}")
        except (OSError, ValueError, KeyError, TypeError):
            gate_problems.append("submitted_selection_manifest_missing_or_invalid")
    runtime_policy_problems = []
    gate: dict[str, Any] = {}
    if gate_path.is_file():
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        if not (gate.get("static_data_ready") and gate.get("valid") and gate.get("full")):
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
            # C2 v3 derives directly from the canonical H2 dataset.  Resolve
            # only the gate-owned relative key under data/xiangqi; never follow
            # the absolute path recorded for human audit evidence.
            source_path = manifest_file.parent / relative
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
        if not gate.get("runtime_policy_certified"):
            runtime_policy_problems.append("c2_runtime_policy_not_certified")
        else:
            cert = gate.get("c2_runtime_certificates", {})
            c2_rows = records_by_family.get("xiangqi-rule-variants", [])
            if c2_rows and all("shortest_proof" in r for r in c2_rows):
                if not cert.get("valid"):
                    runtime_policy_problems.append("c2_shortest_certification_failed")
                for record in c2_rows:
                    proof = record["shortest_proof"]
                    tree_path = repository_root / proof["tree_path"]
                    if (not tree_path.is_file() or
                            hashlib.sha256(tree_path.read_bytes()).hexdigest() != proof["tree_sha256"]):
                        runtime_policy_problems.append(f"c2_shortest_tree_missing_or_changed:{record['id']}")
            else:
                cert_path = manifest_file.parent / "rule_variants/runtime_certificates.json"
                if (not cert.get("valid") or not cert_path.is_file()
                        or hashlib.sha256(cert_path.read_bytes()).hexdigest() != cert.get("sha256")):
                    runtime_policy_problems.append("c2_runtime_certificate_missing_or_changed")
    else:
        gate_problems.append("independent_validation_missing")
        runtime_policy_problems.append("independent_validation_missing")
    technical_ready = not gate_problems and unexpected_duplicate_total == 0 and not unexpected_cross_family
    experiment_ready = technical_ready and not runtime_policy_problems
    return {
        "manifest": str(manifest_file),
        "per_family": per_family,
        "cross_family_exact_fen_overlap_count": len(cross_family),
        "cross_family_exact_fen_overlaps": cross_family,
        "expected_d3_m2_overlap_count": len(expected_d3_m2),
        "expected_h2_c2_overlap_count": len(expected_h2_c2),
        "unexpected_cross_family_overlap_count": len(unexpected_cross_family),
        "unexpected_cross_family_overlaps": unexpected_cross_family,
        "unresolved_external_provenance": unresolved,
        "independent_validation_report": str(gate_path),
        "independent_validation_problems": gate_problems,
        "runtime_policy_problems": runtime_policy_problems,
        "technical_ready": technical_ready,
        "experiment_ready": experiment_ready,
        "release_ready": not unresolved and experiment_ready,
        "limitations": [
            "Exact-FEN auditing cannot prove absence of near-duplicate positions.",
            "Training-data contamination cannot be established from local files alone; document source and model-provider policy.",
        ],
    }
