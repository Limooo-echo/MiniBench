"""Dataset provenance and exact-position leakage audit for Xiangqi releases."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
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
    records_by_family = {
        family: load_records(path, expected_family=family)
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
    return {
        "manifest": str(manifest_file),
        "per_family": per_family,
        "cross_family_exact_fen_overlap_count": len(cross_family),
        "cross_family_exact_fen_overlaps": cross_family,
        "expected_d3_m2_overlap_count": len(expected_cross_family),
        "unexpected_cross_family_overlap_count": len(unexpected_cross_family),
        "unexpected_cross_family_overlaps": unexpected_cross_family,
        "unresolved_external_provenance": unresolved,
        "release_ready": not unresolved and unexpected_duplicate_total == 0
        and not unexpected_cross_family,
        "limitations": [
            "Exact-FEN auditing cannot prove absence of near-duplicate positions.",
            "Training-data contamination cannot be established from local files alone; document source and model-provider policy.",
        ],
    }
