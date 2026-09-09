"""Refresh the legacy-ID mapping and integrity digest after a corpus revision."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/xiangqi/migration_v1_to_v2.json"
DATASETS = {
    "d3-": ROOT / "data/xiangqi/mate_in_one/tasks.jsonl",
    "h2-": ROOT / "data/xiangqi/history/tasks.jsonl",
    "c2-": ROOT / "data/xiangqi/rule_variants/tasks.jsonl",
    "vis-": ROOT / "data/xiangqi/multimodal/tasks.jsonl",
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    all_records = [record for path in DATASETS.values() for record in read_jsonl(path)]
    by_id = {record["id"]: record for record in all_records}
    for prefix, path in DATASETS.items():
        old_ids = sorted(key for key in manifest["task_ids"] if key.startswith(prefix))
        new_ids = sorted(record["id"] for record in read_jsonl(path))
        if len(old_ids) != len(new_ids):
            raise ValueError(f"{prefix}: legacy/current counts differ")
        for old_id, new_id in zip(old_ids, new_ids):
            manifest["task_ids"][old_id] = new_id
    old_scenarios = sorted(manifest["scenario_ids"])
    new_scenarios = sorted({
        record["scenario_id"]
        for record in all_records if "scenario_id" in record
    })
    if len(old_scenarios) != len(new_scenarios):
        raise ValueError("legacy/current C2 scenario counts differ")
    for old_id, new_id in zip(old_scenarios, new_scenarios):
        manifest["scenario_ids"][old_id] = new_id
    projections = []
    for old_id, new_id in sorted(manifest["task_ids"].items()):
        record = by_id[new_id]
        projection = {
            "old_id": old_id,
            "new_id": new_id,
            **{key: record[key] for key in (
                "fen", "agent_color", "goal", "max_plies", "difficulty",
                "piece_count", "oracle",
            )},
        }
        for key in ("scenario_id", "ruleset"):
            if key in record:
                projection[key] = record[key]
        projections.append(projection)
    serialized = json.dumps(
        projections, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    manifest["semantic_sha256"] = sha256(serialized).hexdigest()
    manifest["dataset_revision"] = "xiangqi-corpus-rebuild-2026-08-31"
    manifest["semantic_preservation_from_previous_revision"] = False
    MANIFEST.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"tasks": len(manifest["task_ids"]), "scenarios": len(new_scenarios), "sha256": manifest["semantic_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
