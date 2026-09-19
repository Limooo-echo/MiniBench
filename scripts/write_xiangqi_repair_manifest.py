"""Record retired/retained/changed task identities without transferring old answers."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from minibench.datasets.xiangqi.schema import fen_to_board
from minibench.datasets.xiangqi.validation import validate_position

RELEASE = "xiangqi-2026-09-19-r1"
FOLDERS = ("mate_in_one", "rule_variants", "history", "multimodal")

def read(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def identity(row):
    return {key: row.get(key) for key in ("fen", "agent_color", "goal", "max_plies", "ruleset", "rules")}

def main():
    data = ROOT / "data/xiangqi"
    archive = data / "legacy/xiangqi-corpus-rebuild-2026-08-31"
    h2 = json.loads((data / "history/validation_report.json").read_text(encoding="utf-8"))
    if h2["status"] != "complete":
        raise ValueError("H2 release is incomplete")
    failures = {item["task_id"]: item["reasons"] for item in h2["attempts"]
                if item.get("stage") == "existing" and not item["valid"]}
    exclusions = {item["task_id"]: item.get("reasons", [item.get("reason", "cross_family_fen_overlap")])
                  for item in h2.get("selection_exclusions", []) if item.get("task_id")}
    report = {"release_id": RELEASE, "previous_revision": "xiangqi-corpus-rebuild-2026-08-31",
              "previous_git_commit": "eab452b069b40a1cfaec39fe2b3895d7503484f2",
              "policy": "Old answers remain attached only to archived old tasks. No positional ID mapping to replacement boards.",
              "families": {}}
    for folder in FOLDERS:
        before, after = archive / folder / "tasks.jsonl", data / folder / "tasks.jsonl"
        old, new = read(before), read(after)
        if len(new) != 250 or any(row.get("release_id") != RELEASE for row in new):
            raise ValueError(f"incomplete release: {folder}")
        old_by_id, new_by_id = {r["id"]: r for r in old}, {r["id"]: r for r in new}
        retained, retired, added = [], [], []
        replaced = {r["replaces_id"]: r for r in new if r.get("replaces_id")}
        for row in old:
            task_id = row["id"]
            if task_id in new_by_id:
                if identity(row) != identity(new_by_id[task_id]):
                    raise ValueError(f"changed target reuses old ID: {task_id}")
                retained.append({"id": task_id, "reason": "same_board_and_target_revalidated"})
                continue
            reasons = []
            if folder == "rule_variants":
                board, side = fen_to_board(row["fen"])
                reasons = list(validate_position(board, 1 if side == "red" else -1))
                reasons.append("replaced_legacy_candidate_pool_with_audited_legal_pgn_replay")
            elif folder == "history":
                reasons = failures.get(task_id, []) + exclusions.get(task_id, [])
                if task_id in replaced:
                    reasons = list(reasons) + ["reference_horizon_or_target_changed_new_identity_required"]
                if not reasons:
                    reasons = ["not_selected_under_fixed_strata_and_unique_source_requirements"]
            retired.append({"id": task_id, "identity": identity(row), "reasons": reasons})
        for row in new:
            if row["id"] not in old_by_id:
                added.append({"id": row["id"], "identity": identity(row),
                              "source_id": row.get("source_id"), "replaces_id": row.get("replaces_id")})
        report["families"][folder] = {"old_sha256": sha(before), "new_sha256": sha(after),
            "old_count": len(old), "new_count": len(new), "retained_count": len(retained),
            "retired_count": len(retired), "added_count": len(added),
            "retained": retained, "retired": retired, "added": added}
    destination = data / "repair_manifest.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({key: {k: v for k, v in family.items() if k.endswith("count")}
                      for key, family in report["families"].items()}))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
