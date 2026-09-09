"""Build same-position C2 rule-counterfactual scenarios from audited candidates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from minibench.datasets.xiangqi.schema import RULESETS, board_to_fen


PUBLIC_RULES = {
    "standard": [],
    "horse-no-leg-block": [
        {"kind": "move-modification", "piece": "horse", "effect": "ignore-leg-block"}
    ],
    "chariot-no-center": [
        {"kind": "zone-restriction", "piece": "chariot", "effect": "forbid-center-files"}
    ],
    "soldier-free-retreat": [
        {"kind": "move-modification", "piece": "soldier", "effect": "allow-backward-after-river"}
    ],
}


def build(candidates: dict[str, list[dict]]) -> list[dict]:
    records: list[dict] = []
    sequence = 0
    scenario_number = 0
    for focus in ("horse-no-leg-block", "chariot-no-center", "soldier-free-retreat"):
        items = candidates.get(focus, [])
        if len(items) != 20:
            raise ValueError(f"{focus}: expected 20 audited candidates, got {len(items)}")
        for item in items:
            scenario_number += 1
            scenario_id = f"xiangqi-rule-scenario-c2-{scenario_number:03d}"
            fen = board_to_fen(item["board"], active_color="red")
            piece_count = sum(value != 0 for row in item["board"] for value in row)
            for ruleset in RULESETS:
                sequence += 1
                analysis = item["analysis"][ruleset]
                records.append({
                    "schema_version": 2,
                    "id": f"xiangqi-rule-variants-c2-{sequence:04d}",
                    "family": "xiangqi-rule-variants",
                    "fen": fen,
                    "agent_color": "red",
                    "goal": "best-move-under-rule",
                    "max_plies": 1,
                    "difficulty": "counterfactual",
                    "piece_count": piece_count,
                    "oracle": {
                        "best_move_uci": analysis["best"],
                        "mate_in_plies": None,
                        "evaluation_cp": None,
                    },
                    "tags": sorted({"c2", f"focus:{focus}", f"ruleset:{ruleset}"}),
                    "ruleset": ruleset,
                    "rules": PUBLIC_RULES[ruleset],
                    "scenario_id": scenario_id,
                    "design_stratum": focus,
                    "c2_analysis": {
                        "version": "c2-same-fen-counterfactual-v1",
                        "oracle_depth": 3,
                        "legal_move_count": analysis["legal"],
                        "best_value": analysis["score"],
                        "unique_best_margin": analysis["margin"],
                        "standard_best_uci": item["analysis"]["standard"]["best"],
                        "focus_ruleset": focus,
                        "focus_changes_best_move": (
                            item["analysis"]["standard"]["best"]
                            != item["analysis"][focus]["best"]
                        ),
                    },
                })
    controls = candidates.get("standard-controls", [])
    if len(controls) != 10:
        raise ValueError(f"standard-controls: expected 10, got {len(controls)}")
    for item in controls:
        sequence += 1
        scenario_number += 1
        analysis = item["analysis"]
        fen = board_to_fen(item["board"], active_color="red")
        piece_count = sum(value != 0 for row in item["board"] for value in row)
        records.append({
            "schema_version": 2,
            "id": f"xiangqi-rule-variants-c2-{sequence:04d}",
            "family": "xiangqi-rule-variants",
            "fen": fen,
            "agent_color": "red",
            "goal": "best-move-under-rule",
            "max_plies": 1,
            "difficulty": "control",
            "piece_count": piece_count,
            "oracle": {
                "best_move_uci": analysis["best"],
                "mate_in_plies": None,
                "evaluation_cp": None,
            },
            "tags": ["c2", "control", "ruleset:standard"],
            "ruleset": "standard",
            "rules": [],
            "scenario_id": f"xiangqi-rule-scenario-control-c2-{scenario_number:03d}",
            "design_stratum": "standard-control",
            "c2_analysis": {
                "version": "c2-same-fen-counterfactual-v1",
                "oracle_depth": 3,
                "legal_move_count": analysis["legal"],
                "best_value": analysis["score"],
                "unique_best_margin": analysis["margin"],
                "standard_best_uci": analysis["best"],
                "focus_ruleset": "standard",
                "focus_changes_best_move": False,
            },
        })
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidates = json.loads(args.candidates.read_text(encoding="utf-8"))
    records = build(candidates)
    args.output.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    print(json.dumps({
        "records": len(records),
        "scenarios": len({record["scenario_id"] for record in records}),
        "unique_fens": len({record["fen"] for record in records}),
        "same_fen_complete": all(
            len({record["fen"] for record in records if record["scenario_id"] == scenario}) == 1
            for scenario in {record["scenario_id"] for record in records}
        ),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
