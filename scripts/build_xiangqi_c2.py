"""Build same-position C2 rule-counterfactual scenarios from audited candidates."""
from __future__ import annotations

import argparse
import json
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from minibench.datasets.xiangqi.schema import RULESETS, board_to_fen, validate_record
from generate_xiangqi_c2_candidates import analyse, independently_verify
from minibench.datasets.xiangqi.validation import validate_position, compare_standard_legal_moves
from minibench.datasets.xiangqi import reference

RELEASE_ID = "xiangqi-2026-09-19-r1"


def validate_candidate(item, focus):
    board = item["board"]
    reasons = validate_position(board, 1)
    comparison = compare_standard_legal_moves(board, 1)
    if reasons or not comparison["valid"]:
        raise ValueError(f"invalid candidate: {reasons + comparison['reasons']}")
    if not item.get("source", {}).get("file_sha256"):
        raise ValueError("missing audited PGN replay source")
    names = RULESETS if focus != "standard-controls" else ("standard",)
    for name in names:
        expected = item["analysis"][name] if focus != "standard-controls" else item["analysis"]
        actual = analyse(board, name)
        if actual is None or any(actual[k] != expected[k] for k in ("best", "score", "margin", "legal")):
            raise ValueError(f"stale or non-unique candidate answer: {name}")
    if focus != "standard-controls":
        if item["analysis"][focus]["best"] == item["analysis"]["standard"]["best"]:
            raise ValueError("focus must change the best move")
        independently_verify(item)
    else:
        scores = {m.to_uci(): v for m,v in reference.score_moves(board, 1, 3)}
        expected = item["analysis"]["scores"]
        if scores.keys() != expected.keys() or any(abs(scores[m]-expected[m]) > 1e-6 for m in scores):
            raise ValueError("independent control search disagreement")


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


def position_key(board):
    return hashlib.sha256(board_to_fen(board).encode()).hexdigest()[:10]


def build(candidates: dict[str, list[dict]]) -> list[dict]:
    if set(candidates) != {"horse-no-leg-block", "chariot-no-center", "soldier-free-retreat", "standard-controls"}:
        raise ValueError("expected 20 candidates per focus and 10 controls in exactly four groups")
    all_items = [item for group in candidates.values() for item in group]
    fens = [board_to_fen(item["board"]) for item in all_items]
    if len(fens) != len(set(fens)):
        raise ValueError("duplicate position across C2 scenarios")
    records: list[dict] = []
    sequence = 0
    scenario_number = 0
    for focus in ("horse-no-leg-block", "chariot-no-center", "soldier-free-retreat"):
        items = candidates.get(focus, [])
        if len(items) != 20:
            raise ValueError(f"{focus}: expected 20 audited candidates, got {len(items)}")
        for item in items:
            validate_candidate(item, focus)
            scenario_number += 1
            scenario_id = f"xiangqi-rule-scenario-c2-r1-{scenario_number:03d}-{position_key(item['board'])}"
            fen = board_to_fen(item["board"], active_color="red")
            piece_count = sum(value != 0 for row in item["board"] for value in row)
            for ruleset in RULESETS:
                sequence += 1
                analysis = item["analysis"][ruleset]
                records.append({
                    "schema_version": 2,
                    "release_id": RELEASE_ID,
                    "source": dict(item["source"]),
                    "source_id": "ccpd:" + item["source"]["source_file"],
                    "validation": dict(item["verification"]),
                    "id": f"xiangqi-rule-variants-c2-r1-{sequence:04d}-{position_key(item['board'])}",
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
                        "version": "c2-legal-replay-counterfactual-v2",
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
        validate_candidate(item, "standard-controls")
        sequence += 1
        scenario_number += 1
        analysis = item["analysis"]
        fen = board_to_fen(item["board"], active_color="red")
        piece_count = sum(value != 0 for row in item["board"] for value in row)
        records.append({
            "schema_version": 2,
                    "release_id": RELEASE_ID,
                    "source": dict(item["source"]),
                    "source_id": "ccpd:" + item["source"]["source_file"],
                    "validation": dict(item["verification"]),
            "id": f"xiangqi-rule-variants-c2-r1-{sequence:04d}-{position_key(item['board'])}",
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
            "scenario_id": f"xiangqi-rule-scenario-control-c2-r1-{scenario_number:03d}-{position_key(item['board'])}",
            "design_stratum": "standard-control",
            "c2_analysis": {
                "version": "c2-legal-replay-counterfactual-v2",
                "oracle_depth": 3,
                "legal_move_count": analysis["legal"],
                "best_value": analysis["score"],
                "unique_best_margin": analysis["margin"],
                "standard_best_uci": analysis["best"],
                "focus_ruleset": "standard",
                "focus_changes_best_move": False,
            },
        })
    for record in records:
        validate_record(record, expected_family="xiangqi-rule-variants")
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
