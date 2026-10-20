"""Build the paired multi-ply C2 corpus from independently verified H2 mates.

Every scenario contains the same position twice: standard rules and one focus
rule.  The complete stored line is replayed by both the production and the
independent rule implementations and must end in strict checkmate in both rows.
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
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from minibench.datasets.xiangqi import reference
from minibench.datasets.xiangqi.schema import (
    FAMILY_PATHS,
    RULESETS,
    fen_to_board,
    internal_rules,
    load_records,
    validate_record,
)
from minibench.datasets.xiangqi.variants.board import VariantBoard
from minibench.datasets.xiangqi.variants.mate_search import LOSS_UNIT, replay_strict_mate_line
from minibench.datasets.xiangqi.variants.rules import Rule


FOCI = tuple(ruleset for ruleset in RULESETS if ruleset != "standard")
TARGETS = {
    "horse-no-leg-block": 42,
    "chariot-no-center": 42,
    "soldier-free-retreat": 41,
}
SEARCH_DEPTH = 3
HORIZON_EXTENSION = 4


def _public_rules(ruleset: str) -> list[dict[str, str]]:
    return {
        "standard": [],
        "horse-no-leg-block": [
            {
                "kind": "move-modification",
                "piece": "horse",
                "effect": "ignore-leg-block",
            }
        ],
        "chariot-no-center": [
            {
                "kind": "zone-restriction",
                "piece": "chariot",
                "effect": "forbid-center-files",
            }
        ],
        "soldier-free-retreat": [
            {
                "kind": "move-modification",
                "piece": "soldier",
                "effect": "allow-backward-after-river",
            }
        ],
    }[ruleset]


def _independent_replay(
    board: list[list[int]],
    side: int,
    pv: list[str],
    ruleset: str,
) -> dict[str, Any]:
    state = [row[:] for row in board]
    turn = side
    rules = internal_rules({"ruleset": ruleset})
    for index, uci in enumerate(pv):
        legal = {move.to_uci(): move for move in reference.legal_moves(state, turn, rules)}
        move = legal.get(uci)
        if move is None:
            return {
                "valid": False,
                "terminal": "ongoing",
                "plies": index,
                "reason": f"illegal_ply_{index}:{uci}",
            }
        state = reference.apply_move(state, move)
        turn = -turn
    terminal = reference.terminal_status(state, turn, rules)
    return {
        "valid": terminal == "checkmate" and turn == -side,
        "terminal": terminal,
        "plies": len(pv),
        "reason": "strict_checkmate" if terminal == "checkmate" else "line_did_not_mate",
    }


def _production_replay(
    board: list[list[int]],
    side: int,
    pv: list[str],
    ruleset: str,
) -> dict[str, Any]:
    variant = VariantBoard(
        board,
        [Rule.from_dict(rule) for rule in internal_rules({"ruleset": ruleset})],
    )
    return replay_strict_mate_line(
        variant,
        side,
        pv,
        attacker_side=side,
    )


def _rule_sensitive_along_line(
    board: list[list[int]],
    side: int,
    pv: list[str],
    focus: str,
) -> bool:
    standard = VariantBoard(board, [])
    variant = VariantBoard(
        board,
        [Rule.from_dict(rule) for rule in internal_rules({"ruleset": focus})],
    )
    turn = side
    changed = False
    for uci in pv:
        standard_moves = {move.to_uci(): move for move in standard.legal_moves(turn)}
        variant_moves = {move.to_uci(): move for move in variant.legal_moves(turn)}
        changed = changed or set(standard_moves) != set(variant_moves)
        if uci not in standard_moves or uci not in variant_moves:
            return False
        standard.apply(standard_moves[uci])
        variant.apply(variant_moves[uci])
        turn = -turn
    return changed


def _candidate(source: dict[str, Any], focus: str) -> dict[str, Any] | None:
    analysis = source["h2_analysis"]
    pv = list(analysis["principal_variation_uci"])
    mate_moves = int(analysis["mate_in_moves"])
    if mate_moves < 2 or len(pv) != 2 * mate_moves - 1:
        return None
    board, color = fen_to_board(source["fen"])
    side = 1 if color == "red" else -1
    focus_rules = internal_rules({"ruleset": focus})
    focus_board = VariantBoard(
        board,
        [Rule.from_dict(rule) for rule in focus_rules],
    )
    # Relaxed rules can turn a multi-move standard puzzle into mate-in-one.
    for move in focus_board.legal_moves(side):
        trial = focus_board.copy()
        trial.apply(move)
        if (trial.find_general(-side) is not None and trial._is_in_check(-side)
                and not trial.legal_moves(-side)):
            return None
    # A rule change can retroactively expose a horse attack. Such a position is
    # not a legal start state for the side declared to move and must be retired.
    if focus_board._is_in_check(-side) or reference.in_check(board, -side, focus_rules):
        return None
    verification: dict[str, dict[str, Any]] = {}
    for ruleset in ("standard", focus):
        production = _production_replay(board, side, pv, ruleset)
        independent = _independent_replay(board, side, pv, ruleset)
        if not production["valid"] or not independent["valid"]:
            return None
        if production["terminal"] != independent["terminal"] or production["plies"] != independent["plies"]:
            return None
        verification[ruleset] = {
            "production": production,
            "independent": independent,
        }
    return {
        "source": source,
        "focus": focus,
        "mate_moves": mate_moves,
        "pv": pv,
        "verification": verification,
        "sensitive": _rule_sensitive_along_line(board, side, pv, focus),
    }


def _select_sources(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    source_ids = [str(row.get("id", "")) for row in rows]
    duplicate_ids = sorted(
        source_id
        for source_id, count in Counter(source_ids).items()
        if source_id and count > 1
    )
    if duplicate_ids:
        raise ValueError(
            "duplicate H2 source ids are not allowed: " + ", ".join(duplicate_ids[:5])
        )
    fens = [str(row.get("fen", "")) for row in rows]
    duplicate_fens = sum(count > 1 for count in Counter(fens).values() if count)
    if duplicate_fens:
        raise ValueError(
            f"duplicate H2 source positions are not allowed: {duplicate_fens} duplicate FEN group(s)"
        )
    pools: dict[str, list[dict[str, Any]]] = {focus: [] for focus in FOCI}
    for source in rows:
        for focus in FOCI:
            item = _candidate(source, focus)
            if item is not None:
                pools[focus].append(item)
    for focus in FOCI:
        pools[focus].sort(
            key=lambda item: (
                not item["sensitive"],
                item["mate_moves"] > 5,
                item["mate_moves"],
                item["source"]["piece_count"],
                item["source"]["id"],
            )
        )

    selected: list[dict[str, Any]] = []
    used_sources: set[str] = set()
    # The chariot restriction has the smallest valid pool, so allocate it first.
    for focus in ("chariot-no-center", "horse-no-leg-block", "soldier-free-retreat"):
        choices = [
            item for item in pools[focus] if item["source"]["id"] not in used_sources
        ]
        required = TARGETS[focus]
        if len(choices) < required:
            raise ValueError(
                f"insufficient independently replayed {focus} candidates: "
                f"{len(choices)} < {required}"
            )
        chosen = choices[:required]
        selected.extend(chosen)
        used_sources.update(item["source"]["id"] for item in chosen)
    return selected


def _record(
    item: dict[str, Any],
    *,
    scenario_number: int,
    ruleset: str,
) -> dict[str, Any]:
    source = item["source"]
    focus = item["focus"]
    digest = hashlib.sha256(
        f"{source['id']}|{focus}".encode("utf-8")
    ).hexdigest()[:10]
    scenario_id = f"xiangqi-rule-scenario-c2-r2-{scenario_number:03d}-{digest}"
    suffix = "standard" if ruleset == "standard" else ruleset
    task_id = f"xiangqi-rule-variants-c2-r2-{scenario_number:03d}-{suffix}-{digest}"
    mate_moves = item["mate_moves"]
    reference_plies = len(item["pv"])
    tags = sorted(
        {
            "c2",
            "source:h2-verified-mate",
            f"focus:{focus}",
            f"ruleset:{ruleset}",
            f"mate-band:{source['difficulty']}",
            f"mate-in-moves:{mate_moves}",
            "rule-sensitive" if item["sensitive"] else "rule-control",
        }
    )
    record = {
        "schema_version": 2,
        "release_id": source["release_id"],
        "id": task_id,
        "family": "xiangqi-rule-variants",
        "fen": source["fen"],
        "agent_color": source["agent_color"],
        "goal": "checkmate",
        "max_plies": reference_plies + HORIZON_EXTENSION,
        "difficulty": source["difficulty"],
        "piece_count": source["piece_count"],
        "oracle": {
            "best_move_uci": item["pv"][0],
            "mate_in_plies": reference_plies,
            "evaluation_cp": None,
        },
        "tags": tags,
        "source": deepcopy(source["source"]),
        "source_id": source["source_id"],
        "source_task_id": source["id"],
        "ruleset": ruleset,
        "rules": _public_rules(ruleset),
        "scenario_id": scenario_id,
        "design_stratum": focus,
        "c2_analysis": {
            "version": "c2-multistep-rule-mate-v3",
            "source_task_id": source["id"],
            "focus_ruleset": focus,
            "pair_rulesets": ["standard", focus],
            "mate_in_moves": mate_moves,
            "reference_mate_plies": reference_plies,
            "principal_variation_uci": list(item["pv"]),
            "rule_sensitive_along_reference": bool(item["sensitive"]),
            "search_depth": SEARCH_DEPTH,
            "loss_unit": LOSS_UNIT,
            "max_plies_extension": HORIZON_EXTENSION,
            "verification": deepcopy(item["verification"][ruleset]),
            "interpretation": (
                "The complete reference line is independently replayed under the "
                "active rule and ends in strict checkmate. It is a bounded reference "
                "line, not a proof of the shortest mate against every defence."
            ),
        },
    }
    validate_record(record, expected_family="xiangqi-rule-variants")
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--history",
        type=Path,
        default=ROOT / FAMILY_PATHS["xiangqi-history"],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / FAMILY_PATHS["xiangqi-rule-variants"],
    )
    args = parser.parse_args()
    from minibench.datasets.xiangqi.provenance import require_candidate_outputs
    require_candidate_outputs(args.output)
    rows = load_records(args.history, expected_family="xiangqi-history")
    selected = _select_sources(rows)
    records: list[dict[str, Any]] = []
    for number, item in enumerate(selected, start=1):
        records.append(_record(item, scenario_number=number, ruleset="standard"))
        records.append(_record(item, scenario_number=number, ruleset=item["focus"]))
    records.sort(key=lambda record: record["id"])
    if len(records) != 250 or len({record["id"] for record in records}) != 250:
        raise ValueError("C2 build must produce exactly 250 unique records")
    if not all(record["oracle"]["mate_in_plies"] >= 3 for record in records):
        raise ValueError("C2 must contain only multi-move mates")
    short_or_medium = sum(
        record["c2_analysis"]["mate_in_moves"] <= 5 for record in records
    )
    if short_or_medium <= len(records) // 2:
        raise ValueError("C2 requires a majority of mates within five attacker moves")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
        newline="\n",
    )
    report = {
        "version": "c2-multistep-rule-mate-v3",
        "record_count": len(records),
        "scenario_count": len(records) // 2,
        # Count selected scenarios directly.  Do not infer the focus row from
        # list parity after sorting: task-id ordering is not part of the data
        # contract and may change independently of pair membership.
        "focus_counts": dict(Counter(item["focus"] for item in selected)),
        "ruleset_counts": dict(Counter(record["ruleset"] for record in records)),
        "difficulty_counts": dict(Counter(record["difficulty"] for record in records)),
        "mate_moves_counts": dict(
            sorted(Counter(record["c2_analysis"]["mate_in_moves"] for record in records).items())
        ),
        "within_five_moves": short_or_medium,
        "within_five_moves_rate": short_or_medium / len(records),
        "rule_sensitive_records": sum(
            record["c2_analysis"]["rule_sensitive_along_reference"] for record in records
        ),
        "source_policy": "125 unique H2 sources; standard/focus pairs only",
        "verification": "production and independent full-line strict-checkmate replay",
        "loss_unit": LOSS_UNIT,
    }
    (args.output.parent / "generation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
