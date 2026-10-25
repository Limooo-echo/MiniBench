from __future__ import annotations

import json
import math
from pathlib import Path
import random
import re
from typing import Any, Iterable, Sequence


SCHEMA_VERSION = 2
XIANGQI_FAMILIES = (
    "xiangqi-mate-in-one",
    "xiangqi-rule-variants",
    "xiangqi-history",
    "xiangqi-multimodal",
)
FAMILY_PATHS = {
    "xiangqi-mate-in-one": Path("data/xiangqi/mate_in_one/tasks.jsonl"),
    "xiangqi-rule-variants": Path("data/xiangqi/rule_variants/tasks.jsonl"),
    "xiangqi-history": Path("data/xiangqi/history/tasks.jsonl"),
    "xiangqi-multimodal": Path("data/xiangqi/multimodal/tasks.jsonl"),
}
LEGACY_FAMILY_NAMES = {
    "D3": "xiangqi-mate-in-one",
    "C2": "xiangqi-rule-variants",
    "H2": "xiangqi-history",
    "M2": "xiangqi-multimodal",
}
RULESETS = (
    "standard",
    "horse-no-leg-block",
    "chariot-no-center",
    "soldier-free-retreat",
)
HISTORY_MODES = ("paired", "full-state", "move-history-only")
MULTIMODAL_INPUT_MODES = (
    "text",
    "chinese-piece-image",
    "latin-piece-image",
)

_FEN_POOLS = {
    "K": (1,),
    "A": (2, 3),
    "B": (4, 5),
    "N": (6, 7),
    "R": (8, 9),
    "C": (10, 11),
    "P": (12, 13, 14, 15, 16),
}
_PIECE_TO_FEN = {
    1: "K", 2: "A", 3: "A", 4: "B", 5: "B", 6: "N", 7: "N",
    8: "R", 9: "R", 10: "C", 11: "C", 12: "P", 13: "P",
    14: "P", 15: "P", 16: "P",
}


def fen_to_board(fen: str) -> tuple[list[list[int]], str]:
    parts = fen.strip().split()
    if len(parts) != 6:
        raise ValueError("FEN must have six fields")
    active = parts[1]
    if active not in {"w", "b"}:
        raise ValueError(f"unsupported FEN active color: {active}")
    red_pools = {key: list(values) for key, values in _FEN_POOLS.items()}
    black_pools = {key: list(values) for key, values in _FEN_POOLS.items()}
    board: list[list[int]] = []
    for encoded_row in parts[0].split("/"):
        row: list[int] = []
        for char in encoded_row:
            if char.isdigit():
                row.extend([0] * int(char))
                continue
            piece = char.upper()
            pools = red_pools if char.isupper() else black_pools
            if piece not in pools or not pools[piece]:
                raise ValueError(f"invalid or excessive FEN piece {char!r}")
            piece_id = pools[piece].pop(0)
            row.append(piece_id if char.isupper() else -piece_id)
        if len(row) != 9:
            raise ValueError(f"FEN row must contain 9 files: {encoded_row!r}")
        board.append(row)
    if len(board) != 10:
        raise ValueError("FEN board must contain 10 ranks")
    if sum(value == 1 for row in board for value in row) != 1:
        raise ValueError("FEN must contain exactly one red general")
    if sum(value == -1 for row in board for value in row) != 1:
        raise ValueError("FEN must contain exactly one black general")
    return board, "red" if active == "w" else "black"


def board_to_fen(
    board: Sequence[Sequence[int]], *, active_color: str = "red"
) -> str:
    if active_color not in {"red", "black"}:
        raise ValueError("active_color must be red or black")
    if len(board) != 10 or any(len(row) != 9 for row in board):
        raise ValueError("Xiangqi board must be 10x9")
    encoded_rows: list[str] = []
    for row in board:
        pieces: list[str] = []
        empty = 0
        for raw_value in row:
            value = int(raw_value)
            if value == 0:
                empty += 1
                continue
            if empty:
                pieces.append(str(empty))
                empty = 0
            symbol = _PIECE_TO_FEN.get(abs(value))
            if symbol is None:
                raise ValueError(f"invalid Xiangqi piece id: {value}")
            pieces.append(symbol if value > 0 else symbol.lower())
        if empty:
            pieces.append(str(empty))
        encoded_rows.append("".join(pieces))
    active = "w" if active_color == "red" else "b"
    return f"{'/'.join(encoded_rows)} {active} - - 0 1"


def normalize_tags(tags: Iterable[Any]) -> list[str]:
    return sorted({tag.strip() for tag in tags if isinstance(tag, str) and tag.strip()})


def validate_record(record: dict[str, Any], *, expected_family: str | None = None) -> dict[str, Any]:
    task_id = record.get("id")
    family = record.get("family")
    if record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{task_id or '<record>'}: schema_version must be 2")
    if family not in XIANGQI_FAMILIES:
        raise ValueError(f"{task_id}: unsupported Xiangqi family {family!r}")
    if expected_family is not None and family != expected_family:
        raise ValueError(f"{task_id}: expected family {expected_family}, got {family}")
    if (not isinstance(task_id, str) or not task_id.startswith(f"{family}-")
            or re.fullmatch(r"[A-Za-z0-9_-]+",task_id) is None):
        raise ValueError(f"{task_id!r}: id must begin with {family}-")
    fen = record.get("fen")
    if not isinstance(fen, str):
        raise ValueError(f"{task_id}: fen must be a string")
    board, side_to_move = fen_to_board(fen)
    if board_to_fen(board, active_color=side_to_move) != fen:
        raise ValueError(f"{task_id}: FEN is not in canonical MiniBench form")
    if record.get("agent_color") != side_to_move:
        raise ValueError(f"{task_id}: FEN active color and agent_color disagree")
    from minibench.datasets.xiangqi.validation import validate_position

    position_errors = validate_position(board, 1 if side_to_move == "red" else -1)
    if position_errors:
        raise ValueError(f"{task_id}: invalid initial position: {'; '.join(position_errors)}")
    expected_goal = "checkmate"
    if record.get("goal") != expected_goal:
        raise ValueError(f"{task_id}: goal must be {expected_goal}")
    max_plies = record.get("max_plies")
    if not isinstance(max_plies, int) or max_plies < 1:
        raise ValueError(f"{task_id}: max_plies must be a positive integer")
    difficulty = record.get("difficulty")
    if not isinstance(difficulty, str) or not difficulty:
        raise ValueError(f"{task_id}: difficulty must be a non-empty string")
    piece_count = sum(value != 0 for row in board for value in row)
    if record.get("piece_count") != piece_count:
        raise ValueError(f"{task_id}: piece_count does not match FEN")
    tags = record.get("tags")
    if not isinstance(tags, list) or tags != normalize_tags(tags):
        raise ValueError(f"{task_id}: tags must be sorted, unique non-empty strings")
    oracle = record.get("oracle")
    if not isinstance(oracle, dict):
        raise ValueError(f"{task_id}: oracle must be an object")
    if set(oracle) != {"best_move_uci", "mate_in_plies", "evaluation_cp"}:
        raise ValueError(f"{task_id}: oracle has missing or unsupported fields")
    best_move = oracle["best_move_uci"]
    if not isinstance(best_move, str) or re.fullmatch(
        r"[a-i][0-9][a-i][0-9]", best_move
    ) is None:
        raise ValueError(f"{task_id}: oracle.best_move_uci is invalid")
    mate_in_plies = oracle["mate_in_plies"]
    if mate_in_plies is not None and (
        not isinstance(mate_in_plies, int) or mate_in_plies < 1
    ):
        raise ValueError(f"{task_id}: oracle.mate_in_plies is invalid")
    evaluation_cp = oracle["evaluation_cp"]
    if evaluation_cp is not None and not isinstance(evaluation_cp, (int, float)):
        raise ValueError(f"{task_id}: oracle.evaluation_cp is invalid")
    if family == "xiangqi-mate-in-one":
        _validate_mate_in_one_record(record)
    elif family == "xiangqi-rule-variants":
        from minibench.datasets.xiangqi.reference import in_check

        side = 1 if side_to_move == "red" else -1
        if in_check(board, -side, internal_rules(record)):
            raise ValueError(f"{task_id}: active rules leave side not to move in check")
        _validate_rule_record(record)
    elif family == "xiangqi-history":
        _validate_history_record(record)
    elif family == "xiangqi-multimodal":
        _validate_multimodal_record(record)
    if "shortest_mate_moves" in record or "shortest_proof" in record:
        _validate_shortest_metadata(record)
    derivation=record.get("source",{}).get("derivation")
    if derivation:
        parent_fen=derivation.get("parent_fen")
        moves=derivation.get("moves_uci")
        if (not isinstance(parent_fen,str) or not isinstance(moves,list) or not moves
                or not str(derivation.get("parent_task_id","")).startswith("xiangqi-history-")):
            raise ValueError(f"{task_id}: incomplete derived-position provenance")
        state,color=fen_to_board(parent_fen)
        turn=1 if color=="red" else -1
        from minibench.datasets.xiangqi import reference
        for uci in moves:
            legal={m.to_uci():m for m in reference.legal_moves(state,turn,"standard")}
            if uci not in legal:
                raise ValueError(f"{task_id}: illegal source derivation move")
            state=reference.apply_move(state,legal[uci])
            turn=-turn
        if board_to_fen(state,active_color="red" if turn==1 else "black")!=record["fen"]:
            raise ValueError(f"{task_id}: derived-position path does not reach its FEN")
    return record


def _validate_shortest_metadata(record: dict[str, Any]) -> None:
    """Metadata is not a proof; bind it to a separately checked artifact."""
    task_id = record["id"]
    n = record.get("shortest_mate_moves")
    proof = record.get("shortest_proof")
    if record["family"] not in {"xiangqi-history", "xiangqi-rule-variants"}:
        raise ValueError(f"{task_id}: shortest metadata only applies to multi-move tasks")
    if isinstance(n, bool) or not isinstance(n, int) or not 2 <= n <= 5:
        raise ValueError(f"{task_id}: certified release requires shortest mate in 2..5 moves")
    fields = {"version", "record_key", "tree_path", "tree_sha256",
              "shortest_attacker_moves", "tree_complete"}
    if not isinstance(proof, dict) or set(proof) != fields:
        raise ValueError(f"{task_id}: missing or invalid shortest_proof metadata")
    from minibench.datasets.xiangqi.solution_tree import VERSION, record_key
    if (proof["version"] != VERSION or proof["record_key"] != record_key(record)
            or proof["shortest_attacker_moves"] != n
            or proof["tree_complete"] is not True
            or not isinstance(proof["tree_sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", proof["tree_sha256"]) is None):
        raise ValueError(f"{task_id}: inconsistent shortest proof identity")
    folder = "h2" if record["family"] == "xiangqi-history" else "c2"
    if proof["tree_path"] != f"data/xiangqi/solution_trees/{folder}/{task_id}.json":
        raise ValueError(f"{task_id}: invalid shortest proof path")
    line=record.get("shortest_line_uci")
    if (not isinstance(line,list) or len(line)!=2*n-1
            or any(not isinstance(m,str) or re.fullmatch(r"[a-i][0-9][a-i][0-9]",m) is None for m in line)
            or record["oracle"]["best_move_uci"]!=line[0] or record["oracle"]["mate_in_plies"]!=2*n-1):
        raise ValueError(f"{task_id}: shortest strategy/oracle mismatch")


def _validate_mate_in_one_record(record: dict[str, Any]) -> None:
    analysis = record.get("d3_analysis")
    if not isinstance(analysis, dict):
        raise ValueError(f"{record['id']}: missing d3_analysis")
    expected = {
        "version", "legal_move_count", "mate_moves_uci", "mate_move_count",
        "non_mating_check_count", "piece_count", "difficulty_score",
        "difficulty_factors",
    }
    if set(analysis) != expected or analysis.get("version") != "d3-structural-v2":
        raise ValueError(f"{record['id']}: invalid d3_analysis schema")
    for key in ("legal_move_count", "mate_move_count", "non_mating_check_count", "piece_count"):
        if isinstance(analysis[key], bool) or not isinstance(analysis[key], int) or analysis[key] < 0:
            raise ValueError(f"{record['id']}: invalid d3_analysis.{key}")
    moves = analysis["mate_moves_uci"]
    if (not isinstance(moves, list) or not moves or moves != sorted(set(moves))
            or any(not isinstance(move, str) or re.fullmatch(r"[a-i][0-9][a-i][0-9]", move) is None for move in moves)):
        raise ValueError(f"{record['id']}: invalid d3_analysis.mate_moves_uci")
    if analysis["mate_move_count"] != len(moves) or analysis["piece_count"] != record["piece_count"]:
        raise ValueError(f"{record['id']}: inconsistent d3_analysis counts")
    if record["max_plies"] != 1 or record["oracle"]["mate_in_plies"] != 1:
        raise ValueError(f"{record['id']}: D3 requires an exact one-ply mate horizon")
    if record["oracle"]["best_move_uci"] not in moves:
        raise ValueError(f"{record['id']}: D3 oracle move is not in the complete mate set")
    legal = analysis["legal_move_count"]
    mates = analysis["mate_move_count"]
    near_misses = analysis["non_mating_check_count"]
    if legal < 1 or mates < 1 or mates > legal or near_misses > legal - mates:
        raise ValueError(f"{record['id']}: inconsistent D3 legal/mate/check counts")
    score = analysis["difficulty_score"]
    if (isinstance(score, bool) or not isinstance(score, (int, float))
            or not 0 <= score <= 1 or not math.isfinite(score)):
        raise ValueError(f"{record['id']}: invalid d3_analysis.difficulty_score")
    factors = analysis["difficulty_factors"]
    factor_keys = {
        "choice_pressure", "near_miss_rate", "state_load",
        "choice_pressure_percentile", "near_miss_rate_percentile",
        "state_load_percentile",
    }
    if not isinstance(factors, dict) or set(factors) != factor_keys:
        raise ValueError(f"{record['id']}: invalid d3_analysis.difficulty_factors")
    for key in factor_keys - {"state_load"}:
        value = factors[key]
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not 0 <= value <= 1 or not math.isfinite(value)):
            raise ValueError(f"{record['id']}: invalid D3 difficulty factor {key}")
    load = factors["state_load"]
    if (isinstance(load, bool) or not isinstance(load, int)
            or load != record["piece_count"]):
        raise ValueError(f"{record['id']}: D3 state_load must equal the piece count")
    expected_rates = {
        "choice_pressure": (legal - mates) / legal,
        "near_miss_rate": near_misses / (near_misses + mates),
    }
    for key, expected in expected_rates.items():
        if not math.isclose(factors[key], expected, rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"{record['id']}: D3 {key} disagrees with move counts")
    expected_score = round(math.fsum((
        0.50 * factors["choice_pressure_percentile"],
        0.30 * factors["near_miss_rate_percentile"],
        0.20 * factors["state_load_percentile"],
    )), 6)
    # Persisted six-decimal scores can differ in their last decimal place at a
    # binary floating-point tie across Python versions. This tolerance admits
    # that rounding only; it does not permit a different weighting formula.
    if not math.isclose(score, expected_score, rel_tol=0, abs_tol=1.1e-6):
        raise ValueError(f"{record['id']}: D3 difficulty score disagrees with its weighted factors")


def _validate_rule_record(record: dict[str, Any]) -> None:
    task_id = record["id"]
    scenario_id = record.get("scenario_id")
    if not isinstance(scenario_id, str) or not scenario_id.startswith(
        "xiangqi-rule-scenario-"
    ):
        raise ValueError(f"{task_id}: invalid scenario_id")
    ruleset = record.get("ruleset")
    if ruleset not in RULESETS:
        raise ValueError(f"{task_id}: unsupported ruleset {ruleset!r}")
    expected = {
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
    }[ruleset]
    if record.get("rules") != expected:
        raise ValueError(f"{task_id}: rules do not match ruleset {ruleset}")
    source_task_id = record.get("source_task_id")
    if not isinstance(source_task_id, str) or not source_task_id.startswith("xiangqi-history-"):
        raise ValueError(f"{task_id}: C2 requires an audited H2 source_task_id")
    analysis = record.get("c2_analysis")
    required = {
        "version", "source_task_id", "focus_ruleset", "pair_rulesets",
        "mate_in_moves", "reference_mate_plies", "principal_variation_uci",
        "rule_sensitive_along_reference", "search_depth", "loss_unit",
        "max_plies_extension", "verification", "interpretation",
    }
    if not isinstance(analysis, dict) or set(analysis) != required:
        raise ValueError(f"{task_id}: invalid c2_analysis schema")
    if analysis.get("version") != "c2-multistep-rule-mate-v3":
        raise ValueError(f"{task_id}: unsupported C2 protocol version")
    focus = analysis.get("focus_ruleset")
    if focus not in set(RULESETS) - {"standard"}:
        raise ValueError(f"{task_id}: invalid C2 focus ruleset")
    if analysis.get("pair_rulesets") != ["standard", focus] or ruleset not in {"standard", focus}:
        raise ValueError(f"{task_id}: C2 row is not part of its declared standard/rule pair")
    if record.get("design_stratum") != focus:
        raise ValueError(f"{task_id}: C2 design stratum must equal focus_ruleset")
    if analysis.get("source_task_id") != source_task_id:
        raise ValueError(f"{task_id}: C2 source metadata disagree")
    mate_moves = analysis.get("mate_in_moves")
    if not isinstance(mate_moves, int) or isinstance(mate_moves, bool) or not 2 <= mate_moves <= 12:
        raise ValueError(f"{task_id}: C2 requires a 2..12 move multi-ply mate")
    reference_plies = 2 * mate_moves - 1
    stratification_moves = record.get("shortest_mate_moves",mate_moves)
    if (isinstance(stratification_moves,bool) or not isinstance(stratification_moves,int)
            or not 2<=stratification_moves<=12):
        raise ValueError(f"{task_id}: invalid shortest/reference stratum")
    expected_band = "short" if stratification_moves <= 3 else "medium" if stratification_moves <= 5 else "long"
    if record["difficulty"] != expected_band:
        raise ValueError(f"{task_id}: C2 difficulty does not match mate distance")
    if (analysis.get("reference_mate_plies") != reference_plies
            or ("shortest_proof" not in record and record["oracle"].get("mate_in_plies") != reference_plies)):
        raise ValueError(f"{task_id}: inconsistent C2 reference mate distance")
    horizon_plies = 2*stratification_moves-1
    extension = analysis.get("max_plies_extension")
    # Original source evidence may retain its historical +4 metadata. Current
    # certified tasks never receive extra turns, even after leaving the tree.
    allowed_extensions = {0} if "shortest_proof" in record else {0, 4}
    if (isinstance(extension, bool) or not isinstance(extension, int)
            or extension not in allowed_extensions
            or record["max_plies"] != horizon_plies + extension):
        raise ValueError(f"{task_id}: C2 horizon must match its certified/reference bound")
    pv = analysis.get("principal_variation_uci")
    if (not isinstance(pv, list) or len(pv) != reference_plies
            or any(not isinstance(move, str) or re.fullmatch(r"[a-i][0-9][a-i][0-9]", move) is None for move in pv)):
        raise ValueError(f"{task_id}: invalid C2 principal variation")
    if "shortest_proof" not in record and record["oracle"].get("best_move_uci") != pv[0]:
        raise ValueError(f"{task_id}: C2 oracle move does not match the reference line")
    if analysis.get("search_depth") != 3:
        raise ValueError(f"{task_id}: C2 release requires rule-aware search depth 3")
    from minibench.datasets.xiangqi.variants.mate_search import LOSS_UNIT, replay_strict_mate_line
    if analysis.get("loss_unit") != LOSS_UNIT:
        raise ValueError(f"{task_id}: C2 loss unit is not the registered non-Pikafish scale")
    if not isinstance(analysis.get("rule_sensitive_along_reference"), bool):
        raise ValueError(f"{task_id}: invalid C2 rule-sensitivity flag")
    verification = analysis.get("verification")
    if not isinstance(verification, dict) or set(verification) != {"production", "independent"}:
        raise ValueError(f"{task_id}: invalid C2 verification metadata")
    for name in ("production", "independent"):
        proof = verification.get(name)
        if (not isinstance(proof, dict) or proof.get("valid") is not True
                or proof.get("terminal") != "checkmate" or proof.get("plies") != reference_plies):
            raise ValueError(f"{task_id}: invalid stored C2 {name} replay proof")
    if not isinstance(analysis.get("interpretation"), str) or not analysis["interpretation"]:
        raise ValueError(f"{task_id}: missing C2 claim limitation")

    board, color = fen_to_board(record["fen"])
    side = 1 if color == "red" else -1
    from minibench.datasets.xiangqi.variants.board import VariantBoard
    from minibench.datasets.xiangqi.variants.rules import Rule
    production = replay_strict_mate_line(
        VariantBoard(board, [Rule.from_dict(rule) for rule in internal_rules(record)]),
        side,
        pv,
        attacker_side=side,
    )
    if not production["valid"] or production["terminal"] != "checkmate":
        raise ValueError(f"{task_id}: C2 reference line fails production replay")
    from minibench.datasets.xiangqi import reference
    state = board
    turn = side
    active = internal_rules(record)
    for index, uci in enumerate(pv):
        moves = {move.to_uci(): move for move in reference.legal_moves(state, turn, active)}
        if uci not in moves:
            raise ValueError(f"{task_id}: C2 independent replay rejects ply {index}")
        state = reference.apply_move(state, moves[uci])
        turn = -turn
    if reference.terminal_status(state, turn, active) != "checkmate" or turn != -side:
        raise ValueError(f"{task_id}: C2 independent replay does not end in strict mate")


def _validate_history_record(record: dict[str, Any]) -> None:
    task_id = record["id"]
    if record.get("ruleset", "standard") != "standard" or record.get("rules"):
        raise ValueError(f"{task_id}: H2 requires standard rules without overrides")
    analysis = record.get("h2_analysis")
    required = {
        "version", "mate_in_moves", "reference_mate_plies", "verification_depths",
        "requested_depths", "principal_variation_uci", "validation_runs",
        "source_reused", "interpretation",
    }
    if not isinstance(analysis, dict) or not required.issubset(analysis):
        raise ValueError(f"{task_id}: invalid h2_analysis schema")
    mate_moves = analysis.get("mate_in_moves")
    if not isinstance(mate_moves, int) or isinstance(mate_moves, bool) or not 2 <= mate_moves <= 12:
        raise ValueError(f"{task_id}: H2 reference mate distance must be 2..12 moves")
    reference_plies = 2 * mate_moves - 1
    stratification_moves = record.get("shortest_mate_moves",mate_moves)
    if (isinstance(stratification_moves,bool) or not isinstance(stratification_moves,int)
            or not 2<=stratification_moves<=12):
        raise ValueError(f"{task_id}: invalid shortest/reference stratum")
    expected_band = "short" if stratification_moves <= 3 else "medium" if stratification_moves <= 5 else "long"
    if (analysis.get("reference_mate_plies") != reference_plies
            or ("shortest_proof" not in record and record["oracle"]["mate_in_plies"] != reference_plies)):
        raise ValueError(f"{task_id}: inconsistent H2 reference distance")
    if record["difficulty"] != expected_band:
        raise ValueError(f"{task_id}: H2 difficulty does not match reference distance")
    horizon_plies = 2*stratification_moves-1
    allowed_horizons = ({horizon_plies} if "shortest_proof" in record
                        else {horizon_plies, horizon_plies + 4})
    if record["max_plies"] not in allowed_horizons:
        raise ValueError(f"{task_id}: H2 horizon must match its certified/reference bound")
    if analysis.get("requested_depths") != [16, 20] or analysis.get("source_reused") is not False:
        raise ValueError(f"{task_id}: invalid H2 verification metadata")
    runs = analysis.get("validation_runs")
    if not isinstance(runs, list) or len(runs) != 2:
        raise ValueError(f"{task_id}: H2 requires two engine validation runs")
    pv = analysis.get("principal_variation_uci")
    if (not isinstance(pv, list) or len(pv) != reference_plies
            or any(not isinstance(move, str) or re.fullmatch(r"[a-i][0-9][a-i][0-9]", move) is None for move in pv)):
        raise ValueError(f"{task_id}: invalid H2 principal variation")
    if "shortest_proof" not in record and record["oracle"]["best_move_uci"] != pv[0]:
        raise ValueError(f"{task_id}: H2 oracle move does not match the primary PV")


def _validate_multimodal_record(record: dict[str, Any]) -> None:
    task_id = record["id"]
    if record["max_plies"] != 1 or record["oracle"]["mate_in_plies"] != 1:
        raise ValueError(f"{task_id}: M2 requires an exact one-ply mate horizon")
    source_task_id = record.get("source_task_id")
    if not isinstance(source_task_id, str) or not source_task_id.startswith("xiangqi-mate-in-one-"):
        raise ValueError(f"{task_id}: invalid M2 source_task_id")
    analysis = record.get("m2_analysis")
    required = {"version", "mate_moves_uci", "mate_move_count", "verification"}
    if not isinstance(analysis, dict) or set(analysis) != required:
        raise ValueError(f"{task_id}: invalid m2_analysis schema")
    if analysis.get("version") != "m2-paired-mate-in-one-v1":
        raise ValueError(f"{task_id}: unsupported M2 analysis version")
    moves = analysis.get("mate_moves_uci")
    if (not isinstance(moves, list) or not moves or moves != sorted(set(moves))
            or any(not isinstance(move, str) or re.fullmatch(r"[a-i][0-9][a-i][0-9]", move) is None for move in moves)):
        raise ValueError(f"{task_id}: invalid M2 mate move set")
    if analysis.get("mate_move_count") != len(moves) or record["oracle"]["best_move_uci"] not in moves:
        raise ValueError(f"{task_id}: inconsistent M2 mate metadata")


def load_records(
    path: str | Path, *, expected_family: str | None = None
) -> list[dict[str, Any]]:
    source = Path(path)
    records: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{source}:{line_number}: invalid JSON") from exc
            if not isinstance(raw, dict):
                raise ValueError(f"{source}:{line_number}: expected a JSON object")
            records.append(validate_record(raw, expected_family=expected_family))
    if not records:
        raise ValueError(f"{source} contains no Xiangqi records")
    ids = [record["id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{source}: duplicate Xiangqi task ids")
    return records


def internal_rules(record: dict[str, Any]) -> list[dict[str, Any]]:
    return {
        "standard": [],
        "horse-no-leg-block": [
            {"kind": "move_mod", "piece": "horse", "params": {"mod": "no_leg_restriction"}}
        ],
        "chariot-no-center": [
            {"kind": "zone_limit", "piece": "chariot", "params": {"zone": "not_center_cols"}}
        ],
        "soldier-free-retreat": [
            {"kind": "move_mod", "piece": "soldier", "params": {"mod": "free_retreat"}}
        ],
    }[record.get("ruleset", "standard")]


def runtime_dict(record: dict[str, Any]) -> dict[str, Any]:
    board, side_to_move = fen_to_board(record["fen"])
    payload = dict(record)
    payload.update(
        board=board,
        side_to_move="ally" if side_to_move == "red" else "enemy",
        agent_side="ally" if record["agent_color"] == "red" else "enemy",
    )
    if record["family"] == "xiangqi-rule-variants":
        payload["rules"] = internal_rules(record)
    return payload


def sample_records(
    records: Sequence[dict[str, Any]],
    *,
    count: int,
    seed: int,
    strategy: str = "family-default",
) -> list[dict[str, Any]]:
    if count < 1:
        raise ValueError("sample count must be positive")
    if count > len(records):
        raise ValueError(
            f"sample count {count} exceeds available records {len(records)}"
        )
    family = records[0]["family"]
    if any(record["family"] != family for record in records):
        raise ValueError("cannot sample records from multiple families")
    supported = {
        "family-default", "random", "stratified", "paired-random",
        "paired-stratified",
    }
    if strategy not in supported:
        raise ValueError(f"unsupported Xiangqi sampling strategy {strategy!r}")
    if strategy == "family-default":
        strategy = {
            "xiangqi-mate-in-one": "stratified",
            "xiangqi-history": "stratified",
            "xiangqi-rule-variants": "paired-stratified",
            "xiangqi-multimodal": "stratified",
        }[family]
    if strategy.startswith("paired-") and family != "xiangqi-rule-variants":
        raise ValueError("paired sampling is only defined for C2 scenarios")
    if strategy == "stratified" and family == "xiangqi-rule-variants":
        raise ValueError("C2 requires paired-stratified sampling")
    rng = random.Random(seed)
    if strategy == "random":
        ordered_records = sorted(records, key=lambda item: item["id"])
        return sorted(rng.sample(ordered_records, count), key=lambda item: item["id"])
    if strategy == "stratified":
        preferred = {
            "xiangqi-mate-in-one": ("easy", "medium", "hard"),
            "xiangqi-history": ("short", "medium", "long"),
            "xiangqi-multimodal": ("easy", "medium", "hard"),
        }.get(family, ())
        buckets: dict[str, list[dict[str, Any]]] = {}
        for record in sorted(records, key=lambda item: item["id"]):
            buckets.setdefault(record["difficulty"], []).append(record)
        ordered = [name for name in preferred if buckets.get(name)]
        if not ordered:
            ordered = sorted(name for name, items in buckets.items() if items)
        if not ordered:
            raise ValueError(f"{family} dataset has no difficulty buckets")
        base, remainder = divmod(count, len(ordered))
        selected: list[dict[str, Any]] = []
        remaining = count
        for index, name in enumerate(ordered):
            take = min(base + int(index < remainder), len(buckets[name]))
            selected.extend(rng.sample(buckets[name], take))
            remaining -= take
        for name in sorted(ordered, key=lambda label: (-len(buckets[label]), label)):
            if remaining == 0:
                break
            chosen_ids = {record["id"] for record in selected}
            available = [record for record in buckets[name] if record["id"] not in chosen_ids]
            take = min(remaining, len(available))
            selected.extend(rng.sample(available, take))
            remaining -= take
        if remaining:
            raise ValueError(
                f"sample count {count} exceeds available stratified records for {family}"
            )
        return sorted(selected, key=lambda item: item["id"])
    if family == "xiangqi-rule-variants":
        # C2 is a paired comparison. Do not mix standard-only control records
        # into a sampled rule-comparison run.
        groups: dict[str, list[dict[str, Any]]] = {}
        for record in records:
            groups.setdefault(record.get("scenario_id") or record["id"], []).append(record)
        paired = {}
        for scenario_id, group in groups.items():
            labels = {record.get("design_stratum") for record in group}
            if len(labels) != 1:
                continue
            focus = next(iter(labels))
            if focus not in set(RULESETS) - {"standard"}:
                continue
            if ({record.get("ruleset") for record in group} == {"standard", focus}
                    and len(group) == 2
                    and len({record["fen"] for record in group}) == 1):
                paired[scenario_id] = group
        scenarios = sorted(paired)
        if count > len(scenarios):
            raise ValueError(
                f"sample count {count} exceeds available complete C2 scenarios {len(scenarios)}"
            )
        if strategy == "paired-stratified":
            strata: dict[str, list[str]] = {}
            for scenario_id in scenarios:
                labels = {
                    str(record.get("design_stratum", "unspecified"))
                    for record in paired[scenario_id]
                }
                if len(labels) != 1:
                    raise ValueError(f"C2 scenario {scenario_id} has inconsistent design strata")
                strata.setdefault(labels.pop(), []).append(scenario_id)
            labels = sorted(strata)
            base, remainder = divmod(count, len(labels))
            chosen = []
            remaining = count
            for index, label in enumerate(labels):
                take = min(base + int(index < remainder), len(strata[label]))
                chosen.extend(rng.sample(strata[label], take))
                remaining -= take
            for label in sorted(labels, key=lambda key: (-len(strata[key]), key)):
                if not remaining:
                    break
                available = [item for item in strata[label] if item not in chosen]
                take = min(remaining, len(available))
                chosen.extend(rng.sample(available, take))
                remaining -= take
            if remaining:
                raise ValueError(f"sample count {count} exceeds stratified C2 scenarios")
        else:
            chosen = rng.sample(scenarios, count)
        selected = [record for scenario_id in chosen for record in paired[scenario_id]]
        return sorted(selected, key=lambda item: item["id"])
    ordered_records = sorted(records, key=lambda item: item["id"])
    return sorted(rng.sample(ordered_records, count), key=lambda item: item["id"])
