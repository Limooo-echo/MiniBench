from __future__ import annotations

import json
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
    if not isinstance(task_id, str) or not task_id.startswith(f"{family}-"):
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
    expected_goal = (
        "best-move-under-rule"
        if family == "xiangqi-rule-variants"
        else "checkmate"
    )
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
    if family == "xiangqi-rule-variants":
        _validate_rule_record(record)
        from minibench.datasets.xiangqi.reference import in_check

        side = 1 if side_to_move == "red" else -1
        if in_check(board, -side, internal_rules(record)):
            raise ValueError(f"{task_id}: active rules leave side not to move in check")
    return record


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
        if not isinstance(analysis[key], int) or analysis[key] < 0:
            raise ValueError(f"{record['id']}: invalid d3_analysis.{key}")
    moves = analysis["mate_moves_uci"]
    if (not isinstance(moves, list) or not moves or moves != sorted(set(moves))
            or any(not isinstance(move, str) or re.fullmatch(r"[a-i][0-9][a-i][0-9]", move) is None for move in moves)):
        raise ValueError(f"{record['id']}: invalid d3_analysis.mate_moves_uci")
    if analysis["mate_move_count"] != len(moves) or analysis["piece_count"] != record["piece_count"]:
        raise ValueError(f"{record['id']}: inconsistent d3_analysis counts")
    if not isinstance(analysis["difficulty_score"], (int, float)):
        raise ValueError(f"{record['id']}: invalid d3_analysis.difficulty_score")
    factors = analysis["difficulty_factors"]
    factor_keys = {
        "choice_pressure", "near_miss_rate", "state_load",
        "choice_pressure_percentile", "near_miss_rate_percentile",
        "state_load_percentile",
    }
    if not isinstance(factors, dict) or set(factors) != factor_keys:
        raise ValueError(f"{record['id']}: invalid d3_analysis.difficulty_factors")
    if any(not isinstance(factors[key], (int, float)) for key in factor_keys):
        raise ValueError(f"{record['id']}: non-numeric D3 difficulty factor")


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
        goal="agent_win",
        max_steps=record["max_plies"],
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
        expected_rulesets = set(RULESETS)
        paired = {
            scenario_id: group
            for scenario_id, group in groups.items()
            if {record.get("ruleset") for record in group} == expected_rulesets
            and len(group) == len(expected_rulesets)
            and len({record["fen"] for record in group}) == 1
        }
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
