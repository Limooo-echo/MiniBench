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
HISTORY_MODES = ("full-state", "move-history-only")
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
    if record.get("goal") != "checkmate":
        raise ValueError(f"{task_id}: goal must be checkmate")
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
    if family == "xiangqi-rule-variants":
        _validate_rule_record(record)
    return record


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
    rng = random.Random(seed)
    if family == "xiangqi-mate-in-one":
        buckets = {name: [] for name in ("easy", "medium", "hard")}
        for record in sorted(records, key=lambda item: item["id"]):
            buckets.setdefault(record["difficulty"], []).append(record)
        ordered = [name for name in ("easy", "medium", "hard") if buckets[name]]
        if not ordered:
            ordered = sorted(name for name, items in buckets.items() if items)
        base, remainder = divmod(count, len(ordered))
        allocations = {
            name: base + int(index < remainder) for index, name in enumerate(ordered)
        }
        selected: list[dict[str, Any]] = []
        for name in ordered:
            if allocations[name] > len(buckets[name]):
                raise ValueError(f"not enough {name} records for stratified sampling")
            selected.extend(rng.sample(buckets[name], allocations[name]))
        return sorted(selected, key=lambda item: item["id"])
    if family == "xiangqi-rule-variants":
        buckets = {name: [] for name in RULESETS}
        for record in sorted(records, key=lambda item: item["id"]):
            buckets[record["ruleset"]].append(record)
        exact = {
            name: count * len(items) / len(records) for name, items in buckets.items()
        }
        allocations = {name: int(value) for name, value in exact.items()}
        remaining = count - sum(allocations.values())
        order = sorted(
            RULESETS,
            key=lambda name: (-(exact[name] - allocations[name]), RULESETS.index(name)),
        )
        for name in order[:remaining]:
            allocations[name] += 1
        selected = []
        for name in RULESETS:
            selected.extend(rng.sample(buckets[name], allocations[name]))
        return sorted(selected, key=lambda item: item["id"])
    ordered_records = sorted(records, key=lambda item: item["id"])
    return sorted(rng.sample(ordered_records, count), key=lambda item: item["id"])
