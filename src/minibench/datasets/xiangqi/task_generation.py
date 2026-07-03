from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
import re
from typing import Iterable, Sequence

from minibench.datasets.xiangqi.dataset import xiangqi_task_from_dict
from minibench.datasets.xiangqi.engines.pikafish import PikafishAnalysis


FEN_TO_PIECE_POOLS = {
    "K": (1,),
    "A": (2, 3),
    "B": (4, 5),
    "N": (6, 7),
    "R": (8, 9),
    "C": (10, 11),
    "P": (12, 13, 14, 15, 16),
}

FEN_TAG_RE = re.compile(rb'\[FEN\s+"([^"]+)"\]')


@dataclass(frozen=True)
class FenPosition:
    fen: str
    board: list[list[int]]
    side_to_move: str


@dataclass(frozen=True)
class CcpdFenRecord:
    fen: str
    source_file: str
    source_kind: str


@dataclass(frozen=True)
class GeneratedXiangqiTask:
    record: dict[str, object]
    analysis: PikafishAnalysis


def fen_to_position(fen: str) -> FenPosition:
    parts = fen.strip().split()
    if len(parts) < 2:
        raise ValueError(f"FEN must include board and active color: {fen}")

    active_color = parts[1]
    if active_color == "w":
        side_to_move = "ally"
    elif active_color == "b":
        side_to_move = "enemy"
    else:
        raise ValueError(f"unsupported FEN active color: {active_color}")

    red_pools = {piece: list(ids) for piece, ids in FEN_TO_PIECE_POOLS.items()}
    black_pools = {piece: list(ids) for piece, ids in FEN_TO_PIECE_POOLS.items()}
    board: list[list[int]] = []
    for fen_row in parts[0].split("/"):
        row: list[int] = []
        col = 0
        for ch in fen_row:
            if ch.isdigit():
                empty_count = int(ch)
                row.extend([0] * empty_count)
                col += empty_count
                continue

            piece = ch.upper()
            pools = red_pools if ch.isupper() else black_pools
            if piece not in pools:
                raise ValueError(f"unsupported FEN piece {ch!r}: {fen}")
            if not pools[piece]:
                raise ValueError(f"too many {piece} pieces in FEN: {fen}")

            piece_id = pools[piece].pop(0)
            row.append(piece_id if ch.isupper() else -piece_id)
            col += 1

        if col != 9 or len(row) != 9:
            raise ValueError(f"FEN row must contain 9 files: {fen_row!r}")
        board.append(row)

    if len(board) != 10:
        raise ValueError(f"FEN board must contain 10 ranks: {fen}")
    if not any(1 in row for row in board) or not any(-1 in row for row in board):
        raise ValueError(f"FEN must contain both generals: {fen}")

    return FenPosition(fen=fen, board=board, side_to_move=side_to_move)


def extract_fen_from_pgn(path: Path) -> str | None:
    match = FEN_TAG_RE.search(path.read_bytes())
    if not match:
        return None
    return match.group(1).decode("ascii")


def iter_ccpd_fens(root: Path) -> Iterable[CcpdFenRecord]:
    base = root / "Dataset" if (root / "Dataset").is_dir() else root
    for path in sorted(base.rglob("*.pgn")):
        fen = extract_fen_from_pgn(path)
        if fen is None:
            continue
        try:
            source_file = path.relative_to(root).as_posix()
        except ValueError:
            source_file = path.as_posix()
        yield CcpdFenRecord(
            fen=fen,
            source_file=source_file,
            source_kind=ccpd_source_kind(source_file),
        )


def ccpd_source_kind(source_file: str) -> str:
    patterns = (
        ("kill-tactic", "\u6bba\u5c40"),
        ("fullgame-tactics", "\u5168\u76e4\u6230\u8853"),
        ("endgame", "\u6b98\u5c40"),
        ("midgame", "\u4e2d\u5c40"),
        ("opening", "\u958b\u5c40"),
        ("match", "\u5c0d\u5c40"),
    )
    for kind, pattern in patterns:
        if pattern in source_file:
            return kind
    return "unknown"


def task_record_from_fen(
    *,
    task_id: str,
    fen: str,
    source_file: str,
    source_kind: str,
    category: str,
    analysis: PikafishAnalysis,
    max_steps: int,
) -> dict[str, object]:
    position = fen_to_position(fen)
    if category == "tactical-win":
        goal = "agent_win"
        engine_tag = (
            f"engine:mate-{analysis.score}"
            if analysis.score_kind == "mate"
            else f"engine:agent-plus-{analysis.score}"
        )
    elif category == "advantage-play":
        goal = "agent_survive"
        engine_tag = (
            f"engine:agent-plus-{analysis.score}"
            if analysis.score_kind == "cp"
            else f"engine:mate-{analysis.score}"
        )
    elif category == "survival-defense":
        goal = "agent_survive"
        engine_tag = (
            f"engine:agent-minus-{abs(analysis.score)}"
            if analysis.score_kind == "cp"
            else f"engine:mate-{analysis.score}"
        )
    else:
        raise ValueError(f"unknown generated category: {category}")

    record: dict[str, object] = {
        "id": task_id,
        "board": position.board,
        "side_to_move": position.side_to_move,
        "agent_side": position.side_to_move,
        "opponent": "pikafish",
        "max_steps": max_steps,
        "goal": goal,
        "tags": [
            "xiangqi",
            "ccpd",
            "pikafish-opponent",
            "difficulty:hard",
            f"category:{category}",
            engine_tag,
            "environment",
            f"source-kind:{source_kind}",
        ],
        "source": {
            "dataset": "Chinese-Chess-Practical-Dataset",
            "license": "CC-BY-4.0",
            "file": source_file,
            "kind": source_kind,
            "fen": fen,
        },
        "engine": {
            "name": "pikafish",
            "bestmove": analysis.bestmove,
            "score_kind": analysis.score_kind,
            "score": analysis.score,
            "depth": analysis.depth,
            "pv": list(analysis.pv),
        },
    }
    validate_generated_record(record)
    return record


def endgame_score_bucket(analysis: PikafishAnalysis) -> str:
    if analysis.score_kind == "mate":
        if analysis.score > 0:
            return "mate-for-agent"
        if analysis.score < 0:
            return "mate-against-agent"
        return "balanced"
    if analysis.score >= 500:
        return "winning"
    if analysis.score >= 150:
        return "advantage"
    if analysis.score <= -500:
        return "losing"
    if analysis.score <= -150:
        return "disadvantage"
    return "balanced"


def endgame_task_record_from_fen(
    *,
    task_id: str,
    fen: str,
    source_file: str,
    source_kind: str,
    analysis: PikafishAnalysis | None = None,
    max_steps: int,
    validate: bool = True,
) -> dict[str, object]:
    position = fen_to_position(fen)
    tags = [
        "xiangqi",
        "ccpd",
        "ccpd-endgame",
        "pikafish-opponent",
        "difficulty:hard",
        "category:endgame-play",
        "environment",
        f"source-kind:{source_kind}",
    ]
    label: dict[str, object] = {"kind": "ccpd_endgame", "advantage": "unlabeled"}
    engine: dict[str, object] | None = None
    if analysis is None:
        tags.append("label:unlabeled")
    else:
        advantage = endgame_score_bucket(analysis)
        engine_tag = (
            f"engine:agent-score-{analysis.score}"
            if analysis.score_kind == "cp"
            else f"engine:mate-{analysis.score}"
        )
        tags.extend([f"label:{advantage}", engine_tag])
        label = {
            "kind": "pikafish_static_score",
            "score_kind": analysis.score_kind,
            "score": analysis.score,
            "score_perspective": "agent_side_to_move",
            "advantage": advantage,
            "bestmove": analysis.bestmove,
            "depth": analysis.depth,
            "pv": list(analysis.pv),
        }
        engine = {
            "name": "pikafish",
            "bestmove": analysis.bestmove,
            "score_kind": analysis.score_kind,
            "score": analysis.score,
            "depth": analysis.depth,
            "pv": list(analysis.pv),
        }

    record: dict[str, object] = {
        "id": task_id,
        "board": position.board,
        "side_to_move": position.side_to_move,
        "agent_side": position.side_to_move,
        "opponent": "pikafish",
        "max_steps": max_steps,
        "goal": "agent_survive",
        "tags": tags,
        "source": {
            "dataset": "Chinese-Chess-Practical-Dataset",
            "license": "CC-BY-4.0",
            "file": source_file,
            "kind": source_kind,
            "fen": fen,
        },
        "label": label,
    }
    if engine is not None:
        record["engine"] = engine
    if validate:
        validate_generated_record(record)
    return record


def validate_generated_record(record: dict[str, object]) -> dict[str, object]:
    from minibench.datasets.xiangqi.env import make_xiangqi_env_from_board, strict_legal_actions

    task = xiangqi_task_from_dict(record)
    env = make_xiangqi_env_from_board(task.board, side_to_move=task.side_to_move)
    try:
        if not strict_legal_actions(env):
            raise ValueError(f"{task.id}: generated position has no safe legal actions")
    finally:
        env.close()
    return record


def classify_analysis(
    analysis: PikafishAnalysis,
    *,
    tactical_mate_max: int,
    advantage_cp: int,
    survival_cp: int,
) -> str | None:
    if analysis.score_kind == "mate":
        if 0 < analysis.score <= tactical_mate_max:
            return "tactical-win"
        return None

    if analysis.score_kind != "cp":
        return None
    if analysis.score >= advantage_cp:
        return "advantage-play"
    if analysis.score <= survival_cp:
        return "survival-defense"
    return None


def balanced_records(
    tasks: Sequence[GeneratedXiangqiTask],
    *,
    per_category: int,
    rng: random.Random,
) -> list[dict[str, object]]:
    buckets: dict[str, list[dict[str, object]]] = {
        "tactical-win": [],
        "advantage-play": [],
        "survival-defense": [],
    }
    for item in tasks:
        tags = item.record.get("tags", [])
        if not isinstance(tags, list):
            continue
        category = next(
            (tag.removeprefix("category:") for tag in tags if tag.startswith("category:")),
            None,
        )
        if category in buckets:
            buckets[category].append(item.record)

    records: list[dict[str, object]] = []
    for category in buckets:
        rng.shuffle(buckets[category])
        records.extend(buckets[category][:per_category])
    return records


def write_jsonl(records: Sequence[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
