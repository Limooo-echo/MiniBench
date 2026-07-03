from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import random
import sys
from typing import Any, Sequence


BOARD_ROWS = 10
BOARD_COLS = 9
ALLY_GENERAL = 1
ENEMY_GENERAL = -1

PIECE_TYPES = ("rook", "cannon", "horse", "soldier")
DIFFICULTIES = ("easy", "medium", "hard")

ALLY_PIECE_IDS = {
    "rook": (8, 9),
    "cannon": (10, 11),
    "horse": (6, 7),
    "soldier": (12, 13, 14, 15, 16),
}
DISTRACTOR_ALLY_IDS = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
DISTRACTOR_ENEMY_IDS = tuple(-item for item in DISTRACTOR_ALLY_IDS)
SCREEN_IDS = (12, 13, 14, -12, -13, -14, 2, -2)


@dataclass(frozen=True)
class CaptureCandidate:
    board: list[list[int]]
    piece_type: str
    difficulty: str
    template: str


@dataclass(frozen=True)
class WinningAction:
    action: int
    uci: str


def generate_xiangqi_capture_tasks(
    *,
    output: Path,
    count: int = 100,
    seed: int = 20260702,
    prefix: str = "xq-capture-generated",
    piece_types: Sequence[str] = PIECE_TYPES,
    difficulties: Sequence[str] = DIFFICULTIES,
    max_attempts: int | None = None,
    overwrite: bool = False,
    progress_interval: int | None = 100,
) -> dict[str, Any]:
    if count < 1:
        raise ValueError("count must be positive")
    if output.exists() and not overwrite:
        raise ValueError(f"{output} already exists; pass overwrite=True to replace it")

    piece_types = _validate_choices(piece_types, PIECE_TYPES, "piece_types")
    difficulties = _validate_choices(difficulties, DIFFICULTIES, "difficulties")
    attempts_limit = max_attempts or max(5_000, count * 500)

    rng = random.Random(seed)
    records: list[dict[str, object]] = []
    seen_boards: set[tuple[int, ...]] = set()
    skipped = Counter()

    for attempt in range(1, attempts_limit + 1):
        if len(records) >= count:
            break

        piece_type = rng.choice(tuple(piece_types))
        difficulty = rng.choice(tuple(difficulties))
        try:
            candidate = make_candidate(piece_type, difficulty, rng)
        except ValueError:
            skipped["candidate_error"] += 1
            continue

        key = board_key(candidate.board)
        if key in seen_boards:
            skipped["duplicate_board"] += 1
            continue

        try:
            winning_actions = winning_actions_for_board(candidate.board)
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "gym-xiangqi is required to generate validated Xiangqi capture tasks"
            ) from exc
        except Exception:
            skipped["invalid_position"] += 1
            continue

        if len(winning_actions) != 1:
            skipped[f"winning_actions_{len(winning_actions)}"] += 1
            continue

        seen_boards.add(key)
        task_id = f"{prefix}-{len(records) + 1:03d}"
        record = capture_task_record(
            task_id=task_id,
            candidate=candidate,
            answer=winning_actions[0],
            seed=seed,
        )
        records.append(record)

        if progress_interval and (attempt % progress_interval == 0 or len(records) == count):
            _print_progress(
                attempt=attempt,
                attempts_limit=attempts_limit,
                generated=len(records),
                target=count,
                skipped=skipped,
            )

    if progress_interval:
        print(file=sys.stderr)

    if len(records) < count:
        raise RuntimeError(
            f"generated only {len(records)}/{count} tasks after {attempts_limit} attempts; "
            f"skipped={dict(sorted(skipped.items()))}"
        )

    write_jsonl(records, output)
    return {
        "output": str(output),
        "generated": len(records),
        "seed": seed,
        "piece_types": list(piece_types),
        "difficulties": list(difficulties),
        "attempts": attempt,
        "skipped": dict(sorted(skipped.items())),
        "by_piece": dict(sorted(Counter(record["answer"]["piece"] for record in records).items())),
        "by_difficulty": dict(
            sorted(
                Counter(
                    tag.removeprefix("difficulty:")
                    for record in records
                    for tag in record["tags"]
                    if isinstance(tag, str) and tag.startswith("difficulty:")
                ).items()
            )
        ),
    }


def _validate_choices(
    requested: Sequence[str],
    allowed: Sequence[str],
    name: str,
) -> tuple[str, ...]:
    choices = tuple(item.strip() for item in requested if item.strip())
    if not choices:
        raise ValueError(f"{name} must not be empty")
    invalid = sorted(set(choices) - set(allowed))
    if invalid:
        raise ValueError(f"unknown {name}: {', '.join(invalid)}")
    return choices


def make_candidate(
    piece_type: str,
    difficulty: str,
    rng: random.Random,
) -> CaptureCandidate:
    board = empty_board()
    place(board, 9, 4, ALLY_GENERAL)
    enemy_row, enemy_col = random_enemy_general_square(rng)
    place(board, enemy_row, enemy_col, ENEMY_GENERAL)

    if piece_type == "rook":
        template = place_rook_capture(board, enemy_row, enemy_col, rng)
    elif piece_type == "cannon":
        template = place_cannon_capture(board, enemy_row, enemy_col, rng)
    elif piece_type == "horse":
        template = place_horse_capture(board, enemy_row, enemy_col, rng)
    elif piece_type == "soldier":
        template = place_soldier_capture(board, enemy_row, enemy_col, rng)
    else:
        raise ValueError(f"unsupported piece_type: {piece_type}")

    add_distractors(board, difficulty, rng)
    return CaptureCandidate(
        board=board,
        piece_type=piece_type,
        difficulty=difficulty,
        template=template,
    )


def empty_board() -> list[list[int]]:
    return [[0 for _ in range(BOARD_COLS)] for _ in range(BOARD_ROWS)]


def board_key(board: list[list[int]]) -> tuple[int, ...]:
    return tuple(value for row in board for value in row)


def place(board: list[list[int]], row: int, col: int, value: int) -> None:
    if not in_bounds(row, col):
        raise ValueError(f"square out of bounds: {(row, col)}")
    if board[row][col] != 0:
        raise ValueError(f"square occupied: {(row, col)}")
    board[row][col] = value


def in_bounds(row: int, col: int) -> bool:
    return 0 <= row < BOARD_ROWS and 0 <= col < BOARD_COLS


def random_enemy_general_square(rng: random.Random) -> tuple[int, int]:
    return rng.randint(0, 2), rng.randint(3, 5)


def choose_piece_id(piece_type: str, board: list[list[int]], rng: random.Random) -> int:
    used = {abs(value) for row in board for value in row if value != 0}
    ids = [piece_id for piece_id in ALLY_PIECE_IDS[piece_type] if piece_id not in used]
    if not ids:
        raise ValueError(f"no available id for {piece_type}")
    return rng.choice(ids)


def place_rook_capture(
    board: list[list[int]],
    enemy_row: int,
    enemy_col: int,
    rng: random.Random,
) -> str:
    piece_id = choose_piece_id("rook", board, rng)
    if rng.choice((True, False)):
        cols = [col for col in range(BOARD_COLS) if col != enemy_col and board[enemy_row][col] == 0]
        if not cols:
            raise ValueError("no rook row square")
        place(board, enemy_row, rng.choice(cols), piece_id)
        return "rook-row"

    rows = [row for row in range(BOARD_ROWS) if row != enemy_row and board[row][enemy_col] == 0]
    if not rows:
        raise ValueError("no rook file square")
    place(board, rng.choice(rows), enemy_col, piece_id)
    return "rook-file"


def place_cannon_capture(
    board: list[list[int]],
    enemy_row: int,
    enemy_col: int,
    rng: random.Random,
) -> str:
    piece_id = choose_piece_id("cannon", board, rng)
    placements: list[tuple[int, int, int, int, str]] = []

    for col in range(BOARD_COLS):
        if col == enemy_col or board[enemy_row][col] != 0:
            continue
        between = between_squares(enemy_row, enemy_col, enemy_row, col)
        if between:
            for screen_row, screen_col in between:
                if board[screen_row][screen_col] == 0:
                    placements.append((enemy_row, col, screen_row, screen_col, "cannon-row"))

    for row in range(BOARD_ROWS):
        if row == enemy_row or board[row][enemy_col] != 0:
            continue
        between = between_squares(enemy_row, enemy_col, row, enemy_col)
        if between:
            for screen_row, screen_col in between:
                if board[screen_row][screen_col] == 0:
                    placements.append((row, enemy_col, screen_row, screen_col, "cannon-file"))

    if not placements:
        raise ValueError("no cannon capture placement")

    row, col, screen_row, screen_col, template = rng.choice(placements)
    place(board, row, col, piece_id)
    place(board, screen_row, screen_col, choose_screen_id(board, rng))
    return template


def between_squares(
    row_a: int,
    col_a: int,
    row_b: int,
    col_b: int,
) -> list[tuple[int, int]]:
    if row_a == row_b:
        start, end = sorted((col_a, col_b))
        return [(row_a, col) for col in range(start + 1, end)]
    if col_a == col_b:
        start, end = sorted((row_a, row_b))
        return [(row, col_a) for row in range(start + 1, end)]
    return []


def choose_screen_id(board: list[list[int]], rng: random.Random) -> int:
    used_ally = {value for row in board for value in row if value > 0}
    used_enemy = {value for row in board for value in row if value < 0}
    choices = [
        value
        for value in SCREEN_IDS
        if (value > 0 and value not in used_ally) or (value < 0 and value not in used_enemy)
    ]
    if not choices:
        raise ValueError("no available screen piece")
    return rng.choice(choices)


def place_horse_capture(
    board: list[list[int]],
    enemy_row: int,
    enemy_col: int,
    rng: random.Random,
) -> str:
    piece_id = choose_piece_id("horse", board, rng)
    deltas = [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)]
    rng.shuffle(deltas)
    for dr, dc in deltas:
        row, col = enemy_row + dr, enemy_col + dc
        if not in_bounds(row, col) or board[row][col] != 0:
            continue
        leg_row, leg_col = horse_leg_square(row, col, enemy_row, enemy_col)
        if board[leg_row][leg_col] != 0:
            continue
        place(board, row, col, piece_id)
        return "horse-leg-clear"
    raise ValueError("no horse capture placement")


def horse_leg_square(
    start_row: int,
    start_col: int,
    end_row: int,
    end_col: int,
) -> tuple[int, int]:
    dr = end_row - start_row
    dc = end_col - start_col
    if abs(dr) == 2 and abs(dc) == 1:
        return start_row + (1 if dr > 0 else -1), start_col
    if abs(dr) == 1 and abs(dc) == 2:
        return start_row, start_col + (1 if dc > 0 else -1)
    raise ValueError("not a horse move")


def place_soldier_capture(
    board: list[list[int]],
    enemy_row: int,
    enemy_col: int,
    rng: random.Random,
) -> str:
    piece_id = choose_piece_id("soldier", board, rng)
    candidates = []
    forward_row = enemy_row + 1
    if in_bounds(forward_row, enemy_col) and board[forward_row][enemy_col] == 0:
        candidates.append((forward_row, enemy_col, "soldier-forward"))
    for col in (enemy_col - 1, enemy_col + 1):
        if in_bounds(enemy_row, col) and board[enemy_row][col] == 0:
            candidates.append((enemy_row, col, "soldier-sideways"))
    if not candidates:
        raise ValueError("no soldier capture placement")
    row, col, template = rng.choice(candidates)
    place(board, row, col, piece_id)
    return template


def add_distractors(
    board: list[list[int]],
    difficulty: str,
    rng: random.Random,
) -> None:
    if difficulty == "easy":
        count = rng.randint(0, 1)
    elif difficulty == "medium":
        count = rng.randint(2, 4)
    elif difficulty == "hard":
        count = rng.randint(5, 8)
    else:
        raise ValueError(f"unsupported difficulty: {difficulty}")

    used_ally = {value for row in board for value in row if value > 0}
    used_enemy = {value for row in board for value in row if value < 0}
    available = [
        value
        for value in DISTRACTOR_ALLY_IDS + DISTRACTOR_ENEMY_IDS
        if (value > 0 and value not in used_ally) or (value < 0 and value not in used_enemy)
    ]
    rng.shuffle(available)

    for value in available[:count]:
        empties = [
            (row, col)
            for row in range(BOARD_ROWS)
            for col in range(BOARD_COLS)
            if board[row][col] == 0 and not is_enemy_palace_square(row, col)
        ]
        if not empties:
            return
        row, col = rng.choice(empties)
        board[row][col] = value


def is_enemy_palace_square(row: int, col: int) -> bool:
    return 0 <= row <= 2 and 3 <= col <= 5


def winning_actions_for_board(board: list[list[int]]) -> list[WinningAction]:
    from minibench.datasets.xiangqi.engines.pikafish import action_to_uci
    from minibench.datasets.xiangqi.env import make_xiangqi_env_from_board, strict_legal_actions

    env = make_xiangqi_env_from_board(board, side_to_move="ally")
    try:
        actions = strict_legal_actions(env)
    finally:
        env.close()

    winning: list[WinningAction] = []
    for action in actions:
        trial = make_xiangqi_env_from_board(board, side_to_move="ally")
        try:
            _obs, reward, done, _info = trial.step(action)
            if done and reward >= 100:
                winning.append(WinningAction(action=action, uci=action_to_uci(action)))
        finally:
            trial.close()
    return winning


def capture_task_record(
    *,
    task_id: str,
    candidate: CaptureCandidate,
    answer: WinningAction,
    seed: int,
) -> dict[str, object]:
    return {
        "id": task_id,
        "board": candidate.board,
        "side_to_move": "ally",
        "agent_side": "ally",
        "opponent": "none",
        "max_steps": 1,
        "goal": "capture_enemy_general",
        "tags": [
            "xiangqi",
            "generated",
            "endgame",
            "mate-in-one",
            f"difficulty:{candidate.difficulty}",
            f"piece:{candidate.piece_type}",
            "environment",
        ],
        "answer": {
            "action": answer.action,
            "uci": answer.uci,
            "piece": candidate.piece_type,
            "template": candidate.template,
        },
        "generator": {
            "name": "xiangqi-simple-capture-v1",
            "seed": seed,
            "requires_unique_winning_action": True,
        },
    }


def write_jsonl(records: Sequence[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _print_progress(
    *,
    attempt: int,
    attempts_limit: int,
    generated: int,
    target: int,
    skipped: Counter,
) -> None:
    width = 28
    filled = int(width * generated / target) if target else width
    bar = "#" * filled + "-" * (width - filled)
    top_skipped = ",".join(
        f"{key}:{value}" for key, value in sorted(skipped.items())[:3]
    )
    print(
        f"\r[generate-xiangqi-capture] [{bar}] generated={generated}/{target} "
        f"attempts={attempt}/{attempts_limit} skipped={top_skipped}",
        file=sys.stderr,
        end="",
        flush=True,
    )
