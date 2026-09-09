from __future__ import annotations

import json
import random
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from minibench.datasets.xiangqi.schema import RULESETS, internal_rules
from minibench.datasets.xiangqi.variants.board import VariantBoard
from minibench.datasets.xiangqi.variants.rules import Rule
from minibench.datasets.xiangqi.variants.search import score_moves


def horse_template():
    b = [[0] * 9 for _ in range(10)]
    b[0][6] = -1; b[2][3] = 6; b[3][3] = -12; b[4][4] = -9
    b[5][4] = 4; b[5][7] = -6; b[6][7] = 8; b[8][4] = 1
    return b


def chariot_template():
    b = [[0] * 9 for _ in range(10)]
    b[0][4] = -1; b[1][3] = 12; b[1][5] = 13; b[3][4] = -12
    b[3][5] = 6; b[5][4] = 8; b[7][0] = -6; b[8][0] = -13
    b[8][4] = 1; b[9][0] = 10
    return b


def soldier_template():
    b = [[0] * 9 for _ in range(10)]
    b[0][6] = -1; b[3][1] = -12; b[5][2] = 6
    b[7][3] = 12; b[8][3] = -9; b[9][4] = 1
    return b


RULES = {
    name: [Rule.from_dict(item) for item in internal_rules({"ruleset": name})]
    for name in RULESETS
}


def mirror(board):
    return [list(reversed(row)) for row in board]


def analyse(board, ruleset):
    position = VariantBoard(board, RULES[ruleset])
    if position._is_in_check(1) or position._is_in_check(-1):
        return None
    scored = score_moves(position, 1, 3, 1)
    if len(scored) < 2:
        return None
    margin = scored[0][1] - scored[1][1]
    if margin < 0.5:
        return None
    return {
        "legal": len(position.legal_moves(1)),
        "best": scored[0][0].to_uci(),
        "score": scored[0][1],
        "margin": margin,
    }


def fingerprint(board):
    return tuple(value for row in board for value in row)


def mutate(base, rng, index):
    b = [row[:] for row in (mirror(base) if index % 2 else base)]
    occupied = {(r, c) for r in range(10) for c in range(9) if b[r][c]}
    used = {abs(b[r][c]) for r, c in occupied}
    red_pool = [2, 3, 4, 5, 7, 9, 10, 11, 13, 14, 15, 16]
    black_pool = [-2, -3, -4, -5, -7, -8, -10, -11, -13, -14, -15, -16]
    pool = [piece for piece in red_pool + black_pool if abs(piece) not in used]
    rng.shuffle(pool)
    count = rng.randint(0, 4)
    safe = [(r, c) for r in range(1, 9) for c in range(9)
            if (r, c) not in occupied and c != 4]
    rng.shuffle(safe)
    for piece, (r, c) in zip(pool[:count], safe):
        b[r][c] = piece
    return b


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    results = {}
    for focus, builder in (
        ("horse-no-leg-block", horse_template),
        ("chariot-no-center", chariot_template),
        ("soldier-free-retreat", soldier_template),
    ):
        accepted = []
        seen = set()
        for index in range(5000):
            board = mutate(builder(), rng, index)
            key = fingerprint(board)
            if key in seen:
                continue
            seen.add(key)
            details = {name: analyse(board, name) for name in RULESETS}
            if any(value is None for value in details.values()):
                continue
            if details["standard"]["best"] == details[focus]["best"]:
                continue
            accepted.append({"board": board, "focus": focus, "analysis": details})
            print(f"{focus}: {len(accepted)}/20 after {index + 1}", flush=True)
            if len(accepted) == 20:
                break
        results[focus] = accepted
    existing = {
        fingerprint(item["board"])
        for items in results.values() for item in items
    }
    controls = []
    builders = (horse_template, chariot_template, soldier_template)
    for index in range(10000):
        board = mutate(builders[index % len(builders)](), rng, index + 10000)
        key = fingerprint(board)
        if key in existing:
            continue
        analysis = analyse(board, "standard")
        if analysis is None:
            continue
        existing.add(key)
        controls.append({"board": board, "analysis": analysis})
        if len(controls) == 10:
            break
    if len(controls) != 10:
        raise RuntimeError(f"found only {len(controls)} standard controls")
    results["standard-controls"] = controls
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: len(value) for key, value in results.items()}))


if __name__ == "__main__":
    main()
