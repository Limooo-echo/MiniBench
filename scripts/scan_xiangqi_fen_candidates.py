from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from minibench.datasets.xiangqi.engines.pikafish import PikafishEngine
from minibench.datasets.xiangqi.schema import fen_to_board
from minibench.datasets.xiangqi.variants.board import VariantBoard


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--min-mate", type=int, default=2)
    parser.add_argument("--max-mate", type=int, default=20)
    parser.add_argument("--max-plies-from-end", type=int)
    parser.add_argument(
        "--pikafish",
        type=Path,
        default=None,
        help="Pikafish executable; otherwise use the standard resolver.",
    )
    args = parser.parse_args()

    candidates = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.max_plies_from_end is not None:
        candidates = [
            item
            for item in candidates
            if int(item.get("plies_from_end", 10**9)) <= args.max_plies_from_end
        ]
    seen_fens: set[str] = set()
    counts: dict[str, int] = {}
    accepted = invalid = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    from minibench.datasets.xiangqi.engines.pikafish import (
        resolve_pikafish_executable,
    )

    executable = resolve_pikafish_executable(args.pikafish, start_dir=ROOT)
    with PikafishEngine(executable, timeout=args.timeout) as engine:
        with args.output.open("w", encoding="utf-8", newline="\n") as handle:
            for index, item in enumerate(candidates, start=1):
                fen = item["fen"]
                if fen in seen_fens:
                    continue
                seen_fens.add(fen)
                try:
                    board, color = fen_to_board(fen)
                    side = 1 if color == "red" else -1
                    if not VariantBoard(board, []).legal_moves(side):
                        invalid += 1
                        continue
                    analysis = engine.analyze_fen(fen, depth=args.depth)
                except Exception as exc:
                    invalid += 1
                    print(f"invalid {item['source_file']} ply={item['ply_index']}: {exc}", file=sys.stderr)
                    continue
                if analysis.score_kind == "mate" and args.min_mate <= analysis.score <= args.max_mate:
                    bucket = str(analysis.score)
                    counts[bucket] = counts.get(bucket, 0) + 1
                    accepted += 1
                    payload = {
                        **item,
                        "piece_count": sum(value != 0 for row in board for value in row),
                        "screening": {
                            "depth": analysis.depth,
                            "score_kind": analysis.score_kind,
                            "mate_in_moves": analysis.score,
                            "best_move_uci": analysis.bestmove,
                            "pv": list(analysis.pv),
                        },
                    }
                    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    handle.flush()
                if index % 500 == 0 or index == len(candidates):
                    print(
                        f"scan={index}/{len(candidates)} accepted={accepted} invalid={invalid} "
                        f"buckets={dict(sorted(counts.items()))}",
                        file=sys.stderr,
                        flush=True,
                    )
    print(json.dumps({"candidates": len(candidates), "accepted": accepted, "invalid": invalid, "counts": counts}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
