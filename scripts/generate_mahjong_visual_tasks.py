from __future__ import annotations

import argparse
import json
from pathlib import Path

from minibench.datasets.mahjong.generation import generate_mahjong_visual_tasks


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate paired Mahjong visual tasks.")
    parser.add_argument("--output", type=Path, default=Path("data/mahjong/visual_tasks.jsonl"))
    parser.add_argument("--render-dir", type=Path, default=None)
    parser.add_argument("--count-per-type", type=int, default=15)
    parser.add_argument("--visible-count", type=int, action="append", default=None)
    parser.add_argument("--table-columns", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--prefix", default="mj-visual")
    parser.add_argument("--max-attempts", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    summary = generate_mahjong_visual_tasks(
        output=args.output,
        render_dir=args.render_dir,
        count_per_type=args.count_per_type,
        visible_count=tuple(args.visible_count or (10, 20)),
        table_columns=args.table_columns,
        seed=args.seed,
        prefix=args.prefix,
        max_attempts=args.max_attempts,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
