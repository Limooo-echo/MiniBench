"""Merge JSONL candidate pools deterministically, deduplicating exact FENs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    by_fen: dict[str, dict] = {}
    per_input: dict[str, int] = {}
    for path in args.input:
        count = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            by_fen.setdefault(item["fen"], item)
            count += 1
        per_input[str(path)] = count
    args.output.write_text(
        "".join(json.dumps(by_fen[fen], ensure_ascii=False) + "\n" for fen in sorted(by_fen)),
        encoding="utf-8",
    )
    print(json.dumps({"inputs": per_input, "unique_fens": len(by_fen)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
