from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--positions", type=Path, required=True)
    parser.add_argument("--mates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lookback-turns", type=int, default=10)
    args = parser.parse_args()

    positions = read_jsonl(args.positions)
    mates = read_jsonl(args.mates)
    indexed = {(item["source_file"], item["ply_index"]): item for item in positions}
    selected: dict[str, dict] = {}
    for mate in mates:
        for turns in range(1, args.lookback_turns + 1):
            item = indexed.get((mate["source_file"], mate["ply_index"] - 2 * turns))
            if item is None or item["side_to_move"] != mate["side_to_move"]:
                continue
            payload = dict(item)
            payload["derived_from_screened_mate"] = {
                "ply_index": mate["ply_index"],
                "mate_in_moves": mate["screening"]["mate_in_moves"],
                "lookback_turns": turns,
            }
            selected.setdefault(payload["fen"], payload)
    args.output.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in selected.values()),
        encoding="utf-8",
    )
    print(json.dumps({"input_mates": len(mates), "predecessors": len(selected)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
