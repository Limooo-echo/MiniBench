"""Build the paired M2 corpus from Pikafish-verified D3 mate-in-one positions.

M2 changes only the input representation.  Reusing exactly the D3 positions
across text and the two board renderings prevents position difficulty from
being confounded with modality.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def build(records: list[dict]) -> list[dict]:
    output = []
    for index, source in enumerate(records, start=1):
        analysis = source.get("d3_analysis", {})
        mate_moves = analysis.get("mate_moves_uci")
        if not isinstance(mate_moves, list) or not mate_moves:
            raise ValueError(f"{source.get('id')}: missing verified D3 mate moves")
        record = {
            "schema_version": 2,
            "id": f"xiangqi-multimodal-m2-{index:04d}",
            "family": "xiangqi-multimodal",
            "fen": source["fen"],
            "agent_color": source["agent_color"],
            "goal": "checkmate",
            "max_plies": 1,
            "difficulty": source["difficulty"],
            "piece_count": source["piece_count"],
            "oracle": dict(source["oracle"]),
            "tags": sorted({
                "m2", "mate-in-one", "paired-modalities",
                f"difficulty:{source['difficulty']}",
            }),
            "source_task_id": source["id"],
            "m2_analysis": {
                "version": "m2-paired-mate-in-one-v1",
                "mate_moves_uci": sorted(set(mate_moves)),
                "mate_move_count": len(set(mate_moves)),
                "verification": "exact legality/checkmate plus Pikafish oracle",
            },
        }
        output.append(record)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d3", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = [
        json.loads(line)
        for line in args.d3.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    output = build(records)
    args.output.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in output),
        encoding="utf-8",
    )
    print(json.dumps({
        "total": len(output),
        "difficulty": {
            label: sum(record["difficulty"] == label for record in output)
            for label in ("easy", "medium", "hard")
        },
        "unique_fens": len({record["fen"] for record in output}),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
