"""Turn the verified 100-task Zebra matrix into a compact six-group scorecard."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


MODES = ("deferred_reasoning", "incremental_state", "incremental_state_pro")
TREATMENTS = ("thinking_off", "thinking_on")
SHORT = {"deferred_reasoning": "deferred",
         "incremental_state": "structured",
         "incremental_state_pro": "pro"}


def status(row: dict[str, str], treatment: str, mode: str) -> str:
    prefix = f"{treatment}__{mode}__"
    if row[prefix + "success"] == "True":
        return "OK"
    if row[prefix + "no_answer"] == "True":
        return "CUT" if row[prefix + "final_finish_reason"] == "length" else "NO_ANSWER"
    return "WRONG"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis_dir", type=Path)
    args = parser.parse_args()
    folder = args.analysis_dir.resolve()
    with (folder / "paired_task_matrix.csv").open(encoding="utf-8-sig", newline="") as handle:
        source = list(csv.DictReader(handle))
    if len(source) != 100 or len({row["source_id"] for row in source}) != 100:
        raise ValueError("expected 100 unique puzzle tasks")
    columns = ["source_id", "size", "size_group", "clues", "total_cells"]
    columns += [f"{treatment}_{SHORT[mode]}" for treatment in TREATMENTS for mode in MODES]
    columns += ["correct_of_6", "off_correct_of_3", "on_correct_of_3"]
    output = []
    for row in source:
        item = {name: row[name] for name in columns[:5]}
        for treatment in TREATMENTS:
            for mode in MODES:
                item[f"{treatment}_{SHORT[mode]}"] = status(row, treatment, mode)
        item["off_correct_of_3"] = sum(item[f"thinking_off_{SHORT[mode]}"] == "OK"
                                        for mode in MODES)
        item["on_correct_of_3"] = sum(item[f"thinking_on_{SHORT[mode]}"] == "OK"
                                       for mode in MODES)
        item["correct_of_6"] = item["off_correct_of_3"] + item["on_correct_of_3"]
        output.append(item)
    with (folder / "task_scorecard.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(output)


if __name__ == "__main__":
    main()
