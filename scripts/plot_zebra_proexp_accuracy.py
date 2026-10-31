"""Plot the six verified groups in the 1028 Zebra server experiment."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MODES = (
    ("deferred_reasoning", "Deferred"),
    ("incremental_state", "Structured"),
    ("incremental_state_pro", "Structured Pro"),
)
TREATMENTS = (("thinking_off", "Thinking off", "#436E9A"),
              ("thinking_on", "Thinking on", "#C3A04D"))


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis_dir", type=Path)
    args = parser.parse_args()
    folder = args.analysis_dir.resolve()
    overall = {(r["treatment"], r["mode"]): r
               for r in read_rows(folder / "group_accuracy.csv")}
    hard = {(r["treatment"], r["mode"]): r
            for r in read_rows(folder / "difficulty_accuracy.csv")
            if r["category_type"] == "difficulty" and r["category"] == "hard"}
    assert len(overall) == len(hard) == 6
    assert {int(r["n"]) for r in overall.values()} == {100}
    assert {int(r["n"]) for r in hard.values()} == {33}

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, axes = plt.subplots(2, 1, figsize=(10.6, 7.2), sharex=True,
                             constrained_layout=True)
    x = np.arange(len(MODES))
    width = 0.32
    for ax, rows, title in zip(
        axes,
        (overall, hard),
        ("All 100 puzzles", "Hard subset: 33 puzzles"),
    ):
        for offset, (treatment, label, color) in zip((-width / 2, width / 2), TREATMENTS):
            values = [100 * int(rows[treatment, mode]["success"]) /
                      int(rows[treatment, mode]["n"]) for mode, _ in MODES]
            bars = ax.bar(x + offset, values, width, color=color, label=label,
                          zorder=3)
            for bar, row in zip(bars, (rows[treatment, mode] for mode, _ in MODES)):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                        f'{row["success"]}/{row["n"]}', ha="center", va="bottom",
                        fontsize=10, color="#202C39")
        ax.set(title=title, ylabel="Exact puzzle accuracy (%)", ylim=(0, 110))
        ax.set_yticks((0, 25, 50, 75, 100))
        ax.set_yticklabels(("0%", "25%", "50%", "75%", "100%"))
        ax.grid(axis="y", color="#DDE3E9", linewidth=0.8, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#8C98A5")
    axes[0].legend(loc="upper left", ncol=2, frameon=False)
    axes[1].set_xticks(x, [label for _, label in MODES])
    fig.suptitle("Zebra history-memory accuracy | Direct · Qwen3.8-Flash",
                 fontsize=15, fontweight="bold", color="#202C39")
    fig.savefig(folder / "accuracy_by_mode.png", dpi=210,
                facecolor="white", bbox_inches="tight")


if __name__ == "__main__":
    main()
