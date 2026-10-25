"""Task/condition/metric comparisons for the resumable Xiangqi suite.

Missing results are displayed as N/A, never silently converted to zero.
H2 CP and C2 rule-aware CPE always use separate charts and labels.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Bundled CJK fonts are chess-glyph subsets, not general plot fonts. Use ASCII
# labels and Matplotlib's bundled DejaVu on headless servers; board fonts stay
# unchanged. The Chinese metric descriptions live in docs/cloud-run.md.
NAMES = {"passthrough": "Passthrough", "direct": "Direct", "cot": "CoT",
         "tot": "ToT", "best-of-n": "Best-of-N", "least-to-most": "Least-to-Most",
         "plan-then-solve": "Plan-then-Solve", "critic-refine": "Critic-Refine",
         "self-consistency": "Self-Consistency"}
METRICS = {
    "mate_rate": ("Checkmate rate (bounded run)", True),
    "mate_within_shortest_rate": ("Checkmate within certified shortest bound", True),
    "legality_rate": ("Legal move rate (mean over tasks)", True),
    "recognition_accuracy": ("Board recognition accuracy (text: transcription control)", True),
    "mate_within_prescribed_rate": ("Checkmate rate within reference horizon", True),
    "avg_cp_loss": ("Mean Pikafish CP loss (lower is better)", False),
    "avg_variant_loss": ("Mean rule-aware CPE loss, NOT Pikafish CP (lower is better)", False),
}


def comparison_rows(suite: Path) -> list[dict]:
    plan = json.loads((suite / "plan.json").read_text(encoding="utf-8"))
    rows = []
    entries = plan.get("experiments", plan.get("tasks", {}))
    for key, entry in entries.items():
        family = entry.get("family") or {
            "d3": "xiangqi-mate-in-one", "h2": "xiangqi-history",
            "c2": "xiangqi-rule-variants", "m2": "xiangqi-multimodal",
        }.get(key, "")
        if not family.startswith("xiangqi-"):
            continue
        native = "native" if "experiments" in plan else "single"
        counts = entry.get("condition_counts") or {
            mode: entry["record_count"] for mode in entry["modes"]
        }
        for name in plan["agents"]:
            directory = suite / "results" / key / name
            path = directory / "results.json"
            summary = json.loads(path.read_text()) if path.exists() else {}
            if family == "xiangqi-history":
                buckets = summary.get("by_mode", {})
            elif family == "xiangqi-multimodal":
                buckets = summary.get("by_input_mode", {})
            elif family == "xiangqi-rule-variants" and "experiments" in plan:
                buckets = {"all": summary, **summary.get("by_ruleset", {})}
            else:
                buckets = {native: summary} if summary else {}
            for mode, planned in counts.items():
                bucket = buckets.get(mode, {})
                metrics = ["mate_rate", "legality_rate"]
                if family in ("xiangqi-history", "xiangqi-rule-variants"):
                    metrics = ["mate_within_shortest_rate", "legality_rate"]
                    metrics += ["avg_cp_loss" if family == "xiangqi-history" else "avg_variant_loss"]
                if family == "xiangqi-multimodal":
                    metrics += ["recognition_accuracy"]
                completed = 0
                predictions = directory / "predictions.jsonl"
                if predictions.exists():
                    for line in predictions.read_text().splitlines():
                        if not line.strip():
                            continue
                        record = json.loads(line)
                        if family == "xiangqi-rule-variants" and "experiments" in plan:
                            matches = mode == "all" or record.get("ruleset") == mode
                        else:
                            record_mode = record.get("history_mode") or record.get("input_mode") or native
                            matches = record_mode == mode
                        completed += int(matches)
                for metric in metrics:
                    rows.append({
                        "task": key, "mode": mode, "metric": metric, "agent": name,
                        "value": bucket.get(metric), "completed": completed,
                        "planned": planned,
                    })
    return rows


def plot_suite(suite: str | Path) -> Path:
    with matplotlib.rc_context({"font.family": "DejaVu Sans", "axes.unicode_minus": False}):
        return _plot_suite(suite)


def _plot_suite(suite: str | Path) -> Path:
    suite = Path(suite)
    output = suite / "charts"
    output.mkdir(parents=True, exist_ok=True)
    rows = comparison_rows(suite)
    # JSON is the lossless table; an empty CSV value means unavailable, not zero.
    from minibench.core.checkpoint import atomic_write_json, atomic_write_text
    import io
    atomic_write_json(output / "comparison.json", rows)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=["task", "mode", "metric", "agent", "value", "completed", "planned"])
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(output / "comparison.csv", buffer.getvalue())
    groups = {}
    for row in rows:
        groups.setdefault((row["task"], row["mode"], row["metric"]), []).append(row)
    for (task, mode, metric), values in groups.items():
        title, percent = METRICS[metric]
        fig, ax = plt.subplots(figsize=(15, 6))
        positions = list(range(len(values)))
        colors = plt.get_cmap("tab10").colors
        for index, row in enumerate(values):
            value = row["value"]
            if value is None:
                ax.text(index, 0, "N/A", ha="center", va="bottom", color="#777")
                continue
            displayed = float(value) * (100 if percent else 1)
            ax.bar(index, displayed, color=colors[index % len(colors)], width=.65)
            ax.annotate(f"{displayed:.1f}" + ("%" if percent else ""),
                        (index, displayed), ha="center", va="bottom",
                        xytext=(0, 4), textcoords="offset points", fontsize=9)
        labels = [NAMES.get(row["agent"], row["agent"]) +
                  f"\n{row['completed']}/{row['planned']}" for row in values]
        ax.set_xticks(positions, labels, rotation=20, ha="right")
        ax.set_xlim(-.6, len(values) - .4)
        ax.set_ylabel(title)
        ax.set_title(f"{task.upper()} / {mode} - {title}")
        if percent:
            ax.set_ylim(0, 112)
        else:
            ax.set_ylim(bottom=0)
            ax.margins(y=.2)
        ax.grid(axis="y", linestyle="--", alpha=.3)
        ax.set_axisbelow(True)
        fig.text(.01, .01, "Below each bar: completed/planned. Incomplete results are not final rankings.",
                 fontsize=9, color="#555")
        fig.tight_layout(rect=(0, .04, 1, 1))
        stem = output / f"{task}-{mode}-{metric}"
        fig.savefig(stem.with_suffix(".png"), dpi=160)
        fig.savefig(stem.with_suffix(".svg"))
        plt.close(fig)
    return output
