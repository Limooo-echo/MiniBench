"""Xiangqi history evaluation visualizations.

单组实验: 只生成 difficulty_comparison.png (short / medium / long).
Multi-agent comparison supports full-state and move-history-only modes.
"""
from __future__ import annotations

from pathlib import Path
from math import isfinite

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from minibench.assets.fonts import configure_matplotlib
from minibench.visualization.xiangqi_metrics import metric_mean

DIFFS = ("short", "medium", "long")
DIFF_COLORS = {"short": "#4C72B0", "medium": "#55A868", "long": "#C44E52"}
AGENT_COLORS = [
    "#4C72B0", "#55A868", "#C44E52", "#8172B2",
    "#CCB974", "#64B5CD", "#E78AC3",
]
MODE_COLORS = {"full-state": "#4C72B0", "move-history-only": "#C44E52"}


def visualize_history_results(results, run_dir: str | Path) -> None:
    """Generate per-run visualizations into ``run_dir/visualizations/``."""
    vis_dir = Path(run_dir) / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    with matplotlib.rc_context():
        configure_matplotlib()
        _plot_difficulty_comparison(results, vis_dir)
    print(f"  Visualizations: {vis_dir}/")


def _plot_difficulty_comparison(results, vis_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    metrics = [
        ("Diagnostic Score", "normalized_score", 0.01),
        ("Strict Checkmate Success", "success", 1.0),
        ("Legality", "legality_rate", 1.0),
    ]
    for ax_idx, (title, field, scale) in enumerate(metrics):
        ax = axes[ax_idx]
        vals = []
        for diff in DIFFS:
            subset = [r for r in results if r.difficulty == diff]
            vals.append(metric_mean(subset, field) * scale)
        bars = ax.bar(DIFFS, vals, color=[DIFF_COLORS[d] for d in DIFFS])
        ax.set_xlim(-0.5, len(DIFFS) - 0.5)
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        for bar in bars:
            yval = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                yval + 0.02 if isfinite(yval) else 0.02,
                f"{yval:.2f}" if isfinite(yval) else "missing",
                ha="center", va="bottom", fontsize=10,
            )
    modes = sorted({result.history_mode for result in results})
    mode = " / ".join(modes)
    plt.suptitle(f"History Metrics by Difficulty - {mode}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(vis_dir / "difficulty_comparison.png", dpi=200)
    plt.close()


# ---- Multi-agent history-mode comparison ----

def visualize_history_comparison(
    all_results: dict[str, dict[str, list]],
    output_dir: str | Path,
) -> None:
    """Generate cross-agent comparison charts.

    all_results: {agent_name: {mode: [HistoryResult, ...]}}
    """
    vis_dir = Path(output_dir) / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    with matplotlib.rc_context():
        configure_matplotlib()
        _plot_mode_score_comparison(all_results, vis_dir)
        _plot_mode_success_comparison(all_results, vis_dir)
        _plot_mode_cp_comparison(all_results, vis_dir)
    print(f"\nComparison visualizations saved to {vis_dir}/")


def _agent_mode_values(
    all_results, field: str,
) -> tuple[list[str], dict[str, list[float]]]:
    agents = list(all_results.keys())
    values = {mode: [] for mode in ("full-state", "move-history-only")}
    for agent in agents:
        for mode in ("full-state", "move-history-only"):
            results = all_results.get(agent, {}).get(mode, [])
            fields = {"score": "normalized_score", "success": "success",
                      "cp": "avg_cp_loss", "legality": "legality_rate"}
            value = metric_mean(results, fields[field])
            values[mode].append(value * (100 if field in {"success", "legality"} else 1))
    return agents, values


def _plot_grouped_bars(
    all_results, field: str, title: str, ylabel: str,
    ylim: tuple[float, float], vis_dir: Path, fname: str,
) -> None:
    agents, values = _agent_mode_values(all_results, field)
    x = np.arange(len(agents))
    width = 0.35

    all_vals = [v for vals in values.values() for v in vals if isfinite(v)]
    max_val = max(all_vals, default=1.0)
    top = ylim[1] if ylim[1] is not None else max(1.0, max_val * 1.15)

    fig, ax = plt.subplots(figsize=(12, 5))
    for mode in ("full-state", "move-history-only"):
        offset = -width / 2 if mode == "full-state" else width / 2
        bars = ax.bar(
            x + offset, values[mode], width,
            label=mode, color=MODE_COLORS[mode],
        )
        for bar in bars:
            yval = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                yval + top * 0.01 if isfinite(yval) else top * 0.01,
                ("missing" if not isfinite(yval) else
                 f"{yval:.0f}" if field in ("score", "cp") else f"{yval:.0f}%"),
                ha="center", va="bottom", fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(agents, rotation=20, fontsize=9)
    ax.set_xlim(-0.75, max(1, len(agents)) - 0.25)
    ax.set_ylim(0, top)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(vis_dir / fname, dpi=200)
    plt.close()


def _plot_mode_score_comparison(all_results, vis_dir: Path) -> None:
    _plot_grouped_bars(
        all_results, "score",
        "History Agent Comparison - Diagnostic Score",
        "Diagnostic Score (0-100)", (0, 105), vis_dir, "mode_score_comparison.png",
    )


def _plot_mode_success_comparison(all_results, vis_dir: Path) -> None:
    _plot_grouped_bars(
        all_results, "success",
        "History Agent Comparison - Success Rate",
        "Success Rate (%)", (0, 105), vis_dir, "mode_success_comparison.png",
    )


def _plot_mode_cp_comparison(all_results, vis_dir: Path) -> None:
    _plot_grouped_bars(
        all_results, "cp",
        "History Agent Comparison - Avg CP Loss",
        "CP Loss", (0, None), vis_dir, "mode_cp_comparison.png",
    )
