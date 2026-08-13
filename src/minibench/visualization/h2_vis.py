"""H2 评测可视化模块.

单组实验: 只生成 difficulty_comparison.png (short-mate vs long-mate).
多 agent 对照: 生成 full vs agent_only 的跨 agent 对比图.
"""
from __future__ import annotations

from pathlib import Path
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = [
    "WenQuanYi Zen Hei",
    "SimHei",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

DIFFS = ("short-mate", "long-mate")
DIFF_COLORS = {"short-mate": "#4C72B0", "long-mate": "#C44E52"}
AGENT_COLORS = [
    "#4C72B0", "#55A868", "#C44E52", "#8172B2",
    "#CCB974", "#64B5CD", "#E78AC3",
]
MODE_COLORS = {"full": "#4C72B0", "agent_only": "#C44E52"}


def visualize_h2_results(results, run_dir: str | Path) -> None:
    """Generate per-run visualizations into ``run_dir/visualizations/``."""
    vis_dir = Path(run_dir) / "visualizations"
    vis_dir.mkdir(exist_ok=True)
    _plot_difficulty_comparison(results, vis_dir)
    print(f"  Visualizations: {vis_dir}/")


def _plot_difficulty_comparison(results, vis_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    metrics = [
        ("Score", lambda r: r.normalized_score / 100, False),
        ("Success Rate", lambda r: float(r.success), False),
        ("Legality", lambda r: r.legality_rate, False),
    ]
    for ax_idx, (title, fn, _is_score) in enumerate(metrics):
        ax = axes[ax_idx]
        vals = []
        for diff in DIFFS:
            subset = [r for r in results if r.difficulty == diff]
            vals.append(
                statistics.mean(fn(r) for r in subset) if subset else 0.0
            )
        bars = ax.bar(DIFFS, vals, color=[DIFF_COLORS[d] for d in DIFFS])
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        for bar in bars:
            yval = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                yval + 0.02,
                f"{yval:.2f}",
                ha="center", va="bottom", fontsize=10,
            )
    mode = results[0].history_mode if results else ""
    plt.suptitle(f"H2 Metrics by Difficulty - {mode}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(vis_dir / "difficulty_comparison.png", dpi=200)
    plt.close()


# ---- Multi-agent full vs agent_only comparison ----

def visualize_h2_comparison(
    all_results: dict[str, dict[str, list]],
    output_dir: str | Path,
) -> None:
    """Generate cross-agent comparison charts.

    all_results: {agent_name: {mode: [H2Result, ...]}}
    """
    vis_dir = Path(output_dir) / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    _plot_mode_score_comparison(all_results, vis_dir)
    _plot_mode_success_comparison(all_results, vis_dir)
    _plot_mode_cp_comparison(all_results, vis_dir)
    print(f"\nComparison visualizations saved to {vis_dir}/")


def _agent_mode_values(
    all_results, field: str,
) -> tuple[list[str], dict[str, list[float]]]:
    agents = list(all_results.keys())
    values = {mode: [] for mode in ("full", "agent_only")}
    for agent in agents:
        for mode in ("full", "agent_only"):
            results = all_results.get(agent, {}).get(mode, [])
            if not results:
                values[mode].append(0.0)
                continue
            if field == "score":
                values[mode].append(
                    statistics.mean(r.normalized_score for r in results)
                )
            elif field == "success":
                values[mode].append(
                    statistics.mean(float(r.success) for r in results) * 100
                )
            elif field == "cp":
                cps = [
                    r.avg_cp_loss for r in results
                    if r.avg_cp_loss < 999999
                ]
                values[mode].append(
                    statistics.mean(cps) if cps else 0.0
                )
            elif field == "legality":
                values[mode].append(
                    statistics.mean(r.legality_rate for r in results) * 100
                )
    return agents, values


def _plot_grouped_bars(
    all_results, field: str, title: str, ylabel: str,
    ylim: tuple[float, float], vis_dir: Path, fname: str,
) -> None:
    agents, values = _agent_mode_values(all_results, field)
    x = np.arange(len(agents))
    width = 0.35

    all_vals = [v for vals in values.values() for v in vals]
    max_val = max(all_vals) if all_vals else 1.0
    top = ylim[1] if ylim[1] is not None else max_val * 1.15

    fig, ax = plt.subplots(figsize=(12, 5))
    for mode in ("full", "agent_only"):
        offset = -width / 2 if mode == "full" else width / 2
        bars = ax.bar(
            x + offset, values[mode], width,
            label=mode, color=MODE_COLORS[mode],
        )
        for bar in bars:
            yval = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                yval + top * 0.01,
                f"{yval:.0f}" if field in ("score", "cp") else f"{yval:.0f}%",
                ha="center", va="bottom", fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(agents, rotation=20, fontsize=9)
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
        "H2 Agent Comparison - Score (full vs agent_only)",
        "Score (0-100)", (0, 105), vis_dir, "mode_score_comparison.png",
    )


def _plot_mode_success_comparison(all_results, vis_dir: Path) -> None:
    _plot_grouped_bars(
        all_results, "success",
        "H2 Agent Comparison - Success Rate (full vs agent_only)",
        "Success Rate (%)", (0, 105), vis_dir, "mode_success_comparison.png",
    )


def _plot_mode_cp_comparison(all_results, vis_dir: Path) -> None:
    _plot_grouped_bars(
        all_results, "cp",
        "H2 Agent Comparison - Avg CP Loss (full vs agent_only)",
        "CP Loss", (0, None), vis_dir, "mode_cp_comparison.png",
    )
