"""D3 评测可视化模块.

生成 6 张图表:
  1. per_task_scores.png       — 逐任务得分柱状图
  2. difficulty_comparison.png  — 难度对比 (4 维指标)
  3. score_distribution.png    — 得分分布直方图
  4. cp_loss_scatter.png       — 局面分差散点图
  5. radar_chart.png           — 雷达图 (合法性/正确性/质量/得分)
  6. score_heatmap.png         — 热力图 (难度 × 题号)
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

DIFFS = ("easy", "medium", "hard")
DIFF_COLORS = {"easy": "#4C72B0", "medium": "#55A868", "hard": "#C44E52"}


def visualize_d3_results(results, run_dir: str | Path) -> None:
    """Generate all D3 visualizations into ``run_dir/visualizations/``."""
    vis_dir = Path(run_dir) / "visualizations"
    vis_dir.mkdir(exist_ok=True)

    _plot_per_task_scores(results, vis_dir)
    _plot_difficulty_comparison(results, vis_dir)
    _plot_score_distribution(results, vis_dir)
    _plot_cp_loss_scatter(results, vis_dir)
    _plot_radar_chart(results, vis_dir)
    _plot_score_heatmap(results, vis_dir)

    print(f"\nVisualizations saved to {vis_dir}/")


def _plot_per_task_scores(results, vis_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    task_ids = [r.task_id.split("-")[-1] for r in results]
    scores = [r.normalized_score for r in results]
    colors = [DIFF_COLORS.get(r.difficulty, "#888") for r in results]
    ax.bar(range(len(scores)), scores, color=colors, width=0.8)
    ax.set_xticks(range(len(task_ids)))
    ax.set_xticklabels(task_ids, rotation=60, fontsize=7)
    ax.set_ylabel("Normalized Score (0-100)")
    ax.set_title("D3 Per-Task Scores")
    ax.axhline(y=70, color="gray", linestyle="--", alpha=0.5, label="70 (pass)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(vis_dir / "per_task_scores.png", dpi=200)
    plt.close()


def _plot_difficulty_comparison(results, vis_dir: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    metrics = [
        ("Score", "normalized_score", True),
        ("Legality", "legality_score", False),
        ("Correctness", "correctness_score", False),
        ("Quality", "quality_score", False),
    ]
    for ax_idx, (title, field, is_score) in enumerate(metrics):
        ax = axes[ax_idx]
        vals = []
        for diff in DIFFS:
            subset = [r for r in results if r.difficulty == diff]
            if subset:
                if is_score:
                    vals.append(
                        statistics.mean(getattr(r, field) for r in subset) / 100
                    )
                else:
                    vals.append(
                        statistics.mean(getattr(r, field) for r in subset)
                    )
            else:
                vals.append(0)
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
                ha="center",
                va="bottom",
                fontsize=10,
            )
    plt.suptitle("D3 Metrics by Difficulty", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(vis_dir / "difficulty_comparison.png", dpi=200)
    plt.close()


def _plot_score_distribution(results, vis_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for diff in DIFFS:
        subset = [r.normalized_score for r in results if r.difficulty == diff]
        if subset:
            ax.hist(
                subset,
                bins=range(0, 110, 10),
                alpha=0.6,
                color=DIFF_COLORS[diff],
                label=diff,
                edgecolor="black",
            )
    ax.set_xlabel("Score (0-100)")
    ax.set_ylabel("Count")
    ax.set_title("D3 Score Distribution")
    ax.legend()
    plt.tight_layout()
    plt.savefig(vis_dir / "score_distribution.png", dpi=200)
    plt.close()


def _plot_cp_loss_scatter(results, vis_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    for diff in DIFFS:
        subset = [
            (i, r.cp_loss)
            for i, r in enumerate(results)
            if r.difficulty == diff
        ]
        if subset:
            xs, ys = zip(*subset)
            ax.scatter(
                xs, ys, color=DIFF_COLORS[diff], label=diff, alpha=0.7, s=40
            )
    ax.set_xlabel("Task Index")
    ax.set_ylabel("CP Loss")
    ax.set_title("D3 Position Evaluation Loss (cp)")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(vis_dir / "cp_loss_scatter.png", dpi=200)
    plt.close()


def _plot_radar_chart(results, vis_dir: Path) -> None:
    categories = ["Legality", "Correctness", "Quality", "Score"]
    n = len(categories)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for diff in DIFFS:
        subset = [r for r in results if r.difficulty == diff]
        if not subset:
            continue
        vals = [
            statistics.mean(r.legality_score for r in subset),
            statistics.mean(r.correctness_score for r in subset),
            statistics.mean(r.quality_score for r in subset),
            statistics.mean(r.normalized_score for r in subset) / 100,
        ]
        vals += vals[:1]
        ax.plot(
            angles, vals, "o-", linewidth=2, label=diff,
            color=DIFF_COLORS[diff],
        )
        ax.fill(angles, vals, alpha=0.15, color=DIFF_COLORS[diff])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_title(
        "D3 Radar Chart by Difficulty", fontsize=14, fontweight="bold", pad=20
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig(vis_dir / "radar_chart.png", dpi=200)
    plt.close()


def _plot_score_heatmap(results, vis_dir: Path) -> None:
    matrix = []
    for diff in DIFFS:
        subset = [
            r.normalized_score for r in results if r.difficulty == diff
        ]
        matrix.append(subset if subset else [0])

    max_cols = max(len(row) for row in matrix)
    for row in matrix:
        while len(row) < max_cols:
            row.append(0)

    fig, ax = plt.subplots(figsize=(14, 4))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100)
    ax.set_xticks(range(max_cols))
    ax.set_xticklabels([f"{i+1}" for i in range(max_cols)], fontsize=9)
    ax.set_yticks(range(len(DIFFS)))
    ax.set_yticklabels(DIFFS, fontsize=11)
    ax.set_title("D3 Score Heatmap", fontsize=14, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Score")
    for i, row in enumerate(matrix):
        for j, val in enumerate(row):
            if val > 0:
                ax.text(
                    j, i, f"{val:.0f}", ha="center", va="center",
                    fontsize=8,
                    color="white" if val < 50 else "black",
                )
    plt.tight_layout()
    plt.savefig(vis_dir / "score_heatmap.png", dpi=200)
    plt.close()


# ---- Multi-agent comparison ----

AGENT_COLORS = [
    "#4C72B0", "#55A868", "#C44E52", "#8172B2",
    "#CCB974", "#64B5CD", "#E78AC3", "#A6CE39",
]


def visualize_d3_comparison(
    all_results: dict[str, list],
    output_dir: str | Path,
) -> None:
    """Generate cross-agent comparison chart (score by difficulty)."""
    vis_dir = Path(output_dir) / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    _plot_agent_difficulty_comparison(all_results, vis_dir)
    print(f"\nComparison visualization saved to {vis_dir}/")


def _plot_agent_difficulty_comparison(all_results, vis_dir: Path) -> None:
    """Grouped bar chart: score by difficulty per agent."""
    agents = list(all_results.keys())
    x = np.arange(len(DIFFS))
    width = 0.8 / len(agents)

    fig, ax = plt.subplots(figsize=(10, 5))
    for idx, (agent_name, results) in enumerate(all_results.items()):
        vals = []
        for diff in DIFFS:
            subset = [r for r in results if r.difficulty == diff]
            if subset:
                vals.append(statistics.mean(r.normalized_score for r in subset))
            else:
                vals.append(0)
        offset = (idx - len(agents) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=agent_name,
               color=AGENT_COLORS[idx % len(AGENT_COLORS)])

    ax.set_xticks(x)
    ax.set_xticklabels(DIFFS, fontsize=11)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Score (0-100)")
    ax.set_title("D3 Agent Comparison - Score by Difficulty")
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(vis_dir / "agent_difficulty_comparison.png", dpi=200)
    plt.close()
