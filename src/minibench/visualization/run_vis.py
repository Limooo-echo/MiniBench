"""可视化入口 (参数化).

用法:
  python -m minibench.visualization.run_vis --type xiangqi --id c2-0001 --task c2
  python -m minibench.visualization.run_vis --type one_stroke --id os-001
  python -m minibench.visualization.run_vis --type accuracy --runs runs
  python -m minibench.visualization.run_vis --type tasks --runs runs   # 4任务结果对比图

类型: one_stroke / xiangqi / mahjong / accuracy / tasks
"""
import argparse
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)
sys.path.insert(0, os.path.abspath(os.path.join(CURRENT_DIR, "../../..", "src")))

import onestroke_vis
import xiangqi_vis
import mahjong_vis
import plot_results

PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../.."))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "visualizations")

DATA_PATHS = {
    "one_stroke": os.path.join(PROJECT_ROOT, "data/one_stroke/tasks.jsonl"),
    "mahjong": os.path.join(PROJECT_ROOT, "data/mahjong/tasks.jsonl"),
    # 象棋 4 任务题库 (棋盘可视化)
    "d3": os.path.join(PROJECT_ROOT, "data/d3/d3_250.jsonl"),
    "c2": os.path.join(PROJECT_ROOT, "data/c2/c2_250.jsonl"),
    "h2": os.path.join(PROJECT_ROOT, "data/h2/h2_250.jsonl"),
    "m2": os.path.join(PROJECT_ROOT, "data/m2/m2_250.jsonl"),
}


def main():
    ap = argparse.ArgumentParser(description="MiniBench 可视化入口")
    ap.add_argument("--type", required=True,
                    choices=["one_stroke", "xiangqi", "mahjong", "accuracy", "tasks"])
    ap.add_argument("--id", default="", help="题目 ID (棋盘类可视化)")
    ap.add_argument("--task", default="c2", choices=["d3", "c2", "h2", "m2"],
                    help="象棋 4 任务之一 (--type xiangqi 时用)")
    ap.add_argument("--runs", default=os.path.join(PROJECT_ROOT, "runs"),
                    help="runs 目录 (结果对比可视化)")
    args = ap.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"可视化: type={args.type} id={args.id or '-'} task={args.task}")

    if args.type == "accuracy":
        save = os.path.join(OUTPUT_DIR, "accuracy_comparison.png")
        plot_results.plot_accuracy_comparison(args.runs, save)
        print(f"已生成: {save}")
    elif args.type == "tasks":
        # 4 任务结果对比 (读 runs/ 下各任务结果目录)
        save = os.path.join(OUTPUT_DIR, "task_results_comparison.png")
        plot_results.plot_task_results(args.runs, save)
        print(f"已生成: {save}")
    elif args.type == "xiangqi":
        path = DATA_PATHS[args.task]
        xiangqi_vis.visualize_xiangqi_by_id(args.id, path, OUTPUT_DIR)
    elif args.type == "one_stroke":
        onestroke_vis.visualize_onestroke_by_id(args.id, DATA_PATHS["one_stroke"], OUTPUT_DIR)
    elif args.type == "mahjong":
        mahjong_vis.visualize_mahjong_by_id(args.id, DATA_PATHS["mahjong"], OUTPUT_DIR)


if __name__ == "__main__":
    main()
