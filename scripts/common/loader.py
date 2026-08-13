"""统一任务数据加载: 4 个任务的题库路径 + 抽样后的样本路径."""
from __future__ import annotations

import json
from pathlib import Path

# 项目根 (scripts/common/../..)
ROOT = Path(__file__).resolve().parent.parent.parent

# 任务 -> 250 题库路径
TASK_FILES: dict[str, str] = {
    "d3":     "data/d3/d3_250.jsonl",
    "c2":     "data/c2/c2_250.jsonl",
    "h2":     "data/h2/h2_250.jsonl",
    "m2": "data/m2/m2_250.jsonl",
}

# 旧评测用的小题集 (run_h2_eval 等引用)
LEGACY_FILES: dict[str, str] = {
    "c2": "data/xiangqi/c2_tasks.jsonl",
    "h2": "data/xiangqi/h2_tasks.jsonl",
}

TASK_NAMES = list(TASK_FILES)


def get_task_path(task: str, sample: str | int | None = None) -> Path:
    """返回题集路径. sample=None 全量; sample=int 用 sample_{n}.jsonl (不存在则按比例生成);
    sample=str 非数字 用自定义文件 sample_{name}.jsonl (不生成, 如同棋盘 sample_sb)."""
    if task not in TASK_FILES:
        raise ValueError(f"unknown task: {task}, available: {TASK_NAMES}")
    if sample is None:
        return ROOT / TASK_FILES[task]
    if isinstance(sample, str) and not sample.isdigit():
        return ROOT / f"data/{task}/sample_{sample}.jsonl"   # 自定义文件, 不自动生成
    seed = sample if isinstance(sample, int) else int(sample)
    path = ROOT / f"data/{task}/sample_{seed}.jsonl"
    if not path.exists():
        from scripts.common.sample import sample_tasks
        sample_tasks(task, seed=seed)
    return path


def load_tasks(task: str, *, sample: str | int | None = None) -> list[dict]:
    """加载任务题集 (返回 list[dict]).

    task: d3/c2/h2/vision
    sample: None=全量250题; int=用sample_{n}.jsonl
    """
    path = get_task_path(task, sample)
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
