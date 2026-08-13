"""抽题接口: 从 250 题库按规则抽样.

d3: 每难度(easy/medium/hard)各抽 N 题 -> 3N 题
c2: 按 group 比例抽 N 题
h2/vision: 随机抽 N 题
"""
from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

FILES = {
    "d3": "data/d3/d3_250.jsonl",
    "c2": "data/c2/c2_250.jsonl",
    "h2": "data/h2/h2_250.jsonl",
    "m2": "data/m2/m2_250.jsonl",
}


def _load(p: Path) -> list[dict]:
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _sample_d3(tasks, rng, n=10):
    picked = []
    for diff in ("easy", "medium", "hard"):
        pool = [t for t in tasks if f"difficulty:{diff}" in t["tags"]]
        picked.extend(rng.sample(pool, min(n, len(pool))))
    return picked


def _sample_c2(tasks, rng, n=10):
    groups = {}
    for t in tasks:
        groups.setdefault(t["group"], []).append(t)
    picked, total = [], len(tasks)
    for g, pool in groups.items():
        quota = max(1, round(n * len(pool) / total))
        picked.extend(rng.sample(pool, min(quota, len(pool))))
    if len(picked) > n:
        picked = rng.sample(picked, n)
    else:
        rest = [t for t in tasks if t not in picked]
        picked.extend(rng.sample(rest, n - len(picked)))
    return picked


def _sample_plain(tasks, rng, n=10):
    return rng.sample(tasks, min(n, len(tasks)))


def sample_tasks(task: str, *, seed: int = 42, n: int = 10, out: Path | None = None) -> Path:
    """抽样并写出. 返回输出路径."""
    if task not in FILES:
        raise ValueError(f"unknown task: {task}")
    tasks = _load(ROOT / FILES[task])
    rng = random.Random(seed)
    if task == "d3":
        picked = _sample_d3(tasks, rng, n)
    elif task == "c2":
        picked = _sample_c2(tasks, rng, n)
    else:
        picked = _sample_plain(tasks, rng, n)

    out_path = out or (ROOT / f"data/{task}/sample_{seed}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for t in picked:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"[sample] {task} seed={seed}: {len(picked)} 题 -> {out_path}")
    return out_path


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(FILES))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=10)
    args = ap.parse_args()
    sample_tasks(args.task, seed=args.seed, n=args.n)
