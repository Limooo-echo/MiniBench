# MiniBench

[English](#english) | [中文](#中文)

## English

MiniBench is a small, reproducible benchmark for comparing LLM and
agent-style reasoning behavior. It supports multiple task families, including
multiple-choice questions, Xiangqi, one-stroke graph puzzles, static Mahjong
tile-shape tasks, and local four-player Riichi Mahjong tasks.

The codebase is organized so task families, agents, providers, and experiment
configuration stay separate. Adding a new task family should mostly mean adding
a new package under `src/minibench/datasets/`, a data directory under `data/`,
and one registry entry in `src/minibench/factory/experiments.py`.

### Quick Start

```bash
export PYTHONPATH=src
python -m unittest discover -s tests
```

Run the built-in oracle agent on the multiple-choice benchmark:

```bash
python -m minibench.cli evaluate --agent oracle
```

Run an experiment from YAML:

```bash
./run.sh config/experiments/multiple_choice.yaml
```

Inspect one multiple-choice prompt:

```bash
python -m minibench.cli show-prompt mb-choice-001
```

### Source Layout

```text
src/minibench/
  cli.py                 # command-line entrypoint
  evaluate.py            # YAML/config-driven evaluation runner
  core/                  # shared protocols, prompts, results, run helpers
  agents/                # agent reasoning strategies
  factory/               # agent/provider/config/experiment assembly
  datasets/              # task-family loaders, prompts, evaluators, engines
```

Task-family packages live only under `src/minibench/datasets/`:

```text
src/minibench/datasets/
  multiple_choice/
  xiangqi/
    engines/
  one_stroke/
  mahjong/
  mahjong_riichi/
```

### Data Layout

| Task family | Data file | Command |
| --- | --- | --- |
| Multiple choice | `data/multiple_choice/tasks.jsonl` | `evaluate` |
| Simple Xiangqi | `data/xiangqi/tasks.jsonl` | `evaluate-xiangqi` |
| Hard Xiangqi with Pikafish | `data/xiangqi/hard_tasks.jsonl` | `evaluate-xiangqi` |
| One-stroke graph puzzles | `data/one_stroke/tasks.jsonl` | `evaluate-one-stroke` |
| Static Mahjong tile shapes | `data/mahjong/tasks.jsonl` | `evaluate-mahjong` |
| Four-player Riichi Mahjong v1 | `data/mahjong_riichi/tasks.jsonl` | `evaluate-mahjong-riichi` |

### Agent Architectures

Available agent names:

- `oracle`: returns the gold answer for sanity checks.
- `noisy`: returns loose text for extraction checks.
- `openai-compatible`: direct OpenAI-compatible chat completion baseline.
- `direct`: asks the model to answer directly with the required JSON.
- `cot`: reason first, then finalize to JSON.
- `self-consistency`: sample several reasoning paths and majority vote.
- `tot`: generate candidate reasoning paths, then judge.
- `plan-then-solve`: plan first, solve from the plan, then finalize.
- `critic-refine`: draft, critique, then refine.

Reasoning architectures share these options:

```bash
--samples 3
--reasoning-temperature 0.7
--final-temperature 0.0
--max-reasoning-tokens 512
```

### Provider Examples

DeepSeek:

```bash
export DEEPSEEK_API_KEY="your_key_here"
python -m minibench.cli evaluate --agent cot --provider deepseek
```

Qwen/DashScope:

```bash
export DASHSCOPE_API_KEY="your_key_here"
python -m minibench.cli evaluate --agent self-consistency --provider qwen
```

Custom OpenAI-compatible endpoint:

```bash
export MY_MODEL_API_KEY="your_key_here"
python -m minibench.cli evaluate \
  --agent critic-refine \
  --provider generic \
  --model my-model \
  --base-url https://example.com/v1 \
  --api-key-env MY_MODEL_API_KEY
```

### Task Commands

Multiple choice:

```bash
python -m minibench.cli --tasks data/multiple_choice/tasks.jsonl evaluate \
  --agent openai-compatible \
  --provider deepseek \
  --model deepseek-chat \
  --json-mode \
  --max-tokens 256 \
  --timeout 120
```

Xiangqi:

```bash
python -m minibench.cli evaluate-xiangqi \
  --xiangqi-tasks data/xiangqi/tasks.jsonl \
  --agent cot \
  --provider deepseek \
  --model deepseek-chat \
  --json-mode \
  --max-tokens 256 \
  --timeout 120
```

Hard Xiangqi tasks can use Pikafish as the opponent:

```bash
export PIKAFISH_PATH=/path/to/Pikafish/src/pikafish

python -m minibench.cli evaluate-xiangqi \
  --xiangqi-tasks data/xiangqi/hard_tasks.jsonl \
  --agent openai-compatible \
  --provider deepseek \
  --model deepseek-chat \
  --json-mode \
  --max-tokens 256 \
  --pikafish-depth 8 \
  --timeout 120
```

One-stroke graph puzzles:

```bash
python -m minibench.cli evaluate-one-stroke \
  --one-stroke-tasks data/one_stroke/tasks.jsonl \
  --agent cot \
  --provider deepseek \
  --model deepseek-chat \
  --json-mode \
  --max-tokens 512 \
  --timeout 120
```

Static Mahjong tile-shape tasks:

```bash
python -m minibench.cli evaluate-mahjong \
  --mahjong-tasks data/mahjong/tasks.jsonl \
  --agent cot \
  --provider deepseek \
  --model deepseek-chat \
  --json-mode \
  --max-tokens 512 \
  --timeout 120
```

Four-player Riichi Mahjong:

```bash
python -m minibench.cli evaluate-mahjong-riichi \
  --mahjong-riichi-tasks data/mahjong_riichi/tasks.jsonl \
  --agent openai-compatible \
  --provider deepseek \
  --model deepseek-chat \
  --json-mode \
  --max-tokens 512 \
  --timeout 120
```

By default, seats 1/2/3 use the local shanten baseline bot. To connect external
Mahjong AIs, use `--riichi-opponent external` and provide a wrapper command.

### Adding A New Task Family

For a new family such as `sudoku`, add:

```text
src/minibench/datasets/sudoku/
  __init__.py
  dataset.py
  prompting.py
  evaluation.py

data/sudoku/tasks.jsonl
config/experiments/sudoku.yaml
```

Then register it in `src/minibench/factory/experiments.py` by adding a
`TaskFamilySpec` factory to `TASK_FAMILIES`.

### Output

Each evaluation writes a run directory under `runs/`:

- `predictions.jsonl`: raw outputs and per-instance results.
- `results.json`: aggregate metrics.
- `summary.txt`: short human-readable summary.

## 中文

MiniBench 是一个小型、可复现的 LLM/agent 推理评测项目。当前支持多种任务家族：
选择题、象棋、一笔画、静态麻将牌型，以及本地四人 Riichi Mahjong。

代码结构的原则是：任务、agent、provider、实验配置彼此分离。新增任务时，主要只需要
新增一个 `src/minibench/datasets/<family>/` 包、一份 `data/<family>/tasks.jsonl`
数据文件，以及在 `src/minibench/factory/experiments.py` 里注册一次。

### 快速开始

```bash
export PYTHONPATH=src
python -m unittest discover -s tests
python -m minibench.cli evaluate --agent oracle
./run.sh config/experiments/multiple_choice.yaml
```

### 目录结构

```text
src/minibench/
  cli.py                 # 命令行入口
  evaluate.py            # YAML 配置驱动的评测入口
  core/                  # 公共协议、prompt、结果和 run 辅助逻辑
  agents/                # agent 推理策略
  factory/               # agent/provider/config/experiment 装配
  datasets/              # 各任务家族的读取、prompt、评测和引擎
```

任务家族统一放在 `src/minibench/datasets/` 下，不再使用
`src/minibench/mahjong.py` 或 `src/minibench/xiangqi.py` 这类顶层转发文件。

### 数据文件

| 任务家族 | 数据文件 | 命令 |
| --- | --- | --- |
| 选择题 | `data/multiple_choice/tasks.jsonl` | `evaluate` |
| 简单象棋 | `data/xiangqi/tasks.jsonl` | `evaluate-xiangqi` |
| Pikafish 困难象棋 | `data/xiangqi/hard_tasks.jsonl` | `evaluate-xiangqi` |
| 一笔画 | `data/one_stroke/tasks.jsonl` | `evaluate-one-stroke` |
| 静态麻将牌型 | `data/mahjong/tasks.jsonl` | `evaluate-mahjong` |
| 四人 Riichi Mahjong | `data/mahjong_riichi/tasks.jsonl` | `evaluate-mahjong-riichi` |

### 新增任务

例如新增 `sudoku`：

```text
src/minibench/datasets/sudoku/
  __init__.py
  dataset.py
  prompting.py
  evaluation.py

data/sudoku/tasks.jsonl
config/experiments/sudoku.yaml
```

然后在 `src/minibench/factory/experiments.py` 的 `TASK_FAMILIES` 中注册即可。
