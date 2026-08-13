# MiniBench

[English](#english) | [中文](#中文)

## English

MiniBench is a small, reproducible benchmark for comparing LLM and
agent-style reasoning behavior. It supports multiple task families, including
Zebra logic grids, Xiangqi, one-stroke graph puzzles, static Mahjong tile-shape
tasks, and local four-player Riichi Mahjong tasks.

The codebase is organized so task families, agents, providers, and experiment
configuration stay separate. Adding a new task family should mostly mean adding
a new package under `src/minibench/datasets/`, a data directory under `data/`,
and one registry entry in `src/minibench/factory/experiments.py`.

### Quick Start

```bash
export PYTHONPATH=src
python -m unittest discover -s tests
```

Run the three-task Zebra smoke set with an OpenAI-compatible provider:

```bash
python -m minibench.cli evaluate-zebra --provider deepseek
```

Run an experiment from YAML:

```bash
./run.sh config/experiments/zebra.yaml
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
  zebra/
  xiangqi/
    engines/
  one_stroke/
  mahjong/
  mahjong_solo/
  mahjong_riichi/
```

### Data Layout

| Task family | Data file | Command |
| --- | --- | --- |
| Zebra smoke set | `data/zebra/tasks.jsonl` | `evaluate-zebra` |
| Zebra formal evaluation (15 per difficulty) | `data/zebra/eval.jsonl` | YAML config or `--zebra-tasks` |
| Zebra temporary-rule evaluation | `data/zebra/rule_codebook_eval.jsonl` | `zebra_rule_codebook.yaml` |
| Zebra history evaluation | `data/zebra/history_eval.jsonl` | `zebra_history.yaml` |
| Simple Xiangqi | `data/xiangqi/tasks.jsonl` | `evaluate-xiangqi` |
| Hard Xiangqi with Pikafish | `data/xiangqi/hard_tasks.jsonl` | `evaluate-xiangqi` |
| One-stroke smoke set | `data/one_stroke/tasks.jsonl` | `evaluate-one-stroke` |
| One-stroke A1 direct (10 per difficulty) | `data/one_stroke/a1_direct.jsonl` | `one_stroke_a1.yaml` |
| One-stroke A2 temporary rules (10 per difficulty) | `data/one_stroke/a2_rule_condition.jsonl` | `one_stroke_a2.yaml` |
| One-stroke A3 history (10 per difficulty) | `data/one_stroke/a3_history.jsonl` | `one_stroke_a3_history.yaml` |
| Static Mahjong tile shapes | `data/mahjong/tasks.jsonl` | `evaluate-mahjong` |
| Single-player Riichi Mahjong draw-discard | `data/mahjong_solo/tasks.jsonl` | `evaluate-mahjong-solo` |
| Four-player Riichi Mahjong v1 | `data/mahjong_riichi/tasks.jsonl` | `evaluate-mahjong-riichi` |

### Agent Architectures

Available agent names:

- `openai-compatible`: direct OpenAI-compatible chat completion baseline.
- `direct`: asks the model to answer directly with the required JSON.
- `cot`: reason first, then finalize to JSON.
- `self-consistency`: sample several reasoning paths and ask a judge to select.
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
python -m minibench.cli evaluate-zebra --agent cot --provider deepseek
```

Qwen/DashScope:

```bash
export DASHSCOPE_API_KEY="your_key_here"
python -m minibench.cli evaluate-zebra --agent self-consistency --provider qwen
```

Custom OpenAI-compatible endpoint:

```bash
export MY_MODEL_API_KEY="your_key_here"
python -m minibench.cli evaluate-zebra \
  --agent critic-refine \
  --provider generic \
  --model my-model \
  --base-url https://example.com/v1 \
  --api-key-env MY_MODEL_API_KEY
```

### Task Commands

Zebra direct reasoning:

```bash
python -m minibench.cli evaluate-zebra \
  --zebra-tasks data/zebra/eval.jsonl \
  --agent openai-compatible \
  --provider deepseek \
  --model deepseek-chat \
  --json-mode \
  --max-tokens 4096 \
  --timeout 120
```

The built-in smoke set contains one easy, medium, and hard record from
`WildEval/ZebraLogic`. The formal direct-reasoning set contains 15 records per
difficulty and is used by `config/experiments/zebra.yaml`. `rule_context` is
already wired into prompts. Records
with `capability: history_memory` run both real-chat protocols by default:
`incremental_state` and `deferred_reasoning`. The provider-level message API is
task-agnostic and can also be reused by Xiangqi and Mahjong history evaluators.

The same 45 source ids are reused for the temporary-codebook and history
variants. Counterfactual-rule candidates are kept in a separate unscoreable
JSONL until each changed solution has been manually or solver verified.

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

MiniBench 2.0 A1 direct one-stroke puzzles (no Euler-theorem hint):

```bash
python -m minibench.cli evaluate-one-stroke \
  --one-stroke-tasks data/one_stroke/a1_direct.jsonl \
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

Single-player Riichi Mahjong draw-discard tasks:

```bash
python -m minibench.cli evaluate-mahjong-solo \
  --mahjong-solo-tasks data/mahjong_solo/tasks.jsonl \
  --agent cot \
  --provider deepseek \
  --model deepseek-chat \
  --json-mode \
  --max-tokens 512 \
  --move-scorer shanten \
  --progress \
  --timeout 120
```

`score` is 1 only when the agent wins by tsumo within the draw limit.
`per_move_average_score` is the average discard-quality score over all scored
discards.

To score each discard by agreement with Akochan's recommended action:

```bash
export AKOCHAN_HOME=/path/to/akochan
export AKOCHAN_CONDA_PREFIX="$CONDA_PREFIX"

python -m minibench.cli evaluate-mahjong-solo \
  --mahjong-solo-tasks data/mahjong_solo/tasks_win.jsonl \
  --agent cot \
  --provider deepseek \
  --model deepseek-chat \
  --json-mode \
  --max-tokens 512 \
  --move-scorer akochan-choice \
  --mahjong-ai-command "python examples/akochan_wrapper.py" \
  --mahjong-ai-mode stdio \
  --mahjong-ai-timeout 60 \
  --progress \
  --timeout 120
```

`akochan-choice` is a policy-agreement score: each discard gets 1.0 when it
matches the discard selected by the external Akochan wrapper and 0.0 otherwise.
The local shanten score is still recorded as `shanten_move_score` for context.

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

### Task Generators

Generate one-stroke graph puzzles:

```bash
python scripts/generate_one_stroke_tasks.py \
  --output data/one_stroke/tasks_generated.jsonl \
  --count 200 \
  --min-vertices 4 \
  --max-vertices 8 \
  --seed 20260702 \
  --overwrite
```

Generate simple Xiangqi one-move capture-general tasks. The generator keeps only
positions with exactly one winning legal move:

```bash
python -m minibench.cli generate-xiangqi-capture \
  --count 200 \
  --output data/xiangqi/tasks_generated.jsonl \
  --piece-types rook,cannon,horse,soldier \
  --difficulties easy,medium,hard \
  --seed 20260702 \
  --overwrite \
  --progress-interval 100
```

Convert CCPD endgame FEN records into Pikafish-opponent Xiangqi battle tasks.
This conversion is format-only by default and does not call Pikafish:

```bash
python -m minibench.cli generate-ccpd-endgames \
  --ccpd-root /path/to/Chinese-Chess-Practical-Dataset \
  --output data/xiangqi/ccpd_endgames.jsonl \
  --overwrite \
  --progress-interval 25
```

Add `--engine-label` only when you want Pikafish static-score labels during
generation; this is slower.

Generate single-player Mahjong draw-discard tasks:

```bash
python -m minibench.cli generate-mahjong-solo \
  --output data/mahjong_solo/tasks.jsonl \
  --count 50 \
  --max-draws 18 \
  --seed 20260702 \
  --overwrite
```

Add `--require-oracle-win` to keep only tasks that the local shanten/ukeire
oracle can win within the draw limit.

### One-Stroke Prompt Variants

Run one-stroke puzzles without the Euler theorem hint:

```bash
python -m minibench.cli evaluate-one-stroke \
  --one-stroke-tasks data/one_stroke/tasks.jsonl \
  --agent cot \
  --provider deepseek \
  --model deepseek-chat \
  --json-mode \
  --max-tokens 512 \
  --prompt-variant baseline \
  --progress \
  --timeout 120
```

Run one-stroke puzzles with the Euler theorem hint:

```bash
python -m minibench.cli evaluate-one-stroke \
  --one-stroke-tasks data/one_stroke/tasks.jsonl \
  --agent cot \
  --provider deepseek \
  --model deepseek-chat \
  --json-mode \
  --max-tokens 512 \
  --prompt-variant euler_theorem \
  --progress \
  --timeout 120
```

The formal A1, A2, and A3 configurations are available as YAML experiments. A3 runs
every task once with explicit incremental state and once with raw step history
only, so 30 tasks produce 60 results:

```bash
./run.sh config/experiments/one_stroke.yaml
./run.sh config/experiments/one_stroke_a1.yaml
./run.sh config/experiments/one_stroke_a2.yaml
./run.sh config/experiments/one_stroke_a2_ablation.yaml
./run.sh config/experiments/one_stroke_a3_history.yaml
```

A2 path answers contain both vertex and edge-ID sequences so parallel edges and
edge-specific rules remain unambiguous:

```json
{"path":["A","B","C"],"edge_path":["e01","e02"]}
```

The formal A2 run uses the complete temporary-rule set. The ablation config runs
`full`, `standard`, `drop_key_rule`, and `conflicting_rule`; the last mode removes
the designated key rule and replaces it with a verified logical reverse. Every
mode is scored against its own recomputed constrained-path oracle. The reported
`rule_ignore_rate` is the fraction of ordinary valid Euler paths that violate at
least one displayed temporary rule; malformed or ordinary-invalid paths are not
included in its denominator.

The earlier `one_stroke_euler_theorem.yaml` is retained only as a legacy prompt
ablation for the smoke set; it is not part of the MiniBench 2.0 A1 protocol.

### Xiangqi Evaluation Commands

Run generated simple Xiangqi one-move capture-general tasks:

```bash
python -m minibench.cli evaluate-xiangqi \
  --xiangqi-tasks data/xiangqi/tasks_generated.jsonl \
  --agent cot \
  --provider deepseek \
  --model deepseek-chat \
  --json-mode \
  --max-tokens 256 \
  --progress \
  --timeout 120
```

Run CCPD endgame battle tasks against Pikafish:

```bash
python -m minibench.cli evaluate-xiangqi \
  --xiangqi-tasks data/xiangqi/ccpd_endgames.jsonl \
  --agent openai-compatible \
  --provider deepseek \
  --model deepseek-chat \
  --json-mode \
  --max-tokens 256 \
  --pikafish-path /path/to/Pikafish/src/pikafish \
  --pikafish-eval-file /path/to/Pikafish/src/pikafish.nnue \
  --pikafish-depth 8 \
  --progress \
  --timeout 120
```

To score every agent move with Pikafish, add:

```bash
--score-agent-moves --score-depth 4
```

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
Zebra 逻辑网格、象棋、一笔画、静态麻将牌型，以及本地四人 Riichi Mahjong。

代码结构的原则是：任务、agent、provider、实验配置彼此分离。新增任务时，主要只需要
新增一个 `src/minibench/datasets/<family>/` 包、一份 `data/<family>/tasks.jsonl`
数据文件，以及在 `src/minibench/factory/experiments.py` 里注册一次。

### 快速开始

```bash
export PYTHONPATH=src
python -m unittest discover -s tests
python -m minibench.cli evaluate-zebra --provider deepseek
./run.sh config/experiments/zebra.yaml
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
| Zebra 逻辑网格 | `data/zebra/tasks.jsonl` | `evaluate-zebra` |
| 简单象棋 | `data/xiangqi/tasks.jsonl` | `evaluate-xiangqi` |
| Pikafish 困难象棋 | `data/xiangqi/hard_tasks.jsonl` | `evaluate-xiangqi` |
| 一笔画 | `data/one_stroke/tasks.jsonl` | `evaluate-one-stroke` |
| 静态麻将牌型 | `data/mahjong/tasks.jsonl` | `evaluate-mahjong` |
| 四人 Riichi Mahjong | `data/mahjong_riichi/tasks.jsonl` | `evaluate-mahjong-riichi` |

内置 Zebra smoke 集从 `WildEval/ZebraLogic` 选取 easy、medium、hard 各一题。
provider 的真实多轮消息接口不绑定 Zebra，后续象棋和麻将的历史记忆评测也可直接复用。

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
