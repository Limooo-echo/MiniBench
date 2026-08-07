# MiniBench

[English](#english) | [中文](#中文)

## English

MiniBench is a small, reproducible benchmark for comparing LLM and
agent-style reasoning behavior. It supports multiple task families, including
multiple-choice questions, Xiangqi, one-stroke graph puzzles, static Mahjong
waiting-tile tasks, single-player Riichi Mahjong draw-discard tasks, and
independent Mahjong rule-adaptation channels.

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
  mahjong_solo/
```

### Data Layout

| Task family | Data file | Command |
| --- | --- | --- |
| Multiple choice | `data/multiple_choice/tasks.jsonl` | `evaluate` |
| Simple Xiangqi | `data/xiangqi/tasks.jsonl` | `evaluate-xiangqi` |
| Hard Xiangqi with Pikafish | `data/xiangqi/hard_tasks.jsonl` | `evaluate-xiangqi` |
| One-stroke graph puzzles | `data/one_stroke/tasks.jsonl` | `evaluate-one-stroke` |
| Mahjong 1: text-only static reasoning | `data/mahjong/tasks.jsonl` | `generate-mahjong-static` / `evaluate-mahjong` |
| Mahjong 2: full-hand rule adaptation | `data/mahjong_solo/tasks_win.jsonl` | `generate-mahjong-solo` / `evaluate-mahjong-rules` |
| Mahjong 3: history-only memory comparison | `data/mahjong_solo/tasks_win.jsonl` | `evaluate-mahjong-rules` |
| Mahjong 4: visual tile reasoning | `data/mahjong/visual_tasks.jsonl` | `generate-mahjong-visual` / `evaluate-mahjong` |

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

### Mahjong Benchmarks

All Mahjong evaluators use the same closed-hand shape definition: a regular
four-meld-and-one-pair hand, seven distinct pairs, or thirteen orphans. Calls,
open melds, yaku, riichi, dora, furiten, han, fu, and payments do not affect
benchmark correctness. Mahjong evaluation commands accept `openai-compatible`, `direct`,
`cot`, `self-consistency`, `tot`, `plan-then-solve`, and `critic-refine` agents.
Completed tasks are checkpointed after every item, so a later timeout does not
erase earlier predictions.

#### 1. Text-only static reasoning

The two task types are `winning_tiles` for a 13-tile hand and
`max_wait_discard` for a 14-tile hand. The latter maximizes the number of
distinct structural wait types, not the number of live tile copies.

```bash
python -m minibench.cli generate-mahjong-static \
  --output data/mahjong/tasks_generated.jsonl \
  --count 60 \
  --seed 20260807 \
  --overwrite

RUN_TS=$(date +%Y%m%d-%H%M%S)
python -m minibench.cli evaluate-mahjong \
  --mahjong-tasks data/mahjong/tasks_generated.jsonl \
  --agent cot \
  --provider deepseek \
  --model deepseek-chat \
  --json-mode \
  --reasoning-temperature 0 \
  --final-temperature 0 \
  --max-reasoning-tokens 2048 \
  --max-tokens 1024 \
  --progress \
  --timeout 600 \
  --run-name "${RUN_TS}-deepseek-chat-mahjong-static-cot"
```

`--count` is split as evenly as possible across easy/hard and the two task
types. Easy means the hand is not a one-suit honorless hand; hard is the
generator's full-flush heuristic, not a model-calibrated difficulty rating.
The generator rejects duplicate tasks and requires a unique best discard.
Answers are recomputed locally. Scoring is exact-match binary accuracy, with
`success_rate` and per-type results in `by_task_type`. Generated tasks are
sampled from regular four-meld-and-one-pair shapes; the evaluator itself also
accepts seven pairs and thirteen orphans when those appear in hand-authored data.

#### 2. Full-hand rule adaptation

Generate the shared deterministic solo source tasks once:

```bash
python -m minibench.cli generate-mahjong-solo \
  --output data/mahjong_solo/tasks_generated.jsonl \
  --count 15 \
  --max-draws 50 \
  --require-oracle-win \
  --max-initial-shanten 2 \
  --min-initial-ukeire 12 \
  --max-oracle-win-turn 18 \
  --greedy-simulations 10 \
  --min-greedy-win-rate 0.8 \
  --max-attempts 20000 \
  --seed 20260702 \
  --overwrite
```

The generation filters use the standard-rule shanten/ukeire oracle only. They
select playable walls but are never included in the agent prompt. Modified-rule
shanten is not implemented.

```bash
RUN_TS=$(date +%Y%m%d-%H%M%S)
python -m minibench.cli evaluate-mahjong-rules \
  --mahjong-rule-tasks data/mahjong_solo/tasks_generated.jsonl \
  --observation-mode full-hand \
  --agent cot \
  --provider deepseek \
  --model deepseek-chat \
  --json-mode \
  --samples 3 \
  --reasoning-temperature 0 \
  --final-temperature 0 \
  --max-reasoning-tokens 1024 \
  --max-tokens 512 \
  --progress \
  --timeout 600 \
  --run-name "${RUN_TS}-deepseek-chat-mahjong-rules-full"
```

With no channel selector, each source task expands into eight matched channels:
`standard`, three single modifications, three two-rule combinations, and one
three-rule combination. The modifications are:

- `no_cross_suit_duplicate_sequences`: restrictive; the same numeric sequence
  cannot appear in two suits in one winning decomposition.
- `cyclic_sequences`: additive; `891` and `912` are also legal sequences.
- `red_dragon_wildcard`: additive; every `C` may represent one arbitrary tile.

Use `--rule-channel standard` or a single named channel to run one channel. To
compose rules without writing the canonical `+` name, repeat `--rule`, for
example `--rule cyclic_sequences --rule red_dragon_wildcard`. `--limit` counts
source tasks, so `--limit 1` still runs all selected channels for one source.

All channels share the initial hand, wall, draw limit, full-hand observation,
prompt skeleton, action parser, and three-attempt illegal-action correction
loop. Only the selected rule text and local win validator differ. The matched
baseline is the `standard` channel from `evaluate-mahjong-rules`, not the
standalone `evaluate-mahjong-solo` command.

Success is binary: the agent must legally declare tsumo before the draw limit.
`by_channel` reports active rules, success rate, `variant_only_wins`,
`added_win_opportunity_draws`, and `blocked_standard_win_opportunity_draws`.
The opportunity counters compare the standard and selected validators only on
the 14-tile states reached along the agent's actual path; they are diagnostics,
not an exhaustive search of alternative discards.

#### 3. History-only memory comparison

Use the same generated source file and standard rule in both modes. This keeps
the wall, evaluator, prompt skeleton, and retry mechanism fixed while changing
only whether the current 14-tile hand is shown:

```bash
RUN_TS=$(date +%Y%m%d-%H%M%S)
for MODE in full-hand history-only; do
  python -m minibench.cli evaluate-mahjong-rules \
    --mahjong-rule-tasks data/mahjong_solo/tasks_generated.jsonl \
    --rule-channel standard \
    --observation-mode "$MODE" \
    --agent cot \
    --provider deepseek \
    --model deepseek-chat \
    --json-mode \
    --samples 3 \
    --reasoning-temperature 0 \
    --final-temperature 0 \
    --max-reasoning-tokens 1024 \
    --max-tokens 512 \
    --progress \
    --timeout 600 \
    --run-name "${RUN_TS}-deepseek-chat-mahjong-standard-${MODE}"
done
```

`full-hand` supplies the current hand. `history-only` supplies the initial 13
tiles, current draw, all completed draw/discard turns, cumulative discards, and
remaining draws; the agent must reconstruct its hand. Compare success rates for
the same task IDs. Running modified channels at the same time would mix memory
and rule-adaptation effects.

#### 4. Visual tile reasoning

```bash
python -m minibench.cli generate-mahjong-visual \
  --output data/mahjong/visual_tasks.jsonl \
  --count-per-type 15 \
  --visible-count 10 \
  --visible-count 20 \
  --table-columns 6 \
  --seed 20260803 \
  --overwrite

export DASHSCOPE_API_KEY="your API key"
RUN_TS=$(date +%Y%m%d-%H%M%S)
python -m minibench.cli evaluate-mahjong \
  --mahjong-tasks data/mahjong/visual_tasks.jsonl \
  --agent cot \
  --provider qwen \
  --model qwen3.8-max \
  --json-mode \
  --reasoning-temperature 0 \
  --final-temperature 0 \
  --max-reasoning-tokens 2048 \
  --max-tokens 1024 \
  --extra-body-json '{"enable_thinking":false}' \
  --progress \
  --timeout 600 \
  --run-name "${RUN_TS}-qwen3.8-max-mahjong-visual-cot"

# Paired text-code control: same wait tasks, answers, and output schema.
python -m minibench.cli evaluate-mahjong \
  --mahjong-tasks data/mahjong/visual_tasks.jsonl \
  --goal winning_tiles \
  --input-mode text \
  --agent cot \
  --provider qwen \
  --model qwen3.8-max \
  --json-mode \
  --reasoning-temperature 0 \
  --final-temperature 0 \
  --max-reasoning-tokens 2048 \
  --max-tokens 1024 \
  --extra-body-json '{"enable_thinking":false}' \
  --progress \
  --timeout 600 \
  --run-name "${RUN_TS}-qwen3.8-max-mahjong-visual-text-control-cot"
```

This creates 60 PNG tasks and
`data/mahjong/visual_tasks_visual/index.html`: 15 waits and 15 maximum-live-copy
discards for each 10/20-visible-tile condition. Matching 10/20 tasks use the
same hand, and the first ten public tiles are identical; the 20-tile condition
only appends ten more public tiles. Visual generation also samples regular
four-meld-and-one-pair source shapes. The image is sent as a high-detail data
URL, and the hand is not repeated as text.

`--input-mode text` removes the image only in memory and supplies the exact same
hand and visible tiles as tile codes. Visual-origin tasks still require the same
`hand`, `visible_tiles`, and answer fields, so the control changes only the input
modality. Use `--goal winning_tiles` to compare the 30 paired wait tasks.

For `winning_tiles`, scoring uses all structural waits. For
`max_ukeire_discard`, scoring maximizes the total remaining physical copies
after accounting for the hand, public tiles, and discard; all tied maxima are
accepted. Reports include answer accuracy plus tile-level and exact-match
transcription accuracy for both the hand and visible tiles. These transcription
metrics diagnose vision separately from Mahjong reasoning.

The optional standalone `evaluate-mahjong-solo` command records a Japanese
`win_score` dictionary when the scoring library can produce one, but benchmark
success still depends only on the closed-hand shape and timely tsumo. There is
currently no shanten or Akochan per-move quality score in the standard solo
evaluator.

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

The same two configurations are also available as YAML experiments:

```bash
./run.sh config/experiments/one_stroke.yaml
./run.sh config/experiments/one_stroke_euler_theorem.yaml
```

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
选择题、象棋、一笔画、静态麻将听牌题，以及单人 Riichi Mahjong 摸打题。

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
| 麻将 1：文本静态推理 | `data/mahjong/tasks.jsonl` | `generate-mahjong-static` / `evaluate-mahjong` |
| 麻将 2：全手牌规则适应 | `data/mahjong_solo/tasks_win.jsonl` | `generate-mahjong-solo` / `evaluate-mahjong-rules` |
| 麻将 3：history-only 记忆对照 | `data/mahjong_solo/tasks_win.jsonl` | `evaluate-mahjong-rules` |
| 麻将 4：视觉牌面推理 | `data/mahjong/visual_tasks.jsonl` | `generate-mahjong-visual` / `evaluate-mahjong` |

### 麻将评测

四类麻将题统一只判断闭门手牌结构：普通四面子一对子、七种不同对子或国士无双；
吃碰杠、役、立直、宝牌、振听、番符和点数不影响 benchmark 正确性。所有评测都会在
每道题结束后保存 checkpoint，后续 API 超时不会抹掉已经完成的预测。

#### 1. 文本静态推理

```bash
python -m minibench.cli generate-mahjong-static \
  --output data/mahjong/tasks_generated.jsonl \
  --count 60 \
  --seed 20260807 \
  --overwrite

RUN_TS=$(date +%Y%m%d-%H%M%S)
python -m minibench.cli evaluate-mahjong \
  --mahjong-tasks data/mahjong/tasks_generated.jsonl \
  --agent cot \
  --provider deepseek \
  --model deepseek-chat \
  --json-mode \
  --reasoning-temperature 0 \
  --final-temperature 0 \
  --max-reasoning-tokens 2048 \
  --max-tokens 1024 \
  --progress \
  --timeout 600 \
  --run-name "${RUN_TS}-deepseek-chat-mahjong-static-cot"
```

题目在 easy/hard × `winning_tiles`/`max_wait_discard` 四组间尽量均分。
`max_wait_discard` 最大化不同听牌种类数，不计算剩余实体牌张数。hard 只是“清一色且
无字牌”的生成启发式，不是经过模型标定的难度。生成器用本地算法校验答案、去重，并要求
最佳弃牌唯一。评分是最终答案严格正确的 0/1 分，汇总给出总正确率和 `by_task_type`。
生成器只从普通四面子一对子牌型采样；评测器在手写数据中仍可验证七对子和国士无双。

#### 2. 全手牌规则适应

先生成所有规则通道共用的固定初始手牌和牌墙：

```bash
python -m minibench.cli generate-mahjong-solo \
  --output data/mahjong_solo/tasks_generated.jsonl \
  --count 15 \
  --max-draws 50 \
  --require-oracle-win \
  --max-initial-shanten 2 \
  --min-initial-ukeire 12 \
  --max-oracle-win-turn 18 \
  --greedy-simulations 10 \
  --min-greedy-win-rate 0.8 \
  --max-attempts 20000 \
  --seed 20260702 \
  --overwrite
```

这里的筛选器只使用标准规则向听数/受入和贪心 oracle，筛选结果不会写入 Prompt；修改规则
专用向听数目前没有实现。

```bash
RUN_TS=$(date +%Y%m%d-%H%M%S)
python -m minibench.cli evaluate-mahjong-rules \
  --mahjong-rule-tasks data/mahjong_solo/tasks_generated.jsonl \
  --observation-mode full-hand \
  --agent cot \
  --provider deepseek \
  --model deepseek-chat \
  --json-mode \
  --samples 3 \
  --reasoning-temperature 0 \
  --final-temperature 0 \
  --max-reasoning-tokens 1024 \
  --max-tokens 512 \
  --progress \
  --timeout 600 \
  --run-name "${RUN_TS}-deepseek-chat-mahjong-rules-full"
```

不指定通道时，每道源题展开成 8 个严格配对通道：`standard`、3 个单规则、3 个双规则、
1 个三规则。三条修改规则是：

- `no_cross_suit_duplicate_sequences`：限制型；同一数字顺子不能跨花色重复。
- `cyclic_sequences`：扩展型；额外允许 `891` 和 `912` 顺子。
- `red_dragon_wildcard`：扩展型；每张红中 `C` 可代替任意一张牌。

用 `--rule-channel standard` 或其他完整通道名只跑一个通道；也可重复传入
`--rule`，例如 `--rule cyclic_sequences --rule red_dragon_wildcard`。`--limit`
限制源题数，而不是展开后的实例数。

八个通道共享初始手牌、牌墙、摸牌上限、完整手牌观察、Prompt 骨架、动作解析和每巡三次
非法操作纠错；只有规则文字和本地胡牌判定器不同。严格对照组是同一个命令下的
`standard` 通道，不是单独的 `evaluate-mahjong-solo`。

成功条件是在摸牌上限内合法声明自摸，按 0/1 计分。`by_channel` 还给出
`variant_only_wins`、`added_win_opportunity_draws` 和
`blocked_standard_win_opportunity_draws`。后两个指标只比较 Agent 实际走到的每个
14 张状态，并不穷举其他弃牌路径。

#### 3. history-only 记忆对照

同一批源题只跑标准规则，分别切换是否显示当前完整手牌：

```bash
RUN_TS=$(date +%Y%m%d-%H%M%S)
for MODE in full-hand history-only; do
  python -m minibench.cli evaluate-mahjong-rules \
    --mahjong-rule-tasks data/mahjong_solo/tasks_generated.jsonl \
    --rule-channel standard \
    --observation-mode "$MODE" \
    --agent cot \
    --provider deepseek \
    --model deepseek-chat \
    --json-mode \
    --samples 3 \
    --reasoning-temperature 0 \
    --final-temperature 0 \
    --max-reasoning-tokens 1024 \
    --max-tokens 512 \
    --progress \
    --timeout 600 \
    --run-name "${RUN_TS}-deepseek-chat-mahjong-standard-${MODE}"
done
```

`history-only` 每巡只提供初始 13 张牌、本巡摸牌、此前摸打历史、累计弃牌和剩余摸牌数，
Agent 必须自行重建当前手牌。应按相同 task ID 比较两个模式的成功率；如果同时修改规则，
就会把记忆能力与规则适应混在一起。

#### 4. 视觉牌面推理

```bash
python -m minibench.cli generate-mahjong-visual \
  --output data/mahjong/visual_tasks.jsonl \
  --count-per-type 15 \
  --visible-count 10 \
  --visible-count 20 \
  --table-columns 6 \
  --seed 20260803 \
  --overwrite

export DASHSCOPE_API_KEY="你的 API Key"
RUN_TS=$(date +%Y%m%d-%H%M%S)
python -m minibench.cli evaluate-mahjong \
  --mahjong-tasks data/mahjong/visual_tasks.jsonl \
  --agent cot \
  --provider qwen \
  --model qwen3.8-max \
  --json-mode \
  --reasoning-temperature 0 \
  --final-temperature 0 \
  --max-reasoning-tokens 2048 \
  --max-tokens 1024 \
  --extra-body-json '{"enable_thinking":false}' \
  --progress \
  --timeout 600 \
  --run-name "${RUN_TS}-qwen3.8-max-mahjong-visual-cot"

# 同题牌码对照组：题目、答案和输出结构均与视觉听牌题相同。
python -m minibench.cli evaluate-mahjong \
  --mahjong-tasks data/mahjong/visual_tasks.jsonl \
  --goal winning_tiles \
  --input-mode text \
  --agent cot \
  --provider qwen \
  --model qwen3.8-max \
  --json-mode \
  --reasoning-temperature 0 \
  --final-temperature 0 \
  --max-reasoning-tokens 2048 \
  --max-tokens 1024 \
  --extra-body-json '{"enable_thinking":false}' \
  --progress \
  --timeout 600 \
  --run-name "${RUN_TS}-qwen3.8-max-mahjong-visual-text-control-cot"
```

生成器输出 60 张 PNG 和 `data/mahjong/visual_tasks_visual/index.html`：10/20 张
桌面明牌条件下，各 15 道听牌题和 15 道最大剩余实体受入题。配对的 10/20 张题使用相同
手牌，20 张版本的前 10 张明牌也完全相同，只额外增加后 10 张。图片以 high detail 的
data URL 发送，Prompt 不会用文本泄露牌面。视觉生成器同样只从普通四面子一对子牌型采样。

视觉听牌题按全部结构听牌评分；最大受入题扣除手牌、桌面明牌和弃牌后，最大化剩余实体牌
总张数，并接受所有并列最优答案。除答案正确率外，汇总还分别报告手牌和桌面明牌的逐牌
识别准确率与整组完全识别率，用于区分视觉识别错误和麻将推理错误。

`--input-mode text` 只在内存中移除图片，并把完全相同的手牌和桌面明牌改为牌码输入；
输出仍要求 `hand`、`visible_tiles` 和答案字段，因此只改变输入模态。配合
`--goal winning_tiles` 可直接评测 30 道配对听牌题。

可选的 `evaluate-mahjong-solo` 会在日麻计分库能够返回结果时记录 `win_score`，但成功与否
仍只取决于闭门牌型和是否及时自摸；当前标准 solo 评测没有向听数或 Akochan 每步质量分。

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
