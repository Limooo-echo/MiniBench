# Mini Agent Bench

[English](#english) | [中文](#中文)

## English

Mini Agent Bench is a small, reproducible benchmark for comparing LLM and
agent-style reasoning behavior. It started with single-turn multiple-choice
questions and now also includes Xiangqi, one-stroke graph puzzles, and Riichi
Mahjong tasks.

The project supports multiple agent architectures under `src/minibench/agents/`,
so the same benchmark data can be evaluated with different reasoning strategies
without changing the task files.

### Quick Start

Clone the repository and enter your Python environment:

```bash
git clone <repo-url>
cd MiniBench
export PYTHONPATH=src
```

Run tests:

```bash
python -m unittest discover -s tests
```

Run the built-in oracle agent on the multiple-choice set:

```bash
python -m minibench.cli evaluate --agent oracle
```

Inspect one prompt:

```bash
python -m minibench.cli show-prompt mb-choice-001
```

### Agent Architectures

All multiple-choice architectures use the same evaluator:

```bash
python -m minibench.cli evaluate --agent <agent-name> --provider <provider>
```

Available agent names:

- `oracle`: returns the gold answer; evaluation sanity check only.
- `noisy`: returns loose text; extraction sanity check only.
- `openai-compatible`: direct OpenAI-compatible chat completion baseline.
- `direct`: asks the model to answer directly with the required JSON format.
- `cot`: Chain-of-Thought style; reason first, then finalize to JSON.
- `self-consistency`: sample several reasoning paths and majority vote.
- `tot`: Tree-of-Thought style; generate several candidates, then judge.
- `plan-then-solve`: produce a short plan, solve from the plan, then finalize.
- `critic-refine`: draft an answer, critique it, then return a refined answer.

Reasoning architectures support shared options:

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

SiliconFlow:

```bash
export SILICONFLOW_API_KEY="your_key_here"
python -m minibench.cli evaluate \
  --agent tot \
  --provider siliconflow \
  --model your-siliconflow-model-id
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

Provider shortcuts:

- `deepseek`: `DEEPSEEK_API_KEY`, `https://api.deepseek.com`, default model `deepseek-v4-flash`.
- `qwen`: `DASHSCOPE_API_KEY`, `https://dashscope.aliyuncs.com/compatible-mode/v1`, default model `qwen3.6-plus`.
- `qwen-intl`: `DASHSCOPE_API_KEY`, `https://dashscope-intl.aliyuncs.com/compatible-mode/v1`, default model `qwen3.6-plus`.
- `qwen-us`: `DASHSCOPE_API_KEY`, `https://dashscope-us.aliyuncs.com/compatible-mode/v1`, default model `qwen3.6-plus`.
- `siliconflow`: `SILICONFLOW_API_KEY`, `https://api.siliconflow.cn/v1`, requires `--model`.
- `generic`: requires `--model`, `--base-url`, and `--api-key-env`.

### Task Families

| Task family | Data file | Command |
| --- | --- | --- |
| Multiple choice | `data/tasks-limo.jsonl` | `evaluate` |
| Simple Xiangqi | `data/xiangqi_tasks.jsonl` | `evaluate-xiangqi` |
| Hard Xiangqi with Pikafish | `data/xiangqi_hard_tasks.jsonl` | `evaluate-xiangqi` |
| One-stroke graph puzzles | `data/one_stroke_tasks.jsonl` | `evaluate-one-stroke` |
| Static Riichi Mahjong shapes | `data/mahjong_tasks.jsonl` | `evaluate-mahjong` |
| Four-player Riichi Mahjong v1 | `data/mahjong_riichi_tasks.jsonl` | `evaluate-mahjong-riichi` |

### Source Layout

The source package is organized by benchmark family:

```text
src/minibench/
  cli.py                 # shared command-line entrypoint
  agents/                # model adapters and reasoning architectures
  multiple_choice/       # four-choice dataset, prompting, extraction, scoring
  xiangqi/               # Xiangqi dataset, environment bridge, Pikafish support
  one_stroke/            # one-stroke graph puzzle dataset and evaluator
  mahjong/               # static Mahjong shape tasks
  mahjong_riichi/        # four-player Riichi Mahjong table simulation
```

Each benchmark family keeps its own `dataset.py`, `prompting.py`, and
`evaluation.py` files where applicable. This keeps new task types from piling
up in the `minibench/` root.

### Optional External Engines

Most MiniBench tasks do not require external game engines. Install these only
when you want to run the corresponding opponent mode.

Pikafish is only needed for Xiangqi tasks with `--opponent pikafish`, such as
`data/xiangqi_hard_tasks.jsonl`. Follow the upstream project for platform
details; a typical Linux/WSL source build is:

```bash
git clone https://github.com/official-pikafish/Pikafish.git
cd Pikafish/src
make -j profile-build
export PIKAFISH_PATH="$PWD/pikafish"
```

If your build or release package stores the NNUE network separately, also set:

```bash
export PIKAFISH_EVAL_FILE=/path/to/pikafish.nnue
```

akochan is only needed for four-player Riichi Mahjong when using
`--riichi-opponent external` with `examples/akochan_wrapper.py`. A typical
Linux/WSL build is:

```bash
git clone https://github.com/critter-mj/akochan.git
cd akochan
sudo apt-get install libboost-all-dev
(cd ai_src && make -f Makefile_Linux)
make -f Makefile_Linux
export AKOCHAN_HOME="$PWD"
```

The wrapper will look for `AKOCHAN_HOME/system` or `AKOCHAN_HOME/system.exe`.
If akochan needs shared libraries from the active Python/conda environment, set:

```bash
export AKOCHAN_CONDA_PREFIX="$CONDA_PREFIX"
```

### Run Multiple Choice

```bash
python -m minibench.cli --tasks data/tasks-limo.jsonl evaluate \
  --agent openai-compatible \
  --provider deepseek \
  --model deepseek-chat \
  --json-mode \
  --max-tokens 256 \
  --timeout 120
```

### Run Xiangqi

Static single-move Xiangqi tasks support all generative agent architectures:
`openai-compatible`, `direct`, `cot`, `self-consistency`, `tot`,
`plan-then-solve`, and `critic-refine`. Pikafish battle tasks remain
`openai-compatible` only.

Simple Xiangqi tasks:

```bash
python -m minibench.cli evaluate-xiangqi \
  --xiangqi-tasks data/xiangqi_tasks.jsonl \
  --agent cot \
  --provider deepseek \
  --model deepseek-chat \
  --json-mode \
  --max-tokens 256 \
  --timeout 120
```

Hard Xiangqi tasks can use Pikafish as the opponent. Build Pikafish separately
and point `PIKAFISH_PATH` at the engine binary:

```bash
export PIKAFISH_PATH=/path/to/Pikafish/src/pikafish

python -m minibench.cli evaluate-xiangqi \
  --xiangqi-tasks data/xiangqi_hard_tasks.jsonl \
  --agent openai-compatible \
  --provider deepseek \
  --model deepseek-chat \
  --json-mode \
  --max-tokens 256 \
  --pikafish-depth 8 \
  --timeout 120
```

### Run One-Stroke Puzzles

The agent must return a vertex path that uses every listed undirected edge once:

```json
{"path":["A","B","C","D"]}
```

```bash
python -m minibench.cli evaluate-one-stroke \
  --one-stroke-tasks data/one_stroke_tasks.jsonl \
  --agent cot \
  --provider deepseek \
  --model deepseek-chat \
  --json-mode \
  --max-tokens 512 \
  --timeout 120
```

### Run Riichi Mahjong

Static tile-shape tasks:
These static tasks also support every generative agent architecture.

```bash
python -m minibench.cli evaluate-mahjong \
  --mahjong-tasks data/mahjong_tasks.jsonl \
  --agent cot \
  --provider deepseek \
  --model deepseek-chat \
  --json-mode \
  --max-tokens 512 \
  --timeout 120
```

Four-player local Riichi table:

```bash
python -m minibench.cli evaluate-mahjong-riichi \
  --mahjong-riichi-tasks data/mahjong_riichi_tasks.jsonl \
  --agent openai-compatible \
  --provider deepseek \
  --model deepseek-chat \
  --json-mode \
  --max-tokens 512 \
  --timeout 120
```

By default, seats 1/2/3 use the local shanten baseline bot. To connect external
Mahjong AIs, use `--riichi-opponent external` and provide a wrapper command.
`examples/akochan_wrapper.py` is an example wrapper for akochan:

```bash
export AKOCHAN_HOME=/path/to/akochan
export AKOCHAN_CONDA_PREFIX="$CONDA_PREFIX"

python -m minibench.cli evaluate-mahjong-riichi \
  --mahjong-riichi-tasks data/mahjong_riichi_tasks.jsonl \
  --riichi-opponent external \
  --mahjong-ai-command "python examples/akochan_wrapper.py" \
  --mahjong-ai-mode oneshot \
  --mahjong-ai-timeout 60 \
  --agent openai-compatible \
  --provider deepseek \
  --model deepseek-chat \
  --json-mode \
  --max-tokens 512 \
  --timeout 120
```

The external Mahjong AI wrapper receives JSON requests on stdin and must return
one legal action as JSON, such as `{"action":"discard","tile":"5m"}`,
`{"action":"chi","tiles":["3m","4m","5m"]}`, `{"action":"pon","tile":"P"}`,
or `{"action":"pass"}`.

### Output

Each evaluation creates a directory under `runs/`:

- `predictions.jsonl`: raw outputs and per-instance results.
- `results.json`: aggregate metrics.
- `summary.txt`: short human-readable summary.

## 中文

Mini Agent Bench 是一个小型、可复现的 agent benchmark。它最初支持单轮四选一题，
现在也支持象棋、一笔画图论题和 Riichi 麻将任务。

项目支持 `src/minibench/agents/` 下的多种 agent 架构。也就是说，我们可以不改题库，
用同一批题比较不同“思维组织方式”的效果。

### 快速开始

克隆仓库并进入你的 Python 环境：

```bash
git clone <repo-url>
cd MiniBench
export PYTHONPATH=src
```

运行测试：

```bash
python -m unittest discover -s tests
```

运行 oracle agent，检查评测链路：

```bash
python -m minibench.cli evaluate --agent oracle
```

查看某一道题实际生成的 prompt：

```bash
python -m minibench.cli show-prompt mb-choice-001
```

### Agent 架构

所有选择题架构都走同一个 evaluator：

```bash
python -m minibench.cli evaluate --agent <agent-name> --provider <provider>
```

可用的 agent 名称：

- `oracle`：直接返回标准答案，只用于检查评测链路。
- `noisy`：返回较松散的文本答案，只用于检查抽取逻辑。
- `openai-compatible`：普通 OpenAI-compatible chat completion baseline。
- `direct`：直接要求模型输出最终 JSON 答案。
- `cot`：Chain-of-Thought 风格，先推理，再收敛成 JSON。
- `self-consistency`：多次独立推理，然后多数投票。
- `tot`：Tree-of-Thought 风格，生成多个候选思路，再由 judge 选择。
- `plan-then-solve`：先写简短计划，再按计划求解。
- `critic-refine`：先生成初稿，再 critique，最后修正答案。

### 题型

| 题型 | 数据文件 | 运行命令 |
| --- | --- | --- |
| 选择题 | `data/tasks-limo.jsonl` | `evaluate` |
| 象棋简单题 | `data/xiangqi_tasks.jsonl` | `evaluate-xiangqi` |
| 象棋困难题 | `data/xiangqi_hard_tasks.jsonl` | `evaluate-xiangqi`，可接 Pikafish |
| 一笔画 | `data/one_stroke_tasks.jsonl` | `evaluate-one-stroke` |
| 麻将牌型题 | `data/mahjong_tasks.jsonl` | `evaluate-mahjong` |
| 麻将 Riichi 完整 v1 | `data/mahjong_riichi_tasks.jsonl` | `evaluate-mahjong-riichi` |

### 源码结构

`src/minibench/` 现在按 benchmark 家族分包：

```text
src/minibench/
  cli.py                 # 统一命令行入口
  agents/                # 模型适配器和不同 agent 思维架构
  multiple_choice/       # 四选一题库、prompt、答案抽取和评分
  xiangqi/               # 象棋题库、环境桥接和 Pikafish 支持
  one_stroke/            # 一笔画图题库和评测器
  mahjong/               # 静态麻将牌型题
  mahjong_riichi/        # 四人 Riichi Mahjong 牌桌模拟
```

每个任务家族内部按需保留自己的 `dataset.py`、`prompting.py` 和
`evaluation.py`。这样新增题型时不会继续堆在 `minibench/` 根目录。

### 可选外部引擎

大多数 MiniBench 题目不需要外部游戏引擎。只有在运行对应对战模式时，才需要按需安装下面的引擎。

Pikafish 只用于 `--opponent pikafish` 的象棋任务，例如
`data/xiangqi_hard_tasks.jsonl`。具体平台细节以官方项目为准；Linux/WSL 下常见源码构建方式如下：

```bash
git clone https://github.com/official-pikafish/Pikafish.git
cd Pikafish/src
make -j profile-build
export PIKAFISH_PATH="$PWD/pikafish"
```

如果你的构建或 release 包把 NNUE 网络文件单独放置，也设置：

```bash
export PIKAFISH_EVAL_FILE=/path/to/pikafish.nnue
```

akochan 只用于四人 Riichi 麻将的外部 AI 对手模式，也就是
`--riichi-opponent external` 搭配 `examples/akochan_wrapper.py`。Linux/WSL 下常见构建方式如下：

```bash
git clone https://github.com/critter-mj/akochan.git
cd akochan
sudo apt-get install libboost-all-dev
(cd ai_src && make -f Makefile_Linux)
make -f Makefile_Linux
export AKOCHAN_HOME="$PWD"
```

包装器会查找 `AKOCHAN_HOME/system` 或 `AKOCHAN_HOME/system.exe`。如果 akochan 需要当前
Python/conda 环境里的共享库，再设置：

```bash
export AKOCHAN_CONDA_PREFIX="$CONDA_PREFIX"
```

### Provider 示例

DeepSeek：

```bash
export DEEPSEEK_API_KEY="your_key_here"
python -m minibench.cli evaluate --agent cot --provider deepseek
```

Qwen/DashScope：

```bash
export DASHSCOPE_API_KEY="your_key_here"
python -m minibench.cli evaluate --agent self-consistency --provider qwen
```

自定义 OpenAI-compatible endpoint：

```bash
export MY_MODEL_API_KEY="your_key_here"
python -m minibench.cli evaluate \
  --agent critic-refine \
  --provider generic \
  --model my-model \
  --base-url https://example.com/v1 \
  --api-key-env MY_MODEL_API_KEY
```

### 运行选择题

```bash
python -m minibench.cli --tasks data/tasks-limo.jsonl evaluate \
  --agent openai-compatible \
  --provider deepseek \
  --model deepseek-chat \
  --json-mode \
  --max-tokens 256 \
  --timeout 120
```

### 运行象棋

简单象棋题：

```bash
python -m minibench.cli evaluate-xiangqi \
  --xiangqi-tasks data/xiangqi_tasks.jsonl \
  --agent cot \
  --provider deepseek \
  --model deepseek-chat \
  --json-mode \
  --max-tokens 256 \
  --timeout 120
```

困难象棋题可以接 Pikafish。先单独编译 Pikafish，再把 `PIKAFISH_PATH` 指向引擎：

```bash
export PIKAFISH_PATH=/path/to/Pikafish/src/pikafish

python -m minibench.cli evaluate-xiangqi \
  --xiangqi-tasks data/xiangqi_hard_tasks.jsonl \
  --agent openai-compatible \
  --provider deepseek \
  --model deepseek-chat \
  --json-mode \
  --max-tokens 256 \
  --pikafish-depth 8 \
  --timeout 120
```

### 运行一笔画

模型需要返回一个顶点序列，刚好使用每条无向边一次：

```json
{"path":["A","B","C","D"]}
```

```bash
python -m minibench.cli evaluate-one-stroke \
  --one-stroke-tasks data/one_stroke_tasks.jsonl \
  --agent cot \
  --provider deepseek \
  --model deepseek-chat \
  --json-mode \
  --max-tokens 512 \
  --timeout 120
```

### 运行麻将

静态牌型题：

```bash
python -m minibench.cli evaluate-mahjong \
  --mahjong-tasks data/mahjong_tasks.jsonl \
  --agent cot \
  --provider deepseek \
  --model deepseek-chat \
  --json-mode \
  --max-tokens 512 \
  --timeout 120
```

四人 Riichi 本地牌桌：

```bash
python -m minibench.cli evaluate-mahjong-riichi \
  --mahjong-riichi-tasks data/mahjong_riichi_tasks.jsonl \
  --agent openai-compatible \
  --provider deepseek \
  --model deepseek-chat \
  --json-mode \
  --max-tokens 512 \
  --timeout 120
```

默认 seat 1/2/3 使用本地 shanten baseline bot。若要接外部麻将 AI，可以使用
`--riichi-opponent external` 和包装器命令。`examples/akochan_wrapper.py` 是 akochan
包装器示例：

```bash
export AKOCHAN_HOME=/path/to/akochan
export AKOCHAN_CONDA_PREFIX="$CONDA_PREFIX"

python -m minibench.cli evaluate-mahjong-riichi \
  --mahjong-riichi-tasks data/mahjong_riichi_tasks.jsonl \
  --riichi-opponent external \
  --mahjong-ai-command "python examples/akochan_wrapper.py" \
  --mahjong-ai-mode oneshot \
  --mahjong-ai-timeout 60 \
  --agent openai-compatible \
  --provider deepseek \
  --model deepseek-chat \
  --json-mode \
  --max-tokens 512 \
  --timeout 120
```

### 输出结果

每次评测都会在 `runs/` 下创建一个目录：

- `predictions.jsonl`：每题的原始输出和单题结果。
- `results.json`：总体指标和按 tag 汇总的结果。
- `summary.txt`：简短的人类可读摘要。
