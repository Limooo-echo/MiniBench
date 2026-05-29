# Mini Agent Bench

[English](#english) | [中文](#中文)

## English

Mini Agent Bench is a small, reproducible benchmark for comparing LLM and
agent-style reasoning behavior. It now supports three task families:

- Single-turn multiple-choice questions.
- Xiangqi environment tasks, with optional Pikafish opponent play.
- One-stroke graph puzzles, scored as Euler trail or Euler circuit solutions.

The project also supports several agent architectures under
`src/minibench/agents/`, so the same benchmark data can be evaluated with
different reasoning strategies.

### Quick Start

Run the built-in oracle agent on the multiple-choice set:

```powershell
$env:PYTHONPATH="src"
python -m minibench.cli evaluate --agent oracle
```

Run tests:

```powershell
$env:PYTHONPATH="src"
python -m unittest discover -s tests
```

Inspect one multiple-choice prompt:

```powershell
$env:PYTHONPATH="src"
python -m minibench.cli show-prompt mb-choice-001
```

### Run In WSL

If this repository is stored on a Windows drive, open WSL and enter the project
through `/mnt/<drive-letter-lowercase>/...`:

```bash
cd /mnt/d/benchmark/MiniBench
export PYTHONPATH=src
python3 -m unittest discover -s tests
python3 -m minibench.cli evaluate --agent oracle
```

For model-backed agents, set the provider API key inside WSL:

```bash
cd /mnt/d/benchmark/MiniBench
export PYTHONPATH=src
export DEEPSEEK_API_KEY="your_key_here"
python3 -m minibench.cli evaluate --agent cot --provider deepseek
```

PowerShell and WSL use different environment variable syntax:

```text
PowerShell: $env:DEEPSEEK_API_KEY="your_key_here"
WSL/bash:   export DEEPSEEK_API_KEY="your_key_here"
```

### Agent Architectures

All multiple-choice architectures use the same evaluator:

```powershell
$env:PYTHONPATH="src"
python -m minibench.cli evaluate --agent <agent-name> --provider <provider>
```

Available agent names:

- `oracle`: returns the gold answer; evaluation sanity check only.
- `noisy`: returns a loose text answer; extraction sanity check only.
- `openai-compatible`: direct OpenAI-compatible chat completion baseline.
- `direct`: asks the model to answer directly with the required JSON format.
- `cot`: Chain-of-Thought style; reason first, then finalize to JSON.
- `self-consistency`: sample several reasoning paths and majority vote.
- `tot`: Tree-of-Thought style; generate several candidates, then judge.
- `plan-then-solve`: produce a short plan, solve from the plan, then finalize.
- `critic-refine`: draft an answer, critique it, then return a refined answer.

Reasoning architectures support shared options:

```powershell
--samples 3
--reasoning-temperature 0.7
--final-temperature 0.0
--max-reasoning-tokens 512
```

### Provider Examples

DeepSeek:

```powershell
$env:PYTHONPATH="src"
$env:DEEPSEEK_API_KEY="your_key_here"
python -m minibench.cli evaluate --agent cot --provider deepseek
```

Qwen/DashScope:

```powershell
$env:PYTHONPATH="src"
$env:DASHSCOPE_API_KEY="your_key_here"
python -m minibench.cli evaluate --agent self-consistency --provider qwen
```

SiliconFlow:

```powershell
$env:PYTHONPATH="src"
$env:SILICONFLOW_API_KEY="your_key_here"
python -m minibench.cli evaluate --agent tot --provider siliconflow --model your-siliconflow-model-id
```

Custom OpenAI-compatible endpoint:

```powershell
$env:PYTHONPATH="src"
$env:MY_MODEL_API_KEY="your_key_here"
python -m minibench.cli evaluate `
  --agent openai-compatible `
  --provider generic `
  --model my-model `
  --base-url https://example.com/v1 `
  --api-key-env MY_MODEL_API_KEY
```

Provider shortcuts:

- `deepseek`: `DEEPSEEK_API_KEY`, `https://api.deepseek.com`, default model `deepseek-v4-flash`.
- `qwen`: `DASHSCOPE_API_KEY`, `https://dashscope.aliyuncs.com/compatible-mode/v1`, default model `qwen3.6-plus`.
- `qwen-intl`: `DASHSCOPE_API_KEY`, `https://dashscope-intl.aliyuncs.com/compatible-mode/v1`, default model `qwen3.6-plus`.
- `qwen-us`: `DASHSCOPE_API_KEY`, `https://dashscope-us.aliyuncs.com/compatible-mode/v1`, default model `qwen3.6-plus`.
- `siliconflow`: `SILICONFLOW_API_KEY`, `https://api.siliconflow.cn/v1`, requires `--model`.
- `generic`: requires `--model`, `--base-url`, and `--api-key-env`.

### Xiangqi + Pikafish

The Xiangqi path keeps MiniBench as the test runner, `gym-xiangqi` as the rule
environment, the OpenAI-compatible agent as the tested player, and Pikafish as
the optional high-strength opponent for harder endgames.

Run the simple one-step Xiangqi set:

```bash
cd /mnt/d/benchmark/MiniBench
export PYTHONPATH=src
python -m minibench.cli evaluate-xiangqi \
  --xiangqi-tasks data/xiangqi_tasks.jsonl \
  --predictions path/to/xiangqi_predictions.jsonl
```

Build Pikafish from the source zip in WSL:

```bash
cd /mnt/d/benchmark/pikafish/Pikafish-master/src
make -j"$(nproc)" build ARCH=x86-64
```

Run the hard endgame set against Pikafish:

```bash
cd /mnt/d/benchmark/MiniBench
export PYTHONPATH=src
export PIKAFISH_PATH=/mnt/d/benchmark/pikafish/Pikafish-master/src/pikafish
export DEEPSEEK_API_KEY="your_key_here"
python -m minibench.cli evaluate-xiangqi \
  --xiangqi-tasks data/xiangqi_hard_tasks.jsonl \
  --agent openai-compatible \
  --provider deepseek \
  --pikafish-depth 8
```

Each hard task can set `"opponent": "pikafish"` and `"agent_side": "ally"` or
`"enemy"`. MiniBench alternates turns through `gym-xiangqi`; on the agent turn it
asks for `{"action": 12345}`, and on the opponent turn it asks Pikafish for a UCI
best move and maps it back into the gym action space.

### One-Stroke Puzzles

One-stroke tasks are graph puzzles. The agent must return a vertex sequence that
uses every listed undirected edge exactly once:

```json
{"path":["A","B","C","D"]}
```

Run the built-in one-stroke set with a model:

```bash
cd /mnt/d/benchmark/MiniBench
export PYTHONPATH=src
export DEEPSEEK_API_KEY="your_key_here"
python -m minibench.cli evaluate-one-stroke \
  --one-stroke-tasks data/one_stroke_tasks.jsonl \
  --agent openai-compatible \
  --provider deepseek \
  --json-mode
```

Replay a saved predictions file:

```bash
python -m minibench.cli evaluate-one-stroke \
  --one-stroke-tasks data/one_stroke_tasks.jsonl \
  --predictions path/to/one_stroke_predictions.jsonl
```

### Dataset Files

- `data/tasks-limo.jsonl`: contributor-specific multiple-choice tasks.
- `data/xiangqi_tasks.jsonl`: simple Xiangqi tasks.
- `data/xiangqi_hard_tasks.jsonl`: Pikafish-opponent Xiangqi endgames.
- `data/one_stroke_tasks.jsonl`: one-stroke graph puzzles.

Each evaluation creates a run directory under `runs/` with:

- `predictions.jsonl`: raw outputs and per-instance results.
- `results.json`: aggregate metrics.
- `summary.txt`: short human-readable summary.

## 中文

Mini Agent Bench 是一个小型、可复现的 agent benchmark。现在支持三类任务：

- 单轮四选一题。
- 中国象棋环境题，可选 Pikafish 作为高水平对手。
- 一笔画图论题，按欧拉路径或欧拉回路进行校验。

项目同时支持 `src/minibench/agents/` 下的多种 agent 架构，可以用同一批题比较不同
“思维组织方式”的效果。

### 快速开始

运行 multiple-choice 的 oracle agent，检查评测链路：

```powershell
$env:PYTHONPATH="src"
python -m minibench.cli evaluate --agent oracle
```

运行测试：

```powershell
$env:PYTHONPATH="src"
python -m unittest discover -s tests
```

查看某一道选择题 prompt：

```powershell
$env:PYTHONPATH="src"
python -m minibench.cli show-prompt mb-choice-001
```

### 在 WSL 中运行

如果项目在 Windows 磁盘中，进入 WSL 后路径要写成 `/mnt/<小写盘符>/...`：

```bash
cd /mnt/d/benchmark/MiniBench
export PYTHONPATH=src
python3 -m unittest discover -s tests
python3 -m minibench.cli evaluate --agent oracle
```

调用真实模型时，也要在 WSL 里设置 API key：

```bash
export DEEPSEEK_API_KEY="your_key_here"
python3 -m minibench.cli evaluate --agent cot --provider deepseek
```

### Agent 架构

选择题可以使用这些 agent 名称：

- `oracle`：直接返回标准答案，只用于检查评测链路。
- `noisy`：返回较松散文本，用于检查抽取逻辑。
- `openai-compatible`：普通 OpenAI-compatible chat completion baseline。
- `direct`：直接要求模型输出最终 JSON。
- `cot`：先推理，再收敛为 JSON。
- `self-consistency`：多次采样推理，然后多数投票。
- `tot`：生成多个候选思路，再由 judge 选择。
- `plan-then-solve`：先写计划，再求解。
- `critic-refine`：初答、批判、修正。

### 象棋 + Pikafish

简单题：

```bash
cd /mnt/d/benchmark/MiniBench
export PYTHONPATH=src
python -m minibench.cli evaluate-xiangqi \
  --xiangqi-tasks data/xiangqi_tasks.jsonl \
  --predictions path/to/xiangqi_predictions.jsonl
```

困难题对接 Pikafish：

```bash
cd /mnt/d/benchmark/MiniBench
export PYTHONPATH=src
export PIKAFISH_PATH=/mnt/d/benchmark/pikafish/Pikafish-master/src/pikafish
export DEEPSEEK_API_KEY="your_key_here"
python -m minibench.cli evaluate-xiangqi \
  --xiangqi-tasks data/xiangqi_hard_tasks.jsonl \
  --agent openai-compatible \
  --provider deepseek \
  --pikafish-depth 8
```

agent 需要输出 `{"action": 12345}`。评测器会用 `gym-xiangqi` 执行规则，
在 Pikafish 回合调用 UCI 引擎生成对手走法。

### 一笔画

一笔画任务要求 agent 返回一个顶点序列，必须刚好使用每条无向边一次：

```json
{"path":["A","B","C","D"]}
```

运行内置题：

```bash
cd /mnt/d/benchmark/MiniBench
export PYTHONPATH=src
export DEEPSEEK_API_KEY="your_key_here"
python -m minibench.cli evaluate-one-stroke \
  --one-stroke-tasks data/one_stroke_tasks.jsonl \
  --agent openai-compatible \
  --provider deepseek \
  --json-mode
```

使用已有预测文件复评：

```bash
python -m minibench.cli evaluate-one-stroke \
  --one-stroke-tasks data/one_stroke_tasks.jsonl \
  --predictions path/to/one_stroke_predictions.jsonl
```

### 题库文件

- `data/tasks-limo.jsonl`：Limo 贡献的四选一题。
- `data/xiangqi_tasks.jsonl`：简单象棋题。
- `data/xiangqi_hard_tasks.jsonl`：Pikafish 对手象棋残局题。
- `data/one_stroke_tasks.jsonl`：一笔画图论题。

每次评测都会在 `runs/` 下生成目录，包含 `predictions.jsonl`、`results.json`
和 `summary.txt`。
