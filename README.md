# Mini Agent Bench

[English](#english) | [中文](#中文)

## English

Mini Agent Bench is a small, reproducible benchmark for comparing LLM and
agent-style reasoning behavior. The current task set is intentionally simple:
single-turn multiple-choice questions with fixed answer extraction and scoring.

The project supports multiple agent architectures under `src/minibench/agents/`,
so the same tasks can be evaluated with different reasoning strategies without
changing the benchmark data.

### Quick Start

Run the built-in oracle agent:

```powershell
$env:PYTHONPATH="src"
python -m minibench.cli evaluate --agent oracle
```

Run a noisy baseline that exercises regex fallback extraction:

```powershell
$env:PYTHONPATH="src"
python -m minibench.cli evaluate --agent noisy
```

Inspect the prompt for one task:

```powershell
$env:PYTHONPATH="src"
python -m minibench.cli show-prompt mb-choice-001
```

Run tests:

```powershell
$env:PYTHONPATH="src"
python -m unittest discover -s tests
```

### Run In WSL

If this repository is stored on a Windows drive, open WSL and enter the project
through `/mnt/<drive-letter-lowercase>/...`. For example:

```bash
cd /mnt/d/path/to/MiniBench
export PYTHONPATH=src
python3 -m unittest discover -s tests
python3 -m minibench.cli evaluate --agent oracle
```

For model-backed agents, set the provider API key in WSL:

```bash
cd /mnt/d/path/to/MiniBench
export PYTHONPATH=src
export DEEPSEEK_API_KEY="your_key_here"
python3 -m minibench.cli evaluate --agent cot --provider deepseek
```

WSL uses Linux-style environment variables (`export NAME=value`) instead of
PowerShell's `$env:NAME="value"`.

### Running Agent Architectures

All architecture variants use the same evaluator:

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

The current benchmark tasks do not require real tool calls or environment
actions, so ReAct-style agents are intentionally not implemented yet. These
tasks are better suited for CoT, ToT, self-consistency, planning, and critique
architectures that test reasoning organization.

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
  --agent critic-refine `
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

### Architecture Parameters

Reasoning architectures support these shared options:

```powershell
--samples 3
--reasoning-temperature 0.7
--final-temperature 0.0
--max-reasoning-tokens 512
```

Example:

```powershell
$env:PYTHONPATH="src"
$env:DEEPSEEK_API_KEY="your_key_here"
python -m minibench.cli evaluate `
  --agent self-consistency `
  --provider deepseek `
  --samples 5 `
  --reasoning-temperature 0.8
```

The final answer step is always constrained to a parseable JSON object:

```json
{"answer":"A"}
```

### Agents Package Layout

```text
src/minibench/agents/
  __init__.py
  base.py
  providers.py
  factory.py
  prompts.py
  simple.py
  direct.py
  cot.py
  self_consistency.py
  tree_of_thought.py
  plan_then_solve.py
  critic_refine.py
```

File responsibilities:

- `__init__.py`: compatibility exports, so imports like `from minibench.agents import make_agent` still work.
- `base.py`: common `Agent` interface, `ChatClient` protocol, and `ReasoningConfig`.
- `providers.py`: OpenAI-compatible HTTP adapter, provider defaults, and `resolve_provider`.
- `factory.py`: `make_agent(...)`, available agent registry, and construction logic.
- `prompts.py`: shared prompt templates for final answers, judging, critique, and reasoning.
- `simple.py`: non-LLM helper agents: `OracleAgent`, `NoisyAgent`, and `PredictionFileAgent`.
- `direct.py`: direct-answer architecture.
- `cot.py`: Chain-of-Thought style architecture.
- `self_consistency.py`: multi-sample reasoning plus majority vote.
- `tree_of_thought.py`: candidate generation plus judge selection.
- `plan_then_solve.py`: planning phase followed by solving and finalization.
- `critic_refine.py`: draft, critique, and refinement architecture.

The evaluator only depends on one stable interface:

```python
raw_output = agent.generate(prompt, task)
```

### Dataset Format

Each line in `data/tasks-limo.jsonl` is one task contributed by Limo. See
`docs/task-authoring.md` for the full authoring and tag schema.

```json
{
  "id": "mb-choice-001",
  "question": "Question text goes here.",
  "options": {
    "A": "Option A",
    "B": "Option B",
    "C": "Option C",
    "D": "Option D"
  },
  "correct_option": "B",
  "tags": [
    "format:multiple-choice",
    "turn:single",
    "source:synthetic",
    "domain:tool-use",
    "skill:tool-selection",
    "difficulty:easy"
  ]
}
```

### Output

Each evaluation creates a directory under `runs/`:

- `predictions.jsonl`: raw output, extracted answer, and per-instance result.
- `results.json`: aggregate counts and accuracy.
- `summary.txt`: short human-readable summary.

## 中文

Mini Agent Bench 是一个小型、可复现的 agent benchmark。当前题库先保持简单：
单轮四选一任务，统一做答案抽取和评分。

现在项目已经支持 `src/minibench/agents/` 下的多种 agent 架构。也就是说，
我们可以不改题库，用同一批题比较不同“思维组织方式”的效果。

### 快速开始

运行内置 oracle agent，用来检查评测链路是否正常：

```powershell
$env:PYTHONPATH="src"
python -m minibench.cli evaluate --agent oracle
```

运行 noisy baseline，用来检查正则兜底抽取是否正常：

```powershell
$env:PYTHONPATH="src"
python -m minibench.cli evaluate --agent noisy
```

查看某一道题实际生成的 prompt：

```powershell
$env:PYTHONPATH="src"
python -m minibench.cli show-prompt mb-choice-001
```

运行测试：

```powershell
$env:PYTHONPATH="src"
python -m unittest discover -s tests
```

### 在 WSL 中运行

如果项目在 Windows 磁盘里，进入 WSL 后路径要写成
`/mnt/<小写盘符>/...`。例如：

```bash
cd /mnt/d/path/to/MiniBench
export PYTHONPATH=src
python3 -m unittest discover -s tests
python3 -m minibench.cli evaluate --agent oracle
```

如果要调用真实模型，在 WSL 里也需要重新设置 API key：

```bash
cd /mnt/d/path/to/MiniBench
export PYTHONPATH=src
export DEEPSEEK_API_KEY="your_key_here"
python3 -m minibench.cli evaluate --agent cot --provider deepseek
```

PowerShell 和 WSL 的环境变量写法不同：

```text
PowerShell: $env:DEEPSEEK_API_KEY="your_key_here"
WSL/bash:   export DEEPSEEK_API_KEY="your_key_here"
```

如果你的 WSL 里还没有 Python，可以先检查：

```bash
python3 --version
```

这个项目目前没有第三方依赖，所以只要 Python 版本满足 `>=3.10`，通常就可以直接运行。

### 调用不同的 Agent 架构

所有架构都走同一个 evaluator：

```powershell
$env:PYTHONPATH="src"
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

当前题目不需要真实工具调用或环境交互，所以暂时没有实现 ReAct。现在更适合先比较
CoT、ToT、self-consistency、planning、critique 这类提升思维能力的架构。

### Provider 示例

DeepSeek：

```powershell
$env:PYTHONPATH="src"
$env:DEEPSEEK_API_KEY="your_key_here"
python -m minibench.cli evaluate --agent cot --provider deepseek
```

Qwen/DashScope：

```powershell
$env:PYTHONPATH="src"
$env:DASHSCOPE_API_KEY="your_key_here"
python -m minibench.cli evaluate --agent self-consistency --provider qwen
```

SiliconFlow：

```powershell
$env:PYTHONPATH="src"
$env:SILICONFLOW_API_KEY="your_key_here"
python -m minibench.cli evaluate --agent tot --provider siliconflow --model your-siliconflow-model-id
```

自定义 OpenAI-compatible endpoint：

```powershell
$env:PYTHONPATH="src"
$env:MY_MODEL_API_KEY="your_key_here"
python -m minibench.cli evaluate `
  --agent critic-refine `
  --provider generic `
  --model my-model `
  --base-url https://example.com/v1 `
  --api-key-env MY_MODEL_API_KEY
```

Provider 快捷配置：

- `deepseek`：`DEEPSEEK_API_KEY`，`https://api.deepseek.com`，默认模型 `deepseek-v4-flash`。
- `qwen`：`DASHSCOPE_API_KEY`，`https://dashscope.aliyuncs.com/compatible-mode/v1`，默认模型 `qwen3.6-plus`。
- `qwen-intl`：`DASHSCOPE_API_KEY`，`https://dashscope-intl.aliyuncs.com/compatible-mode/v1`，默认模型 `qwen3.6-plus`。
- `qwen-us`：`DASHSCOPE_API_KEY`，`https://dashscope-us.aliyuncs.com/compatible-mode/v1`，默认模型 `qwen3.6-plus`。
- `siliconflow`：`SILICONFLOW_API_KEY`，`https://api.siliconflow.cn/v1`，需要显式传 `--model`。
- `generic`：需要同时传 `--model`、`--base-url`、`--api-key-env`。

### 架构参数

思维型 agent 共用这些参数：

```powershell
--samples 3
--reasoning-temperature 0.7
--final-temperature 0.0
--max-reasoning-tokens 512
```

例如增加 self-consistency 的采样次数：

```powershell
$env:PYTHONPATH="src"
$env:DEEPSEEK_API_KEY="your_key_here"
python -m minibench.cli evaluate `
  --agent self-consistency `
  --provider deepseek `
  --samples 5 `
  --reasoning-temperature 0.8
```

最终答案步骤始终要求输出可解析 JSON：

```json
{"answer":"A"}
```

### Agents 包结构

原来的单文件 agent 模块已经迁移为 package：

```text
src/minibench/agents/
  __init__.py
  base.py
  providers.py
  factory.py
  prompts.py
  simple.py
  direct.py
  cot.py
  self_consistency.py
  tree_of_thought.py
  plan_then_solve.py
  critic_refine.py
```

各文件职责：

- `__init__.py`：兼容导出，保证 `from minibench.agents import make_agent` 这类旧导入仍然可用。
- `base.py`：公共 `Agent` 接口、`ChatClient` 协议、`ReasoningConfig` 配置。
- `providers.py`：OpenAI-compatible HTTP 调用、provider 默认配置、`resolve_provider`。
- `factory.py`：`make_agent(...)`、agent registry、创建不同 agent 的逻辑。
- `prompts.py`：最终答案、judge、critique、reasoning 等公共 prompt 模板。
- `simple.py`：不依赖 LLM 的辅助 agent：`OracleAgent`、`NoisyAgent`、`PredictionFileAgent`。
- `direct.py`：直接回答架构。
- `cot.py`：Chain-of-Thought 架构。
- `self_consistency.py`：多次采样推理 + 多数投票。
- `tree_of_thought.py`：候选思路生成 + judge 选择。
- `plan_then_solve.py`：先计划、再解题、最后格式化。
- `critic_refine.py`：初答、批判、修正。

Evaluator 只依赖一个稳定接口：

```python
raw_output = agent.generate(prompt, task)
```

这样题库、答案抽取、评分逻辑就不会和具体 agent 架构绑死。

### 题库格式

`data/tasks-limo.jsonl` 中每一行是一道 Limo 贡献的任务。完整出题规范见
`docs/task-authoring.md`。

```json
{
  "id": "mb-choice-001",
  "question": "Question text goes here.",
  "options": {
    "A": "Option A",
    "B": "Option B",
    "C": "Option C",
    "D": "Option D"
  },
  "correct_option": "B",
  "tags": [
    "format:multiple-choice",
    "turn:single",
    "source:synthetic",
    "domain:tool-use",
    "skill:tool-selection",
    "difficulty:easy"
  ]
}
```

每道题必须包含 `A`、`B`、`C`、`D` 四个选项。评测器会抽取一个选项字母，
然后和 `correct_option` 比较。

### 输出结果

每次评测都会在 `runs/` 下创建一个目录：

- `predictions.jsonl`：每题的原始输出、抽取答案和单题结果。
- `results.json`：总体数量、正确数、准确率和按 tag 汇总的结果。
- `summary.txt`：简短的人类可读摘要。

### 当前设计

当前评测流程是：

```text
读取任务
  -> 构造 prompt
  -> 运行选定 agent 架构
  -> 收集原始输出
  -> JSON 抽取，失败后正则兜底
  -> 和标准答案比较
  -> 写出运行结果
```

这个 benchmark 先从四选一任务开始，是为了把“架构推理能力”先和真实工具、
浏览器、文件系统、游戏环境等复杂度分开。之后可以继续加入交互环境、轨迹记录、
过程指标和 ReAct 风格工具 agent。
