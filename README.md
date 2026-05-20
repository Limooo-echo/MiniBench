# Mini Agent Bench

[English](#english) | [中文](#中文)

## English

Mini Agent Bench is a small starter benchmark for agent evaluation. The current
version focuses on a clean multiple-choice evaluation loop:

1. Load benchmark questions.
2. Build a constrained prompt.
3. Collect model or agent output.
4. Prefer JSON parsing, then fall back to regex extraction.
5. Compare the extracted option letter with the gold option.
6. Write JSONL predictions and aggregate metrics.

The design borrows from AgentBoard and SWE-bench, but starts with the smallest
useful unit. AgentBoard motivates process-aware metrics and diagnostic
dimensions; SWE-bench motivates reproducible instances, prediction files, and
separate evaluation artifacts.

### Quick Start

Run the built-in oracle agent:

```powershell
$env:PYTHONPATH="src"
python -m minibench.cli evaluate --agent oracle
```

Run a noisier agent that exercises regex fallback extraction:

```powershell
$env:PYTHONPATH="src"
python -m minibench.cli evaluate --agent noisy
```

Inspect the prompt for one task:

```powershell
$env:PYTHONPATH="src"
python -m minibench.cli show-prompt mb-choice-001
```

Replay an existing predictions file:

```powershell
$env:PYTHONPATH="src"
python -m minibench.cli evaluate --predictions path\to\predictions.jsonl
```

Run an OpenAI-compatible provider:

```powershell
$env:PYTHONPATH="src"
$env:DEEPSEEK_API_KEY="your_key_here"
python -m minibench.cli evaluate --agent openai-compatible --provider deepseek
```

```powershell
$env:PYTHONPATH="src"
$env:SILICONFLOW_API_KEY="your_key_here"
python -m minibench.cli evaluate --agent openai-compatible --provider siliconflow --model your-siliconflow-model-id
```

Provider shortcuts:

- `deepseek`: `DEEPSEEK_API_KEY`, `https://api.deepseek.com`, default model `deepseek-v4-flash`.
- `qwen`: `DASHSCOPE_API_KEY`, `https://dashscope.aliyuncs.com/compatible-mode/v1`, default model `qwen3.6-plus`.
- `qwen-intl`: `DASHSCOPE_API_KEY`, `https://dashscope-intl.aliyuncs.com/compatible-mode/v1`, default model `qwen3.6-plus`.
- `qwen-us`: `DASHSCOPE_API_KEY`, `https://dashscope-us.aliyuncs.com/compatible-mode/v1`, default model `qwen3.6-plus`.
- `siliconflow`: `SILICONFLOW_API_KEY`, `https://api.siliconflow.cn/v1`, requires `--model`.

Use a custom OpenAI-compatible endpoint:

```powershell
$env:PYTHONPATH="src"
$env:MY_MODEL_API_KEY="your_key_here"
python -m minibench.cli evaluate --agent openai-compatible --provider generic --model my-model --base-url https://example.com/v1 --api-key-env MY_MODEL_API_KEY
```

### Dataset Format

Each line in `data/tasks.jsonl` is one task. See `docs/task-authoring.md` for
the full authoring and tag schema.

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
  "tags": ["format:multiple-choice", "turn:single", "source:synthetic", "domain:tool-use", "skill:tool-selection", "difficulty:easy"]
}
```

Every task must contain exactly four options labeled `A`, `B`, `C`, and `D`.
The evaluator extracts one option letter and compares it with `correct_option`.
Default answer extractors and prompt constraints are applied when those fields
are omitted.

### Output

Each evaluation creates a directory under `runs/`:

- `predictions.jsonl`: raw output, extracted answer, and per-instance result.
- `results.json`: aggregate counts and accuracy.
- `summary.txt`: a short human-readable summary.

### Next Steps

- Add multi-step task environments with trajectories.
- Track progress rate over intermediate states.
- Add weighted diagnostic scores over normalized tags.
- Add provider-specific metadata such as latency, token usage, and cost.

## 中文

Mini Agent Bench 是一个小型 agent benchmark 起步项目。当前版本先把“四选一题目评测”的最小闭环跑通：

1. 读取 benchmark 题目。
2. 构造受约束的 prompt。
3. 收集模型或 agent 输出。
4. 优先解析 JSON，失败时用正则兜底抽取答案。
5. 将抽取出的选项字母和标准答案比较。
6. 写出 JSONL 预测文件和汇总指标。

这个设计参考了 AgentBoard 和 SWE-bench，但从最小可用单元开始。AgentBoard 启发我们后续加入过程指标和能力诊断；SWE-bench 启发我们把题目、预测和评测结果都做成可复现的文件。

### 快速开始

运行内置 oracle agent。它直接读取标准答案，适合检查评测链路是否正常：

```powershell
$env:PYTHONPATH="src"
python -m minibench.cli evaluate --agent oracle
```

运行 noisy agent。它会输出不那么规整的文本，用来测试正则兜底抽取：

```powershell
$env:PYTHONPATH="src"
python -m minibench.cli evaluate --agent noisy
```

查看某一道题实际生成的 prompt：

```powershell
$env:PYTHONPATH="src"
python -m minibench.cli show-prompt mb-choice-001
```

用已有预测文件重新评分：

```powershell
$env:PYTHONPATH="src"
python -m minibench.cli evaluate --predictions path\to\predictions.jsonl
```

### 接入外部模型

运行 DeepSeek：

```powershell
$env:PYTHONPATH="src"
$env:DEEPSEEK_API_KEY="your_key_here"
python -m minibench.cli evaluate --agent openai-compatible --provider deepseek
```

运行硅基流动模型：

```powershell
$env:PYTHONPATH="src"
$env:SILICONFLOW_API_KEY="your_key_here"
python -m minibench.cli evaluate --agent openai-compatible --provider siliconflow --model your-siliconflow-model-id
```

可用 provider 快捷方式：

- `deepseek`: 使用 `DEEPSEEK_API_KEY`，地址为 `https://api.deepseek.com`，默认模型为 `deepseek-v4-flash`。
- `qwen`: 使用 `DASHSCOPE_API_KEY`，地址为 `https://dashscope.aliyuncs.com/compatible-mode/v1`，默认模型为 `qwen3.6-plus`。
- `qwen-intl`: 使用 `DASHSCOPE_API_KEY`，地址为 `https://dashscope-intl.aliyuncs.com/compatible-mode/v1`，默认模型为 `qwen3.6-plus`。
- `qwen-us`: 使用 `DASHSCOPE_API_KEY`，地址为 `https://dashscope-us.aliyuncs.com/compatible-mode/v1`，默认模型为 `qwen3.6-plus`。
- `siliconflow`: 使用 `SILICONFLOW_API_KEY`，地址为 `https://api.siliconflow.cn/v1`，需要显式传入 `--model`。

自定义 OpenAI-compatible 接口：

```powershell
$env:PYTHONPATH="src"
$env:MY_MODEL_API_KEY="your_key_here"
python -m minibench.cli evaluate --agent openai-compatible --provider generic --model my-model --base-url https://example.com/v1 --api-key-env MY_MODEL_API_KEY
```

### 题库格式

`data/tasks.jsonl` 中每一行是一道题。完整出题规范和 tag schema 见 `docs/task-authoring.md`。

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
  "tags": ["format:multiple-choice", "turn:single", "source:synthetic", "domain:tool-use", "skill:tool-selection", "difficulty:easy"]
}
```

每道题必须刚好包含 `A`、`B`、`C`、`D` 四个选项。评测器会抽取一个选项字母，并与 `correct_option` 比较。省略 `answer_extractors` 和 `prompt_constraints` 时，会自动使用默认规则。

### 输出结果

每次评测都会在 `runs/` 下创建一个目录：

- `predictions.jsonl`: 每题的原始输出、抽取答案和单题结果。
- `results.json`: 总体指标和按 tag 汇总的指标。
- `summary.txt`: 简短的人类可读总结。

### 下一步

- 增加带 trajectory 的多步任务环境。
- 记录中间状态的 progress rate。
- 基于规范化 tags 做加权能力诊断。
- 记录 provider 相关元数据，比如 latency、token usage 和 cost。

