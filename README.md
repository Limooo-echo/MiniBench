# Mini Agent Bench

This repository is a deliberately small starter benchmark for agent evaluation.
It implements the first end-to-end loop:

1. Load benchmark questions.
2. Build a constrained prompt.
3. Collect model or agent output.
4. Prefer JSON parsing, then fall back to regex extraction.
5. Compare the extracted option letter with the gold option.
6. Write JSONL predictions and aggregate metrics.

The design borrows the spirit of AgentBoard and SWE-bench, but starts with the
smallest useful unit. AgentBoard motivates process-aware metrics and diagnostic
dimensions; SWE-bench motivates reproducible instances, prediction files, and
separate evaluation artifacts.

## Quick Start

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

Run an OpenAI-compatible model provider:

```powershell
$env:PYTHONPATH="src"
$env:DEEPSEEK_API_KEY="your_key_here"
python -m minibench.cli evaluate --agent openai-compatible --provider deepseek
```

```powershell
$env:PYTHONPATH="src"
$env:DASHSCOPE_API_KEY="your_key_here"
python -m minibench.cli evaluate --agent openai-compatible --provider qwen
```

Use a custom OpenAI-compatible endpoint:

```powershell
$env:PYTHONPATH="src"
$env:MY_MODEL_API_KEY="your_key_here"
python -m minibench.cli evaluate --agent openai-compatible --provider generic --model my-model --base-url https://example.com/v1 --api-key-env MY_MODEL_API_KEY
```

Run a SiliconFlow model:

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

You can pass provider-specific request fields with `--extra-body-json`, for
example:

```powershell
python -m minibench.cli evaluate --agent openai-compatible --provider deepseek --extra-body-json '{\"reasoning_effort\":\"high\"}'
```

## Dataset Format

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

## Output

Each evaluation creates a directory under `runs/`:

- `predictions.jsonl`: raw output, extracted answer, and per-instance result.
- `results.json`: aggregate counts and accuracy.
- `summary.txt`: a short human-readable summary.

## Next Steps

- Add multi-step task environments with trajectories.
- Track progress rate over intermediate states.
- Add weighted diagnostic scores over normalized tags.
- Add provider-specific metadata such as latency, token usage, and cost.
