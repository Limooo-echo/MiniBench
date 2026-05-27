# Mini Agent Bench

Mini Agent Bench is a small, reproducible benchmark for comparing LLM and
agent-style reasoning behavior. The current task set is intentionally simple:
single-turn multiple-choice questions with fixed answer extraction and scoring.

The project now supports multiple agent architectures under
`src/minibench/agents/`, so the same tasks can be evaluated with different
reasoning strategies without changing the benchmark data.

## Quick Start

Run the built-in oracle agent. This is useful for checking that the evaluation
loop itself works:

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

Replay an existing predictions file:

```powershell
$env:PYTHONPATH="src"
python -m minibench.cli evaluate --predictions path\to\predictions.jsonl
```

## Running Different Agent Architectures

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

The reasoning architectures support these shared options:

```powershell
--samples 3
--reasoning-temperature 0.7
--final-temperature 0.0
--max-reasoning-tokens 512
```

Examples:

```powershell
$env:PYTHONPATH="src"
$env:DEEPSEEK_API_KEY="your_key_here"
python -m minibench.cli evaluate `
  --agent self-consistency `
  --provider deepseek `
  --samples 5 `
  --reasoning-temperature 0.8
```

```powershell
$env:PYTHONPATH="src"
$env:DEEPSEEK_API_KEY="your_key_here"
python -m minibench.cli evaluate `
  --agent tot `
  --provider deepseek `
  --samples 4 `
  --max-reasoning-tokens 800
```

The final answer step is always constrained to a parseable JSON object such as:

```json
{"answer":"A"}
```

## Agents Package Layout

The old single-file agent module has been migrated to a package:

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

- `__init__.py`: compatibility exports, so existing imports like `from minibench.agents import make_agent` still work.
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

The evaluator still only depends on the stable interface:

```python
raw_output = agent.generate(prompt, task)
```

This keeps the benchmark data, answer extraction, and scoring independent from
the architecture used to produce the answer.

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

Every task must contain exactly four options labeled `A`, `B`, `C`, and `D`.
The evaluator extracts one option letter and compares it with `correct_option`.
Default answer extractors and prompt constraints are applied when those fields
are omitted.

## Output

Each evaluation creates a directory under `runs/`:

- `predictions.jsonl`: raw output, extracted answer, and per-instance result.
- `results.json`: aggregate counts and accuracy.
- `summary.txt`: short human-readable summary.

## Current Design

The current evaluation loop is:

```text
load task
  -> build prompt
  -> run selected agent architecture
  -> collect raw output
  -> JSON extraction, then regex fallback
  -> compare extracted option with gold option
  -> write run artifacts
```

The benchmark intentionally starts with multiple-choice tasks so architecture
experiments can be isolated from environment/tool complexity. Later versions can
add interactive environments, trajectories, process metrics, and ReAct-style
tool agents without changing this first stable interface.
