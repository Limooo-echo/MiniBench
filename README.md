# Mini Agent Bench

This repository is a deliberately small starter benchmark for agent evaluation.
It implements the first end-to-end loop:

1. Load benchmark questions.
2. Build a constrained prompt.
3. Collect model or agent output.
4. Prefer JSON parsing, then fall back to regex extraction.
5. Compare the extracted answer with the reference answer.
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
python -m minibench.cli show-prompt mb-arithmetic-001
```

Replay an existing predictions file:

```powershell
$env:PYTHONPATH="src"
python -m minibench.cli evaluate --predictions path\to\predictions.jsonl
```

## Dataset Format

Each line in `data/tasks.jsonl` is one task:

```json
{
  "id": "mb-arithmetic-001",
  "question": "What is 17 * 23?",
  "reference_answers": ["391"],
  "reference_regex": null,
  "answer_extractors": ["(?i)answer\\s*[:=]\\s*([A-Za-z0-9_. -]+)"],
  "prompt_constraints": ["Return exactly one JSON object.", "Use the schema {\"answer\": \"...\"}."],
  "tags": ["arithmetic"]
}
```

`reference_answers` supports exact normalized matching. `reference_regex` is
useful when several surface forms should be accepted.

## Output

Each evaluation creates a directory under `runs/`:

- `predictions.jsonl`: raw output, extracted answer, and per-instance result.
- `results.json`: aggregate counts and accuracy.
- `summary.txt`: a short human-readable summary.

## Next Steps

- Add multi-step task environments with trajectories.
- Track progress rate over intermediate states.
- Add capability tags and weighted diagnostic scores.
- Add real model adapters after API keys and model choices are settled.

