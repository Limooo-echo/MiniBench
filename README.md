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

## Dataset Format

Each line in `data/tasks.jsonl` is one task:

```json
{
  "id": "mb-choice-001",
  "question": "Which number is the result of 17 * 23?",
  "options": {
    "A": "340",
    "B": "391",
    "C": "421",
    "D": "529"
  },
  "correct_option": "B",
  "answer_extractors": ["(?i)answer\\s*[:=]\\s*([ABCD])"],
  "prompt_constraints": ["Return exactly one JSON object.", "Use the schema {\"answer\": \"A\"}."],
  "tags": ["arithmetic", "multiple-choice"]
}
```

Every task must contain exactly four options labeled `A`, `B`, `C`, and `D`.
The evaluator extracts one option letter and compares it with `correct_option`.

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
