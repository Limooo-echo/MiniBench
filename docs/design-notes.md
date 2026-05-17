# Design Notes

## First Loop

The first implementation follows this minimal loop:

```text
question selection
  -> constrained prompt
  -> agent output
  -> JSON extraction
  -> regex fallback
  -> answer comparison
  -> run artifacts
```

This intentionally starts smaller than AgentBoard. The current unit is a
single-turn task, but the code separates dataset loading, prompting, extraction,
scoring, and run writing so the task unit can later become a multi-step
environment.

## Borrowed Ideas

From AgentBoard:

- Keep evaluation deterministic where possible.
- Track process information instead of only a final pass/fail result.
- Preserve tags that can later become capability dimensions.

From SWE-bench:

- Keep instances and predictions as explicit JSONL files.
- Store every run under an isolated output directory.
- Make the evaluator independent of the model that produced predictions.

## Near-Term Roadmap

1. Add an `Environment` interface with `reset`, `step`, and `state`.
2. Save full trajectories for multi-step tasks.
3. Add progress-rate scoring for intermediate states.
4. Add weighted diagnostic dimensions over tags.
5. Add real model adapters after deciding provider and key handling.

