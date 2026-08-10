# Design Notes

## First Loop

The first implementation follows this minimal loop:

```text
task selection
  -> task-family prompt
  -> agent output
  -> JSON extraction
  -> task-family scoring
  -> run artifacts
```

The code separates dataset loading, prompting, extraction, scoring, and run
writing. The provider supports both single prompts and real `system/user/assistant`
message histories, so Zebra, Xiangqi, and Mahjong can define independent
multi-turn protocols without flattening history into one prompt.

## Borrowed Ideas

From AgentBoard:

- Keep evaluation deterministic where possible.
- Track process information instead of only a final pass/fail result.
- Preserve tags that can later become capability dimensions.

From SWE-bench:

- Keep instances and predictions as explicit JSONL files.
- Store every run under an isolated output directory.
- Make the evaluator independent of the model that produced predictions.

## Model Adapter

The first real model adapter is `openai-compatible`. It uses the chat
completions request shape directly through the Python standard library, so the
benchmark does not need an SDK dependency. Provider shortcuts fill in common
defaults for DeepSeek, Qwen/DashScope, and SiliconFlow, while `generic` keeps the
adapter open to local servers and other compatible APIs.

## Task Taxonomy

Task sets use normalized `prefix:value` tags documented in
`docs/task-authoring.md`. Family, capability, source, size, and difficulty tags
support AgentBoard-style diagnostic summaries.

## Near-Term Roadmap

1. Add an `Environment` interface with `reset`, `step`, and `state`.
2. Save full trajectories for multi-step tasks.
3. Add progress-rate scoring for intermediate states.
4. Add weighted diagnostic dimensions over tags.
5. Add provider-specific result metadata such as latency, token usage, and cost.
