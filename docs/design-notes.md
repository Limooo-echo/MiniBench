# Design Notes

## Evaluation Loop

All current task families follow this loop:

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

`factory/providers.py` provides `OpenAICompatibleClient`. It uses the chat
completions request shape directly through the Python standard library, so the
benchmark does not need an SDK dependency. Provider shortcuts fill in common
defaults for DeepSeek, Qwen/DashScope, and SiliconFlow, while `generic` keeps the
adapter open to local servers and other compatible APIs.

The transport client is not an Agent. `factory/agents.py` composes it with
`passthrough` or a reasoning architecture from `agents/`. There is no deprecated
combined client/agent wrapper and no old-name fallback.

## Task Taxonomy

Task sets use normalized `prefix:value` tags documented in
`docs/task-authoring.md`. Family, capability, source, size, and difficulty tags
support AgentBoard-style diagnostic summaries.

## Code Ownership

- `cli.py` composes command groups; `commands/` owns CLI arguments and handlers.
- `evaluate.py` loads/validates a YAML configuration and starts the experiment.
- `factory/` owns configuration validation, agent construction and HTTP transport.
- `experiments/registry.py` is the sole task-family registry. `runner.py` selects
  inputs and routes to the task's checkpoint orchestration; `dispatch.py` forwards
  supported evaluation parameters. `selection.py` and `metadata.py` own the
  shared input-selection and reproducibility helpers.
- `datasets/` owns task data formats, prompts, generators, rules and evaluators.
- `agents/` and `core/prompts.py` own reasoning architectures and prompt versions.
- `scoring/` builds offline capability scores from existing observations.
- `visualization/` displays tasks and run artifacts; it does not change scoring.

Multi-step tasks already persist trajectories and intermediate results. Preserve
their task-specific protocols rather than imposing a new environment abstraction.
Resume checks use input/configuration/source fingerprints and reject version
mixing. Archived data and results are not silently migrated.
