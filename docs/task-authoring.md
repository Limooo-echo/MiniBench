# Task Authoring Guide

This guide defines how to add multiple-choice tasks to `data/tasks.jsonl`.
The current benchmark is intentionally simple: every task is a single-turn
question with four options, and the evaluator scores only the selected option
letter.

## Task Shape

Each JSONL line is one task:

```json
{"id":"mb-choice-051","question":"Question text goes here.","options":{"A":"Option A","B":"Option B","C":"Option C","D":"Option D"},"correct_option":"B","tags":["format:multiple-choice","turn:single","source:synthetic","domain:tool-use","skill:tool-selection","difficulty:easy"]}
```

Required fields:

- `id`: unique task id. Use `mb-choice-###`.
- `question`: concise question text.
- `options`: exactly four options labeled `A`, `B`, `C`, and `D`.
- `correct_option`: one of `A`, `B`, `C`, or `D`.
- `tags`: normalized tags for later analysis.

Optional fields:

- `answer_extractors`: regexes for unusual outputs. If omitted, the default
  A-D extractors are used.
- `prompt_constraints`: output rules. If omitted, the default JSON-only
  constraints are used.

## Tag Schema

Use flat tags with a `prefix:value` pattern. Every task should have exactly one
tag from each required group.

Required groups:

- `format:multiple-choice`
- `turn:single`
- `source:synthetic`, `source:agentboard-inspired`, or `source:swebench-inspired`
- `domain:<domain>`
- `skill:<skill>`
- `difficulty:easy`, `difficulty:medium`, or `difficulty:hard`

Recommended domains:

- `domain:agent-evaluation`
- `domain:prompting`
- `domain:tool-use`
- `domain:planning`
- `domain:self-reflection`
- `domain:software-engineering`
- `domain:web-information`
- `domain:world-modeling`

Recommended skills:

- `skill:metric-understanding`
- `skill:format-following`
- `skill:tool-selection`
- `skill:planning`
- `skill:self-reflection`
- `skill:test-understanding`
- `skill:information-extraction`
- `skill:state-tracking`
- `skill:grounding`
- `skill:world-modeling`
- `skill:patch-reasoning`
- `skill:error-diagnosis`

Optional secondary tags are allowed when useful:

- `topic:progress-rate`
- `topic:trajectory`
- `topic:partial-observability`
- `topic:json`
- `topic:regex`
- `topic:docker`
- `topic:patch`

## Category Plan

The first seed set targets eight buckets:

1. Agent evaluation concepts: success rate, progress rate, grounding,
   trajectories, partial observability.
2. Prompt and JSON format following: machine parsing, invalid extra text,
   schema compliance.
3. Tool selection: search, calculator, file read, code execution, database query.
4. Planning: choose the next action in a small goal-directed sequence.
5. Self-reflection: identify the mistake in a failed trajectory.
6. SWE-bench-style software engineering: failing tests, patches, containers,
   repo state, resolution.
7. Web and information extraction: read short webpage-like snippets and extract
   the correct value.
8. World modeling: track state changes after actions.

## Quality Rules

- Keep the question answerable from the prompt alone.
- Make exactly one option clearly correct.
- Avoid trick wording unless the task is explicitly about instruction following.
- Keep option lengths roughly balanced.
- Do not use real private data, secrets, credentials, or copyrighted dataset
  instances copied from another benchmark.
- Prefer small synthetic scenarios inspired by mature benchmark patterns.

## Validation

After editing tasks, run:

```powershell
$env:PYTHONPATH="src"
python -m unittest discover -s tests
python -m minibench.cli evaluate --agent oracle
```

Use `show-prompt` to inspect a task:

```powershell
python -m minibench.cli show-prompt mb-choice-051
```

