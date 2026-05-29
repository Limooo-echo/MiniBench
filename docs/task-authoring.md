# Task Authoring Guide

MiniBench currently has three task families:

- Multiple-choice tasks in contributor-specific files such as `data/tasks-limo.jsonl`.
- Xiangqi environment tasks in `data/xiangqi_tasks.jsonl` and `data/xiangqi_hard_tasks.jsonl`.
- One-stroke graph puzzles in `data/one_stroke_tasks.jsonl`.

## Multiple-Choice Tasks

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

- `answer_extractors`: regexes for unusual outputs. If omitted, default A-D
  extractors are used.
- `prompt_constraints`: output rules. If omitted, default JSON-only constraints
  are used.

## Xiangqi Tasks

Xiangqi tasks describe a `gym-xiangqi` board plus the side and goal to evaluate.
Simple tasks usually have no opponent and expect the agent to find a one-step
winning action. Hard tasks can set `opponent` to `pikafish`.

```json
{"id":"xq-capture-general-011","board":[[0,0,0,0,-1,0,0,0,0],[0,0,0,0,0,0,0,0,0]],"side_to_move":"ally","agent_side":"ally","goal":"capture_enemy_general","opponent":"none","max_steps":1,"tags":["xiangqi","endgame","difficulty:easy"]}
```

The real board must contain 10 rows and 9 columns. Use existing Xiangqi files as
templates because piece ids are inherited from `gym-xiangqi`.

Important fields:

- `side_to_move`: `ally` or `enemy`.
- `agent_side`: side controlled by the tested agent.
- `goal`: currently `capture_enemy_general` or `agent_win`.
- `opponent`: `none` or `pikafish`.
- `max_steps`: maximum environment steps before the task is failed.

## One-Stroke Tasks

One-stroke tasks are undirected graph puzzles. The agent must return a vertex
path that traverses every listed edge exactly once.

```json
{"id":"os-example-011","vertices":["A","B","C","D"],"edges":[["A","B"],["B","C"],["C","D"]],"start":"A","end":"D","tags":["one-stroke","euler-trail","difficulty:easy"]}
```

Required fields:

- `id`: unique task id. Use `os-...`.
- `vertices`: unique vertex labels.
- `edges`: non-empty list of two-vertex undirected edges.
- `tags`: normalized tags.

Optional fields:

- `start`: required first vertex.
- `end`: required final vertex.

The loader validates that the graph has a one-stroke solution under the supplied
start and end constraints. Parallel edges are accepted by the evaluator if they
appear as repeated edge entries, but self-loops are not supported.

## Tag Schema

Use flat tags with a `prefix:value` pattern where useful. Multiple-choice tasks
should keep the original normalized groups:

- `format:multiple-choice`
- `turn:single`
- `source:synthetic`, `source:agentboard-inspired`, or `source:swebench-inspired`
- `domain:<domain>`
- `skill:<skill>`
- `difficulty:easy`, `difficulty:medium`, or `difficulty:hard`

Environment and game tasks can add task-family tags such as:

- `xiangqi`
- `one-stroke`
- `euler-trail`
- `euler-circuit`
- `pikafish-opponent`
- `difficulty:easy`, `difficulty:medium`, or `difficulty:hard`

## Validation

After editing tasks, run:

```powershell
$env:PYTHONPATH="src"
python -m unittest discover -s tests
python -m minibench.cli evaluate --agent oracle
```

In WSL:

```bash
cd /mnt/d/benchmark/MiniBench
export PYTHONPATH=src
python3 -m unittest discover -s tests
```

Inspect one multiple-choice prompt:

```powershell
python -m minibench.cli show-prompt mb-choice-051
```
