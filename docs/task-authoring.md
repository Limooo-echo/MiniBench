# Task Authoring Guide

MiniBench currently has several task families:

- Multiple-choice tasks in contributor-specific files such as `data/tasks-limo.jsonl`.
- Xiangqi environment tasks in `data/xiangqi_tasks.jsonl` and `data/xiangqi_hard_tasks.jsonl`.
- One-stroke graph puzzles in `data/one_stroke_tasks.jsonl`.
- Riichi Mahjong tile-shape tasks in `data/mahjong_tasks.jsonl`.
- Local Riichi Mahjong v1 table tasks in `data/mahjong_riichi_tasks.jsonl`.

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
- `goal`: currently `capture_enemy_general`, `agent_win`, or `agent_survive`.
- `opponent`: `none` or `pikafish`.
- `max_steps`: maximum environment steps before the task is failed.

Hard Pikafish tasks should use category tags:

- `category:tactical-win`: the agent side has a forced tactical win.
- `category:advantage-play`: the agent side starts clearly ahead and should not
  throw the position.
- `category:survival-defense`: the agent side starts worse and should survive
  the move horizon.

`agent_win` succeeds when the agent captures the opposing general, or when the
agent's last move leaves Pikafish with `bestmove (none)`. `agent_survive`
succeeds when the agent reaches `max_steps` without illegal moves or an opponent
win; an agent win also succeeds.

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

## Mahjong Tasks

Mahjong tasks use Riichi tile notation and are scored with the `mahjong` Python
package. They currently test tile-shape reasoning, not full yaku or score
calculation.

```json
{"id":"mj-example-009","goal":"winning_tiles","hand":["2m","3m","4p","5p","6p","7s","8s","9s","E","E","E","5m","5m"],"tags":["mahjong","riichi","goal:winning_tiles","difficulty:easy"]}
```

Required fields:

- `id`: unique task id. Use `mj-...`.
- `goal`: `winning_tiles` or `tenpai_discard`.
- `hand`: tile list.
- `tags`: normalized tags.

Tile notation:

- `1m`-`9m`: characters/manzu.
- `1p`-`9p`: dots/pinzu.
- `1s`-`9s`: bamboo/souzu.
- `E`, `S`, `W`, `N`: winds.
- `P`, `F`, `C`: white, green, and red dragons.

For `winning_tiles`, the hand should have 13 tiles and the agent must return
`{"winning_tiles":["3m","6m"]}`. For `tenpai_discard`, the hand should have 14
tiles and the agent must return `{"discard":"5m"}`. The loader validates that
each task has at least one correct answer.

## Riichi Mahjong V1 Tasks

Riichi v1 tasks use a local four-player table with one LLM-controlled seat and
three shanten-based API bots. Compared with `mahjong-table`, this path adds
riichi declarations, chi/pon/kan calls, tsumo, ron, yaku/fu/point checks, and
score deltas through the `mahjong` Python package.

The built-in file contains seeded hands. Each task is random-looking but
reproducible because its wall shuffle is derived from the `seed` field.

Riichi v1 writes both a strict win flag and softer per-seat benchmark scores.
The strict `success` flag is true only when seat 0 wins. The per-seat score gives
`1.0` to the winner, `0.25` to non-winners on another player's tsumo, `0.5` to
all seats on a normal draw, and `0.0` to the deal-in seat on ron while uninvolved
seats receive a rank-based survival score.

```json
{"id":"mj-riichi-006","seed":666,"agent_seat":0,"max_draws":70,"starting_scores":[25000,25000,25000,25000],"tags":["mahjong","riichi-full-v1","four-player","difficulty:medium"]}
```

Required fields:

- `id`: unique task id. Use `mj-riichi-...`.
- `seed`: integer seed for the reproducible wall shuffle.
- `tags`: normalized tags.

Optional fields:

- `agent_seat`: currently expected to be `0`.
- `max_draws`: maximum draw events before the hand is treated as a draw.
- `starting_scores`: four integer scores, defaulting to 25000 each.

Agent actions:

- `{"action":"discard","tile":"5m"}`
- `{"action":"riichi","discard":"5m"}`
- `{"action":"kan","tile":"5m"}` on a legal closed kan turn

When another player discards a tile that seat 0 can call, MiniBench prompts the
agent separately for a call decision:

- `{"action":"pass"}`
- `{"action":"chi","tiles":["3m","4m","5m"]}`
- `{"action":"pon","tile":"5m"}`
- `{"action":"kan","tile":"5m"}`

This v1 path does not yet implement dora, ura-dora, ippatsu, added-kan/chankan,
multi-ron, or complete round bookkeeping.

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
- `mahjong`
- `four-player`
- `riichi-full-v1`
- `riichi`
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
cd /path/to/MiniBench
export PYTHONPATH=src
python3 -m unittest discover -s tests
```

Inspect one multiple-choice prompt:

```powershell
python -m minibench.cli show-prompt mb-choice-051
```
