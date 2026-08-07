# Task Authoring Guide

MiniBench currently has several task families:

- Multiple-choice tasks in contributor-specific files such as `data/tasks-limo.jsonl`.
- Xiangqi environment tasks in `data/xiangqi_tasks.jsonl` and `data/xiangqi_hard_tasks.jsonl`.
- One-stroke graph puzzles in `data/one_stroke_tasks.jsonl`.
- Riichi Mahjong waiting-tile tasks in `data/mahjong/tasks.jsonl`.

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
{"id":"mj-easy-wait-001","goal":"winning_tiles","hand":["3p","3p","2s","3s","4s","4s","5s","6s","7s","9s","W","W","W"],"tags":["easy","task:winning_tiles"]}
```

Required fields:

- `id`: unique task id. Use `mj-...`.
- `goal`: `winning_tiles`, `max_wait_discard`, or `max_ukeire_discard`.
- `hand`: tile list.
- `tags`: exactly one difficulty tag followed by the matching task-type tag.

Tile notation:

- `1m`-`9m`: characters/manzu.
- `1p`-`9p`: dots/pinzu.
- `1s`-`9s`: bamboo/souzu.
- `E`, `S`, `W`, `N`: winds.
- `P`, `F`, `C`: white, green, and red dragons.

For `winning_tiles`, the hand has 13 tiles and the agent returns the complete
`winning_tiles` list. For `max_wait_discard`, the hand has 14 tiles and the
agent returns one best `discard`. Tied best discards are all accepted. To keep
this task different from a plain tenpai-discard question, author each hand with
at least two tenpai discards whose winning-tile counts differ. The loader
validates that each task has at least one correct answer.

Visual tasks additionally provide `visible_tiles`, `table_columns`, and an
optional relative `image` path. Their tags end with `visual`. For
`max_ukeire_discard`, the correct discard maximizes the total number of live
winning copies rather than the number of distinct wait types. Each tile has
four copies; copies in the remaining hand, `visible_tiles`, and the selected
discard are unavailable. The loader rejects any hand/table combination that
contains more than four physical copies of a tile.

When `image` is present, `load_mahjong_tasks` resolves it relative to the JSONL
file and the agent sends the raster image as an OpenAI-compatible `image_url`
data URL. Use PNG, JPEG, WebP, or GIF and a vision-capable provider model. The
visual prompt intentionally omits the textual tile identities.

A full flush contains numbered tiles from exactly one suit and no honors. The
built-in matrix contains 15 tasks for each combination of goal and difficulty:
non-full-flush is `easy`, while full-flush is `hard`.

## Mahjong Rule-Variant Tasks

Rule-variant evaluation reuses ordinary single-player draw-discard tasks. Each
source task is evaluated once under each rule channel, with the same initial
hand, seed, fixed wall, and draw limit.

```json
{"id":"mj-solo-001","seed":12345,"initial_hand":["1m","2m","3m","4p","5p","6p","7s","8s","9s","E","E","N","N"],"wall":["C","4m"],"max_draws":2,"round_wind":"E","seat_wind":"E","tags":["mahjong","solo-draw-discard"]}
```

Required fields:

- `id`: unique task id.
- `initial_hand`: exactly 13 concealed tiles.
- `wall`: fixed hidden draw sequence with at least `max_draws` tiles.
- `max_draws`: maximum number of draw-discard turns.
- `seed`, `round_wind`, `seat_wind`, and `tags`: the same fields used by
  ordinary Mahjong solo tasks.

The loader expands each source record into eight channels: a `standard`
baseline, each of `no_cross_suit_duplicate_sequences`, `cyclic_sequences`, and
`red_dragon_wildcard` alone, all three two-rule combinations, and the three-rule
combination. Every channel uses the same draw-discard loop, prompt template, and
correction mechanism. On every turn the agent returns either
`{"action":"tsumo"}` or a discard from its current hand. The local evaluator
applies all rules selected for that channel simultaneously without exposing
legality or future wall tiles in the prompt. Results label wins that are
possible only under the modified configuration and standard wins blocked by a
restrictive configuration.

The CLI supports `--observation-mode full-hand` and `--observation-mode
history-only` for all eight channels. Full-hand mode lists the current 14 tiles.
History-only mode instead provides the initial hand, current draw, completed
draw/discard history, cumulative discards, and remaining draw count. The
underlying hand state and rule validator are unchanged.

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

The static Mahjong set is the exception: use exactly `easy` or `hard` followed
by `task:winning_tiles` or `task:max_wait_discard`. Do not add task-family,
shape, or prefixed difficulty tags.

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
