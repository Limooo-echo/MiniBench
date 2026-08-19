# Xiangqi schema-v2 data card

## Scope

MiniBench 0.2.0 ships 1,000 Xiangqi endgame records: 250 records in each of
`xiangqi-mate-in-one`, `xiangqi-rule-variants`, `xiangqi-history`, and
`xiangqi-multimodal`. The positions, answers, oracle values, and scoring
semantics are preserved from the earlier corpus; names and storage schema are
new.

FEN is the sole persisted board representation. Loaders assign stable internal
piece IDs in row-major FEN order only while a task is running. Every record has
`schema_version: 2`, a family-prefixed ID, an active color matching
`agent_color`, both generals, a positive `max_plies`, an exact `piece_count`, a
normalized `oracle` object, and sorted unique non-empty `tags`.

## Coordinates and notation

Files are `a` through `i` from Red's left to right; ranks are `0` through `9`
from Red's home side toward Black. UCI moves concatenate the start and end
squares: `a0a1`. A FEN record lists ranks 9 down to 0. Uppercase FEN pieces are
Red; lowercase pieces are Black. `w` means Red to move and `b` means Black.

`max_plies` counts half-moves, not full move pairs. `oracle.best_move_uci` is
the preferred first move, `oracle.mate_in_plies` is the preserved mating
horizon when available, and `oracle.evaluation_cp` is nullable.

## Families

### `xiangqi-mate-in-one`

```text
      a  b  c  d  e  f  g  h  i
  9   ·  ·  ·  ·  ·  将  ·  ·  ·
  8   ·  ·  ·  ·  ·  ·  ·  ·  ·
  7   ·  ·  ·  ·  ·  ·  ·  马  ·
  6   ·  ·  ·  ·  ·  ·  ·  ·  ·
  5   ·  ·  ·  ·  ·  ·  ·  炮  ·
  4   ·  ·  ·  ·  ·  ·  ·  ·  ·
  3   ·  ·  车  ·  兵  ·  ·  ·  兵
  2   ·  ·  ·  ·  ·  ·  ·  ·  ·
  1   ·  ·  ·  ·  帅  ·  ·  ·  ·
  0   ·  ·  ·  ·  ·  ·  ·  ·  ·
```

The evaluator scores legal action selection, oracle agreement, checkmate goal
achievement, and centipawn loss. Sampling is difficulty-stratified; remainders
are assigned in easy → medium → hard order.

```bash
minibench run-task xiangqi-mate-in-one --agent openai-compatible \
  --sample-seed 42 --sample-count 10 --pikafish-depth 8
```

### `xiangqi-rule-variants`

```text
      a  b  c  d  e  f  g  h  i
  9   ·  ·  象  士  ·  将  ·  ·  ·
  8   ·  ·  ·  ·  士  ·  ·  ·  ·
  7   ·  ·  ·  ·  ·  ·  ·  ·  ·
  6   卒  ·  ·  ·  卒  ·  ·  ·  炮
  5   ·  ·  ·  ·  ·  ·  ·  ·  ·
  4   车  ·  ·  ·  ·  ·  ·  ·  ·
  3   兵  ·  车  ·  ·  ·  ·  ·  兵
  2   ·  ·  ·  ·  ·  ·  ·  ·  ·
  1   ·  ·  ·  ·  帅  ·  ·  ·  ·
  0   ·  ·  ·  仕  ·  仕  ·  ·  ·
```

Each record contains a shared `scenario_id`, one of the four `ruleset` values,
and a structured rule list. The modified rules are horse leg-block ignored,
chariots forbidden from center files, and soldiers allowed to retreat after
crossing the river. Sampling preserves ruleset proportions with the largest
remainder method.

```bash
minibench run-task xiangqi-rule-variants --agent openai-compatible \
  --sample-seed 42 --sample-count 10 --search-depth 3
```

### `xiangqi-history`

```text
      a  b  c  d  e  f  g  h  i
  9   ·  ·  ·  ·  ·  将  ·  ·  ·
  8   ·  ·  ·  ·  ·  ·  ·  ·  ·
  7   ·  ·  ·  ·  ·  ·  ·  ·  ·
  6   ·  ·  车  ·  ·  ·  ·  ·  ·
  5   ·  ·  ·  ·  ·  ·  ·  ·  ·
  4   ·  ·  ·  ·  ·  ·  炮  ·  ·
  3   ·  砲  ·  ·  ·  ·  ·  ·  ·
  2   ·  ·  ·  ·  ·  ·  ·  ·  ·
  1   ·  ·  ·  帅  ·  ·  ·  ·  ·
  0   ·  ·  ·  ·  ·  ·  ·  ·  ·
```

`full-state` shows the current board each turn. `move-history-only` shows the
initial state and move history, requiring the model to reconstruct the board.

```bash
minibench run-task xiangqi-history --agent openai-compatible \
  --sample-seed 42 --sample-count 10 --history-mode full-state \
  --pikafish-depth 8 --pikafish-timeout 60
```

### `xiangqi-multimodal`

```text
      a  b  c  d  e  f  g  h  i
  9   ·  ·  ·  将  ·  ·  ·  ·  ·
  8   ·  ·  ·  ·  士  ·  ·  ·  ·
  7   ·  ·  ·  ·  象  ·  ·  车  ·
  6   卒  ·  炮  ·  ·  ·  ·  ·  卒
  5   ·  ·  ·  ·  ·  ·  ·  ·  ·
  4   ·  ·  ·  ·  ·  ·  ·  ·  马
  3   兵  ·  ·  ·  ·  ·  ·  ·  兵
  2   相  ·  ·  ·  ·  ·  车  ·  相
  1   ·  ·  ·  ·  帅  ·  ·  ·  ·
  0   ·  ·  ·  仕  ·  仕  ·  ·  ·
```

Input modes are `text`, `chinese-piece-image`, and `latin-piece-image`. Both
image modes use the bundled MiniBench Noto Sans CJK SC subset, so no operating
system font installation is required.

```bash
minibench run-task xiangqi-multimodal --agent openai-compatible \
  --sample-seed 42 --sample-count 10 \
  --input-modes text,chinese-piece-image,latin-piece-image
```

## Inspection and reproducibility

Use `minibench inspect-xiangqi` for terminal, formatted JSON, or PNG output.
`minibench build-xiangqi-gallery` embeds all FEN records, the font subset, CSS,
and JavaScript in one offline HTML file. Runs store the resolved configuration,
data SHA256, schema version, renderer version, and relevant dependency versions.

The checked-in migration map covers all 1,000 task IDs and all 70 shared rule
scenarios. The migration command refuses in-place operation and existing output,
supports dry runs, reports unknown IDs, and never rewrites model free text.
