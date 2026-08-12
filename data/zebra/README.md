# Zebra data

MiniBench expects ZeroEval-compatible JSONL records with `id`, `size`, `puzzle`,
`solution.header`, and `solution.rows`. Optional extension fields are
`capability`, `rule_context`, and `clue_turns`.

The original public `allenai/ZebraLogicBench` test split masks every gold
solution cell as `___`. The importer therefore uses the public
`WildEval/ZebraLogic` mirror, whose `grid_mode/test` records include the real
solution grids.

`tasks.jsonl` is the three-task smoke set. Refresh it with:

```bash
python scripts/import_zebra_tasks.py --smoke-per-difficulty 1
```

`eval.jsonl` is the formal 45-task direct-reasoning set: 15 tasks per
difficulty. It excludes the smoke ids, covers every grid size, and balances
low/middle/high clue-count thirds within each size. Reproduce it from the local
dataset archive with:

```bash
python scripts/import_zebra_tasks.py \
  --source-parquet "D:/AAALimoWork/CS/Seminar/PAPER/Agent/ZebraLogic/grid_mode/test-00000-of-00001.parquet" \
  --evaluation-per-difficulty 15 \
  --exclude-task-file data/zebra/tasks.jsonl \
  --seed 20260810 \
  --overwrite
```

The importer rejects masked solutions. Both smoke and evaluation selections
are deterministic for a fixed seed.

## Paired variants

The frozen 45 source ids in `eval.jsonl` are also used by:

- `rule_codebook_eval.jsonl`: scoreable temporary-codebook rules. The clue
  text uses per-puzzle `[R1]`, `[R2]`, ... macros and keeps the original gold.
- `history_eval.jsonl`: scoreable history tasks. The background stays in
  `puzzle`, while the official clues move to `clue_turns` in their original
  order.
- `rule_counterfactual_candidates.jsonl`: 45 counterfactual rule candidates
  awaiting manual/solver review. Every record deliberately has `solution:
  null`, `validation_status: pending_manual_review`, and a `not-scoreable` tag.

Regenerate all three paired files with:

```bash
python scripts/derive_zebra_variants.py --seed 20260810 --overwrite
```

Do not point an experiment config at the counterfactual candidate file. A
candidate becomes scoreable only after its new solution is independently
verified to be unique and different from the original solution.
