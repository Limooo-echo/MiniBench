# Zebra data

MiniBench expects ZeroEval-compatible JSONL records with `id`, `size`, `puzzle`,
`solution.header`, and `solution.rows`. Optional extension fields are
`capability`, `rule_context`, and `clue_turns`.

The original public `allenai/ZebraLogicBench` test split masks every gold
solution cell as `___`. The importer therefore uses the public
`WildEval/ZebraLogic` mirror, whose `grid_mode/test` records include the real
solution grids.

Refresh the three-task smoke set with:

```bash
python scripts/import_zebra_tasks.py --smoke-per-difficulty 1
```

The importer rejects masked solutions and writes easy, medium, and hard records
deterministically.
