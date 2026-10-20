# Zebra data

The formal release uses the locally downloaded public `WildEval/ZebraLogic`
`grid_mode/test` pool. Its 1,000 records contain real answer grids; the original
`allenai/ZebraLogicBench` test split masks answers and cannot be used for scoring.
The archive is at
`D:/AAALimoWork/CS/Seminar/Dataset/ZebraLogic/grid_mode/test-00000-of-00001.parquet`.
Its reviewed SHA256 is
`545cd5cce9c54575c7f74769b8f1250b8bb94f5745519c83b99b8db01f95924d`.

## Formal release

| File | Records | Source and answer relationship |
| --- | ---: | --- |
| `eval.jsonl` | 100 | Direct tasks; retains all original 45 raw records |
| `rule_codebook_eval.jsonl` | 100 | Same 100 direct sources and gold, reversible per-puzzle macros |
| `rule_counterfactual_eval.jsonl` | 100 | Independently solved changed rules; new unique gold |
| `rule_eval.jsonl` | 200 | Codebook and Counterfactual together, each variant exactly once |
| `history_eval.jsonl` | 100 | Same direct sources, gold, and official clue order |

Each standalone set has 34 easy, 33 medium, and 33 hard records. Source selection
covers all 25 grid sizes with counts differing by at most one within difficulty,
and balances low/middle/high clue-count thirds. Those thirds are frozen against
the complete public pool before excluding smoke tasks or failed candidates.
Retained records remain unchanged; the release manifest records their current
full-pool bands separately from legacy sampling tags.

Codebook and Counterfactual share 69 source IDs. The remaining 31 Counterfactual
sources come from the same original pool. Counterfactual requires the new answer
to differ from the original, so it cannot always reuse the direct source set.
Use the manifest's intersection for paired comparisons and all 100 records for
each mode's full score. The combined rule module gives each rule mode 50% weight.

History runs the same 100 records under `incremental_state` and
`deferred_reasoning`, producing 200 evaluation instances. Source IDs, clue order,
and final gold remain paired. `tasks.jsonl` remains the three-record smoke set.

## Independent verification

`src/minibench/datasets/zebra/validation.py` reads each attribute domain from the
puzzle background and parses every complete clue using a reviewed finite grammar
and explicit entity aliases. It does not infer relationships from the answer.
Z3 assigns integer house positions with per-category `AllDifferent` constraints,
finds a solution, blocks that entire model, and requires the second check to be
UNSAT. A separate integer evaluator checks every clause on the first model.
Only then is the independently computed grid compared against official gold.
Unparsed or ambiguous clues, inconsistent domains, UNSAT, multiple solutions,
unknown/timeout, and gold mismatch fail closed.

All 1,000 originals passed. The builder enumerated all 3,286 applicable rule
overrides, transforming every matching parsed clause including negations. There
were 898 unique-answer variants, 778 UNSAT variants, and 1,610 multiple-answer
variants; 660 source puzzles supplied at least one unique changed answer. The
eight declared operators were considered; `directly_right_of` has no applicable
clues in this source pool, so seven operators appear in the accepted release. The
accepted Counterfactual records have `validation_status: verified_unique`, new
`solution`, original gold in audit metadata, affected clue indices, and proof
evidence. No pending candidate is directly promoted.

`validation.jsonl` contains a certificate for every standalone formal record.
`pool_proofs.json` records the original and every applicable override attempt.
`release_manifest.json` records source hashes, selections, fills, intersection,
quotas, operator counts, and artifact hashes. Old 45-record releases and pending
candidates are preserved under `legacy/20260810/`; the old pending candidate file
remains explicitly unscoreable and is excluded from experiment configs.

## Reproduce

Install generation dependencies with `pip install -e ".[zebra-generation]"`.
Stage and inspect a release before replacing the formal files:

```bash
python scripts/build_zebra_release.py --target-total 100 --seed 20261013
python scripts/build_zebra_release.py --target-total 100 --seed 20261013 --publish
```

The builder verifies the source hash, reuses proofs only when the validator hash
and timeout match, stages and re-verifies every formal record, and publishes only
after all count, uniqueness, gold, pairing, and loader checks pass. Rebuilding
uses the frozen legacy 45 records to reproduce the same selection. It makes no
model calls. The old importer and derivation APIs remain available:

```bash
python scripts/import_zebra_tasks.py --source-parquet "D:/AAALimoWork/CS/Seminar/Dataset/ZebraLogic/grid_mode/test-00000-of-00001.parquet" --target-total 100 --seed 20261013 --output tmp/zebra-import.jsonl
python scripts/derive_zebra_variants.py --source data/zebra/eval.jsonl --target-total 100 --output-dir tmp/zebra-variants --seed 20261013
```

The generic importer selects raw records; it does not replace the certified
builder. The generic derivation command still produces explicitly unscoreable
Counterfactual candidates, and cannot certify new Counterfactual gold.
