# Third-party notices

## WildEval/ZeroEval

MiniBench's Zebra prompt and JSON-grid template are adapted from
`src/templates/ZEBRA_GRID.py` and `src/_TEMPLATES.py` in
[WildEval/ZeroEval](https://github.com/WildEval/ZeroEval). The last-complete-JSON
parser and scoring semantics are adapted from `src/evaluation/eval_utils.py` and
`src/evaluation/zebra_grid_eval.py` in the same project.

ZeroEval is licensed under the Apache License 2.0. A copy is included at
`licenses/ZeroEval-APACHE-2.0.txt`. MiniBench modifications include typed local
data models, MiniBench difficulty mapping, run summaries, and reusable true
multi-turn provider support.

The Zebra smoke records come from the `grid_mode/test` split of
[WildEval/ZebraLogic](https://huggingface.co/datasets/WildEval/ZebraLogic).

## Noto Sans CJK SC

MiniBench includes glyph-subset builds of Noto Sans CJK SC Regular and Bold for
portable Xiangqi rendering. The original fonts are published by the Noto CJK
project under the SIL Open Font License 1.1. A copy is included at
`licenses/NotoSansCJK-OFL-1.1.txt`; source URLs, checksums, subset coverage, and
generated-file checksums are recorded in
`src/minibench/assets/fonts/PROVENANCE.md`.
