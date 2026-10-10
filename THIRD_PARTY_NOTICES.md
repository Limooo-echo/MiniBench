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

## Chinese Chess Practical Dataset (CCPD)

The Xiangqi corpus derives source positions from
[Yvonne761/Chinese-Chess-Practical-Dataset](https://github.com/Yvonne761/Chinese-Chess-Practical-Dataset),
revision `368a47a947773dd8692c026e286dd19b6277b993`, under CC-BY-4.0.
Source PGN bytes in `data/xiangqi/sources/ccpd` are unmodified; MiniBench selects
positions and adds task records, reference lines, difficulty and paired rules.
Attribution, original paths, source hashes and a license copy are included there.
This source license does not by itself license all MiniBench-authored code/data.

## Pikafish (external dependency)

[official-pikafish/Pikafish](https://github.com/official-pikafish/Pikafish) is an
external engine under GPL-3.0-or-later. `scripts/cloud_pikafish.sh` fetches its
pinned official source and network for local compilation into ignored `vendor`.
No engine binary is bundled or claimed as MiniBench-authored code; any separate
redistribution must retain and satisfy its own license/source obligations.

## MiniBench licensing status

The team has not yet selected a top-level license for MiniBench-authored work.
No new project-wide license is inferred from the third-party notices above.
