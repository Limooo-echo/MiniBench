# MiniBench one-stroke tasks

Every undirected edge must be used exactly once; vertices may be revisited.
The formal tracks use semantic names throughout task IDs, configs, and scores.

## Files and protocols

- `direct.jsonl`: 100 direct-reasoning tasks, with 34 easy, 33 medium, and
  33 hard tasks. Easy contains 24 solvable and 10 unsolvable graphs; medium
  and hard each contain 23 solvable and 10 unsolvable
  graphs. IDs use `direct-<difficulty>-<NN>`. The formal `baseline` prompt
  contains no Euler-theorem hint.
- `history.jsonl`: 100 history-memory tasks, with 34 easy, 33 medium and 33 hard, with IDs
  `history-<difficulty>-<NN>` and capability `history_memory`. Evaluation
  runs each graph under both `incremental_state` and `step_history_only`,
  yielding 200 instance results from the same 100 graphs and event sequences.
- `multimodal.jsonl`: 100 tasks paired to direct tasks through
  `source_task_id`. IDs use `multimodal-<difficulty>-<NN>`. Each record has
  one `image` path, relative to the JSONL file, pointing to
  `images/<task-id>.png`. The renderer version is `multimodal-v3`.
  Main evaluation uses `image` (100 results); paired ablation uses
  `text` and `image` (200 results). Graph difficulty remains easy/medium/hard;
  there are no separate image difficulty variants.
- `tasks.jsonl`: the earlier compatibility/smoke set.
- `tasks_generated.jsonl`: the existing generated set, retained independently.

History tasks identify parallel edges as `e01`, `e02`, and so on in original
edge-list order. Each event supplies the current vertex and its original static
incident edges, but never the authoritative used-edge set. In
`incremental_state`, the agent writes that state after each event; in
`step_history_only`, it may acknowledge only the step number. Hard tasks include
a reversible exploratory move followed by a strict LIFO undo event. The move
is different from the selected oracle step; it is not necessarily a dead end.

## Provenance and reproducibility

The original 30 direct records and 29 distinct history inputs are preserved.
The exact duplicate `history-easy-09` is retired in favor of `history-easy-03`.
All original 30 history records remain in `legacy/pre-20261013/history.jsonl`
for inspecting old results.

New easy direct tasks are sampled deterministically from the BSD-3-Clause
NetworkX Graph Atlas; the pinned archive and license are bundled under
`sources/`. Its SHA-256 is
`73fc416df0164923607751cb759f4ae81deb5f6550bf25be59c86de3b747e41d`.
Other new tasks use independent seeded combinations of cycles, chords and,
where allowed, parallel edges. Raw unlicensed Cocos/Yuyaoo levels are not copied.
Both generators use seed `20261013`.

Direct vertex counts are easy 4–6, medium 7–9 and hard 9–12; only hard direct
graphs use parallel edges. History vertex counts use the same ranges and event
counts are easy 4–6, medium 7–12 and hard 12–20. History graphs may contain
parallel edges. New candidates reject endpoint-aware multigraph isomorphism
against retained baseline graphs and other new graphs. Existing graph-isomorphic
histories with different events remain, except the exact retired duplicate.

Regenerate the formal files and images with:

```powershell
python scripts/build_one_stroke_direct_history.py --total 100 --seed 20261013
python scripts/build_one_stroke_multimodal.py --overwrite
```

The builder cross-checks the production loader with NetworkX MultiGraph Euler
paths under fixed endpoints. It verifies full oracle edge multisets, independently
replays each history move and LIFO undo, and saves a valid residual completion.
Unsolvable records include connectivity, odd-degree or endpoint proofs in
`validation-20261013.json`; per-image pairs and hashes are in
`multimodal-validation-20261013.json`.

New direct candidates also pass deterministic visual clearance checks against
the eventual image-ID layout: nodes remain separated and nonincident nodes stay
clear of straight and parallel Bezier edges. All 100 images receive visual QA.
Original 30 image bytes remain preserved under renderer `multimodal-v3`.

## Scoring and interpretation

- `direct_score` is direct path/no-solution accuracy. Any valid Euler trail is
  accepted, not only the stored oracle. Here `direct` names the benchmark
  track; the shipped config uses the `passthrough` agent.
- History `success` and `history_final_score` measure final completion.
  The official `history_score` is `history_joint_score`: final completion,
  final response schema, all intermediate response schemas, and (for
  `incremental_state`) every recorded state must all be correct.
  `history_intermediate_protocol_score` and `history_protocol_score` expose
  intermediate-only and full-protocol validity, while `history_state_score`
  measures recorded state accuracy. The current set contains no history that
  ends without a valid completion and measures within-transcript context tracking.
- Multimodal reports `multimodal_path_score`,
  `multimodal_transcription_score`, and `multimodal_joint_score` with equal
  weighting across represented graph difficulties. `multimodal_score` is an
  alias for path score. Paired `text`/`image` results estimate the visual gap.
- `json_format_exact_rate` checks that the complete response is one strict
  JSON object; `response_schema_valid_rate` also requires the exact fields
  requested by the track. Lenient answer extraction remains available for
  semantic accuracy.

Each configured run creates a manifest and empty checkpoint before the first
model call, then atomically checkpoints each completed task/mode item. The
manifest records resolved config, provider/model identity, data/code/image
fingerprints, Git state, and the work plan. Protocol-generation totals are
direct=100, history=2198, and multimodal main=100/paired=200; the history total
is computed as `2 * sum(len(history_events) + 1)`; reasoning wrappers may
add internal calls on final turns. Compare tracks only with appropriately
paired designs and controlled model identities; shipped text and image configs
use different model families.

## Migration

The rule-conditioned track has been removed. Old numbered paths, task IDs,
score fields, and image modes have no aliases in the new interface. Existing
run artifacts remain unchanged; create a new run for the new dataset and
protocol instead of resuming an old run. Formal configs default to a fresh
run name with `on_existing: error`. The smoke and generated collections remain
available under their existing names.
