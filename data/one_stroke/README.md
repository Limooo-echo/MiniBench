# MiniBench one-stroke tasks

Every undirected edge must be used exactly once; vertices may be revisited.
The formal tracks use semantic names throughout task IDs, configs, and scores.

## Files and protocols

- `direct.jsonl`: 30 direct-reasoning tasks, with 10 easy, 10 medium, and
  10 hard tasks. Each difficulty contains seven solvable and three unsolvable
  graphs. IDs use `direct-<difficulty>-<NN>`. The formal `baseline` prompt
  contains no Euler-theorem hint.
- `history.jsonl`: 30 history-memory tasks, again 10 per difficulty, with IDs
  `history-<difficulty>-<NN>` and capability `history_memory`. Evaluation
  runs each graph under both `incremental_state` and `step_history_only`,
  yielding 60 instance results.
- `multimodal.jsonl`: 30 tasks paired to direct tasks through
  `source_task_id`. IDs use `multimodal-<difficulty>-<NN>`. Each record has
  one `image` path, relative to the JSONL file, pointing to
  `images/<task-id>.png`. The renderer version is `multimodal-v3`.
  Main evaluation uses `image` (30 results); paired ablation uses
  `text` and `image` (60 results). Graph difficulty remains easy/medium/hard;
  there are no separate image difficulty variants.
- `tasks.jsonl`: the earlier compatibility/smoke set.
- `tasks_generated.jsonl`: the existing generated set, retained independently.

History tasks identify parallel edges as `e01`, `e02`, and so on in original
edge-list order. Each event supplies the current vertex and its original static
incident edges, but never the authoritative used-edge set. In
`incremental_state`, the agent writes that state after each event; in
`step_history_only`, it may acknowledge only the step number. Hard tasks include
a reversible wrong move followed by an undo event.

## Provenance and reproducibility

The small easy-graph motifs are inspired by the BSD-3-Clause NetworkX Graph
Atlas. Larger tasks are handcrafted deterministic combinations of cycles,
bridges, chords, and parallel edges, informed by locally archived one-stroke
datasets. No raw unlicensed level is copied into this repository.

Regenerate the formal files and images with:

```powershell
python scripts/build_one_stroke_direct_history.py
python scripts/build_one_stroke_multimodal.py --overwrite
```

The loader independently checks graph solvability, complete oracle paths,
history legality, LIFO undo semantics, and the existence of a valid completion
after the recorded history.

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
direct=30, history=652, and multimodal main=30/paired=60; reasoning wrappers may
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
