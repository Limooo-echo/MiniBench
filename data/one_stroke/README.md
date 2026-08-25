# MiniBench 2.0 one-stroke tasks

The formal A1/A2/A3 sets use the graph-theoretic definition of one-stroke drawing:
each undirected edge must be used exactly once; vertices may be revisited.

## Files

- `a1_direct.jsonl`: 30 direct-reasoning tasks, with 10 easy, 10 medium, and
  10 hard tasks. Every difficulty contains seven solvable and three unsolvable
  graphs. The formal prompt is `baseline` and contains no Euler-theorem hint.
- `a2_rule_condition.jsonl`: 30 temporary-rule tasks, with 10 per difficulty.
  Every difficulty contains eight rule-solvable and two rule-unsolvable tasks;
  all underlying standard graphs remain solvable. A2 answers return both `path`
  and `edge_path` so edge-specific constraints and parallel edges are unambiguous.
- `a3_history.jsonl`: 30 history-memory tasks, again 10 per difficulty. A full
  evaluation runs every graph under both `incremental_state` and
  `step_history_only`, yielding 60 instance results.
- `a4_multimodal.jsonl`: 30 multimodal tasks paired to A1 source tasks. Each task
  has clear and challenge image variants and can also run in structured-text mode.
- `tasks.jsonl`: the earlier compatibility/smoke set.

History tasks identify parallel edges as `e01`, `e02`, and so on in original
edge-list order. Each event supplies the current vertex and its original static
incident edges, but never the authoritative used-edge set. In
`incremental_state`, the agent writes that state after each event; in
`step_history_only`, it may acknowledge only the step number. Hard tasks include
a reversible wrong move followed by an undo event.

A2 supports start/end, first/last edge, directed-edge, edge-order, exact
checkpoint, adjacent/nonconsecutive-edge, and edge-step-window constraints. The
four rule modes are `full`, `standard`, `drop_key_rule`, and
`conflicting_rule`. The conflicting mode removes the key rule and inserts a
verified logical reverse; the loader requires that the replacement world is
solvable and that the key and reverse rules cannot be jointly satisfied.

## Provenance and reproducibility

The small easy-graph motifs are inspired by the BSD-3-Clause NetworkX Graph
Atlas. Larger tasks are newly handcrafted deterministic combinations of cycles,
bridges, chords, and parallel edges, informed by the structures observed in the
locally archived one-stroke datasets. No raw unlicensed level is copied into
this repository. Regenerate the formal files with:

```powershell
python scripts/build_one_stroke_a1_a3.py
python scripts/build_one_stroke_a2.py
```

The loader independently checks graph solvability, complete oracle paths,
history legality, LIFO undo semantics, and the existence of a valid completion
after the recorded history. For A2 it also recomputes all constrained-path
oracles and validates every reverse-rule ablation.

## Scoring and interpretation contract

- A1 accepts any valid Euler trail, not only the stored oracle. `a1_score` is
  direct path/no-solution accuracy. Here `direct` names the benchmark track,
  not the `DirectAgent` architecture; the shipped A1 config uses the raw
  `openai-compatible` agent deterministically.
- A2's `a2_score` is the `full`-rule accuracy. `rule_ignore_rate` is conditional
  on outputs that are valid for the standard graph and have active rules. In the
  current easy split, `standard` and `drop_key_rule` can be identical when the
  key rule is the only temporary rule; treat that pair as a negative-control,
  not evidence of rule sensitivity.
- A3 keeps `success` and `a3_final_score` as backward-compatible final-completion
  accuracy. The official `a3_score` is `a3_joint_score`: the final completion,
  the final response schema, every intermediate response schema, and (for
  `incremental_state`) every recorded state must all be correct.
  `a3_intermediate_protocol_score` and `a3_protocol_score` expose the
  intermediate-only and full-protocol views. The current v1 data deliberately contains
  no history that ends without a valid completion, and measures within-transcript
  context tracking rather than persistent memory across runs.
- A4 reports `a4_path_score`, `a4_transcription_score`, and `a4_joint_score`.
  `a4_score` remains an alias for path score. The main A4 config runs only the
  challenge image; use the ablation config to estimate text/clear/challenge gaps.
  In v1, the 10 easy clear/challenge image pairs are byte-identical, so they are
  negative-controls rather than degradation contrasts.
- `json_format_exact_rate` records whether the complete response is one strict
  JSON object; `response_schema_valid_rate` separately requires the exact fields
  for A1/A2/A3/A4. Legacy answer extraction remains lenient for backward-compatible
  task accuracy.

Every configured run creates its manifest and empty checkpoint before the first
model call, then atomically checkpoints each completed task/mode work item. The
manifest records the resolved config, provider/model identity, data and code
fingerprints (including A4 image bytes), Git state, and the complete work plan.
The shipped protocol-generation totals are A1=30, A2 main=30/A2 ablation=120,
A3=652, and A4 main=30/A4 ablation=90; reasoning wrappers may add internal calls
on final turns. Compare scores across A1-A4
only when the paired design, model identity, and fingerprints support that
comparison; the shipped A1-A3 and A4 configs use different default model families.
