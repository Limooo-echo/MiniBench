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
