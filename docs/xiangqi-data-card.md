# Xiangqi schema-v2 data card

## Scope and release contents

MiniBench 0.2.0 contains four Xiangqi families with 250 records each, for a
total of 1,000 JSONL records:

| Code | Family | Primary experimental condition | Records |
| --- | --- | --- | ---: |
| D3 | `xiangqi-mate-in-one` | Direct mate-in-one | 250 |
| C2 | `xiangqi-rule-variants` | Best move under a temporary rule | 250 |
| H2 | `xiangqi-history` | Full board state versus move history only | 250 |
| M2 | `xiangqi-multimodal` | Text versus two board-image encodings | 250 |

FEN is the persisted board representation. Every record uses
`schema_version: 2`, a family-prefixed ID, an `agent_color` matching the side
to move, both generals, a positive per-task `max_plies`, an exact
`piece_count`, a normalized `oracle` object, and sorted unique tags.

The release manifest is `data/xiangqi/provenance.json`. Exact-position reuse
and unresolved provenance are reported by `data/xiangqi/release_audit.json`.
The repository does not currently declare a top-level licence for its own code
and generated data; the audit deliberately reports that fact instead of
inferring a licence.

## Coordinates and output

Files are `a` through `i` from Red's left to right. Ranks are `0` through `9`
from Red's home side toward Black. A UCI move concatenates origin and
destination, for example `a0a1`. FEN lists ranks 9 down to 0; uppercase pieces
are Red, lowercase pieces are Black, `w` means Red to move, and `b` means
Black to move.

Agents generate one free UCI move at a time. The benchmark does not expose a
numbered legal-move list. Prompts describe the coordinate system, current
information condition, output schema, and normal legality checks; they do not
reveal the oracle move.

`oracle.best_move_uci` is a preferred engine or exact-rule move. A different
move is not automatically wrong when it also achieves the task goal.
`oracle.mate_in_plies` is the shortest verified mating horizon when available.

## D3: direct mate-in-one

- Corpus: 250 exact mate-in-one positions.
- Horizon: every task has `max_plies: 1` and `oracle.mate_in_plies: 1`.
- Verification: `d3_analysis.mate_moves_uci` stores every legal mating move;
  the preferred move must be a member of this set.
- Difficulty: 80 easy, 85 medium, and 85 hard. The deterministic structural
  score combines legal-choice pressure, non-mating checking near misses, and
  board state load, followed by fixed quantile assignment.
- Sampling: `stratified`; the formal YAML samples 30 tasks, 10 per level.
- Runtime evaluation: exact legality and checkmate decide task success.
  Pikafish supplies preferred-move and centipawn-loss diagnostics.

The positions originate from the earlier MiniBench D3 corpus and were
re-annotated with the deterministic `d3-structural-v2` procedure. Because the
legacy corpus did not retain upstream per-position source fields, no more
specific provenance is claimed.

## H2: history and multi-step mate

- Corpus: 250 unique positions replayed from 250 distinct games in the
  [Chinese Chess Practical Dataset](https://github.com/Yvonne761/Chinese-Chess-Practical-Dataset),
  revision `368a47a947773dd8692c026e286dd19b6277b993`, licensed CC-BY-4.0.
- Distribution: 80 short mate-in-2/3, 80 medium mate-in-4/5, and 90 long
  mate-in-6 or more. Exact mate-in-moves counts are 49/31 for 2/3, 50/30 for
  4/5, and 40/20/19/5/3/2/1 for 6 through 12.
- Verification: each released position was independently analysed at
  Pikafish depths 16 and 20. Both depths had to return the same positive mate
  distance, a legal principal variation, and the same difficulty band.
- Horizon: the task-specific limit is `2 * mate_in_moves + 1` plies, which
  leaves one additional agent turn beyond the shortest engine line. Limits
  range from 5 to 25 plies and must not be replaced with one global cutoff.
- Opponent: Pikafish; the formal configuration uses depth 16.
- Conditions: `full-state` provides the current board every agent turn.
  `move-history-only` provides the initial board once and then continues the
  same multi-turn conversation using only both players' UCI moves.
- Sampling: one stratified draw of 30 positions, 10 from each horizon band,
  is reused in both information conditions (`history_mode: paired`).

Exact per-position validation evidence is stored in
`data/xiangqi/history/validation_report.json`.

## C2: temporary rule variants

- Corpus: 60 complete paired scenarios. Each scenario repeats one FEN under
  four rule cards: standard, horse without leg blocking, chariot forbidden
  from centre files, and soldier allowed to retreat after crossing the river.
  These produce 240 counterfactual records. Ten additional standard-only
  controls bring the file to 250 records.
- Goal: generate the best free UCI move under the displayed rule, not merely a
  legal move and not necessarily a checkmate.
- Oracle: deterministic rule engine with depth-3 search. Released paired
  scenarios require a unique best move margin and a focus rule that changes
  the standard best move.
- Evaluation: legality uses the active rule card. Task success is exact goal
  completion; best-value regret and value quality are diagnostic measures.
- Sampling: `paired-stratified`. A count of 10 means ten scenarios, stratified
  across the three rule foci; all four same-FEN rule records are retained, so
  the run evaluates 40 records. Standard-only controls are not mixed into the
  formal paired comparison.

## M2: paired multimodal mate-in-one

- Corpus: the same 250 positions and exact mate sets as D3. This overlap is
  intentional and isolates input modality instead of position difficulty.
- Modes: `text`, `chinese-piece-image`, and `latin-piece-image`. Both image
  modes use bundled deterministic Noto Sans CJK font subsets.
- Horizon: one agent move only (`max_plies: 1`); M2 is no longer a multi-step
  game against an opponent.
- Sampling: 30 source positions, 10 per D3 difficulty level. Every selected
  position is evaluated in all three modes, giving 90 paired observations.
- Evaluation: exact rules accept every legal mating move. Pikafish separately
  reconfirms that the initial position is mate-in-one and supplies its
  preferred move and CP diagnostics. No shallow pseudo-CP oracle is used for
  correctness.
- Primary comparison: within-position change in mate success from text to each
  image mode. Legality, preferred-move match, and CP loss are diagnostics.

## Scoring interpretation

Task-native result files retain success, legality, preferred/optimal move
agreement, CP or value loss, and diagnostic quality fields. These help explain
failure modes, but they are not all added together.

The official MiniBench four-dimensional scorer is `minibench.scoring`. Under
its v1 policy, every Xiangqi item uses the original task success indicator
`Y`; Xiangqi partial credit is zero. Legality, oracle agreement, CP loss, and
value regret remain auxiliary diagnostics. This prevents an engine-quality
proxy from overriding whether the requested goal was actually completed.

## Reproduction and audit

Dataset entry points are:

- `scripts/annotate_xiangqi_d3.py`
- `scripts/extract_ccpd_positions.py`
- `scripts/scan_xiangqi_fen_candidates.py`
- `scripts/select_ccpd_predecessors.py`
- `scripts/build_xiangqi_h2.py`
- `scripts/generate_xiangqi_c2_candidates.py`
- `scripts/build_xiangqi_c2.py`
- `scripts/build_xiangqi_m2_from_d3.py`
- `scripts/audit_xiangqi_release.py`

Before a release, run:

```bash
python scripts/audit_xiangqi_release.py --strict
```

The strict audit intentionally remains unsuccessful until the project declares
its own top-level licence. Structural validity, expected D3/M2 overlap, C2
pairing, same-family duplicates, and external-source fields are still reported
separately.

For a small end-to-end API acceptance run with exact samples saved before
evaluation:

```bash
export DASHSCOPE_API_KEY
python scripts/run_xiangqi_smoke.py
```

The default smoke run uses Direct Qwen with thinking disabled, samples D3/H2/M2
at 6 positions each and C2 at 3 paired scenarios, and writes the exact selected
JSONL files. It is an engineering acceptance report, not a second scoring
system. To reuse those positions with a web model while local code performs
legality, opponent, and scoring steps:

```bash
python scripts/xiangqi_web_test.py --suite-dir runs/xiangqi-smoke-<timestamp>
```

Every normal Xiangqi run writes `selected_tasks.jsonl`,
`resolved_config.yaml`, and `run_metadata.json`, including data and config
hashes, renderer/schema versions, dependency information, model request
identity without secrets, Git state, and the Pikafish binary fingerprint when
available.

The migration map covers all 1,000 task IDs and all 70 C2 scenario IDs. The
migration command refuses in-place replacement, supports dry runs, reports
unknown IDs, and never rewrites model free text.

Known limitations are public-corpus contamination risk, absence of a declared
MiniBench top-level licence, and the fact that exact-FEN auditing cannot detect
all near-duplicate positions.
