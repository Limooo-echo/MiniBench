# Xiangqi reasoning release r1

Release: `xiangqi-2026-09-19-r1`. Runtime protocol: `xiangqi-reasoning-v2`.
The four families and schema version 2 are retained. This benchmark compares
registered Agent architectures **within the same base-model profile**.

| Task | Input / action | Success | Main score |
|---|---|---|---|
| D3 | Complete text board; one freely generated UCI move | Legal strict mate in one; every mating move accepted | D; easy/medium/hard equally weighted |
| C2 | Same board with standard or one of three rule cards; one free move | Legal maximum utility at exactly three plies | R; three nonstandard rule groups equally weighted |
| H2 | Paired full current board / initial board plus complete move history | Strict checkmate within the task horizon against fixed Pikafish | H; history-only, reference-distance bands equally weighted |
| M2 | Paired text / Chinese-piece image / Latin-piece image | Same exact mate-in-one criterion as D3 | V; the two image modes weighted one half each |

Every Xiangqi item remains `s=Y, P=0`. Xiangqi task weights within D/R/H/V are
0.20/0.35/0.20/0.35. Standard-rule, full-state and text controls explain differences;
they do not enter the corresponding R/H/V main scores. Existing process scores,
legality, CP loss and utility regret are diagnostics. The radar represents an Agent
architecture, grouped by base model. Tokens, model calls and elapsed time accompany
ability scores. Unknown token usage for a known call stays in the coverage denominator.

## Coordinates and rules

Files a–i run from Red's left to right; rank 0 is Red's home side. FEN lists ranks
9 down to 0, uppercase Red and lowercase Black. `w` means Red to move. An answer
is a JSON object such as `{"move":"a0a1"}`; no candidate list, board transcription
or explanation is requested.

The variants change only these rules:

- Horse: ignore the blocking horse leg.
- Chariot: a destination in files d/e/f is forbidden (crossing these files is allowed).
- Soldier: a backward step is allowed only while **currently** across the river
  (Red ranks 5–9, Black ranks 0–4). Once back on its own half, backward and sideways
  movement are forbidden again.

Own palace, advisor/elephant locations, piece counts, non-facing generals and
side-to-move consistency are checked before accepting positions. The player who
just moved cannot already be checked under the active rule. A checked player to
move may legally be required to evade check. Structural constraints alone do not
prove reachability from the standard opening.

Checkmate and stalemate are distinct terminal types. Both lose for the side to
move in Xiangqi search; only actual checkmate completes D3/H2/M2. Capturing a
missing general is not a substitute for strict checkmate.

## Data and independent verification

Each family contains 250 records. D3 keeps the original 250 valid positions and
re-enumerates every mating move. Difficulty remains 80 easy / 85 medium / 85 hard;
its structural index uses choice pressure .50, checking near misses .30 and state
load .20. M2 is derived one-to-one from these records, with identical boards,
answers, source IDs and difficulty. Legacy D3 lacked upstream per-position
provenance; that limitation is retained rather than invented.

C2 is rebuilt from legal PGN replay in CCPD revision
`368a47a947773dd8692c026e286dd19b6277b993`. It contains 60 paired scenarios (20 per
focus rule), each under all four conditions, plus 10 standard controls. Cross-scenario
boards are unique. The generation seed is 20260831; search depth is 3. Every paired
condition must have a unique best move separated by at least .5, and the focus rule
must change the standard best move. Failure to meet the quota produces a shortage
report and no releasable dataset. The r1 candidate pool is the first 2,000 sorted
PGN files, up to 1,000 tail plies, Red to move, at most 18 pieces. The complete pool
hash and scan counts are in `data/xiangqi/rule_variants/generation_report.json`.

C2 utility preserves the existing design: Red material minus Black material;
general 10000, advisor 2, elephant 2, horse 4, chariot 9, cannon 4.5, soldier 1.
Checking Black adds 1 and checking Red subtracts 1. Terminal win/loss is ±10000,
including stalemate at the search horizon. Unused legacy local-goal rules retain
an agent-signed 5000 bonus. Root values always use Red's perspective; Black
minimizes them. All equally best moves within absolute tolerance `1e-6` pass.
This is fixed-depth tactical rule application, not unlimited-depth global optimality.

H2 keeps 80 short (reference mate in 2–3 moves), 80 medium (4–5), and 90 long (6–12),
with unique source games. Every accepted position stores complete depth-16 and
depth-20 reference routes, requested and actual search depth, score and independent
replay evidence. Both routes must actually end in strict checkmate, with no illegal
move, truncation, early termination, repetition, engine error or stalemate.
`max_plies = 2 * reference_mate_moves + 1`. These are **engine reference distances**;
two routes do not prove a shortest forced mate or cover all legal defences. Runtime
results measure performance against the registered fixed opponent only.

Standard legal move sets are checked against pinned `cchess==1.25.5`, after
hand-specified interface calibration (self-check/facing generals, cannon screen,
checkmate/stalemate). `reference.py` implements independent movement, attack and
search logic without calling `VariantBoard` movement/check methods. C2 compares
all root utilities against its exhaustive depth-three search. D3/M2 independently
enumerate complete answer sets. Any rule disagreement blocks release.

## Runtime and evidence

Pikafish uses `Threads=1`, `Hash=128` MB and clears hash for every independent
analysis. Formal and smoke H2 use depth 16; D3/M2 diagnostics use depth 8. H2 dataset
validation uses 16 and 20. Timeouts are failure limits, not a time-based search budget.
The r1 local validation uses the official Pikafish 2026-09-06 Windows universal
binary through WSL and its accompanying NNUE. Exact binary/NNUE hashes, source
revision and settings are recorded in provenance and run metadata. Supply the same
artifacts via `PIKAFISH_PATH` / `PIKAFISH_EVAL_FILE` to reproduce the opponent.

Text and image conditions use the same architecture generation parameters and
answer opportunities. Evaluators add no model retry; transport retries remain in
the provider, while the architecture's registered sampling/reasoning/format repair
is unchanged and charged to its recorded cost. Every task/mode initializes fresh
board and conversation state. H2 history-only requires `generate_messages` with
the full conversation; unsupported agents receive an explicit configuration error.

| Status | Meaning | Main score |
|---|---|---|
| `ok` | Judged legal answer; goal may succeed or fail | 1 or 0 from goal |
| `invalid` | Returned but unparseable or illegal answer, including failed format repair | 0 |
| `error` | API/transport, configuration, judge or necessary opponent failure | null / missing |

An optional diagnostic-engine error leaves CP unavailable and does not erase a
precisely judged outcome. Full answers, all actions, structured error stage/reason,
termination and model usage are retained. Selected tasks, configuration, versions
and fingerprints are saved before calls; each result is fsynced. Interrupted runs
retain prior results and same-name runs cannot overwrite them. Old results without
error evidence cannot have a missing/incorrect distinction reconstructed retroactively.

History/image gaps measure a change in task success under changed information.
The output does not separately measure memory accuracy or image recognition accuracy.

## Frozen experiments and reproduction

`data/xiangqi/evaluation_samples/manifest.json` freezes the actual records and hashes
for every architecture. Smoke: D3/H2/M2 six positions each, C2 three scenarios;
48 observations total. Formal: D3/H2/M2 30 each, C2 ten scenarios; 220 observations.
D3/M2 share the exact selected source positions. H2 evaluates both information
modes; M2 all three representations. Architecture default configurations must be
registered with their actual model profile and reused consistently.

From the repository root, with the Xiangqi generation dependencies installed:

```bash
python scripts/annotate_xiangqi_d3.py --in-place
python scripts/build_xiangqi_m2_from_d3.py --d3 data/xiangqi/mate_in_one/tasks.jsonl --output data/xiangqi/multimodal/tasks.jsonl
python scripts/build_xiangqi_c2.py --candidates data/xiangqi/rule_variants/candidates.json --output data/xiangqi/rule_variants/tasks.jsonl
python scripts/freeze_xiangqi_evaluation.py
python scripts/write_xiangqi_repair_manifest.py
python scripts/validate_xiangqi_data.py --workers 4 --output data/xiangqi/independent_validation.json
python scripts/audit_xiangqi_release.py --technical --output data/xiangqi/release_audit.json
python -m unittest discover -s tests
python scripts/accept_xiangqi_protocol.py --output-dir output/acceptance/new-check --pikafish-path "$PIKAFISH_PATH"
python scripts/run_xiangqi_smoke.py --prepare-only
```

The acceptance script uses deterministic/scripted agents and a real local engine;
it makes no language-model request. `--prepare-only` likewise makes no paid call.
To run a real model later, use the prepared configs and a new run directory. No
real model ranking is required to accept this repair.

To rebuild C2 candidates from the fixed downloaded CCPD source:

```bash
python scripts/extract_ccpd_positions.py --source "$CCPD_ROOT" --output tmp/c2-positions.jsonl --tail-plies 1000 --max-games 2000
python scripts/generate_xiangqi_c2_candidates.py --positions tmp/c2-positions.jsonl --source "$CCPD_ROOT" --output tmp/c2-candidates.json --report tmp/c2-generation.json --seed 20260831 --workers 6 --max-pieces 18
```

The 48 source PGNs used by C2 are archived byte-for-byte under
`data/xiangqi/sources/ccpd/`, with their fixed revision and SHA-256 manifest. The
independent gate replays their source prefixes and checks all transitions.

Rebuild H2 after D3 and C2, from the same fixed CCPD revision (wait for each
extraction to finish). A matching checkpoint can reuse completed engine certificates:

```bash
python scripts/extract_ccpd_positions.py --source "$CCPD_ROOT" --output tmp/ccpd-h2-focused.jsonl --include-subdir Dataset/殺局_殺法_練習題 --include-subdir Dataset/殘局 --tail-plies 32
python scripts/extract_ccpd_positions.py --source "$CCPD_ROOT" --output tmp/ccpd-h2-fullgame.jsonl --include-subdir Dataset/全盤戰術 --include-subdir Dataset/中局 --tail-plies 48
python scripts/build_xiangqi_h2.py --existing data/xiangqi/legacy/xiangqi-corpus-rebuild-2026-08-31/history/tasks.jsonl --pool tmp/ccpd-h2-focused.jsonl --pool tmp/ccpd-h2-fullgame.jsonl --source-root "$CCPD_ROOT" --output data/xiangqi/history/tasks.jsonl --report data/xiangqi/history/validation_report.json --checkpoint tmp/h2-validation.checkpoint.jsonl --workers 8 --seed 20260919 --depth-a 16 --depth-b 20 --timeout 120 --pikafish "$PIKAFISH_PATH"
```

H2 excludes current D3/C2 FENs by default, records the excluded dataset hashes,
and verifies those inputs did not change during selection. Different PGN filenames
do not imply different positions. The final gate also rejects unintended cross-family
FEN overlap; only the planned D3/M2 pairing is allowed.

The generator checkpoints atomically. Resume requires matching input/config/code
hashes; code changes require explicit `--revalidate-resume`. H2's exact generation
configuration, pools, rejected old IDs and both engine routes are in its validation
report; `python scripts/build_xiangqi_h2.py --help` documents rebuilding with a new
checkpoint. The original 2026-08-31 files are preserved under `data/xiangqi/legacy/`.
The old ID migration command verifies/uses that archive only. Changed boards or
goals (including H2 half-move limits) receive new IDs; no old answer is attached to
a replacement board. `data/xiangqi/repair_manifest.json` records retained, retired
and added IDs with per-task retirement reasons and immutable old-file hashes.

`config/scoring/manifest.example.yaml` uses current data and frozen selections.
`manifest.legacy-v1.example.yaml` is for archived results only. Empty `runs` means
missing evaluations and does not create a ranking. The strict external provenance
audit additionally reports the repository's still-undeclared top-level licence;
this repair does not choose a licence. Public CCPD may overlap training data, and
local validation cannot establish contamination absence.
