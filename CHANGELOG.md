# Changelog

## Unreleased — C2 100 paired positions and separate group accounting

- Expand C2 to 100 standard conditions and 100 same-position focus variants.
  Keep 50 short/50 medium baseline positions; independently certify each
  variant's real distance rather than forcing the same band or mating line.
- Prioritize changes to retained optimal strategies, first moves and mate
  distance; preserve paired impact evidence and trace every derived position.
  Only complete independently re-proved trees may pass the publishing gate.
- Separate H2 full-state/history and C2 standard/variant outcomes, tokens, time
  and configured-price estimates. Preserve per-rule detail and physical API
  retry/error accounting without counting replay as another billed request.
- Export condition-metrics JSON/CSV beside comparison charts. Full nine-Agent
  runs contain 5,400 conditions; the 30-condition direct pilot is unchanged.

## Unreleased — H2/C2 whole-episode legality conditioning

- Add mate within the certified shortest bound conditional on every Agent
  decision returning a legal move. Keep legal non-mating games in the denominator
  and preserve the original mate and move-legality metrics.
- Persist exact decision counts and expose numerator, denominator, eligible
  coverage and unknown evidence in native, browser and cloud reports/plots.
  Infrastructure errors are excluded; empty denominators remain N/A.

## Unreleased — paired D3/M2 perception-gated metrics

- Preserve raw mate, legality, transcription and model-output evidence. Apply
  the same exact full-board transcription and true-board legality gate to both
  the D3 text condition and M2 image condition.
- Add checkmate conditional on that gate and joint success over all scorable
  conditions, with explicit numerators, denominators and eligible counts.
  Empty denominators and unavailable legacy evidence remain unknown, not zero.
- Expose the additional metrics in native/API and browser summaries and cloud
  comparison tables/plots without replacing the original scores or prompts.

## 2026-10-24 — independent Xiangqi strategy verification

- Add an independent bounded solver using the production rules backend rather
  than the tree exporter's reference backend. Recompute shortest distance and
  every declared optimal attack/defence edge; a positive-line replay alone is
  not an exhaustive proof.
- Persist per-task verified/invalid/unknown results and content/code bindings.
  Budget exhaustion stays unknown; stale or missing evidence blocks selected
  certified tasks before model calls in standalone and cloud runners.
- Add deletion, forged-distance, stale-evidence and resumable-CLI regressions.

## Unreleased — strict certified shortest-mate horizon

- H2/C2 allow legal off-tree moves to continue against their existing fallback
  opponent, but the total attacker budget is the certified shortest `n` moves,
  counted from the initial position. Check for checkmate after every move; no
  counter reset or `n+2` extension is allowed.
- Report and plot only mate within that shortest bound, legality and the existing
  task-specific loss. Version the new protocol separately from archived runs.

## 2026-10-20 — integrated release branch

- Merge the latest Mahjong 100-item groups and the current main branch's
  one-stroke/Zebra releases and modular command/experiment layout.
- Preserve the certified Xiangqi 100-item corpora and complete shortest-mate
  trees, paired C2 conditions, shared D3/M2 positions, and Chinese-image M2 input.
- Wire the Xiangqi decision deadline, resource accounting, tree configuration
  and uncapped request setting through the new orchestration modules. Add
  regression tests for these module boundaries; do not retain the removed
  `factory.experiments` compatibility module.
- Experimental outputs, credentials and local environments are not submitted.

## Unreleased — project organization

- Split CLI handlers into `commands/` and experiment orchestration into
  `experiments/`. Removed `factory.experiments`; use the new modules directly.
- Removed the deprecated `openai-compatible` agent name and
  `OpenAICompatibleAgent`. Use `passthrough`, or compose `OpenAICompatibleClient`
  with an existing Agent. Transport and reasoning algorithms are unchanged.
- Removed duplicate `one_stroke.yaml`; use `one_stroke_direct.yaml`.
  Task data, experimental settings, prompt versions and prompt text are unchanged.
- Run fingerprints include the extracted orchestration modules. Existing
  results remain readable, but resume requires the same source fingerprint.
  Zebra uses only its current sidecar lock; obsolete `.run.lock` entries are
  treated as unexpected files, never silently reused or removed.
- Removed hand-written old folder/agent aliases from result plots. Visualization
  helpers scope their font settings locally, close their figures and accept
  filename-only output paths.

## 0.2.0 — 2026-08-19

Breaking Xiangqi schema and interface release.

- Replaced D3, C2, H2, and M2 public task names with
  `xiangqi-mate-in-one`, `xiangqi-rule-variants`, `xiangqi-history`, and
  `xiangqi-multimodal`. The old names are not runtime aliases.
- Converted all 1,000 formal records to schema version 2 with FEN-only board
  persistence, normalized oracle fields, readable enums, and clean tags.
- Added a complete v1→v2 mapping and a safe JSON/JSONL/run-directory migration
  command that preserves raw model text.
- Replaced ambiguous `variant_a/b/c`, `img_cn`, `img_ab`, `full`, and
  `agent_only` values with descriptive enums.
- Added four executable YAML configs, unified task/suite commands, deterministic
  in-memory sampling, standardized run artifacts, terminal/JSON/PNG inspection,
  and a self-contained 1,000-task HTML gallery.
- Bundled OFL-licensed Noto Sans CJK SC Regular/Bold glyph subsets and routed
  Xiangqi rendering through package-owned font paths.
- Reworked image regression checks to compare decoded pixels and emit expected,
  actual, diff, dependency, and font diagnostics. The two earlier CI failures
  were PNG byte-encoding differences, not GitHub Actions service failures.
- Pinned Python 3.10.20 and the CI dependency stack; constrained NumPy below 2
  for the legacy Gym dependency; moved official JavaScript actions to Node 24
  releases pinned by immutable commit SHA.
