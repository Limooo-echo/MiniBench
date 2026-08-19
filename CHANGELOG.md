# Changelog

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
