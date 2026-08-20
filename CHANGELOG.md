# Changelog

## [0.2.0] - 2026-08-20

### Features

- Consolidation generation is retried once. `/api/finglmqa/consolidate` now
  routes its chat completion through `_generated_text`, which makes a second
  attempt after a one-second pause when the model errors or returns an empty
  completion. A second failure falls back to concatenating the per-report
  answers, exactly as the route already did on the first failure.
- The TabGR quota is a flat two rows. `_table_quota` no longer inspects the
  question; `TABLE_QUOTA = 2` applies to every request, so a narrative question
  now carries two table rows instead of one.
- Repaired the three tracked symlinks under `FinGLMQA/refs/`. They pointed at
  `hyfin/../../A2RAG`, which resolves to `FinGLMQA/A2RAG` and does not exist, so
  `start.sh` could not find the A2RAG runtime from a fresh clone.

### Design Rationale

- Retry was scoped to consolidation only. Per-report answer generation already
  degrades to the deterministic extractive builder, and retrying the whole QA
  request from the browser would repeat retrieval and burn a second slot against
  the gateway's hourly expensive-route limit. Consolidation is the one
  generation step whose only fallback is a plain concatenation.
- The old quota fell to one row unless the question contained a term from a
  literal, untranslated list of Chinese numeric words. That gave an English
  question the lower quota purely because it had not been translated yet. A flat
  quota removes the asymmetry and the list along with it.
- The quota was not raised past two. `LIMITS["max_evidence_top_k"]` is 5 and
  `_table_quota` returns `min(TABLE_QUOTA, top_k - 1)`, so a larger quota would
  eat the text channel. Raising the frozen limit would mean editing
  `src/config/composition_patterns.json`, which is SHA-256 pinned by
  `runs/phase_10/immutable_inputs_manifest.json` and verified by the supervisor
  at startup.

### Notes & Caveats

- At the default `top_k` of 5 the fused evidence set is now 2 table rows and 3
  text chunks, down from 4 text chunks for questions without a numeric term.
  Answers to narrative questions draw on one less text chunk than before.
- `TABLE_QUOTA` cannot exceed 4 without also raising
  `LIMITS["max_evidence_top_k"]` in both `contracts.py` and
  `src/config/composition_patterns.json`, and regenerating the phase-10
  immutable manifest on a machine that holds the full data bundle.
- The retry consumes a second call against the online model on every failed
  consolidation. The gateway's `DEMO_RATE_LIMIT_EXPENSIVE` budget counts the
  inbound request, not the upstream calls, so the limit itself is unchanged.

## [0.1.0] - 2026-08-19

- Initial public release.
