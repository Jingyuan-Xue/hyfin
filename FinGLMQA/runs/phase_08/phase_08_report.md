# Phase 8 Completion Report

## Outcome

Phase 8 is complete. The implemented QA path is a deterministic static DAG:

`QuestionAnalysis → ScopePlan → CompositionPlan/SubPlan[] → structured execution → NumericAuthorizationSet → evidence execution → Composer`

All Gates 0–8 pass. Phase 6 and Phase 7 inputs remained immutable.

## Delivered behavior

- Four public backends: `fact`, `sql`, `formula`, and `evidence`.
- Exactly nine topology patterns with deterministic matching, ordering, IDs, required-node policy, minimum usable results, comparison quorum, and no dynamic expansion.
- Independent single-document evidence nodes for company/year sections; no joint retrieval, cross-document reranking, evidence transfer, fused summary, or cross-company causal inference.
- Static zero-call rejection for dynamic winner explanation, two-dimensional company/year fan-out, unsupported composition, and hard-limit violations.
- Read-only selected-fact and metadata repositories, registered QuerySpec SQL, Decimal formula execution, and strict Phase 9 `missing_fact_requests` for direct fact, formula operand, and unique document SQL misses.
- NumericAuthorizationSet and citation-scope enforcement before evidence claims become usable.
- Deterministic QATrace and separate runtime QATelemetry.

## W4 review fixes

The three read-only reviewers found real issues that were fixed before release:

- raw retrieved chunks could carry unauthorized numbers into QAAnswer even when the accepted claim was safe;
- same-valued wrong metrics could borrow an authorization;
- an allow-listed worker chunk could forge content/line/section metadata;
- metadata records and malformed formula provenance could be washed through structured adapters;
- request hints and report/metric-year phrasing could misroute scope;
- historical/renamed company aliases left nine executable benchmark rows blocked;
- mixed structured+narrative multi-report questions returned unsupported instead of clarification;
- Gate 2 compared only backend/operation and overstated planning correctness.

The release now removes raw content from official artifacts, derives one unique metric/formula and year for every financial claim, exact-matches worker chunks to hash-pinned Phase 7 evidence, validates metadata/formula lineage, separates hints from text axes, supports reviewed historical aliases, emits blocked placeholders for ambiguous reports, and compares complete planning projections.

## Verification

- Phase 8 tests: 169 passed.
- Gate 1: schemas, contracts, limits, and nine-pattern registry passed.
- Gate 2: 1,003/1,003 benchmark and 40/40 General full projections passed; deviations 0.
- Gates 3–5: 9 exact fact pins, registered SQL/report-year separation, and 6 exact formula pins passed.
- Gates 6–8: core pipeline, Composer/quorum, global provenance suppression, authorization, portability, and real evidence repeatability passed.
- Real evidence: two fresh fixed-CPU BGE-M3 runs returned byte-identical answer and deterministic trace; telemetry differed as expected.
- Real trace SHA-256: `6ce39bdced732c413dabdf812a4076380359ab7c2f3237a2cae27a9f842d1539`.
- Evidence provider fingerprint: `dec6b55fa69c489d0d2e86abf605ae1916016780cf9d3dc25e14700cbb799648`.
- Registry semantic SHA-256: `330f4e3a6c832950310fae4d67e0219dccd51357d39c5f2c0812bf5442432eda`.
- Phase 6 DuckDB pin: `b3e8fed65ddc1ccd5954083a4df64f3eab2150294cae08a11424f3bc5744f278`.
- Phase 6 JSONL pin: `abeb4b3b221aac74705b84c80469c03b23fd8638d67004c75dd7a512c6841405`.

The integrated Gate 8 report additionally confirms that official answer/trace trees contain no raw chunk `content`, host absolute paths, runtime telemetry fields, or known unauthorized financial renderings.

## Commands

```bash
.venv/bin/python scripts/validate_phase_08_gate1.py
.venv/bin/python scripts/validate_phase_08_gate2.py
.venv/bin/python scripts/validate_phase_08_gates_3_5.py
.venv/bin/python scripts/validate_phase_08_gates_6_8.py
```

## Deferred boundaries

- Phase 9 may consume only strict `missing_fact_requests`, validate TabGR-derived facts through the selected-fact-equivalent provenance gate, and write a separate supplemental store.
- Phase 10 owns service lifecycle, GPU workers, concurrency, timeout, logging, and projections without changing QA semantics.
- A later multi-document phase may introduce result-driven evidence targets, explicit document sets, joint retrieval/reranking, and fused synthesis under new versioned contracts.
- Multi-turn context resolution remains out of scope.
