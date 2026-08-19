# Phase 9: TabGR Supplemental Fact Completion

## Status

- Planning and implementation complete; Gates 0-7 passed.
- Revision 2 incorporates the unit-key, Phase 8 port, TabGR scoring, and
  Phase 6 eligibility review findings. All Gate arithmetic below is over the
  exact Phase 8 request key, including `normalized_unit`.
- Plan date: 2026-07-14.
- Phase 9 consumes immutable Phase 4/5/6 artifacts, the frozen Phase 8
  `missing_fact_requests` contract, and the local read-only TabGR runtime at
  `refs/tabgr_runtime`.
- Phase 10 service lifecycle (FastAPI, GPU workers, ports, concurrency,
  timeouts) remains out of scope. Later multi-document composition remains out
  of scope.

## Objective

Build a deterministic offline supplementation pipeline:

    MissingFactRequest (strict Phase 8 contract)
      -> conflict-group guard
      -> candidate table shortlist (FallbackCandidateIndex + audited
         document-scoped expansion)
      -> TabGR QG-PPR cell retrieval (deterministic, no LLM)
      -> metric / year / unit / value validation (Phase 6-equivalent rules)
      -> conflict check against every Phase 6 fact value
      -> provenance gate (Phase 4 cell-exact)
      -> supplemental fact store, separate from the immutable Phase 6 store

A TabGR result is never an answer. Every accepted supplemental fact must pass
validation equivalent to Phase 6 selected-fact eligibility plus an exact
provenance gate, and is always cited with an explicit `supplemental` source
marker. Every rejected request terminates with a stable failure code and a
complete audit row. No request may fail silently or fall back to weaker
validation.

## Feasibility Review (measured 2026-07-14, read-only)

The request grid is unit-specific because `normalized_unit` is mandatory in
`finglmqa.phase8.missing_fact_request.v1`: 170 documents x 3 in-window metric
years x (9 single-unit metrics + 2 explicit 股本 units) = 5,610 exact request
slots. Against the frozen Phase 6 store (`selected_financial_facts` = 4,189
rows and 4,189 exact metric/unit slots):

- 1,421 exact slots have no selected fact and are the maximum Phase 8
  missing-request universe.
  - 501 slots are withheld by unresolved multi-value conflict groups
    (1,365 `unresolved_conflict` rows retained in `financial_facts`).
  - 30 slots have only `low_confidence` non-conflict fact rows.
  - 890 slots have no fact row for the exact metric/unit key:
    - 79 expose only unit-null metric-year-resolvable Phase 5 candidates;
    - 44 mix unit-null and incompatible-unit candidates;
    - 259 expose only a different normalized unit (almost entirely the other
      股本 representation);
    - 508 have no metric-year-resolvable Phase 5 candidate and require table
      discovery beyond the Phase 5 alias net (43,540 `tabgr_ready` tables).
  - By metric, the 890 exact-unit no-fact slots are dominated by 股本 (717)
    and 总资产 (69); by year offset the counts are Y=243, Y-1=218, Y-2=429.
- `A601298_青岛港_2019年年度报告` accounts for 33 exact-unit missing slots,
  all of which have no exact-unit fact row; 8 unit-specific slots (股本 in 元
  and 股 for 2017/2018/2019, plus 基本每股收益 2018/2019) expose only
  unit-null Phase 5 candidates. It is the named acceptance fixture for
  explicit unit-evidence recovery.
- Consequence: for all 890 recoverable exact-unit slots, the Phase 8
  `candidate_table_ids` list (which filters on the requested `normalized_unit`)
  is empty. The plan therefore requires an audited, deterministic
  document-scoped shortlist expansion in addition to `FallbackCandidateIndex`.

The earlier 5,100 / 4,187 / 913 figures are retained only as a unitless
metric-slot diagnostic and are forbidden in Gate arithmetic or request counts.

TabGR runtime feasibility:

- The QG-PPR path (`build_graphs/graph_to_text_triple_full.py`:
  `grouped_string_with_cell_merges_w` -> `_build_neighbors`,
  `_build_personalization`, `_run_ppr`, renderers) is pure deterministic
  Python (dict/Counter/math, fixed 30-iteration PageRank, no randomness, no
  LLM call, no torch). The module imports `networkx` at module scope; the
  imported symbol is not used by any function on this path.
- The FinGLMQA `.venv` has no `networkx`, so the module currently fails to
  import. `serve_api.py` additionally requires `fastapi`/`pydantic` and is a
  service; running it belongs to Phase 10.
- TabGR's LLM reasoning stack (`qa/`, `request_llm/`, `prompts_engine/`,
  `environment.yml` GPU dependencies) is not needed and must not be used.
- Decision (see Planning Decisions): install exactly `networkx==3.5` as the
  single new `.venv` dependency, record its installed package-tree hash and
  interpreter version, and import the QG-PPR functions in-process through a
  FinGLMQA-owned adapter; do not start `serve_api.py`, do not add
  `fastapi`/`pydantic`, do not call any TabGR LLM module.

## Scope Decision

Phase 9 handles exactly one gap class: slots with no Phase 6 fact row for the
exact request key including `normalized_unit` (890 slots maximum; realized
scope is whichever of them arrive as strict `missing_fact_requests`).
Supplementation is executed as an offline batch over
a deterministic synthetic request universe shaped by the same
`finglmqa.phase8.missing_fact_request.v1` contract, so that online Phase 8
requests are answered by pure lookup into the supplemental store and no TabGR
code runs inside the QA path.

Gap classes explicitly not supplemented, each with a distinct terminal status:

- 501 conflict-withheld slots: the values already exist with full provenance;
  the blocker is value-selection ambiguity, not missing data. TabGR would
  re-extract the same conflicting sources. Resolution requires an audited
  curation policy (adjusted vs parent-only vs segment values) that is out of
  Phase 9 scope. Requests hitting these slots terminate with
  `SUPPLEMENT_CONFLICT_GROUP_OPEN` and cite the conflict group ID.
- 30 low-confidence-withheld slots: a fact row exists; Phase 6 confidence
  policy owns it. Terminal status `SUPPLEMENT_FACT_WITHHELD`.
- Any slot where a differing value already exists in `financial_facts` for the
  same exact group key `(document_id, stock_code, metric_year, statement,
  canonical_metric, normalized_unit)`: terminal `SUPPLEMENT_VALUE_CONFLICT`;
  Phase 9 never
  overrides or arbitrates Phase 6 values.

## Non-goals

- No modification of any Phase 1-8 frozen artifact, contract, or semantic.
  Phase 8 QA analysis, planning, execution, Composer, and status semantics are
  not redefined.
- No resolution or curation of the 501 unresolved conflict groups.
- No unit inference without explicit in-document unit evidence (reaffirms the
  Phase 6 Qingdao Port decision).
- No recovery of the 44,237 mixed-narrative table chunks; that is evidence
  coverage requiring fragment embeddings, not fact supplementation.
- No TabGR LLM reasoning, no `serve_api.py` process, no GPU use, no new
  service. Service lifecycle is Phase 10.
- No answering from TabGR output directly; no relaxation of Phase 8 numeric
  authorization or citation scope.
- No new metrics beyond the 10 canonical selected-fact metrics and no changes
  to `src/config/metric_aliases.json` or `src/config/unit_rules.json`.

## Immutable Inputs

Gate 0 records a manifest (`runs/phase_09/immutable_inputs_manifest.json`,
path/SHA-256/size/mtime, same conventions as Phase 8) covering at least:

- `data/facts/financial_facts.duckdb`
  `b3e8fed65ddc1ccd5954083a4df64f3eab2150294cae08a11424f3bc5744f278`
- `data/facts/financial_facts.jsonl`
  `abeb4b3b221aac74705b84c80469c03b23fd8638d67004c75dd7a512c6841405`
- `data/indexes/canonical_metric_candidates.jsonl`
  `8371c8a2e9f62d5dfd09d0fd6d14bfe14a1ca85d0273d9e0e2a02430952e5099`
- `data/corpus_package/tabgr_table_corpus.jsonl`
  `a6190b8c8e2f8bafe0f1ae7e0d5a7dcb7ca6de6c3acc4c3c41b7775d4336e369`
- `data/corpus_package/table_cells.jsonl`
  `41c14754ac8875498550b7986ff7a2ba5d61f3ea3839b11f4a7e249d6d1bf6a4`
- `data/indexes/tabgr_table_index.jsonl`
  `ee4a58887a535e1ce8427f6fb3a9a5f15e20bbe2cb927f3256f674aae4367291`
- `data/corpus_package/company_year_index.jsonl`
  `fb605221d096159435f24cdc8651e4679b039667b2dd7826290bd657ab6b7b00`
- `runs/phase_06/reports/candidate_decisions.jsonl`
  `588b321a612297ebe5fc5dfe4548d8c502a99653ec958839d1b46e9539282077`
- `runs/phase_06/reports/conflict_groups.jsonl`
  `8e1c6b56921bb98c952519351411c1541cef7df8d9e6c5401096d70520b814ad`
- `src/config/metric_aliases.json`
  `12f786608da8508741df11975070bf631eebcb919083a11a1ee0de843ba15ddc`
- `src/config/unit_rules.json`
  `ef948c839cc4ae040ddeb5a6b3c1f5ffdd8a167eb2f251dc4f1d80511ff52cfb`
- External TabGR sources consumed in-process (recorded with the `external:`
  prefix):
  - `external:tabgr_runtime/build_graphs/graph_to_text_triple_full.py`
    `7d193807d5f74b3281c8bd52c0d6da76f1f149cd5e92c4c82b47de4b8708d316`
  - any sibling TabGR module the adapter imports, hash-pinned at Gate 0.

All inputs open read-only. The supplemental builder must be verified not to
change hash, size, or mtime of any manifest entry (Phase 7 helper precedent).

## Architecture

### Request universe

`scripts/build_supplement_requests.py` deterministically enumerates the batch
universe from immutable inputs only:

- documents and report years from `company_year_index.jsonl`;
- metric years Y, Y-1, Y-2 per document;
- the 10 canonical metrics with catalog expected units (元, 元/股, ratio);
  股本 has no default unit and is enumerated under both 元 and 股, mirroring
  the Phase 8 rule that share-capital requests carry an explicit unit;
- slots already covered by `selected_financial_facts` are excluded;
- each remaining slot becomes a `finglmqa.phase8.missing_fact_request.v1`
  object with `origin_operation="fact_lookup"`, a reserved deterministic
  batch `subplan_id` (`sp_` + first 16 hex of the SHA-256 of the canonical
  slot key), a canonical `requirement_id`, and `candidate_table_ids` from the
  real read-only `FallbackCandidateIndex`.

Online requests from Phase 8 traces are matched to batch outcomes by the slot
key `(document_id, stock_code, report_year, metric_year, canonical_metric,
normalized_unit)`, never by `requirement_id` (which binds the caller's
`subplan_id`).

### Supplementation pipeline (per request)

1. Contract validation: `validate_missing_fact_request`, canonical
   `requirement_id`, document/company/year consistency against
   `company_year_index.jsonl`. Any violation is `SUPPLEMENT_REQUEST_INVALID`.
2. Conflict-group guard: query `financial_facts`/`conflict_groups` for the
   slot's group key. An unresolved conflict group terminates with
   `SUPPLEMENT_CONFLICT_GROUP_OPEN`; a withheld low-confidence fact terminates
   with `SUPPLEMENT_FACT_WITHHELD`; an already-selected fact terminates with
   `SUPPLEMENT_ALREADY_SELECTED` (defensive; excluded upstream).
3. Candidate table shortlist, in fixed priority order:
   a. `request.candidate_table_ids` (unit-compatible Phase 5 candidates);
   b. document-scoped alias expansion: tables of the same `document_id` with
      `tabgr_ready=true` whose Phase 5 candidate rows match the canonical
      metric at any unit (captures the unit-null Qingdao pattern);
   c. document-scoped lexical expansion: tables of the same `document_id`
      whose section path, caption, or header/matrix text contains a frozen
      metric alias (longest-match, `metric_aliases.json` only), capped at 40
      tables per request, ordered by `(table_index, table_id)`.
   An empty shortlist terminates with `SUPPLEMENT_NO_CANDIDATE_TABLE`.
4. TabGR QG-PPR cell retrieval: the public
   `grouped_string_with_cell_merges_w` renderer does not expose PPR scores, so
   the adapter hash-pins and calls the same source module's private scoring
   chain `_parse_triples -> _build_index -> _build_neighbors ->
   _build_personalization -> _run_ppr`, then calls the public renderer only as
   a consistency oracle. To avoid the upstream ASCII-only question tokenizer,
   personalization uses `question=None` and exact normalized `targets` built
   from frozen metric aliases, metric-year renderings, and unit renderings;
   Chinese aliases therefore participate through target-value matches rather
   than the lossy question-token path. Pinned parameters remain `model=llama3`,
   alpha 0.35, w_row 0.7, w_col 0.3, 30 iterations. Ranked triples are mapped
   back to Phase 4 cell coordinates; duplicate or otherwise non-unique triple
   mappings are discarded with an audit counter. Scores are recorded as
   strings at eight decimals with
   total-order tie breaks `(score desc, row_index, col_index)`. TabGR ranking
   only orders candidate cells; it never accepts a value.
5. Cell validation (acceptance is validation-only, ranking-blind):
   - metric: the cell's row/column label must longest-match exactly one frozen
     alias of the requested canonical metric; conservative Phase 5 exclusions
     apply (no 营业收入/主营业务收入 collapse, no generic 净利润 or 每股收益);
   - metric year: resolved with the Phase 6-equivalent rules already mirrored
     in `_candidate_metric_year` (comparison/quarter/partial-duration
     rejection before explicit/relative routing, instant-metric date rules,
     report-year window Y..Y-2) and must equal the requested year;
   - unit: explicit in-document evidence only (row/column label, table unit
     hint, header or section unit declaration such as 单位：元), normalized by
     frozen `unit_rules.json`, must equal the requested `normalized_unit`;
     absent or ambiguous evidence is `SUPPLEMENT_UNIT_UNRESOLVED`;
   - value: Decimal parse of the raw cell with Phase 5 normalization
     semantics; any normalization warning (including
     `integer_unit_has_fraction`) rejects; magnitude/sign plausibility follows
     Phase 6 rules;
   - Phase 6 eligibility mirror: reconstruct the Phase 5 candidate fields from
     the exact Phase 4 cell/table record, derive financial context from the
     frozen Phase 5 terms, and apply the Phase 6 metric-source, year-confidence,
     unit-source, main-scope, non-company/parent-only `-0.30`, and Y-2 scoring
     rules. A single value must reach the frozen 0.70 threshold. Phase 9 is
     stricter than Phase 6 for multiple distinct values and never resolves a
     conflict by confidence margin;
   - agreement: if validation accepts more than one distinct value for the
     slot, terminate with `SUPPLEMENT_VALUE_CONFLICT` and record every value;
     equal values from multiple cells merge into one fact with multiple
     provenance entries (Phase 6 dedup key: normalized decimal value).
6. Cross-store conflict check: the accepted value is compared against every
   `financial_facts` row (not only selected) sharing the exact group key,
   including `normalized_unit`; any
   differing retained value is `SUPPLEMENT_VALUE_CONFLICT`.
7. Provenance gate: the accepted cell must exactly match its
   `table_cells.jsonl` row (`table_id`, `row_index`, `col_index`, raw value,
   line range, section path, source markdown path) and its table must be
   `tabgr_ready=true` in `tabgr_table_index.jsonl` with matching
   `raw_markdown_sha1`. Any mismatch is `SUPPLEMENT_PROVENANCE_FAILED` and the
   whole request is rejected (fail-closed, no partial acceptance).
8. Accepted facts are written to the supplemental store with complete
   provenance, the TabGR trace fingerprint, and `fact_source="supplemental_tabgr"`.

### Determinism and repeatability

- No LLM, no network, no GPU, no randomness. All floats are fixed-iteration
  pure-Python arithmetic; scores serialize as eight-decimal strings.
- Canonical JSON everywhere: UTF-8, sorted keys, compact separators, one final
  newline; Decimal values as strings.
- Deterministic trace (input fingerprints, per-request decision path, scores,
  failure codes) is separated from runtime telemetry (timestamps, duration,
  PID, host paths), following the Phase 8 QATrace/QATelemetry convention.
- Two full builder runs from the same manifest must produce byte-identical
  supplemental JSONL, decisions JSONL, and deterministic trace.

## Contracts

New versioned schemas (Phase 8 schemas are not modified):

- `finglmqa.phase9.supplemental_fact.v1`: `supplemental_fact_id`
  (deterministic hash of slot key + normalized value + the sorted complete
  source-coordinate set),
  `schema_version`, slot key fields, `normalized_value` (Decimal string),
  `normalized_unit`, `statement`, `fact_source="supplemental_tabgr"`,
  `validation_versions` (alias/unit/year rule fingerprints),
  `tabgr_trace_fingerprint`, `source_table_id`, `source_row_index`,
  `source_col_index`, `source_line_start/end`, `source_markdown`,
  `provenance_json`, `created_from_requirement_ids` (sorted). When equal values
  have multiple sources, the singular source fields use the minimum
  `(table_index, row_index, col_index, table_id)` coordinate and all sources
  remain in sorted `provenance_json`.
- `finglmqa.phase9.supplement_decision.v1`: one row per request slot with
  `decision_status=accepted|rejected`, a null failure code iff accepted,
  shortlist provenance (tier a/b/c and table IDs), ranked-cell fingerprints,
  and every rejected value.
- `finglmqa.phase9.supplement_lookup_result.v1`: the read API result for the
  QA path, mirroring `fact_lookup_result.v1` plus mandatory `fact_source`.

Phase 8 integration (minimal, versioned, default-off):

- The existing `FactRepository`, `FactLookupPort`, `fact_lookup_result.v1`,
  `FactRecord`, and their validators remain unchanged. A new
  `SupplementAwareFactRepository` composes the selected repository with the
  read-only supplemental repository and still implements the frozen v1 port.
  Phase 10 enables it only by explicit dependency injection into the already
  existing `Phase8Pipeline(fact_repository=...)` boundary.
- The wrapper first queries selected facts unchanged; only on `not_found` does
  it query `finglmqa.phase9.supplement_lookup_result.v1`. A supplemental hit is
  mapped into a valid v1 FactRecord, with `fact_source="supplemental_tabgr"`
  embedded inside its canonical `provenance_json`. This preserves formula and
  direct-fact port conformance instead of sending an unknown v2 schema to
  Phase 8 consumers.
- Selected facts always shadow supplemental facts; a supplemental fact can
  never override, duplicate, or re-rank a selected fact.
- Citations built from supplemental facts carry the marker through the
  existing open citation-provenance object and are rendered with an explicit
  supplemental source note by the Composer. The conditional rendering path
  emits no new field or byte when the marker is absent. Numeric authorization
  rules are unchanged: a
  supplemental fact authorizes numbers exactly as a canonical fact does, but
  its citation can never present itself as a Phase 6 selected fact.
- `missing_fact_requests` emission in Phase 8 is unchanged: if the pipeline
  runs with the supplemental store disabled or the store has no accepted fact
  for the slot, the request is still emitted verbatim.

## Failure Codes

All Phase 9 terminal codes are prefixed `SUPPLEMENT_` and are stable:

- `SUPPLEMENT_REQUEST_INVALID`
- `SUPPLEMENT_ALREADY_SELECTED`
- `SUPPLEMENT_CONFLICT_GROUP_OPEN`
- `SUPPLEMENT_FACT_WITHHELD`
- `SUPPLEMENT_NO_CANDIDATE_TABLE`
- `SUPPLEMENT_CELL_NOT_FOUND` (shortlist yielded no metric-matching cell)
- `SUPPLEMENT_YEAR_UNRESOLVED`
- `SUPPLEMENT_UNIT_UNRESOLVED`
- `SUPPLEMENT_VALUE_INVALID` (parse/warning/plausibility)
- `SUPPLEMENT_ELIGIBILITY_REJECTED` (Phase 6 scope/confidence gate)
- `SUPPLEMENT_VALUE_CONFLICT`
- `SUPPLEMENT_PROVENANCE_FAILED`
- `SUPPLEMENT_RUNTIME_UNAVAILABLE` (TabGR import/hash-pin failure; the whole
  batch fails closed, never silently degrades to lexical-only acceptance)

Per-cell terminal-code precedence is frozen by the deepest validation stage
reached: no metric match -> `CELL_NOT_FOUND`; metric but no year ->
`YEAR_UNRESOLVED`; year but no unit -> `UNIT_UNRESOLVED`; unit but no valid
value -> `VALUE_INVALID`; valid value below the eligibility gate ->
`ELIGIBILITY_REJECTED`. A provenance mismatch on any would-be accepted source
overrides these codes with `PROVENANCE_FAILED`; multiple valid distinct values
then override with `VALUE_CONFLICT`. Request/group guards run before all cell
codes in the numbered pipeline order.

## Pipeline Plans

- `src/finglmqa/tabgr_adapter.py`: hash-pin check of imported TabGR sources,
  edge-list loading from `tabgr_table_corpus.jsonl`, target-set construction
  from frozen aliases/years/units, private scoring-chain invocation with pinned
  parameters, public-renderer consistency check, triple-to-cell mapping, and
  deterministic score serialization.
- `src/finglmqa/supplement_validation.py`: metric/year/unit/value validators
  reusing the frozen rule configs; every function pure and covered by
  adversarial tests.
- `src/finglmqa/supplement_store.py`: supplemental DuckDB + JSONL writer,
  read-only `SupplementalFactRepository`, and v1-conformant
  `SupplementAwareFactRepository` wrapper.
- `scripts/build_supplement_requests.py`: request-universe builder.
- `scripts/build_supplemental_facts.py`: the reproducible Phase 9 command
  (universe -> pipeline -> stores -> reports), single deterministic entry.
- `scripts/validate_phase_09_gate*.py`: gate validators.

Storage: `data/facts/supplemental_facts.duckdb` (tables `supplemental_facts`,
`supplement_decisions`, `build_metadata`) plus canonical
`data/facts/supplemental_facts.jsonl` and
`data/schemas/supplemental_facts.schema.json`. The Phase 6 database is never
opened for writing.

## File Layout

    runs/phase_09/
      phase_09_plan.md                      (this file)
      immutable_inputs_manifest.json        (Gate 0)
      gate0_report.json ... gate7_report.json
      reports/
        supplement_requests.jsonl           (batch universe)
        supplement_decisions.jsonl          (one row per slot)
        supplement_summary.json             (counts by failure code / metric / offset)
        qingdao_port_case_report.json       (named fixture outcome)
      waves/                                (per-wave path/hash/mtime manifests)
    data/facts/supplemental_facts.duckdb
    data/facts/supplemental_facts.jsonl
    data/schemas/supplemental_facts.schema.json
    src/finglmqa/tabgr_adapter.py
    src/finglmqa/supplement_validation.py
    src/finglmqa/supplement_store.py
    scripts/build_supplement_requests.py
    scripts/build_supplemental_facts.py
    scripts/validate_phase_09_gate0.py ... gate7

## Gates

- Gate 0 — Baseline: immutable manifest with all hash pins above;
  `networkx==3.5` installed via one recorded `uv pip install`, with resolved
  version, Python version, and installed package-tree hash in
  `runs/phase_09/dependency_manifest.json`; TabGR module import succeeds and
  source hash pins match; Phase 6/7/8 artifact hashes unchanged.
- Gate 1 — Contracts: the three new schemas validate; the supplement-aware
  wrapper passes the unchanged v1 FactLookupPort contract; failure-code enum frozen;
  request-universe builder produces a contract-valid, sorted, byte-stable
  file whose exact-unit slot arithmetic reconciles exactly (5,610 grid, 4,189
  covered, 1,421 missing, 501/30/890 classes).
- Gate 2 — Adapter determinism: two fresh-process QG-PPR runs over a pinned
  table sample return byte-identical ranked-cell serializations; triple-to-
  cell mapping is exact against `table_cells.jsonl`; a corrupted TabGR source
  file fails closed with `SUPPLEMENT_RUNTIME_UNAVAILABLE`.
- Gate 3 — Validation adversarial suite: wrong-metric alias, near-synonym
  collapse attempts, comparison/quarter/partial-duration labels, relative-year
  edge cases, missing/ambiguous unit, non-company/parent-only scope, confidence
  below 0.70, forged provenance fields, fractional person-unit, and
  conflicting-value injections all reject with the exact expected code. The
  Flyada 2019 revenue validator fixture must reproduce the selected value, but
  a full pipeline request terminates at `SUPPLEMENT_ALREADY_SELECTED` before
  TabGR and writes no fact. The China Railway Signal partial-duration candidate
  must stay rejected.
- Gate 4 — Full batch build: `build_supplemental_facts.py` runs the complete
  1,421-slot universe; every slot has exactly one terminal decision row; zero
  accepted facts overlap a selected-fact slot key; zero accepted facts have a
  differing `financial_facts` value; two runs are byte-identical.
- Gate 5 — Store integrity: DuckDB and JSONL reconcile row-for-row; schema
  file validates; Phase 6 database hash/size/mtime unchanged after the build.
- Gate 6 — Phase 8 integration: with supplemental disabled, the full Phase 8
  Gate 2 oracle (1,003 + 40) and the real-evidence trace re-run byte-identical
  to the frozen release; with supplemental enabled, a supplemented lookup
  passes the unchanged v1 FactLookupPort contract with
  `fact_source="supplemental_tabgr"` in provenance, direct-fact and formula
  paths both render the supplemental marker, citation scope validation passes,
  and a non-supplemented slot still emits the verbatim missing-fact request.
- Gate 7 — Reports and audits: summary report with per-code/per-metric/
  per-offset counts, Qingdao Port named-case report, read-only independent
  audit that no official artifact contains raw table content beyond cited
  cells, host absolute paths, or telemetry fields; governance docs updated
  (implementation stage).

## Validation Matrix

| Check | Gate | Method |
| --- | --- | --- |
| Immutable inputs unchanged | 0, 5, 6 | manifest re-hash |
| TabGR source hash pins | 0, 2 | SHA-256 vs pinned constants |
| Request-universe reconciliation (5,610/1,421/501/30/890) | 1 | independent recount from DuckDB + index |
| Contract validity of every request/decision/fact row | 1, 4 | validators over full files |
| QG-PPR repeatability | 2 | two fresh processes, byte compare |
| Ranking never accepts a value | 3 | adversarial: top-ranked wrong cell must reject |
| Metric/year/unit/value/provenance rejections | 3 | exact-code adversarial suite |
| No selected-slot overlap, no conflicting acceptance | 4 | full-store cross join checks |
| Batch byte-identical repeatability | 4 | two full runs |
| Phase 8 frozen behavior with supplemental off | 6 | Gate 2 oracle + trace byte compare |
| Supplemental marker end-to-end | 6 | pipeline case with citation/source assertions |
| Qingdao Port named fixture | 7 | recovered with explicit unit evidence, or explicit failure code with audited reason |

## Completion Targets

- 100% of the 1,421-slot universe terminates with exactly one deterministic
  decision row; zero silent failures.
- Zero accepted supplemental facts that violate metric/year/unit/value/
  provenance validation or conflict with any retained Phase 6 value
  (hard target: 0).
- Recovery of the 890 exact-unit no-fact slots is reported, not promised: the summary
  must state accepted/rejected counts per failure code. Fail-closed rejection
  of all 890 is an acceptable (if disappointing) outcome; fabricated
  acceptance is not.
- Qingdao Port case closed one way or the other with an audited report.
- Phase 8 regression: byte-identical with supplemental disabled; the complete
  existing Phase 8 test suite and all frozen Gates still pass.
- Phase 6/7/8 artifact hashes unchanged.

## Execution

One implementation subagent executes W0-W4 sequentially and may not start
nested agents. It owns new Phase 9 schemas, modules, scripts, tests, run
artifacts, and the narrowly conditional supplemental-marker branch in
`src/finglmqa/composer.py`. It must not edit the existing Phase 8 port,
repository, formula, authorization, schema, registry, or planning contracts.
The primary agent owns this plan, independently reviews every Gate, repairs any
failed invariant, reruns the full validation suite, and performs the final
governance update.

- W0 / Gates 0-1: immutable/dependency manifests, schemas, failure codes, and
  the 1,421-request exact-unit universe.
- W1 / Gates 2-3: TabGR adapter, validation/eligibility mirror, and adversarial
  tests.
- W2 / Gates 4-5: supplemental store, full batch builder, two-run byte
  comparison, and immutable-input recheck.
- W3 / Gate 6: v1-conformant composed repository, conditional Composer marker,
  direct-fact/formula integration tests, and frozen Phase 8 regression.
- W4 / Gate 7: reports and audits. Governance docs are proposed by the
  subagent and finalized by the primary after independent verification.

## Handoff

- Phase 10 receives: an unchanged default Phase 8 pipeline, the explicitly
  injectable v1-conformant `SupplementAwareFactRepository`, and a documented
  enablement switch. Phase 10 decides service-level enablement; it may not
  alter validation or acceptance semantics.
- A later curation phase receives the untouched 501 conflict groups, the 30
  withheld facts, and the complete Phase 9 decision audit as its input
  inventory.
- A later evidence-rebuild phase receives the mixed-narrative table chunk
  problem unchanged.

## Rollback

- The supplemental store is additive and default-off. Full rollback is:
  delete `data/facts/supplemental_facts.duckdb`,
  `data/facts/supplemental_facts.jsonl`,
  `data/schemas/supplemental_facts.schema.json`, and revert the Phase 9
  source/script files. No Phase 6/7/8 artifact is modified, so no restore is
  needed.
- Because Phase 8 defaults to supplemental-off and Gate 6 proves byte-identical
  behavior in that mode, disabling the flag is an instant behavioral rollback
  without file deletion.
- `networkx==3.5` can be uninstalled with one `uv pip uninstall`; nothing outside
  Phase 9 imports it.

## Planning Decisions

- Decision: install exactly `networkx==3.5` as the single new `.venv`
  dependency, record its installed tree fingerprint, and run
  TabGR QG-PPR in-process via a FinGLMQA-owned adapter; never start
  `serve_api.py` or import TabGR LLM modules.
  Rationale: the QG-PPR path is deterministic pure Python but its module
  imports `networkx` at top level; one small, pure-Python install (Phase 6
  `duckdb` precedent: single recorded `uv pip install`) is cheaper and more
  faithful than re-vendoring TabGR code, while a FastAPI service belongs to
  Phase 10 and would add two more dependencies. Phase 4's pure-Python adapter
  precedent covered table structuring only; Phase 9 actually executes the
  graph ranking, so importing the pinned original source is the honest
  "really runs TabGR" option. Fail-closed if import or hash pins fail.
- Decision: run supplementation as an offline batch over a synthetic request
  universe shaped by the strict Phase 8 contract; the QA path only performs
  lookups.
  Rationale: the exact-unit universe is small (1,421 slots) and fully
  enumerable; batch
  execution gives complete audit coverage, keeps TabGR and `networkx` out of
  the QA hot path, and makes online behavior a deterministic store lookup.
- Decision: exclude the 501 conflict groups and 30 low-confidence withheld
  facts from supplementation with dedicated terminal codes.
  Rationale: their values already exist with provenance; the gap is a selection
  policy question. Re-extracting them via TabGR cannot add evidence and risks
  laundering an unresolved conflict into a "supplemental" fact.
- Decision: allow a bounded, deterministic document-scoped shortlist expansion
  (metric-at-any-unit candidates, then frozen-alias lexical match, cap 40)
  beyond `candidate_table_ids`.
  Rationale: measured exact-unit data shows `FallbackCandidateIndex` returns
  empty for all 890 recoverable slots: 123 have some unit-null evidence, 259
  expose only the other unit, and 508 have no year-resolvable candidate.
  Without expansion Phase 9 would be a no-op; the
  expansion stays inside the request's single document and is fully audited
  per tier.
- Decision: acceptance is validation-only; TabGR ranking merely orders cells.
  Rationale: preserves the no-LLM/no-heuristic-acceptance governance line;
  a top-ranked but wrong-year/wrong-unit cell must still reject.
- Decision: integrate through a separate, explicitly injected
  `SupplementAwareFactRepository` that remains conformant to the frozen v1
  FactLookupPort; selected facts always shadow and supplemental source identity
  is carried inside canonical provenance.
  Rationale: returning a v2 record through the existing port would be rejected
  by the frozen FormulaExecutor/validator. Composition plus provenance marking
  keeps the contract byte-stable and still makes the source explicit end to end.
- Decision: 股本 batch slots are enumerated under both 元 and 股.
  Rationale: Phase 8 refuses a default unit for share capital because selected
  facts carry both; the batch universe must mirror exactly the requests
  Phase 8 is allowed to emit.

## Decision

Proceed with Phase 9 as specified: a deterministic, fail-closed, batch-first
TabGR supplementation pipeline targeting only exact-unit no-fact gaps (890
slots measured inside a 1,421-request audit universe), a separate versioned
supplemental store, and a default-off, v1-conformant composed repository.
Conflict-group curation,
mixed-narrative evidence recovery, and service lifecycle remain explicitly
deferred.
