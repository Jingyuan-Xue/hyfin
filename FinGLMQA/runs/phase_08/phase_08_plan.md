# Phase 8: Static Composition QA

## Status

- Implementation complete; Gates 0–8 passed on 2026-07-13.
- Contract date: 2026-07-13.
- Phase 8 consumes immutable Phase 6 selected facts and Phase 7 evidence.
- Phase 9 TabGR completion and Phase 10 service lifecycle are out of scope.

## Objective

Build a deterministic static DAG:

    QuestionAnalysis
      -> ScopePlan
      -> CompositionPlan / ordered SubPlan[]
      -> structured execution
      -> NumericAuthorizationSet
      -> evidence execution
      -> Composer

The only public backends are `fact`, `sql`, `formula`, and `evidence`.
Metadata lookup is `backend=fact, operation=metadata_lookup`.

Every SubPlan is frozen before execution. `depends_on_subplan_ids` expresses
completion dependencies only. Results may authorize numbers in an already
planned evidence node, but may not create a new node or change a node target.

Therefore data-dependent questions such as “find the highest-revenue company,
then explain its reasons” return `COMPOSITION_UNSUPPORTED`, with no SQL or
evidence calls.

## Multi-document Boundary

Phase 8 permits multiple independent single-document evidence SubPlans. The
Composer may render company or report-year sections side by side. It may not:

- jointly retrieve or rerank evidence from multiple documents;
- transfer one company's evidence to another company;
- infer cross-company causal explanations;
- fuse multi-document narrative into a new conclusion.

Structured fact/formula results may support explicit numeric comparison. Later
multi-document work owns dynamic document-set discovery, joint retrieval, and
cross-document synthesis.

Each ready evidence SubPlan binds exactly one `document_id`. A
`company_documents` scope is never passed directly to A2RAG. If a narrative
question omits report year and multiple reports match, the intended evidence
node is blocked with `RESOLVER_AMBIGUOUS`; sibling company nodes may proceed.

## Hard Limits

- maximum company output axis: 5;
- maximum period output axis: 5;
- maximum SubPlans, including blocked placeholders: 12;
- maximum evidence `top_k` per SubPlan: 5;
- maximum emitted evidence chunks per request: 25;
- simultaneous multi-company and multi-period output axes: unsupported.

Formula operand years are not output periods. A three-company growth-rate
question is one entity axis with three formula nodes and is supported.

Limit violations produce `COMPOSITION_LIMIT_EXCEEDED`, no CompositionPlan, no
SubPlans, and no backend calls.

## Frozen Analysis and Scope Contracts

QuestionAnalysis records normalized question/hash, ordered company/year/metric/
formula mentions, ordered concerns, six intents, evidence kinds, output axes,
dynamic target dependency, unsupported markers, and ambiguities.

The six intents are:

- lookup
- compare
- rank
- aggregate
- calculate
- narrative

`explain` and `summarize` are narrative modes. Evidence kinds are
`structured_fact`, `table`, and `narrative`. Table-only evidence excluded from
the Phase 7 text index must not be routed to evidence.

Scope kinds and direct capabilities are frozen in `contracts.py`:

- `single_document`: all four backends may execute compatible operations;
- `company_documents`: fact/sql direct; formula/evidence require a unique
  single-document subscope;
- `multi_company_documents`: resolve and narrow each entity independently;
- `explicit_document_set`: contract only in Phase 8;
- `corpus`: SQL rank/aggregate only.

## Nine Static Topologies

The registry is `src/config/composition_patterns.json`. It contains exactly:

1. `single_node`
2. `parallel_concerns`
3. `entity_list`
4. `entity_compare`
5. `period_list`
6. `period_compare`
7. `single_document_bundle`
8. `entity_section_bundle`
9. `period_section_bundle`

Each registry entry freezes allowed scopes/intents, static shape, ordering,
required/optional selector, minimum usable results, comparison quorum selector,
measure compatibility, composition policy, and `dynamic_expansion=false`.

All explicit user concerns are required. Phase 8 creates no optional helper
nodes. Matching precedence is boundary rejection, narrative topology, entity
topology, period topology, parallel concerns, then single node. Multiple valid
matches fail with `ROUTE_AMBIGUOUS`; priority cannot hide selector overlap.

Ordering is question mention order: entity, then output period, then concern.
Within an entity/period section structured nodes precede its evidence node.

Benchmark type is validator-only metadata. The production planner never reads
it. The Gate 2 oracle maps the current selected benchmark as follows:

- type1: one fact node;
- type1-2: `parallel_concerns`, normally two fact nodes;
- type2-1: one formula node;
- type2-2: `period_compare`, blocked unsupported metadata nodes;
- type3-1: one single-document evidence node.

SQL is reserved for registered document/corpus QuerySpecs such as rank,
aggregate, equality, or series queries. It is not used merely to collapse two
ordinary field lookups into one node.

## SubPlan and Execution Rules

Every SubPlan records canonical ID/ordinal, planning state, intended backend and
operation, entity/period/concern keys, scope reference, completion dependencies,
numeric-authorization dependencies, required flag, declared scope, payload, and
planning failure.

- ready: non-empty operation payload and no planning failure;
- blocked: no payload, stable planning failure, zero backend calls;
- metadata remains a fact operation;
- evidence requires one declared document;
- dependencies only reference earlier nodes;
- authorization dependencies are a subset of completion dependencies.

Completion dependency does not mean success dependency. An evidence node may
still emit non-financial narrative when a structured node fails, but it receives
no authorization from that failed node.

Formula and SQL executors may read FactRepository internally. Those reads are
part of the selected executor and do not create hidden fact SubPlans or backend
fallback.

## Usability and Composer

Usability is derived by the boundary validator, not asserted by an executor:

- fact: value, unit, and provenance;
- formula: Decimal result plus fully cited operands;
- SQL: QuerySpec-valid rows with contributing fact IDs;
- evidence: at least one valid claim; chunks alone are not usable.

All required results usable and quorum satisfied yields `ok`. Any usable result
plus a failed required node yields `partial`. If comparison quorum is not met
but one result is usable, return `partial + COMPOSITION_QUORUM_NOT_MET`, retain
the successful value, and emit no comparison conclusion.

When no result is usable, precedence is:

    error > blocked > needs_clarification > fallback_required > unsupported > not_found

Planning blocked maps to an unexecuted SubPlanResult: resolver missing becomes
not_found; resolver/metric/unit ambiguity becomes needs_clarification;
unsupported metric/scope/table evidence becomes unsupported; safety failures
become blocked.

Any `PROVENANCE_VALIDATION_FAILED` overrides all other outcomes. The entire
answer is blocked and answer text/result/citations are cleared; only safe status
and validation trace remain.

Incomplete corpus coverage returns a partial SQL result with warning
`CORPUS_COVERAGE_INCOMPLETE`. It must not claim a full-corpus maximum or create
hundreds of missing-fact requests.

## Numeric Authorization and Citation Scope

NumericAuthorization is a union:

- `canonical_fact`: canonical metric and one metric year;
- `formula_result`: formula/version, target year, and operand years;
- `sql_result`: QuerySpec/result-row/measure identity.

Every authorization records source node/backend/row, company/entity, exact
document when used by evidence, Decimal value/unit, output formatting, sorted
renderings, and source/provenance citation IDs.

EvidenceExecutor owns provider call, chunk scope validation, deterministic span
selection, number classification/filtering, citation construction, and
SubPlanResult creation. It may authorize a financial number only for the same
entity and document. Formula results cannot masquerade as canonical facts.

Citation kinds are fact, metadata, evidence, formula_derivation, and
sql_derivation. Derivation citations reference their underlying fact citations.
Claim, entity, document, SubPlan, and citation scope must agree.

## Missing Fact Requests

`fallback_request` is removed. Phase 8 emits only `missing_fact_requests[]`.
Each request includes a canonical requirement ID, origin operation, formula and
operand role when applicable, node/document/company/report/metric years,
canonical metric, catalog-provided unit, and sorted candidate table IDs.

Requests are allowed only for a uniquely resolved company, report, metric,
metric year, and expected unit. They may originate from direct fact, formula
operand, or document-scoped SQL. Metadata, unknown metrics, evidence misses,
unit ambiguity, and corpus coverage never generate requests. Share capital has
no default query unit because selected facts contain both 元 and 股.

The read-only helper is named `FallbackCandidateIndex`. Phase 8 does not execute
TabGR.

## Trace and Telemetry

QATrace contains only deterministic semantics and workspace-relative artifact
fingerprints. QATelemetry separately contains timestamps, duration, PIDs,
device/process details, cache path, resource data, and exception stacks.

Canonical JSON is UTF-8, sorted keys, compact separators, and one final newline.
Decimal and dense scores are strings. Phase 8 preserves Phase 7 ranking semantics
and records score to eight decimals plus document chunk ordinal. Repeatability is
defined for the frozen model/runtime/device.

The registry semantic SHA-256 hashes parsed canonical JSON. The immutable
manifest separately names the raw file SHA-256; the two hashes are never mixed.

## Gates and Ownership

The primary agent owns contracts, schemas, errors, trace/telemetry, pattern
registry, pipeline, CLI, official validator, run artifacts, environment, and
governance docs.

- W0 / Gates 0–1: baseline, contracts, registry, fixtures, 1,003-row oracle,
  40 General gold cases.
- W1 / Gate 2: analyzer/catalog, resolver, and composition planner in disjoint
  agent-owned files.
- W2 / Gates 3–5: repositories, SQL, and formula in disjoint files; fake and
  real FactLookupPort run the same conformance suite.
- W3 / Gates 6–7: provider transport, EvidenceExecutor, and Composer/global
  validator in disjoint files; primary integrates pipeline.
- W4 / Gate 8: full benchmark/capability run, read-only independent audits,
  reports, and governance updates.

The workspace is not a Git repository. Every wave records a recursive path/hash/
mtime manifest. Agents may modify only their explicit file allow-list and must
not edit Phase 1–7 artifacts.

## Validation

Gate 2 uses a 1,003-row UID-authoritative benchmark oracle plus 40 manually
reviewed General cases covering multi-intent, partial company resolution,
multiple formulas, multiple output periods, dynamic-target rejection,
multi-report ambiguity, metadata/financial mixtures, and corpus-rank-plus-reason
rejection.

Determinism and correctness are separate gates: two identical decompositions
prove repeatability; the independent General gold set proves expected behavior.

Mandatory adversarial cases include zero-call dynamic planning, no silent
backend switch, over-decomposition, company-local failure, comparison quorum,
scope-limit rejection, evidence without claims, cross-company citation poisoning,
missing-request eligibility, fake/real port conformance, telemetry exclusion,
and unchanged Phase 6/7 hashes.

## Subsequent Phases

Phase 9 consumes missing requests, validates TabGR output through the same
metric/year/unit/conflict/provenance gates, and stores supplemental facts outside
the immutable Phase 6 store with an explicit supplemental source marker.

Phase 10 owns only API and runtime lifecycle: FastAPI, workers, GPU, concurrency,
timeouts, logs, and projection. It cannot redefine Phase 8 analysis, planning,
execution, Composer, or status semantics.

Later multi-document work may enable explicit document sets, SQL-result-driven
evidence targets, joint retrieval/reranking, and fused synthesis. Multi-turn
context resolution remains unscheduled.
