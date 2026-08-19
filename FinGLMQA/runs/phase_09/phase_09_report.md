# Phase 9 Completion Report

## Outcome

Phase 9 is complete and Gates 0–7 pass. The offline batch enumerated the
strict exact-unit universe, executed the hash-pinned TabGR QG-PPR scorer, and
terminated all 1,421 requests without modifying any Phase 1–8 artifact.

The accepted supplemental-fact count is **0**. This is a valid fail-closed
outcome under the frozen plan: none of the 890 executable no-exact-fact slots
simultaneously met the metric, year, explicit-unit, value, Phase 6 confidence,
conflict, and cell-exact provenance gates. The one initially plausible
share-capital cell (`48,946,000.0o`) was rejected after reproducing Phase 5's
`unknown_unit_suffix` semantics; it is not present in the store.

## Exact-unit inventory

- Grid: 5,610 slots (170 documents × 3 years × 11 metric/unit keys).
- Phase 6 selected coverage: 4,189.
- MissingFactRequest audit universe: 1,421.
- Conflict groups withheld: 501.
- Low-confidence facts withheld: 30.
- No exact-unit fact: 890.

The 1,421 decisions end as:

- `SUPPLEMENT_CONFLICT_GROUP_OPEN`: 501
- `SUPPLEMENT_FACT_WITHHELD`: 30
- `SUPPLEMENT_YEAR_UNRESOLVED`: 469
- `SUPPLEMENT_UNIT_UNRESOLVED`: 376
- `SUPPLEMENT_NO_CANDIDATE_TABLE`: 24
- `SUPPLEMENT_CELL_NOT_FOUND`: 15
- `SUPPLEMENT_VALUE_INVALID`: 6

## Implementation

- Installed only `networkx==3.5`; the 597-file installed tree fingerprint is
  `183cdf8ad9a8dd17fb4bc7d43d8dd4c42a4c61d2672020d5531b7121119c4410`.
- Added three Phase 9 contracts/schemas and a canonical 1,421-row request
  builder.
- Added a TabGR adapter that hash-pins upstream source, invokes the private
  scoring chain, verifies the public renderer, maps triples to exact Phase 4
  cell coordinates, and emits eight-place scores.
- Added validation mirroring Phase 6 year, unit, value, scope, confidence,
  conflict, and provenance rules. Matrix/header/row checks independently
  detect forged cell coordinates or values.
- Added a separate DuckDB/JSONL supplemental store and a default-off,
  v1-conformant `SupplementAwareFactRepository`. Selected facts always shadow
  supplemental rows.
- Added a conditional Composer source marker. With no supplemental provenance
  marker, all 171 Phase 8 tests remain unchanged; marked direct facts and
  formulas render `补充来源：TabGR`.

## Determinism and integrity

Two complete 1,421-slot builds produced identical canonical JSONL bytes:

- Supplemental facts JSONL (0 rows):
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Decisions JSONL (1,421 rows):
  `1365a501b68cac4956fe359a35a9d194df4af83752a2ca52798c20508e175411`
- Deterministic trace JSONL (1,421 rows):
  `a9d40e46e679319b0d12e1ae7e57405bfae497e4881702244cd095bbcc9fc97e`

DuckDB and JSONL reconcile row-for-row. DuckDB file bytes are not used as the
repeatability oracle because storage metadata differs between fresh database
files; semantic tables and canonical JSONL are the frozen comparison surface.

Immutable release hashes remained unchanged, including:

- Phase 6 DuckDB:
  `b3e8fed65ddc1ccd5954083a4df64f3eab2150294cae08a11424f3bc5744f278`
- Phase 6 JSONL:
  `abeb4b3b221aac74705b84c80469c03b23fd8638d67004c75dd7a512c6841405`
- Phase 7 evidence chunks:
  `7dd6d793b6a2e70f91f867dd7163bbae7adff3ade8ce2c7277607cdb3d6ce7d5`
- Phase 7 document map:
  `c5fa29ffef0008cf8ce5e5fc3d186afac9a4beae1c8521c1804f4b4ec1644164`
- Phase 8 Gate 2 report:
  `bfc0e1d39b78daf3716b4226d31ce581747ae376a510b0fc6ec8325fb7ef83bb`
- Phase 8 Gates 6–8 report:
  `430f7d75e258a1151a7251e68920b181db76636b48aded87b173e40c43eccf7a`

## Gates

- Gate 0: 12 immutable pins, exact dependency, and TabGR import passed.
- Gate 1: schemas, v1 wrapper, request contracts, and unit arithmetic passed.
- Gate 2: fresh-process scoring bytes, exact cell mapping, renderer oracle,
  score precision, and corrupted-source fail-closed behavior passed.
- Gate 3: 14 adversarial tests passed, including Flyada, China Railway Signal,
  parent-only scope, unknown suffix, unit ambiguity, and forged provenance.
- Gate 4: 1,421 decisions, zero overlaps/conflicts, and two-build JSONL
  repeatability passed.
- Gate 5: separate store reconciliation and immutable-input checks passed.
- Gate 6: 171/171 Phase 8 tests passed; contract-only supplemental fixtures
  passed direct/formula marker and missing-request behavior. Two fresh real
  BGE-M3 runs produced answer/trace bytes identical to each other and the
  frozen Phase 8 release (`trace_hash=6ce39bdc...1539`).
- Gate 7: Qingdao Port closed as 0 accepted / 33 audited rejections; official
  outputs contain no raw-table payload, host absolute path, or telemetry field.

## Handoff

Phase 10 may inject `SupplementAwareFactRepository` explicitly, but enabling it
currently adds no production facts because the accepted store is empty.
Conflict curation (501 groups), withheld low-confidence facts (30), and wider
unit recovery remain separate future work. Phase 10 must not weaken Phase 9
validation or change Phase 8 QA semantics.
