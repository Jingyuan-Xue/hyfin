# Architecture

How HyFin serves its three browser-facing modules, and where each answer's
evidence comes from. For installation see [DEPLOY.md](DEPLOY.md).

---

## Processes

Three processes, started in dependency order by `start.sh`:

```
browser
   │
   ▼  :4173
┌──────────────────────────────────────────────┐
│ Gateway  icdm_demo.stable_backend            │  access token, rate limits
│          └─ icdm_demo.final_backend          │  API, static files, translation
└───────┬───────────────────────────┬──────────┘
        │ :8010                     │ :8012
        ▼                           ▼
┌───────────────────────┐   ┌──────────────────────┐
│ FinGLMQA service      │   │ Risk exposure API    │
│ (Python 3.14)         │   │ serves frozen 2023   │
│   └─ QA worker        │   │ SW3 artifacts        │
│      (Python 3.10)    │   └──────────────────────┘
│      A2RAG + TabGR    │
└───────────────────────┘
```

The QA worker is a **separate interpreter** launched as a subprocess over a
stdin/stdout protocol. That boundary exists so the embedding stack (torch,
BGE-M3) never shares a process with the service, and so a worker crash degrades
readiness instead of taking the API down. It is single-concurrency by design.

Only the gateway binds a public interface. The QA and risk services bind
`127.0.0.1`.

---

## The three modules

### Industry exposure

Reads **precomputed** artifacts — dense-hybrid labels plus cached A2RAG text
evidence and TabGR table evidence from `05_evidence/*.json`. No retrieval runs
at request time. This is a replay of a frozen experiment, not a live pipeline.

### Risk exposure

Serves validated 2023 risk-exposure artifacts for the SW3 cohort from
`risk_exposure_method/output/`. Also static.

### QA

The only live path. Retrieval and generation run per request against the
170-report corpus.

---

## The QA request path

```
POST /api/finglmqa/qa
      │
      ├─ translate question to Chinese if needed (display stays English)
      ├─ optionally scope the question to one company/year
      ▼
   FinGLMQA /api/v1/qa
      │
      ▼
   Phase 8 pipeline (deterministic)
      │
      ├─ metric/metadata questions ──▶ frozen DuckDB fact store, no LLM
      │
      └─ narrative questions ────────▶ A2RAGTabGRHybridEvidenceProvider
                                          ├─ A2RAG dense text chunks (BGE-M3)
                                          └─ TabGR structured table rows
                                                    │
                                                    ▼
                                          online model composes the answer
                                          from retrieved evidence only
```

Numeric claims are gated: a value may appear in an answer only if it is
authorized by the fact store or by cited evidence. Citations carry
`provenance.evidence_chunk_id`; table-channel rows are prefixed `tabgr:`.

### Evidence channel mixing

`hybrid_evidence_provider.py` fuses the two channels by **deterministic
interleaving**, not by score:

- `_table_quota` allots at most 2 table rows out of `top_k`, and only 1 unless
  the question contains one of a fixed list of Chinese numeric terms — 营业收入
  (revenue), 利润 (profit), 多少 (how much), and roughly twenty more. The list is
  literal and untranslated: an English question only reaches the higher quota
  after the gateway has translated it to Chinese.
- Table chunks are emitted with `score: 0`, then every chunk is re-scored by
  rank position.

This is deliberate — A2RAG cosine similarity and TabGR lexical scores are not
calibrated to a common scale, so comparing them numerically would be
meaningless. The consequence is that the table channel's contribution is capped
by the quota rather than earned by relevance. See
[Known limitations](../README.md#known-limitations).

### Multi-report questions

Up to three reports can be selected. Retrieval runs **once per report**,
sequentially, because the worker is single-concurrency. A single
`/api/finglmqa/consolidate` call then merges the per-report answers, with every
citation still labelled by company and year.

Per-report answers are restored to their original Chinese before consolidation,
so the model reasons over retrieved text rather than a translation. When several
reports are selected, ask the question without naming a company — the selection
defines the scope.

---

## Integrity gates

Two independent mechanisms keep the demo honest:

**Frozen input manifest.** `FinGLMQA/runs/phase_10/immutable_inputs_manifest.json`
SHA-256 pins the fact store and corpus. The supervisor verifies it at startup
and re-checks cheaply thereafter; a mismatch means the service never reports
ready. This is why pinned artifacts must not be recompressed or reformatted.

**Measured readiness.** `/health/ready` reports retrieval state read off the
provider that was actually constructed — `tabgr_document_count` is
`len(retriever.document_ids)`, and `evidence_provider_version` names the class in
use. `/api/v1/meta` derives `online_evidence` from those values, so a service
running without a usable TabGR index advertises `a2rag` rather than claiming
hybrid retrieval it cannot perform. `selfcheck.sh` asserts against these
measured fields.

---

## Language handling

Retrieval and reasoning always run in Chinese. English mode translates only the
display projection, server-side, through the Tencent adapter. Canonical Chinese
questions, document ids, numeric table cells and original evidence are preserved
unchanged; translations are cached by content hash in
`runtime/translation_cache.sqlite3`.

Credentials live in `.env` and are read only by the gateway. They are never
exposed to frontend JavaScript.

---

## Where the data lives

| Path | Used by | Live at request time |
|---|---|---|
| `FinGLMQA/data/facts/*.duckdb` | metric answers | yes |
| `FinGLMQA/data/indexes/a2rag_index/` | text retrieval | yes |
| `FinGLMQA/data/indexes/type3/.../tabgr/` | table retrieval | yes |
| `models/models--BAAI--bge-m3/` | query embedding | yes |
| `output/.../04_a2rag_docs/` | source markdown for citations | yes |
| `risk_exposure_method/output/` | risk module | served as-is |
| `FinGLMQA/runs/phase_*/` | manifests and reports | startup verification only |
