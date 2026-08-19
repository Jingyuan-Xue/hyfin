# scripts

Executable entry points only. Reusable business logic belongs in
`src/finglmqa/`.

Type 3 A2RAG + TabGR experiment entry points:

- `build_type3_corpus_profile.py` — builds the corpus and the sanitized question
  profile from explicit read-only inputs.
- `validate_type3_a2rag_tabgr.py` — audits the Phase 1 corpus, the questions, the
  frozen v8/v10 runs and the ablations.

These entry points write only into the `data/**/type3` and
`runs/type3_a2rag_tabgr_experiment_*` namespaces. They must never overwrite
official Phase artifacts or historical Type 3 outputs.

> Running the demo does not require anything in this directory. Use `./start.sh`
> at the repository root instead — see [docs/DEPLOY.md](../../docs/DEPLOY.md).

## Build entry points

| Script | Phase / purpose |
|---|---|
| `scan_corpus.py` | Phase 2 document and company-year index |
| `parse_markdown_documents.py` | Phase 3 Markdown parsing |
| `build_tabgr_tables.py` | Phase 4 TabGR structuring |
| `build_metric_unit_layer.py` | Phase 5 metric and unit layer |
| `build_financial_fact_store.py` | Phase 6 official fact store |
| `build_a2rag_text_index.py` | Phase 7 evidence allow-list |
| `build_supplement_requests.py` | Phase 9 missing-fact requests |
| `build_supplemental_facts.py` | Phase 9 supplemental audit and storage |

Rebuilding a Phase 6 artifact changes its SHA-256 and therefore invalidates
`runs/phase_10/immutable_inputs_manifest.json`. The service will refuse to start
until that manifest is regenerated.

## Query and service

- `query_phase_08.py` — CLI for the official pipeline.
- `query_type3_evidence.py` — Phase 7 document-scoped retrieval; also the worker
  process the live service launches.
- `serve_phase_10_api.py`, `serve_phase_10_worker.py` — internal service entry points.
- `start_finglmqa.sh`, `status_finglmqa.sh`, `stop_finglmqa.sh` — the only
  supported lifecycle entry points for the QA service on its own.

## Gates and finalization

- `prepare_*`, `capture_*`, `record_*` — freeze inputs or execute run waves.
- `validate_phase_*` — Phase gates.
- `finalize_phase_10.py`, `finalize_phase_11.py` — release and governance wrap-up.

## Evaluation and experiments

- `eval_phase_10_http.py`, `eval_finglmqa.sh` — official HTTP evaluation.
- `eval_type3_no_llm*.py` — Type 3 experiments without an LLM.
- `score_no_llm_benchmark.py` — BGE-M3 scoring of frozen answers.
- `build_table_evidence_index.py`, `run_table_qwen_experiment.py` — table experiments.
- `run_type3_qwen36_organization_v8.py`, `project_type3_qwen36_fallback_v8.py` —
  Qwen passage selection and safe fallback.
- `run_type3_qwen36_faceted_v9.py`, `project_type3_qwen36_unanimous_v9.py` —
  faceted recall, id selection and 3/3 projection.
- `run_type3_qwen36_coverage_v10.py` — hybrid candidate recall, Qwen coverage
  planning and 2/3/3/3 projection.

Script arguments are defined by `--help` and the corresponding run report. Never
write host paths, PIDs, temporary directories or benchmark reference answers
into answer-generation logic.
