# src

`src/finglmqa/` is the reusable business package. `src/config/` holds
configuration constrained by a semantic hash.

## Module layers

| Layer | Modules |
|---|---|
| Analysis and planning | `analyzer.py`, `resolver.py`, `composition.py`, `metric_catalog.py` |
| Contracts and errors | `contracts.py`, `ports.py`, `errors.py`, `trace.py` |
| Structured execution | `repositories.py`, `structured_execution.py`, `sql_engine.py`, `formula_engine.py` |
| Evidence execution | `evidence_provider.py`, `evidence_executor.py`, `authorization.py` |
| Composition and output | `composer.py`, `pipeline.py` |
| Phase 9 | `supplement_*`, `tabgr_adapter.py` |
| Phase 10 | `service_*`, `qwen_shadow.py` |
| Type 3 experiments | `table_evidence.py`, `type3_v7*`, `type3_v8.py`, `type3_qwen36_organizer_v8.py`, `type3_qwen36_faceted_v9.py`, `type3_qwen36_coverage_v10.py` |
| Type 3 dual-channel contracts | `type3_corpus_profile.py`; all downstream A2RAG / TabGR / fusion modules take an explicit corpus profile |

The hybrid retrieval provider that serves live QA is
`hybrid_evidence_provider.py`, which fuses the A2RAG text channel with the
TabGR table channel. See [ARCHITECTURE.md](../../docs/ARCHITECTURE.md).

## Configuration

- `config/composition_patterns.json` — the nine static topology patterns.
- `config/company_aliases.json` — audited historical and legal company-name aliases.
- `config/metric_aliases.json` — canonical metric dictionary.
- `config/unit_rules.json` — unit normalization rules.

Editing configuration can change the semantic hash, the planning oracle and the
answer trace. Fixtures, manifests, gates and version numbers must be updated
together. In particular, do not quietly add a tenth composition pattern at
runtime.
