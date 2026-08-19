#!/usr/bin/env python3
"""Run the frozen Type 3 hybrid-recall + Qwen coverage-planner v10 experiment."""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from finglmqa.contracts import semantic_sha256  # noqa: E402
from finglmqa.type3_qwen36_coverage_v10 import (  # noqa: E402
    MAX_CANDIDATES,
    MAX_DENSE_CHUNKS_PER_FACET,
    MAX_FACETS,
    PROFILE_VERSION,
    PROMPT_CONTRACT_HASH,
    PROMPT_VERSION,
    RESULT_SCHEMA,
    SELECTOR_SEEDS,
    Type3Qwen36CoverageV10,
)
from finglmqa.type3_qwen36_faceted_v9 import compact_text  # noqa: E402
from query_type3_evidence import (  # noqa: E402
    Type3EvidenceRetriever,
    heading_rerank_adjustment,
)
from run_type3_qwen36_faceted_v9 import (  # noqa: E402
    OpenAICompatibleClient,
    VLLMV9Server,
    load_cases,
    projection,
    run_repeats,
    safety_validation,
    sha256_file,
    vllm_version,
    write_json,
    write_jsonl,
)


DEFAULT_V8_DIR = ROOT / "runs/type3_no_llm_experiment_v8"
DEFAULT_OUTPUT_ROOT = ROOT / "runs/type3_qwen36_hybrid_coverage_v10"
DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUT_ROOT / "full"
DEFAULT_MODEL_PATH = ROOT / "refs/qwen_model"
DEFAULT_VLLM_BIN = Path(
    "/home/coder/demo/exposure_pipeline_workspace/.venv-vllm-auto/bin/vllm"
)
DEFAULT_MODEL_NAME = "finglmqa-qwen3.6-27b-v10"


class ExpandedDenseRetriever:
    """Read-only top-15 BGE-M3 adapter over the frozen Phase 7 vectors."""

    def __init__(self, base: Type3EvidenceRetriever) -> None:
        self.base = base
        self._cache: dict[tuple[str, str, int], dict[str, Any]] = {}

    def retrieve_for_document(
        self, document_id: str, question: str, top_k: int = 15
    ) -> dict[str, Any]:
        if document_id not in self.base.document_map:
            raise KeyError(f"unknown document_id: {document_id}")
        count = min(max(int(top_k), 1), 20)
        key = (document_id, question, count)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        self.base._load_dense_vectors()
        document = self.base.document_map[document_id]
        candidate_ids = list(document["chunk_ids"])
        missing = [value for value in candidate_ids if value not in self.base._dense_by_id]
        if missing:
            raise RuntimeError(f"evidence IDs lack dense vectors: {missing[:5]}")
        matrix = np.vstack([
            self.base._dense_by_id[chunk_id] for chunk_id in candidate_ids
        ]).astype(np.float32)
        query_vector = np.asarray(
            self.base._load_embedding_model().batch_encode([question])[0],
            dtype=np.float32,
        )
        query_norm = float(np.linalg.norm(query_vector)) or 1.0
        candidate_norms = np.linalg.norm(matrix, axis=1)
        candidate_norms[candidate_norms == 0] = 1.0
        dense_scores = (matrix @ query_vector) / (candidate_norms * query_norm)
        dense_order = sorted(
            range(len(candidate_ids)),
            key=lambda index: (-float(dense_scores[index]), index + 1, candidate_ids[index]),
        )
        pool_size = min(len(candidate_ids), max(80, count * 10))
        reranked: list[tuple[float, int, str, Mapping[str, Any]]] = []
        for index in dense_order[:pool_size]:
            chunk_id = candidate_ids[index]
            evidence = self.base.evidence_by_id[chunk_id]
            adjusted = float(dense_scores[index]) + heading_rerank_adjustment(
                question, evidence
            )
            reranked.append((adjusted, index + 1, chunk_id, evidence))
        reranked.sort(key=lambda row: (-row[0], row[1], row[2]))
        chunks = []
        for rank, (score, _ordinal, chunk_id, evidence) in enumerate(reranked[:count], 1):
            chunks.append({
                "rank": rank,
                "score": format(score, ".8f"),
                "evidence_chunk_id": chunk_id,
                "document_id": evidence["document_id"],
                "company_name": evidence["company_name"],
                "stock_code": evidence["stock_code"],
                "report_year": evidence["report_year"],
                "section_path": evidence["section_path"],
                "semantic_tags": evidence["semantic_tags"],
                "line_range": evidence["line_range"],
                "source_markdown": evidence["source_markdown"],
                "content": evidence["content"],
            })
        result = {
            "retrieval_method": "bge_m3_dense_top15_plus_heading_rerank",
            "candidate_prefilter": "resolved_document_allow_list",
            "prefilter_applied_before_scoring": True,
            "candidate_document_id": document_id,
            "candidate_chunk_count": len(candidate_ids),
            "dense_candidate_pool_size": pool_size,
            "top_k": count,
            "chunks": chunks,
        }
        self._cache[key] = result
        return result


def _task_cache_key(case: Mapping[str, Any]) -> str:
    return semantic_sha256({
        "question": case["question"],
        "document_id": case["document_id"],
        "baseline_answer": case["baseline_answer"],
        "baseline_citations": case["baseline_citations"],
    })


def _refresh_fingerprint(value: dict[str, Any]) -> None:
    value["result_fingerprint"] = semantic_sha256({
        "schema_version": value["schema_version"],
        "profile_version": value["profile_version"],
        "case_id": value["case_id"],
        "question": value["question"],
        "document_id": value["document_id"],
        "answer": value["answer"],
        "citations": value["citations"],
        "status": value["status"],
        "facets": value["facets"],
        "selected_fragment_ids": value["selected_fragment_ids"],
        "gate_report": value["gate_report"],
    })


def run_cases(
    cases: list[dict[str, Any]], *, engine: Type3Qwen36CoverageV10
) -> tuple[list[dict[str, Any]], int]:
    cache: dict[str, dict[str, Any]] = {}
    results: list[dict[str, Any]] = []
    for ordinal, case in enumerate(cases, 1):
        key = _task_cache_key(case)
        if key in cache:
            result = copy.deepcopy(cache[key])
            result["case_id"] = str(case["case_id"])
            _refresh_fingerprint(result)
            cache_status = "cache"
        else:
            result = engine.answer(**case)
            cache[key] = copy.deepcopy(result)
            cache_status = "inference"
        results.append(result)
        print(
            f"[{ordinal}/{len(cases)}] {case['case_id']} {cache_status} "
            f"{result['planner_outcome']} {result['selector_outcome']} "
            f"candidates={result['candidate_count']} "
            f"selected={len(result['selected_fragment_ids'])}",
            file=sys.stderr,
            flush=True,
        )
    return results, len(cache)


UNANIMOUS_PROFILE = PROFILE_VERSION + "-unanimous-3of3"


def unanimous_project(result: Mapping[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(dict(result))
    runs = value.get("selector_runs")
    ordered: list[list[str]] = []
    if (
        isinstance(runs, list)
        and len(runs) == 3
        and all(isinstance(run, Mapping) and run.get("status") == "ok" for run in runs)
    ):
        for run in runs:
            ordered.append([
                str(row["fragment_id"])
                for row in run.get("selections") or []
                if isinstance(row, Mapping) and isinstance(row.get("fragment_id"), str)
            ])
    common = set(ordered[0]).intersection(*map(set, ordered[1:])) if len(ordered) == 3 else set()
    projection_by_id = {
        str(row["fragment_id"]): row
        for row in value.get("selected_fragment_projection") or []
        if isinstance(row, Mapping) and isinstance(row.get("fragment_id"), str)
    }
    selected_ids = [
        fragment_id for fragment_id in (ordered[0] if ordered else [])
        if fragment_id in common and fragment_id in projection_by_id
    ]
    baseline = str(value.get("baseline_answer") or "").strip()
    if baseline:
        baseline_key = compact_text(baseline)
        selected_ids = [
            fragment_id for fragment_id in selected_ids
            if compact_text(str(projection_by_id[fragment_id].get("text") or "")) not in baseline_key
        ]
    selected_projection = [projection_by_id[fragment_id] for fragment_id in selected_ids]
    parts = [str(row["text"]).strip() for row in selected_projection if str(row["text"]).strip()]
    if baseline:
        parts.append(baseline)
    answer = "\n".join(parts)
    citations = []
    for citation in value.get("citations") or []:
        if not isinstance(citation, Mapping):
            continue
        candidate_id = citation.get("candidate_id")
        if isinstance(candidate_id, str) and candidate_id.startswith("v9frag_"):
            if candidate_id not in selected_ids:
                continue
        citations.append(dict(citation))

    covered_facets: set[str] = set()
    if len(ordered) == 3:
        for run in runs:
            for selection in run.get("selections") or []:
                if selection.get("fragment_id") in selected_ids:
                    covered_facets.update(str(item) for item in selection.get("facet_ids") or [])
    core_ids = set(value["coverage_report"]["core_facet_ids"])
    value.update({
        "profile_version": UNANIMOUS_PROFILE,
        "answer": answer,
        "citations": citations,
        "status": "ok" if answer else "not_found",
        "selector_outcome": "unanimous_selected" if selected_ids else "unanimous_empty",
        "selected_fragment_ids": selected_ids,
        "selected_fragment_projection": selected_projection,
        "coverage_report": {
            **value["coverage_report"],
            "selected_core_facet_ids": sorted(core_ids.intersection(covered_facets)),
            "selected_facet_ids": sorted(covered_facets),
            "core_selection_coverage_ratio": round(
                len(core_ids.intersection(covered_facets)) / max(1, len(core_ids)), 8
            ),
            "consensus_policy": "three_valid_runs_and_three_selection_votes",
        },
        "unanimous_policy": {
            "required_valid_runs": 3,
            "required_selection_votes": 3,
            "baseline_retained_as_exact_suffix": True,
        },
    })
    _refresh_fingerprint(value)
    return value


def freeze_manifest(
    *,
    source_hashes: Mapping[str, str],
    model_path: Path,
    model_name: str,
    vllm_binary: Path,
) -> dict[str, Any]:
    generation = {
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "planner_seed": 0,
        "selector_seeds": list(SELECTOR_SEEDS),
        "max_tokens": 1024,
        "enable_thinking": True,
        "structured_outputs": True,
        "max_facets": MAX_FACETS,
        "max_candidates": MAX_CANDIDATES,
        "dense_chunks_per_facet": MAX_DENSE_CHUNKS_PER_FACET,
        "retrieval_channels": ["bge_m3_dense", "document_bm25", "source_neighbour"],
    }
    config_path = model_path.resolve() / "config.json"
    manifest = {
        "schema_version": "finglmqa.experimental.type3_qwen36_coverage_v10.freeze.v1",
        "frozen_before_model_invocation": True,
        "frozen_before_scoring": True,
        "profile_version": PROFILE_VERSION,
        "prompt_version": PROMPT_VERSION,
        "prompt_contract_sha256": PROMPT_CONTRACT_HASH,
        "generation_config": generation,
        "generation_config_sha256": semantic_sha256(generation),
        "source_hashes": dict(sorted(source_hashes.items())),
        "index_hashes": {
            "evidence_chunks_sha256": sha256_file(
                ROOT / "data/corpus_package/evidence_chunks.jsonl"
            ),
            "document_map_sha256": sha256_file(
                ROOT / "data/indexes/a2rag_index/document_chunk_map.jsonl"
            ),
            "dense_index_manifest_sha256": sha256_file(
                ROOT / "data/indexes/a2rag_index/index_manifest.json"
            ),
        },
        "code_hashes": {
            "coverage_v10_sha256": sha256_file(
                ROOT / "src/finglmqa/type3_qwen36_coverage_v10.py"
            ),
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "base_retriever_sha256": sha256_file(ROOT / "scripts/query_type3_evidence.py"),
        },
        "model_identity": {
            "served_name": model_name,
            "requested_path": model_path.relative_to(ROOT).as_posix(),
            "snapshot_path": model_path.resolve().as_posix(),
            "snapshot_revision": model_path.resolve().name,
            "config_sha256": sha256_file(config_path),
            "vllm_version": vllm_version(vllm_binary),
            "dtype": "bfloat16",
            "max_model_len": 24576,
        },
        "answer_chain_consumed_fields": [
            "case_id (join identity only)", "question", "resolved document_id",
            "v8 answer", "v8 citations", "document-scoped evidence",
        ],
        "forbidden_benchmark_fields_consumed": [],
        "case_company_year_rules": False,
        "manifest_fingerprint": "",
    }
    manifest["manifest_fingerprint"] = semantic_sha256({
        key: value for key, value in manifest.items() if key != "manifest_fingerprint"
    })
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v8-dir", type=Path, default=DEFAULT_V8_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--vllm-bin", type=Path, default=DEFAULT_VLLM_BIN)
    parser.add_argument("--base-url")
    parser.add_argument("--port", type=int, default=8013)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--repeat-count", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--embedding-cache",
        type=Path,
        default=Path(os.environ.get("FINGLMQA_EMBEDDING_CACHE", "/home/coder/demo/models")),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if not output_dir.is_relative_to(DEFAULT_OUTPUT_ROOT.resolve()):
        raise RuntimeError("output-dir must remain under v10 experiment root")
    if args.limit is not None and not 1 <= args.limit <= 260:
        raise RuntimeError("limit must be between 1 and 260")
    if not 0 <= args.repeat_count <= 260:
        raise RuntimeError("repeat-count must be between 0 and 260")

    cases, source_hashes = load_cases(args.v8_dir.resolve())
    if args.limit is not None:
        cases = cases[:args.limit]
    frozen = freeze_manifest(
        source_hashes=source_hashes,
        model_path=args.model_path,
        model_name=args.model,
        vllm_binary=args.vllm_bin,
    )
    write_json(output_dir / "freeze_manifest.json", frozen)

    os.environ["FINGLMQA_EMBEDDING_CACHE"] = args.embedding_cache.resolve().as_posix()
    base_retriever = Type3EvidenceRetriever(
        root=ROOT,
        device=args.device,
        model_cache=args.embedding_cache,
        load_dense=True,
    )
    retriever = ExpandedDenseRetriever(base_retriever)
    server: VLLMV9Server | None = None
    try:
        if args.base_url:
            base_url = args.base_url
            served_name = args.model
        else:
            server = VLLMV9Server(
                binary=args.vllm_bin,
                model_path=args.model_path,
                served_name=args.model,
                port=args.port,
            )
            server.start()
            base_url = server.base_url
            served_name = server.served_name
        engine = Type3Qwen36CoverageV10(
            OpenAICompatibleClient(base_url),
            retriever,
            root=ROOT,
            model=served_name,
        )
        results, unique_tasks = run_cases(cases, engine=engine)
        repeats = run_repeats(
            cases,
            engine=engine,
            repeat_count=min(args.repeat_count, len(cases)),
        )
    finally:
        if server is not None:
            server.stop()

    validation = safety_validation(cases, results, repeats)
    validation["schema_version"] = "finglmqa.experimental.type3_qwen36_coverage_v10.safety.v1"
    unanimous_results = [unanimous_project(row) for row in results]
    unanimous_repeats = [unanimous_project(row) for row in repeats]
    unanimous_validation = safety_validation(cases, unanimous_results, unanimous_repeats)
    unanimous_validation["schema_version"] = (
        "finglmqa.experimental.type3_qwen36_coverage_v10.unanimous_safety.v1"
    )
    write_jsonl(output_dir / "results.jsonl", results)
    write_jsonl(output_dir / "repeat_results.jsonl", repeats)
    write_jsonl(
        output_dir / "http_evaluation.jsonl",
        (projection(row, answer_field="answer", profile=PROFILE_VERSION) for row in results),
    )
    deterministic_profile = PROFILE_VERSION + "-qwen-plan-deterministic-selector"
    write_jsonl(
        output_dir / "ablations/qwen_plan_deterministic/http_evaluation.jsonl",
        (
            projection(
                row,
                answer_field="qwen_plan_deterministic_selector_answer",
                profile=deterministic_profile,
            )
            for row in results
        ),
    )
    no_qwen_profile = PROFILE_VERSION + "-no-qwen-same-index"
    write_jsonl(
        output_dir / "ablations/no_qwen_same_index/http_evaluation.jsonl",
        (
            projection(
                row,
                answer_field="no_qwen_same_index_answer",
                profile=no_qwen_profile,
            )
            for row in results
        ),
    )
    write_json(output_dir / "safety_validation.json", validation)
    unanimous_dir = DEFAULT_OUTPUT_ROOT / (
        "unanimous" if output_dir == DEFAULT_OUTPUT_DIR.resolve()
        else f"{output_dir.name}_unanimous"
    )
    write_jsonl(unanimous_dir / "results.jsonl", unanimous_results)
    write_jsonl(unanimous_dir / "repeat_results.jsonl", unanimous_repeats)
    write_jsonl(
        unanimous_dir / "http_evaluation.jsonl",
        (
            projection(row, answer_field="answer", profile=UNANIMOUS_PROFILE)
            for row in unanimous_results
        ),
    )
    write_json(unanimous_dir / "safety_validation.json", unanimous_validation)
    unanimous_report = {
        "schema_version": "finglmqa.experimental.type3_qwen36_coverage_v10.unanimous_report.v1",
        "profile_version": UNANIMOUS_PROFILE,
        "rows": len(unanimous_results),
        "nonempty_answers": sum(bool(row["answer"].strip()) for row in unanimous_results),
        "selected_row_count": sum(bool(row["selected_fragment_ids"]) for row in unanimous_results),
        "selected_fragment_count_distribution": dict(sorted(Counter(
            len(row["selected_fragment_ids"]) for row in unanimous_results
        ).items())),
        "average_core_selection_coverage_ratio": round(sum(
            float(row["coverage_report"]["core_selection_coverage_ratio"])
            for row in unanimous_results
        ) / max(1, len(unanimous_results)), 8),
        "repeat_final_projection_exact": unanimous_validation["repeat_final_projection_exact"],
        "safety_validation_passed": unanimous_validation["passed"],
        "source_freeze_manifest_fingerprint": frozen["manifest_fingerprint"],
        "artifacts": {
            "results_sha256": sha256_file(unanimous_dir / "results.jsonl"),
            "repeat_results_sha256": sha256_file(unanimous_dir / "repeat_results.jsonl"),
            "http_evaluation_sha256": sha256_file(unanimous_dir / "http_evaluation.jsonl"),
            "safety_validation_sha256": sha256_file(unanimous_dir / "safety_validation.json"),
        },
        "benchmark_scoring_used_for_projection": False,
    }
    write_json(unanimous_dir / "run_report.json", unanimous_report)
    report = {
        "schema_version": "finglmqa.experimental.type3_qwen36_coverage_v10.run_report.v1",
        "profile_version": PROFILE_VERSION,
        "result_schema_version": RESULT_SCHEMA,
        "input_rows": len(cases),
        "unique_inference_tasks": unique_tasks,
        "evaluation_duplicate_cache_hits": len(cases) - unique_tasks,
        "terminal_rows": len(results),
        "nonempty_answers": sum(bool(row["answer"].strip()) for row in results),
        "planner_outcome_counts": dict(sorted(Counter(
            row["planner_outcome"] for row in results
        ).items())),
        "selector_outcome_counts": dict(sorted(Counter(
            row["selector_outcome"] for row in results
        ).items())),
        "selected_fragment_count_distribution": dict(sorted(Counter(
            len(row["selected_fragment_ids"]) for row in results
        ).items())),
        "candidate_count_distribution": dict(sorted(Counter(
            int(row["candidate_count"]) for row in results
        ).items())),
        "average_core_selection_coverage_ratio": round(sum(
            float(row["coverage_report"]["core_selection_coverage_ratio"])
            for row in results
        ) / max(1, len(results)), 8),
        "safety_validation_passed": validation["passed"],
        "repeat_final_projection_exact": validation["repeat_final_projection_exact"],
        "repeat_count": len(repeats),
        "unanimous_safety_validation_passed": unanimous_validation["passed"],
        "unanimous_repeat_final_projection_exact": unanimous_validation[
            "repeat_final_projection_exact"
        ],
        "freeze_manifest_fingerprint": frozen["manifest_fingerprint"],
        "artifacts": {
            "results_sha256": sha256_file(output_dir / "results.jsonl"),
            "repeat_results_sha256": sha256_file(output_dir / "repeat_results.jsonl"),
            "http_evaluation_sha256": sha256_file(output_dir / "http_evaluation.jsonl"),
            "deterministic_http_sha256": sha256_file(
                output_dir / "ablations/qwen_plan_deterministic/http_evaluation.jsonl"
            ),
            "no_qwen_same_index_http_sha256": sha256_file(
                output_dir / "ablations/no_qwen_same_index/http_evaluation.jsonl"
            ),
            "safety_validation_sha256": sha256_file(output_dir / "safety_validation.json"),
            "unanimous_http_evaluation_sha256": sha256_file(
                unanimous_dir / "http_evaluation.jsonl"
            ),
        },
        "benchmark_fields_loaded_by_answer_chain": ["case_id", "question"],
        "forbidden_benchmark_fields_loaded_by_answer_chain": [],
        "benchmark_scoring_used_for_prompt_or_rule_selection": False,
    }
    write_json(output_dir / "run_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if validation["passed"] and unanimous_validation["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
