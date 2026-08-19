#!/usr/bin/env python3
"""Run the 260 Type-3 questions through the experimental table/Qwen path.

The script does not import or modify the official service pipeline.  It reads
the frozen Phase 8 decomposition oracle only to obtain a unique document scope,
retrieves document-local A2RAG and experimental table fragments, and invokes
the fail-closed extractive organizer.  Raw prompts are never persisted.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from finglmqa.contracts import canonical_json_bytes, semantic_sha256  # noqa: E402
from finglmqa.evidence_provider import (  # noqa: E402
    A2RAGWarmWorkerTransport,
    DocumentScopedEvidenceProvider,
)
from finglmqa.qwen_answer_organizer import (  # noqa: E402
    OpenAICompatibleChatClient,
    QwenAnswerOrganizer,
    RESULT_SCHEMA,
)
from finglmqa.qwen_shadow import VLLMShadowServer  # noqa: E402
from finglmqa.table_evidence import TableEvidenceIndex  # noqa: E402


DEFAULT_ORACLE = ROOT / "runs/phase_08/benchmark_decomposition_oracle.jsonl"
DEFAULT_TABLE_INDEX = ROOT / "runs/table_evidence_experiment/table_evidence_fragments.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "runs/table_qwen_experiment"
DEFAULT_MODEL_CACHE = Path(
    os.environ.get("FINGLMQA_EVIDENCE_MODEL_CACHE", "/home/coder/demo/models")
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise RuntimeError(f"{path.name}:{line_number} is not an object")
            rows.append(value)
    return rows


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(canonical_json_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        for row in rows:
            handle.write(canonical_json_bytes(dict(row)))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def type3_cases(path: Path) -> list[dict[str, Any]]:
    rows = [
        row for row in read_jsonl(path)
        if row.get("source", {}).get("benchmark_type") == "3-1"
    ]
    rows.sort(key=lambda row: int(row["source"]["selected_ordinal"]))
    if len(rows) != 260:
        raise RuntimeError(f"expected 260 Type-3 oracle rows, found {len(rows)}")
    if len({row["case_id"] for row in rows}) != len(rows):
        raise RuntimeError("Type-3 oracle contains duplicate case IDs")
    return rows


def case_scope(row: Mapping[str, Any]) -> dict[str, Any]:
    projection = row["expected_planning_projection"]
    plan = projection["plan"]
    ready_evidence = [
        subplan for subplan in plan["subplans"]
        if subplan["backend"] == "evidence" and subplan["planning_state"] == "ready"
    ]
    if len(ready_evidence) != 1:
        raise RuntimeError("Type-3 experimental case must have exactly one ready evidence SubPlan")
    subplan = ready_evidence[0]
    documents = subplan["declared_scope"]["document_ids"]
    document_id = subplan["payload"]["document_id"]
    if documents != [document_id]:
        raise RuntimeError("Type-3 SubPlan did not freeze one matching document_id")
    entities = projection["scope"]["entities"]
    if len(entities) != 1 or entities[0]["status"] != "unique":
        raise RuntimeError("Type-3 scope did not uniquely resolve one company")
    entity = entities[0]
    docs = entity["documents"]
    if len(docs) != 1 or docs[0]["document_id"] != document_id:
        raise RuntimeError("Type-3 resolved entity/document scope is inconsistent")
    return {
        "case_id": row["case_id"],
        "question": row["source"]["question"],
        "document_id": document_id,
        "company": entity["identity"]["stock_name"],
        "stock_code": entity["identity"]["stock_code"],
        "report_year": docs[0]["report_year"],
    }


def a2rag_evidence(provider_result: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "evidence_id": chunk["chunk_id"],
            "source_kind": "a2rag_text",
            "document_id": chunk["document_id"],
            "company": chunk["company"],
            "stock_code": chunk["stock_code"],
            "report_year": chunk["report_year"],
            "content": chunk["content"],
            "source_markdown": chunk["source_markdown"],
            "line_range": list(chunk["line_range"]),
            # This experiment does not import structured numeric authority
            # from Phase 8.  Text containing digits therefore fails closed.
            "numeric_authorization": "not_authorized_for_answer",
        }
        for chunk in provider_result["chunks"]
    ]


def table_evidence(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for fragment in rows:
        authorization = fragment["provenance"]["numeric_authorization"]
        if authorization != "not_authorized_for_answer":
            raise RuntimeError("table fragment attempted to authorize answer numbers")
        result.append({
            "evidence_id": fragment["fragment_id"],
            "source_kind": fragment["fragment_kind"],
            "document_id": fragment["document_id"],
            "company": fragment["company_name"],
            "stock_code": fragment["stock_code"],
            "report_year": fragment["report_year"],
            "content": fragment["content"],
            "source_markdown": fragment["source_markdown"],
            "line_range": list(fragment["source_line_range"]),
            "numeric_authorization": authorization,
        })
    return result


def retrieve_case_evidence(
    scope: Mapping[str, Any],
    *,
    provider: DocumentScopedEvidenceProvider,
    table_index: TableEvidenceIndex,
    text_top_k: int,
    table_top_k: int,
) -> list[dict[str, Any]]:
    request = {
        "document_id": scope["document_id"],
        "question": scope["question"],
        "top_k": text_top_k,
    }
    provider_result = provider.retrieve(request)
    identity = (
        provider_result["document_id"], provider_result["company"],
        provider_result["stock_code"], provider_result["report_year"],
    )
    expected = (
        scope["document_id"], scope["company"], scope["stock_code"], scope["report_year"],
    )
    if identity != expected:
        raise RuntimeError("A2RAG provider identity differs from the frozen oracle scope")
    narratives = table_index.search(
        document_id=scope["document_id"],
        question=scope["question"],
        top_k=table_top_k,
        fragment_kinds=["mixed_narrative"],
    )
    table_rows = table_index.search(
        document_id=scope["document_id"],
        question=scope["question"],
        top_k=table_top_k,
        fragment_kinds=["table_row"],
    )
    return [
        *a2rag_evidence(provider_result),
        *table_evidence(narratives),
        *table_evidence(table_rows),
    ]


def retrieval_error_result(
    scope: Mapping[str, Any], organizer: QwenAnswerOrganizer,
) -> dict[str, Any]:
    result = {
        "schema_version": RESULT_SCHEMA,
        "case_id": scope["case_id"],
        "question": scope["question"],
        "scope": {
            key: scope[key] for key in ("document_id", "company", "stock_code", "report_year")
        },
        "status": "error",
        "generator_outcome": "retrieval_error",
        "answer": "",
        "accepted_claim_projection": [],
        "citation_projection": [],
        "rejections": [],
        "source_snapshot": [],
        "authorization_snapshot": [],
        "model_config": organizer.model_config,
        "result_fingerprint": "",
    }
    result["result_fingerprint"] = semantic_sha256({
        key: value for key, value in result.items() if key != "result_fingerprint"
    })
    return result


def http_projection(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "case_id": result["case_id"],
        "kind": "benchmark",
        "oracle_match": True,
        "request": {"question": result["question"]},
        "response": {
            "answer": result["answer"],
            "status": result["status"],
            "errors": (
                [{"error_code": result["generator_outcome"]}]
                if result["status"] == "error" else []
            ),
        },
    }


def execute(
    *,
    cases: Sequence[Mapping[str, Any]],
    organizer: QwenAnswerOrganizer,
    provider: DocumentScopedEvidenceProvider,
    table_index: TableEvidenceIndex,
    text_top_k: int,
    table_top_k: int,
    repeat_count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    results: list[dict[str, Any]] = []
    repeat_inputs: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    for ordinal, oracle in enumerate(cases):
        scope = case_scope(oracle)
        try:
            evidence = retrieve_case_evidence(
                scope, provider=provider, table_index=table_index,
                text_top_k=text_top_k, table_top_k=table_top_k,
            )
            result = organizer.organize(**scope, evidence=evidence)
            if ordinal < repeat_count:
                # A private in-memory snapshot permits an independent second
                # model call without persisting or reconstructing a prompt.
                repeat_inputs.append((scope, evidence))
        except Exception:
            # Runtime details are deliberately not copied into deterministic
            # result artifacts.  The terminal class is sufficient for audit.
            result = retrieval_error_result(scope, organizer)
        results.append(result)
        print(
            f"[{ordinal + 1}/{len(cases)}] {scope['case_id']} {result['generator_outcome']}",
            file=sys.stderr,
            flush=True,
        )

    repeats: list[dict[str, Any]] = []
    for ordinal, (scope, evidence) in enumerate(repeat_inputs):
        try:
            result = organizer.organize(**scope, evidence=evidence)
        except Exception:
            result = retrieval_error_result(scope, organizer)
        repeats.append(result)
        print(
            f"[repeat {ordinal + 1}/{len(repeat_inputs)}] {scope['case_id']} "
            f"{result['generator_outcome']}",
            file=sys.stderr,
            flush=True,
        )
    return results, repeats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", type=Path, default=DEFAULT_ORACLE)
    parser.add_argument("--table-index", type=Path, default=DEFAULT_TABLE_INDEX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--base-url", help="Use an already-running OpenAI-compatible server")
    parser.add_argument("--model", default="finglmqa-qwen3.6-27b")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "auto"))
    parser.add_argument("--model-cache", type=Path, default=DEFAULT_MODEL_CACHE)
    parser.add_argument("--text-top-k", type=int, default=5, choices=range(1, 6), metavar="1..5")
    parser.add_argument("--table-top-k", type=int, default=5, choices=range(1, 21), metavar="1..20")
    parser.add_argument("--repeat-count", type=int, default=30)
    parser.add_argument("--limit", type=int, help="Developer smoke-test prefix; omit for all 260")
    args = parser.parse_args()
    if args.repeat_count < 0 or args.repeat_count > 260:
        parser.error("--repeat-count must be between 0 and 260")
    if args.limit is not None and not 1 <= args.limit <= 260:
        parser.error("--limit must be between 1 and 260")

    cases = type3_cases(args.oracle)
    if args.limit is not None:
        cases = cases[: args.limit]
    repeat_count = min(args.repeat_count, len(cases))
    table_index = TableEvidenceIndex(args.table_index)
    transport = A2RAGWarmWorkerTransport(
        device=args.device,
        model_cache=args.model_cache,
        timeout_seconds=90,
    )
    provider = DocumentScopedEvidenceProvider(transport)

    server: VLLMShadowServer | None = None
    try:
        if args.base_url:
            base_url = args.base_url
            model = args.model
        else:
            server = VLLMShadowServer()
            server.start()
            base_url = server.base_url
            model = server.served_name
        organizer = QwenAnswerOrganizer(
            OpenAICompatibleChatClient(base_url),
            model=model,
        )
        transport.ping()
        results, repeats = execute(
            cases=cases,
            organizer=organizer,
            provider=provider,
            table_index=table_index,
            text_top_k=args.text_top_k,
            table_top_k=args.table_top_k,
            repeat_count=repeat_count,
        )
    finally:
        transport.close(force=True)
        if server is not None:
            server.stop()

    atomic_write_jsonl(args.output_dir / "results.jsonl", results)
    atomic_write_jsonl(args.output_dir / "results_repeat.jsonl", repeats)
    atomic_write_jsonl(
        args.output_dir / "http_evaluation.jsonl",
        [http_projection(row) for row in results],
    )
    outcomes = Counter(row["generator_outcome"] for row in results)
    statuses = Counter(row["status"] for row in results)
    summary = {
        "schema_version": "finglmqa.experimental.table_qwen_summary.v1",
        "result_schema_version": RESULT_SCHEMA,
        "expected_case_count": len(cases),
        "terminal_case_count": len(results),
        "repeat_case_count": len(repeats),
        "status_counts": dict(sorted(statuses.items())),
        "generator_outcome_counts": dict(sorted(outcomes.items())),
        "accepted_claim_count": sum(len(row["accepted_claim_projection"]) for row in results),
        "nonempty_answer_count": sum(bool(row["answer"]) for row in results),
        "all_cases_terminal": len(results) == len(cases),
        "all_cases_retrieved": outcomes.get("retrieval_error", 0) == 0,
        "model_config": organizer.model_config,
        "results_semantic_sha256": semantic_sha256(results),
        "repeat_results_semantic_sha256": semantic_sha256(repeats),
    }
    atomic_write_json(args.output_dir / "summary.json", summary)
    return 0 if summary["all_cases_terminal"] and summary["all_cases_retrieved"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
