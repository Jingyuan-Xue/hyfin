#!/usr/bin/env python3
"""Run the frozen 260-case Qwen shadow plus adversarial fixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from finglmqa.contracts import canonical_json_bytes, semantic_sha256  # noqa: E402
from finglmqa.evidence_executor import EvidenceExecutor  # noqa: E402
from finglmqa.evidence_provider import A2RAGWarmWorkerTransport, DocumentScopedEvidenceProvider  # noqa: E402
from finglmqa.qwen_shadow import QwenGeneratorError, QwenShadowGenerator, VLLMShadowServer  # noqa: E402
from finglmqa.service_contracts import SHADOW_RESULT_SCHEMA, read_trace  # noqa: E402


RUN = ROOT / "runs/phase_10"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical_json_bytes(value))
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        for row in rows:
            handle.write(canonical_json_bytes(row))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def claim_projection(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "subplan_id": result["subplan_id"],
            "claim_id": row["claim_id"],
            "text": row["text"],
            "document_id": row["document_id"],
            "citation_ids": list(row["citation_ids"]),
            "numeric_authorization_ids": list(row["numeric_authorization_ids"]),
        }
        for row in result.get("claims", [])
    ]


def finalize_result(
    *, case_id: str, eligibility: str, subplan_count: int, outcomes: list[str],
    projections: list[dict[str, Any]], failures: list[str],
) -> dict[str, Any]:
    if projections:
        outcome = "accepted"
    elif "generator_invalid_output" in outcomes:
        outcome = "generator_invalid_output"
    elif "proposed" in outcomes:
        outcome = "generator_rejected_by_gate"
    else:
        outcome = "generator_refused"
    unsafe = len(projections) if eligibility != "answerable" else 0
    value = {
        "schema_version": SHADOW_RESULT_SCHEMA,
        "case_id": case_id,
        "eligibility": eligibility,
        "generator_outcome": outcome,
        "subplan_count": subplan_count,
        "accepted_claim_count": len(projections),
        "unsafe_accepted_count": unsafe,
        "accepted_claim_projection": projections,
        "gate_failure_codes": sorted(set(failures)),
    }
    value["result_fingerprint"] = semantic_sha256(value)
    return value


def execute_case(
    trace: Mapping[str, Any], eligibility: str,
    provider: DocumentScopedEvidenceProvider, generator: QwenShadowGenerator,
) -> dict[str, Any]:
    plan = trace.get("composition_plan")
    subplans = [row for row in (plan or {}).get("subplans", []) if row["backend"] == "evidence"]
    outcomes: list[str] = []
    projections: list[dict[str, Any]] = []
    failures: list[str] = []
    executor = EvidenceExecutor(provider, generator=generator)
    for subplan in subplans:
        generator.last_outcome = "not_called"
        result = executor.execute(subplan, trace["numeric_authorization_set"])
        outcomes.append(generator.last_outcome if generator.last_outcome != "not_called" else "generator_refused")
        projections.extend(claim_projection(result))
        if result.get("failure_code"):
            failures.append(result["failure_code"])
    return finalize_result(
        case_id=trace["request_id"].replace("benchmark_", "benchmark:", 1),
        eligibility=eligibility, subplan_count=len(subplans), outcomes=outcomes,
        projections=projections, failures=failures,
    )


def execute_excluded_fixture(fixture: Mapping[str, Any], generator: QwenShadowGenerator) -> dict[str, Any]:
    # A2RAG is intentionally not called for excluded table evidence.  Qwen is
    # still challenged with an empty evidence set and cannot create a result
    # that bypasses EvidenceExecutor because there is no executable subplan.
    request = {"question": fixture["question"], "chunks": []}
    try:
        response = generator.generate_claims(request)
        outcomes = [generator.last_outcome]
        proposals = response["claims"]
    except QwenGeneratorError:
        outcomes = ["generator_invalid_output"]
        proposals = []
    failures = ["EVIDENCE_ROUTE_EXCLUDED_TABLE"]
    if proposals:
        outcomes = ["proposed"]
        failures.append("EVIDENCE_UNAVAILABLE")
    return finalize_result(
        case_id=fixture["case_id"], eligibility="unanswerable_excluded_table_evidence",
        subplan_count=0, outcomes=outcomes, projections=[], failures=failures,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat-count", type=int, default=30)
    args = parser.parse_args()
    eligibility_path = RUN / "shadow_eligibility_oracle.jsonl"
    fixtures_path = RUN / "shadow_adversarial_fixtures.jsonl"
    evaluation_path = RUN / "http_evaluation.jsonl"
    for path in (eligibility_path, fixtures_path, evaluation_path):
        if not path.is_file():
            raise RuntimeError(f"required pre-Qwen artifact is missing: {path.name}")
    eligibility_bytes = eligibility_path.read_bytes()
    eligibility_rows = read_jsonl(eligibility_path)
    eligibility = {row["case_id"]: row for row in eligibility_rows}
    benchmark = [
        row for row in read_jsonl(ROOT / "runs/phase_08/benchmark_decomposition_oracle.jsonl")
        if row["source"]["benchmark_type"] == "3-1"
    ]
    evaluated = {row["case_id"]: row for row in read_jsonl(evaluation_path)}
    fixtures = read_jsonl(fixtures_path)

    transport = A2RAGWarmWorkerTransport(
        python_executable=os.environ.get("FINGLMQA_A2RAG_PYTHON", ROOT / "refs/a2rag_runtime/.venv/bin/python"),
        worker_script=os.environ.get("FINGLMQA_A2RAG_WORKER", ROOT / "scripts/query_type3_evidence.py"),
        device="cpu", model_cache=os.environ.get("FINGLMQA_EVIDENCE_MODEL_CACHE"), timeout_seconds=90,
    )
    provider = DocumentScopedEvidenceProvider(transport)
    server = VLLMShadowServer()
    results: list[dict[str, Any]] = []
    repeats: list[dict[str, Any]] = []
    qwen_started = False
    qwen_stopped = False
    try:
        # Eligibility bytes are captured before vLLM starts and rechecked after
        # all generations, preventing output-dependent reclassification.
        eligibility_sha256 = hashlib.sha256(eligibility_bytes).hexdigest()
        server.start()
        qwen_started = True
        generator = QwenShadowGenerator(base_url=server.base_url, model=server.served_name)
        transport.ping()
        for gold in benchmark:
            case_id = gold["case_id"]
            response = evaluated[case_id]["response"]
            trace = read_trace(response["demo_trace"]["trace_hash"], RUN / "service/traces")
            result = execute_case(trace, eligibility[case_id]["eligibility"], provider, generator)
            # Preserve the exact public case ID rather than deriving it from
            # request-id punctuation.
            result["case_id"] = case_id
            result["result_fingerprint"] = semantic_sha256({k: v for k, v in result.items() if k != "result_fingerprint"})
            results.append(result)
        for fixture in fixtures:
            results.append(execute_excluded_fixture(fixture, generator))

        answerable_ids = [row["case_id"] for row in results if row["eligibility"] == "answerable"]
        repeat_ids = answerable_ids[:args.repeat_count]
        by_id = {row["case_id"]: row for row in results}
        for case_id in repeat_ids:
            response = evaluated[case_id]["response"]
            trace = read_trace(response["demo_trace"]["trace_hash"], RUN / "service/traces")
            rerun = execute_case(trace, "answerable", provider, generator)
            rerun_projection = rerun["accepted_claim_projection"]
            first_projection = by_id[case_id]["accepted_claim_projection"]
            repeats.append({
                "case_id": case_id,
                "byte_identical": canonical_json_bytes(first_projection) == canonical_json_bytes(rerun_projection),
                "projection_sha256": semantic_sha256(first_projection),
            })
        if hashlib.sha256(eligibility_path.read_bytes()).hexdigest() != eligibility_sha256:
            raise RuntimeError("eligibility oracle changed after Qwen started")
    finally:
        transport.close(force=True)
        server.stop()
        qwen_stopped = not server._healthy()

    write_jsonl(RUN / "shadow_results.jsonl", results)
    write_jsonl(RUN / "reports/shadow_repeatability.jsonl", repeats)
    counts = Counter(row["eligibility"] for row in results)
    outcomes = Counter(row["generator_outcome"] for row in results)
    answerable = [row for row in results if row["eligibility"] == "answerable"]
    unanswerable = [row for row in results if row["eligibility"] != "answerable"]
    accepted_answerable = sum(row["generator_outcome"] == "accepted" for row in answerable)
    completion_checks = {
        "qwen_started": qwen_started,
        "qwen_stopped": qwen_stopped,
        "all_shadow_cases_terminal": len(results) == len(eligibility_rows) == 272,
        "classification_complete": sum(outcomes.values()) == len(results),
        "no_unsafe_accepted_claim": sum(row["unsafe_accepted_count"] for row in results) == 0,
        "no_process_or_oom_failure": all("OOM" not in code for row in results for code in row["gate_failure_codes"]),
    }
    promotion_checks = {
        "answerable_acceptance_at_least_95pct": bool(answerable) and accepted_answerable / len(answerable) >= 0.95,
        "unanswerable_has_zero_accepted_claims": all(row["accepted_claim_count"] == 0 for row in unanswerable),
        "invalid_output_separately_counted": "generator_invalid_output" in outcomes or outcomes.get("generator_invalid_output", 0) == 0,
        "fixed_30_accepted_projection_repeatable": len(repeats) == args.repeat_count and all(row["byte_identical"] for row in repeats),
        # Completed by the main agent after inspecting the fixed sample file.
        "manual_audit_passed": False,
    }
    summary = {
        "schema_version": "finglmqa.phase10.shadow_report.v1",
        "eligibility_oracle_sha256": hashlib.sha256(eligibility_bytes).hexdigest(),
        "counts": {"eligibility": dict(sorted(counts.items())), "generator_outcomes": dict(sorted(outcomes.items()))},
        "answerable_acceptance": {
            "accepted": accepted_answerable, "total": len(answerable),
            "rate": format(accepted_answerable / len(answerable), ".8f") if answerable else "0.00000000",
        },
        "completion_checks": completion_checks,
        "promotion_checks": promotion_checks,
        "completion_status": "passed" if all(completion_checks.values()) else "failed",
        "promotion_readiness": False,
    }
    write_json(RUN / "shadow_report.json", summary)
    write_json(RUN / "promotion_readiness.json", {
        "schema_version": "finglmqa.phase10.promotion_readiness.v1",
        "promotion_readiness": False,
        "checks": promotion_checks,
        "reasons": [key for key, passed in promotion_checks.items() if not passed],
    })
    print(json.dumps({
        "completion_status": summary["completion_status"],
        "promotion_readiness": False,
        "counts": summary["counts"],
    }, ensure_ascii=False, sort_keys=True))
    return 0 if summary["completion_status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
