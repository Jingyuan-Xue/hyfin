#!/usr/bin/env python3
"""Freeze Qwen eligibility from official, pre-Qwen evidence observations."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from finglmqa.contracts import canonical_json_bytes  # noqa: E402
from finglmqa.service_contracts import SHADOW_ELIGIBILITY_SCHEMA  # noqa: E402


RUN = ROOT / "runs/phase_10"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        for row in rows:
            handle.write(canonical_json_bytes(row))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> int:
    evaluation_path = RUN / "http_evaluation.jsonl"
    if not evaluation_path.is_file():
        raise RuntimeError("Gate 7 HTTP evaluation must finish before eligibility is frozen")
    oracle = read_jsonl(ROOT / "runs/phase_08/benchmark_decomposition_oracle.jsonl")
    narrative = [row for row in oracle if row["source"]["benchmark_type"] == "3-1"]
    if len(narrative) != 260:
        raise RuntimeError("frozen narrative universe is not 260 rows")
    evaluated = {row["case_id"]: row for row in read_jsonl(evaluation_path)}

    eligibility: list[dict[str, Any]] = []
    for gold in narrative:
        observed = evaluated.get(gold["case_id"])
        if observed is None:
            raise RuntimeError(f"HTTP result missing for {gold['case_id']}")
        response = observed["response"]
        answerable = bool(response["answer"].strip() and response["citations"] and response["status"] in {"ok", "partial"})
        category = "answerable" if answerable else "unanswerable_absent_evidence"
        basis = [
            "phase8_production_resolver_and_planner",
            "cpu_a2rag_official_result",
            "deterministic_evidence_executor_gate",
            f"official_trace:{response['demo_trace']['trace_hash']}",
        ]
        eligibility.append({
            "schema_version": SHADOW_ELIGIBILITY_SCHEMA,
            "case_id": gold["case_id"],
            "question_sha256": gold["source"]["question_sha256"],
            "eligibility": category,
            "basis": basis,
            "review_state": "mechanical",
        })

    company_rows = read_jsonl(ROOT / "data/corpus_package/company_year_index.jsonl")
    companies = {row["document_id"]: row for row in company_rows if row.get("status") == "unique"}
    audit_samples = read_jsonl(ROOT / "runs/phase_07/reports/table_exclusion_samples.jsonl")
    fixtures: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for sample in audit_samples:
        document_id = sample["document_id"]
        identity = companies.get(document_id)
        heading = sample["heading_path"][-1] if sample.get("heading_path") else "相关事项"
        key = (document_id, heading)
        if identity is None or key in seen:
            continue
        seen.add(key)
        ordinal = len(fixtures) + 1
        case_id = f"shadow-excluded-table-{ordinal:02d}"
        question = (
            f"根据{identity['company_full']}{identity['report_year']}年年度报告，"
            f"请概括“{heading}”表格所披露的信息。"
        )
        fixture = {
            "schema_version": "finglmqa.phase10.shadow_adversarial_fixture.v1",
            "case_id": case_id,
            "question": question,
            "question_sha256": hashlib.sha256(question.encode("utf-8")).hexdigest(),
            "document_id": document_id,
            "stock_code": identity["stock_code"],
            "company": identity["stock_name"],
            "report_year": int(identity["report_year"]),
            "audit_content_sha256": sample["content_sha256"],
            "audit_classification": sample["classification"],
        }
        fixtures.append(fixture)
        eligibility.append({
            "schema_version": SHADOW_ELIGIBILITY_SCHEMA,
            "case_id": case_id,
            "question_sha256": fixture["question_sha256"],
            "eligibility": "unanswerable_excluded_table_evidence",
            "basis": [
                "phase7_table_exclusion_audit",
                "table_evidence_forbidden_from_a2rag",
                f"audit_content_sha256:{sample['content_sha256']}",
            ],
            "review_state": "mechanical",
        })
        if len(fixtures) == 12:
            break
    if len(fixtures) != 12:
        raise RuntimeError("could not freeze twelve excluded-table adversarial fixtures")

    # Ordering is part of the oracle: 260 benchmark rows, then 12 adversarial rows.
    write_jsonl(RUN / "shadow_adversarial_fixtures.jsonl", fixtures)
    write_jsonl(RUN / "shadow_eligibility_oracle.jsonl", eligibility)
    print(json.dumps({
        "rows": len(eligibility),
        "answerable": sum(row["eligibility"] == "answerable" for row in eligibility),
        "unanswerable_absent_evidence": sum(row["eligibility"] == "unanswerable_absent_evidence" for row in eligibility),
        "unanswerable_excluded_table_evidence": 12,
        "eligibility_sha256": hashlib.sha256((RUN / "shadow_eligibility_oracle.jsonl").read_bytes()).hexdigest(),
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
