#!/usr/bin/env python3
"""Validate Phase 8 repositories, SQL, and formula gates on real artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from finglmqa.analyzer import QuestionAnalyzer  # noqa: E402
from finglmqa.composition import TopologyCompositionPlanner  # noqa: E402
from finglmqa.contracts import canonical_json_bytes, semantic_sha256  # noqa: E402
from finglmqa.formula_engine import FormulaExecutor  # noqa: E402
from finglmqa.repositories import FactRepository, FallbackCandidateIndex  # noqa: E402
from finglmqa.resolver import ScopeResolver  # noqa: E402
from finglmqa.sql_engine import SQLExecutor  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(canonical_json_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def plan_question(question: str, request_id: str) -> dict[str, Any]:
    request = {
        "schema_version": "finglmqa.phase8.qa_request.v1",
        "request_id": request_id,
        "question": question,
        "locale": "zh-CN",
    }
    analysis = QuestionAnalyzer().analyze(request)
    scope = ScopeResolver().resolve(analysis, request)
    return TopologyCompositionPlanner().plan(analysis, scope)


def numeric_text(value: str) -> str:
    match = re.search(r"[-+]?\d+(?:\.\d+)?", str(value).replace(",", ""))
    if not match:
        raise ValueError(f"expected numeric fixture value: {value!r}")
    return match.group(0)


def main() -> int:
    facts_path = ROOT / "data/facts/financial_facts.duckdb"
    facts_jsonl = ROOT / "data/facts/financial_facts.jsonl"
    immutable_before = {"duckdb": sha256_file(facts_path), "jsonl": sha256_file(facts_jsonl)}
    repository = FactRepository(facts_path)
    candidates = FallbackCandidateIndex()
    fixture_manifest = json.loads((ROOT / "runs/phase_08/supported_fixture_manifest.json").read_text(encoding="utf-8"))

    fact_rows: list[dict[str, Any]] = []
    for index, fixture in enumerate(fixture_manifest["type1_exact"]):
        plan = plan_question(fixture["question"], f"gate3_fact_{index}")
        subplan = plan["subplans"][0]
        payload = subplan["payload"]
        lookup = repository.lookup_fact({
            "schema_version": "finglmqa.phase8.fact_lookup_request.v1",
            "requirement_id": f"gate3_fact_{index}",
            "document_id": payload["document_id"],
            "stock_code": payload["stock_code"],
            "report_year": payload["report_year"],
            "metric_year": payload["metric_year"],
            "canonical_metric": payload["canonical_metric"],
            "normalized_unit": payload["normalized_unit"],
        })
        expected = numeric_text(fixture["answer_prompt"]["prom_answer"])
        actual = lookup["records"][0]["normalized_value"] if lookup["status"] == "found" else None
        fact_rows.append({
            "uid": fixture["uid"], "status": lookup["status"], "expected": expected,
            "actual": actual, "passed": actual is not None and Decimal(actual) == Decimal(expected),
        })

    formula_executor = FormulaExecutor(repository, fallback_candidates=candidates)
    formula_rows: list[dict[str, Any]] = []
    for index, fixture in enumerate(fixture_manifest["formula_exact"]):
        plan = plan_question(fixture["question"], f"gate5_formula_{index}")
        subplan = plan["subplans"][0]
        result = formula_executor.execute(subplan)
        expected = numeric_text(fixture["answer_prompt"]["prom_answer"])
        rendered = None
        if result["status"] == "ok":
            rendered_decimal = (Decimal(result["result"]["value"]) * Decimal("100")).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
            rendered = format(rendered_decimal, ".2f")
        formula_rows.append({
            "uid": fixture["uid"], "status": result["status"], "expected_percent": expected,
            "actual_percent": rendered, "passed": rendered == format(Decimal(expected), ".2f"),
            "operand_fact_ids": [row["fact_id"] for row in result["result"]["operands"]] if result["result"] else [],
        })

    sql_executor = SQLExecutor(repository)
    sql_cases: list[dict[str, Any]] = []
    for case_id, question, expected_kind in (
        ("rank", "2019年所有公司中营业收入最高的公司是谁？", "rank"),
        ("average", "2019年所有公司营业收入的平均值是多少？", "aggregate"),
    ):
        plan = plan_question(question, f"gate4_{case_id}")
        subplan = plan["subplans"][0]
        execution = sql_executor.execute(subplan)
        sql_cases.append({
            "case_id": case_id,
            "pattern_id": plan["pattern_id"],
            "query_kind": execution["query_spec"]["query_kind"],
            "status": execution["status"],
            "row_count": len(execution["rows"]),
            "coverage_complete": execution["coverage"]["complete"],
            "warning_codes": [row["warning_code"] for row in execution["warnings"]],
            "contributing_fact_ids": [
                fact_id for row in execution["rows"] for fact_id in row["contributing_fact_ids"]
            ],
            "report_years": execution["query_spec"]["report_years"],
            "contributing_report_years": sorted({
                source["report_year"]
                for row in execution["rows"] for source in row["derivation_inputs"]
            }),
            "passed": (
                execution["query_spec"]["query_kind"] == expected_kind
                and bool(execution["rows"])
                and all(row["contributing_fact_ids"] for row in execution["rows"])
                and execution["query_spec"]["report_years"] == [2019]
                and all(
                    source["report_year"] == 2019
                    for row in execution["rows"] for source in row["derivation_inputs"]
                )
                and (
                    execution["coverage"]["complete"]
                    or "CORPUS_COVERAGE_INCOMPLETE" in [row["warning_code"] for row in execution["warnings"]]
                )
            ),
        })

    test_command = [
        str(ROOT / ".venv/bin/python"), "-m", "unittest", "-v",
        "tests.test_phase08_fact_lookup_port", "tests.test_phase08_repositories",
        "tests.test_phase08_real_fact_lookup_port", "tests.test_phase08_sql_engine",
        "tests.test_phase08_formula_engine",
    ]
    completed = subprocess.run(
        test_command, cwd=ROOT, env={**os.environ, "PYTHONPATH": "src", "PYTHONDONTWRITEBYTECODE": "1"},
        text=True, capture_output=True,
    )
    immutable_after = {"duckdb": sha256_file(facts_path), "jsonl": sha256_file(facts_jsonl)}
    checks = {
        "gate3_fact_exact_9": len(fact_rows) == 9 and all(row["passed"] for row in fact_rows),
        "gate3_phase6_immutable": immutable_before == immutable_after,
        "gate3_repository_fingerprint_matches": repository.repository_fingerprint == immutable_before["duckdb"],
        "gate4_sql_registered_cases": len(sql_cases) == 2 and all(row["passed"] for row in sql_cases),
        "gate4_report_year_metric_year_separated": all(
            row["report_years"] == [2019] and row["contributing_report_years"] == [2019]
            for row in sql_cases
        ),
        "gate5_formula_exact_6": len(formula_rows) == 6 and all(row["passed"] for row in formula_rows),
        "gate5_formula_provenance_complete": all(len(row["operand_fact_ids"]) >= 2 for row in formula_rows),
        "w2_unit_tests_pass": completed.returncode == 0,
    }
    report = {
        "schema_version": "finglmqa.phase8.gates_3_5_report.v1",
        "checks": checks,
        "facts": fact_rows,
        "sql": sql_cases,
        "formulas": formula_rows,
        "phase6_hashes_before": immutable_before,
        "phase6_hashes_after": immutable_after,
        "test_command": test_command,
        "test_stdout": completed.stdout,
        "test_stderr": completed.stderr,
        "artifact_fingerprints": {
            "repository": repository.repository_fingerprint,
            "fallback_candidates": candidates.repository_fingerprint,
            "report_semantic_inputs": semantic_sha256({"facts": fact_rows, "sql": sql_cases, "formulas": formula_rows}),
        },
    }
    report["all_checks_passed"] = all(checks.values())
    atomic_json(ROOT / "runs/phase_08/gates_3_5_report.json", report)
    print(json.dumps({"all_checks_passed": report["all_checks_passed"], "checks": checks}, ensure_ascii=False, indent=2))
    return 0 if report["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
