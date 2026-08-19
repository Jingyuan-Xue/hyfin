#!/usr/bin/env python3
"""Run the Phase 8 decomposition oracle without executing any backend."""

from __future__ import annotations

import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(ROOT / "src"))

from finglmqa.analyzer import QuestionAnalyzer  # noqa: E402
from finglmqa.composition import CompositionPlanningError, TopologyCompositionPlanner  # noqa: E402
from finglmqa.contracts import canonical_json_bytes, semantic_sha256  # noqa: E402
from finglmqa.resolver import ScopeResolver  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(canonical_json_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        for row in rows:
            handle.write(canonical_json_bytes(row))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def request(case_id: str, question: str) -> dict[str, Any]:
    return {
        "schema_version": "finglmqa.phase8.qa_request.v1",
        "request_id": case_id.replace(":", "_"),
        "question": question,
        "locale": "zh-CN",
    }


def execute_planning(
    analyzer: QuestionAnalyzer,
    resolver: ScopeResolver,
    planner: TopologyCompositionPlanner,
    req: Mapping[str, Any],
) -> dict[str, Any]:
    analysis = analyzer.analyze(req)
    scope = resolver.resolve(analysis, req)
    try:
        plan = planner.plan(analysis, scope)
    except CompositionPlanningError as exc:
        terminal = exc.as_dict()
        return {
            "analysis": analysis, "scope": scope, "plan": None, "terminal": terminal,
            "backend_call_count": 0,
        }
    return {
        "analysis": analysis, "scope": scope, "plan": plan, "terminal": None,
        "backend_call_count": 0,
    }


def signature(subplan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "backend": subplan["backend"],
        "operation": subplan["operation"],
        "planning_state": subplan["planning_state"],
    }


def analysis_projection(analysis: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "company_mentions": [
            {key: row[key] for key in ("raw_text", "span", "mention_ordinal", "hint_source")}
            for row in analysis["company_mentions"]
        ],
        "year_mentions": [
            {key: row[key] for key in ("raw_text", "span", "years", "role", "is_output_axis", "mention_ordinal")}
            for row in analysis["year_mentions"]
        ],
        "concerns": [
            {
                key: row.get(key)
                for key in (
                    "concern_id", "mention_ordinal", "kind", "raw_text", "canonical_metric",
                    "formula_id", "metadata_field", "normalized_unit", "unit_source",
                )
            }
            for row in analysis["concerns"]
        ],
        "intents": list(analysis["intents"]),
        "narrative_mode": analysis["narrative_mode"],
        "evidence_kinds": list(analysis["evidence_kinds"]),
        "output_entity_axis": list(analysis["output_entity_axis"]),
        "output_period_axis": list(analysis["output_period_axis"]),
        "dynamic_target_dependency": analysis["dynamic_target_dependency"],
        "unsupported_marker_codes": [row["code"] for row in analysis["unsupported_markers"]],
    }


def scope_projection(scope: Mapping[str, Any]) -> dict[str, Any]:
    corpus = scope.get("corpus_scope")
    return {
        "scope_kind": scope["scope_kind"],
        "report_year_constraints": list(scope["report_year_constraints"]),
        "metric_years": list(scope["metric_years"]),
        "entities": [
            {
                "entity_key": row["entity_key"],
                "mention": row["mention"],
                "mention_ordinal": row["mention_ordinal"],
                "status": row["status"],
                "identity": row["identity"],
                "documents": [
                    {
                        "document_id": document["document_id"],
                        "stock_code": document["stock_code"],
                        "report_year": document["report_year"],
                    }
                    for document in row["document_set"]
                ],
                "finding_codes": [finding["failure_code"] for finding in row["findings"]],
            }
            for row in scope["entity_resolutions"]
        ],
        "corpus_scope": (
            {
                "scope_version": corpus["scope_version"],
                "source_fingerprint": corpus["source_fingerprint"],
                "company_count": corpus["company_count"],
                "document_count": corpus["document_count"],
                "report_years": corpus["report_years"],
                "allowed_operations": corpus["allowed_operations"],
            }
            if corpus is not None else None
        ),
        "finding_codes": [finding["failure_code"] for finding in scope["findings"]],
    }


def plan_projection(plan: Mapping[str, Any]) -> dict[str, Any]:
    ordinal_by_id = {row["subplan_id"]: row["ordinal"] for row in plan["subplans"]}
    return {
        "pattern_id": plan["pattern_id"],
        "pattern_version": plan["pattern_version"],
        "registry_semantic_sha256": plan["registry_semantic_sha256"],
        "output_kind": plan["output_kind"],
        "subplans": [
            {
                "ordinal": row["ordinal"],
                "planning_state": row["planning_state"],
                "backend": row["backend"],
                "operation": row["operation"],
                "entity_key": row["entity_key"],
                "period_key": row["period_key"],
                "concern_key": row["concern_key"],
                "required": row["required"],
                "declared_scope": row["declared_scope"],
                "payload": row["payload"],
                "planning_failure": row["planning_failure"],
                "depends_on_ordinals": [ordinal_by_id[value] for value in row["depends_on_subplan_ids"]],
                "authorization_source_ordinals": [
                    ordinal_by_id[value] for value in row["authorization_source_subplan_ids"]
                ],
            }
            for row in plan["subplans"]
        ],
        "composition_policy": plan["composition_policy"],
        "numeric_answer_policy": plan["numeric_answer_policy"],
        "limit_evaluation": plan["limit_evaluation"],
        "dynamic_expansion": plan["dynamic_expansion"],
    }


def frozen_projection(actual: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "analysis": analysis_projection(actual["analysis"]),
        "scope": scope_projection(actual["scope"]),
        "plan": plan_projection(actual["plan"]) if actual["plan"] is not None else None,
        "terminal": actual["terminal"],
        "backend_call_count": actual["backend_call_count"],
    }


def benchmark_check(actual: Mapping[str, Any], gold: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    expected_projection = gold.get("expected_planning_projection")
    if expected_projection is None:
        return ["gold_missing_full_planning_projection"]
    observed_projection = frozen_projection(actual)
    if observed_projection != expected_projection:
        errors.append(
            "full_projection_sha256:"
            f"{semantic_sha256(observed_projection)}!={semantic_sha256(expected_projection)}"
        )
    if gold["capability"]["executable_in_phase8"]:
        plan = actual["plan"]
        if plan is None or any(row["planning_state"] != "ready" for row in plan["subplans"]):
            errors.append("executable_case_is_not_fully_ready")
        expected_stock = gold["expected_resolution"]["expected_stock_code"]
        resolved_stocks = {
            row["identity"]["stock_code"]
            for row in actual["scope"]["entity_resolutions"]
            if row["status"] == "unique" and row["identity"] is not None
        }
        if expected_stock is not None and expected_stock not in resolved_stocks:
            errors.append(f"expected_stock_not_resolved:{expected_stock}")
    expected = gold["expected_composition"]
    if expected["registry_semantic_sha256"] != planner_registry_hash(actual):
        errors.append("gold_registry_hash_does_not_match_plan")
    return errors


def planner_registry_hash(actual: Mapping[str, Any]) -> str | None:
    return actual["plan"]["registry_semantic_sha256"] if actual["plan"] is not None else None


def general_signature_matches(actual: Mapping[str, Any], expected_text: str) -> bool:
    parts = expected_text.split(":")
    if actual["backend"] != parts[0]:
        return False
    if len(parts) == 1:
        return True
    if parts[1] in {"ready", "blocked"}:
        return actual["planning_state"] == parts[1]
    return actual["operation"] == parts[1]


def general_check(actual: Mapping[str, Any], gold: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    expected = gold["expected"]
    full = gold.get("expected_planning_projection")
    if full is None:
        return ["gold_missing_full_planning_projection"]
    observed_full = frozen_projection(actual)
    if observed_full != full:
        errors.append(
            "full_projection_sha256:"
            f"{semantic_sha256(observed_full)}!={semantic_sha256(full)}"
        )
    plan, terminal = actual["plan"], actual["terminal"]
    if expected["pattern_id"] is None:
        if terminal is None:
            errors.append("expected_terminal_but_plan_emitted")
            return errors
        if terminal["failure_code"] != expected["failure_code"]:
            errors.append(f"failure:{terminal['failure_code']}!={expected['failure_code']}")
        if terminal["subplans"] or terminal["backend_call_count"] != 0:
            errors.append("terminal_not_zero_call")
        return errors
    if plan is None:
        errors.append(f"unexpected_terminal:{terminal['failure_code']}")
        return errors
    if plan["pattern_id"] != expected["pattern_id"]:
        errors.append(f"pattern:{plan['pattern_id']}!={expected['pattern_id']}")
    observed = [signature(row) for row in plan["subplans"]]
    wanted = expected["ordered_subplan_signatures"]
    if len(observed) != len(wanted):
        errors.append(f"subplan_count:{len(observed)}!={len(wanted)}")
    else:
        for index, (row, text) in enumerate(zip(observed, wanted)):
            if not general_signature_matches(row, text):
                errors.append(f"signature[{index}]:{row!r}!={text}")
    failure = expected.get("failure_code")
    if failure:
        blocked_codes = [
            row["planning_failure"]["failure_code"]
            for row in plan["subplans"]
            if row["planning_state"] == "blocked"
        ]
        if failure not in blocked_codes:
            errors.append(f"missing_blocked_failure:{failure}:{blocked_codes!r}")
    return errors


def synthetic_records() -> list[dict[str, Any]]:
    return [
        {
            "aliases": ["示例公司", "示例多报告公司"], "document_id": "A900001_示例公司_2019年年度报告",
            "stock_code": "900001", "stock_name": "示例公司", "company_full": "示例公司股份有限公司", "report_year": 2019,
        },
        {
            "aliases": ["示例公司", "示例多报告公司"], "document_id": "A900001_示例公司_2020年年度报告",
            "stock_code": "900001", "stock_name": "示例公司", "company_full": "示例公司股份有限公司", "report_year": 2020,
        },
        {
            "aliases": ["飞亚达", "飞亚达精密科技股份有限公司"], "document_id": "A000026_飞亚达_2019年年度报告",
            "stock_code": "000026", "stock_name": "飞亚达", "company_full": "飞亚达精密科技股份有限公司", "report_year": 2019,
        },
        {
            "aliases": ["示例歧义公司"], "document_id": "A900002_示例甲_2019年年度报告",
            "stock_code": "900002", "stock_name": "示例甲", "company_full": "示例甲股份有限公司", "report_year": 2019,
        },
        {
            "aliases": ["示例歧义公司"], "document_id": "A900003_示例乙_2019年年度报告",
            "stock_code": "900003", "stock_name": "示例乙", "company_full": "示例乙股份有限公司", "report_year": 2019,
        },
    ]


def main() -> int:
    analyzer = QuestionAnalyzer()
    production_resolver = ScopeResolver()
    synthetic_resolver = ScopeResolver(records=synthetic_records())
    planner = TopologyCompositionPlanner()
    oracle = read_jsonl(ROOT / "runs/phase_08/benchmark_decomposition_oracle.jsonl")
    general = read_jsonl(ROOT / "runs/phase_08/general_decomposition_gold.jsonl")
    deviations: list[dict[str, Any]] = []
    benchmark_passed = 0
    deterministic_passed = 0
    benchmark_patterns: Counter[str] = Counter()
    for gold in oracle:
        req = request(gold["case_id"], gold["source"]["question"])
        first = execute_planning(analyzer, production_resolver, planner, req)
        second = execute_planning(analyzer, production_resolver, planner, req)
        deterministic = canonical_json_bytes(first) == canonical_json_bytes(second)
        deterministic_passed += int(deterministic)
        errors = benchmark_check(first, gold)
        if not deterministic:
            errors.append("non_deterministic")
        if errors:
            deviations.append({"case_id": gold["case_id"], "kind": "benchmark", "errors": errors})
        else:
            benchmark_passed += 1
        if first["plan"] is not None:
            benchmark_patterns[first["plan"]["pattern_id"]] += 1

    general_passed = 0
    synthetic_case_ids = {"G05", "G07", "G08", "G26", "G27", "G28", "G29", "G30"}
    general_rows: list[dict[str, Any]] = []
    for gold in general:
        req = request(gold["case_id"], gold["question"])
        resolver = synthetic_resolver if gold["case_id"] in synthetic_case_ids else production_resolver
        first = execute_planning(analyzer, resolver, planner, req)
        second = execute_planning(analyzer, resolver, planner, req)
        errors = general_check(first, gold)
        deterministic = canonical_json_bytes(first) == canonical_json_bytes(second)
        if not deterministic:
            errors.append("non_deterministic")
        general_rows.append({
            "case_id": gold["case_id"],
            "errors": errors,
            "pattern_id": first["plan"]["pattern_id"] if first["plan"] else None,
            "failure_code": first["terminal"]["failure_code"] if first["terminal"] else None,
            "subplans": [signature(row) for row in first["plan"]["subplans"]] if first["plan"] else [],
        })
        if errors:
            deviations.append({"case_id": gold["case_id"], "kind": "general", "errors": errors})
        else:
            general_passed += 1

    forbidden_runtime_tokens = ("benchmark_decomposition_oracle", "general_decomposition_gold", "answers.jsonl", "selected_questions")
    production_sources = [
        ROOT / "src/finglmqa/analyzer.py", ROOT / "src/finglmqa/metric_catalog.py",
        ROOT / "src/finglmqa/resolver.py", ROOT / "src/finglmqa/composition.py",
    ]
    source_text = "\n".join(path.read_text(encoding="utf-8") for path in production_sources)
    leakage_tokens = [token for token in forbidden_runtime_tokens if token in source_text]
    observed_backend_calls = sum(
        row["backend_call_count"]
        for row in [
            *[execute_planning(analyzer, production_resolver, planner, request(
                gold["case_id"], gold["source"]["question"]
            )) for gold in oracle[:1]],
            *[execute_planning(
                analyzer,
                synthetic_resolver if gold["case_id"] in synthetic_case_ids else production_resolver,
                planner,
                request(gold["case_id"], gold["question"]),
            ) for gold in general[:1]],
        ]
    )
    checks = {
        "benchmark_1003_pass": benchmark_passed == len(oracle) == 1003,
        "benchmark_1003_deterministic": deterministic_passed == len(oracle),
        "general_40_pass": general_passed == len(general) == 40,
        "no_oracle_runtime_leakage": not leakage_tokens,
        "zero_backend_calls_during_gate2": observed_backend_calls == 0,
    }
    report = {
        "schema_version": "finglmqa.phase8.gate2_report.v1",
        "checks": checks,
        "benchmark": {
            "total": len(oracle), "passed": benchmark_passed,
            "deterministic": deterministic_passed, "actual_patterns": dict(sorted(benchmark_patterns.items())),
        },
        "general": {"total": len(general), "passed": general_passed, "rows": general_rows},
        "deviation_count": len(deviations),
        "runtime_leakage_tokens": leakage_tokens,
        "artifacts": {
            "oracle_semantic_sha256": semantic_sha256(oracle),
            "general_gold_semantic_sha256": semantic_sha256(general),
            "registry_semantic_sha256": planner.registry_semantic_sha256,
        },
    }
    report["all_checks_passed"] = all(checks.values())
    atomic_json(ROOT / "runs/phase_08/gate2_report.json", report)
    atomic_jsonl(ROOT / "runs/phase_08/reports/gate2_deviations.jsonl", deviations)
    print(json.dumps({
        "all_checks_passed": report["all_checks_passed"], "checks": checks,
        "benchmark_passed": benchmark_passed, "general_passed": general_passed,
        "deviation_count": len(deviations),
    }, ensure_ascii=False, indent=2))
    return 0 if report["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
