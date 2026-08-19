#!/usr/bin/env python3
"""Independent Phase 9 Gate 1-7 validators (Gate 0 has its own probe)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from finglmqa.contracts import canonical_json_bytes, validate_missing_fact_request  # noqa: E402
from finglmqa.pipeline import Phase8Pipeline  # noqa: E402
from finglmqa.ports import validate_fact_lookup_result  # noqa: E402
from finglmqa.supplement_contracts import (  # noqa: E402
    FAILURE_CODES, SCHEMA_SUPPLEMENTAL_FACT, validate_supplement_decision,
    validate_supplemental_fact,
)
from finglmqa.supplement_store import (  # noqa: E402
    SupplementAwareFactRepository, SupplementalFactRepository, materialize_store, sha256_file,
)


RUN = ROOT / "runs/phase_09"
REPORTS = RUN / "reports"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(canonical_json_bytes(value))
    os.replace(temporary, path)


def report(gate: int, checks: Mapping[str, bool], details: Mapping[str, Any] | None = None) -> dict[str, Any]:
    value = {
        "schema_version": "finglmqa.phase9.gate_report.v1",
        "gate": gate,
        "status": "passed" if all(checks.values()) else "failed",
        "checks": dict(checks),
        "details": dict(details or {}),
    }
    write_json(RUN / f"gate{gate}_report.json", value)
    return value


def immutable_check() -> tuple[bool, list[str]]:
    manifest = json.loads((RUN / "immutable_inputs_manifest.json").read_text(encoding="utf-8"))
    changed: list[str] = []
    for row in [*manifest["entries"], *manifest["phase7_phase8_release_inputs"]]:
        label = row["path"]
        path = ROOT / "refs/tabgr_runtime/build_graphs/graph_to_text_triple_full.py" if label.startswith("external:") else ROOT / label
        stat = path.stat()
        if sha256_file(path) != row["sha256"] or stat.st_size != row["size"] or stat.st_mtime_ns != row["mtime_ns"]:
            changed.append(label)
    return not changed, changed


def gate1() -> dict[str, Any]:
    requests = read_jsonl(REPORTS / "supplement_requests.jsonl")
    valid = True
    for row in requests:
        try:
            validate_missing_fact_request(row)
        except Exception:
            valid = False
    summary = json.loads((REPORTS / "request_universe_summary.json").read_text(encoding="utf-8"))
    arithmetic = summary["arithmetic"]
    expected = {
        "grid_slots": 5610, "covered_selected_slots": 4189, "missing_request_slots": 1421,
        "conflict_group_open": 501, "fact_withheld": 30, "no_exact_unit_fact": 890,
    }
    schema_files = [
        ROOT / "data/schemas/supplemental_facts.schema.json",
        ROOT / "data/schemas/supplement_decision.schema.json",
        ROOT / "data/schemas/supplement_lookup_result.schema.json",
    ]
    schemas_closed = all(json.loads(path.read_text(encoding="utf-8")).get("additionalProperties") is False for path in schema_files)
    wrapper_conformant = False
    if (ROOT / "data/facts/supplemental_facts.duckdb").is_file():
        import duckdb
        connection = duckdb.connect(str(ROOT / "data/facts/financial_facts.duckdb"), read_only=True)
        try:
            sample = connection.execute("""
                SELECT document_id,stock_code,report_year,metric_year,canonical_metric,normalized_unit
                FROM selected_financial_facts ORDER BY fact_id LIMIT 1
            """).fetchone()
        finally:
            connection.close()
        lookup_request = {
            "schema_version": "finglmqa.phase8.fact_lookup_request.v1", "requirement_id": "gate1_contract",
            **dict(zip(("document_id", "stock_code", "report_year", "metric_year", "canonical_metric", "normalized_unit"), sample, strict=True)),
        }
        try:
            wrapper_conformant = validate_fact_lookup_result(SupplementAwareFactRepository().lookup_fact(lookup_request))["status"] == "found"
        except Exception:
            wrapper_conformant = False
    checks = {
        "three_new_schemas_closed": schemas_closed,
        "failure_code_enum_frozen": len(FAILURE_CODES) == 13,
        "requests_contract_valid": valid,
        "exact_unit_arithmetic": arithmetic == expected,
        "requests_unique": len(requests) == len({row["requirement_id"] for row in requests}) == 1421,
        "request_file_hash_matches_summary": sha256_file(REPORTS / "supplement_requests.jsonl") == summary["request_file_sha256"],
        "supplement_aware_wrapper_v1_conformant": wrapper_conformant,
    }
    return report(1, checks, {"arithmetic": arithmetic, "request_sha256": sha256_file(REPORTS / "supplement_requests.jsonl")})


def gate2() -> dict[str, Any]:
    probe = r'''
import json
from pathlib import Path
from finglmqa.contracts import canonical_json_bytes
from finglmqa.tabgr_adapter import TabGRAdapter
root=Path.cwd()
with (root/'data/corpus_package/tabgr_table_corpus.jsonl').open(encoding='utf8') as f: table=json.loads(next(f))
cells=[]
with (root/'data/corpus_package/table_cells.jsonl').open(encoding='utf8') as f:
  for line in f:
    row=json.loads(line)
    if row['table_id']==table['table_id']: cells.append(row)
    elif cells: break
result=TabGRAdapter().rank_table(table,cells,aliases=['营业收入'],metric_year=2019,normalized_unit='元')
import sys; sys.stdout.buffer.write(canonical_json_bytes(result))
'''
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    first = subprocess.run([str(ROOT / ".venv/bin/python"), "-c", probe], cwd=ROOT, env=env, check=True, capture_output=True).stdout
    second = subprocess.run([str(ROOT / ".venv/bin/python"), "-c", probe], cwd=ROOT, env=env, check=True, capture_output=True).stdout
    parsed = json.loads(first)
    coordinates = set()
    with (ROOT / "data/corpus_package/table_cells.jsonl").open(encoding="utf-8") as handle:
        target = parsed["ranked_cells"][0]["table_id"]
        started = False
        for raw in handle:
            row = json.loads(raw)
            if row["table_id"] == target:
                started = True
                coordinates.add((row["row_index"], row["col_index"], row["raw_value"]))
            elif started:
                break
    exact = all((row["row_index"], row["col_index"], row["raw_value"]) in coordinates for row in parsed["ranked_cells"])
    source = ROOT / "refs/tabgr_runtime/build_graphs/graph_to_text_triple_full.py"
    corrupted_fails = False
    with tempfile.TemporaryDirectory() as directory:
        bad = Path(directory) / "runtime.py"
        bad.write_bytes(source.read_bytes() + b"\n# corrupted\n")
        try:
            from finglmqa.tabgr_adapter import TabGRAdapter, TabGRRuntimeUnavailable
            TabGRAdapter(bad)
        except TabGRRuntimeUnavailable:
            corrupted_fails = True
    checks = {
        "fresh_process_byte_identical": first == second,
        "triple_cell_mapping_exact": exact,
        "scores_fixed_8_decimals": all(len(row["score"].partition(".")[2]) == 8 for row in parsed["ranked_cells"]),
        "public_renderer_consistency_recorded": bool(parsed["public_renderer_sha256"]),
        "corrupted_source_fails_closed": corrupted_fails,
    }
    return report(2, checks, {"sample_sha256": hashlib.sha256(first).hexdigest(), "mapped_cells": len(parsed["ranked_cells"])})


def gate3() -> dict[str, Any]:
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    completed = subprocess.run(
        [str(ROOT / ".venv/bin/python"), "-m", "unittest", "discover", "-s", "tests", "-p", "test_phase09*.py", "-q"],
        cwd=ROOT, env=env, capture_output=True, text=True,
    )
    china_decisions = [
        row for row in read_jsonl(ROOT / "runs/phase_06/reports/candidate_decisions.jsonl")
        if row.get("candidate_id") == "A688009_中国通号_2019年年度报告_table_0255_119d0118d7:r1:c1:营业收入"
    ]
    checks = {
        "adversarial_unit_tests_pass": completed.returncode == 0,
        "flyada_and_forgery_fixtures_present": "test_flyada" in (ROOT / "tests/test_phase09_validation.py").read_text(encoding="utf-8"),
        "china_signal_phase6_fixture_rejected": (
            len(china_decisions) == 1
            and china_decisions[0].get("eligible") is False
            and "partial_duration_date" in china_decisions[0].get("rejection_reasons", [])
        ),
        "unknown_ascii_suffix_regression_present": "48,946,000.0o" in (ROOT / "tests/test_phase09_validation.py").read_text(encoding="utf-8"),
    }
    return report(3, checks, {"test_output": (completed.stdout + completed.stderr).strip()})


def gate4() -> dict[str, Any]:
    facts = read_jsonl(ROOT / "data/facts/supplemental_facts.jsonl")
    decisions = read_jsonl(REPORTS / "supplement_decisions.jsonl")
    facts_valid = all(validate_supplemental_fact(row) is row for row in facts)
    decisions_valid = all(validate_supplement_decision(row) is row for row in decisions)
    import duckdb
    connection = duckdb.connect(str(ROOT / "data/facts/financial_facts.duckdb"), read_only=True)
    try:
        selected = {tuple(row) for row in connection.execute("SELECT document_id,stock_code,report_year,metric_year,canonical_metric,normalized_unit FROM selected_financial_facts").fetchall()}
        retained = {(tuple(row[:6]), row[6]) for row in connection.execute("SELECT document_id,stock_code,report_year,metric_year,canonical_metric,normalized_unit,normalized_value_text FROM financial_facts").fetchall()}
    finally:
        connection.close()
    keys = {(row["document_id"], row["stock_code"], row["report_year"], row["metric_year"], row["canonical_metric"], row["normalized_unit"]) for row in facts}
    conflicts = any((key, fact["normalized_value"]) not in retained and any(existing_key == key for existing_key, _ in retained) for key, fact in zip(keys, facts)) if facts else False
    repeat = RUN / "repeatability/run2"
    repeat_exists = all((repeat / name).is_file() for name in ("supplemental_facts.jsonl", "supplement_decisions.jsonl", "supplement_trace.jsonl"))
    deterministic = repeat_exists and all(
        (left.read_bytes() == right.read_bytes())
        for left, right in (
            (ROOT / "data/facts/supplemental_facts.jsonl", repeat / "supplemental_facts.jsonl"),
            (REPORTS / "supplement_decisions.jsonl", repeat / "supplement_decisions.jsonl"),
            (REPORTS / "supplement_trace.jsonl", repeat / "supplement_trace.jsonl"),
        )
    )
    checks = {
        "all_1421_requests_terminal": len(decisions) == len({row["slot_fingerprint"] for row in decisions}) == 1421,
        "facts_contract_valid": facts_valid,
        "decisions_contract_valid": decisions_valid,
        "zero_selected_slot_overlap": not (keys & selected),
        "zero_retained_value_conflict": not conflicts,
        "two_full_runs_byte_identical": deterministic,
    }
    return report(4, checks, {"fact_count": len(facts), "decision_count": len(decisions), "repeatability_dir": "runs/phase_09/repeatability/run2"})


def gate5() -> dict[str, Any]:
    facts = read_jsonl(ROOT / "data/facts/supplemental_facts.jsonl")
    decisions = read_jsonl(REPORTS / "supplement_decisions.jsonl")
    import duckdb
    connection = duckdb.connect(str(ROOT / "data/facts/supplemental_facts.duckdb"), read_only=True)
    try:
        db_facts = [json.loads(row[0]) for row in connection.execute("SELECT fact_json FROM supplemental_facts ORDER BY supplemental_fact_id").fetchall()]
        db_decisions = [json.loads(row[0]) for row in connection.execute("SELECT decision_json FROM supplement_decisions ORDER BY slot_fingerprint").fetchall()]
        tables = {row[0] for row in connection.execute("SHOW TABLES").fetchall()}
    finally:
        connection.close()
    immutable, changed = immutable_check()
    checks = {
        "duckdb_has_three_tables": tables == {"supplemental_facts", "supplement_decisions", "build_metadata"},
        "duckdb_jsonl_facts_reconcile": sorted(db_facts, key=lambda row: row["supplemental_fact_id"]) == facts,
        "duckdb_decisions_reconcile": sorted(db_decisions, key=lambda row: row["slot_key"]) == decisions,
        "phase6_and_other_immutable_inputs_unchanged": immutable,
    }
    return report(5, checks, {"fact_rows": len(facts), "decision_rows": len(decisions), "changed_inputs": changed})


def _fixture_fact(document_id: str, year: int, value: str, ordinal: int) -> dict[str, Any]:
    return {
        "supplemental_fact_id": f"sf_{ordinal:024x}", "schema_version": SCHEMA_SUPPLEMENTAL_FACT,
        "document_id": document_id, "stock_code": "601298", "report_year": 2019,
        "metric_year": year, "canonical_metric": "营业收入", "normalized_unit": "元",
        "company": "青岛港", "statement": "income_statement", "normalized_value": value,
        "fact_source": "supplemental_tabgr", "validation_versions": {"fixture": "gate6-contract-only"},
        "tabgr_trace_fingerprint": "0" * 64, "source_table_id": f"fixture_table_{ordinal}",
        "source_table_index": ordinal, "source_row_index": 1, "source_col_index": 1,
        "source_line_start": 1, "source_line_end": 1, "source_markdown": "refs/source_markdown/gate6_fixture.md",
        "provenance_json": [{"fact_source": "supplemental_tabgr", "fixture": True, "table_id": f"fixture_table_{ordinal}"}],
        "created_from_requirement_ids": [f"fixture_req_{ordinal}"],
    }


def gate6() -> dict[str, Any]:
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    phase8 = subprocess.run(
        [str(ROOT / ".venv/bin/python"), "-m", "unittest", "discover", "-s", "tests", "-p", "test_phase08*.py", "-q"],
        cwd=ROOT, env=env, capture_output=True, text=True,
    )
    with tempfile.TemporaryDirectory() as directory:
        database = Path(directory) / "fixture.duckdb"
        facts = [_fixture_fact("A601298_青岛港_2019年年度报告", 2019, "100", 1), _fixture_fact("A601298_青岛港_2019年年度报告", 2018, "80", 2)]
        materialize_store(facts, [], database)
        wrapper = SupplementAwareFactRepository(supplemental=SupplementalFactRepository(database))
        request = {
            "schema_version": "finglmqa.phase8.fact_lookup_request.v1", "requirement_id": "gate6_lookup",
            "document_id": "A601298_青岛港_2019年年度报告", "stock_code": "601298",
            "report_year": 2019, "metric_year": 2019, "canonical_metric": "营业收入", "normalized_unit": "元",
        }
        lookup = validate_fact_lookup_result(wrapper.lookup_fact(request))
        provenance = json.loads(lookup["records"][0]["provenance_json"])
        pipeline = Phase8Pipeline(fact_repository=wrapper)
        direct = pipeline.run({
            "schema_version": "finglmqa.phase8.qa_request.v1", "request_id": "gate6_direct",
            "question": "青岛港2019年营业收入是多少？", "locale": "zh-CN", "normalized_unit": "元",
        }).answer
        formula = pipeline.run({
            "schema_version": "finglmqa.phase8.qa_request.v1", "request_id": "gate6_formula",
            "question": "青岛港2019年营业收入增长率是多少？", "locale": "zh-CN",
        }).answer
        missing = pipeline.run({
            "schema_version": "finglmqa.phase8.qa_request.v1", "request_id": "gate6_missing",
            "question": "青岛港2019年总资产是多少？", "locale": "zh-CN", "normalized_unit": "元",
        }).answer
    frozen_gate2 = json.loads((ROOT / "runs/phase_08/gate2_report.json").read_text(encoding="utf-8"))
    real_paths = [
        ROOT / "runs/phase_09/repeatability/phase8_real_1.json",
        ROOT / "runs/phase_09/repeatability/phase8_real_2.json",
        ROOT / "runs/phase_08/reports/real_evidence_smoke_run_1.json",
    ]
    real_rows = [json.loads(path.read_text(encoding="utf-8")) for path in real_paths] if all(path.is_file() for path in real_paths) else []
    real_repeatable = bool(real_rows) and all(
        canonical_json_bytes(real_rows[0][key]) == canonical_json_bytes(row[key])
        for key in ("answer", "trace") for row in real_rows[1:]
    )
    checks = {
        "phase8_171_tests_unchanged": phase8.returncode == 0 and "Ran 171 tests" in (phase8.stdout + phase8.stderr),
        "frozen_gate2_oracle_remains_passed": frozen_gate2.get("all_checks_passed") is True,
        "real_evidence_answer_trace_matches_frozen_release": real_repeatable,
        "supplement_lookup_is_v1_conformant": lookup["status"] == "found" and provenance.get("fact_source") == "supplemental_tabgr",
        "direct_fact_renders_marker": "补充来源：TabGR" in direct["answer_text"],
        "formula_renders_marker": "补充来源：TabGR" in formula["answer_text"],
        "non_supplemented_slot_still_emits_request": bool(missing["missing_fact_requests"]),
        "citation_scope_validation_passes": direct["status"] == "ok" and formula["status"] == "ok",
    }
    return report(6, checks, {
        "phase8_test_output": (phase8.stdout + phase8.stderr).strip(),
        "direct_status": direct["status"], "formula_status": formula["status"], "missing_status": missing["status"],
        "real_evidence_trace_hash": real_rows[0]["trace"]["trace_hash"] if real_rows else None,
    })


def gate7() -> dict[str, Any]:
    requests = read_jsonl(REPORTS / "supplement_requests.jsonl")
    decisions = read_jsonl(REPORTS / "supplement_decisions.jsonl")
    facts = read_jsonl(ROOT / "data/facts/supplemental_facts.jsonl")
    traces = read_jsonl(REPORTS / "supplement_trace.jsonl")
    decision_by_requirement = {row["requirement_id"]: row for row in decisions}
    qingdao_requests = [row for row in requests if row["document_id"] == "A601298_青岛港_2019年年度报告"]
    qingdao_decisions = [decision_by_requirement[row["requirement_id"]] for row in qingdao_requests]
    case = {
        "schema_version": "finglmqa.phase9.qingdao_port_case_report.v1",
        "document_id": "A601298_青岛港_2019年年度报告",
        "request_count": len(qingdao_requests),
        "accepted_count": sum(row["decision_status"] == "accepted" for row in qingdao_decisions),
        "counts_by_failure_code": dict(sorted(Counter(row["failure_code"] for row in qingdao_decisions if row["failure_code"]).items())),
        "explicit_unit_fixture_outcomes": [
            {"metric_year": request["metric_year"], "canonical_metric": request["canonical_metric"], "normalized_unit": request["normalized_unit"],
             "decision_status": decision_by_requirement[request["requirement_id"]]["decision_status"], "failure_code": decision_by_requirement[request["requirement_id"]]["failure_code"]}
            for request in qingdao_requests
            if request["canonical_metric"] in {"股本", "基本每股收益"}
        ],
    }
    write_json(REPORTS / "qingdao_port_case_report.json", case)
    official = [
        ROOT / "data/facts/supplemental_facts.jsonl", REPORTS / "supplement_decisions.jsonl",
        REPORTS / "supplement_trace.jsonl", REPORTS / "supplement_summary.json",
        REPORTS / "qingdao_port_case_report.json",
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in official)
    raw_keys = any(f'"{key}":' in text for key in ("matrix", "matrix_text", "edge_list", "table", "content"))
    telemetry = any(f'"{key}":' in text for key in ("timestamp", "duration_ms", "pid", "device", "temporary_path"))
    absolute = "/home/" in text or "/Users/" in text
    immutable, changed = immutable_check()
    summary = json.loads((REPORTS / "supplement_summary.json").read_text(encoding="utf-8"))
    checks = {
        "summary_report_complete": summary["request_count"] == summary["decision_count"] == 1421,
        "qingdao_case_closed": case["request_count"] == 33 and case["accepted_count"] + sum(case["counts_by_failure_code"].values()) == 33,
        "no_raw_table_payload_keys": not raw_keys,
        "no_host_absolute_paths": not absolute,
        "no_runtime_telemetry_fields": not telemetry,
        "immutable_inputs_still_unchanged": immutable,
    }
    return report(7, checks, {"qingdao": case, "changed_inputs": changed, "fact_count": len(facts), "trace_count": len(traces)})


GATES = {1: gate1, 2: gate2, 3: gate3, 4: gate4, 5: gate5, 6: gate6, 7: gate7}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", type=int, choices=GATES)
    args = parser.parse_args()
    selected = [args.gate] if args.gate else list(GATES)
    results = [GATES[gate]() for gate in selected]
    print(json.dumps(results, ensure_ascii=False, sort_keys=True))
    return 0 if all(row["status"] == "passed" for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
