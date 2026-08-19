#!/usr/bin/env python3
"""Validate integrated Phase 8 evidence, composition, and release gates."""

from __future__ import annotations

import hashlib
import gc
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from finglmqa.contracts import canonical_json_bytes  # noqa: E402
from finglmqa.pipeline import Phase8Pipeline  # noqa: E402


PHASE6_PINS = {
    "data/facts/financial_facts.duckdb": "b3e8fed65ddc1ccd5954083a4df64f3eab2150294cae08a11424f3bc5744f278",
    "data/facts/financial_facts.jsonl": "abeb4b3b221aac74705b84c80469c03b23fd8638d67004c75dd7a512c6841405",
}


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


def qa_request(request_id: str, question: str) -> dict[str, Any]:
    return {
        "schema_version": "finglmqa.phase8.qa_request.v1",
        "request_id": request_id,
        "question": question,
        "locale": "zh-CN",
    }


def run_command(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": "src", "PYTHONDONTWRITEBYTECODE": "1"},
        text=True,
        capture_output=True,
    )
    return {
        "command": command,
        "return_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def tree_has_key(value: Any, forbidden: str) -> bool:
    if isinstance(value, dict):
        return forbidden in value or any(tree_has_key(child, forbidden) for child in value.values())
    if isinstance(value, list):
        return any(tree_has_key(child, forbidden) for child in value)
    return False


def tree_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [text for child in value.values() for text in tree_strings(child)]
    if isinstance(value, list):
        return [text for child in value for text in tree_strings(child)]
    return []


def real_smoke_command(output: Path, model_cache: Path) -> list[str]:
    return [
        ".venv/bin/python", "scripts/query_phase_08.py",
        "飞亚达2019年面临哪些经营风险？",
        "--request-id", "phase8-real-evidence-smoke",
        "--device", "cpu",
        "--model-cache", str(model_cache),
        "--output", output.relative_to(ROOT).as_posix(),
    ]


def main() -> int:
    phase6_before = {path: sha256_file(ROOT / path) for path in PHASE6_PINS}
    gate_runs = [
        run_command([".venv/bin/python", "scripts/validate_phase_08_gate1.py"]),
        run_command([".venv/bin/python", "scripts/validate_phase_08_gate2.py"]),
        run_command([".venv/bin/python", "scripts/validate_phase_08_gates_3_5.py"]),
    ]
    tests = run_command([
        ".venv/bin/python", "-m", "unittest", "discover", "-v",
        "-s", "tests", "-p", "test_phase08*.py",
    ])

    pipeline = Phase8Pipeline()
    core_cases: list[dict[str, Any]] = []
    for case_id, question, expected_status, expected_pattern in (
        ("fact", "2019年飞亚达营业收入是多少？", "ok", "single_node"),
        ("formula", "2019年飞亚达营业收入增长率是多少？", "ok", "single_node"),
        ("sql_rank", "2019年所有公司中营业收入最高的公司是谁？", "partial", "single_node"),
        ("concern_compare", "飞亚达2019年营业收入和归母净利润哪个更高？", "ok", "parallel_concerns"),
        ("dynamic_target", "2019年收入最高的是哪家公司，为什么？", "unsupported", None),
        (
            "company_limit",
            "比较飞亚达、东阿阿胶、金岭矿业、中信特钢、冀东装备和厦门港务2019年营业收入。",
            "unsupported",
            None,
        ),
        (
            "two_dimensional",
            "比较飞亚达和东阿阿胶2018、2019年营业收入。",
            "unsupported",
            None,
        ),
    ):
        run = pipeline.run(qa_request(f"gate8-{case_id}", question))
        core_cases.append({
            "case_id": case_id,
            "status": run.answer["status"],
            "pattern_id": run.answer["composition_pattern_id"],
            "failure_codes": [row.get("failure_code") for row in run.answer["errors"]],
            "backend_call_count": run.trace["composition_decision"]["backend_call_count"],
            "trace_hash": run.trace["trace_hash"],
            "numeric_authorization_count": len(run.trace["numeric_authorization_set"]["items"]),
            "passed": (
                run.answer["status"] == expected_status
                and run.answer["composition_pattern_id"] == expected_pattern
                and (
                    case_id not in {"dynamic_target", "company_limit", "two_dimensional"}
                    or run.trace["composition_decision"]["backend_call_count"] == 0
                )
            ),
        })

    formula = next(row for row in core_cases if row["case_id"] == "formula")
    rank = next(row for row in core_cases if row["case_id"] == "sql_rank")
    # Release the in-process repository/candidate-index snapshot before the
    # real BGE worker loads model weights in a child process.
    del pipeline
    gc.collect()
    real_paths = [
        ROOT / "runs/phase_08/reports/real_evidence_smoke_run_1.json",
        ROOT / "runs/phase_08/reports/real_evidence_smoke_run_2.json",
    ]
    model_cache = Path(os.environ.get("FINGLMQA_MODEL_CACHE", ROOT.parent / "models")).resolve()
    real_commands = [real_smoke_command(path, model_cache) for path in real_paths]
    real_command_runs = [run_command(command) for command in real_commands]
    if any(row["return_code"] != 0 for row in real_command_runs):
        print(json.dumps({
            "all_checks_passed": False,
            "real_smoke_return_codes": [row["return_code"] for row in real_command_runs],
        }, ensure_ascii=False, indent=2))
        return 1
    real_runs = [json.loads(path.read_text(encoding="utf-8")) for path in real_paths]
    answer_and_trace = [
        {"answer": row["answer"], "trace": row["trace"]} for row in real_runs
    ]
    dynamic_trace_keys = {
        "started_at", "started_at_utc", "finished_at", "finished_at_utc",
        "elapsed_seconds", "process_id", "pid", "device", "temporary_path",
    }
    trace_dynamic_keys = sorted(
        key for key in dynamic_trace_keys if any(tree_has_key(row["trace"], key) for row in real_runs)
    )
    official_strings = [text for row in answer_and_trace for text in tree_strings(row)]
    absolute_official_paths = sorted({text for text in official_strings if text.startswith("/")})
    leaked_raw_financial_renderings = sorted({
        token for token in ("370,421.07万元", "8.93%", "3,704,210,734.90")
        if any(token in text for text in official_strings)
    })
    real_repeatability = {
        "statuses": [row["answer"]["status"] for row in real_runs],
        "trace_byte_identical": canonical_json_bytes(real_runs[0]["trace"]) == canonical_json_bytes(real_runs[1]["trace"]),
        "answer_byte_identical": canonical_json_bytes(real_runs[0]["answer"]) == canonical_json_bytes(real_runs[1]["answer"]),
        "telemetry_distinct": real_runs[0]["telemetry"] != real_runs[1]["telemetry"],
        "trace_hash": real_runs[0]["trace"]["trace_hash"],
        "provider_fingerprint": real_runs[0]["trace"]["artifact_fingerprints"].get("evidence_provider"),
        "retrieved_chunk_ids": real_runs[0]["trace"]["subplan_traces"][0]["trace"].get("retrieved_chunk_ids", []),
        "all_sources_portable": all(
            not str(citation["provenance"].get("source_markdown", "")).startswith("/")
            for citation in real_runs[0]["answer"]["citations"]
        ),
        "real_commands": [
            {
                "python": ".venv/bin/python",
                "script": "scripts/query_phase_08.py",
                "question_sha256": hashlib.sha256("飞亚达2019年面临哪些经营风险？".encode()).hexdigest(),
                "device": "cpu",
                "model_cache": "external:models",
                "output": path.relative_to(ROOT).as_posix(),
                "return_code": run["return_code"],
            }
            for path, run in zip(real_paths, real_command_runs)
        ],
        "source_fingerprints": {
            name: sha256_file(ROOT / path)
            for name, path in {
                "evidence_provider": "src/finglmqa/evidence_provider.py",
                "evidence_executor": "src/finglmqa/evidence_executor.py",
                "pipeline": "src/finglmqa/pipeline.py",
                "phase7_evidence_chunks": "data/corpus_package/evidence_chunks.jsonl",
                "phase7_index_manifest": "data/indexes/a2rag_index/index_manifest.json",
            }.items()
        },
        "raw_content_keys_in_official_artifacts": any(
            tree_has_key(row, "content") for row in answer_and_trace
        ),
        "absolute_paths_in_official_artifacts": absolute_official_paths,
        "dynamic_trace_keys": trace_dynamic_keys,
        "leaked_raw_financial_renderings": leaked_raw_financial_renderings,
    }
    phase6_after = {path: sha256_file(ROOT / path) for path in PHASE6_PINS}
    reports = {
        "gate1": json.loads((ROOT / "runs/phase_08/gate1_contract_report.json").read_text(encoding="utf-8")),
        "gate2": json.loads((ROOT / "runs/phase_08/gate2_report.json").read_text(encoding="utf-8")),
        "gates3_5": json.loads((ROOT / "runs/phase_08/gates_3_5_report.json").read_text(encoding="utf-8")),
    }
    checks = {
        "gate1_2_3_5_revalidation_passed": all(row["return_code"] == 0 for row in gate_runs)
        and all(row["all_checks_passed"] for row in reports.values()),
        "all_phase8_unit_tests_passed": tests["return_code"] == 0,
        "core_pipeline_matrix_passed": all(row["passed"] for row in core_cases),
        "formula_authorizes_result_and_two_operands": formula["numeric_authorization_count"] == 3,
        "sql_authorization_v2_present": rank["numeric_authorization_count"] >= 1,
        "real_evidence_two_runs_ok": real_repeatability["statuses"] == ["ok", "ok"],
        "real_answer_and_trace_byte_identical": real_repeatability["trace_byte_identical"]
        and real_repeatability["answer_byte_identical"],
        "runtime_telemetry_excluded": real_repeatability["telemetry_distinct"],
        "real_evidence_sources_portable": real_repeatability["all_sources_portable"],
        "real_evidence_has_claim_citations": bool(real_runs[0]["answer"]["citations"])
        and bool(real_repeatability["retrieved_chunk_ids"]),
        "real_smoke_bound_to_current_sources": all(
            row["return_code"] == 0 for row in real_command_runs
        ) and real_repeatability["provider_fingerprint"]
        == real_runs[1]["trace"]["artifact_fingerprints"].get("evidence_provider"),
        "official_artifacts_exclude_raw_chunk_content": not real_repeatability[
            "raw_content_keys_in_official_artifacts"
        ],
        "official_artifacts_have_no_host_absolute_paths": not real_repeatability[
            "absolute_paths_in_official_artifacts"
        ],
        "deterministic_trace_excludes_runtime_fields": not real_repeatability[
            "dynamic_trace_keys"
        ],
        "real_empty_authorization_has_no_raw_financial_leak": not real_repeatability[
            "leaked_raw_financial_renderings"
        ] and all(not row["trace"]["numeric_authorization_set"]["items"] for row in real_runs),
        "phase6_hashes_pinned_and_unchanged": phase6_before == PHASE6_PINS == phase6_after,
    }
    report = {
        "schema_version": "finglmqa.phase8.gates_6_8_report.v1",
        "checks": checks,
        "core_cases": core_cases,
        "real_evidence_repeatability": real_repeatability,
        "phase6_hashes_before": phase6_before,
        "phase6_hashes_after": phase6_after,
        "upstream_gate_runs": gate_runs,
        "unit_test_run": tests,
        "report_fingerprints": {
            name: sha256_file(ROOT / path)
            for name, path in {
                "gate1": "runs/phase_08/gate1_contract_report.json",
                "gate2": "runs/phase_08/gate2_report.json",
                "gates3_5": "runs/phase_08/gates_3_5_report.json",
            }.items()
        },
    }
    report["all_checks_passed"] = all(checks.values())
    atomic_json(ROOT / "runs/phase_08/gates_6_8_report.json", report)
    print(json.dumps({"all_checks_passed": report["all_checks_passed"], "checks": checks}, ensure_ascii=False, indent=2))
    return 0 if report["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
