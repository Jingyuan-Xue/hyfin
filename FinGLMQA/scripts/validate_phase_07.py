#!/usr/bin/env python3
"""Run independent evidence, structural, and regression-pin Phase 7 checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import selectors
import subprocess
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from query_type3_evidence import CompanyYearResolver, Type3EvidenceRetriever, portable_path


SCHEMA_REPORT = "finglmqa.phase7.type3_evidence_report.v2"
VALIDATOR_VERSION = "phase7-type3-evidence-validator-v2"
PHASE6_REGRESSION_PINS = {
    "financial_facts_duckdb": "b3e8fed65ddc1ccd5954083a4df64f3eab2150294cae08a11424f3bc5744f278",
    "financial_facts_jsonl": "abeb4b3b221aac74705b84c80469c03b23fd8638d67004c75dd7a512c6841405",
}
FLYADA_REVENUE_REGRESSION_FIXTURE = {
    "name": "flyada_2019_revenue_phase6_regression_pin",
    "canonical_metric": "营业收入",
    "metric_year": 2019,
    "normalized_value": "3704210734.9",
    "normalized_unit": "元",
}
HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s*(.*?)\s*#*\s*$")


def workspace_root() -> Path:
    return Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_state(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"sha256": sha256_file(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")
    tmp.replace(path)


def compact_query_result(result: dict[str, Any]) -> dict[str, Any]:
    copied = json.loads(json.dumps(result, ensure_ascii=False, default=str))
    retrieval = copied.get("retrieval")
    if retrieval:
        for chunk in retrieval.get("chunks") or []:
            content = str(chunk.get("content") or "")
            chunk["content_sha256"] = hashlib.sha256(content.encode("utf-8")).hexdigest()
            chunk["content_preview"] = content[:600]
            chunk.pop("content", None)
    return copied


def independent_text_form(value: str) -> str:
    """Independent validator canonicalization: NFKC + whitespace collapse."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value)).strip()


def independent_heading_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value.strip().strip("#").strip())
    return "".join(char for char in normalized if not char.isspace())


def parse_sections_independently(source_text: str) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    lines = source_text.splitlines(keepends=True)
    headings: list[dict[str, Any]] = []
    offset = 0
    for line_no, line in enumerate(lines, start=1):
        match = HEADING_RE.match(line.rstrip("\r\n"))
        if match:
            label = match.group(2).strip().rstrip("#").strip()
            headings.append({
                "line": line_no,
                "level": len(match.group(1)),
                "label": label,
                "key": independent_heading_key(label),
                "body_char_start": offset + len(line),
            })
        offset += len(line)
    for index, heading in enumerate(headings):
        heading["end_line"] = len(source_text.splitlines())
        heading["end_char"] = len(source_text)
        for later in headings[index + 1 :]:
            if later["level"] <= heading["level"]:
                heading["end_line"] = later["line"] - 1
                heading["end_char"] = later["body_char_start"] - len(lines[later["line"] - 1])
                break
    return headings, {row["line"]: row for row in headings}


def independent_markdown_trace_audit(root: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    source_cache: dict[str, tuple[str, list[dict[str, Any]], dict[int, dict[str, Any]]]] = {}
    failures: list[dict[str, Any]] = []
    reason_counts: dict[str, int] = {}
    checked = 0
    failed_rows = 0
    for row in rows:
        source_ref = str(row["source_markdown"])
        if source_ref not in source_cache:
            source_path = Path(source_ref)
            if not source_path.is_absolute():
                source_path = root / source_path
            source_text = source_path.read_text(encoding="utf-8", errors="replace")
            headings, by_line = parse_sections_independently(source_text)
            source_cache[source_ref] = (source_text, headings, by_line)
        source_text, headings, by_line = source_cache[source_ref]
        checked += 1
        provenance = row.get("provenance") or {}
        heading_line = int(provenance.get("section_heading_line") or 0)
        section = by_line.get(heading_line)
        start_line, end_line = row.get("line_range") or [0, 0]
        expected_key = independent_heading_key(str((row.get("section_path") or [""])[-1]))
        reasons: list[str] = []
        if section is None:
            reasons.append("stored_heading_line_is_not_markdown_heading")
        else:
            if section["key"] != expected_key:
                reasons.append("heading_metadata_incompatible")
            if not (section["line"] <= start_line <= end_line <= section["end_line"]):
                reasons.append("line_range_outside_section")
        char_start = int(provenance.get("source_character_start") or -1)
        char_end = int(provenance.get("source_character_end") or -1)
        if not (0 <= char_start < char_end <= len(source_text)):
            reasons.append("invalid_character_offsets")
        elif independent_text_form(source_text[char_start:char_end]) != independent_text_form(str(row["content"])):
            reasons.append("content_differs_under_independent_canonicalization")

        compatible_count = 0
        if expected_key:
            canonical_content = independent_text_form(str(row["content"]))
            for candidate_section in headings:
                if candidate_section["key"] != expected_key:
                    continue
                section_body = independent_text_form(
                    source_text[candidate_section["body_char_start"] : candidate_section["end_char"]]
                )
                compatible_count += section_body.count(canonical_content)
            if compatible_count != 1:
                reasons.append("independent_section_match_not_unique")
        if reasons:
            failed_rows += 1
            for reason in reasons:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
            if len(failures) < 100:
                failures.append({
                    "evidence_chunk_id": row["evidence_chunk_id"],
                    "document_id": row["document_id"],
                    "line_range": row["line_range"],
                    "section_path": row["section_path"],
                    "section_heading_line": heading_line,
                    "independent_compatible_match_count": compatible_count,
                    "reasons": reasons,
                })
    return {
        "proof_scope": "All emitted evidence rows; independently reparses Markdown headings and section ends, checks stored character slice under NFKC+collapsed whitespace, and rejects repeated compatible content.",
        "checked": checked,
        "passed": checked - failed_rows,
        "rows_with_failures": failed_rows,
        "reason_counts": dict(sorted(reason_counts.items())),
        "failure_samples": failures,
        "all_passed": not reason_counts,
    }


def known_alignment_regressions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {row["evidence_chunk_id"]: row for row in rows}
    baichuan = by_id.get("chunk-535c9be327c67fb5fe7d98ca053b6711")
    rejected_culprit = "chunk-f67d578ac1b26138760702f381ffcf88" not in by_id
    guancheng_checkbox = by_id.get("chunk-58712985e129f960737945f99563e4b4")
    guancheng_equity = by_id.get("chunk-b6ff3f60e7ba500ac5e4da4f93fdd0b6")
    checks = {
        "baichuan_heading_692_content_694": bool(
            baichuan
            and baichuan["line_range"] == [694, 694]
            and baichuan["provenance"].get("section_heading_line") == 692
        ),
        "guancheng_cross_section_false_match_rejected": rejected_culprit,
        "guancheng_checkbox_restored_to_2170": bool(
            guancheng_checkbox and guancheng_checkbox["line_range"] == [2170, 2170]
        ),
        "guancheng_equity_restored_without_global_fallback": bool(
            guancheng_equity
            and guancheng_equity["line_range"] == [2174, 2176]
            and guancheng_equity["provenance"].get("alignment_method") == "section_constrained_unique_exact"
        ),
    }
    return {"checks": checks, "all_passed": all(checks.values())}


def read_json_line_with_timeout(stream: Any, timeout_seconds: float) -> dict[str, Any]:
    selector = selectors.DefaultSelector()
    selector.register(stream, selectors.EVENT_READ)
    try:
        events = selector.select(timeout_seconds)
        if not events:
            raise TimeoutError(f"worker produced no JSONL message within {timeout_seconds}s")
        line = stream.readline()
        if not line:
            raise EOFError("worker stdout closed before a response")
        return json.loads(line)
    finally:
        selector.close()


def fresh_worker_query(
    root: Path,
    device: str,
    model_cache: Path | None,
    request: dict[str, Any],
    timeout_seconds: float,
) -> dict[str, Any]:
    command = [
        (root / "refs/a2rag_runtime/.venv/bin/python").as_posix(),
        (root / "scripts/query_type3_evidence.py").as_posix(),
        "--serve",
        "--device",
        device,
    ]
    if model_cache is not None:
        command.extend(["--model-cache", model_cache.as_posix()])
    process = subprocess.Popen(
        command,
        cwd=root,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    started = time.monotonic()
    messages: list[dict[str, Any]] = []
    try:
        assert process.stdout is not None and process.stdin is not None
        ready = read_json_line_with_timeout(process.stdout, timeout_seconds)
        messages.append(ready)
        if ready.get("type") != "ready":
            raise RuntimeError(f"worker did not emit READY first: {ready}")
        process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
        process.stdin.flush()
        response = read_json_line_with_timeout(process.stdout, timeout_seconds)
        messages.append(response)
        if response.get("type") != "result" or response.get("request_id") != request["request_id"]:
            raise RuntimeError(f"worker query response invalid: {response}")
        shutdown_id = f"shutdown-{request['request_id']}"
        process.stdin.write(json.dumps({"type": "shutdown", "request_id": shutdown_id}) + "\n")
        process.stdin.flush()
        shutdown = read_json_line_with_timeout(process.stdout, timeout_seconds)
        messages.append(shutdown)
        if shutdown.get("type") != "shutdown_ack" or shutdown.get("request_id") != shutdown_id:
            raise RuntimeError(f"worker shutdown acknowledgement invalid: {shutdown}")
        process.wait(timeout=timeout_seconds)
        if process.returncode != 0:
            raise RuntimeError(f"worker exit={process.returncode}")
        return {
            "status": "ok",
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "messages": messages,
            "result": response["result"],
            "cleanup": "shutdown_ack_then_exit_0",
        }
    except Exception:
        process.kill()
        process.wait(timeout=10)
        stderr = process.stderr.read() if process.stderr else ""
        raise RuntimeError(f"fresh worker failed; stderr tail={stderr[-3000:]}")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)


def main() -> int:
    root = workspace_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=root / "runs/phase_07/type3_evidence_report.json")
    parser.add_argument("--device", default=os.environ.get("FINGLMQA_DEVICE", "auto"))
    parser.add_argument("--model-cache", type=Path)
    parser.add_argument("--worker-timeout", type=float, default=180.0)
    args = parser.parse_args()

    database = root / "data/facts/financial_facts.duckdb"
    facts_jsonl = root / "data/facts/financial_facts.jsonl"
    phase6_before = {"financial_facts_duckdb": file_state(database), "financial_facts_jsonl": file_state(facts_jsonl)}
    pin_checks = {
        name: phase6_before[name]["sha256"] == expected
        for name, expected in PHASE6_REGRESSION_PINS.items()
    }
    pin_message = (
        "Phase 6 immutable baseline matches the named regression pins."
        if all(pin_checks.values())
        else "REGRESSION PIN MISMATCH: Phase 6 inputs differ from the audited baseline; this may be an intentionally refreshed pin, but Phase 7 validation must stop being treated as a data-integrity pass until the baseline is separately audited and the named pins are updated."
    )

    retriever = Type3EvidenceRetriever(
        root=root,
        device=args.device,
        model_cache=args.model_cache,
        load_dense=True,
    )
    rows = retriever.evidence_rows
    expected_documents = len(json.loads((root / "data/corpus_package/corpus_manifest.json").read_text(encoding="utf-8"))["documents"])
    required_fields = {
        "document_id", "company_name", "stock_code", "report_year", "section_path",
        "semantic_tags", "line_range", "source_markdown", "content",
    }
    evidence_ids = [row["evidence_chunk_id"] for row in rows]
    document_ids = {row["document_id"] for row in rows}
    schema_missing = [row.get("evidence_chunk_id") for row in rows if not required_fields.issubset(row)]
    independent_trace = independent_markdown_trace_audit(root, rows)
    alignment_regressions = known_alignment_regressions(rows)

    qualitative_cases = [
        ("risk_flyada_2019", "飞亚达", 2019, "飞亚达2019年主要经营风险和应对措施是什么？", "risk"),
        ("rd_gigadevice_2019", "兆易创新", 2019, "兆易创新2019年在研发和技术创新方面有哪些重点进展？", "rd"),
        ("business_kailun_2020", "开润股份", 2020, "开润股份2020年的主要业务、产品和经营模式是什么？", "business"),
        ("staff_flyada_2019", "飞亚达", 2019, "飞亚达2019年的员工培训和人员发展情况如何？", "staff"),
    ]
    qualitative_results: list[dict[str, Any]] = []
    for case_id, company, year, question, expected_tag in qualitative_cases:
        result = retriever.query(company, year, question, top_k=5)
        chunks = result["retrieval"]["chunks"]
        resolved = result["resolver"]["document_id"]
        qualitative_results.append({
            "id": case_id,
            "expected_tag": expected_tag,
            "expected_tag_hit": any(expected_tag in chunk["semantic_tags"] for chunk in chunks),
            "company_year_isolation_passed": all(chunk["document_id"] == resolved for chunk in chunks),
            "result": compact_query_result(result),
        })

    missing_company = retriever.query("不存在的公司", 2019, "该公司的主要风险是什么？", top_k=3)
    concatenated_company = retriever.query("飞亚达冠城大通", 2019, "主要风险是什么？", top_k=3)
    wrong_year = retriever.query("飞亚达", 2020, "主要风险是什么？", top_k=3)
    stock_code_boundary = retriever.query("600067", 2019, "主要风险是什么？", top_k=3)
    synthetic_rows = [dict(row) for row in retriever.resolver_rows]
    duplicate = dict(synthetic_rows[0])
    duplicate["document_id"] += "_synthetic_duplicate"
    ambiguous_resolution = CompanyYearResolver([*synthetic_rows, duplicate]).resolve(
        str(synthetic_rows[0]["aliases"][0]), int(synthetic_rows[0]["report_year"])
    )
    adversarial_checks = {
        "missing_company_has_no_retrieval": missing_company["resolver"]["status"] == "missing" and missing_company["retrieval"] is None,
        "concatenated_alias_has_no_retrieval": concatenated_company["resolver"]["status"] == "missing" and concatenated_company["retrieval"] is None,
        "wrong_year_has_no_retrieval": wrong_year["resolver"]["status"] == "missing" and wrong_year["retrieval"] is None,
        "stock_code_resolves_only_expected_document": bool(
            stock_code_boundary["resolver"].get("document_id") == "A600067_冠城大通_2019年年度报告"
            and all(
                chunk["document_id"] == "A600067_冠城大通_2019年年度报告"
                for chunk in stock_code_boundary["retrieval"]["chunks"]
            )
        ),
        "synthetic_duplicate_fails_ambiguous": ambiguous_resolution["status"] == "ambiguous",
    }

    numeric_cases = {
        "flyada_revenue_fixture": retriever.query("飞亚达", 2019, "飞亚达2019年营业收入是多少？", top_k=3),
        "growth_formula_required": retriever.query("飞亚达", 2019, "飞亚达2019年营业收入增长率是多少？", top_k=3),
        "report_vs_metric_year": retriever.query("飞亚达", 2019, "2019年报披露的2018年营业收入是多少？", top_k=3),
        "multi_year_fail_closed": retriever.query("飞亚达", 2019, "2019年报中2017年和2018年营业收入分别是多少？", top_k=3),
        "unit_ambiguous_share_capital": retriever.query("再升科技", 2020, "再升科技2020年股本是多少？", top_k=3),
        "explicit_share_unit": retriever.query("再升科技", 2020, "再升科技2020年股本是多少股？", top_k=3),
        "two_profit_metrics": retriever.query("再升科技", 2020, "归母净利润和扣非净利润分别是多少？", top_k=3),
        "unrecognized_numeric": retriever.query("飞亚达", 2019, "飞亚达2019年的广告费用是多少？", top_k=3),
    }
    revenue_facts = numeric_cases["flyada_revenue_fixture"]["fact_injection"]["facts"]
    fixture = FLYADA_REVENUE_REGRESSION_FIXTURE
    flyada_fixture_passed = any(
        fact["canonical_metric"] == fixture["canonical_metric"]
        and fact["metric_year"] == fixture["metric_year"]
        and fact["normalized_value"] == fixture["normalized_value"]
        and fact["normalized_unit"] == fixture["normalized_unit"]
        for fact in revenue_facts
    )
    numeric_checks = {
        "growth_is_formula_required_without_level_fact": (
            numeric_cases["growth_formula_required"]["fact_injection"]["status"] == "formula_required"
            and not numeric_cases["growth_formula_required"]["fact_injection"]["facts"]
        ),
        "report_year_2019_metric_year_2018": (
            numeric_cases["report_vs_metric_year"]["fact_injection"]["status"] == "selected_facts_injected"
            and numeric_cases["report_vs_metric_year"]["fact_injection"]["year_policy"]["metric_year"] == 2018
            and all(fact["metric_year"] == 2018 for fact in numeric_cases["report_vs_metric_year"]["fact_injection"]["facts"])
        ),
        "multiple_metric_years_fail_closed": (
            numeric_cases["multi_year_fail_closed"]["fact_injection"]["status"] == "metric_year_ambiguous"
            and not numeric_cases["multi_year_fail_closed"]["fact_injection"]["facts"]
        ),
        "share_capital_without_unit_fails_ambiguous": (
            numeric_cases["unit_ambiguous_share_capital"]["fact_injection"]["status"] == "unit_ambiguous"
            and not numeric_cases["unit_ambiguous_share_capital"]["fact_injection"]["facts"]
        ),
        "share_capital_explicit_shares_selects_only_shares": (
            numeric_cases["explicit_share_unit"]["fact_injection"]["status"] == "selected_facts_injected"
            and numeric_cases["explicit_share_unit"]["fact_injection"]["facts"]
            and all(fact["normalized_unit"] == "股" for fact in numeric_cases["explicit_share_unit"]["fact_injection"]["facts"])
        ),
        "two_profit_metrics_recognized_and_injected": (
            numeric_cases["two_profit_metrics"]["fact_injection"]["status"] == "selected_facts_injected"
            and numeric_cases["two_profit_metrics"]["fact_injection"]["recognized_metrics"]
            == ["归属于上市公司股东的净利润", "扣除非经常性损益后的净利润"]
        ),
        "unrecognized_numeric_fails_closed": (
            numeric_cases["unrecognized_numeric"]["fact_injection"]["status"] == "metric_not_recognized"
            and not numeric_cases["unrecognized_numeric"]["fact_injection"]["facts"]
        ),
    }

    worker_request = {
        "type": "query",
        "request_id": "fresh-risk-query",
        "company": "飞亚达",
        "report_year": 2019,
        "question": "飞亚达2019年主要经营风险和应对措施是什么？",
        "top_k": 5,
    }
    fresh_workers = [
        fresh_worker_query(root, args.device, args.model_cache, worker_request, args.worker_timeout)
        for _ in range(2)
    ]
    rankings = [
        [
            [chunk["evidence_chunk_id"], chunk["score"]]
            for chunk in lifecycle["result"]["retrieval"]["chunks"]
        ]
        for lifecycle in fresh_workers
    ]
    fresh_worker_stable = rankings[0] == rankings[1]

    table_audit = json.loads((root / "runs/phase_07/reports/table_exclusion_audit.json").read_text(encoding="utf-8"))
    table_classification_count = sum(
        1 for line in (root / "runs/phase_07/reports/table_exclusion_classification.jsonl").open(encoding="utf-8") if line.strip()
    )
    document_chunk_sets = [set(row["chunk_ids"]) for row in retriever.document_rows]
    disjoint_chunk_allow_lists = sum(len(values) for values in document_chunk_sets) == len(set().union(*document_chunk_sets))
    absolute_evidence_paths = [
        row["evidence_chunk_id"] for row in rows
        if Path(str(row["source_markdown"])).is_absolute()
    ]
    global_fallback_count = sum(
        row["provenance"].get("alignment_method") == "global_unique_section_compatible_exact" for row in rows
    )

    phase6_after = {"financial_facts_duckdb": file_state(database), "financial_facts_jsonl": file_state(facts_jsonl)}
    phase6_unchanged = phase6_before == phase6_after

    structural_invariants = {
        "evidence_schema_complete": not schema_missing,
        "evidence_chunk_ids_unique": len(evidence_ids) == len(set(evidence_ids)),
        "document_coverage_matches_manifest": len(document_ids) == expected_documents,
        "document_allow_lists_are_disjoint": disjoint_chunk_allow_lists,
        "all_cursor_updates_monotonic": all(row["provenance"].get("cursor_monotonic") is True for row in rows),
        "all_successful_alignments_have_unique_section_match": all(
            row["provenance"].get("compatible_content_match_count") == 1 for row in rows
        ),
        "table_classification_covers_all_excluded_table_chunks": (
            table_classification_count == table_audit["classified_chunks"]
            == retriever.index_manifest["counts"]["excluded_by_reason"]["contains_table_html"]
        ),
        "generated_evidence_source_paths_are_workspace_relative": not absolute_evidence_paths,
        "no_unexplained_global_fallback": global_fallback_count == 0,
    }
    independent_evidence_checks = {
        "independent_markdown_trace_all_rows": independent_trace["all_passed"],
        "known_alignment_regressions": alignment_regressions["all_passed"],
        "qualitative_tag_hits": all(row["expected_tag_hit"] for row in qualitative_results),
        "qualitative_public_query_isolation": all(row["company_year_isolation_passed"] for row in qualitative_results),
        "adversarial_public_boundary_checks": all(adversarial_checks.values()),
        "numeric_policy_boundaries": all(numeric_checks.values()),
        "fresh_worker_lifecycles_same_top_ids_and_scores": fresh_worker_stable,
    }
    regression_pins = {
        "phase6_duckdb_sha256": pin_checks["financial_facts_duckdb"],
        "phase6_facts_jsonl_sha256": pin_checks["financial_facts_jsonl"],
        "phase6_files_unchanged_during_validation": phase6_unchanged,
        FLYADA_REVENUE_REGRESSION_FIXTURE["name"]: flyada_fixture_passed,
    }
    all_checks_passed = (
        all(independent_evidence_checks.values())
        and all(structural_invariants.values())
        and all(regression_pins.values())
    )

    reports_dir = args.output.parent / "reports"
    numeric_report_path = reports_dir / "numeric_boundary_cases.json"
    worker_report_path = reports_dir / "fresh_worker_repeatability.json"
    atomic_write_json(numeric_report_path, {
        "schema_version": "finglmqa.phase7.numeric_boundary_cases.v1",
        "checks": numeric_checks,
        "all_passed": all(numeric_checks.values()),
        "cases": {key: compact_query_result(value) for key, value in numeric_cases.items()},
    })
    atomic_write_json(worker_report_path, {
        "schema_version": "finglmqa.phase7.fresh_worker_repeatability.v1",
        "protocol": {
            "ready_first": True,
            "request_id_required": True,
            "stdout": "JSONL only",
            "stderr": "logs",
            "flush": "after every message",
            "timeout_seconds": args.worker_timeout,
            "cleanup": "shutdown/shutdown_ack/exit-0; timeout client kills process",
            "single_concurrency": True,
        },
        "lifecycles": [
            {
                "elapsed_seconds": row["elapsed_seconds"],
                "message_types": [message["type"] for message in row["messages"]],
                "cleanup": row["cleanup"],
            }
            for row in fresh_workers
        ],
        "rankings": rankings,
        "stable_top_ids_and_scores": fresh_worker_stable,
    })

    report = {
        "schema_version": SCHEMA_REPORT,
        "validator_version": VALIDATOR_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_model": {
            "note": "Evidence checks, structural invariants, and regression pins have different proof strength and are reported separately; there is no aggregate N/N claim.",
            "independent_evidence_checks": independent_evidence_checks,
            "structural_invariants": structural_invariants,
            "regression_pins": regression_pins,
        },
        "all_checks_passed": all_checks_passed,
        "index_counts": retriever.index_manifest["counts"],
        "independent_markdown_trace_audit": independent_trace,
        "known_alignment_regressions": alignment_regressions,
        "qualitative_retrieval_cases": qualitative_results,
        "adversarial_company_boundary": {
            "proof_scope": "Public resolver/query/worker boundary only. The retrieval-only adapter intentionally accepts a previously resolved document_id and is not itself a company-name resolver.",
            "checks": adversarial_checks,
            "cases": {
                "missing_company": compact_query_result(missing_company),
                "concatenated_company": compact_query_result(concatenated_company),
                "wrong_year": compact_query_result(wrong_year),
                "stock_code_boundary": compact_query_result(stock_code_boundary),
                "ambiguous_synthetic": ambiguous_resolution,
            },
        },
        "numeric_policy": {
            "checks": numeric_checks,
            "machine_readable_report": portable_path(numeric_report_path, root),
        },
        "fresh_process_repeatability": {
            "stable_top_ids_and_scores": fresh_worker_stable,
            "rankings": rankings,
            "machine_readable_report": portable_path(worker_report_path, root),
        },
        "phase6_immutable_baseline": {
            "expected_sha256": PHASE6_REGRESSION_PINS,
            "pin_checks": pin_checks,
            "pin_message": pin_message,
            "before": phase6_before,
            "after": phase6_after,
            "unchanged_during_validation": phase6_unchanged,
        },
        "regression_fixture_governance": {
            "fixture": FLYADA_REVENUE_REGRESSION_FIXTURE,
            "passed": flyada_fixture_passed,
            "stale_pin_message": "If a separately audited Phase 6 rebuild legitimately changes this value, update this named regression fixture and its decision record; a mismatch alone is not evidence of data corruption.",
        },
        "table_exclusion_audit": table_audit,
        "schema_audit": {
            "required_fields": sorted(required_fields),
            "evidence_rows": len(rows),
            "missing_required_field_rows": len(schema_missing),
            "missing_required_field_samples": schema_missing[:20],
            "document_count": len(document_ids),
            "expected_document_count_from_manifest": expected_documents,
            "absolute_source_path_rows": len(absolute_evidence_paths),
            "global_fallback_count": global_fallback_count,
        },
        "commands": {
            "build": "refs/a2rag_runtime/.venv/bin/python scripts/build_a2rag_text_index.py",
            "validate": "FINGLMQA_EMBEDDING_CACHE=<local-cache> refs/a2rag_runtime/.venv/bin/python scripts/validate_phase_07.py --device auto",
            "worker": "refs/a2rag_runtime/.venv/bin/python scripts/query_type3_evidence.py --serve --device auto",
        },
    }
    atomic_write_json(args.output, report)
    print(json.dumps({
        "all_checks_passed": all_checks_passed,
        "independent_evidence_checks": independent_evidence_checks,
        "structural_invariants": structural_invariants,
        "regression_pins": regression_pins,
        "report": args.output.as_posix(),
    }, ensure_ascii=False, indent=2))
    return 0 if all_checks_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
