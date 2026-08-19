#!/usr/bin/env python3
"""Gate 6/7 HTTP evaluation for the isolated Phase 10 service.

The evaluator deliberately observes only the public HTTP surface.  Planning
projections are reconstructed from the persisted deterministic trace and
compared with the frozen Phase 8 oracle.  The eight General rows whose gold
resolver is explicitly synthetic are still sent through HTTP, but are audited
as production-safe resolver failures rather than compared to a fictional
production corpus.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping

import httpx


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from finglmqa.contracts import canonical_json_bytes, semantic_sha256, validate_qa_trace  # noqa: E402
from finglmqa.service_contracts import validate_service_projection  # noqa: E402
from validate_phase_08_gate2 import analysis_projection, plan_projection, scope_projection  # noqa: E402


RUN = ROOT / "runs/phase_10"
SYNTHETIC_GENERAL_IDS = frozenset({"G05", "G07", "G08", "G26", "G27", "G28", "G29", "G30"})
SERVICE_ERROR_CODES = frozenset({
    "SERVICE_PAYLOAD_INVALID", "SERVICE_PAYLOAD_TOO_LARGE", "SERVICE_QUEUE_FULL",
    "SERVICE_NOT_READY", "SERVICE_TIMEOUT", "SERVICE_WORKER_RESTARTED",
})


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


def request(case_id: str, question: str) -> dict[str, Any]:
    return {
        "schema_version": "finglmqa.phase8.qa_request.v1",
        "request_id": case_id.replace(":", "_"),
        "question": question,
        "locale": "zh-CN",
        "trace_delivery": "reference",
    }


def terminal_projection(response: Mapping[str, Any], trace: Mapping[str, Any]) -> dict[str, Any] | None:
    if trace["composition_plan"] is not None:
        return None
    errors = response["errors"]
    if not errors:
        return None
    error = errors[0]
    return {
        "status": response["status"],
        "failure_code": error["failure_code"],
        "message": error["message"],
        "details": error["details"],
        "subplans": [],
        "backend_call_count": trace["composition_decision"]["backend_call_count"],
    }


def planning_projection(response: Mapping[str, Any], trace: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "analysis": analysis_projection(trace["question_analysis"]),
        "scope": scope_projection(trace["scope_plan"]),
        "plan": plan_projection(trace["composition_plan"]) if trace["composition_plan"] is not None else None,
        "terminal": terminal_projection(response, trace),
        # This is the planning oracle's contract, not the later execution count.
        "backend_call_count": 0,
    }


def post(client: httpx.Client, payload: Mapping[str, Any]) -> tuple[bytes, dict[str, Any], dict[str, Any]]:
    response = client.post("/v1/qa", content=canonical_json_bytes(payload), headers={"content-type": "application/json"})
    if response.status_code != 200:
        raise RuntimeError(f"service HTTP failure: {response.status_code}")
    raw = response.content
    if not raw.endswith(b"\n"):
        raise RuntimeError("service response is not canonical newline-terminated JSON")
    projection = validate_service_projection(response.json())
    reference = projection["demo_trace"]["trace_reference"]
    trace_response = client.get(reference)
    if trace_response.status_code != 200:
        raise RuntimeError("trace reference could not be dereferenced")
    trace = validate_qa_trace(trace_response.json())
    if hashlib.sha256(trace_response.content).hexdigest() != projection["demo_trace"]["trace_file_sha256"]:
        raise RuntimeError("trace_file_sha256 does not match delivered trace bytes")
    return raw, projection, trace


def exact_pin_ok(projection: Mapping[str, Any], fixture: Mapping[str, Any]) -> bool:
    expected = str(fixture["answer_prompt"]["prom_answer"])
    rendered = projection["answer"]
    number = re.compile(r"[-+]?\d+(?:\.\d+)?")
    expected_match = number.search(expected.replace(",", ""))
    rendered_match = number.search(rendered.replace(",", ""))
    return bool(
        projection["status"] == "ok"
        and expected_match and rendered_match
        and Decimal(expected_match.group()) == Decimal(rendered_match.group())
    )


def select_repeatability(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    buckets = (("1", 10), ("2-1", 5), ("3-1", 10))
    for benchmark_type, count in buckets:
        selected.extend([
            row for row in rows if row["source"]["benchmark_type"] == benchmark_type
        ][:count])
    if len(selected) != 25:
        raise RuntimeError("could not select the frozen 25-case repeatability set")
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8010")
    parser.add_argument("--timeout", type=float, default=130.0)
    parser.add_argument("--skip-full", action="store_true", help="run Gate 6 pins/repeatability only")
    args = parser.parse_args()

    benchmark = read_jsonl(ROOT / "runs/phase_08/benchmark_decomposition_oracle.jsonl")
    general = read_jsonl(ROOT / "runs/phase_08/general_decomposition_gold.jsonl")
    fixtures = json.loads((ROOT / "runs/phase_08/supported_fixture_manifest.json").read_text(encoding="utf-8"))
    fixture_by_uid = {
        row["uid"]: row for group in (fixtures["type1_exact"], fixtures["formula_exact"]) for row in group
    }
    oracle_by_uid = {row["source"]["uid"]: row for row in benchmark}

    output_rows: list[dict[str, Any]] = []
    deviations: list[dict[str, Any]] = []
    schema_valid = 0
    production_oracle_pass = 0
    synthetic_safe = 0
    status_counts: Counter[str] = Counter()
    pattern_counts: Counter[str] = Counter()

    with httpx.Client(base_url=args.base_url, timeout=args.timeout) as client:
        ready = client.get("/health/ready")
        if ready.status_code != 200 or ready.json().get("ready") is not True:
            raise RuntimeError("Phase 10 service is not ready")

        pin_rows: list[dict[str, Any]] = []
        for uid, fixture in fixture_by_uid.items():
            gold = oracle_by_uid[uid]
            _, projection, trace = post(client, request(gold["case_id"], gold["source"]["question"]))
            pin_rows.append({
                "uid": uid,
                "fixture_pin": gold["capability"]["fixture_pin"],
                "status": projection["status"],
                "passed": exact_pin_ok(projection, fixture),
                "trace_hash": trace["trace_hash"],
            })

        repeat_rows: list[dict[str, Any]] = []
        for gold in select_repeatability(benchmark):
            payload = request(gold["case_id"], gold["source"]["question"])
            first, first_projection, _ = post(client, payload)
            second, second_projection, _ = post(client, payload)
            repeat_rows.append({
                "case_id": gold["case_id"],
                "benchmark_type": gold["source"]["benchmark_type"],
                "byte_identical": first == second,
                "semantic_trace_hash": first_projection["demo_trace"]["semantic_trace_hash"],
                "trace_hash": second_projection["demo_trace"]["trace_hash"],
            })

        semantic_gold = benchmark[0]
        left_payload = request("phase10-semantic-a", semantic_gold["source"]["question"])
        right_payload = request("phase10-semantic-b", semantic_gold["source"]["question"])
        _, left, _ = post(client, left_payload)
        _, right, _ = post(client, right_payload)
        semantic_identity = {
            "trace_hash_differs": left["demo_trace"]["trace_hash"] != right["demo_trace"]["trace_hash"],
            "semantic_trace_hash_equal": (
                left["demo_trace"]["semantic_trace_hash"] == right["demo_trace"]["semantic_trace_hash"]
            ),
        }

        gate6_checks = {
            "nine_exact_fact_pins": sum(row["passed"] for row in pin_rows if row["fixture_pin"] == "type1_exact") == 9,
            "six_exact_formula_pins": sum(row["passed"] for row in pin_rows if row["fixture_pin"] == "formula_exact") == 6,
            "fixed_25_canonical_bodies_byte_identical": len(repeat_rows) == 25 and all(row["byte_identical"] for row in repeat_rows),
            "different_request_id_changes_trace_hash": semantic_identity["trace_hash_differs"],
            "different_request_id_preserves_semantic_hash": semantic_identity["semantic_trace_hash_equal"],
        }
        gate6 = {
            "schema_version": "finglmqa.phase10.gate_report.v1", "gate": 6,
            "status": "passed" if all(gate6_checks.values()) else "failed",
            "checks": gate6_checks, "pins": pin_rows, "repeatability": repeat_rows,
            "semantic_identity": semantic_identity,
        }
        write_json(RUN / "gate6_http_report.json", gate6)
        if args.skip_full:
            print(json.dumps({"gate6": gate6["status"]}, ensure_ascii=False, sort_keys=True))
            return 0 if gate6["status"] == "passed" else 1

        for gold in benchmark:
            payload = request(gold["case_id"], gold["source"]["question"])
            _, projection, trace = post(client, payload)
            observed = planning_projection(projection, trace)
            expected = gold["expected_planning_projection"]
            oracle_match = observed == expected
            if oracle_match:
                production_oracle_pass += 1
            else:
                deviations.append({
                    "case_id": gold["case_id"], "kind": "benchmark_planning",
                    "observed_sha256": semantic_sha256(observed),
                    "expected_sha256": semantic_sha256(expected),
                })
            schema_valid += 1
            status_counts[projection["status"]] += 1
            pattern_counts[str(projection["demo_trace"]["composition_pattern_id"])] += 1
            output_rows.append({
                "case_id": gold["case_id"], "kind": "benchmark", "request": payload,
                "response": projection, "oracle_match": oracle_match,
            })

        for gold in general:
            payload = request(gold["case_id"], gold["question"])
            _, projection, trace = post(client, payload)
            observed = planning_projection(projection, trace)
            expected = gold["expected_planning_projection"]
            if gold["case_id"] in SYNTHETIC_GENERAL_IDS:
                codes = {row["failure_code"] for row in projection["errors"]}
                safe = not codes & SERVICE_ERROR_CODES and projection["status"] in {
                    "ok", "partial", "not_found", "needs_clarification", "unsupported", "fallback_required",
                }
                synthetic_safe += int(safe)
                oracle_match: bool | None = None
                if not safe:
                    deviations.append({"case_id": gold["case_id"], "kind": "synthetic_fixture_not_safe"})
            else:
                oracle_match = observed == expected
                production_oracle_pass += int(oracle_match)
                if not oracle_match:
                    deviations.append({
                        "case_id": gold["case_id"], "kind": "general_planning",
                        "observed_sha256": semantic_sha256(observed),
                        "expected_sha256": semantic_sha256(expected),
                    })
            schema_valid += 1
            status_counts[projection["status"]] += 1
            pattern_counts[str(projection["demo_trace"]["composition_pattern_id"])] += 1
            output_rows.append({
                "case_id": gold["case_id"], "kind": "general", "request": payload,
                "response": projection, "oracle_match": oracle_match,
                "fixture_layer": gold.get("fixture_layer"),
            })

    write_jsonl(RUN / "http_evaluation.jsonl", output_rows)
    write_jsonl(RUN / "reports/http_oracle_deviations.jsonl", deviations)
    expected_production = 1003 + (40 - len(SYNTHETIC_GENERAL_IDS))
    gate7_checks = {
        "benchmark_1003_executed": len([row for row in output_rows if row["kind"] == "benchmark"]) == 1003,
        "general_40_executed": len([row for row in output_rows if row["kind"] == "general"]) == 40,
        "all_1043_schema_valid": schema_valid == 1043,
        "production_oracle_rows_match": production_oracle_pass == expected_production,
        "synthetic_fixture_rows_fail_safely": synthetic_safe == len(SYNTHETIC_GENERAL_IDS),
        "no_service_errors": not any(
            error["failure_code"] in SERVICE_ERROR_CODES
            for row in output_rows for error in row["response"]["errors"]
        ),
        "no_unexplained_internal_error": not any(
            error["failure_code"] == "INTERNAL_ERROR"
            for row in output_rows for error in row["response"]["errors"]
        ),
        "no_provenance_leak": not any(
            error["failure_code"] == "PROVENANCE_VALIDATION_FAILED"
            for row in output_rows for error in row["response"]["errors"]
        ),
        "output_preserves_oracle_order": (
            [row["case_id"] for row in output_rows]
            == [row["case_id"] for row in benchmark] + [row["case_id"] for row in general]
        ),
    }
    gate7 = {
        "schema_version": "finglmqa.phase10.gate_report.v1", "gate": 7,
        "status": "passed" if all(gate7_checks.values()) else "failed",
        "checks": gate7_checks,
        "counts": {
            "schema_valid": schema_valid, "production_oracle_pass": production_oracle_pass,
            "production_oracle_total": expected_production, "synthetic_fixture_safe": synthetic_safe,
            "synthetic_fixture_total": len(SYNTHETIC_GENERAL_IDS), "deviations": len(deviations),
            "statuses": dict(sorted(status_counts.items())), "patterns": dict(sorted(pattern_counts.items())),
        },
        "artifacts": {
            "evaluation": "runs/phase_10/http_evaluation.jsonl",
            "evaluation_sha256": hashlib.sha256((RUN / "http_evaluation.jsonl").read_bytes()).hexdigest(),
            "deviations": "runs/phase_10/reports/http_oracle_deviations.jsonl",
        },
    }
    write_json(RUN / "gate7_http_report.json", gate7)
    print(json.dumps({"gate6": gate6["status"], "gate7": gate7["status"], "counts": gate7["counts"]}, ensure_ascii=False, sort_keys=True))
    return 0 if gate6["status"] == gate7["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
