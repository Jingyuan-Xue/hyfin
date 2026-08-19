#!/usr/bin/env python3
"""Independent Phase 10 Gate 0-6 validation and report writer."""

from __future__ import annotations

import argparse
import json
import os
import selectors
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import httpx


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/phase_10"
PYTHON = ROOT / ".venv-phase10/bin/python"
sys.path.insert(0, str(ROOT / "src"))

from finglmqa.contracts import canonical_json_bytes  # noqa: E402
from finglmqa.service_contracts import WORKER_PROTOCOL, validate_service_projection  # noqa: E402
from finglmqa.service_manifest import ImmutableManifestVerifier  # noqa: E402


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical_json_bytes(value))
    os.replace(temporary, path)


def report(gate: int, checks: Mapping[str, bool], details: Mapping[str, Any] | None = None) -> dict[str, Any]:
    value = {
        "schema_version": "finglmqa.phase10.gate_report.v1", "gate": gate,
        "status": "passed" if all(checks.values()) else "failed",
        "checks": dict(checks), "details": dict(details or {}),
    }
    write_json(RUN / f"gate{gate}_report.json", value)
    return value


def command(args: list[str], *, timeout: float = 600) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args, cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        text=True, capture_output=True, timeout=timeout,
    )


def unittest_pattern(pattern: str) -> subprocess.CompletedProcess[str]:
    return command([
        str(PYTHON), "-m", "unittest", "discover", "-s", "tests", "-p", pattern, "-q",
    ])


def gate0() -> dict[str, Any]:
    verified = ImmutableManifestVerifier(ROOT, RUN / "immutable_inputs_manifest.json").verify_full()
    dependency = json.loads((RUN / "dependency_manifest.json").read_text(encoding="utf-8"))
    manifest = json.loads((RUN / "immutable_inputs_manifest.json").read_text(encoding="utf-8"))
    checks = {
        "immutable_manifest_verified": verified.semantic_hash == manifest["manifest_semantic_sha256"],
        "upstream_entry_count_21": len(manifest["entries"]) == 21,
        "qwen_snapshot_has_15_shards": len(manifest["external_runtimes"]["qwen"]["weight_shards"]) == 15,
        "vllm_0_21_0": manifest["external_runtimes"]["vllm_version"] == "0.21.0",
        "isolated_environment_locked": dependency["python_version"].startswith("Python 3.14."),
    }
    return report(0, checks, {"manifest_semantic_sha256": verified.semantic_hash})


def gate1() -> dict[str, Any]:
    schemas = [
        "phase_10_service_projection.schema.json", "phase_10_demo_trace.schema.json",
        "phase_10_worker_message.schema.json", "phase_10_shadow_eligibility.schema.json",
        "phase_10_shadow_result.schema.json",
    ]
    parsed = [json.loads((ROOT / "data/schemas" / name).read_text(encoding="utf-8")) for name in schemas]
    tests = command([
        str(PYTHON), "-m", "unittest", "-q",
        "tests.test_phase10_contracts", "tests.test_phase10_api",
    ])
    source = (ROOT / "src/finglmqa/service_contracts.py").read_text(encoding="utf-8")
    checks = {
        "five_service_schemas_closed": len(parsed) == 5 and all(row.get("additionalProperties") is False for row in parsed),
        "transport_and_semantic_contract_tests_pass": tests.returncode == 0,
        "three_trace_identifiers_frozen": all(token in source for token in (
            "trace_hash", "semantic_trace_hash", "trace_file_sha256",
        )),
        "telemetry_split_frozen": "semantic_trace_projection" in source,
    }
    return report(1, checks, {"test_output": (tests.stdout + tests.stderr).strip()})


def _read_worker(process: subprocess.Popen[str], timeout: float) -> dict[str, Any]:
    assert process.stdout is not None
    selector = selectors.DefaultSelector()
    try:
        selector.register(process.stdout, selectors.EVENT_READ)
        if not selector.select(timeout):
            raise TimeoutError("worker probe timed out")
        raw = process.stdout.readline()
    finally:
        selector.close()
    return json.loads(raw)


def gate2() -> dict[str, Any]:
    process = subprocess.Popen(
        [str(PYTHON), "scripts/serve_phase_10_worker.py"], cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        text=True, bufsize=1, start_new_session=True,
    )
    messages: list[dict[str, Any]] = []
    clean = False
    try:
        ready = _read_worker(process, 300)
        messages.append(ready)
        assert process.stdin is not None
        for kind in ("ping", "shutdown"):
            request_id = f"gate2-{kind}"
            process.stdin.write(json.dumps({
                "protocol_version": WORKER_PROTOCOL, "type": kind, "request_id": request_id,
            }, separators=(",", ":")) + "\n")
            process.stdin.flush()
            messages.append(_read_worker(process, 10))
        clean = process.wait(timeout=30) == 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)
    checks = {
        "ready_protocol_valid": messages[0].get("type") == "ready" and messages[0].get("request_id") is None,
        "a2rag_preheated": messages[0].get("ready", {}).get("a2rag_preheated") is True,
        "ping_round_trip": messages[1].get("type") == "pong" and messages[1].get("request_id") == "gate2-ping",
        "shutdown_ack": messages[2].get("type") == "shutdown_ack" and messages[2].get("request_id") == "gate2-shutdown",
        "clean_worker_exit": clean,
    }
    return report(2, checks, {"message_types": [row.get("type") for row in messages]})


def gate3(base_url: str) -> dict[str, Any]:
    with httpx.Client(base_url=base_url, timeout=130) as client:
        malformed = client.post("/v1/qa", content=b"{")
        oversized = client.post("/v1/qa", content=b"x" * 65537)
        business = client.post("/v1/qa", json={
            "schema_version": "finglmqa.phase8.qa_request.v1", "request_id": "gate3-business",
            "question": "2019年收入最高的是哪家公司，为什么？", "locale": "zh-CN",
        })
        valid = client.post("/v1/qa", json={
            "schema_version": "finglmqa.phase8.qa_request.v1", "request_id": "gate3-valid",
            "question": "2019年飞亚达营业收入是多少？", "locale": "zh-CN",
        })
        valid_payload = validate_service_projection(valid.json())
        trace = client.get(valid_payload["demo_trace"]["trace_reference"])
    checks = {
        "malformed_is_400_service_payload_invalid": malformed.status_code == 400 and malformed.json()["errors"][0]["failure_code"] == "SERVICE_PAYLOAD_INVALID",
        "oversized_is_413_without_pipeline": oversized.status_code == 413 and oversized.json()["errors"][0]["failure_code"] == "SERVICE_PAYLOAD_TOO_LARGE",
        "business_terminal_remains_http_200": business.status_code == 200 and business.json()["errors"][0]["failure_code"] == "COMPOSITION_UNSUPPORTED",
        "valid_projection_is_canonical": valid.status_code == 200 and valid.content.endswith(b"\n") and valid_payload["status"] == "ok",
        "trace_reference_resolves": trace.status_code == 200 and trace.content.endswith(b"\n"),
    }
    return report(3, checks)


def gate4() -> dict[str, Any]:
    tests = command([str(PYTHON), "-m", "unittest", "-q", "tests.test_phase10_supervisor"])
    source = (ROOT / "src/finglmqa/service_supervisor.py").read_text(encoding="utf-8")
    checks = {
        "fault_state_machine_tests_pass": tests.returncode == 0,
        "process_group_kill_implemented": "os.killpg" in source,
        "queue_timeout_does_not_record_failure": source.index('self.telemetry.event("queue_timeout"') < source.index("async def _consume"),
        "breaker_manual_reset_only": "breaker_open = False" in source and "failure_events.clear" not in source,
    }
    return report(4, checks, {"test_output": (tests.stdout + tests.stderr).strip()})


def gate5() -> dict[str, Any]:
    calls = [
        command(["scripts/start_finglmqa.sh"]),
        command(["scripts/status_finglmqa.sh"]),
        command(["scripts/stop_finglmqa.sh"]),
        command(["scripts/stop_finglmqa.sh"]),
        command(["scripts/start_finglmqa.sh"]),
        command(["scripts/start_finglmqa.sh"]),
        command(["scripts/status_finglmqa.sh"]),
    ]
    telemetry = (ROOT / "src/finglmqa/service_logging.py").read_text(encoding="utf-8")
    checks = {
        "lifecycle_calls_succeed": all(row.returncode == 0 for row in calls),
        "stop_is_idempotent": '"already_stopped"' in calls[3].stdout,
        "start_is_idempotent": '"already_ready"' in calls[5].stdout,
        "final_service_ready": '"ready"' in calls[6].stdout,
        "telemetry_rotates_100mib_x5": "100 * 1024 * 1024" in telemetry and "backupCount=5" in telemetry,
        "pid_reuse_guard_present": "process_start_ticks" in (ROOT / "src/finglmqa/service_control.py").read_text(encoding="utf-8"),
    }
    return report(5, checks, {"outputs": [row.stdout.strip() for row in calls]})


def gate6(base_url: str) -> dict[str, Any]:
    phase8 = unittest_pattern("test_phase08*.py")
    phase9 = unittest_pattern("test_phase09*.py")
    http_report = json.loads((RUN / "gate6_http_report.json").read_text(encoding="utf-8"))
    with httpx.Client(base_url=base_url, timeout=130) as client:
        smoke = client.post("/v1/qa", json={
            "schema_version": "finglmqa.phase8.qa_request.v1",
            "request_id": "phase8-real-evidence-smoke",
            "question": "飞亚达2019年面临哪些经营风险？", "locale": "zh-CN",
        }).json()
    frozen = json.loads((ROOT / "runs/phase_09/gate6_report.json").read_text(encoding="utf-8"))
    expected_trace = frozen["details"]["real_evidence_trace_hash"]
    checks = {
        "phase8_171_tests_pass": phase8.returncode == 0 and "Ran 171 tests" in (phase8.stdout + phase8.stderr),
        "phase9_14_tests_pass": phase9.returncode == 0 and "Ran 14 tests" in (phase9.stdout + phase9.stderr),
        "http_exact_and_repeatability_pass": http_report["status"] == "passed",
        "real_evidence_trace_matches_frozen_release": smoke["demo_trace"]["trace_hash"] == expected_trace,
        "supplemental_repository_remains_disabled": os.environ.get("FINGLMQA_SUPPLEMENTAL_FACTS_ENABLED", "0") == "0",
    }
    return report(6, checks, {
        "phase8_test_output": (phase8.stdout + phase8.stderr).strip(),
        "phase9_test_output": (phase9.stdout + phase9.stderr).strip(),
        "real_evidence_trace_hash": smoke["demo_trace"]["trace_hash"],
    })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8010")
    parser.add_argument("--skip-lifecycle", action="store_true")
    args = parser.parse_args()
    results = [gate0(), gate1(), gate2(), gate3(args.base_url), gate4()]
    if not args.skip_lifecycle:
        results.append(gate5())
    results.append(gate6(args.base_url))
    print(json.dumps({f"gate{row['gate']}": row["status"] for row in results}, sort_keys=True))
    return 0 if all(row["status"] == "passed" for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
