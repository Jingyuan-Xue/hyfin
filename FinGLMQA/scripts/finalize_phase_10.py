#!/usr/bin/env python3
"""Finalize Gate 8, promotion readiness, release manifest, and report."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Mapping

import httpx


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/phase_10"
sys.path.insert(0, str(ROOT / "src"))

from finglmqa.contracts import canonical_json_bytes  # noqa: E402
from finglmqa.service_manifest import ImmutableManifestVerifier  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical_json_bytes(value))
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_has_key(value: Any, name: str) -> bool:
    if isinstance(value, dict):
        return name in value or any(tree_has_key(child, name) for child in value.values())
    if isinstance(value, list):
        return any(tree_has_key(child, name) for child in value)
    return False


def main() -> int:
    results = read_jsonl(RUN / "shadow_results.jsonl")
    eligibility = read_jsonl(RUN / "shadow_eligibility_oracle.jsonl")
    repeats = read_jsonl(RUN / "reports/shadow_repeatability.jsonl")
    audit = read_jsonl(RUN / "reports/manual_shadow_audit.jsonl")
    if any(row.get("review") is None for row in audit):
        raise RuntimeError("manual shadow audit has not been finalized")
    answerable_results = [row for row in results if row["eligibility"] == "answerable"]
    unanswerable_results = [row for row in results if row["eligibility"] != "answerable"]
    answerable_audit = [row for row in audit if row["eligibility"] == "answerable"]
    unanswerable_audit = [row for row in audit if row["eligibility"] != "answerable"]
    accepted = sum(row["generator_outcome"] == "accepted" for row in answerable_results)
    promotion_checks = {
        "answerable_acceptance_at_least_95pct": accepted / len(answerable_results) >= 0.95,
        "unanswerable_has_zero_accepted_claims": all(row["accepted_claim_count"] == 0 for row in unanswerable_results),
        "invalid_output_separately_counted": all(row["generator_outcome"] in {
            "accepted", "generator_refused", "generator_invalid_output", "generator_rejected_by_gate",
        } for row in results),
        "fixed_30_accepted_projection_repeatable": len(repeats) == 30 and all(row["byte_identical"] for row in repeats),
        "manual_answerable_relevance_at_least_90pct": (
            sum(row["review"]["relevant"] and row["review"]["citation_sufficient"] for row in answerable_audit)
            / len(answerable_audit) >= 0.90
        ),
        "manual_unanswerable_has_no_unsupported_output": all(
            not row["review"]["unsupported_output"] for row in unanswerable_audit
        ),
    }
    promotion_ready = all(promotion_checks.values())
    promotion = {
        "schema_version": "finglmqa.phase10.promotion_readiness.v1",
        "promotion_readiness": promotion_ready,
        "checks": promotion_checks,
        "reasons": [key for key, passed in promotion_checks.items() if not passed],
    }
    write_json(RUN / "promotion_readiness.json", promotion)

    with httpx.Client(base_url="http://127.0.0.1:8010", timeout=130) as client:
        ready_response = client.get("/health/ready")
        official = client.post("/v1/qa", json={
            "schema_version": "finglmqa.phase8.qa_request.v1",
            "request_id": "phase8-real-evidence-smoke",
            "question": "飞亚达2019年面临哪些经营风险？", "locale": "zh-CN",
        })
    qwen_stopped = False
    try:
        httpx.get("http://127.0.0.1:8011/v1/models", timeout=2)
    except httpx.HTTPError:
        qwen_stopped = True
    frozen_trace = json.loads((ROOT / "runs/phase_09/gate6_report.json").read_text(encoding="utf-8"))["details"]["real_evidence_trace_hash"]
    official_payload = official.json()
    manifest = ImmutableManifestVerifier(ROOT, RUN / "immutable_inputs_manifest.json").verify_full()
    official_rows = read_jsonl(RUN / "http_evaluation.jsonl")
    raw_content_key = any(tree_has_key(row["response"], "content") for row in official_rows)
    logs = b"".join(path.read_bytes() for path in sorted((ROOT / "logs/phase_10").glob("*.jsonl")))
    forbidden_log_tokens = [token for token in (b"/home/", "飞亚达".encode(), b"prompt", b"environment") if token in logs]
    completion_checks = {
        "all_272_shadow_cases_terminal": len(results) == len(eligibility) == 272,
        "all_generator_outcomes_classified": all(row["generator_outcome"] in {
            "accepted", "generator_refused", "generator_invalid_output", "generator_rejected_by_gate",
        } for row in results),
        "zero_unsafe_accepted_claims": sum(row["unsafe_accepted_count"] for row in results) == 0,
        "no_process_or_oom_failure": all("OOM" not in code for row in results for code in row["gate_failure_codes"]),
        "official_answer_unchanged_after_qwen": official.status_code == 200 and official_payload["demo_trace"]["trace_hash"] == frozen_trace,
        "official_service_ready": ready_response.status_code == 200 and ready_response.json().get("ready") is True,
        "qwen_shadow_stopped": qwen_stopped,
        "immutable_manifest_reverified": bool(manifest.semantic_hash),
        "official_artifacts_exclude_raw_chunks": not raw_content_key,
        "logs_are_redacted": not forbidden_log_tokens,
    }
    shadow_report = json.loads((RUN / "shadow_report.json").read_text(encoding="utf-8"))
    shadow_report["promotion_checks"] = promotion_checks
    shadow_report["promotion_readiness"] = promotion_ready
    shadow_report["manual_audit"] = {
        "answerable": len(answerable_audit), "unanswerable": len(unanswerable_audit),
        "artifact": "runs/phase_10/reports/manual_shadow_audit.jsonl",
    }
    write_json(RUN / "shadow_report.json", shadow_report)

    gate8 = {
        "schema_version": "finglmqa.phase10.gate_report.v1", "gate": 8,
        "status": "passed" if all(completion_checks.values()) else "failed",
        "checks": completion_checks,
        "promotion_readiness": promotion_ready,
        "promotion_checks": promotion_checks,
        "details": {
            "manifest_semantic_sha256": manifest.semantic_hash,
            "eligibility_sha256": sha256_file(RUN / "shadow_eligibility_oracle.jsonl"),
            "shadow_results_sha256": sha256_file(RUN / "shadow_results.jsonl"),
            "forbidden_log_tokens": [token.decode("utf-8", errors="replace") for token in forbidden_log_tokens],
        },
    }
    write_json(RUN / "gate8_report.json", gate8)

    release_paths = [
        "phase_10_plan.md", "phase_10_report.md",
        "immutable_inputs_manifest.json", "dependency_manifest.json",
        *[f"gate{index}_report.json" for index in range(0, 9)],
        "gate6_http_report.json", "gate7_http_report.json", "http_evaluation.jsonl",
        "shadow_eligibility_oracle.jsonl", "shadow_adversarial_fixtures.jsonl",
        "shadow_results.jsonl", "shadow_report.json", "promotion_readiness.json",
        "reports/shadow_repeatability.jsonl", "reports/manual_shadow_audit.jsonl",
    ]
    artifacts = []
    for relative in release_paths:
        path = RUN / relative
        if path.is_file():
            artifacts.append({"path": f"runs/phase_10/{relative}", "sha256": sha256_file(path), "size": path.stat().st_size})
    release = {
        "schema_version": "finglmqa.phase10.release_manifest.v1",
        "service_version": "phase10-service-v1",
        "promotion_readiness": promotion_ready,
        "artifacts": artifacts,
    }
    release["manifest_semantic_sha256"] = hashlib.sha256(canonical_json_bytes(release)).hexdigest()
    write_json(RUN / "release_manifest.json", release)
    print(json.dumps({"gate8": gate8["status"], "promotion_readiness": promotion_ready}, sort_keys=True))
    return 0 if gate8["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
