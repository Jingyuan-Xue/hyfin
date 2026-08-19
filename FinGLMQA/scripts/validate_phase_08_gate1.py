#!/usr/bin/env python3
"""Validate and report the frozen Phase 8 Gate 1 contract surface."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from finglmqa.contracts import (  # noqa: E402
    LIMITS,
    PATTERN_IDS,
    canonical_json_bytes,
    semantic_sha256,
    validate_pattern_registry,
)
from finglmqa.errors import (  # noqa: E402
    ALL_FAILED_STATUS_PRECEDENCE,
    BLOCKED_PLAN_STATUS_BY_CODE,
    FAILURE_CODES,
    PRECOMPOSITION_STATUS_BY_CODE,
)

PHASE8_SCHEMAS = (
    "qa_request.schema.json",
    "question_analysis.schema.json",
    "scope_plan.schema.json",
    "composition_plan.schema.json",
    "subplan_result.schema.json",
    "numeric_authorization.schema.json",
    "numeric_authorization_set.schema.json",
    "missing_fact_request.schema.json",
    "qa_answer.schema.json",
    "qa_trace.schema.json",
    "qa_telemetry.schema.json",
    "fact_lookup_request.schema.json",
    "fact_lookup_result.schema.json",
    "selected_fact_filters.schema.json",
    "phase_10_service_projection.schema.json",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = canonical_json_bytes(value)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> int:
    schemas: list[dict[str, Any]] = []
    schema_dir = ROOT / "data/schemas"
    for name in PHASE8_SCHEMAS:
        path = schema_dir / name
        payload = json.loads(path.read_text(encoding="utf-8"))
        schemas.append({
            "path": path.relative_to(ROOT).as_posix(),
            "id": payload["$id"],
            "draft": payload["$schema"],
            "additional_properties": payload.get("additionalProperties"),
            "sha256": sha256_file(path),
        })
    registry_path = ROOT / "src/config/composition_patterns.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    validate_pattern_registry(registry)
    command = [
        str(ROOT / ".venv/bin/python"), "-m", "unittest", "-v",
        "tests.test_phase08_contracts", "tests.test_phase08_fact_lookup_port",
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": "src", "PYTHONDONTWRITEBYTECODE": "1"},
        text=True,
        capture_output=True,
    )
    report = {
        "schema_version": "finglmqa.phase8.gate1_contract_report.v1",
        "checks": {
            "all_schema_files_parse": len(schemas) == len(PHASE8_SCHEMAS),
            "schema_ids_unique": len({row["id"] for row in schemas}) == len(schemas),
            "schemas_reject_unknown_top_level_fields": all(row["additional_properties"] is False for row in schemas),
            "pattern_registry_valid": True,
            "pattern_count_9": tuple(row["pattern_id"] for row in registry["patterns"]) == PATTERN_IDS,
            "limits_frozen": registry["limits"] == LIMITS,
            "contract_unit_tests_pass": completed.returncode == 0,
        },
        "schemas": schemas,
        "pattern_registry": {
            "semantic_sha256": semantic_sha256(registry),
            "file_sha256": sha256_file(registry_path),
            "pattern_ids": list(PATTERN_IDS),
        },
        "status_truth_table": {
            "precomposition": PRECOMPOSITION_STATUS_BY_CODE,
            "blocked_plan": BLOCKED_PLAN_STATUS_BY_CODE,
            "all_failed_precedence": list(ALL_FAILED_STATUS_PRECEDENCE),
            "provenance_override": {
                "failure_code": "PROVENANCE_VALIDATION_FAILED",
                "answer_status": "blocked",
                "suppress_all_outputs": True,
            },
        },
        "failure_codes": sorted(FAILURE_CODES),
        "test_command": command,
        "test_stdout": completed.stdout,
        "test_stderr": completed.stderr,
        "ownership": {
            "evidence_executor": [
                "EvidenceProviderPort", "claim_builder", "numeric_filtering",
                "citation_construction", "SubPlanResult",
            ],
            "generator_port_scope": "draft_claims_only",
            "fallback_candidate_index_executes_fallback": False,
        },
    }
    report["all_checks_passed"] = all(report["checks"].values())
    report["contract_surface_semantic_sha256"] = semantic_sha256({
        "schemas": schemas,
        "registry": report["pattern_registry"],
        "status_truth_table": report["status_truth_table"],
        "failure_codes": report["failure_codes"],
        "ownership": report["ownership"],
    })
    atomic_write(ROOT / "runs/phase_08/gate1_contract_report.json", report)
    print(json.dumps({"all_checks_passed": report["all_checks_passed"], "checks": report["checks"]}, ensure_ascii=False, indent=2))
    return 0 if report["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
