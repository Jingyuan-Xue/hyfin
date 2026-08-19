"""Phase 10 service-only contracts around the frozen Phase 8 pipeline."""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from .contracts import (
    ContractValidationError,
    canonical_json_bytes,
    semantic_sha256,
    validate_qa_answer,
    validate_qa_request,
    validate_qa_trace,
)


SERVICE_VERSION = "phase10-service-v1"
WORKER_PROTOCOL = "finglmqa.phase10.qa_worker.v1"
DEMO_TRACE_SCHEMA = "finglmqa.phase10.demo_trace.v1"
SHADOW_ELIGIBILITY_SCHEMA = "finglmqa.phase10.shadow_eligibility.v1"
SHADOW_RESULT_SCHEMA = "finglmqa.phase10.shadow_result.v1"

SERVICE_FAILURE_CODES = frozenset({
    "SERVICE_PAYLOAD_INVALID",
    "SERVICE_PAYLOAD_TOO_LARGE",
    "SERVICE_QUEUE_FULL",
    "SERVICE_NOT_READY",
    "SERVICE_TIMEOUT",
    "SERVICE_WORKER_RESTARTED",
})

_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_RUNTIME_MARKERS = (
    "timestamp", "generated_at", "started_at", "finished_at", "latency", "elapsed",
    "duration", "process_id", "pid", "worker_generation", "device", "gpu",
    "temporary_path", "runtime_path",
)


class ServiceContractError(ValueError):
    """A Phase 10 service boundary object is invalid."""


def validate_wire_request(payload: Any) -> dict[str, Any]:
    """Apply the complete frozen QARequest schema before invoking the pipeline."""

    if not isinstance(payload, dict):
        raise ServiceContractError("request JSON root must be an object")
    try:
        return validate_qa_request(payload)
    except (ContractValidationError, TypeError, ValueError) as exc:
        raise ServiceContractError(str(exc)) from exc


def semantic_trace_projection(trace: Mapping[str, Any]) -> dict[str, Any]:
    """Remove request identity and any runtime field from an already valid trace."""

    checked = validate_qa_trace(deepcopy(dict(trace)))

    def scrub(value: Any) -> Any:
        if isinstance(value, dict):
            result: dict[str, Any] = {}
            for key, item in value.items():
                lower = key.lower()
                # Phase 8's composition_plan_id and plan_fingerprint are
                # derived from the full planning payload, which contains the
                # caller request_id.  They are request-instance identities,
                # not QA semantics, so the Phase 10 semantic projection drops
                # them alongside request_id.
                if key in {"request_id", "trace_hash", "composition_plan_id", "plan_fingerprint"}:
                    continue
                if any(marker in lower for marker in _RUNTIME_MARKERS):
                    continue
                result[key] = scrub(item)
            return result
        if isinstance(value, list):
            return [scrub(item) for item in value]
        return deepcopy(value)

    return scrub(checked)


def semantic_trace_hash(trace: Mapping[str, Any]) -> str:
    return semantic_sha256(semantic_trace_projection(trace))


def trace_file_sha256(trace: Mapping[str, Any]) -> str:
    validate_qa_trace(deepcopy(dict(trace)))
    return hashlib.sha256(canonical_json_bytes(trace)).hexdigest()


def _ordered_subplan_summary(answer: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "subplan_id": row["subplan_id"],
            "backend": row["backend"],
            "operation": row["operation"],
            "status": row["status"],
            "failure_code": row["failure_code"],
        }
        for row in answer["subplans"]
    ]


def build_service_projection(
    answer: Mapping[str, Any],
    trace: Mapping[str, Any],
    *,
    trace_delivery: str,
) -> dict[str, Any]:
    checked_answer = validate_qa_answer(deepcopy(dict(answer)))
    checked_trace = validate_qa_trace(deepcopy(dict(trace)))
    if checked_answer["request_id"] != checked_trace["request_id"]:
        raise ServiceContractError("answer and trace request IDs differ")
    if trace_delivery not in {"inline", "reference"}:
        raise ServiceContractError("trace_delivery must be inline or reference")
    complete_hash = checked_trace["trace_hash"]
    semantic_hash = semantic_trace_hash(checked_trace)
    file_hash = trace_file_sha256(checked_trace)
    demo: dict[str, Any] = {
        "schema_version": DEMO_TRACE_SCHEMA,
        "request_id": checked_answer["request_id"],
        "composition_pattern_id": checked_answer["composition_pattern_id"],
        "subplans": _ordered_subplan_summary(checked_answer),
        "missing_fact_requests": deepcopy(checked_answer["missing_fact_requests"]),
        "trace_hash": complete_hash,
        "semantic_trace_hash": semantic_hash,
        "trace_file_sha256": file_hash,
        "trace_delivery": trace_delivery,
    }
    if trace_delivery == "inline":
        demo["trace"] = checked_trace
    else:
        demo["trace_reference"] = f"/v1/traces/{complete_hash}"
    projection = {
        "answer": checked_answer["answer_text"],
        "status": checked_answer["status"],
        "citations": deepcopy(checked_answer["citations"]),
        "errors": deepcopy(checked_answer["errors"]),
        "warnings": deepcopy(checked_answer["warnings"]),
        "demo_trace": demo,
    }
    return validate_service_projection(projection)


def build_service_error(
    failure_code: str,
    message: str,
    *,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if failure_code not in SERVICE_FAILURE_CODES:
        raise ServiceContractError("unknown service failure code")
    projection = {
        "answer": "",
        "status": "error",
        "citations": [],
        "errors": [{
            "failure_code": failure_code,
            "message": message,
            "details": dict(details or {}),
        }],
        "warnings": [],
        "demo_trace": {
            "schema_version": DEMO_TRACE_SCHEMA,
            "service_failure_code": failure_code,
        },
    }
    return validate_service_projection(projection, allow_service_error=True)


def validate_service_projection(
    payload: Any, *, allow_service_error: bool = False,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ServiceContractError("service projection must be an object")
    fields = {"answer", "status", "citations", "errors", "warnings", "demo_trace"}
    if set(payload) != fields:
        raise ServiceContractError("service projection fields are not exact")
    if not isinstance(payload["answer"], str) or not isinstance(payload["status"], str):
        raise ServiceContractError("service projection answer/status are invalid")
    for name in ("citations", "errors", "warnings"):
        if not isinstance(payload[name], list):
            raise ServiceContractError(f"service projection {name} must be an array")
    demo = payload["demo_trace"]
    if not isinstance(demo, dict) or demo.get("schema_version") != DEMO_TRACE_SCHEMA:
        raise ServiceContractError("demo_trace schema is invalid")
    if allow_service_error:
        code = demo.get("service_failure_code")
        if code not in SERVICE_FAILURE_CODES or payload["status"] != "error":
            raise ServiceContractError("service failure projection is invalid")
        return payload
    required = {
        "schema_version", "request_id", "composition_pattern_id", "subplans",
        "missing_fact_requests", "trace_hash", "semantic_trace_hash",
        "trace_file_sha256", "trace_delivery",
    }
    if not required.issubset(demo):
        raise ServiceContractError("demo_trace fields are incomplete")
    for name in ("trace_hash", "semantic_trace_hash", "trace_file_sha256"):
        if not isinstance(demo[name], str) or not _HEX64_RE.fullmatch(demo[name]):
            raise ServiceContractError(f"demo_trace {name} is invalid")
    delivery = demo["trace_delivery"]
    if delivery == "inline":
        if set(demo) != required | {"trace"}:
            raise ServiceContractError("inline demo_trace fields are not exact")
        validate_qa_trace(demo["trace"])
    elif delivery == "reference":
        if set(demo) != required | {"trace_reference"}:
            raise ServiceContractError("reference demo_trace fields are not exact")
        if demo["trace_reference"] != f"/v1/traces/{demo['trace_hash']}":
            raise ServiceContractError("trace reference does not match trace hash")
    else:
        raise ServiceContractError("demo_trace trace_delivery is invalid")
    return payload


def persist_trace(trace: Mapping[str, Any], directory: str | Path) -> tuple[Path, str]:
    checked = validate_qa_trace(deepcopy(dict(trace)))
    encoded = canonical_json_bytes(checked)
    file_hash = hashlib.sha256(encoded).hexdigest()
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"{checked['trace_hash']}.json"
    if target.exists():
        if target.read_bytes() != encoded:
            raise ServiceContractError("existing trace hash path contains different bytes")
        return target, file_hash
    temporary = target.with_suffix(".json.tmp")
    temporary.write_bytes(encoded)
    temporary.replace(target)
    return target, file_hash


def read_trace(trace_hash: str, directory: str | Path) -> dict[str, Any]:
    if not _HEX64_RE.fullmatch(trace_hash):
        raise ServiceContractError("trace hash path is invalid")
    path = Path(directory) / f"{trace_hash}.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ServiceContractError("trace does not exist or is invalid") from exc
    checked = validate_qa_trace(value)
    if checked["trace_hash"] != trace_hash:
        raise ServiceContractError("trace file name and content disagree")
    return checked


__all__ = [
    "DEMO_TRACE_SCHEMA", "SERVICE_FAILURE_CODES", "SERVICE_VERSION",
    "SHADOW_ELIGIBILITY_SCHEMA", "SHADOW_RESULT_SCHEMA", "WORKER_PROTOCOL",
    "ServiceContractError", "build_service_error", "build_service_projection",
    "persist_trace", "read_trace", "semantic_trace_hash",
    "semantic_trace_projection", "trace_file_sha256", "validate_service_projection",
    "validate_wire_request",
]
