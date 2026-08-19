"""Phase 9-only contracts for the additive supplemental fact pipeline.

The Phase 8 contracts are deliberately not imported for extension: the only
shared boundary is the already-frozen MissingFactRequest validator and the v1
FactLookupPort used by :class:`SupplementAwareFactRepository`.
"""

from __future__ import annotations

import json
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping

from .contracts import ContractValidationError, semantic_sha256


SCHEMA_SUPPLEMENTAL_FACT = "finglmqa.phase9.supplemental_fact.v1"
SCHEMA_SUPPLEMENT_DECISION = "finglmqa.phase9.supplement_decision.v1"
SCHEMA_SUPPLEMENT_LOOKUP = "finglmqa.phase9.supplement_lookup_result.v1"
FACT_SOURCE = "supplemental_tabgr"

FAILURE_CODES = frozenset({
    "SUPPLEMENT_REQUEST_INVALID",
    "SUPPLEMENT_ALREADY_SELECTED",
    "SUPPLEMENT_CONFLICT_GROUP_OPEN",
    "SUPPLEMENT_FACT_WITHHELD",
    "SUPPLEMENT_NO_CANDIDATE_TABLE",
    "SUPPLEMENT_CELL_NOT_FOUND",
    "SUPPLEMENT_YEAR_UNRESOLVED",
    "SUPPLEMENT_UNIT_UNRESOLVED",
    "SUPPLEMENT_VALUE_INVALID",
    "SUPPLEMENT_ELIGIBILITY_REJECTED",
    "SUPPLEMENT_VALUE_CONFLICT",
    "SUPPLEMENT_PROVENANCE_FAILED",
    "SUPPLEMENT_RUNTIME_UNAVAILABLE",
})

SLOT_FIELDS = (
    "document_id", "stock_code", "report_year", "metric_year",
    "canonical_metric", "normalized_unit",
)


def canonical_slot_key(value: Mapping[str, Any]) -> list[Any]:
    return [value.get(field) for field in SLOT_FIELDS]


def slot_fingerprint(value: Mapping[str, Any]) -> str:
    return semantic_sha256(canonical_slot_key(value))


def _exact(value: Any, fields: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractValidationError(f"{name} must be an object")
    missing = sorted(fields - set(value))
    extras = sorted(set(value) - fields)
    if missing or extras:
        raise ContractValidationError(f"{name} fields mismatch; missing={missing}, extras={extras}")
    return value


def _text(value: Any, name: str, *, nullable: bool = False) -> str | None:
    if nullable and value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ContractValidationError(f"{name} must be a non-empty string")
    return value


def _integer(value: Any, name: str, *, nullable: bool = False) -> int | None:
    if nullable and value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractValidationError(f"{name} must be an integer")
    return value


def _decimal(value: Any, name: str) -> str:
    text = _text(value, name)
    try:
        number = Decimal(text)
    except InvalidOperation as exc:
        raise ContractValidationError(f"{name} must be a Decimal string") from exc
    if not number.is_finite():
        raise ContractValidationError(f"{name} must be finite")
    return text


def validate_supplemental_fact(value: Any) -> dict[str, Any]:
    fields = {
        "supplemental_fact_id", "schema_version", *SLOT_FIELDS, "company",
        "statement", "normalized_value", "fact_source", "validation_versions",
        "tabgr_trace_fingerprint", "source_table_id", "source_table_index",
        "source_row_index", "source_col_index", "source_line_start", "source_line_end",
        "source_markdown", "provenance_json", "created_from_requirement_ids",
    }
    obj = _exact(value, fields, "SupplementalFact")
    if obj["schema_version"] != SCHEMA_SUPPLEMENTAL_FACT:
        raise ContractValidationError("SupplementalFact.schema_version is unsupported")
    if obj["fact_source"] != FACT_SOURCE:
        raise ContractValidationError("SupplementalFact.fact_source is invalid")
    for field in (
        "supplemental_fact_id", "document_id", "stock_code", "company",
        "canonical_metric", "normalized_unit", "statement", "tabgr_trace_fingerprint",
        "source_table_id", "source_markdown",
    ):
        _text(obj[field], f"SupplementalFact.{field}")
    for field in ("report_year", "metric_year", "source_table_index", "source_row_index", "source_col_index"):
        _integer(obj[field], f"SupplementalFact.{field}")
    for field in ("source_line_start", "source_line_end"):
        _integer(obj[field], f"SupplementalFact.{field}", nullable=True)
    _decimal(obj["normalized_value"], "SupplementalFact.normalized_value")
    if not isinstance(obj["validation_versions"], dict) or not obj["validation_versions"]:
        raise ContractValidationError("SupplementalFact.validation_versions must be a non-empty object")
    provenance = obj["provenance_json"]
    if not isinstance(provenance, list) or not provenance or any(not isinstance(row, dict) for row in provenance):
        raise ContractValidationError("SupplementalFact.provenance_json must be a non-empty object array")
    requirement_ids = obj["created_from_requirement_ids"]
    if not isinstance(requirement_ids, list) or not requirement_ids or requirement_ids != sorted(set(requirement_ids)):
        raise ContractValidationError("SupplementalFact.created_from_requirement_ids must be sorted and unique")
    return obj


def validate_supplement_decision(value: Any) -> dict[str, Any]:
    fields = {
        "schema_version", "slot_key", "slot_fingerprint", "requirement_id",
        "decision_status", "failure_code", "request_class", "conflict_group_ids",
        "shortlist", "ranked_cell_fingerprints", "rejected_values",
        "accepted_fact_id", "audit_counts", "trace_fingerprint",
    }
    obj = _exact(value, fields, "SupplementDecision")
    if obj["schema_version"] != SCHEMA_SUPPLEMENT_DECISION:
        raise ContractValidationError("SupplementDecision.schema_version is unsupported")
    slot = obj["slot_key"]
    if not isinstance(slot, list) or len(slot) != len(SLOT_FIELDS):
        raise ContractValidationError("SupplementDecision.slot_key is invalid")
    if obj["slot_fingerprint"] != semantic_sha256(slot):
        raise ContractValidationError("SupplementDecision.slot_fingerprint is invalid")
    _text(obj["requirement_id"], "SupplementDecision.requirement_id")
    if obj["decision_status"] not in {"accepted", "rejected"}:
        raise ContractValidationError("SupplementDecision.decision_status is invalid")
    code = obj["failure_code"]
    if obj["decision_status"] == "accepted":
        if code is not None or obj["accepted_fact_id"] is None:
            raise ContractValidationError("accepted decision must have fact ID and null failure code")
    elif code not in FAILURE_CODES or obj["accepted_fact_id"] is not None:
        raise ContractValidationError("rejected decision must have a known failure code and null fact ID")
    _text(obj["request_class"], "SupplementDecision.request_class")
    for field in ("conflict_group_ids", "ranked_cell_fingerprints", "rejected_values"):
        if not isinstance(obj[field], list):
            raise ContractValidationError(f"SupplementDecision.{field} must be an array")
    if obj["conflict_group_ids"] != sorted(set(obj["conflict_group_ids"])):
        raise ContractValidationError("SupplementDecision.conflict_group_ids must be sorted and unique")
    if not isinstance(obj["shortlist"], list) or any(not isinstance(row, dict) for row in obj["shortlist"]):
        raise ContractValidationError("SupplementDecision.shortlist must be an object array")
    if not isinstance(obj["audit_counts"], dict):
        raise ContractValidationError("SupplementDecision.audit_counts must be an object")
    _text(obj["trace_fingerprint"], "SupplementDecision.trace_fingerprint")
    return obj


def validate_supplement_lookup_result(value: Any) -> dict[str, Any]:
    fields = {"schema_version", "requirement_id", "status", "records", "repository_fingerprint", "fact_source"}
    obj = _exact(value, fields, "SupplementLookupResult")
    if obj["schema_version"] != SCHEMA_SUPPLEMENT_LOOKUP or obj["fact_source"] != FACT_SOURCE:
        raise ContractValidationError("SupplementLookupResult schema/source is invalid")
    _text(obj["requirement_id"], "SupplementLookupResult.requirement_id")
    _text(obj["repository_fingerprint"], "SupplementLookupResult.repository_fingerprint")
    if obj["status"] not in {"found", "not_found", "ambiguous"} or not isinstance(obj["records"], list):
        raise ContractValidationError("SupplementLookupResult status/records are invalid")
    expected = {"found": 1, "not_found": 0}
    if obj["status"] in expected and len(obj["records"]) != expected[obj["status"]]:
        raise ContractValidationError("SupplementLookupResult status/record cardinality mismatch")
    if obj["status"] == "ambiguous" and len(obj["records"]) < 2:
        raise ContractValidationError("ambiguous SupplementLookupResult requires multiple records")
    for record in obj["records"]:
        validate_supplemental_fact(record)
    return obj


def canonical_provenance_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


__all__ = [
    "FACT_SOURCE", "FAILURE_CODES", "SCHEMA_SUPPLEMENTAL_FACT",
    "SCHEMA_SUPPLEMENT_DECISION", "SCHEMA_SUPPLEMENT_LOOKUP", "SLOT_FIELDS",
    "canonical_provenance_text", "canonical_slot_key", "slot_fingerprint",
    "validate_supplement_decision", "validate_supplement_lookup_result",
    "validate_supplemental_fact",
]
