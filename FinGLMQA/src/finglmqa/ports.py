"""Frozen Phase 8 backend ports and boundary validators.

Ports exchange JSON-compatible dictionaries.  The fake implementations used by
unit tests and the production repositories must pass the same validators; this
prevents a test double from silently offering behavior the real store cannot.
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol, runtime_checkable

from .contracts import ContractValidationError

FACT_LOOKUP_STATUSES = frozenset({"found", "not_found", "ambiguous"})


def _exact_fields(value: Mapping[str, Any], required: set[str], name: str) -> None:
    missing = sorted(required - set(value))
    extras = sorted(set(value) - required)
    if missing or extras:
        raise ContractValidationError(f"{name} fields mismatch; missing={missing}, extras={extras}")


def _text(value: Any, name: str, *, nullable: bool = False) -> str | None:
    if nullable and value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ContractValidationError(f"{name} must be a non-empty string")
    return value


def validate_fact_lookup_request(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractValidationError("FactLookupRequest must be an object")
    fields = {
        "schema_version", "requirement_id", "document_id", "stock_code", "report_year",
        "metric_year", "canonical_metric", "normalized_unit",
    }
    _exact_fields(value, fields, "FactLookupRequest")
    if value["schema_version"] != "finglmqa.phase8.fact_lookup_request.v1":
        raise ContractValidationError("FactLookupRequest.schema_version is unsupported")
    for field in ("requirement_id", "document_id", "stock_code", "canonical_metric", "normalized_unit"):
        _text(value[field], f"FactLookupRequest.{field}")
    for field in ("report_year", "metric_year"):
        if isinstance(value[field], bool) or not isinstance(value[field], int):
            raise ContractValidationError(f"FactLookupRequest.{field} must be an integer")
    return value


def validate_fact_record(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractValidationError("FactRecord must be an object")
    fields = {
        "fact_id", "document_id", "stock_code", "company", "report_year", "metric_year",
        "canonical_metric", "normalized_value", "normalized_unit", "source_table_id",
        "source_line_start", "source_line_end", "provenance_json",
    }
    _exact_fields(value, fields, "FactRecord")
    for field in (
        "fact_id", "document_id", "stock_code", "company", "canonical_metric",
        "normalized_value", "normalized_unit", "source_table_id", "provenance_json",
    ):
        _text(value[field], f"FactRecord.{field}")
    for field in ("report_year", "metric_year"):
        if isinstance(value[field], bool) or not isinstance(value[field], int):
            raise ContractValidationError(f"FactRecord.{field} must be an integer")
    for field in ("source_line_start", "source_line_end"):
        if value[field] is not None and (isinstance(value[field], bool) or not isinstance(value[field], int)):
            raise ContractValidationError(f"FactRecord.{field} must be an integer or null")
    return value


def validate_fact_lookup_result(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractValidationError("FactLookupResult must be an object")
    fields = {"schema_version", "requirement_id", "status", "records", "repository_fingerprint"}
    _exact_fields(value, fields, "FactLookupResult")
    if value["schema_version"] != "finglmqa.phase8.fact_lookup_result.v1":
        raise ContractValidationError("FactLookupResult.schema_version is unsupported")
    _text(value["requirement_id"], "FactLookupResult.requirement_id")
    if value["status"] not in FACT_LOOKUP_STATUSES:
        raise ContractValidationError("FactLookupResult.status is invalid")
    _text(value["repository_fingerprint"], "FactLookupResult.repository_fingerprint")
    records = value["records"]
    if not isinstance(records, list):
        raise ContractValidationError("FactLookupResult.records must be an array")
    for record in records:
        validate_fact_record(record)
    if [record["fact_id"] for record in records] != sorted(record["fact_id"] for record in records):
        raise ContractValidationError("FactLookupResult.records must be ordered by fact_id")
    expected = {"found": 1, "not_found": 0}
    if value["status"] in expected and len(records) != expected[value["status"]]:
        raise ContractValidationError("FactLookupResult status/record cardinality mismatch")
    if value["status"] == "ambiguous" and len(records) < 2:
        raise ContractValidationError("ambiguous FactLookupResult requires at least two records")
    return value


def validate_selected_fact_filters(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractValidationError("SelectedFactFilters must be an object")
    fields = {
        "schema_version", "document_ids", "stock_codes", "report_years", "metric_years",
        "canonical_metrics", "normalized_units",
    }
    _exact_fields(value, fields, "SelectedFactFilters")
    if value["schema_version"] != "finglmqa.phase8.selected_fact_filters.v1":
        raise ContractValidationError("SelectedFactFilters.schema_version is unsupported")
    for field in ("document_ids", "stock_codes", "canonical_metrics", "normalized_units"):
        rows = value[field]
        if not isinstance(rows, list) or any(not isinstance(row, str) or not row for row in rows):
            raise ContractValidationError(f"SelectedFactFilters.{field} must be a string array")
        if rows != sorted(set(rows)):
            raise ContractValidationError(f"SelectedFactFilters.{field} must be sorted and unique")
    for field in ("report_years", "metric_years"):
        rows = value[field]
        if not isinstance(rows, list) or any(isinstance(row, bool) or not isinstance(row, int) for row in rows):
            raise ContractValidationError(f"SelectedFactFilters.{field} must be an integer array")
        if rows != sorted(set(rows)):
            raise ContractValidationError(f"SelectedFactFilters.{field} must be sorted and unique")
    return value


@runtime_checkable
class FactLookupPort(Protocol):
    """Exact, document-scoped selected-fact lookup used by fact/formula/SQL."""

    def lookup_fact(self, request: Mapping[str, Any]) -> dict[str, Any]: ...


@runtime_checkable
class SelectedFactQueryPort(Protocol):
    """Read-only allow-list scan used by validated SQL QuerySpecs.

    Every filter is an allow-list.  Empty arrays mean no restriction; callers
    cannot provide SQL text, expressions, column names, or ordering clauses.
    Returned FactRecords use the same shape and stable fact-id order as exact
    lookup results.
    """

    def query_selected_facts(self, filters: Mapping[str, Any]) -> list[dict[str, Any]]: ...


@runtime_checkable
class MetadataLookupPort(Protocol):
    def lookup_metadata(self, request: Mapping[str, Any]) -> dict[str, Any]: ...


@runtime_checkable
class FallbackCandidateIndexPort(Protocol):
    """Discovery-only candidate index; Phase 8 never executes a fallback."""

    def candidate_table_ids(self, request: Mapping[str, Any]) -> list[str]: ...


@runtime_checkable
class EvidenceProviderPort(Protocol):
    """Retrieves within the one document already frozen on an evidence SubPlan."""

    def retrieve(self, request: Mapping[str, Any]) -> dict[str, Any]: ...


@runtime_checkable
class GeneratorPort(Protocol):
    """Builds draft claims; EvidenceExecutor owns all later numeric/citation gates."""

    def generate_claims(self, request: Mapping[str, Any]) -> dict[str, Any]: ...
