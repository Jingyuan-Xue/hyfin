"""Adapters from frozen structured SubPlans to contract-valid results.

The repositories and SQL engine intentionally expose narrow execution ports.
This module owns the Phase 8 envelope around those ports: deterministic
citations, strict Phase 9 request generation, and conversion to SubPlanResult.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import (
    SCHEMA_MISSING_FACT_REQUEST,
    SCHEMA_SUBPLAN_RESULT,
    make_requirement_id,
    semantic_sha256,
    validate_missing_fact_request,
    validate_subplan_result,
)
from .errors import status_for_blocked_plan
from .ports import FactLookupPort, FallbackCandidateIndexPort, MetadataLookupPort
from .repositories import (
    METADATA_REQUEST_SCHEMA,
    validate_metadata_lookup_result,
)
from .sql_engine import SQLExecutionError, SQLExecutor


STRUCTURED_EXECUTION_VERSION = "phase8-structured-adapter-v1"


def _stable_id(prefix: str, value: Any, length: int = 20) -> str:
    return prefix + semantic_sha256(value)[:length]


def portable_semantic_payload(value: Any) -> Any:
    """Remove host-specific prefixes while preserving source identity."""

    if isinstance(value, list):
        return [portable_semantic_payload(row) for row in value]
    if not isinstance(value, dict):
        return value
    result: dict[str, Any] = {}
    for key, child in value.items():
        if key == "source_markdown" and isinstance(child, str):
            marker = "/refs/source_markdown/"
            result[key] = "refs/source_markdown/" + child.split(marker, 1)[1] if marker in child else Path(child).name
        elif key == "provenance_json" and isinstance(child, str):
            try:
                parsed = json.loads(child)
            except json.JSONDecodeError:
                result[key] = child
            else:
                result[key] = json.dumps(
                    portable_semantic_payload(parsed), ensure_ascii=False,
                    sort_keys=True, separators=(",", ":"),
                )
        else:
            result[key] = portable_semantic_payload(child)
    return result


def blocked_subplan_result(subplan: Mapping[str, Any]) -> dict[str, Any]:
    """Materialize a blocked planner placeholder without backend execution."""

    failure = subplan.get("planning_failure") or {
        "failure_code": "INTERNAL_ERROR",
        "message": "Blocked SubPlan has no planning failure",
        "details": {},
    }
    code = str(failure.get("failure_code") or "INTERNAL_ERROR")
    result = _base_result(subplan, backend=str(subplan.get("backend") or "fact"))
    result.update({
        "planning_state": "blocked",
        "execution_state": "not_executed",
        "status": status_for_blocked_plan(code),
        "failure_code": code,
        "errors": [{
            "failure_code": code,
            "message": str(failure.get("message") or "SubPlan was blocked during planning"),
            "details": dict(failure.get("details") or {}),
        }],
        "trace": {
            "adapter_version": STRUCTURED_EXECUTION_VERSION,
            "planning_failure_code": code,
            "backend_called": False,
        },
    })
    validate_subplan_result(result)
    return result


class FactSubPlanExecutor:
    """Execute exact selected facts and identity metadata through their ports."""

    def __init__(
        self,
        fact_port: FactLookupPort,
        metadata_port: MetadataLookupPort,
        *,
        fallback_candidates: FallbackCandidateIndexPort | None = None,
    ) -> None:
        self.fact_port = fact_port
        self.metadata_port = metadata_port
        self.fallback_candidates = fallback_candidates

    def execute(self, subplan: Mapping[str, Any]) -> dict[str, Any]:
        if subplan.get("planning_state") == "blocked":
            return blocked_subplan_result(subplan)
        if subplan.get("backend") != "fact":
            return self._failure(subplan, "error", "INTERNAL_ERROR", "Fact executor received another backend")
        if subplan.get("operation") == "metadata_lookup":
            return self._metadata(subplan)
        if subplan.get("operation") != "fact_lookup":
            return self._failure(subplan, "unsupported", "METRIC_UNRECOGNIZED", "Fact operation is unsupported")
        return self._fact(subplan)

    def _fact(self, subplan: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(subplan.get("payload") or {})
        required = (
            "document_id", "stock_code", "report_year", "metric_year",
            "canonical_metric", "normalized_unit",
        )
        if any(payload.get(field) in (None, "") for field in required):
            return self._failure(
                subplan, "blocked", "PROVENANCE_VALIDATION_FAILED",
                "Exact fact identity is incomplete",
            )
        requirement_id = _stable_id("lookup_", {
            "subplan_id": subplan["subplan_id"],
            "identity": {field: payload[field] for field in required},
        })
        request = {
            "schema_version": "finglmqa.phase8.fact_lookup_request.v1",
            "requirement_id": requirement_id,
            **{field: payload[field] for field in required},
        }
        try:
            response = self.fact_port.lookup_fact(request)
        except Exception as exc:
            return self._failure(
                subplan, "error", "INTERNAL_ERROR", "FactLookupPort failed",
                details={"exception_type": type(exc).__name__},
            )
        if response.get("requirement_id") != requirement_id:
            return self._failure(
                subplan, "blocked", "PROVENANCE_VALIDATION_FAILED",
                "FactLookupPort returned a mismatched requirement identity",
            )
        if response.get("status") == "not_found":
            missing = self._missing_request(subplan, payload)
            return self._failure(
                subplan, "fallback_required", "SELECTED_FACT_MISSING",
                "The uniquely resolved selected fact is absent; Phase 8 does not run fallback",
                missing_fact_requests=[missing],
                trace={"lookup_requirement_id": requirement_id, "lookup_status": "not_found"},
            )
        records = response.get("records")
        if response.get("status") != "found" or not isinstance(records, list) or len(records) != 1:
            return self._failure(
                subplan, "blocked", "FACT_UNRESOLVED_CONFLICT",
                "Exact selected-fact lookup was not unique",
                details={"fact_ids": sorted(str(row.get("fact_id")) for row in records or [])},
                trace={"lookup_requirement_id": requirement_id, "lookup_status": response.get("status")},
            )
        record = records[0]
        expected = {
            "document_id": payload["document_id"], "stock_code": payload["stock_code"],
            "report_year": payload["report_year"], "metric_year": payload["metric_year"],
            "canonical_metric": payload["canonical_metric"], "normalized_unit": payload["normalized_unit"],
        }
        if any(record.get(field) != value for field, value in expected.items()):
            return self._failure(
                subplan, "blocked", "PROVENANCE_VALIDATION_FAILED",
                "FactLookupPort returned a fact outside the exact identity",
                details={"fact_id": record.get("fact_id")},
            )
        try:
            provenance = portable_semantic_payload(json.loads(record["provenance_json"]))
        except (TypeError, json.JSONDecodeError):
            return self._failure(
                subplan, "blocked", "PROVENANCE_VALIDATION_FAILED",
                "Selected fact provenance is not valid JSON",
                details={"fact_id": record.get("fact_id")},
            )
        citation_id = _stable_id("cite_fact_", [subplan["subplan_id"], record["fact_id"]])
        citation = {
            "citation_id": citation_id,
            "citation_kind": "fact",
            "subplan_id": subplan["subplan_id"],
            "entity_key": subplan.get("entity_key"),
            "document_id": record["document_id"],
            "source_citation_ids": [],
            "provenance": {
                "fact_id": record["fact_id"],
                "source_table_id": record["source_table_id"],
                "source_line_start": record["source_line_start"],
                "source_line_end": record["source_line_end"],
                "fact_provenance": provenance,
            },
        }
        output = _base_result(subplan, backend="fact")
        output.update({
            "status": "ok",
            "result": {
                "value": record["normalized_value"],
                "normalized_unit": record["normalized_unit"],
                "canonical_metric": record["canonical_metric"],
                "metric_year": record["metric_year"],
                "report_year": record["report_year"],
                "fact_id": record["fact_id"],
                "entity_key": subplan.get("entity_key"),
                "company": record["company"],
                "stock_code": record["stock_code"],
                "document_id": record["document_id"],
                "provenance": citation["provenance"],
            },
            "citations": [citation],
            "trace": {
                "adapter_version": STRUCTURED_EXECUTION_VERSION,
                "lookup_requirement_id": requirement_id,
                "lookup_status": "found",
                "repository_fingerprint": response.get("repository_fingerprint"),
                "fact_id": record["fact_id"],
            },
        })
        validate_subplan_result(output)
        return output

    def _metadata(self, subplan: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(subplan.get("payload") or {})
        if not payload.get("stock_code") or not payload.get("metadata_field"):
            return self._failure(subplan, "not_found", "RESOLVER_MISSING", "Metadata identity is incomplete")
        requirement_id = _stable_id("metadata_", [subplan["subplan_id"], payload])
        request = {
            "schema_version": METADATA_REQUEST_SCHEMA,
            "requirement_id": requirement_id,
            "entity_key": payload.get("entity_key"),
            "document_id": payload.get("document_id"),
            "stock_code": payload["stock_code"],
            "metadata_field": payload["metadata_field"],
        }
        try:
            response = self.metadata_port.lookup_metadata(request)
            validate_metadata_lookup_result(response)
        except Exception as exc:
            return self._failure(
                subplan, "blocked", "PROVENANCE_VALIDATION_FAILED",
                "MetadataLookupPort returned an invalid result",
                details={"exception_type": type(exc).__name__},
            )
        if response["requirement_id"] != requirement_id:
            return self._failure(
                subplan, "blocked", "PROVENANCE_VALIDATION_FAILED",
                "MetadataLookupPort returned a mismatched requirement identity",
            )
        records = response.get("records")
        if response.get("status") == "not_found" or not records:
            return self._failure(subplan, "not_found", "SELECTED_FACT_MISSING", "Metadata is absent")
        if response.get("status") != "found" or len(records) != 1:
            return self._failure(
                subplan, "needs_clarification", "FACT_UNRESOLVED_CONFLICT",
                "Metadata lookup resolved to multiple values",
            )
        record = records[0]
        document_id = payload.get("document_id")
        expected_identity = {
            "metadata_field": payload["metadata_field"],
            "entity_key": payload.get("entity_key"),
            "stock_code": payload["stock_code"],
        }
        if any(record.get(field) != value for field, value in expected_identity.items()):
            return self._failure(
                subplan, "blocked", "PROVENANCE_VALIDATION_FAILED",
                "MetadataLookupPort returned a record outside the exact identity",
            )
        if document_id is not None and record["document_ids"] != [document_id]:
            return self._failure(
                subplan, "blocked", "PROVENANCE_VALIDATION_FAILED",
                "MetadataLookupPort returned a record outside the exact document scope",
            )
        citation_id = _stable_id("cite_metadata_", [subplan["subplan_id"], record["metadata_field"], record["value"]])
        citation = {
            "citation_id": citation_id,
            "citation_kind": "metadata",
            "subplan_id": subplan["subplan_id"],
            "entity_key": subplan.get("entity_key"),
            "document_id": document_id,
            "source_citation_ids": [],
            "provenance": dict(record["provenance"]),
        }
        output = _base_result(subplan, backend="fact")
        output.update({
            "status": "ok",
            "result": {
                "value": str(record["value"]),
                "normalized_unit": "text",
                "metadata_field": record["metadata_field"],
                "entity_key": subplan.get("entity_key"),
                "company": None,
                "stock_code": payload["stock_code"],
                "document_id": document_id,
                "provenance": citation["provenance"],
            },
            "citations": [citation],
            "trace": {
                "adapter_version": STRUCTURED_EXECUTION_VERSION,
                "metadata_requirement_id": requirement_id,
                "repository_fingerprint": response.get("repository_fingerprint"),
            },
        })
        validate_subplan_result(output)
        return output

    def _missing_request(self, subplan: Mapping[str, Any], payload: Mapping[str, Any]) -> dict[str, Any]:
        request = {
            "schema_version": SCHEMA_MISSING_FACT_REQUEST,
            "requirement_id": "pending",
            "origin_operation": "fact_lookup",
            "formula_id": None,
            "operand_role": None,
            "subplan_id": subplan["subplan_id"],
            "document_id": payload["document_id"],
            "stock_code": payload["stock_code"],
            "report_year": payload["report_year"],
            "metric_year": payload["metric_year"],
            "canonical_metric": payload["canonical_metric"],
            "normalized_unit": payload["normalized_unit"],
            "candidate_table_ids": [],
        }
        request["requirement_id"] = make_requirement_id(request)
        if self.fallback_candidates is not None:
            try:
                values = self.fallback_candidates.candidate_table_ids(request)
                request["candidate_table_ids"] = sorted(set(str(value) for value in values if value))
            except Exception:
                request["candidate_table_ids"] = []
        validate_missing_fact_request(request)
        return request

    @staticmethod
    def _failure(
        subplan: Mapping[str, Any],
        status: str,
        code: str,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
        missing_fact_requests: Sequence[Mapping[str, Any]] = (),
        trace: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        output = _base_result(subplan, backend="fact")
        output.update({
            "status": status,
            "failure_code": code,
            "errors": [{"failure_code": code, "message": message, "details": dict(details or {})}],
            "missing_fact_requests": [dict(row) for row in missing_fact_requests],
            "trace": dict(trace or {"adapter_version": STRUCTURED_EXECUTION_VERSION}),
        })
        validate_subplan_result(output)
        return output


class SQLSubPlanExecutor:
    """Wrap the registered SQL engine in the common SubPlanResult contract."""

    def __init__(
        self,
        executor: SQLExecutor,
        *,
        fallback_candidates: FallbackCandidateIndexPort | None = None,
    ) -> None:
        self.executor = executor
        self.fallback_candidates = fallback_candidates

    def execute(self, subplan: Mapping[str, Any]) -> dict[str, Any]:
        if subplan.get("planning_state") == "blocked":
            return blocked_subplan_result(subplan)
        try:
            execution = self.executor.execute(subplan)
        except SQLExecutionError as exc:
            output = _base_result(subplan, backend="sql")
            output.update({
                "status": exc.status,
                "failure_code": exc.failure_code,
                "errors": [{"failure_code": exc.failure_code, "message": exc.message, "details": dict(exc.details)}],
                "trace": {"adapter_version": STRUCTURED_EXECUTION_VERSION, "query_rejected": True},
            })
            validate_subplan_result(output)
            return output
        citations = self._citations(subplan, execution["rows"])
        status = execution["status"]
        missing_requests: list[dict[str, Any]] = []
        if status == "not_found":
            missing = self._document_missing_request(subplan)
            if missing is not None:
                missing_requests = [missing]
                status = "fallback_required"
        output = _base_result(subplan, backend="sql")
        output.update({
            "status": status,
            "result": ({
                "rows": execution["rows"],
                "query_spec": execution["query_spec"],
                "coverage": execution["coverage"],
            } if execution["rows"] else None),
            "citations": citations,
            "failure_code": "SELECTED_FACT_MISSING" if missing_requests else execution["failure_code"],
            "errors": ([{
                "failure_code": "SELECTED_FACT_MISSING",
                "message": "The uniquely scoped document fact is absent; Phase 8 does not run fallback",
                "details": {},
            }] if missing_requests else []),
            "warnings": execution["warnings"],
            "missing_fact_requests": missing_requests,
            "trace": {
                "adapter_version": STRUCTURED_EXECUTION_VERSION,
                "sql_execution_id": execution["execution_id"],
                "query_fingerprint": execution["query_fingerprint"],
                "executor_version": execution["executor_version"],
            },
        })
        validate_subplan_result(output)
        return output

    def _document_missing_request(self, subplan: Mapping[str, Any]) -> dict[str, Any] | None:
        """Create a Phase 9 request only for a uniquely bound document query."""

        if subplan.get("operation") != "document_query":
            return None
        payload = subplan.get("payload")
        if not isinstance(payload, Mapping):
            return None
        documents = payload.get("document_ids")
        entities = payload.get("entity_keys")
        report_years = payload.get("report_years")
        if not (
            isinstance(documents, list) and len(documents) == 1
            and isinstance(entities, list) and len(entities) == 1
            and isinstance(report_years, list) and len(report_years) == 1
        ):
            return None
        request = {
            "schema_version": SCHEMA_MISSING_FACT_REQUEST,
            "requirement_id": "pending",
            "origin_operation": "document_query",
            "formula_id": None,
            "operand_role": None,
            "subplan_id": subplan["subplan_id"],
            "document_id": documents[0],
            "stock_code": entities[0],
            "report_year": report_years[0],
            "metric_year": payload.get("metric_year"),
            "canonical_metric": payload.get("canonical_metric"),
            "normalized_unit": payload.get("normalized_unit"),
            "candidate_table_ids": [],
        }
        try:
            request["requirement_id"] = make_requirement_id(request)
            validate_missing_fact_request(request)
        except Exception:
            return None
        if self.fallback_candidates is not None:
            try:
                values = self.fallback_candidates.candidate_table_ids(request)
                request["candidate_table_ids"] = sorted(set(
                    str(value) for value in values if isinstance(value, str) and value
                ))
            except Exception:
                request["candidate_table_ids"] = []
        validate_missing_fact_request(request)
        return request

    @staticmethod
    def _citations(subplan: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        citations: list[dict[str, Any]] = []
        for row in rows:
            by_document: dict[str, list[str]] = {}
            for source in row["derivation_inputs"]:
                citation_id = _stable_id("cite_fact_", [subplan["subplan_id"], row["result_row_id"], source["fact_id"]])
                by_document.setdefault(source["document_id"], []).append(citation_id)
                citations.append({
                    "citation_id": citation_id,
                    "citation_kind": "fact",
                    "subplan_id": subplan["subplan_id"],
                    "entity_key": source["stock_code"],
                    "document_id": source["document_id"],
                    "source_citation_ids": [],
                    "provenance": {
                        "fact_id": source["fact_id"],
                        "source_table_id": source["source_table_id"],
                        "source_line_start": source["source_line_start"],
                        "source_line_end": source["source_line_end"],
                    },
                })
            for document_id, source_ids in sorted(by_document.items()):
                citations.append({
                    "citation_id": _stable_id("cite_sql_", [subplan["subplan_id"], row["result_row_id"], document_id]),
                    "citation_kind": "sql_derivation",
                    "subplan_id": subplan["subplan_id"],
                    "entity_key": row.get("entity_key"),
                    "document_id": document_id,
                    "source_citation_ids": sorted(source_ids),
                    "provenance": {
                        "query_spec_id": row["query_spec_id"],
                        "result_row_id": row["result_row_id"],
                        "measure_id": row["measure_id"],
                        "operation": row["operation"],
                        "rank": row["rank"],
                        "aggregate_operator": row["aggregate_operator"],
                        "contributing_fact_ids": sorted(row["contributing_fact_ids"]),
                    },
                })
        return citations


def _base_result(subplan: Mapping[str, Any], *, backend: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_SUBPLAN_RESULT,
        "subplan_id": str(subplan.get("subplan_id") or "invalid_subplan"),
        "backend": backend,
        "operation": str(subplan.get("operation") or "invalid_operation"),
        "planning_state": "ready",
        "execution_state": "executed",
        "status": "error",
        "result": None,
        "claims": [],
        "citations": [],
        "failure_code": None,
        "errors": [],
        "warnings": [],
        "missing_fact_requests": [],
        "trace": {"adapter_version": STRUCTURED_EXECUTION_VERSION},
    }


__all__ = [
    "FactSubPlanExecutor",
    "SQLSubPlanExecutor",
    "STRUCTURED_EXECUTION_VERSION",
    "blocked_subplan_result",
    "portable_semantic_payload",
]
