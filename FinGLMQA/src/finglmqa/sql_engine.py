"""Validated static QuerySpec execution for Phase 8.

This module is intentionally not a general SQL surface.  It accepts two
registered semantic operations and translates them to the frozen
``SelectedFactQueryPort`` allow-list.  No SQL text, expression, column name,
or user-provided ordering clause reaches the repository.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import re
from typing import Any, Mapping, Sequence

from .contracts import canonical_json_bytes, semantic_sha256
from .metric_catalog import MetricCatalog
from .ports import (
    SelectedFactQueryPort,
    validate_fact_record,
    validate_selected_fact_filters,
)


SQL_EXECUTION_SCHEMA = "finglmqa.phase8.sql_execution.v1"
SELECTED_FACT_FILTER_SCHEMA = "finglmqa.phase8.selected_fact_filters.v1"

QUERY_SPECS: dict[str, dict[str, Any]] = {
    "phase8.rank.v1": {
        "query_kind": "rank",
        "aggregate_operators": [],
        "order_directions": ["asc", "desc"],
        "default_order_direction": "desc",
        "default_limit": 1,
        "maximum_limit": 5,
    },
    "phase8.aggregate.v1": {
        "query_kind": "aggregate",
        "aggregate_operators": ["average", "count", "sum"],
        "order_directions": [],
        "default_order_direction": None,
        "default_limit": None,
        "maximum_limit": None,
    },
}

_COMMON_PAYLOAD_FIELDS = {
    "query_spec_id",
    "query_kind",
    "canonical_metric",
    "metric_year",
    "normalized_unit",
    "document_ids",
    "entity_keys",
    "report_years",
    "scope_company_count",
    "scope_document_count",
}
_RANK_FIELDS = _COMMON_PAYLOAD_FIELDS | {"order_direction", "limit"}
_AGGREGATE_FIELDS = _COMMON_PAYLOAD_FIELDS | {"aggregate_operator"}
_UNSAFE_TEXT_RE = re.compile(
    r"(?:\x00|;|--|/\*|\*/|\b(?:attach|alter|copy|delete|drop|insert|pragma|select|union|update)\b)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SQLExecutionError(Exception):
    """Fail-closed error emitted before a result can be composed."""

    failure_code: str
    message: str
    details: Mapping[str, Any]

    def __post_init__(self) -> None:
        Exception.__init__(self, self.message)

    @property
    def status(self) -> str:
        return "blocked" if self.failure_code == "SQL_SAFETY_REJECTED" else "unsupported"

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "failure_code": self.failure_code,
            "message": self.message,
            "details": dict(self.details),
        }


class SQLExecutor:
    """Execute one static rank or aggregate QuerySpec over selected facts."""

    executor_version = "1.0.0"

    def __init__(
        self,
        fact_query_port: SelectedFactQueryPort,
        *,
        metric_catalog: MetricCatalog | None = None,
    ) -> None:
        if not callable(getattr(fact_query_port, "query_selected_facts", None)):
            raise TypeError("fact_query_port must implement query_selected_facts")
        self.fact_query_port = fact_query_port
        self.metric_catalog = metric_catalog or MetricCatalog()

    def execute(self, subplan_or_payload: Mapping[str, Any]) -> dict[str, Any]:
        """Validate, query through the allow-list port, and derive stable rows.

        A full SubPlan and its operation payload are both accepted.  A blocked
        or non-SQL SubPlan is rejected before the repository is invoked.
        """

        payload, subplan_id = self._extract_payload(subplan_or_payload)
        spec = self._validate_payload(payload)
        filters = self._filters(payload)
        try:
            raw_records = self.fact_query_port.query_selected_facts(filters)
        except SQLExecutionError:
            raise
        except Exception as exc:
            self._safety(
                "SelectedFactQueryPort rejected or failed the validated allow-list request",
                exception_type=type(exc).__name__,
            )
        records = self._validate_records(raw_records, payload, filters)

        if spec["query_kind"] == "rank":
            rows = self._rank_rows(records, payload)
        else:
            rows = self._aggregate_rows(records, payload)
        coverage = self._coverage(records, payload)
        warnings = (
            [{
                "warning_code": "CORPUS_COVERAGE_INCOMPLETE",
                "message": "The scoped selected-fact coverage is incomplete for this QuerySpec.",
                "missing_document_ids": coverage["missing_document_ids"],
                "missing_entity_keys": coverage["missing_entity_keys"],
            }]
            if not coverage["complete"] else []
        )
        status = "partial" if rows and warnings else ("ok" if rows else "not_found")
        normalized_spec = self._normalized_spec(payload)
        query_fingerprint = semantic_sha256({
            "query_spec": normalized_spec,
            "filters": filters,
        })
        execution_id = "sqlx_" + semantic_sha256({
            "subplan_id": subplan_id,
            "query_fingerprint": query_fingerprint,
            "rows": rows,
            "coverage": coverage,
        })[:16]
        result = {
            "schema_version": SQL_EXECUTION_SCHEMA,
            "execution_id": execution_id,
            "executor_version": self.executor_version,
            "subplan_id": subplan_id,
            "query_spec": normalized_spec,
            "query_fingerprint": query_fingerprint,
            "status": status,
            "failure_code": None,
            "rows": rows,
            "coverage": coverage,
            "warnings": warnings,
            # Phase 8 never invokes fallback.  The present planner payload has
            # no report-year identity for a missing document, so fabricating a
            # strict MissingFactRequest seed would violate the contract.
            "missing_fact_requirement_seeds": [],
        }
        # This assertion also catches accidental float/non-JSON additions.
        canonical_json_bytes(result)
        return result

    @staticmethod
    def _extract_payload(value: Mapping[str, Any]) -> tuple[dict[str, Any], str | None]:
        if not isinstance(value, Mapping) or not value:
            raise SQLExecutionError("SQL_SAFETY_REJECTED", "SQL input must be a non-empty object", {})
        if "backend" not in value:
            return dict(value), None
        if value.get("planning_state") != "ready":
            raise SQLExecutionError(
                "SQL_SAFETY_REJECTED", "A blocked SQL SubPlan cannot execute", {}
            )
        if value.get("backend") != "sql":
            raise SQLExecutionError(
                "SQL_SAFETY_REJECTED", "SQLExecutor accepts only backend=sql SubPlans", {}
            )
        if value.get("operation") not in {"corpus_query", "document_query"}:
            raise SQLExecutionError(
                "SQL_UNSUPPORTED_QUERY", "The SQL SubPlan operation is not registered",
                {"operation": value.get("operation")},
            )
        payload = value.get("payload")
        if not isinstance(payload, dict) or not payload:
            raise SQLExecutionError(
                "SQL_SAFETY_REJECTED", "A ready SQL SubPlan requires a non-empty payload", {}
            )
        subplan_id = value.get("subplan_id")
        if subplan_id is not None and not isinstance(subplan_id, str):
            raise SQLExecutionError(
                "SQL_SAFETY_REJECTED", "subplan_id must be a string when present", {}
            )
        return dict(payload), subplan_id

    def _validate_payload(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, Mapping) or not payload:
            self._safety("QuerySpec payload must be a non-empty object")
        spec_id = payload.get("query_spec_id")
        if not isinstance(spec_id, str) or spec_id not in QUERY_SPECS:
            self._unsupported("Unknown static QuerySpec", query_spec_id=spec_id)
        spec = QUERY_SPECS[spec_id]
        if payload.get("query_kind") != spec["query_kind"]:
            self._unsupported(
                "QuerySpec ID and query_kind disagree",
                query_spec_id=spec_id,
                query_kind=payload.get("query_kind"),
            )
        allowed = _RANK_FIELDS if spec["query_kind"] == "rank" else _AGGREGATE_FIELDS
        extras = sorted(set(payload) - allowed)
        missing = sorted(_COMMON_PAYLOAD_FIELDS - set(payload))
        if extras or missing:
            self._safety("QuerySpec payload fields are not allow-listed", extras=extras, missing=missing)

        metric = self._safe_text(payload.get("canonical_metric"), "canonical_metric")
        if not self.metric_catalog.is_selected_fact_metric(metric):
            self._unsupported(
                "QuerySpec metric is not in the Phase 6 selected-fact catalog",
                canonical_metric=metric,
            )
        unit = self._safe_text(payload.get("normalized_unit"), "normalized_unit")
        expected_unit = self.metric_catalog.expected_unit(metric)
        if expected_unit is not None and unit != expected_unit:
            self._unsupported(
                "QuerySpec unit is incompatible with the metric catalog",
                canonical_metric=metric,
                normalized_unit=unit,
                expected_unit=expected_unit,
            )
        if expected_unit is None and metric == "股本" and unit not in {"元", "元/股", "股"}:
            self._unsupported(
                "股本 QuerySpec requires an explicit selected-fact unit",
                normalized_unit=unit,
            )
        year = payload.get("metric_year")
        if isinstance(year, bool) or not isinstance(year, int) or not 1900 <= year <= 2200:
            self._safety("metric_year must be an integer in the supported year range")
        for field in ("document_ids", "entity_keys"):
            self._safe_string_array(payload.get(field), field)
        report_years = payload.get("report_years")
        if (
            not isinstance(report_years, list)
            or any(isinstance(year, bool) or not isinstance(year, int) or not 1900 <= year <= 2200 for year in report_years)
            or report_years != sorted(set(report_years))
        ):
            self._safety("report_years must be a sorted unique integer allow-list")
        for field in ("scope_company_count", "scope_document_count"):
            count = payload.get(field)
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                self._safety(f"{field} must be a nonnegative integer")

        if spec["query_kind"] == "rank":
            direction = payload.get("order_direction", spec["default_order_direction"])
            if direction not in spec["order_directions"]:
                self._unsupported("Unsupported rank order direction", order_direction=direction)
            limit = payload.get("limit", spec["default_limit"])
            if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= spec["maximum_limit"]:
                self._unsupported("Rank limit must be an integer from 1 through 5", limit=limit)
        else:
            operator = payload.get("aggregate_operator")
            if operator not in spec["aggregate_operators"]:
                self._unsupported(
                    "Aggregate QuerySpec requires an explicit registered operator",
                    aggregate_operator=operator,
                )
        return spec

    def _filters(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        filters = {
            "schema_version": SELECTED_FACT_FILTER_SCHEMA,
            "document_ids": sorted(payload["document_ids"]),
            # Real resolver entity keys are stock codes.  Document IDs remain
            # the stronger boundary whenever both arrays are populated.
            "stock_codes": sorted(payload["entity_keys"]),
            "report_years": list(payload["report_years"]),
            "metric_years": [payload["metric_year"]],
            "canonical_metrics": [payload["canonical_metric"]],
            "normalized_units": [payload["normalized_unit"]],
        }
        try:
            return validate_selected_fact_filters(filters)
        except Exception as exc:
            self._safety("QuerySpec could not be converted to a valid allow-list", reason=str(exc))

    def _validate_records(
        self,
        raw_records: Any,
        payload: Mapping[str, Any],
        filters: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        if not isinstance(raw_records, list):
            self._safety("SelectedFactQueryPort must return an array")
        records: list[dict[str, Any]] = []
        for index, raw in enumerate(raw_records):
            try:
                record = validate_fact_record(raw)
            except Exception as exc:
                self._safety("SelectedFactQueryPort returned an invalid FactRecord", index=index, reason=str(exc))
            try:
                Decimal(record["normalized_value"])
            except (InvalidOperation, ValueError):
                self._safety("SelectedFactQueryPort returned a non-Decimal normalized_value", index=index)
            if not Decimal(record["normalized_value"]).is_finite():
                self._safety("SelectedFactQueryPort returned a non-finite normalized_value", index=index)
            records.append(record)
        fact_ids = [record["fact_id"] for record in records]
        if fact_ids != sorted(fact_ids) or len(fact_ids) != len(set(fact_ids)):
            self._safety("SelectedFactQueryPort results must have unique fact IDs in stable order")

        allowed_documents = set(filters["document_ids"])
        allowed_stocks = set(filters["stock_codes"])
        for record in records:
            if (
                record["canonical_metric"] != payload["canonical_metric"]
                or record["metric_year"] != payload["metric_year"]
                or record["normalized_unit"] != payload["normalized_unit"]
                or (filters["report_years"] and record["report_year"] not in filters["report_years"])
                or (allowed_documents and record["document_id"] not in allowed_documents)
                or (allowed_stocks and record["stock_code"] not in allowed_stocks)
            ):
                self._safety(
                    "SelectedFactQueryPort returned a row outside the validated allow-list",
                    fact_id=record["fact_id"],
                )
        units = {record["normalized_unit"] for record in records}
        if len(units) > 1:
            self._safety("QuerySpec result contains mixed units", units=sorted(units))
        return records

    def _rank_rows(
        self, records: Sequence[Mapping[str, Any]], payload: Mapping[str, Any]
    ) -> list[dict[str, Any]]:
        direction = payload.get("order_direction", "desc")
        ordered = sorted(
            records,
            key=lambda row: (
                Decimal(row["normalized_value"]),
                row["stock_code"],
                row["document_id"],
                row["fact_id"],
            ),
            reverse=direction == "desc",
        )
        # ``reverse=True`` would reverse textual tie breakers too.  Restore the
        # frozen rule: value follows direction; all identity ties are ascending.
        ordered = sorted(
            ordered,
            key=lambda row: (row["stock_code"], row["document_id"], row["fact_id"]),
        )
        ordered = sorted(
            ordered,
            key=lambda row: Decimal(row["normalized_value"]),
            reverse=direction == "desc",
        )
        return [
            self._record_row(record, payload, ordinal=index, operation="rank")
            for index, record in enumerate(ordered[: payload.get("limit", 1)], start=1)
        ]

    def _aggregate_rows(
        self, records: Sequence[Mapping[str, Any]], payload: Mapping[str, Any]
    ) -> list[dict[str, Any]]:
        if not records:
            return []
        operator = payload["aggregate_operator"]
        values = [Decimal(row["normalized_value"]) for row in records]
        if operator == "count":
            value = Decimal(len(values))
            result_unit = "count"
        elif operator == "sum":
            value = sum(values, Decimal(0))
            result_unit = payload["normalized_unit"]
        else:
            value = sum(values, Decimal(0)) / Decimal(len(values))
            result_unit = payload["normalized_unit"]
        contributing = [row["fact_id"] for row in records]
        measure_id = self._measure_id(payload, operation=f"aggregate:{operator}")
        derivation_inputs = [self._derivation_input(row) for row in records]
        row_seed = {
            "query_spec_id": payload["query_spec_id"],
            "measure_id": measure_id,
            "aggregate_operator": operator,
            "contributing_fact_ids": contributing,
        }
        return [{
            "query_spec_id": payload["query_spec_id"],
            "result_row_id": "sqlr_" + semantic_sha256(row_seed)[:16],
            "measure_id": measure_id,
            "entity_key": None,
            "document_id": None,
            "stock_code": None,
            "company": None,
            "metric_year": payload["metric_year"],
            "canonical_metric": payload["canonical_metric"],
            "value": self._decimal_text(value),
            "normalized_unit": result_unit,
            "operation": "aggregate",
            "rank": None,
            "aggregate_operator": operator,
            "contributing_fact_ids": contributing,
            "derivation_inputs": derivation_inputs,
        }]

    def _record_row(
        self,
        record: Mapping[str, Any],
        payload: Mapping[str, Any],
        *,
        ordinal: int,
        operation: str,
    ) -> dict[str, Any]:
        measure_id = self._measure_id(payload, operation=operation)
        seed = {
            "query_spec_id": payload["query_spec_id"],
            "measure_id": measure_id,
            "fact_id": record["fact_id"],
            "rank": ordinal,
        }
        return {
            "query_spec_id": payload["query_spec_id"],
            "result_row_id": "sqlr_" + semantic_sha256(seed)[:16],
            "measure_id": measure_id,
            "entity_key": record["stock_code"],
            "document_id": record["document_id"],
            "stock_code": record["stock_code"],
            "company": record["company"],
            "report_year": record["report_year"],
            "metric_year": record["metric_year"],
            "canonical_metric": record["canonical_metric"],
            "value": self._decimal_text(Decimal(record["normalized_value"])),
            "normalized_unit": record["normalized_unit"],
            "operation": operation,
            "rank": ordinal,
            "aggregate_operator": None,
            "contributing_fact_ids": [record["fact_id"]],
            "derivation_inputs": [self._derivation_input(record)],
        }

    @staticmethod
    def _derivation_input(record: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "fact_id": record["fact_id"],
            "document_id": record["document_id"],
            "stock_code": record["stock_code"],
            "company": record["company"],
            "report_year": record["report_year"],
            "metric_year": record["metric_year"],
            "canonical_metric": record["canonical_metric"],
            "normalized_value": SQLExecutor._decimal_text(Decimal(record["normalized_value"])),
            "normalized_unit": record["normalized_unit"],
            "source_table_id": record["source_table_id"],
            "source_line_start": record["source_line_start"],
            "source_line_end": record["source_line_end"],
        }

    @staticmethod
    def _coverage(records: Sequence[Mapping[str, Any]], payload: Mapping[str, Any]) -> dict[str, Any]:
        observed_documents = sorted({row["document_id"] for row in records})
        observed_entities = sorted({row["stock_code"] for row in records})
        expected_documents = sorted(payload["document_ids"])
        expected_entities = sorted(payload["entity_keys"])
        missing_documents = sorted(set(expected_documents) - set(observed_documents))
        missing_entities = sorted(set(expected_entities) - set(observed_entities))
        count_incomplete = (
            len(observed_documents) < payload["scope_document_count"]
            or len(observed_entities) < payload["scope_company_count"]
        )
        return {
            "expected_company_count": payload["scope_company_count"],
            "observed_company_count": len(observed_entities),
            "expected_document_count": payload["scope_document_count"],
            "observed_document_count": len(observed_documents),
            "expected_document_ids": expected_documents,
            "observed_document_ids": observed_documents,
            "missing_document_ids": missing_documents,
            "expected_entity_keys": expected_entities,
            "observed_entity_keys": observed_entities,
            "missing_entity_keys": missing_entities,
            "complete": not (missing_documents or missing_entities or count_incomplete),
        }

    @staticmethod
    def _normalized_spec(payload: Mapping[str, Any]) -> dict[str, Any]:
        result = {key: payload[key] for key in sorted(_COMMON_PAYLOAD_FIELDS)}
        if payload["query_kind"] == "rank":
            result.update({
                "order_direction": payload.get("order_direction", "desc"),
                "limit": payload.get("limit", 1),
            })
        else:
            result["aggregate_operator"] = payload["aggregate_operator"]
        return result

    @staticmethod
    def _measure_id(payload: Mapping[str, Any], *, operation: str) -> str:
        return "sqlm_" + semantic_sha256({
            "query_spec_id": payload["query_spec_id"],
            "canonical_metric": payload["canonical_metric"],
            "metric_year": payload["metric_year"],
            "normalized_unit": payload["normalized_unit"],
            "operation": operation,
        })[:16]

    @staticmethod
    def _decimal_text(value: Decimal) -> str:
        text = format(value, "f")
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        return text or "0"

    @staticmethod
    def _safe_text(value: Any, field: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise SQLExecutionError(
                "SQL_SAFETY_REJECTED", f"{field} must be a non-empty string", {}
            )
        if _UNSAFE_TEXT_RE.search(value):
            raise SQLExecutionError(
                "SQL_SAFETY_REJECTED", f"{field} contains a rejected SQL fragment", {"field": field}
            )
        return value

    @classmethod
    def _safe_string_array(cls, value: Any, field: str) -> list[str]:
        if not isinstance(value, list):
            cls._safety(f"{field} must be an array")
        result = [cls._safe_text(row, field) for row in value]
        if len(result) != len(set(result)):
            cls._safety(f"{field} must not contain duplicates")
        return result

    @staticmethod
    def _unsupported(message: str, **details: Any) -> None:
        raise SQLExecutionError("SQL_UNSUPPORTED_QUERY", message, details)

    @staticmethod
    def _safety(message: str, **details: Any) -> None:
        raise SQLExecutionError("SQL_SAFETY_REJECTED", message, details)


def execute_sql(
    subplan_or_payload: Mapping[str, Any], fact_query_port: SelectedFactQueryPort
) -> dict[str, Any]:
    """Functional convenience wrapper around :class:`SQLExecutor`."""

    return SQLExecutor(fact_query_port).execute(subplan_or_payload)


__all__ = [
    "QUERY_SPECS",
    "SQL_EXECUTION_SCHEMA",
    "SQLExecutionError",
    "SQLExecutor",
    "execute_sql",
]
