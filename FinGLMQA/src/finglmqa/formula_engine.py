"""Fail-closed Decimal formula execution for frozen Phase 8 plans.

The executor accepts only formula IDs registered by :class:`MetricCatalog`.
Formula text is descriptive metadata and is never evaluated.  Missing operands
produce discovery-only Phase 9 requests; this module never executes a fallback.
"""

from __future__ import annotations

import hashlib
import json
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN, localcontext
from typing import Any, Callable, Mapping, Sequence

from .contracts import (
    ContractValidationError,
    SCHEMA_MISSING_FACT_REQUEST,
    SCHEMA_SUBPLAN_RESULT,
    canonical_json_bytes,
    make_requirement_id,
    validate_missing_fact_request,
    validate_subplan_result,
)
from .metric_catalog import FormulaDefinition, MetricCatalog
from .ports import (
    FactLookupPort,
    FallbackCandidateIndexPort,
    validate_fact_lookup_request,
    validate_fact_lookup_result,
)


FORMULA_ENGINE_VERSION = "phase8-decimal-formula-v1"
FORMULA_VERSION = "1.0.0"
DECIMAL_PRECISION = 50
DECIMAL_ROUNDING = "ROUND_HALF_EVEN"


_GROWTH_IDS = frozenset({
    "revenue_growth_rate.v1",
    "total_assets_growth_rate.v1",
    "net_assets_growth_rate.v1",
    "parent_net_profit_growth_rate.v1",
    "sales_expense_growth_rate.v1",
    "operating_profit_growth_rate.v1",
    "administrative_expense_growth_rate.v1",
    "current_liabilities_growth_rate.v1",
    "intangible_assets_growth_rate.v1",
    "financial_expense_growth_rate.v1",
    "rd_expense_growth_rate.v1",
    "net_profit_growth_rate.v1",
    "total_liabilities_growth_rate.v1",
    "cash_growth_rate.v1",
    "investment_income_growth_rate.v1",
    "fixed_assets_growth_rate.v1",
    "cash_equivalents_growth_rate.v1",
})

_SIMPLE_RATIO_IDS = frozenset({
    "operating_cost_ratio.v1",
    "investment_income_revenue_ratio.v1",
    "administrative_expense_ratio.v1",
    "financial_expense_ratio.v1",
    "operating_margin.v1",
    "net_profit_margin.v1",
    "current_ratio.v1",
    "cash_ratio.v1",
    "debt_asset_ratio.v1",
    "current_liabilities_ratio.v1",
    "noncurrent_liabilities_ratio.v1",
    "rd_revenue_ratio.v1",
    "rd_profit_ratio.v1",
    "rd_staff_ratio.v1",
})

_SPECIAL_OPERATOR_BY_ID = {
    "gross_margin.v1": "gross_margin",
    "quick_ratio.v1": "quick_ratio",
    "three_expense_ratio.v1": "three_expense_ratio",
    "rd_expense_share.v1": "rd_expense_share",
    "postgraduate_staff_ratio.v1": "postgraduate_staff_ratio",
}


class FormulaExecutionError(ValueError):
    """Raised only for programmer errors in direct calculator use."""


def _decimal_text(value: Decimal) -> str:
    if not value.is_finite():
        raise FormulaExecutionError("formula produced a non-finite Decimal")
    if value.is_zero():
        return "0"
    return format(value, "f")


def _stable_id(prefix: str, value: Any, length: int = 20) -> str:
    return prefix + hashlib.sha256(canonical_json_bytes(value)).hexdigest()[:length]


def _growth(values: Mapping[str, Decimal]) -> Decimal:
    return (values["current"] - values["previous"]) / values["previous"]


def _simple_ratio(values: Mapping[str, Decimal]) -> Decimal:
    return values["numerator"] / values["denominator"]


def _gross_margin(values: Mapping[str, Decimal]) -> Decimal:
    return (values["revenue"] - values["cost"]) / values["revenue"]


def _quick_ratio(values: Mapping[str, Decimal]) -> Decimal:
    return (values["current_assets"] - values["inventory"]) / values["denominator"]


def _three_expense_ratio(values: Mapping[str, Decimal]) -> Decimal:
    numerator = values["sales"] + values["administrative"] + values["financial"]
    return numerator / values["denominator"]


def _rd_expense_share(values: Mapping[str, Decimal]) -> Decimal:
    denominator = values["rd"] + values["sales"] + values["financial"] + values["administrative"]
    return values["rd"] / denominator


def _postgraduate_staff_ratio(values: Mapping[str, Decimal]) -> Decimal:
    return (values["master"] + values["doctor"]) / values["denominator"]


_CALCULATORS: dict[str, Callable[[Mapping[str, Decimal]], Decimal]] = {
    "growth": _growth,
    "simple_ratio": _simple_ratio,
    "gross_margin": _gross_margin,
    "quick_ratio": _quick_ratio,
    "three_expense_ratio": _three_expense_ratio,
    "rd_expense_share": _rd_expense_share,
    "postgraduate_staff_ratio": _postgraduate_staff_ratio,
}


def _operator_for(formula_id: str) -> str | None:
    if formula_id in _GROWTH_IDS:
        return "growth"
    if formula_id in _SIMPLE_RATIO_IDS:
        return "simple_ratio"
    return _SPECIAL_OPERATOR_BY_ID.get(formula_id)


class FormulaExecutor:
    """Execute a ready ``formula_compute`` SubPlan over exact selected facts.

    ``fallback_candidates`` is a discovery index only.  It may attach stable
    table IDs to a MissingFactRequest; it is never asked for values and cannot
    turn a missing operand into an executable fact.
    """

    def __init__(
        self,
        facts: FactLookupPort,
        *,
        metric_catalog: MetricCatalog | None = None,
        fallback_candidates: FallbackCandidateIndexPort | None = None,
    ) -> None:
        self.facts = facts
        self.metric_catalog = metric_catalog or MetricCatalog()
        self.fallback_candidates = fallback_candidates
        registered = {formula.formula_id for formula in self.metric_catalog.formulas}
        executable = _GROWTH_IDS | _SIMPLE_RATIO_IDS | set(_SPECIAL_OPERATOR_BY_ID)
        if registered != executable:
            missing = sorted(registered - executable)
            stale = sorted(executable - registered)
            raise FormulaExecutionError(
                f"formula dispatch/catalog mismatch; missing={missing}, stale={stale}"
            )

    def execute(self, subplan: Mapping[str, Any]) -> dict[str, Any]:
        """Return a contract-valid ``SubPlanResult`` without raising on data failures."""

        invalid = self._validate_subplan(subplan)
        if invalid is not None:
            safe_subplan = subplan if isinstance(subplan, Mapping) else {}
            return self._failure(safe_subplan, "error", "INVALID_REQUEST", invalid)
        payload = subplan["payload"]
        formula_id = payload["formula_id"]
        definition = self.metric_catalog.formula(formula_id)
        operator_id = _operator_for(formula_id)
        if definition is None or operator_id is None:
            return self._failure(
                subplan,
                "unsupported",
                "METRIC_UNRECOGNIZED",
                "Formula ID is not in the frozen Phase 8 registry",
                trace={"formula_id": formula_id},
            )
        if payload["normalized_unit"] != definition.normalized_unit:
            return self._failure(
                subplan,
                "blocked",
                "FORMULA_UNIT_MISMATCH",
                "Planned formula unit does not match the metric catalog",
                trace={
                    "formula_id": formula_id,
                    "planned_unit": payload["normalized_unit"],
                    "catalog_unit": definition.normalized_unit,
                },
            )

        lookups: list[dict[str, Any]] = []
        missing: list[tuple[str, str, int, str]] = []
        conflict: dict[str, Any] | None = None
        for role, metric, year_offset in definition.operands:
            expected_unit = self.metric_catalog.expected_unit(metric)
            metric_year = payload["target_year"] + year_offset
            if expected_unit is None:
                conflict = {
                    "failure_code": "UNIT_AMBIGUOUS",
                    "message": "Formula operand has no unique catalog unit",
                    "details": {"operand_role": role, "canonical_metric": metric},
                }
                break
            request = self._lookup_request(subplan, role, metric, metric_year, expected_unit)
            try:
                response = self.facts.lookup_fact(request)
                validate_fact_lookup_result(response)
            except Exception as exc:  # repository boundary failures are fail-closed
                return self._failure(
                    subplan,
                    "error",
                    "INTERNAL_ERROR",
                    "FactLookupPort returned an invalid result",
                    trace={"lookup_requirement_id": request["requirement_id"], "exception_type": type(exc).__name__},
                )
            if response["requirement_id"] != request["requirement_id"]:
                return self._failure(
                    subplan,
                    "error",
                    "INTERNAL_ERROR",
                    "FactLookupPort requirement ID does not match the request",
                    trace={"lookup_requirement_id": request["requirement_id"]},
                )
            if response["status"] == "not_found":
                missing.append((role, metric, metric_year, expected_unit))
                lookups.append({"role": role, "request": request, "response": response})
                continue
            if response["status"] == "ambiguous":
                conflict = {
                    "failure_code": "FACT_UNRESOLVED_CONFLICT",
                    "message": "Formula operand lookup returned multiple selected facts",
                    "details": {"operand_role": role, "fact_ids": [row["fact_id"] for row in response["records"]]},
                }
                lookups.append({"role": role, "request": request, "response": response})
                break

            record = response["records"][0]
            mismatch = self._fact_mismatch(request, record)
            if mismatch is not None:
                code = "FORMULA_UNIT_MISMATCH" if mismatch == "normalized_unit" else "FACT_UNRESOLVED_CONFLICT"
                conflict = {
                    "failure_code": code,
                    "message": "Formula operand fact does not match its exact lookup identity",
                    "details": {"operand_role": role, "mismatch_field": mismatch, "fact_id": record["fact_id"]},
                }
                lookups.append({"role": role, "request": request, "response": response})
                break
            try:
                provenance_payload = json.loads(record["provenance_json"])
            except (json.JSONDecodeError, TypeError):
                provenance_payload = None
            provenance_is_structured = isinstance(provenance_payload, dict) or (
                isinstance(provenance_payload, list)
                and bool(provenance_payload)
                and all(isinstance(item, dict) for item in provenance_payload)
            )
            if not provenance_is_structured:
                return self._failure(
                    subplan,
                    "blocked",
                    "PROVENANCE_VALIDATION_FAILED",
                    "Formula operand provenance is not a valid structured JSON record",
                    details={"operand_role": role, "fact_id": record["fact_id"]},
                    trace=self._lookup_trace(formula_id, operator_id, [*lookups, {
                        "role": role, "request": request, "response": response,
                    }]),
                )
            try:
                value = Decimal(record["normalized_value"])
            except InvalidOperation:
                conflict = {
                    "failure_code": "FACT_UNRESOLVED_CONFLICT",
                    "message": "Formula operand is not a valid Decimal",
                    "details": {"operand_role": role, "fact_id": record["fact_id"]},
                }
                lookups.append({"role": role, "request": request, "response": response})
                break
            if not value.is_finite():
                conflict = {
                    "failure_code": "FACT_UNRESOLVED_CONFLICT",
                    "message": "Formula operand is not finite",
                    "details": {"operand_role": role, "fact_id": record["fact_id"]},
                }
                lookups.append({"role": role, "request": request, "response": response})
                break
            lookups.append({"role": role, "request": request, "response": response, "value": value})

        if conflict is not None:
            return self._failure(
                subplan,
                "blocked",
                conflict["failure_code"],
                conflict["message"],
                details=conflict["details"],
                trace=self._lookup_trace(formula_id, operator_id, lookups),
            )
        if missing:
            requests = [
                self._missing_request(subplan, definition, role, metric, metric_year, expected_unit)
                for role, metric, metric_year, expected_unit in missing
            ]
            return self._failure(
                subplan,
                "fallback_required",
                "FORMULA_OPERAND_MISSING",
                "One or more exact formula operands are missing; Phase 8 does not execute fallback",
                details={"missing_operand_roles": [row[0] for row in missing]},
                missing_fact_requests=requests,
                trace=self._lookup_trace(formula_id, operator_id, lookups),
            )

        values = {row["role"]: row["value"] for row in lookups}
        denominator_roles, denominator_value = self._denominator(operator_id, values)
        if denominator_value.is_zero():
            return self._failure(
                subplan,
                "blocked",
                "FORMULA_ZERO_DENOMINATOR",
                "Formula denominator is zero",
                details={"denominator_roles": list(denominator_roles)},
                trace=self._lookup_trace(formula_id, operator_id, lookups),
            )
        try:
            with localcontext() as context:
                context.prec = DECIMAL_PRECISION
                context.rounding = ROUND_HALF_EVEN
                value = _CALCULATORS[operator_id](values)
        except (ArithmeticError, KeyError, FormulaExecutionError) as exc:
            return self._failure(
                subplan,
                "error",
                "INTERNAL_ERROR",
                "Registered formula execution failed",
                trace={**self._lookup_trace(formula_id, operator_id, lookups), "exception_type": type(exc).__name__},
            )

        citations, operand_rows = self._citations_and_operands(subplan, definition, lookups)
        derivation_citation = self._derivation_citation(subplan, definition, operator_id, citations, lookups)
        citations.append(derivation_citation)
        result = {
            "value": _decimal_text(value),
            "normalized_unit": definition.normalized_unit,
            "formula_id": definition.formula_id,
            "formula_version": FORMULA_VERSION,
            "canonical_formula": definition.canonical_formula,
            "target_year": payload["target_year"],
            "operands": operand_rows,
            "derivation": {
                "engine_version": FORMULA_ENGINE_VERSION,
                "operator_id": operator_id,
                "operand_roles": [role for role, _, _ in definition.operands],
                "decimal_precision": DECIMAL_PRECISION,
                "decimal_rounding": DECIMAL_ROUNDING,
                "percentage_scale": "100" if definition.normalized_unit == "ratio" else "1",
                "default_render_precision": 2,
                "default_render_rounding": "ROUND_HALF_UP",
            },
        }
        output = self._base_result(subplan)
        output.update({
            "status": "ok",
            "result": result,
            "citations": citations,
            "trace": self._lookup_trace(formula_id, operator_id, lookups),
        })
        validate_subplan_result(output)
        return output

    @staticmethod
    def _validate_subplan(subplan: Mapping[str, Any]) -> str | None:
        if not isinstance(subplan, Mapping):
            return "SubPlan must be an object"
        if not isinstance(subplan.get("subplan_id"), str) or not subplan["subplan_id"]:
            return "SubPlan.subplan_id is required"
        if subplan.get("backend") != "formula" or subplan.get("operation") != "formula_compute":
            return "FormulaExecutor accepts only formula/formula_compute SubPlans"
        if subplan.get("planning_state") != "ready":
            return "FormulaExecutor accepts only ready SubPlans"
        payload = subplan.get("payload")
        if not isinstance(payload, Mapping):
            return "Formula SubPlan.payload must be an object"
        required_strings = ("entity_key", "document_id", "stock_code", "formula_id", "normalized_unit")
        if any(not isinstance(payload.get(field), str) or not payload[field] for field in required_strings):
            return "Formula payload identity/formula/unit fields must be non-empty strings"
        for field in ("report_year", "target_year"):
            if isinstance(payload.get(field), bool) or not isinstance(payload.get(field), int):
                return f"Formula payload {field} must be an integer"
        return None

    def _lookup_request(
        self,
        subplan: Mapping[str, Any],
        role: str,
        metric: str,
        metric_year: int,
        normalized_unit: str,
    ) -> dict[str, Any]:
        payload = subplan["payload"]
        identity = [
            subplan["subplan_id"], payload["document_id"], payload["stock_code"],
            payload["report_year"], metric_year, metric, normalized_unit, role,
        ]
        request = {
            "schema_version": "finglmqa.phase8.fact_lookup_request.v1",
            "requirement_id": _stable_id("lookup_", identity),
            "document_id": payload["document_id"],
            "stock_code": payload["stock_code"],
            "report_year": payload["report_year"],
            "metric_year": metric_year,
            "canonical_metric": metric,
            "normalized_unit": normalized_unit,
        }
        validate_fact_lookup_request(request)
        return request

    @staticmethod
    def _fact_mismatch(request: Mapping[str, Any], record: Mapping[str, Any]) -> str | None:
        for field in (
            "document_id", "stock_code", "report_year", "metric_year",
            "canonical_metric", "normalized_unit",
        ):
            if record[field] != request[field]:
                return field
        return None

    def _missing_request(
        self,
        subplan: Mapping[str, Any],
        formula: FormulaDefinition,
        role: str,
        metric: str,
        metric_year: int,
        normalized_unit: str,
    ) -> dict[str, Any]:
        payload = subplan["payload"]
        request = {
            "schema_version": SCHEMA_MISSING_FACT_REQUEST,
            "requirement_id": "pending",
            "origin_operation": "formula_compute",
            "formula_id": formula.formula_id,
            "operand_role": role,
            "subplan_id": subplan["subplan_id"],
            "document_id": payload["document_id"],
            "stock_code": payload["stock_code"],
            "report_year": payload["report_year"],
            "metric_year": metric_year,
            "canonical_metric": metric,
            "normalized_unit": normalized_unit,
            "candidate_table_ids": [],
        }
        request["requirement_id"] = make_requirement_id(request)
        if self.fallback_candidates is not None:
            try:
                candidates = self.fallback_candidates.candidate_table_ids(request)
                if not isinstance(candidates, list) or any(
                    not isinstance(table_id, str) or not table_id for table_id in candidates
                ):
                    raise ContractValidationError("FallbackCandidateIndexPort returned invalid table IDs")
                request["candidate_table_ids"] = sorted(set(candidates))
            except Exception:
                # Candidate discovery is advisory.  The uniquely bound request is
                # still valid and Phase 9 may build/retry its own candidate index.
                request["candidate_table_ids"] = []
        validate_missing_fact_request(request)
        return request

    @staticmethod
    def _denominator(
        operator_id: str, values: Mapping[str, Decimal]
    ) -> tuple[tuple[str, ...], Decimal]:
        if operator_id == "growth":
            return ("previous",), values["previous"]
        if operator_id == "gross_margin":
            return ("revenue",), values["revenue"]
        if operator_id == "rd_expense_share":
            roles = ("rd", "sales", "financial", "administrative")
            return roles, sum((values[role] for role in roles), Decimal("0"))
        return ("denominator",), values["denominator"]

    @staticmethod
    def _lookup_trace(
        formula_id: str,
        operator_id: str,
        lookups: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        return {
            "formula_engine_version": FORMULA_ENGINE_VERSION,
            "formula_id": formula_id,
            "operator_id": operator_id,
            "operand_lookups": [
                {
                    "operand_role": row["role"],
                    "requirement_id": row["request"]["requirement_id"],
                    "status": row["response"]["status"],
                    "repository_fingerprint": row["response"]["repository_fingerprint"],
                }
                for row in lookups
            ],
        }

    @staticmethod
    def _citations_and_operands(
        subplan: Mapping[str, Any],
        formula: FormulaDefinition,
        lookups: Sequence[Mapping[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        citations: list[dict[str, Any]] = []
        operands: list[dict[str, Any]] = []
        payload = subplan["payload"]
        for row in lookups:
            record = row["response"]["records"][0]
            citation_id = _stable_id("cite_fact_", [subplan["subplan_id"], row["role"], record["fact_id"]])
            try:
                provenance_payload = json.loads(record["provenance_json"])
            except (json.JSONDecodeError, TypeError):
                raise AssertionError("operand provenance was not validated before calculation")
            if not (
                isinstance(provenance_payload, dict)
                or (
                    isinstance(provenance_payload, list)
                    and provenance_payload
                    and all(isinstance(item, dict) for item in provenance_payload)
                )
            ):
                raise AssertionError("operand provenance must be structured JSON")
            citations.append({
                "citation_id": citation_id,
                "citation_kind": "fact",
                "subplan_id": subplan["subplan_id"],
                "entity_key": payload["entity_key"],
                "document_id": payload["document_id"],
                "source_citation_ids": [],
                "provenance": {
                    "fact_id": record["fact_id"],
                    "source_table_id": record["source_table_id"],
                    "source_line_start": record["source_line_start"],
                    "source_line_end": record["source_line_end"],
                    "fact_provenance": provenance_payload,
                },
            })
            operands.append({
                "operand_role": row["role"],
                "canonical_metric": record["canonical_metric"],
                "metric_year": record["metric_year"],
                "value": record["normalized_value"],
                "normalized_unit": record["normalized_unit"],
                "fact_id": record["fact_id"],
                "citation_ids": [citation_id],
                "fact_record": dict(record),
            })
        return citations, operands

    @staticmethod
    def _derivation_citation(
        subplan: Mapping[str, Any],
        formula: FormulaDefinition,
        operator_id: str,
        citations: Sequence[Mapping[str, Any]],
        lookups: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        payload = subplan["payload"]
        source_ids = [row["citation_id"] for row in citations]
        citation_id = _stable_id(
            "cite_formula_",
            [subplan["subplan_id"], formula.formula_id, payload["target_year"], source_ids],
        )
        return {
            "citation_id": citation_id,
            "citation_kind": "formula_derivation",
            "subplan_id": subplan["subplan_id"],
            "entity_key": payload["entity_key"],
            "document_id": payload["document_id"],
            "source_citation_ids": source_ids,
            "provenance": {
                "formula_id": formula.formula_id,
                "formula_version": FORMULA_VERSION,
                "canonical_formula": formula.canonical_formula,
                "operator_id": operator_id,
                "target_year": payload["target_year"],
                "operand_fact_ids": [row["response"]["records"][0]["fact_id"] for row in lookups],
                "engine_version": FORMULA_ENGINE_VERSION,
            },
        }

    @staticmethod
    def _base_result(subplan: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_SUBPLAN_RESULT,
            "subplan_id": str(subplan.get("subplan_id") or "invalid_subplan"),
            "backend": "formula",
            "operation": "formula_compute",
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
            "trace": {"formula_engine_version": FORMULA_ENGINE_VERSION},
        }

    def _failure(
        self,
        subplan: Mapping[str, Any],
        status: str,
        code: str,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
        missing_fact_requests: Sequence[Mapping[str, Any]] = (),
        trace: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        output = self._base_result(subplan)
        output.update({
            "status": status,
            "failure_code": code,
            "errors": [{"failure_code": code, "message": message, "details": dict(details or {})}],
            "missing_fact_requests": [dict(row) for row in missing_fact_requests],
            "trace": dict(trace or {"formula_engine_version": FORMULA_ENGINE_VERSION}),
        })
        validate_subplan_result(output)
        return output


__all__ = [
    "DECIMAL_PRECISION",
    "FORMULA_ENGINE_VERSION",
    "FORMULA_VERSION",
    "FormulaExecutionError",
    "FormulaExecutor",
]
