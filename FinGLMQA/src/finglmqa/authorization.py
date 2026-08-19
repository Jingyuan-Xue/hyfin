"""Build the frozen numeric authorization boundary for Phase 8 evidence."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any, Mapping, Sequence

from .contracts import (
    SCHEMA_NUMERIC_AUTHORIZATION,
    SCHEMA_NUMERIC_AUTHORIZATION_SET,
    is_subplan_result_usable,
    semantic_sha256,
    validate_numeric_authorization_set,
)


AUTHORIZATION_BUILDER_VERSION = "phase8-numeric-authorization-v1"


def _decimal_text(value: Decimal) -> str:
    if not value.is_finite():
        raise ValueError("numeric authorization cannot contain a non-finite Decimal")
    if value.is_zero():
        return "0"
    return format(value, "f")


def _renderings(value: Decimal, unit: str) -> list[str]:
    raw = _decimal_text(value)
    values = {raw}
    if value == value.to_integral_value():
        values.add(format(value, ".0f"))
    else:
        values.add(format(value.normalize(), "f"))
        values.add(format(value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP), ".2f"))
    if unit == "ratio":
        percent = (value * Decimal("100")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        values.update({format(percent, ".2f"), f"{percent:.2f}%"})
    elif unit in {"元", "股"}:
        values.add(f"{raw}{unit}")
        for scale, suffix in ((Decimal("10000"), f"万{unit}"), (Decimal("100000000"), f"亿{unit}")):
            scaled = value / scale
            exact = _decimal_text(scaled)
            rounded = format(scaled.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP), ".2f")
            values.update({f"{exact}{suffix}", f"{rounded}{suffix}"})
    elif unit == "元/股":
        values.add(f"{raw}元/股")
    return sorted(values)


def build_numeric_authorization_set(
    subplans: Sequence[Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Authorize only usable, provenance-backed structured outputs."""

    plan_by_id = {str(row["subplan_id"]): row for row in subplans}
    items: list[dict[str, Any]] = []
    for result in results:
        if result.get("backend") == "evidence" or not is_subplan_result_usable(result):
            continue
        plan = plan_by_id.get(str(result.get("subplan_id")))
        if plan is None:
            continue
        if result["backend"] == "fact" and result["operation"] == "fact_lookup":
            items.append(_fact_authorization(plan, result))
        elif result["backend"] == "formula":
            items.extend(_formula_authorizations(plan, result))
        elif result["backend"] == "sql":
            items.extend(_sql_authorizations(plan, result))
        # Textual metadata is intentionally never a numeric authorization.
    items = _deduplicate_authorizations(items)
    items.sort(key=lambda row: (row["source_subplan_id"], row["source_result_row"], row["authorization_id"]))
    payload = {
        "schema_version": SCHEMA_NUMERIC_AUTHORIZATION_SET,
        "items": items,
        "set_fingerprint": "pending",
    }
    payload["set_fingerprint"] = semantic_sha256({
        "schema_version": payload["schema_version"], "items": items,
    })
    validate_numeric_authorization_set(payload)
    return payload


def _fact_authorization(plan: Mapping[str, Any], result: Mapping[str, Any]) -> dict[str, Any]:
    row = result["result"]
    value = _decimal(row["value"])
    source = [citation["citation_id"] for citation in result["citations"] if citation["citation_kind"] == "fact"]
    item = _base(plan, result, row, value, row["normalized_unit"], source, source, 0)
    item["measure"] = {
        "kind": "canonical_fact",
        "canonical_metric": row["canonical_metric"],
        "metric_year": row["metric_year"],
    }
    return _finish(item)


def _formula_authorizations(plan: Mapping[str, Any], result: Mapping[str, Any]) -> list[dict[str, Any]]:
    row = result["result"]
    value = _decimal(row["value"])
    source = sorted({
        citation_id for operand in row["operands"] for citation_id in operand["citation_ids"]
    })
    provenance = sorted(
        citation["citation_id"] for citation in result["citations"]
        if citation["citation_kind"] == "formula_derivation"
    )
    item = _base(plan, result, {
        "entity_key": plan.get("entity_key"),
        "company": row["operands"][0]["fact_record"]["company"],
        "document_id": plan["payload"]["document_id"],
    }, value, row["normalized_unit"], source, provenance, 0)
    item["measure"] = {
        "kind": "formula_result",
        "formula_id": row["formula_id"],
        "formula_version": row["formula_version"],
        "target_year": row["target_year"],
        "operand_years": sorted({operand["metric_year"] for operand in row["operands"]}),
    }
    items = [_finish(item)]
    for index, operand in enumerate(row["operands"], start=1):
        record = operand["fact_record"]
        operand_value = _decimal(operand["value"])
        operand_item = _base(
            plan,
            result,
            {
                "entity_key": plan.get("entity_key"),
                "company": record["company"],
                "document_id": record["document_id"],
            },
            operand_value,
            operand["normalized_unit"],
            operand["citation_ids"],
            operand["citation_ids"],
            index,
        )
        operand_item["measure"] = {
            "kind": "canonical_fact",
            "canonical_metric": operand["canonical_metric"],
            "metric_year": operand["metric_year"],
        }
        items.append(_finish(operand_item))
    return items


def _sql_authorizations(plan: Mapping[str, Any], result: Mapping[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    citations = result["citations"]
    for index, row in enumerate(result["result"]["rows"]):
        value = _decimal(row["value"])
        fact_ids = set(row["contributing_fact_ids"])
        source = sorted(
            citation["citation_id"] for citation in citations
            if citation["citation_kind"] == "fact" and citation["provenance"].get("fact_id") in fact_ids
        )
        provenance = sorted(
            citation["citation_id"] for citation in citations
            if citation["citation_kind"] == "sql_derivation"
            and citation["provenance"].get("result_row_id") == row["result_row_id"]
        )
        item = _base(plan, result, row, value, row["normalized_unit"], source, provenance, index)
        item["measure"] = {
            "kind": "sql_result",
            "query_spec_id": row["query_spec_id"],
            "result_row_id": row["result_row_id"],
            "measure_id": row["measure_id"],
            "canonical_metric": row["canonical_metric"],
            "metric_year": row["metric_year"],
        }
        items.append(_finish(item))
    return items


def _base(
    plan: Mapping[str, Any],
    result: Mapping[str, Any],
    row: Mapping[str, Any],
    value: Decimal,
    unit: str,
    source_ids: Sequence[str],
    provenance_ids: Sequence[str],
    row_index: int,
) -> dict[str, Any]:
    if not source_ids or not provenance_ids:
        raise ValueError("usable structured result lacks numeric authorization provenance")
    return {
        "schema_version": SCHEMA_NUMERIC_AUTHORIZATION,
        "authorization_id": "pending",
        "source_subplan_id": result["subplan_id"],
        "source_backend": result["backend"],
        "source_result_row": row_index,
        "entity_key": str(row.get("entity_key") or plan.get("entity_key") or "corpus"),
        "company": str(row.get("company") or row.get("stock_code") or plan.get("entity_key") or "corpus"),
        "document_id": row.get("document_id") or (plan.get("payload") or {}).get("document_id"),
        "measure": {},
        "normalized_value": _decimal_text(value),
        "normalized_unit": str(unit),
        "output_kind": "percentage_ratio" if unit == "ratio" else "decimal",
        "precision": 2,
        "rounding": "ROUND_HALF_UP",
        "allowed_renderings": _renderings(value, str(unit)),
        "source_citation_ids": sorted(set(source_ids)),
        "provenance_citation_ids": sorted(set(provenance_ids)),
    }


def _finish(item: dict[str, Any]) -> dict[str, Any]:
    unhashed = dict(item)
    unhashed.pop("authorization_id", None)
    item["authorization_id"] = "auth_" + semantic_sha256(unhashed)[:20]
    return item


def _deduplicate_authorizations(items: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Avoid ambiguous duplicate grants for the same value and measure.

    A direct fact SubPlan is preferred over the same fact exposed as a formula
    operand. Formula operands remain authorized when the bundle contains only
    a formula plus narrative evidence.
    """

    priority = {"fact": 0, "formula": 1, "sql": 2}

    def identity(item: Mapping[str, Any]) -> str:
        return semantic_sha256({
            "entity_key": item["entity_key"],
            "document_id": item["document_id"],
            "measure": item["measure"],
            "normalized_value": item["normalized_value"],
            "normalized_unit": item["normalized_unit"],
        })

    selected: dict[str, dict[str, Any]] = {}
    for item in sorted(
        items,
        key=lambda row: (
            priority[row["source_backend"]], row["source_subplan_id"],
            row["source_result_row"], row["authorization_id"],
        ),
    ):
        selected.setdefault(identity(item), item)
    return list(selected.values())


def _decimal(value: Any) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError("structured value is not a Decimal") from exc
    if not result.is_finite():
        raise ValueError("structured value is not finite")
    return result


__all__ = [
    "AUTHORIZATION_BUILDER_VERSION",
    "build_numeric_authorization_set",
]
