"""Deterministic Phase 8 answer composition and provenance validation.

The composer is deliberately execution-blind.  It accepts the already-frozen
``CompositionPlan`` and one result per executable SubPlan, validates that every
claim/citation stays inside its declared entity/document scope, derives the
frozen quorum/status policy, and renders sectioned output.  It never creates a
new SubPlan and evidence is never used for a structured comparison.
"""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import re
from typing import Any, Mapping, Sequence

from .contracts import (
    SCHEMA_ANSWER,
    SCHEMA_SUBPLAN_RESULT,
    ContractValidationError,
    canonical_json_bytes,
    is_subplan_result_usable,
    semantic_sha256,
    validate_composition_plan,
    validate_qa_answer,
    validate_subplan_result,
)
from .errors import dominant_failed_status, status_for_blocked_plan
from .metric_catalog import FORMULAS, METADATA_FIELDS


COMPOSER_VERSION = "1.0.0"
_CROSS_ENTITY_INFERENCE_RE = re.compile(
    r"(?:主要(?:是)?因为|原因(?:是|在于)|导致|因此|从而|优于|劣于|高于|低于|超过|"
    r"不及|相比|相较|较.{0,12}(?:高|低|多|少|好|差|快|慢))"
)
_FORMULA_LABELS = {row.formula_id: row.aliases[0] for row in FORMULAS}
_METADATA_LABELS = {field: aliases[0] for field, aliases in METADATA_FIELDS.items()}


def _contains_supplemental_marker(value: Any) -> bool:
    """Detect Phase 9 provenance without changing the unmarked render path."""

    if isinstance(value, Mapping):
        if value.get("fact_source") == "supplemental_tabgr":
            return True
        return any(_contains_supplemental_marker(child) for child in value.values())
    if isinstance(value, list):
        return any(_contains_supplemental_marker(child) for child in value)
    return False


@dataclass(frozen=True)
class ProvenanceViolation:
    """One deterministic global provenance or answer-scope violation."""

    violation_code: str
    message: str
    subplan_id: str | None = None
    details: Mapping[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "violation_code": self.violation_code,
            "message": self.message,
            "subplan_id": self.subplan_id,
            "details": dict(self.details or {}),
        }
        return payload


class GlobalAnswerProvenanceValidator:
    """Validate backend results against their frozen SubPlan scopes."""

    validator_version = "1.0.0"

    def validate(
        self,
        composition_plan: Mapping[str, Any],
        subplan_results: Sequence[Mapping[str, Any]],
    ) -> list[ProvenanceViolation]:
        plan = validate_composition_plan(deepcopy(dict(composition_plan)))
        plan_by_id = {row["subplan_id"]: row for row in plan["subplans"]}
        violations: list[ProvenanceViolation] = []
        seen_result_ids: set[str] = set()
        global_citation_ids: dict[str, str] = {}

        for ordinal, raw_result in enumerate(subplan_results):
            if not isinstance(raw_result, Mapping):
                violations.append(ProvenanceViolation(
                    "malformed_subplan_result",
                    "A backend result is not an object.",
                    details={"result_ordinal": ordinal},
                ))
                continue
            result = deepcopy(dict(raw_result))
            subplan_id = result.get("subplan_id")
            if not isinstance(subplan_id, str) or subplan_id not in plan_by_id:
                violations.append(ProvenanceViolation(
                    "unknown_subplan_result",
                    "A backend result does not reference a frozen SubPlan.",
                    subplan_id=subplan_id if isinstance(subplan_id, str) else None,
                    details={"result_ordinal": ordinal},
                ))
                continue
            if subplan_id in seen_result_ids:
                violations.append(ProvenanceViolation(
                    "duplicate_subplan_result",
                    "More than one backend result was supplied for a SubPlan.",
                    subplan_id=subplan_id,
                ))
                continue
            seen_result_ids.add(subplan_id)
            subplan = plan_by_id[subplan_id]

            try:
                validate_subplan_result(result)
            except ContractValidationError as exc:
                violations.append(ProvenanceViolation(
                    "invalid_subplan_result_contract",
                    "A backend result violates the frozen SubPlanResult contract.",
                    subplan_id=subplan_id,
                    details={"contract_error": str(exc)},
                ))
                continue

            for field in ("backend", "operation", "planning_state"):
                if result[field] != subplan[field]:
                    violations.append(ProvenanceViolation(
                        "subplan_identity_mismatch",
                        "Backend output identity differs from its frozen SubPlan.",
                        subplan_id=subplan_id,
                        details={
                            "field": field,
                            "planned": subplan[field],
                            "actual": result[field],
                        },
                    ))
            if subplan["planning_state"] == "blocked" and result["execution_state"] != "not_executed":
                violations.append(ProvenanceViolation(
                    "blocked_subplan_executed",
                    "A planning-blocked SubPlan was executed.",
                    subplan_id=subplan_id,
                ))

            if result.get("failure_code") == "PROVENANCE_VALIDATION_FAILED" or any(
                isinstance(row, Mapping)
                and row.get("failure_code") == "PROVENANCE_VALIDATION_FAILED"
                for row in result.get("errors", [])
            ):
                violations.append(ProvenanceViolation(
                    "backend_provenance_failure",
                    "A backend reported a provenance validation failure.",
                    subplan_id=subplan_id,
                ))

            self._validate_result_scope(
                plan, subplan, result, global_citation_ids, violations
            )
        return violations

    def _validate_result_scope(
        self,
        plan: Mapping[str, Any],
        subplan: Mapping[str, Any],
        result: Mapping[str, Any],
        global_citation_ids: dict[str, str],
        violations: list[ProvenanceViolation],
    ) -> None:
        subplan_id = subplan["subplan_id"]
        expected_entity = subplan["entity_key"]
        allowed_documents = set(subplan["declared_scope"]["document_ids"])
        local_citations: dict[str, Mapping[str, Any]] = {}

        for citation in result["citations"]:
            citation_id = citation["citation_id"]
            if citation_id in local_citations:
                violations.append(ProvenanceViolation(
                    "duplicate_citation_id",
                    "A SubPlan emitted a duplicate citation ID.",
                    subplan_id=subplan_id,
                    details={"citation_id": citation_id},
                ))
            local_citations[citation_id] = citation
            prior_owner = global_citation_ids.get(citation_id)
            if prior_owner is not None and prior_owner != subplan_id:
                violations.append(ProvenanceViolation(
                    "cross_subplan_citation_id",
                    "A citation ID is shared by different SubPlans.",
                    subplan_id=subplan_id,
                    details={"citation_id": citation_id, "prior_subplan_id": prior_owner},
                ))
            global_citation_ids[citation_id] = subplan_id

            if citation["subplan_id"] != subplan_id:
                violations.append(ProvenanceViolation(
                    "citation_subplan_mismatch",
                    "A citation references a different SubPlan.",
                    subplan_id=subplan_id,
                    details={"citation_id": citation_id, "actual": citation["subplan_id"]},
                ))
            if expected_entity is not None and citation["entity_key"] != expected_entity:
                violations.append(ProvenanceViolation(
                    "citation_entity_mismatch",
                    "A citation crosses the SubPlan company scope.",
                    subplan_id=subplan_id,
                    details={
                        "citation_id": citation_id,
                        "expected_entity_key": expected_entity,
                        "actual_entity_key": citation["entity_key"],
                    },
                ))
            citation_document = citation["document_id"]
            if (
                citation_document is not None
                and allowed_documents
                and citation_document not in allowed_documents
            ):
                violations.append(ProvenanceViolation(
                    "citation_document_mismatch",
                    "A citation crosses the SubPlan document scope.",
                    subplan_id=subplan_id,
                    details={
                        "citation_id": citation_id,
                        "allowed_document_ids": sorted(allowed_documents),
                        "actual_document_id": citation_document,
                    },
                ))

        citation_ids = set(local_citations)
        for citation in result["citations"]:
            missing_sources = sorted(set(citation["source_citation_ids"]) - citation_ids)
            if missing_sources:
                violations.append(ProvenanceViolation(
                    "unresolved_source_citation",
                    "A derivation citation references a citation outside its SubPlan.",
                    subplan_id=subplan_id,
                    details={
                        "citation_id": citation["citation_id"],
                        "missing_source_citation_ids": missing_sources,
                    },
                ))

        for claim_ordinal, claim in enumerate(result["claims"]):
            if not isinstance(claim, Mapping):
                violations.append(ProvenanceViolation(
                    "malformed_claim",
                    "A claim is not an object.",
                    subplan_id=subplan_id,
                    details={"claim_ordinal": claim_ordinal},
                ))
                continue
            claim_citations = claim.get("citation_ids")
            if not isinstance(claim_citations, list) or not claim_citations:
                violations.append(ProvenanceViolation(
                    "claim_without_citation",
                    "A claim lacks local citation IDs.",
                    subplan_id=subplan_id,
                    details={"claim_ordinal": claim_ordinal},
                ))
                continue
            missing = sorted(
                citation_id if isinstance(citation_id, str) else f"<{type(citation_id).__name__}>"
                for citation_id in claim_citations
                if not isinstance(citation_id, str) or citation_id not in citation_ids
            )
            if missing:
                violations.append(ProvenanceViolation(
                    "claim_citation_mismatch",
                    "A claim references a citation outside its SubPlan.",
                    subplan_id=subplan_id,
                    details={"claim_ordinal": claim_ordinal, "citation_ids": missing},
                ))
            if expected_entity is not None and claim.get("entity_key") != expected_entity:
                violations.append(ProvenanceViolation(
                    "claim_entity_mismatch",
                    "A claim crosses the SubPlan company scope.",
                    subplan_id=subplan_id,
                    details={
                        "claim_ordinal": claim_ordinal,
                        "expected_entity_key": expected_entity,
                        "actual_entity_key": (
                            claim.get("entity_key")
                            if isinstance(claim.get("entity_key"), (str, type(None)))
                            else f"<{type(claim.get('entity_key')).__name__}>"
                        ),
                    },
                ))
            if subplan["backend"] == "evidence":
                entity = next(
                    (
                        row for row in plan["scope_plan"]["entity_resolutions"]
                        if row["entity_key"] == expected_entity
                    ),
                    None,
                )
                company = claim.get("company")
                allowed_company_labels: set[str] = {expected_entity} if expected_entity else set()
                if entity is not None:
                    allowed_company_labels.add(entity["mention"])
                    if entity["identity"] is not None:
                        allowed_company_labels.update({
                            entity["identity"]["stock_code"],
                            entity["identity"]["stock_name"],
                            entity["identity"]["company_full"],
                        })
                if not isinstance(company, str) or company not in allowed_company_labels:
                    violations.append(ProvenanceViolation(
                        "claim_company_mismatch",
                        "An evidence claim's company label is outside its SubPlan scope.",
                        subplan_id=subplan_id,
                        details={
                            "claim_ordinal": claim_ordinal,
                            "allowed_company_labels": sorted(allowed_company_labels),
                            "actual_company": (
                                company if isinstance(company, (str, type(None)))
                                else f"<{type(company).__name__}>"
                            ),
                        },
                    ))
            claim_document = claim.get("document_id")
            if allowed_documents and (
                not isinstance(claim_document, str) or claim_document not in allowed_documents
            ):
                violations.append(ProvenanceViolation(
                    "claim_document_mismatch",
                    "A claim crosses the SubPlan document scope.",
                    subplan_id=subplan_id,
                    details={
                        "claim_ordinal": claim_ordinal,
                        "allowed_document_ids": sorted(allowed_documents),
                        "actual_document_id": (
                            claim_document if isinstance(claim_document, (str, type(None)))
                            else f"<{type(claim_document).__name__}>"
                        ),
                    },
                ))
            if subplan["backend"] == "evidence":
                self._validate_evidence_claim_text(
                    plan, subplan, claim, claim_ordinal, violations
                )

    @staticmethod
    def _validate_evidence_claim_text(
        plan: Mapping[str, Any],
        subplan: Mapping[str, Any],
        claim: Mapping[str, Any],
        claim_ordinal: int,
        violations: list[ProvenanceViolation],
    ) -> None:
        text = claim.get("text", claim.get("claim_text"))
        if not isinstance(text, str) or not text.strip():
            violations.append(ProvenanceViolation(
                "empty_evidence_claim",
                "An evidence claim has no deterministic text.",
                subplan_id=subplan["subplan_id"],
                details={"claim_ordinal": claim_ordinal},
            ))
            return
        if not _CROSS_ENTITY_INFERENCE_RE.search(text):
            return

        current = subplan["entity_key"]
        for entity in plan["scope_plan"]["entity_resolutions"]:
            if entity["entity_key"] == current:
                continue
            aliases = [entity["mention"]]
            if entity["identity"] is not None:
                aliases.extend([
                    entity["identity"]["stock_name"],
                    entity["identity"]["company_full"],
                    entity["identity"]["stock_code"],
                ])
            matched = sorted({alias for alias in aliases if alias and alias in text})
            if matched:
                violations.append(ProvenanceViolation(
                    "cross_entity_evidence_inference",
                    "An evidence claim performs forbidden cross-company inference.",
                    subplan_id=subplan["subplan_id"],
                    details={"claim_ordinal": claim_ordinal, "matched_other_entity_aliases": matched},
                ))


class Composer:
    """Compose a contract-valid ``QAAnswer`` from a frozen static plan."""

    composer_version = COMPOSER_VERSION

    def __init__(self, *, provenance_validator: GlobalAnswerProvenanceValidator | None = None) -> None:
        self.provenance_validator = provenance_validator or GlobalAnswerProvenanceValidator()

    def compose(
        self,
        request_id: str,
        composition_plan: Mapping[str, Any],
        subplan_results: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        if not isinstance(request_id, str) or not request_id.strip():
            raise ValueError("request_id must be a non-empty string")
        plan = validate_composition_plan(deepcopy(dict(composition_plan)))
        supplied = [deepcopy(dict(row)) if isinstance(row, Mapping) else row for row in subplan_results]
        violations = self.provenance_validator.validate(plan, supplied)
        if violations:
            return self._provenance_blocked_answer(request_id, plan, violations)

        results = self._ordered_results(plan, supplied)
        usable = [row for row in results if is_subplan_result_usable(row)]
        usable_ids = {row["subplan_id"] for row in usable}
        quorum = self._evaluate_quorum(plan, results, usable_ids)
        required_ids = plan["composition_policy"]["required_subplan_ids"]
        result_by_id = {row["subplan_id"]: row for row in results}
        required_all_ok = all(result_by_id[subplan_id]["status"] == "ok" for subplan_id in required_ids)

        if usable:
            status = "ok" if required_all_ok and quorum["met"] else "partial"
        else:
            status = dominant_failed_status([result_by_id[row]["status"] for row in required_ids])

        errors, warnings = self._aggregate_findings(results)
        if usable and not quorum["met"]:
            errors.append({
                "failure_code": "COMPOSITION_QUORUM_NOT_MET",
                "message": "Structured comparison quorum was not met; no comparison conclusion was produced.",
                "details": {
                    "required_count": quorum["required_count"],
                    "groups": quorum["groups"],
                },
            })
        if quorum["scope_reduced"]:
            warnings.append({
                "warning_code": "COMPARISON_SCOPE_REDUCED",
                "message": "The structured comparison includes only successful entities or periods.",
                "details": {"groups": quorum["groups"]},
            })

        sections = self._build_sections(plan, results, usable_ids)
        comparison = self._build_comparison(plan, results, usable_ids, quorum)
        answer_result = None
        if usable:
            answer_result = {
                "composition_policy_id": plan["composition_policy"]["policy_id"],
                "section_axis": plan["composition_policy"]["section_axis"],
                "sections": sections,
                "comparison": comparison,
                "unavailable_subplans": [
                    {
                        "subplan_id": row["subplan_id"],
                        "status": row["status"],
                        "failure_code": row["failure_code"],
                    }
                    for row in results if row["subplan_id"] not in usable_ids
                ],
            }

        citations = [
            deepcopy(citation)
            for row in results if row["subplan_id"] in usable_ids
            for citation in row["citations"]
        ]
        trace = {
            "composer_version": self.composer_version,
            "plan_fingerprint": plan["plan_fingerprint"],
            "status": status,
            "usable_subplan_ids": [row["subplan_id"] for row in results if row["subplan_id"] in usable_ids],
            "failed_required_subplan_ids": [
                row for row in required_ids if result_by_id[row]["status"] != "ok"
            ],
            "quorum": quorum,
            "provenance_valid": True,
        }
        answer = {
            "schema_version": SCHEMA_ANSWER,
            "request_id": request_id,
            "status": status,
            "composition_pattern_id": plan["pattern_id"],
            "document_scope": deepcopy(plan["scope_plan"]),
            "subplans": results,
            "answer_text": self._render_answer_text(plan, results, usable_ids, sections, comparison),
            "result": answer_result,
            "citations": citations,
            "trace": trace,
            "errors": errors,
            "warnings": warnings,
            "missing_fact_requests": [
                deepcopy(request)
                for row in results
                for request in row["missing_fact_requests"]
            ],
        }
        validate_qa_answer(answer)
        canonical_json_bytes(answer)
        return answer

    def _ordered_results(
        self,
        plan: Mapping[str, Any],
        supplied: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        by_id = {row["subplan_id"]: deepcopy(dict(row)) for row in supplied}
        ordered: list[dict[str, Any]] = []
        for subplan in plan["subplans"]:
            if subplan["planning_state"] == "blocked":
                ordered.append(self._blocked_plan_result(subplan))
            elif subplan["subplan_id"] in by_id:
                ordered.append(by_id[subplan["subplan_id"]])
            else:
                ordered.append(self._missing_execution_result(subplan))
        for result in ordered:
            validate_subplan_result(result)
        return ordered

    @staticmethod
    def _blocked_plan_result(subplan: Mapping[str, Any]) -> dict[str, Any]:
        failure = deepcopy(subplan["planning_failure"])
        code = failure["failure_code"]
        result = {
            "schema_version": SCHEMA_SUBPLAN_RESULT,
            "subplan_id": subplan["subplan_id"],
            "backend": subplan["backend"],
            "operation": subplan["operation"],
            "planning_state": "blocked",
            "execution_state": "not_executed",
            "status": status_for_blocked_plan(code),
            "result": None,
            "claims": [],
            "citations": [],
            "failure_code": code,
            "errors": [failure],
            "warnings": [],
            "missing_fact_requests": [],
            "trace": {"source": "composition_plan", "planning_failure": failure},
        }
        validate_subplan_result(result)
        return result

    @staticmethod
    def _missing_execution_result(subplan: Mapping[str, Any]) -> dict[str, Any]:
        error = {
            "failure_code": "INTERNAL_ERROR",
            "message": "No backend result was supplied for a ready required SubPlan.",
            "details": {"subplan_id": subplan["subplan_id"]},
        }
        result = {
            "schema_version": SCHEMA_SUBPLAN_RESULT,
            "subplan_id": subplan["subplan_id"],
            "backend": subplan["backend"],
            "operation": subplan["operation"],
            "planning_state": "ready",
            "execution_state": "not_executed",
            "status": "error",
            "result": None,
            "claims": [],
            "citations": [],
            "failure_code": "INTERNAL_ERROR",
            "errors": [error],
            "warnings": [],
            "missing_fact_requests": [],
            "trace": {"source": "composer", "backend_result_missing": True},
        }
        validate_subplan_result(result)
        return result

    @staticmethod
    def _evaluate_quorum(
        plan: Mapping[str, Any],
        results: Sequence[Mapping[str, Any]],
        usable_ids: set[str],
    ) -> dict[str, Any]:
        selector = plan["composition_policy"]["quorum_selector"]
        if selector is None:
            return {
                "required": False,
                "met": True,
                "required_count": None,
                "groups": [],
                "scope_reduced": False,
            }
        subplan_by_id = {row["subplan_id"]: row for row in plan["subplans"]}
        result_by_id = {row["subplan_id"]: row for row in results}
        groups: OrderedDict[str, list[Mapping[str, Any]]] = OrderedDict()
        group_by = selector["group_by"]
        for subplan in plan["subplans"]:
            if (
                subplan["backend"] not in selector["eligible_backends"]
                or subplan["operation"] not in selector["eligible_operations"]
            ):
                continue
            group_key = "__global__" if group_by == "global" else subplan["concern_key"]
            groups.setdefault(group_key, []).append(subplan)

        required_count = selector["count"]
        group_rows: list[dict[str, Any]] = []
        met = bool(groups)
        scope_reduced = False
        for concern_key, members in groups.items():
            usable_members = [row for row in members if row["subplan_id"] in usable_ids]
            compatible_members = Composer._compatible_members(
                plan, usable_members, result_by_id
            )
            distinct_key = selector["distinct_key"]
            distinct_values = {
                row[distinct_key] for row in compatible_members if row[distinct_key] is not None
            }
            group_met = len(distinct_values) >= required_count
            met = met and group_met
            if group_met and len(usable_members) < len(members):
                scope_reduced = True
            group_rows.append({
                "concern_key": concern_key,
                "planned_subplan_ids": [row["subplan_id"] for row in members],
                "usable_subplan_ids": [row["subplan_id"] for row in compatible_members],
                "unavailable_subplan_ids": [
                    row["subplan_id"] for row in members
                    if row["subplan_id"] not in usable_ids
                ],
                "incompatible_subplan_ids": [
                    row["subplan_id"] for row in usable_members
                    if row["subplan_id"] not in {item["subplan_id"] for item in compatible_members}
                ],
                "usable_distinct_count": len(distinct_values),
                "required_count": required_count,
                "met": group_met,
            })
        return {
            "required": True,
            "met": met,
            "required_count": required_count,
            "groups": group_rows,
            "scope_reduced": scope_reduced,
        }

    @staticmethod
    def _compatible_members(
        plan: Mapping[str, Any],
        members: Sequence[Mapping[str, Any]],
        result_by_id: Mapping[str, Mapping[str, Any]],
    ) -> list[Mapping[str, Any]]:
        compatibility = plan["composition_policy"]["comparison_measure_compatibility"]
        if compatibility == "not_applicable":
            return list(members)
        buckets: OrderedDict[tuple[str, str | None], list[Mapping[str, Any]]] = OrderedDict()
        for subplan in members:
            result = result_by_id[subplan["subplan_id"]]["result"]
            unit = result.get("normalized_unit") if isinstance(result, Mapping) else None
            measure = "*" if compatibility == "same_unit" else subplan["concern_key"]
            buckets.setdefault((measure, unit), []).append(subplan)
        if not buckets:
            return []
        # All members of one concern should have the same measure/unit.  When a
        # backend violates that expectation, select the largest stable bucket;
        # excluded results remain visible but cannot satisfy comparison quorum.
        return max(buckets.values(), key=lambda rows: len(rows))

    @staticmethod
    def _aggregate_findings(
        results: Sequence[Mapping[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        errors: list[dict[str, Any]] = []
        warnings: list[dict[str, Any]] = []
        for result in results:
            for row in result["errors"]:
                item = deepcopy(dict(row)) if isinstance(row, Mapping) else {"message": str(row)}
                item.setdefault("subplan_id", result["subplan_id"])
                errors.append(item)
            for row in result["warnings"]:
                item = deepcopy(dict(row)) if isinstance(row, Mapping) else {"message": str(row)}
                item.setdefault("subplan_id", result["subplan_id"])
                warnings.append(item)
        return errors, warnings

    @staticmethod
    def _build_sections(
        plan: Mapping[str, Any],
        results: Sequence[Mapping[str, Any]],
        usable_ids: set[str],
    ) -> list[dict[str, Any]]:
        axis = plan["composition_policy"]["section_axis"]
        subplan_by_id = {row["subplan_id"]: row for row in plan["subplans"]}
        sections: OrderedDict[str, dict[str, Any]] = OrderedDict()
        for result in results:
            if result["subplan_id"] not in usable_ids:
                continue
            subplan = subplan_by_id[result["subplan_id"]]
            if axis == "entity":
                key = subplan["entity_key"] or "unscoped"
            elif axis == "period":
                key = subplan["period_key"] or "unscoped"
            elif axis == "concern":
                key = subplan["concern_key"]
            else:
                key = "answer"
            section = sections.setdefault(key, {
                "section_key": key,
                "entity_key": subplan["entity_key"] if axis == "entity" else None,
                "period_key": subplan["period_key"] if axis == "period" else None,
                "items": [],
            })
            section["items"].append({
                "subplan_id": result["subplan_id"],
                "backend": result["backend"],
                "operation": result["operation"],
                "concern_key": subplan["concern_key"],
                "status": result["status"],
                "result": deepcopy(result["result"]),
                "claims": deepcopy(result["claims"]),
            })
        return list(sections.values())

    @staticmethod
    def _build_comparison(
        plan: Mapping[str, Any],
        results: Sequence[Mapping[str, Any]],
        usable_ids: set[str],
        quorum: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        if not quorum["required"] or not quorum["met"]:
            return None
        subplan_by_id = {row["subplan_id"]: row for row in plan["subplans"]}
        result_by_id = {row["subplan_id"]: row for row in results}
        groups: list[dict[str, Any]] = []
        for group in quorum["groups"]:
            rows: list[dict[str, Any]] = []
            for subplan_id in group["usable_subplan_ids"]:
                subplan = subplan_by_id[subplan_id]
                result = result_by_id[subplan_id]
                # Quorum excludes evidence by contract; this assertion makes
                # accidental future registry widening fail closed in tests.
                if result["backend"] == "evidence" or subplan_id not in usable_ids:
                    continue
                rows.append({
                    "subplan_id": subplan_id,
                    "entity_key": subplan["entity_key"],
                    "period_key": subplan["period_key"],
                    "concern_key": subplan["concern_key"],
                    "value": Composer._display_value(result["result"]),
                    "normalized_unit": (
                        result["result"].get("normalized_unit")
                        if isinstance(result["result"], Mapping) else None
                    ),
                })
            groups.append({
                "concern_key": group["concern_key"],
                "rows": rows,
                "missing_subplan_ids": group["unavailable_subplan_ids"],
            })
        return {
            "basis": "structured_results_only",
            "groups": groups,
            "cross_entity_causal_inference": False,
        }

    @staticmethod
    def _display_value(result: Any) -> str | None:
        if not isinstance(result, Mapping):
            return None
        value = result.get("value")
        if value is not None:
            if result.get("normalized_unit") == "ratio":
                try:
                    percent = (Decimal(str(value)) * Decimal("100")).quantize(
                        Decimal("0.01"), rounding=ROUND_HALF_UP
                    )
                except (InvalidOperation, ValueError):
                    return str(value)
                return format(percent, ".2f")
            return str(value)
        rows = result.get("rows")
        if isinstance(rows, list) and rows:
            rendered: list[str] = []
            for row in rows:
                if not isinstance(row, Mapping):
                    continue
                value_text = str(row.get("value", ""))
                unit = Composer._display_unit(row.get("normalized_unit"))
                if row.get("operation") == "rank":
                    company = row.get("company") or row.get("stock_code") or row.get("entity_key")
                    rank = row.get("rank")
                    prefix = f"第{rank}名 " if rank is not None else ""
                    rendered.append(f"{prefix}{company}={value_text}{unit}")
                else:
                    operator = row.get("aggregate_operator") or "aggregate"
                    rendered.append(f"{operator}={value_text}{unit}")
            return "；".join(rendered) if rendered else None
        return None

    @staticmethod
    def _display_unit(unit: Any) -> str:
        if unit in (None, "", "text"):
            return ""
        if unit == "ratio":
            return "%"
        return str(unit)

    @staticmethod
    def _concern_label(subplan: Mapping[str, Any]) -> str:
        payload = subplan.get("payload")
        if isinstance(payload, Mapping):
            metadata = payload.get("metadata_field")
            if isinstance(metadata, str) and metadata:
                return _METADATA_LABELS.get(metadata, metadata)
            metric = payload.get("canonical_metric")
            if isinstance(metric, str) and metric:
                return metric
            formula = payload.get("formula_id")
            if isinstance(formula, str) and formula:
                return _FORMULA_LABELS.get(formula, formula)
        return str(subplan["concern_key"])

    def _render_answer_text(
        self,
        plan: Mapping[str, Any],
        results: Sequence[Mapping[str, Any]],
        usable_ids: set[str],
        sections: Sequence[Mapping[str, Any]],
        comparison: Mapping[str, Any] | None,
    ) -> str:
        if not usable_ids:
            return ""
        entity_names = {
            row["entity_key"]: (
                row["identity"]["stock_name"] if row["identity"] is not None else row["mention"]
            )
            for row in plan["scope_plan"]["entity_resolutions"]
        }
        subplan_by_id = {row["subplan_id"]: row for row in plan["subplans"]}
        result_by_id = {row["subplan_id"]: row for row in results}
        lines: list[str] = []
        show_headers = len(sections) > 1 or plan["composition_policy"]["section_axis"] != "none"
        for section in sections:
            if show_headers:
                if section["entity_key"] is not None:
                    label = entity_names.get(section["entity_key"], section["entity_key"])
                elif section["period_key"] is not None:
                    label = f"{section['period_key']}年"
                elif section["items"]:
                    label = self._concern_label(subplan_by_id[section["items"][0]["subplan_id"]])
                else:
                    label = section["section_key"]
                lines.append(f"【{label}】")
            for item in section["items"]:
                if item["backend"] == "evidence":
                    for claim in item["claims"]:
                        text = claim.get("text", claim.get("claim_text"))
                        if isinstance(text, str) and text.strip():
                            lines.append(text.strip())
                else:
                    value = self._display_value(item["result"])
                    if value is not None:
                        unit = item["result"].get("normalized_unit") if isinstance(item["result"], Mapping) else None
                        suffix = self._display_unit(unit)
                        label = self._concern_label(subplan_by_id[item["subplan_id"]])
                        rendered = f"{label}：{value}{suffix}"
                        backend_result = result_by_id[item["subplan_id"]]
                        if _contains_supplemental_marker(backend_result.get("citations", [])):
                            rendered += "（补充来源：TabGR）"
                        lines.append(rendered)

        for subplan in plan["subplans"]:
            result = result_by_id[subplan["subplan_id"]]
            if subplan["subplan_id"] not in usable_ids:
                label = entity_names.get(subplan["entity_key"], subplan["entity_key"] or subplan["period_key"] or "")
                prefix = f"{label} " if label else ""
                code = result["failure_code"] or result["status"]
                lines.append(f"{prefix}{self._concern_label(subplan)}：未完成（{code}）")

        if comparison is not None:
            for group in comparison["groups"]:
                rendered: list[str] = []
                for row in group["rows"]:
                    if group["concern_key"] == "__global__":
                        label = self._concern_label(subplan_by_id[row["subplan_id"]])
                    else:
                        label = (
                            entity_names.get(row["entity_key"], row["entity_key"])
                            if row["entity_key"] is not None else f"{row['period_key']}年"
                        )
                    unit = row["normalized_unit"]
                    suffix = self._display_unit(unit)
                    rendered.append(f"{label}={row['value']}{suffix}")
                if rendered:
                    group_label = "跨指标" if group["concern_key"] == "__global__" else group["concern_key"]
                    lines.append(f"结构化比较（{group_label}）：" + "；".join(rendered))
        return "\n".join(lines)

    def _provenance_blocked_answer(
        self,
        request_id: str,
        plan: Mapping[str, Any],
        violations: Sequence[ProvenanceViolation],
    ) -> dict[str, Any]:
        error = {
            "failure_code": "PROVENANCE_VALIDATION_FAILED",
            "message": "Global answer provenance validation failed; all outputs were suppressed.",
            "details": {"violations": [row.as_dict() for row in violations]},
        }
        sanitized: list[dict[str, Any]] = []
        for subplan in plan["subplans"]:
            row = {
                "schema_version": SCHEMA_SUBPLAN_RESULT,
                "subplan_id": subplan["subplan_id"],
                "backend": subplan["backend"],
                "operation": subplan["operation"],
                "planning_state": subplan["planning_state"],
                "execution_state": "not_executed" if subplan["planning_state"] == "blocked" else "executed",
                "status": "blocked",
                "result": None,
                "claims": [],
                "citations": [],
                "failure_code": "PROVENANCE_VALIDATION_FAILED",
                "errors": [deepcopy(error)],
                "warnings": [],
                "missing_fact_requests": [],
                "trace": {"source": "composer", "outputs_suppressed": True},
            }
            validate_subplan_result(row)
            sanitized.append(row)
        answer = {
            "schema_version": SCHEMA_ANSWER,
            "request_id": request_id,
            "status": "blocked",
            "composition_pattern_id": plan["pattern_id"],
            "document_scope": deepcopy(plan["scope_plan"]),
            "subplans": sanitized,
            "answer_text": "",
            "result": None,
            "citations": [],
            "trace": {
                "composer_version": self.composer_version,
                "plan_fingerprint": plan["plan_fingerprint"],
                "status": "blocked",
                "provenance_valid": False,
                "violation_fingerprint": semantic_sha256([row.as_dict() for row in violations]),
            },
            "errors": [error],
            "warnings": [],
            "missing_fact_requests": [],
        }
        validate_qa_answer(answer)
        canonical_json_bytes(answer)
        return answer


__all__ = [
    "COMPOSER_VERSION",
    "Composer",
    "GlobalAnswerProvenanceValidator",
    "ProvenanceViolation",
]
