"""Stable Phase 8 failure codes and status precedence."""

from __future__ import annotations

FAILURE_CODES = frozenset({
    "INVALID_REQUEST",
    "COMPOSITION_UNSUPPORTED",
    "COMPOSITION_LIMIT_EXCEEDED",
    "COMPOSITION_QUORUM_NOT_MET",
    "ROUTE_AMBIGUOUS",
    "RESOLVER_MISSING",
    "RESOLVER_AMBIGUOUS",
    "RESOLVER_CONFLICT",
    "METRIC_UNRECOGNIZED",
    "METRIC_AMBIGUOUS",
    "METRIC_YEAR_MISSING",
    "UNIT_AMBIGUOUS",
    "SELECTED_FACT_MISSING",
    "FACT_UNRESOLVED_CONFLICT",
    "FORMULA_OPERAND_MISSING",
    "FORMULA_ZERO_DENOMINATOR",
    "FORMULA_UNIT_MISMATCH",
    "SQL_UNSUPPORTED_QUERY",
    "SQL_SAFETY_REJECTED",
    "EVIDENCE_UNAVAILABLE",
    "UNSUPPORTED_GENERAL_KNOWLEDGE",
    "GENERATOR_UNAVAILABLE",
    "PROVENANCE_VALIDATION_FAILED",
    "FALLBACK_REQUIRED_PHASE9_DISABLED",
    "INTERNAL_ERROR",
})

ANSWER_STATUSES = frozenset({
    "ok",
    "partial",
    "needs_clarification",
    "not_found",
    "unsupported",
    "fallback_required",
    "blocked",
    "error",
})

SUBPLAN_STATUSES = ANSWER_STATUSES

WARNING_CODES = frozenset({
    "CORPUS_COVERAGE_INCOMPLETE",
    "COMPARISON_SCOPE_REDUCED",
    "EVIDENCE_ONLY",
})

ALL_FAILED_STATUS_PRECEDENCE = (
    "error",
    "blocked",
    "needs_clarification",
    "fallback_required",
    "unsupported",
    "not_found",
)

PRECOMPOSITION_STATUS_BY_CODE = {
    "INVALID_REQUEST": "error",
    "ROUTE_AMBIGUOUS": "needs_clarification",
    "COMPOSITION_UNSUPPORTED": "unsupported",
    "COMPOSITION_LIMIT_EXCEEDED": "unsupported",
    "UNSUPPORTED_GENERAL_KNOWLEDGE": "unsupported",
}

BLOCKED_PLAN_STATUS_BY_CODE = {
    "RESOLVER_MISSING": "not_found",
    "RESOLVER_AMBIGUOUS": "needs_clarification",
    "RESOLVER_CONFLICT": "needs_clarification",
    "METRIC_UNRECOGNIZED": "unsupported",
    "METRIC_AMBIGUOUS": "needs_clarification",
    "METRIC_YEAR_MISSING": "needs_clarification",
    "COMPOSITION_UNSUPPORTED": "unsupported",
    "COMPOSITION_LIMIT_EXCEEDED": "unsupported",
    "SQL_SAFETY_REJECTED": "blocked",
    "PROVENANCE_VALIDATION_FAILED": "blocked",
}


def status_for_blocked_plan(failure_code: str) -> str:
    """Map a planning failure to its non-executed subplan status."""

    return BLOCKED_PLAN_STATUS_BY_CODE.get(failure_code, "blocked")


def dominant_failed_status(statuses: list[str]) -> str:
    """Return the frozen top-level status when no result is usable."""

    for candidate in ALL_FAILED_STATUS_PRECEDENCE:
        if candidate in statuses:
            return candidate
    return "error"
