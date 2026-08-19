"""Boundary validators for the frozen Phase 8 JSON contracts.

The core intentionally uses plain dictionaries and standard-library validators.
This keeps the on-disk JSON schemas and the runtime representation identical and
avoids a second, framework-specific source of truth.
"""

from __future__ import annotations

import hashlib
import json
import re
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable, Mapping

from .errors import ANSWER_STATUSES, FAILURE_CODES, SUBPLAN_STATUSES

SCHEMA_REQUEST = "finglmqa.phase8.qa_request.v1"
SCHEMA_ANALYSIS = "finglmqa.phase8.question_analysis.v1"
SCHEMA_SCOPE = "finglmqa.phase8.scope_plan.v1"
SCHEMA_COMPOSITION = "finglmqa.phase8.composition_plan.v1"
SCHEMA_SUBPLAN_RESULT = "finglmqa.phase8.subplan_result.v1"
SCHEMA_ANSWER = "finglmqa.phase8.qa_answer.v1"
SCHEMA_TRACE = "finglmqa.phase8.qa_trace.v1"
SCHEMA_TELEMETRY = "finglmqa.phase8.qa_telemetry.v1"
# Gate 1 contract revision 2 adds canonical metric/year to SQL-derived
# authorizations so the evidence executor can prove the same metric scope.
SCHEMA_NUMERIC_AUTHORIZATION = "finglmqa.phase8.numeric_authorization.v2"
SCHEMA_NUMERIC_AUTHORIZATION_SET = "finglmqa.phase8.numeric_authorization_set.v1"
SCHEMA_MISSING_FACT_REQUEST = "finglmqa.phase8.missing_fact_request.v1"

LIMITS = {
    "max_companies": 5,
    "max_years": 5,
    "max_subplans": 12,
    "max_evidence_top_k": 5,
    "max_evidence_chunks": 25,
}

BACKENDS = frozenset({"fact", "sql", "formula", "evidence"})
INTENTS = frozenset({"lookup", "compare", "rank", "aggregate", "calculate", "narrative"})
NARRATIVE_MODES = frozenset({"explain", "summarize"})
EVIDENCE_KINDS = frozenset({"structured_fact", "table", "narrative"})
YEAR_ROLES = frozenset({"report_year", "metric_year", "formula_operand", "output_period", "corpus_year", "unresolved"})
CONCERN_KINDS = frozenset({"metric", "formula", "metadata", "narrative"})
SCOPE_KINDS = frozenset({
    "single_document",
    "company_documents",
    "multi_company_documents",
    "explicit_document_set",
    "corpus",
})
PLANNING_STATES = frozenset({"ready", "blocked"})
EXECUTION_STATES = frozenset({"executed", "not_executed"})
ENTITY_RESOLUTION_STATUSES = frozenset({"unique", "missing", "ambiguous", "conflict"})

PATTERN_IDS = (
    "single_node",
    "parallel_concerns",
    "entity_list",
    "entity_compare",
    "period_list",
    "period_compare",
    "single_document_bundle",
    "entity_section_bundle",
    "period_section_bundle",
)

_SUBPLAN_ID_RE = re.compile(r"^sp_[0-9a-f]{16}$")
_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]+$")


class ContractValidationError(ValueError):
    """Raised when a public Phase 8 boundary payload is invalid."""


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize deterministic semantic JSON as UTF-8 with one final newline."""

    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def semantic_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def validate_pattern_registry(payload: Any) -> dict[str, Any]:
    obj = _object(payload, "CompositionPatternRegistry")
    fields = {"schema_version", "registry_version", "limits", "match_precedence", "patterns"}
    _required(obj, fields, "CompositionPatternRegistry")
    _no_unknown(obj, fields, "CompositionPatternRegistry")
    if obj["schema_version"] != "finglmqa.phase8.composition_pattern_registry.v1":
        raise ContractValidationError("CompositionPatternRegistry.schema_version is unsupported")
    _string(obj["registry_version"], "CompositionPatternRegistry.registry_version")
    if _object(obj["limits"], "CompositionPatternRegistry.limits") != LIMITS:
        raise ContractValidationError("CompositionPatternRegistry.limits must match frozen Phase 8 limits")
    expected_precedence = [
        "boundary_rejection", "narrative_topology", "entity_topology", "period_topology",
        "parallel_concerns", "single_node",
    ]
    if obj["match_precedence"] != expected_precedence:
        raise ContractValidationError("CompositionPatternRegistry.match_precedence is not frozen order")
    patterns = _array(obj["patterns"], "CompositionPatternRegistry.patterns")
    if [pattern.get("pattern_id") for pattern in patterns] != list(PATTERN_IDS):
        raise ContractValidationError("CompositionPatternRegistry must contain exactly the nine frozen patterns")
    priorities: set[int] = set()
    for index, pattern in enumerate(patterns):
        path = f"CompositionPatternRegistry.patterns[{index}]"
        pattern = _object(pattern, path)
        required = {
            "pattern_id", "version", "priority", "allowed_scope_kinds", "allowed_intents", "shape",
            "ordering_rule", "required_selector", "optional_selector", "minimum_usable_results",
            "quorum_selector", "quorum_applies_when_intent", "comparison_measure_compatibility", "allow_cross_document_claims",
            "allow_cross_entity_causal_inference", "composition_policy_id", "dynamic_expansion",
        }
        _required(pattern, required, path)
        _no_unknown(pattern, required, path)
        _enum(pattern["pattern_id"], PATTERN_IDS, f"{path}.pattern_id")
        _string(pattern["version"], f"{path}.version")
        priority = _integer(pattern["priority"], f"{path}.priority", minimum=0)
        if priority in priorities:
            raise ContractValidationError("CompositionPatternRegistry priorities must be unique")
        priorities.add(priority)
        scopes = _unique_strings(pattern["allowed_scope_kinds"], f"{path}.allowed_scope_kinds")
        intents = _unique_strings(pattern["allowed_intents"], f"{path}.allowed_intents")
        if not set(scopes).issubset(SCOPE_KINDS) or not set(intents).issubset(INTENTS):
            raise ContractValidationError(f"{path} contains an invalid scope or intent")
        for field in ("shape", "ordering_rule", "required_selector", "optional_selector", "composition_policy_id"):
            _string(pattern[field], f"{path}.{field}")
        if pattern["required_selector"] != "all_explicit_subplans" or pattern["optional_selector"] != "none":
            raise ContractValidationError(f"{path} violates Phase 8 required/optional policy")
        if _integer(pattern["minimum_usable_results"], f"{path}.minimum_usable_results", minimum=1) != 1:
            raise ContractValidationError(f"{path}.minimum_usable_results must be 1")
        if pattern["quorum_selector"] is not None:
            quorum = _object(pattern["quorum_selector"], f"{path}.quorum_selector")
            _required(quorum, ("group_by", "distinct_key", "eligible_backends", "eligible_operations", "count"), f"{path}.quorum_selector")
            if pattern["quorum_applies_when_intent"] != "compare":
                raise ContractValidationError(f"{path}.quorum_applies_when_intent must be compare")
            group_by = _enum(quorum["group_by"], {"concern", "global"}, f"{path}.quorum_selector.group_by")
            distinct_key = _enum(
                quorum["distinct_key"], {"entity_key", "period_key", "concern_key"},
                f"{path}.quorum_selector.distinct_key",
            )
            if group_by == "global" and distinct_key != "concern_key":
                raise ContractValidationError(f"{path} global quorum must count distinct concerns")
        elif pattern["quorum_applies_when_intent"] is not None:
            raise ContractValidationError(f"{path}.quorum_applies_when_intent must be null without quorum")
        _enum(
            pattern["comparison_measure_compatibility"],
            {"not_applicable", "same_unit", "same_measure_and_unit"},
            f"{path}.comparison_measure_compatibility",
        )
        for field in ("allow_cross_document_claims", "allow_cross_entity_causal_inference", "dynamic_expansion"):
            if _boolean(pattern[field], f"{path}.{field}"):
                raise ContractValidationError(f"{path}.{field} must be false")
    return obj


def make_subplan_id(
    registry_semantic_sha256: str,
    pattern_id: str,
    pattern_version: str,
    ordinal: int,
    entity_key: str | None,
    period_key: str | None,
    concern_key: str,
) -> str:
    canonical_tuple = [
        registry_semantic_sha256, pattern_id, pattern_version, ordinal, entity_key, period_key, concern_key,
    ]
    return "sp_" + hashlib.sha256(canonical_json_bytes(canonical_tuple)).hexdigest()[:16]


def make_requirement_id(payload: Mapping[str, Any]) -> str:
    keys = (
        "origin_operation", "formula_id", "operand_role", "subplan_id", "document_id",
        "stock_code", "report_year", "metric_year", "canonical_metric", "normalized_unit",
    )
    return "req_" + hashlib.sha256(
        canonical_json_bytes([payload.get(key) for key in keys])
    ).hexdigest()[:20]


def _object(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractValidationError(f"{path} must be an object")
    return value


def _array(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise ContractValidationError(f"{path} must be an array")
    return value


def _string(value: Any, path: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ContractValidationError(f"{path} must be a non-empty string")
    return value


def _integer(value: Any, path: str, *, minimum: int | None = None, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractValidationError(f"{path} must be an integer")
    if minimum is not None and value < minimum:
        raise ContractValidationError(f"{path} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ContractValidationError(f"{path} must be <= {maximum}")
    return value


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise ContractValidationError(f"{path} must be a boolean")
    return value


def _required(obj: Mapping[str, Any], fields: Iterable[str], path: str) -> None:
    missing = [field for field in fields if field not in obj]
    if missing:
        raise ContractValidationError(f"{path} missing required fields: {missing}")


def _no_unknown(obj: Mapping[str, Any], allowed: Iterable[str], path: str) -> None:
    extras = sorted(set(obj) - set(allowed))
    if extras:
        raise ContractValidationError(f"{path} has unknown fields: {extras}")


def _enum(value: Any, values: Iterable[str], path: str) -> str:
    text = _string(value, path)
    if text not in values:
        raise ContractValidationError(f"{path} must be one of {sorted(values)}")
    return text


def _unique_strings(value: Any, path: str, *, maximum: int | None = None) -> list[str]:
    rows = _array(value, path)
    result = [_string(row, f"{path}[{index}]") for index, row in enumerate(rows)]
    if len(result) != len(set(result)):
        raise ContractValidationError(f"{path} must not contain duplicates")
    if maximum is not None and len(result) > maximum:
        raise ContractValidationError(f"{path} must contain at most {maximum} items")
    return result


def _decimal_string(value: Any, path: str) -> str:
    text = _string(value, path)
    try:
        parsed = Decimal(text)
    except InvalidOperation as exc:
        raise ContractValidationError(f"{path} must be a Decimal string") from exc
    if not parsed.is_finite():
        raise ContractValidationError(f"{path} must be finite")
    return text


def validate_qa_request(payload: Any) -> dict[str, Any]:
    obj = _object(payload, "QARequest")
    allowed = {
        "schema_version", "request_id", "question", "locale", "company", "report_year",
        "metric_years", "canonical_metrics", "normalized_unit", "top_k", "generation_mode",
        "trace_delivery",
    }
    _required(obj, ("schema_version", "request_id", "question", "locale"), "QARequest")
    _no_unknown(obj, allowed, "QARequest")
    if obj["schema_version"] != SCHEMA_REQUEST:
        raise ContractValidationError("QARequest.schema_version is unsupported")
    _string(obj["request_id"], "QARequest.request_id")
    _string(obj["question"], "QARequest.question")
    _string(obj["locale"], "QARequest.locale")
    if "company" in obj:
        _string(obj["company"], "QARequest.company")
    if "report_year" in obj:
        _integer(obj["report_year"], "QARequest.report_year", minimum=1900, maximum=2200)
    if "metric_years" in obj:
        years = obj["metric_years"]
        _array(years, "QARequest.metric_years")
        if len(years) != len(set(years)):
            raise ContractValidationError("QARequest.metric_years has duplicates")
        for index, year in enumerate(years):
            _integer(year, f"QARequest.metric_years[{index}]", minimum=1900, maximum=2200)
    if "canonical_metrics" in obj:
        _unique_strings(obj["canonical_metrics"], "QARequest.canonical_metrics")
    if "normalized_unit" in obj:
        _string(obj["normalized_unit"], "QARequest.normalized_unit")
    if "top_k" in obj:
        _integer(obj["top_k"], "QARequest.top_k", minimum=1)
    if obj.get("generation_mode", "disabled") not in {"disabled", "fake", "external"}:
        raise ContractValidationError("QARequest.generation_mode is invalid")
    if obj.get("trace_delivery", "inline") not in {"inline", "reference"}:
        raise ContractValidationError("QARequest.trace_delivery is invalid")
    return obj


def validate_question_analysis(payload: Any) -> dict[str, Any]:
    obj = _object(payload, "QuestionAnalysis")
    fields = {
        "schema_version", "analysis_version", "request_id", "question_sha256", "normalized_question",
        "company_mentions", "year_mentions", "metric_mentions", "formula_mentions", "concerns",
        "cardinalities", "intents", "narrative_mode", "evidence_kinds", "output_entity_axis",
        "output_period_axis", "dynamic_target_dependency", "unsupported_markers", "ambiguity_findings",
    }
    _required(obj, fields, "QuestionAnalysis")
    _no_unknown(obj, fields, "QuestionAnalysis")
    if obj["schema_version"] != SCHEMA_ANALYSIS:
        raise ContractValidationError("QuestionAnalysis.schema_version is unsupported")
    _string(obj["analysis_version"], "QuestionAnalysis.analysis_version")
    _string(obj["request_id"], "QuestionAnalysis.request_id")
    question_sha = _string(obj["question_sha256"], "QuestionAnalysis.question_sha256")
    if not re.fullmatch(r"[0-9a-f]{64}", question_sha):
        raise ContractValidationError("QuestionAnalysis.question_sha256 is invalid")
    _string(obj["normalized_question"], "QuestionAnalysis.normalized_question")
    companies = _array(obj["company_mentions"], "QuestionAnalysis.company_mentions")
    for index, mention in enumerate(companies):
        mention = _object(mention, f"QuestionAnalysis.company_mentions[{index}]")
        mention_path = f"QuestionAnalysis.company_mentions[{index}]"
        _required(mention, ("raw_text", "span", "mention_ordinal", "hint_source"), mention_path)
        _string(mention["raw_text"], f"{mention_path}.raw_text")
        span = _array(mention["span"], f"{mention_path}.span")
        if len(span) != 2:
            raise ContractValidationError(f"{mention_path}.span must contain start/end")
        start = _integer(span[0], f"{mention_path}.span[0]", minimum=0)
        end = _integer(span[1], f"{mention_path}.span[1]", minimum=0)
        if end <= start:
            raise ContractValidationError(f"{mention_path}.span must be increasing")
        _integer(mention["mention_ordinal"], f"{mention_path}.mention_ordinal", minimum=0)
        _enum(mention["hint_source"], {"question", "request_hint"}, f"{mention_path}.hint_source")
    years = _array(obj["year_mentions"], "QuestionAnalysis.year_mentions")
    expanded_years: set[int] = set()
    for index, signal in enumerate(years):
        signal_path = f"QuestionAnalysis.year_mentions[{index}]"
        signal = _object(signal, signal_path)
        _required(signal, ("raw_text", "span", "mention_ordinal", "years", "role", "is_output_axis"), signal_path)
        _string(signal["raw_text"], f"{signal_path}.raw_text")
        span = _array(signal["span"], f"{signal_path}.span")
        if len(span) != 2:
            raise ContractValidationError(f"{signal_path}.span must contain start/end")
        _integer(signal["mention_ordinal"], f"{signal_path}.mention_ordinal", minimum=0)
        _enum(signal["role"], YEAR_ROLES, f"{signal_path}.role")
        _boolean(signal["is_output_axis"], f"{signal_path}.is_output_axis")
        for year in _array(signal["years"], f"{signal_path}.years"):
            expanded_years.add(_integer(year, f"{signal_path}.years[]", minimum=1900, maximum=2200))
    metrics = _array(obj["metric_mentions"], "QuestionAnalysis.metric_mentions")
    for index, mention in enumerate(metrics):
        path = f"QuestionAnalysis.metric_mentions[{index}]"
        mention = _object(mention, path)
        _required(mention, ("raw_text", "span", "mention_ordinal", "candidates", "status"), path)
        _string(mention["raw_text"], f"{path}.raw_text")
        _array(mention["span"], f"{path}.span")
        _integer(mention["mention_ordinal"], f"{path}.mention_ordinal", minimum=0)
        _unique_strings(mention["candidates"], f"{path}.candidates")
        _enum(mention["status"], {"unique", "missing", "ambiguous"}, f"{path}.status")
    _array(obj["formula_mentions"], "QuestionAnalysis.formula_mentions")
    concerns = _array(obj["concerns"], "QuestionAnalysis.concerns")
    concern_ids: list[str] = []
    for index, concern in enumerate(concerns):
        path = f"QuestionAnalysis.concerns[{index}]"
        concern = _object(concern, path)
        required = ("concern_id", "mention_ordinal", "kind", "raw_text", "canonical_metric", "formula_id", "metadata_field")
        _required(concern, required, path)
        concern_ids.append(_string(concern["concern_id"], f"{path}.concern_id"))
        _integer(concern["mention_ordinal"], f"{path}.mention_ordinal", minimum=0)
        _enum(concern["kind"], CONCERN_KINDS, f"{path}.kind")
        _string(concern["raw_text"], f"{path}.raw_text")
        for field in ("canonical_metric", "formula_id", "metadata_field"):
            if concern[field] is not None:
                _string(concern[field], f"{path}.{field}")
        if "normalized_unit" in concern and concern["normalized_unit"] is not None:
            _string(concern["normalized_unit"], f"{path}.normalized_unit")
        if "unit_source" in concern:
            _enum(
                concern["unit_source"],
                {"question", "request_hint", "catalog", "formula", "none", "ambiguous"},
                f"{path}.unit_source",
            )
    if len(concern_ids) != len(set(concern_ids)):
        raise ContractValidationError("QuestionAnalysis.concern_id values must be unique")
    cardinalities = _object(obj["cardinalities"], "QuestionAnalysis.cardinalities")
    _required(cardinalities, ("companies", "years", "metrics", "concerns"), "QuestionAnalysis.cardinalities")
    expected_cardinality = {
        "companies": len(companies), "years": len(expanded_years), "metrics": len(metrics), "concerns": len(concerns),
    }
    for name, expected in expected_cardinality.items():
        if _integer(cardinalities[name], f"QuestionAnalysis.cardinalities.{name}", minimum=0) != expected:
            raise ContractValidationError(f"QuestionAnalysis.cardinalities.{name} is inconsistent")
    intents = _unique_strings(obj["intents"], "QuestionAnalysis.intents")
    if not intents or not set(intents).issubset(INTENTS):
        raise ContractValidationError("QuestionAnalysis.intents must contain only frozen intents")
    narrative_mode = obj["narrative_mode"]
    if narrative_mode is not None:
        _enum(narrative_mode, NARRATIVE_MODES, "QuestionAnalysis.narrative_mode")
        if "narrative" not in intents:
            raise ContractValidationError("narrative_mode requires narrative intent")
    evidence_kinds = _unique_strings(obj["evidence_kinds"], "QuestionAnalysis.evidence_kinds")
    if not set(evidence_kinds).issubset(EVIDENCE_KINDS):
        raise ContractValidationError("QuestionAnalysis.required_evidence_kinds is invalid")
    output_entities = _array(obj["output_entity_axis"], "QuestionAnalysis.output_entity_axis")
    output_periods = _array(obj["output_period_axis"], "QuestionAnalysis.output_period_axis")
    for index, year in enumerate(output_periods):
        _integer(year, f"QuestionAnalysis.output_period_axis[{index}]", minimum=1900, maximum=2200)
    _boolean(obj["dynamic_target_dependency"], "QuestionAnalysis.dynamic_target_dependency")
    _array(obj["unsupported_markers"], "QuestionAnalysis.unsupported_markers")
    _array(obj["ambiguity_findings"], "QuestionAnalysis.ambiguity_findings")
    return obj


def validate_scope_plan(payload: Any) -> dict[str, Any] | None:
    if payload is None:
        return None
    obj = _object(payload, "ScopePlan")
    fields = {
        "schema_version", "scope_plan_id", "scope_kind", "resolver_version", "phase8_capabilities",
        "entity_resolutions", "report_year_constraints", "metric_years", "corpus_scope",
        "explicit_document_ids", "findings", "resolution_skipped_reason",
    }
    _required(obj, fields, "ScopePlan")
    _no_unknown(obj, fields, "ScopePlan")
    if obj["schema_version"] != SCHEMA_SCOPE:
        raise ContractValidationError("ScopePlan.schema_version is unsupported")
    _string(obj["scope_plan_id"], "ScopePlan.scope_plan_id")
    scope_kind = _enum(obj["scope_kind"], SCOPE_KINDS, "ScopePlan.scope_kind")
    _string(obj["resolver_version"], "ScopePlan.resolver_version")
    capabilities = _object(obj["phase8_capabilities"], "ScopePlan.phase8_capabilities")
    _required(capabilities, BACKENDS, "ScopePlan.phase8_capabilities")
    _no_unknown(capabilities, BACKENDS, "ScopePlan.phase8_capabilities")
    capability_values = {"direct", "entity_subscope", "single_document_subscope", "contract_only", "forbidden"}
    for backend in BACKENDS:
        _enum(capabilities[backend], capability_values, f"ScopePlan.phase8_capabilities.{backend}")
    expected_capabilities = {
        "single_document": {"fact": "direct", "sql": "direct", "formula": "direct", "evidence": "direct"},
        "company_documents": {"fact": "direct", "sql": "direct", "formula": "single_document_subscope", "evidence": "single_document_subscope"},
        "multi_company_documents": {"fact": "entity_subscope", "sql": "entity_subscope", "formula": "single_document_subscope", "evidence": "single_document_subscope"},
        "explicit_document_set": {backend: "contract_only" for backend in BACKENDS},
        "corpus": {"fact": "forbidden", "sql": "direct", "formula": "forbidden", "evidence": "forbidden"},
    }
    if capabilities != expected_capabilities[scope_kind]:
        raise ContractValidationError("ScopePlan.phase8_capabilities does not match the frozen scope matrix")
    entities = _array(obj["entity_resolutions"], "ScopePlan.entity_resolutions")
    for index, entity in enumerate(entities):
        entity_path = f"ScopePlan.entity_resolutions[{index}]"
        entity = _object(entity, entity_path)
        _required(entity, ("entity_key", "mention", "mention_ordinal", "status", "identity", "document_set", "findings"), entity_path)
        _string(entity["entity_key"], f"{entity_path}.entity_key")
        _string(entity["mention"], f"{entity_path}.mention")
        _integer(entity["mention_ordinal"], f"{entity_path}.mention_ordinal", minimum=0)
        _enum(entity["status"], ENTITY_RESOLUTION_STATUSES, f"{entity_path}.status")
        identity = entity["identity"]
        if entity["status"] == "unique":
            identity = _object(identity, f"{entity_path}.identity")
            _required(identity, ("stock_code", "stock_name", "company_full"), f"{entity_path}.identity")
            for field in ("stock_code", "stock_name", "company_full"):
                _string(identity[field], f"{entity_path}.identity.{field}")
        elif identity is not None:
            raise ContractValidationError(f"{entity_path}.identity must be null unless resolution is unique")
        documents = _array(entity["document_set"], f"{entity_path}.document_set")
        document_order: list[tuple[int, str]] = []
        for doc_index, document in enumerate(documents):
            doc_path = f"{entity_path}.document_set[{doc_index}]"
            document = _object(document, doc_path)
            _required(document, ("document_id", "stock_code", "report_year", "artifact_fingerprint"), doc_path)
            _string(document["document_id"], f"{doc_path}.document_id")
            _string(document["stock_code"], f"{doc_path}.stock_code")
            _integer(document["report_year"], f"{doc_path}.report_year", minimum=1900, maximum=2200)
            fingerprint = _string(document["artifact_fingerprint"], f"{doc_path}.artifact_fingerprint")
            if not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
                raise ContractValidationError(f"{doc_path}.artifact_fingerprint is invalid")
            document_order.append((document["report_year"], document["document_id"]))
        if document_order != sorted(document_order):
            raise ContractValidationError(f"{entity_path}.document_set must be stably sorted")
        _array(entity["findings"], f"{entity_path}.findings")
    for field in ("report_year_constraints", "metric_years"):
        values = _array(obj[field], f"ScopePlan.{field}")
        if len(values) != len(set(values)):
            raise ContractValidationError(f"ScopePlan.{field} has duplicates")
        for index, year in enumerate(values):
            _integer(year, f"ScopePlan.{field}[{index}]", minimum=1900, maximum=2200)
    _array(obj["findings"], "ScopePlan.findings")
    if scope_kind == "corpus":
        _object(obj["corpus_scope"], "ScopePlan.corpus_scope")
    elif obj["corpus_scope"] is not None:
        raise ContractValidationError("ScopePlan.corpus_scope is valid only for corpus scope")
    explicit_ids = _unique_strings(obj["explicit_document_ids"], "ScopePlan.explicit_document_ids")
    if scope_kind != "explicit_document_set" and explicit_ids:
        raise ContractValidationError("explicit_document_ids is contract-only")
    if obj["resolution_skipped_reason"] is not None:
        _string(obj["resolution_skipped_reason"], "ScopePlan.resolution_skipped_reason")
    return obj


def validate_subplan(payload: Any, *, path: str = "SubPlan") -> dict[str, Any]:
    obj = _object(payload, path)
    fields = {
        "subplan_id", "ordinal", "planning_state", "backend", "operation", "entity_key", "period_key",
        "concern_key", "scope_ref", "depends_on_subplan_ids", "authorization_source_subplan_ids",
        "required", "declared_scope", "payload", "planning_failure",
    }
    _required(obj, fields, path)
    _no_unknown(obj, fields, path)
    subplan_id = _string(obj["subplan_id"], f"{path}.subplan_id")
    if not _SUBPLAN_ID_RE.fullmatch(subplan_id):
        raise ContractValidationError(f"{path}.subplan_id is not canonical")
    _integer(obj["ordinal"], f"{path}.ordinal", minimum=0, maximum=LIMITS["max_subplans"] - 1)
    state = _enum(obj["planning_state"], PLANNING_STATES, f"{path}.planning_state")
    backend = _enum(obj["backend"], BACKENDS, f"{path}.backend")
    _string(obj["operation"], f"{path}.operation")
    for field in ("entity_key", "period_key"):
        if obj[field] is not None:
            _string(obj[field], f"{path}.{field}")
    _string(obj["concern_key"], f"{path}.concern_key")
    _string(obj["scope_ref"], f"{path}.scope_ref")
    dependencies = _unique_strings(obj["depends_on_subplan_ids"], f"{path}.depends_on_subplan_ids")
    for dependency in dependencies:
        if not _SUBPLAN_ID_RE.fullmatch(dependency):
            raise ContractValidationError(f"{path}.depends_on_subplan_ids contains a non-canonical ID")
    authorization_dependencies = _unique_strings(
        obj["authorization_source_subplan_ids"], f"{path}.authorization_source_subplan_ids"
    )
    if not set(authorization_dependencies).issubset(dependencies):
        raise ContractValidationError(f"{path}.authorization_source_subplan_ids must be completion dependencies")
    _boolean(obj["required"], f"{path}.required")
    scope = _object(obj["declared_scope"], f"{path}.declared_scope")
    _required(scope, ("scope_kind", "document_ids"), f"{path}.declared_scope")
    scope_kind = _enum(scope["scope_kind"], SCOPE_KINDS, f"{path}.declared_scope.scope_kind")
    document_ids = _unique_strings(scope["document_ids"], f"{path}.declared_scope.document_ids")
    if state == "ready":
        body = _object(obj["payload"], f"{path}.payload")
        if not body:
            raise ContractValidationError(f"{path}.payload must be non-empty when ready")
        if obj["planning_failure"] is not None:
            raise ContractValidationError(f"{path}.planning_failure must be null when ready")
        if backend == "evidence":
            if scope_kind != "single_document" or len(document_ids) != 1:
                raise ContractValidationError("every ready evidence subplan requires single_document scope")
            _required(body, ("document_id", "question", "top_k"), f"{path}.payload")
            _string(body["document_id"], f"{path}.payload.document_id")
            if body["document_id"] != document_ids[0]:
                raise ContractValidationError(f"{path}.payload.document_id must match declared scope")
            _string(body["question"], f"{path}.payload.question")
            _integer(body["top_k"], f"{path}.payload.top_k", minimum=1, maximum=LIMITS["max_evidence_top_k"])
    else:
        if obj["payload"] not in (None, {}):
            raise ContractValidationError(f"{path}.payload must be null or empty when blocked")
        failure = _object(obj["planning_failure"], f"{path}.planning_failure")
        _required(failure, ("failure_code", "message", "details"), f"{path}.planning_failure")
        _enum(failure["failure_code"], FAILURE_CODES, f"{path}.planning_failure.failure_code")
        _string(failure["message"], f"{path}.planning_failure.message")
        _object(failure["details"], f"{path}.planning_failure.details")
    return obj


def validate_composition_plan(payload: Any) -> dict[str, Any]:
    obj = _object(payload, "CompositionPlan")
    fields = {
        "schema_version", "composition_plan_id", "pattern_id", "pattern_version",
        "registry_semantic_sha256", "registry_file_sha256", "scope_plan", "scope_plan_id",
        "subplans", "composition_policy", "output_kind", "numeric_answer_policy", "limit_evaluation",
        "dynamic_expansion", "plan_fingerprint",
    }
    _required(obj, fields, "CompositionPlan")
    _no_unknown(obj, fields, "CompositionPlan")
    if obj["schema_version"] != SCHEMA_COMPOSITION:
        raise ContractValidationError("CompositionPlan.schema_version is unsupported")
    _string(obj["composition_plan_id"], "CompositionPlan.composition_plan_id")
    _enum(obj["pattern_id"], PATTERN_IDS, "CompositionPlan.pattern_id")
    _string(obj["pattern_version"], "CompositionPlan.pattern_version")
    registry_hash = _string(obj["registry_semantic_sha256"], "CompositionPlan.registry_semantic_sha256")
    if not re.fullmatch(r"[0-9a-f]{64}", registry_hash):
        raise ContractValidationError("CompositionPlan.registry_semantic_sha256 is invalid")
    file_hash = _string(obj["registry_file_sha256"], "CompositionPlan.registry_file_sha256")
    if not re.fullmatch(r"[0-9a-f]{64}", file_hash):
        raise ContractValidationError("CompositionPlan.registry_file_sha256 is invalid")
    scope_plan = validate_scope_plan(obj["scope_plan"])
    scope_plan_id = _string(obj["scope_plan_id"], "CompositionPlan.scope_plan_id")
    if scope_plan is None or scope_plan["scope_plan_id"] != scope_plan_id:
        raise ContractValidationError("CompositionPlan.scope_plan_id must reference ScopePlan")
    subplans = _array(obj["subplans"], "CompositionPlan.subplans")
    if not subplans or len(subplans) > LIMITS["max_subplans"]:
        raise ContractValidationError("CompositionPlan.subplans must contain 1..12 items")
    seen: set[str] = set()
    evidence_budget = 0
    for index, subplan in enumerate(subplans):
        validated = validate_subplan(subplan, path=f"CompositionPlan.subplans[{index}]")
        if validated["ordinal"] != index:
            raise ContractValidationError("CompositionPlan subplan ordinals must be contiguous and ordered")
        expected_id = make_subplan_id(
            registry_hash, obj["pattern_id"], obj["pattern_version"], index,
            validated["entity_key"], validated["period_key"], validated["concern_key"],
        )
        if validated["subplan_id"] != expected_id:
            raise ContractValidationError("CompositionPlan contains a non-canonical subplan_id")
        subplan_id = validated["subplan_id"]
        if subplan_id in seen:
            raise ContractValidationError("CompositionPlan.subplan_id values must be unique")
        unknown_dependencies = set(validated["depends_on_subplan_ids"]) - seen
        if unknown_dependencies:
            raise ContractValidationError("CompositionPlan dependencies must refer to earlier subplans")
        seen.add(subplan_id)
        if validated["planning_state"] == "ready" and validated["backend"] == "evidence":
            evidence_budget += int(validated["payload"]["top_k"])
    if evidence_budget > LIMITS["max_evidence_chunks"]:
        raise ContractValidationError("CompositionPlan evidence chunk budget exceeded")
    policy = _object(obj["composition_policy"], "CompositionPlan.composition_policy")
    policy_fields = {
        "policy_id", "required_subplan_ids", "optional_subplan_ids", "minimum_usable_results",
        "quorum_selector", "section_axis", "comparison_measure_compatibility",
        "allow_cross_document_claims", "allow_cross_entity_causal_inference",
    }
    _required(policy, policy_fields, "CompositionPlan.composition_policy")
    _no_unknown(policy, policy_fields, "CompositionPlan.composition_policy")
    _string(policy["policy_id"], "CompositionPlan.composition_policy.policy_id")
    required_ids = _unique_strings(policy["required_subplan_ids"], "CompositionPlan.composition_policy.required_subplan_ids")
    optional_ids = _unique_strings(policy["optional_subplan_ids"], "CompositionPlan.composition_policy.optional_subplan_ids")
    if optional_ids:
        raise ContractValidationError("Phase 8 v1 does not emit optional helper subplans")
    if set(required_ids) != seen:
        raise ContractValidationError("all Phase 8 v1 subplans must be required")
    minimum = _integer(policy["minimum_usable_results"], "CompositionPlan.composition_policy.minimum_usable_results", minimum=1)
    if minimum > len(subplans):
        raise ContractValidationError("minimum_usable_results exceeds subplan count")
    quorum = policy["quorum_selector"]
    if quorum is not None:
        quorum = _object(quorum, "CompositionPlan.composition_policy.quorum_selector")
        quorum_fields = {"group_by", "distinct_key", "eligible_backends", "eligible_operations", "count"}
        _required(quorum, quorum_fields, "CompositionPlan.composition_policy.quorum_selector")
        _no_unknown(quorum, quorum_fields, "CompositionPlan.composition_policy.quorum_selector")
        _enum(quorum["group_by"], {"concern", "global"}, "CompositionPlan.composition_policy.quorum_selector.group_by")
        _enum(quorum["distinct_key"], {"entity_key", "period_key", "concern_key"}, "CompositionPlan.composition_policy.quorum_selector.distinct_key")
        eligible_backends = _unique_strings(quorum["eligible_backends"], "CompositionPlan.composition_policy.quorum_selector.eligible_backends")
        if not eligible_backends or not set(eligible_backends).issubset(BACKENDS - {"evidence"}):
            raise ContractValidationError("comparison quorum cannot count evidence results")
        _unique_strings(quorum["eligible_operations"], "CompositionPlan.composition_policy.quorum_selector.eligible_operations")
        _integer(quorum["count"], "CompositionPlan.composition_policy.quorum_selector.count", minimum=2, maximum=LIMITS["max_companies"])
    _enum(policy["section_axis"], {"none", "entity", "period", "concern"}, "CompositionPlan.composition_policy.section_axis")
    _enum(
        policy["comparison_measure_compatibility"],
        {"not_applicable", "same_unit", "same_measure_and_unit"},
        "CompositionPlan.composition_policy.comparison_measure_compatibility",
    )
    if _boolean(policy["allow_cross_document_claims"], "CompositionPlan.composition_policy.allow_cross_document_claims"):
        raise ContractValidationError("cross-document claims are forbidden in Phase 8")
    if _boolean(policy["allow_cross_entity_causal_inference"], "CompositionPlan.composition_policy.allow_cross_entity_causal_inference"):
        raise ContractValidationError("cross-entity causal inference is forbidden in Phase 8")
    _string(obj["output_kind"], "CompositionPlan.output_kind")
    _object(obj["numeric_answer_policy"], "CompositionPlan.numeric_answer_policy")
    limits = _object(obj["limit_evaluation"], "CompositionPlan.limit_evaluation")
    _required(limits, ("company_count", "output_period_count", "subplan_count", "evidence_chunk_budget", "within_limits"), "CompositionPlan.limit_evaluation")
    _integer(limits["company_count"], "CompositionPlan.limit_evaluation.company_count", minimum=0, maximum=LIMITS["max_companies"])
    _integer(limits["output_period_count"], "CompositionPlan.limit_evaluation.output_period_count", minimum=0, maximum=LIMITS["max_years"])
    _integer(limits["subplan_count"], "CompositionPlan.limit_evaluation.subplan_count", minimum=1, maximum=LIMITS["max_subplans"])
    _integer(limits["evidence_chunk_budget"], "CompositionPlan.limit_evaluation.evidence_chunk_budget", minimum=0, maximum=LIMITS["max_evidence_chunks"])
    if not _boolean(limits["within_limits"], "CompositionPlan.limit_evaluation.within_limits"):
        raise ContractValidationError("a CompositionPlan cannot be emitted when limits are exceeded")
    if _boolean(obj["dynamic_expansion"], "CompositionPlan.dynamic_expansion"):
        raise ContractValidationError("dynamic subplan expansion is forbidden in Phase 8")
    plan_fingerprint = _string(obj["plan_fingerprint"], "CompositionPlan.plan_fingerprint")
    if not re.fullmatch(r"[0-9a-f]{64}", plan_fingerprint):
        raise ContractValidationError("CompositionPlan.plan_fingerprint is invalid")
    unhashed = dict(obj)
    unhashed.pop("plan_fingerprint")
    if plan_fingerprint != semantic_sha256(unhashed):
        raise ContractValidationError("CompositionPlan.plan_fingerprint is not canonical")
    return obj


def validate_numeric_authorization(payload: Any) -> dict[str, Any]:
    obj = _object(payload, "NumericAuthorization")
    fields = {
        "schema_version", "authorization_id", "source_subplan_id", "source_backend", "source_result_row",
        "entity_key", "company", "document_id", "measure", "normalized_value", "normalized_unit",
        "output_kind", "precision", "rounding", "allowed_renderings", "source_citation_ids",
        "provenance_citation_ids",
    }
    _required(obj, fields, "NumericAuthorization")
    _no_unknown(obj, fields, "NumericAuthorization")
    if obj["schema_version"] != SCHEMA_NUMERIC_AUTHORIZATION:
        raise ContractValidationError("NumericAuthorization.schema_version is unsupported")
    for field in ("authorization_id", "source_subplan_id", "entity_key", "company", "normalized_unit", "output_kind", "rounding"):
        _string(obj[field], f"NumericAuthorization.{field}")
    _enum(obj["source_backend"], BACKENDS - {"evidence"}, "NumericAuthorization.source_backend")
    _integer(obj["source_result_row"], "NumericAuthorization.source_result_row", minimum=0)
    if obj["document_id"] is not None:
        _string(obj["document_id"], "NumericAuthorization.document_id")
    measure = _object(obj["measure"], "NumericAuthorization.measure")
    measure_kind = _enum(measure.get("kind"), {"canonical_fact", "formula_result", "sql_result"}, "NumericAuthorization.measure.kind")
    if measure_kind == "canonical_fact":
        measure_fields = {"kind", "canonical_metric", "metric_year"}
        _required(measure, measure_fields, "NumericAuthorization.measure")
        _no_unknown(measure, measure_fields, "NumericAuthorization.measure")
        _string(measure["canonical_metric"], "NumericAuthorization.measure.canonical_metric")
        _integer(measure["metric_year"], "NumericAuthorization.measure.metric_year", minimum=1900, maximum=2200)
    elif measure_kind == "formula_result":
        measure_fields = {"kind", "formula_id", "formula_version", "target_year", "operand_years"}
        _required(measure, measure_fields, "NumericAuthorization.measure")
        _no_unknown(measure, measure_fields, "NumericAuthorization.measure")
        _string(measure["formula_id"], "NumericAuthorization.measure.formula_id")
        _string(measure["formula_version"], "NumericAuthorization.measure.formula_version")
        _integer(measure["target_year"], "NumericAuthorization.measure.target_year", minimum=1900, maximum=2200)
        operand_years = _array(measure["operand_years"], "NumericAuthorization.measure.operand_years")
        for index, year in enumerate(operand_years):
            _integer(year, f"NumericAuthorization.measure.operand_years[{index}]", minimum=1900, maximum=2200)
    else:
        measure_fields = {
            "kind", "query_spec_id", "result_row_id", "measure_id",
            "canonical_metric", "metric_year",
        }
        _required(measure, measure_fields, "NumericAuthorization.measure")
        _no_unknown(measure, measure_fields, "NumericAuthorization.measure")
        for field in ("query_spec_id", "result_row_id", "measure_id", "canonical_metric"):
            _string(measure[field], f"NumericAuthorization.measure.{field}")
        _integer(measure["metric_year"], "NumericAuthorization.measure.metric_year", minimum=1900, maximum=2200)
    _decimal_string(obj["normalized_value"], "NumericAuthorization.normalized_value")
    _integer(obj["precision"], "NumericAuthorization.precision", minimum=0, maximum=12)
    source_citations = _unique_strings(obj["source_citation_ids"], "NumericAuthorization.source_citation_ids")
    provenance_citations = _unique_strings(obj["provenance_citation_ids"], "NumericAuthorization.provenance_citation_ids")
    if not source_citations or not provenance_citations:
        raise ContractValidationError("NumericAuthorization requires source and provenance citations")
    renderings = _unique_strings(obj["allowed_renderings"], "NumericAuthorization.allowed_renderings")
    if not renderings or renderings != sorted(renderings):
        raise ContractValidationError("NumericAuthorization.allowed_renderings cannot be empty")
    return obj


def validate_numeric_authorization_set(payload: Any) -> dict[str, Any]:
    obj = _object(payload, "NumericAuthorizationSet")
    fields = {"schema_version", "items", "set_fingerprint"}
    _required(obj, fields, "NumericAuthorizationSet")
    _no_unknown(obj, fields, "NumericAuthorizationSet")
    if obj["schema_version"] != SCHEMA_NUMERIC_AUTHORIZATION_SET:
        raise ContractValidationError("NumericAuthorizationSet.schema_version is unsupported")
    items = _array(obj["items"], "NumericAuthorizationSet.items")
    for item in items:
        validate_numeric_authorization(item)
    expected_order = sorted(
        items,
        key=lambda item: (item["source_subplan_id"], item["source_result_row"], item["authorization_id"]),
    )
    if items != expected_order:
        raise ContractValidationError("NumericAuthorizationSet.items must be stably sorted")
    fingerprint = _string(obj["set_fingerprint"], "NumericAuthorizationSet.set_fingerprint")
    if fingerprint != semantic_sha256({"schema_version": obj["schema_version"], "items": items}):
        raise ContractValidationError("NumericAuthorizationSet.set_fingerprint is invalid")
    return obj


def validate_missing_fact_request(payload: Any) -> dict[str, Any]:
    obj = _object(payload, "MissingFactRequest")
    fields = {
        "schema_version", "requirement_id", "origin_operation", "formula_id", "operand_role",
        "subplan_id", "document_id", "stock_code", "report_year", "metric_year",
        "canonical_metric", "normalized_unit", "candidate_table_ids",
    }
    _required(obj, fields, "MissingFactRequest")
    _no_unknown(obj, fields, "MissingFactRequest")
    if obj["schema_version"] != SCHEMA_MISSING_FACT_REQUEST:
        raise ContractValidationError("MissingFactRequest.schema_version is unsupported")
    for field in ("requirement_id", "origin_operation", "subplan_id", "document_id", "stock_code", "canonical_metric", "normalized_unit"):
        _string(obj[field], f"MissingFactRequest.{field}")
    for field in ("formula_id", "operand_role"):
        if obj[field] is not None:
            _string(obj[field], f"MissingFactRequest.{field}")
    origin = obj["origin_operation"]
    if origin not in {"fact_lookup", "formula_compute", "document_query"}:
        raise ContractValidationError("MissingFactRequest.origin_operation is unsupported")
    if origin == "formula_compute":
        if obj["formula_id"] is None or obj["operand_role"] is None:
            raise ContractValidationError("Formula MissingFactRequest requires formula_id and operand_role")
        from .metric_catalog import MetricCatalog
        formula = MetricCatalog().formula(obj["formula_id"])
        if formula is None or obj["operand_role"] not in {role for role, _, _ in formula.operands}:
            raise ContractValidationError("Formula MissingFactRequest identity is not registered")
    elif obj["formula_id"] is not None or obj["operand_role"] is not None:
        raise ContractValidationError("Non-formula MissingFactRequest cannot bind formula fields")
    for field in ("report_year", "metric_year"):
        _integer(obj[field], f"MissingFactRequest.{field}", minimum=1900, maximum=2200)
    table_ids = _unique_strings(obj["candidate_table_ids"], "MissingFactRequest.candidate_table_ids")
    if table_ids != sorted(table_ids):
        raise ContractValidationError("MissingFactRequest.candidate_table_ids must be sorted")
    expected = make_requirement_id(obj)
    if obj["requirement_id"] != expected:
        raise ContractValidationError("MissingFactRequest.requirement_id is not canonical")
    return obj


def validate_citation(payload: Any, *, path: str = "Citation") -> dict[str, Any]:
    obj = _object(payload, path)
    fields = {
        "citation_id", "citation_kind", "subplan_id", "entity_key", "document_id",
        "source_citation_ids", "provenance",
    }
    _required(obj, fields, path)
    _no_unknown(obj, fields, path)
    _string(obj["citation_id"], f"{path}.citation_id")
    kind = _enum(
        obj["citation_kind"],
        {"fact", "metadata", "evidence", "formula_derivation", "sql_derivation"},
        f"{path}.citation_kind",
    )
    _string(obj["subplan_id"], f"{path}.subplan_id")
    if obj["entity_key"] is not None:
        _string(obj["entity_key"], f"{path}.entity_key")
    if obj["document_id"] is not None:
        _string(obj["document_id"], f"{path}.document_id")
    if kind in {"fact", "evidence", "formula_derivation", "sql_derivation"} and obj["document_id"] is None:
        raise ContractValidationError(f"{path}.document_id is required for {kind}")
    source_ids = _unique_strings(obj["source_citation_ids"], f"{path}.source_citation_ids")
    if kind in {"formula_derivation", "sql_derivation"} and not source_ids:
        raise ContractValidationError(f"{path} derivation citations require source citations")
    _object(obj["provenance"], f"{path}.provenance")
    return obj


def validate_subplan_result(payload: Any, *, path: str = "SubPlanResult") -> dict[str, Any]:
    obj = _object(payload, path)
    fields = {
        "schema_version", "subplan_id", "backend", "operation", "planning_state", "execution_state",
        "status", "result", "claims", "citations", "failure_code", "errors", "warnings",
        "missing_fact_requests", "trace",
    }
    _required(obj, fields, path)
    _no_unknown(obj, fields, path)
    if obj["schema_version"] != SCHEMA_SUBPLAN_RESULT:
        raise ContractValidationError(f"{path}.schema_version is unsupported")
    _string(obj["subplan_id"], f"{path}.subplan_id")
    backend = _enum(obj["backend"], BACKENDS, f"{path}.backend")
    _string(obj["operation"], f"{path}.operation")
    planning_state = _enum(obj["planning_state"], PLANNING_STATES, f"{path}.planning_state")
    execution_state = _enum(obj["execution_state"], EXECUTION_STATES, f"{path}.execution_state")
    status = _enum(obj["status"], SUBPLAN_STATUSES, f"{path}.status")
    if planning_state == "blocked" and execution_state != "not_executed":
        raise ContractValidationError(f"{path} blocked plans cannot execute")
    claims = _array(obj["claims"], f"{path}.claims")
    citations = _array(obj["citations"], f"{path}.citations")
    for index, citation in enumerate(citations):
        validate_citation(citation, path=f"{path}.citations[{index}]")
    if obj["failure_code"] is not None:
        _enum(obj["failure_code"], FAILURE_CODES, f"{path}.failure_code")
    _array(obj["errors"], f"{path}.errors")
    _array(obj["warnings"], f"{path}.warnings")
    requests = _array(obj["missing_fact_requests"], f"{path}.missing_fact_requests")
    for request in requests:
        validate_missing_fact_request(request)
    if (status == "fallback_required") != bool(requests):
        raise ContractValidationError(f"{path} fallback_required and missing_fact_requests must be equivalent")
    _object(obj["trace"], f"{path}.trace")
    usable = is_subplan_result_usable(obj)
    if status == "ok" and not usable:
        raise ContractValidationError(f"{path} status=ok requires a backend-valid usable result")
    if backend == "evidence" and status in {"ok", "partial"} and not claims:
        raise ContractValidationError(f"{path} evidence chunks without claims are not usable")
    return obj


def is_subplan_result_usable(payload: Mapping[str, Any]) -> bool:
    """Derive usability from backend output; executors cannot self-assert it."""

    if payload.get("status") not in {"ok", "partial"} or payload.get("result") is None:
        return False
    backend = payload.get("backend")
    result = payload.get("result")
    if not isinstance(result, dict):
        return False
    if backend == "fact":
        return all(result.get(field) not in (None, "", []) for field in ("value", "normalized_unit", "provenance"))
    if backend == "formula":
        operands = result.get("operands")
        return bool(
            result.get("value") not in (None, "")
            and result.get("normalized_unit")
            and isinstance(operands, list)
            and operands
            and all(isinstance(row, dict) and row.get("citation_ids") for row in operands)
        )
    if backend == "sql":
        rows = result.get("rows")
        return bool(
            isinstance(rows, list)
            and rows
            and all(isinstance(row, dict) and row.get("contributing_fact_ids") for row in rows)
        )
    if backend == "evidence":
        claims = payload.get("claims")
        return bool(
            isinstance(claims, list)
            and claims
            and all(isinstance(claim, dict) and claim.get("citation_ids") for claim in claims)
        )
    return False


def validate_qa_answer(payload: Any) -> dict[str, Any]:
    obj = _object(payload, "QAAnswer")
    fields = {
        "schema_version", "request_id", "status", "composition_pattern_id", "document_scope",
        "subplans", "answer_text", "result", "citations", "trace", "errors", "warnings",
        "missing_fact_requests",
    }
    _required(obj, fields, "QAAnswer")
    _no_unknown(obj, fields, "QAAnswer")
    if obj["schema_version"] != SCHEMA_ANSWER:
        raise ContractValidationError("QAAnswer.schema_version is unsupported")
    _string(obj["request_id"], "QAAnswer.request_id")
    status = _enum(obj["status"], ANSWER_STATUSES, "QAAnswer.status")
    if obj["composition_pattern_id"] is not None:
        _enum(obj["composition_pattern_id"], PATTERN_IDS, "QAAnswer.composition_pattern_id")
    validate_scope_plan(obj["document_scope"])
    results = _array(obj["subplans"], "QAAnswer.subplans")
    if len(results) > LIMITS["max_subplans"]:
        raise ContractValidationError("QAAnswer.subplans exceeds the limit")
    for index, result in enumerate(results):
        validate_subplan_result(result, path=f"QAAnswer.subplans[{index}]")
    _string(obj["answer_text"], "QAAnswer.answer_text", allow_empty=True)
    citations = _array(obj["citations"], "QAAnswer.citations")
    for index, citation in enumerate(citations):
        validate_citation(citation, path=f"QAAnswer.citations[{index}]")
    errors = _array(obj["errors"], "QAAnswer.errors")
    _array(obj["warnings"], "QAAnswer.warnings")
    missing = _array(obj["missing_fact_requests"], "QAAnswer.missing_fact_requests")
    for request in missing:
        validate_missing_fact_request(request)
    flattened = [item for result in results for item in result["missing_fact_requests"]]
    if missing != flattened:
        raise ContractValidationError("QAAnswer.missing_fact_requests must preserve subplan order without merging")
    provenance_failure = any(
        isinstance(error, dict) and error.get("failure_code") == "PROVENANCE_VALIDATION_FAILED"
        for error in errors
    )
    if provenance_failure and (
        status != "blocked" or obj["answer_text"] or obj["result"] not in (None, {}, []) or citations
    ):
        raise ContractValidationError("provenance failure must block the whole answer and suppress outputs")
    if obj["composition_pattern_id"] is None and results:
        raise ContractValidationError("pre-composition failures cannot contain subplans")
    return obj


def validate_qa_trace(payload: Any) -> dict[str, Any]:
    obj = _object(payload, "QATrace")
    fields = {
        "schema_version", "request_id", "question_analysis", "scope_plan", "composition_plan",
        "composition_decision", "subplan_traces", "numeric_authorization_set", "composer",
        "artifact_fingerprints", "trace_hash",
    }
    _required(obj, fields, "QATrace")
    _no_unknown(obj, fields, "QATrace")
    if obj["schema_version"] != SCHEMA_TRACE:
        raise ContractValidationError("QATrace.schema_version is unsupported")
    _string(obj["request_id"], "QATrace.request_id")
    if obj["question_analysis"] is not None:
        validate_question_analysis(obj["question_analysis"])
    validate_scope_plan(obj["scope_plan"])
    if obj["composition_plan"] is not None:
        validate_composition_plan(obj["composition_plan"])
    _object(obj["composition_decision"], "QATrace.composition_decision")
    _array(obj["subplan_traces"], "QATrace.subplan_traces")
    validate_numeric_authorization_set(obj["numeric_authorization_set"])
    _object(obj["composer"], "QATrace.composer")
    _object(obj["artifact_fingerprints"], "QATrace.artifact_fingerprints")
    trace_hash = _string(obj["trace_hash"], "QATrace.trace_hash")
    if not re.fullmatch(r"[0-9a-f]{64}", trace_hash):
        raise ContractValidationError("QATrace.trace_hash is invalid")
    unhashed = dict(obj)
    unhashed.pop("trace_hash")
    if trace_hash != semantic_sha256(unhashed):
        raise ContractValidationError("QATrace.trace_hash does not match deterministic trace content")
    forbidden = {"timestamp", "generated_at", "latency", "elapsed", "duration", "pid", "process_id", "temporary_path"}
    stack: list[tuple[str, Any]] = [("QATrace", obj)]
    while stack:
        path, current = stack.pop()
        if isinstance(current, dict):
            for key, value in current.items():
                normalized_key = key.lower()
                if any(marker in normalized_key for marker in forbidden):
                    raise ContractValidationError(f"{path}.{key} is runtime telemetry, not deterministic trace")
                stack.append((f"{path}.{key}", value))
        elif isinstance(current, list):
            stack.extend((f"{path}[]", value) for value in current)
        elif isinstance(current, float):
            raise ContractValidationError(f"{path} contains a float; deterministic numbers must be strings")
        elif isinstance(current, str) and current.startswith("/"):
            raise ContractValidationError(f"{path} contains an absolute path")
    return obj
