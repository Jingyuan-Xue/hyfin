"""Deterministic static topology planning for Phase 8.

The planner is deliberately execution-blind: it freezes every node before a
backend is called and raises :class:`CompositionPlanningError` for terminal
precomposition outcomes.  The exception always carries an empty subplan list,
which makes the zero-backend-call boundary explicit to the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from .contracts import (
    LIMITS,
    SCHEMA_COMPOSITION,
    canonical_json_bytes,
    make_subplan_id,
    semantic_sha256,
    validate_composition_plan,
    validate_pattern_registry,
    validate_question_analysis,
    validate_scope_plan,
)
from .errors import PRECOMPOSITION_STATUS_BY_CODE


DEFAULT_REGISTRY_PATH = Path(__file__).resolve().parents[1] / "config" / "composition_patterns.json"


@dataclass(frozen=True)
class CompositionPlanningError(Exception):
    """Terminal planning result for which no CompositionPlan may be emitted."""

    failure_code: str
    message: str
    details: Mapping[str, Any]

    def __post_init__(self) -> None:
        Exception.__init__(self, self.message)

    @property
    def status(self) -> str:
        return PRECOMPOSITION_STATUS_BY_CODE.get(self.failure_code, "unsupported")

    @property
    def subplans(self) -> list[dict[str, Any]]:
        return []

    @property
    def backend_call_count(self) -> int:
        return 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "failure_code": self.failure_code,
            "message": self.message,
            "details": dict(self.details),
            "subplans": [],
            "backend_call_count": 0,
        }


class TopologyCompositionPlanner:
    """Choose one frozen topology and materialize its ordered SubPlans."""

    planner_version = "1.0.0"

    def __init__(
        self,
        registry_path: str | Path = DEFAULT_REGISTRY_PATH,
        *,
        evidence_top_k: int = 5,
    ) -> None:
        self.registry_path = Path(registry_path)
        raw = self.registry_path.read_bytes()
        self.registry_file_sha256 = hashlib.sha256(raw).hexdigest()
        self.registry = validate_pattern_registry(json.loads(raw))
        self.registry_semantic_sha256 = semantic_sha256(self.registry)
        self.evidence_top_k = evidence_top_k
        self._patterns = {row["pattern_id"]: row for row in self.registry["patterns"]}

    def plan(self, analysis: Mapping[str, Any], scope_plan: Mapping[str, Any]) -> dict[str, Any]:
        """Return a validated CompositionPlan or raise a zero-call terminal error."""

        self._check_raw_boundaries(analysis, scope_plan)
        checked_analysis = validate_question_analysis(dict(analysis))
        checked_scope = validate_scope_plan(dict(scope_plan))
        if checked_scope is None:  # defensive; public signature requires a scope
            self._terminal("COMPOSITION_UNSUPPORTED", "Scope resolution did not produce a plan")

        pattern_id = self._match_pattern(checked_analysis, checked_scope)
        pattern = self._patterns[pattern_id]
        if checked_scope["scope_kind"] not in pattern["allowed_scope_kinds"]:
            self._terminal(
                "COMPOSITION_UNSUPPORTED",
                "Matched topology does not allow the resolved scope",
                pattern_id=pattern_id,
                scope_kind=checked_scope["scope_kind"],
            )
        disallowed_intents = sorted(set(checked_analysis["intents"]) - set(pattern["allowed_intents"]))
        if disallowed_intents:
            self._terminal(
                "COMPOSITION_UNSUPPORTED",
                "Matched topology does not allow all analyzed intents",
                pattern_id=pattern_id,
                disallowed_intents=disallowed_intents,
            )

        drafts = self._draft_subplans(pattern_id, checked_analysis, checked_scope)
        if not drafts:
            self._terminal("COMPOSITION_UNSUPPORTED", "No executable or blocked explicit concern was planned")
        if len(drafts) > LIMITS["max_subplans"]:
            self._terminal(
                "COMPOSITION_LIMIT_EXCEEDED",
                "Static decomposition exceeds the SubPlan limit",
                limit=LIMITS["max_subplans"],
                actual=len(drafts),
            )

        evidence_budget = sum(
            int(row["payload"]["top_k"])
            for row in drafts
            if row["planning_state"] == "ready" and row["backend"] == "evidence"
        )
        if self.evidence_top_k > LIMITS["max_evidence_top_k"] or evidence_budget > LIMITS["max_evidence_chunks"]:
            self._terminal(
                "COMPOSITION_LIMIT_EXCEEDED",
                "Evidence retrieval budget exceeds a frozen limit",
                top_k=self.evidence_top_k,
                evidence_chunk_budget=evidence_budget,
                max_top_k=LIMITS["max_evidence_top_k"],
                max_chunks=LIMITS["max_evidence_chunks"],
            )

        subplans = self._finalize_subplans(drafts, pattern)
        compare = "compare" in checked_analysis["intents"]
        quorum = pattern["quorum_selector"] if compare else None
        section_axis = self._section_axis(pattern_id)
        policy = {
            "policy_id": pattern["composition_policy_id"],
            "required_subplan_ids": [row["subplan_id"] for row in subplans],
            "optional_subplan_ids": [],
            "minimum_usable_results": pattern["minimum_usable_results"],
            "quorum_selector": quorum,
            "section_axis": section_axis,
            "comparison_measure_compatibility": (
                pattern["comparison_measure_compatibility"] if compare else "not_applicable"
            ),
            "allow_cross_document_claims": False,
            "allow_cross_entity_causal_inference": False,
        }
        plan: dict[str, Any] = {
            "schema_version": SCHEMA_COMPOSITION,
            "composition_plan_id": self._composition_plan_id(
                checked_analysis, checked_scope, pattern_id, subplans
            ),
            "pattern_id": pattern_id,
            "pattern_version": pattern["version"],
            "registry_semantic_sha256": self.registry_semantic_sha256,
            "registry_file_sha256": self.registry_file_sha256,
            "scope_plan": checked_scope,
            "scope_plan_id": checked_scope["scope_plan_id"],
            "subplans": subplans,
            "composition_policy": policy,
            "output_kind": self._output_kind(checked_analysis, pattern_id),
            "numeric_answer_policy": {
                "structured_results_are_numeric_authority": True,
                "evidence_financial_numbers_require_authorization": True,
                "comparison_requires_same_measure_and_unit": (
                    compare and pattern["comparison_measure_compatibility"] == "same_measure_and_unit"
                ),
            },
            "limit_evaluation": {
                "company_count": self._company_count(checked_analysis, checked_scope),
                "output_period_count": len(checked_analysis["output_period_axis"]),
                "subplan_count": len(subplans),
                "evidence_chunk_budget": evidence_budget,
                "within_limits": True,
            },
            "dynamic_expansion": False,
        }
        plan["plan_fingerprint"] = semantic_sha256(plan)
        return validate_composition_plan(plan)

    def _check_raw_boundaries(
        self, analysis: Mapping[str, Any], scope_plan: Mapping[str, Any]
    ) -> None:
        marker_rows = analysis.get("unsupported_markers", [])
        markers: dict[str, Any] = {}
        if isinstance(marker_rows, list):
            for row in marker_rows:
                if isinstance(row, str):
                    markers[row] = {"code": row}
                elif isinstance(row, Mapping):
                    code = row.get("code") or row.get("marker") or row.get("failure_code")
                    if isinstance(code, str):
                        markers[code] = dict(row)
        limit_markers = [
            name for name in ("max_companies_exceeded", "max_years_exceeded", "evidence_top_k_exceeded")
            if name in markers
        ]
        if limit_markers:
            self._terminal(
                "COMPOSITION_LIMIT_EXCEEDED",
                "Question analysis recorded a frozen planning limit violation",
                markers=[markers[name] for name in limit_markers],
            )
        unsupported_markers = [
            name for name in ("two_dimensional_output_axis", "dynamic_subplan_expansion_required")
            if name in markers
        ]
        if unsupported_markers:
            self._terminal(
                "COMPOSITION_UNSUPPORTED",
                "Question analysis requires an unsupported composition topology",
                markers=[markers[name] for name in unsupported_markers],
            )
        # ``table_evidence_unavailable_phase7`` is intentionally not a global
        # rejection.  The analyzer represents that explicit table task with a
        # narrative-carrier concern; _evidence_draft turns only that node into
        # a blocked COMPOSITION_UNSUPPORTED placeholder.  Structured siblings
        # can therefore remain usable without ever invoking A2RAG for a table.
        if "no_supported_concern" in markers:
            self._terminal(
                "COMPOSITION_UNSUPPORTED",
                "Question analysis found no supported Phase 8 concern",
                marker=markers["no_supported_concern"],
            )
        if bool(analysis.get("dynamic_target_dependency")):
            self._terminal(
                "COMPOSITION_UNSUPPORTED",
                "Execution-result-dependent SubPlan creation is outside Phase 8",
                dynamic_expansion=False,
            )

        entities = analysis.get("output_entity_axis", [])
        periods = analysis.get("output_period_axis", [])
        scope_entities = scope_plan.get("entity_resolutions", [])
        company_count = max(
            len(entities) if isinstance(entities, list) else 0,
            len(scope_entities) if isinstance(scope_entities, list) else 0,
        )
        period_count = len(periods) if isinstance(periods, list) else 0
        if company_count > LIMITS["max_companies"]:
            self._terminal(
                "COMPOSITION_LIMIT_EXCEEDED", "Company axis exceeds the frozen limit",
                limit=LIMITS["max_companies"], actual=company_count,
            )
        if period_count > LIMITS["max_years"]:
            self._terminal(
                "COMPOSITION_LIMIT_EXCEEDED", "Output period axis exceeds the frozen limit",
                limit=LIMITS["max_years"], actual=period_count,
            )
        if company_count > 1 and period_count > 1:
            self._terminal(
                "COMPOSITION_UNSUPPORTED",
                "Simultaneous entity and period output fan-out is outside Phase 8",
                company_count=company_count,
                output_period_count=period_count,
            )
        if self.evidence_top_k < 1 or self.evidence_top_k > LIMITS["max_evidence_top_k"]:
            self._terminal(
                "COMPOSITION_LIMIT_EXCEEDED", "Evidence top_k exceeds the frozen limit",
                limit=LIMITS["max_evidence_top_k"], actual=self.evidence_top_k,
            )
        if scope_plan.get("scope_kind") == "explicit_document_set":
            self._terminal(
                "COMPOSITION_UNSUPPORTED",
                "Explicit multi-document execution is contract-only in Phase 8",
            )

    def _match_pattern(self, analysis: Mapping[str, Any], scope: Mapping[str, Any]) -> str:
        intents = set(analysis["intents"])
        narrative = "narrative" in intents
        compare = "compare" in intents
        entity_count = self._company_count(analysis, scope)
        period_count = len(analysis["output_period_axis"])
        concerns = self._ordered_concerns(analysis)
        structured_count = sum(row["kind"] != "narrative" for row in concerns)
        narrative_count = sum(row["kind"] == "narrative" for row in concerns)

        # SQL rank/aggregate over an already-known scope is one static node.  A
        # result-dependent narrative target is rejected earlier by analysis.
        if ("rank" in intents or "aggregate" in intents) and not narrative:
            candidates = ["single_node"] if len(concerns) == 1 else []
        elif narrative:
            candidates = []
            if entity_count > 1 and period_count <= 1:
                candidates.append("entity_section_bundle")
            if period_count > 1 and entity_count <= 1:
                candidates.append("period_section_bundle")
            if entity_count <= 1 and period_count <= 1 and structured_count and narrative_count:
                candidates.append("single_document_bundle")
            if entity_count <= 1 and period_count <= 1 and not structured_count and narrative_count == 1:
                candidates.append("single_node")
        elif entity_count > 1:
            candidates = ["entity_compare" if compare else "entity_list"]
        elif period_count > 1:
            candidates = ["period_compare" if compare else "period_list"]
        elif len(concerns) > 1:
            candidates = ["parallel_concerns"]
        elif len(concerns) == 1:
            candidates = ["single_node"]
        else:
            candidates = []

        if not candidates:
            self._terminal(
                "COMPOSITION_UNSUPPORTED", "No frozen topology can express the analyzed request",
                entity_count=entity_count, period_count=period_count,
                concern_count=len(concerns), intents=sorted(intents),
            )
        if len(candidates) != 1:
            self._terminal(
                "ROUTE_AMBIGUOUS", "More than one frozen topology matched",
                pattern_ids=sorted(candidates),
            )
        return candidates[0]

    def _draft_subplans(
        self, pattern_id: str, analysis: Mapping[str, Any], scope: Mapping[str, Any]
    ) -> list[dict[str, Any]]:
        concerns = self._ordered_concerns(analysis)
        structured = [row for row in concerns if row["kind"] != "narrative"]
        narratives = [row for row in concerns if row["kind"] == "narrative"]
        entities = self._ordered_entities(analysis, scope)
        periods = list(analysis["output_period_axis"])

        if pattern_id == "single_node":
            return [self._node_draft(analysis, scope, concerns[0], entities[0] if entities else None, None)]
        if pattern_id == "parallel_concerns":
            return [
                self._node_draft(analysis, scope, concern, entities[0] if entities else None, None)
                for concern in structured
            ]
        if pattern_id in {"entity_list", "entity_compare"}:
            return [
                self._node_draft(analysis, scope, concern, entity, None)
                for entity in entities for concern in structured
            ]
        if pattern_id in {"period_list", "period_compare"}:
            entity = entities[0] if entities else None
            return [
                self._node_draft(analysis, scope, concern, entity, period)
                for period in periods for concern in structured
            ]
        if pattern_id == "single_document_bundle":
            entity = entities[0] if entities else None
            rows = [self._node_draft(analysis, scope, concern, entity, None) for concern in structured]
            rows.append(self._evidence_draft(analysis, scope, narratives[0], entity, None, rows))
            return rows
        if pattern_id == "entity_section_bundle":
            rows: list[dict[str, Any]] = []
            for entity in entities:
                section = [self._node_draft(analysis, scope, concern, entity, None) for concern in structured]
                rows.extend(section)
                rows.append(self._evidence_draft(analysis, scope, narratives[0], entity, None, section))
            return rows
        if pattern_id == "period_section_bundle":
            rows = []
            entity = entities[0] if entities else None
            for period in periods:
                section = [self._node_draft(analysis, scope, concern, entity, period) for concern in structured]
                rows.extend(section)
                rows.append(self._evidence_draft(analysis, scope, narratives[0], entity, period, section))
            return rows
        raise AssertionError(f"unknown frozen pattern: {pattern_id}")

    def _node_draft(
        self,
        analysis: Mapping[str, Any],
        scope: Mapping[str, Any],
        concern: Mapping[str, Any],
        entity: Mapping[str, Any] | None,
        period: int | None,
    ) -> dict[str, Any]:
        kind = concern["kind"]
        intents = set(analysis["intents"])
        if "rank" in intents or "aggregate" in intents:
            backend, operation = "sql", "corpus_query" if scope["scope_kind"] == "corpus" else "document_query"
        elif kind == "formula":
            backend, operation = "formula", "formula_compute"
        elif kind == "metadata":
            backend, operation = "fact", "metadata_lookup"
        elif kind == "narrative":
            return self._evidence_draft(analysis, scope, concern, entity, period, [])
        else:
            backend, operation = "fact", "fact_lookup"

        failure = self._entity_failure(entity)
        # Period structured nodes in a narrative bundle are report-scoped and
        # bind that section's report.  Pure period fact/formula fan-out may use
        # comparative metric years from one source annual report instead.
        is_narrative_period = period is not None and "narrative" in intents
        document = self._select_document(
            scope,
            entity,
            period=period if is_narrative_period else None,
            narrative=is_narrative_period,
        )
        if failure is None and kind == "metadata" and period is not None:
            failure = self._failure(
                "COMPOSITION_UNSUPPORTED",
                "Metadata cannot be compared or repeated on a metric-year axis",
                period=period,
            )
        if failure is None and kind == "metric" and concern.get("canonical_metric") is None:
            failure = self._metric_failure(analysis, concern)
        if (
            failure is None
            and kind in {"metric", "formula"}
            and concern.get("unit_source") == "ambiguous"
        ):
            failure = self._failure(
                "UNIT_AMBIGUOUS",
                "The requested unit is missing or incompatible with the catalog measure",
                concern_key=concern["concern_id"],
            )
        if (
            failure is None
            and kind == "metric"
            and concern.get("canonical_metric") == "股本"
            and not self._normalized_unit(analysis, concern)
        ):
            failure = self._failure(
                "UNIT_AMBIGUOUS",
                "股本 has both currency and share-count selected facts; an explicit unit is required",
                concern_key=concern["concern_id"],
            )
        if failure is None and kind == "formula" and concern.get("formula_id") is None:
            failure = self._failure(
                "METRIC_UNRECOGNIZED", "Formula concern was not recognized",
                concern_key=concern["concern_id"],
            )
        metric_year = (
            self._formula_target_year(analysis, scope, concern, period)
            if kind == "formula"
            else self._metric_year(scope, period)
        )
        if metric_year is None and document is not None:
            metric_year = int(document["report_year"])
        if failure is None and backend in {"fact", "formula"} and kind != "metadata" and entity is not None:
            if document is None and len(entity.get("document_set", [])) > 1:
                failure = self._failure(
                    "RESOLVER_AMBIGUOUS", "A unique source report is required",
                    entity_key=entity["entity_key"],
                )
        if failure is None and kind in {"metric", "formula"} and metric_year is None:
            failure = self._failure(
                "METRIC_YEAR_MISSING", "A unique metric year is required",
                concern_key=concern["concern_id"],
            )
        documents = [document["document_id"]] if document is not None else []
        declared_kind = "single_document" if document is not None else scope["scope_kind"]
        if failure is not None:
            payload = None
            state = "blocked"
        else:
            state = "ready"
            payload = self._structured_payload(
                backend, operation, analysis, scope, concern, entity, document, metric_year
            )
        return {
            "planning_state": state,
            "backend": backend,
            "operation": operation,
            "entity_key": entity["entity_key"] if entity is not None else None,
            "period_key": str(period) if period is not None else None,
            "concern_key": concern["concern_id"],
            "scope_ref": scope["scope_plan_id"],
            "depends_on_subplan_ids": [],
            "authorization_source_subplan_ids": [],
            "required": True,
            "declared_scope": {"scope_kind": declared_kind, "document_ids": documents},
            "payload": payload,
            "planning_failure": failure,
        }

    def _evidence_draft(
        self,
        analysis: Mapping[str, Any],
        scope: Mapping[str, Any],
        concern: Mapping[str, Any],
        entity: Mapping[str, Any] | None,
        period: int | None,
        section_drafts: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        failure = self._entity_failure(entity)
        document = self._select_document(scope, entity, period=period, narrative=True)
        if failure is None and "table" in analysis["evidence_kinds"] and "narrative" not in analysis["evidence_kinds"]:
            failure = self._failure(
                "COMPOSITION_UNSUPPORTED",
                "Table-only evidence is excluded from the Phase 7 text index",
                concern_key=concern["concern_id"],
            )
        if failure is None and document is None:
            code = "RESOLVER_AMBIGUOUS" if entity and len(entity.get("document_set", [])) > 1 else "RESOLVER_MISSING"
            failure = self._failure(
                code, "Evidence requires exactly one resolved annual report",
                entity_key=entity["entity_key"] if entity else None,
                period=period,
            )
        if failure is None:
            state = "ready"
            payload: dict[str, Any] | None = {
                "document_id": document["document_id"],
                "question": analysis["normalized_question"],
                "top_k": self.evidence_top_k,
            }
            documents = [document["document_id"]]
        else:
            state, payload, documents = "blocked", None, []
        return {
            "planning_state": state,
            "backend": "evidence",
            "operation": "document_retrieval",
            "entity_key": entity["entity_key"] if entity is not None else None,
            "period_key": str(period) if period is not None else None,
            "concern_key": concern["concern_id"],
            "scope_ref": scope["scope_plan_id"],
            # Draft references are replaced with canonical IDs after ordinals are known.
            "_dependency_drafts": list(section_drafts),
            "required": True,
            "declared_scope": {"scope_kind": "single_document", "document_ids": documents},
            "payload": payload,
            "planning_failure": failure,
        }

    def _finalize_subplans(
        self, drafts: Sequence[dict[str, Any]], pattern: Mapping[str, Any]
    ) -> list[dict[str, Any]]:
        ids: list[str] = []
        for ordinal, draft in enumerate(drafts):
            ids.append(make_subplan_id(
                self.registry_semantic_sha256,
                pattern["pattern_id"],
                pattern["version"],
                ordinal,
                draft["entity_key"],
                draft["period_key"],
                draft["concern_key"],
            ))
        draft_identity = {id(draft): ids[index] for index, draft in enumerate(drafts)}
        finalized: list[dict[str, Any]] = []
        for ordinal, draft in enumerate(drafts):
            row = dict(draft)
            dependency_drafts = row.pop("_dependency_drafts", [])
            dependencies = [draft_identity[id(dep)] for dep in dependency_drafts]
            authorization = [
                draft_identity[id(dep)] for dep in dependency_drafts
                if dep["backend"] in {"fact", "formula", "sql"}
            ]
            row["subplan_id"] = ids[ordinal]
            row["ordinal"] = ordinal
            row["depends_on_subplan_ids"] = dependencies
            row["authorization_source_subplan_ids"] = authorization
            finalized.append(row)
        return finalized

    def _structured_payload(
        self,
        backend: str,
        operation: str,
        analysis: Mapping[str, Any],
        scope: Mapping[str, Any],
        concern: Mapping[str, Any],
        entity: Mapping[str, Any] | None,
        document: Mapping[str, Any] | None,
        metric_year: int | None,
    ) -> dict[str, Any]:
        identity = entity.get("identity") if entity else None
        normalized_unit = self._normalized_unit(analysis, concern)
        common = {
            "entity_key": entity["entity_key"] if entity else None,
            "document_id": document["document_id"] if document else None,
            "stock_code": identity["stock_code"] if identity else None,
        }
        if backend == "fact" and operation == "metadata_lookup":
            return {**common, "metadata_field": concern["metadata_field"]}
        if backend == "fact":
            return {
                **common,
                "report_year": document["report_year"] if document else None,
                "metric_year": metric_year,
                "canonical_metric": concern["canonical_metric"],
                "normalized_unit": normalized_unit,
            }
        if backend == "formula":
            return {
                **common,
                "report_year": document["report_year"] if document else None,
                "target_year": metric_year,
                "formula_id": concern["formula_id"],
                "normalized_unit": normalized_unit,
            }
        query_kind = "rank" if "rank" in analysis["intents"] else "aggregate"
        payload = {
            "query_spec_id": f"phase8.{query_kind}.v1",
            "query_kind": query_kind,
            "canonical_metric": concern.get("canonical_metric"),
            "metric_year": metric_year,
            "normalized_unit": normalized_unit,
            "document_ids": self._scope_document_ids(scope),
            "entity_keys": [row["entity_key"] for row in self._ordered_entities(analysis, scope)],
            # Corpus year is a report-set boundary, while metric_year selects
            # the value column inside each report.  Both are required: without
            # this filter a 2020 report's 2019 comparison column could leak
            # into a query explicitly scoped to 2019 annual reports.
            "report_years": sorted(scope["report_year_constraints"]),
            "scope_company_count": (
                scope["corpus_scope"]["company_count"]
                if scope["scope_kind"] == "corpus" else len(scope["entity_resolutions"])
            ),
            "scope_document_count": (
                scope["corpus_scope"]["document_count"]
                if scope["scope_kind"] == "corpus" else len(self._scope_document_ids(scope))
            ),
        }
        question = analysis["normalized_question"]
        if query_kind == "rank":
            limit_match = re.search(r"(?:Top\s*|前\s*)([1-5])", question, flags=re.IGNORECASE)
            payload.update({
                "order_direction": "asc" if re.search(r"最低|最少|最小", question) else "desc",
                "limit": int(limit_match.group(1)) if limit_match else 1,
            })
        else:
            if re.search(r"平均值|均值|平均", question):
                operator = "average"
            elif re.search(r"合计|总和|汇总", question):
                operator = "sum"
            elif re.search(r"公司数量|多少家公司|几家公司", question):
                operator = "count"
            else:
                operator = None
            payload["aggregate_operator"] = operator
        return payload

    @staticmethod
    def _ordered_concerns(analysis: Mapping[str, Any]) -> list[dict[str, Any]]:
        return [
            row for _, row in sorted(
                enumerate(analysis["concerns"]), key=lambda pair: (pair[1]["mention_ordinal"], pair[0])
            )
        ]

    def _ordered_entities(
        self, analysis: Mapping[str, Any], scope: Mapping[str, Any]
    ) -> list[dict[str, Any]]:
        rows = [
            row for _, row in sorted(
                enumerate(scope["entity_resolutions"]),
                key=lambda pair: (pair[1]["mention_ordinal"], pair[0]),
            )
        ]
        axis = analysis["output_entity_axis"]
        if not axis:
            return rows
        by_key: dict[Any, dict[str, Any]] = {}
        for row in rows:
            by_key[row["entity_key"]] = row
            by_key[row["mention"]] = row
            if row.get("identity"):
                for name in ("stock_code", "stock_name", "company_full"):
                    by_key[row["identity"].get(name)] = row
        selected: list[dict[str, Any]] = []
        for ordinal, key in enumerate(axis):
            row = by_key.get(key)
            if row is None and ordinal < len(rows):
                row = rows[ordinal]
            if row is None:
                row = {
                    "entity_key": str(key), "mention": str(key), "mention_ordinal": ordinal,
                    "status": "missing", "identity": None, "document_set": [], "findings": [],
                }
            selected.append(row)
        return selected

    @staticmethod
    def _entity_failure(entity: Mapping[str, Any] | None) -> dict[str, Any] | None:
        if entity is None or entity.get("status") == "unique":
            return None
        code = {
            "missing": "RESOLVER_MISSING",
            "ambiguous": "RESOLVER_AMBIGUOUS",
            "conflict": "RESOLVER_CONFLICT",
        }[entity["status"]]
        return TopologyCompositionPlanner._failure(
            code, "Entity identity was not uniquely resolved", entity_key=entity["entity_key"]
        )

    @staticmethod
    def _metric_failure(
        analysis: Mapping[str, Any], concern: Mapping[str, Any]
    ) -> dict[str, Any]:
        matching = [
            row for row in analysis["metric_mentions"] if row.get("raw_text") == concern.get("raw_text")
        ]
        code = "METRIC_AMBIGUOUS" if any(row.get("status") == "ambiguous" for row in matching) else "METRIC_UNRECOGNIZED"
        return TopologyCompositionPlanner._failure(
            code, "Metric concern was not uniquely recognized", concern_key=concern["concern_id"]
        )

    def _select_document(
        self,
        scope: Mapping[str, Any],
        entity: Mapping[str, Any] | None,
        *,
        period: int | None,
        narrative: bool = False,
    ) -> dict[str, Any] | None:
        if entity is None:
            return None
        documents = list(entity.get("document_set", []))
        if not documents:
            return None
        wanted: list[int] = []
        if narrative and period is not None:
            wanted = [period]
        elif scope["report_year_constraints"]:
            wanted = list(scope["report_year_constraints"])
        matches = [row for row in documents if not wanted or row["report_year"] in wanted]
        return matches[0] if len(matches) == 1 else None

    @staticmethod
    def _metric_year(scope: Mapping[str, Any], period: int | None) -> int | None:
        if period is not None:
            return period
        years = scope["metric_years"]
        if len(years) == 1:
            return years[0]
        report_years = scope["report_year_constraints"]
        if len(report_years) == 1:
            return report_years[0]
        return None

    @staticmethod
    def _formula_target_year(
        analysis: Mapping[str, Any],
        scope: Mapping[str, Any],
        concern: Mapping[str, Any],
        period: int | None,
    ) -> int | None:
        if period is not None:
            return period
        target_years: list[int] = []
        for mention in analysis["formula_mentions"]:
            if not isinstance(mention, Mapping) or mention.get("formula_id") != concern.get("formula_id"):
                continue
            raw_targets = mention.get("target_years", [])
            if isinstance(raw_targets, list):
                target_years.extend(
                    int(year) for year in raw_targets
                    if isinstance(year, int) and not isinstance(year, bool)
                )
        target_years = list(dict.fromkeys(target_years))
        if len(target_years) == 1:
            return target_years[0]
        return TopologyCompositionPlanner._metric_year(scope, None)

    @staticmethod
    def _normalized_unit(
        analysis: Mapping[str, Any], concern: Mapping[str, Any]
    ) -> str | None:
        direct = concern.get("normalized_unit")
        if isinstance(direct, str) and direct:
            return direct
        if concern.get("kind") == "formula":
            for mention in analysis["formula_mentions"]:
                if (
                    isinstance(mention, Mapping)
                    and mention.get("formula_id") == concern.get("formula_id")
                    and isinstance(mention.get("normalized_unit"), str)
                    and mention["normalized_unit"]
                ):
                    return mention["normalized_unit"]
        return None

    @staticmethod
    def _scope_document_ids(scope: Mapping[str, Any]) -> list[str]:
        return sorted({
            doc["document_id"]
            for entity in scope["entity_resolutions"] for doc in entity["document_set"]
        })

    @staticmethod
    def _failure(code: str, message: str, **details: Any) -> dict[str, Any]:
        return {"failure_code": code, "message": message, "details": details}

    @staticmethod
    def _company_count(analysis: Mapping[str, Any], scope: Mapping[str, Any]) -> int:
        return max(len(analysis["output_entity_axis"]), len(scope["entity_resolutions"]))

    @staticmethod
    def _section_axis(pattern_id: str) -> str:
        if pattern_id in {"entity_list", "entity_compare", "entity_section_bundle"}:
            return "entity"
        if pattern_id in {"period_list", "period_compare", "period_section_bundle"}:
            return "period"
        if pattern_id == "parallel_concerns":
            return "concern"
        return "none"

    @staticmethod
    def _output_kind(analysis: Mapping[str, Any], pattern_id: str) -> str:
        intents = set(analysis["intents"])
        if "narrative" in intents and len(intents) > 1:
            return "sectioned_bundle" if pattern_id.endswith("section_bundle") else "bundle"
        if "narrative" in intents:
            return "sectioned_narrative" if pattern_id.endswith("section_bundle") else "narrative"
        if "compare" in intents:
            return "comparison"
        if "rank" in intents:
            return "ranked"
        if "aggregate" in intents:
            return "aggregate"
        if pattern_id in {"entity_list", "period_list", "parallel_concerns"}:
            return "list"
        return "scalar"

    def _composition_plan_id(
        self,
        analysis: Mapping[str, Any],
        scope: Mapping[str, Any],
        pattern_id: str,
        subplans: Sequence[Mapping[str, Any]],
    ) -> str:
        seed = [
            self.planner_version,
            self.registry_semantic_sha256,
            analysis["request_id"],
            analysis["question_sha256"],
            scope["scope_plan_id"],
            pattern_id,
            [row["subplan_id"] for row in subplans],
        ]
        return "cp_" + hashlib.sha256(canonical_json_bytes(seed)).hexdigest()[:20]

    @staticmethod
    def _terminal(code: str, message: str, **details: Any) -> None:
        raise CompositionPlanningError(code, message, details)


__all__ = ["CompositionPlanningError", "TopologyCompositionPlanner"]
