"""End-to-end static Phase 8 QA pipeline.

Execution order is deliberately fixed:

QuestionAnalysis -> ScopePlan -> CompositionPlan -> structured results ->
NumericAuthorizationSet -> evidence results -> Composer.

No executor may create a SubPlan, and pre-composition terminal decisions make
zero backend calls.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .analyzer import QuestionAnalyzer
from .authorization import build_numeric_authorization_set
from .composer import Composer
from .composition import CompositionPlanningError, TopologyCompositionPlanner
from .contracts import (
    SCHEMA_ANSWER,
    SCHEMA_NUMERIC_AUTHORIZATION_SET,
    SCHEMA_SUBPLAN_RESULT,
    ContractValidationError,
    semantic_sha256,
    validate_numeric_authorization_set,
    validate_qa_answer,
    validate_qa_request,
    validate_subplan_result,
)
from .errors import PRECOMPOSITION_STATUS_BY_CODE
from .evidence_executor import EvidenceExecutor
from .evidence_provider import A2RAGWarmWorkerTransport, DocumentScopedEvidenceProvider
from .formula_engine import FormulaExecutor
from .repositories import FactRepository, FallbackCandidateIndex, MetadataRepository
from .resolver import ScopeResolver
from .sql_engine import SQLExecutor
from .structured_execution import (
    FactSubPlanExecutor,
    SQLSubPlanExecutor,
    blocked_subplan_result,
    portable_semantic_payload,
)
from .trace import TelemetryRecorder, finalize_trace


PIPELINE_VERSION = "phase8-static-dag-v1"
ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class PipelineRun:
    answer: dict[str, Any]
    trace: dict[str, Any]
    telemetry: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "answer": deepcopy(self.answer),
            "trace": deepcopy(self.trace),
            "telemetry": deepcopy(self.telemetry),
        }


class Phase8Pipeline:
    """Run one question through the frozen static Phase 8 DAG."""

    def __init__(
        self,
        *,
        analyzer: QuestionAnalyzer | None = None,
        resolver: ScopeResolver | None = None,
        planner: TopologyCompositionPlanner | None = None,
        fact_repository: FactRepository | None = None,
        metadata_repository: MetadataRepository | None = None,
        fallback_candidates: FallbackCandidateIndex | None = None,
        evidence_executor: EvidenceExecutor | None = None,
        composer: Composer | None = None,
    ) -> None:
        self.analyzer = analyzer or QuestionAnalyzer()
        self.resolver = resolver or ScopeResolver()
        self.planner = planner or TopologyCompositionPlanner()
        self.fact_repository = fact_repository or FactRepository()
        self.metadata_repository = metadata_repository or MetadataRepository()
        self.fallback_candidates = fallback_candidates or FallbackCandidateIndex()
        self.fact_executor = FactSubPlanExecutor(
            self.fact_repository,
            self.metadata_repository,
            fallback_candidates=self.fallback_candidates,
        )
        self.formula_executor = FormulaExecutor(
            self.fact_repository,
            fallback_candidates=self.fallback_candidates,
        )
        self.sql_executor = SQLSubPlanExecutor(
            SQLExecutor(self.fact_repository), fallback_candidates=self.fallback_candidates
        )
        self.evidence_executor = evidence_executor
        self.composer = composer or Composer()

    def run(self, raw_request: Mapping[str, Any]) -> PipelineRun:
        request_id = (
            str(raw_request.get("request_id"))
            if isinstance(raw_request, Mapping) and isinstance(raw_request.get("request_id"), str)
            and raw_request.get("request_id")
            else "invalid_request"
        )
        telemetry = TelemetryRecorder(request_id)
        analysis: dict[str, Any] | None = None
        scope: dict[str, Any] | None = None
        plan: dict[str, Any] | None = None
        authorization_set = _authorization_subset([], None)
        backend_calls = 0
        try:
            try:
                request = validate_qa_request(dict(raw_request))
            except (ContractValidationError, TypeError, ValueError) as exc:
                answer, trace = self._terminal(
                    request_id=request_id,
                    status="error",
                    failure_code="INVALID_REQUEST",
                    message="QA request violates the frozen Phase 8 contract",
                    details={"contract_error": str(exc)},
                    analysis=None,
                    scope=None,
                    authorization_set=authorization_set,
                )
                return PipelineRun(answer, trace, telemetry.finish(runtime=self._runtime(backend_calls)))

            request_id = request["request_id"]
            try:
                analysis = self.analyzer.analyze(request)
                scope = self.resolver.resolve(analysis, request)
                plan = self.planner.plan(analysis, scope)
            except CompositionPlanningError as exc:
                answer, trace = self._terminal(
                    request_id=request_id,
                    status=exc.status,
                    failure_code=exc.failure_code,
                    message=exc.message,
                    details=dict(exc.details),
                    analysis=analysis,
                    scope=scope,
                    authorization_set=authorization_set,
                )
                return PipelineRun(answer, trace, telemetry.finish(runtime=self._runtime(backend_calls)))
            except Exception as exc:
                answer, trace = self._terminal(
                    request_id=request_id,
                    status="error",
                    failure_code="INTERNAL_ERROR",
                    message="Analysis, scope resolution, or static planning failed",
                    details={"exception_type": type(exc).__name__},
                    analysis=analysis,
                    scope=scope,
                    authorization_set=authorization_set,
                )
                return PipelineRun(answer, trace, telemetry.finish(runtime=self._runtime(backend_calls)))

            results: list[dict[str, Any]] = []
            structured_results: list[dict[str, Any]] = []
            for subplan in plan["subplans"]:
                if subplan["backend"] == "evidence":
                    continue
                if subplan["planning_state"] == "blocked":
                    result = blocked_subplan_result(subplan)
                else:
                    backend_calls += 1
                    result = self._execute_structured(subplan)
                result = portable_semantic_payload(result)
                validate_subplan_result(result)
                results.append(result)
                structured_results.append(result)

            try:
                authorization_set = build_numeric_authorization_set(
                    plan["subplans"], structured_results
                )
            except Exception as exc:
                poisoned = self._execution_failure(
                    next((row for row in plan["subplans"] if row["backend"] != "evidence"), plan["subplans"][0]),
                    status="blocked",
                    failure_code="PROVENANCE_VALIDATION_FAILED",
                    message="Numeric authorization construction failed",
                    details={"exception_type": type(exc).__name__},
                )
                results = [poisoned]
                authorization_set = _authorization_subset([], None)

            for subplan in plan["subplans"]:
                if subplan["backend"] != "evidence":
                    continue
                if subplan["planning_state"] == "blocked":
                    result = blocked_subplan_result(subplan)
                elif self.evidence_executor is None:
                    result = self._execution_failure(
                        subplan,
                        status="error",
                        failure_code="GENERATOR_UNAVAILABLE",
                        message="Evidence execution is not configured for this pipeline instance",
                    )
                else:
                    backend_calls += 1
                    scoped_authorizations = _authorization_subset(
                        authorization_set["items"],
                        set(subplan["authorization_source_subplan_ids"]),
                    )
                    try:
                        result = self.evidence_executor.execute(subplan, scoped_authorizations)
                    except Exception as exc:
                        result = self._execution_failure(
                            subplan,
                            status="error",
                            failure_code="INTERNAL_ERROR",
                            message="Evidence executor failed",
                            details={"exception_type": type(exc).__name__},
                        )
                result = portable_semantic_payload(result)
                validate_subplan_result(result)
                results.append(result)

            results_by_id = {row["subplan_id"]: row for row in results}
            ordered_results = [
                results_by_id[row["subplan_id"]]
                for row in plan["subplans"] if row["subplan_id"] in results_by_id
            ]
            answer = self.composer.compose(request_id, plan, ordered_results)
            trace = finalize_trace({
                "request_id": request_id,
                "question_analysis": analysis,
                "scope_plan": scope,
                "composition_plan": plan,
                "composition_decision": {
                    "status": "planned",
                    "failure_code": None,
                    "backend_call_count": backend_calls,
                    "static_subplan_count": len(plan["subplans"]),
                },
                "subplan_traces": [
                    {"subplan_id": row["subplan_id"], "trace": row["trace"]}
                    for row in answer["subplans"]
                ],
                "numeric_authorization_set": authorization_set,
                "composer": answer["trace"],
                "artifact_fingerprints": self._artifact_fingerprints(plan),
            })
            answer["trace"] = {
                "composer": answer["trace"],
                "qa_trace_hash": trace["trace_hash"],
            }
            answer = portable_semantic_payload(answer)
            validate_qa_answer(answer)
            return PipelineRun(answer, trace, telemetry.finish(runtime=self._runtime(backend_calls)))
        except Exception as exc:  # final fail-closed envelope
            answer, trace = self._terminal(
                request_id=request_id,
                status="error",
                failure_code="INTERNAL_ERROR",
                message="Phase 8 pipeline failed closed",
                details={"exception_type": type(exc).__name__},
                analysis=analysis,
                scope=scope,
                authorization_set=_authorization_subset([], None),
            )
            return PipelineRun(answer, trace, telemetry.finish(runtime=self._runtime(backend_calls)))

    def answer(self, request: Mapping[str, Any]) -> dict[str, Any]:
        return self.run(request).answer

    def _execute_structured(self, subplan: Mapping[str, Any]) -> dict[str, Any]:
        if subplan["backend"] == "fact":
            return self.fact_executor.execute(subplan)
        if subplan["backend"] == "formula":
            return self.formula_executor.execute(subplan)
        if subplan["backend"] == "sql":
            return self.sql_executor.execute(subplan)
        return self._execution_failure(
            subplan, status="error", failure_code="INTERNAL_ERROR",
            message="Structured dispatcher received an unknown backend",
        )

    @staticmethod
    def _execution_failure(
        subplan: Mapping[str, Any],
        *,
        status: str,
        failure_code: str,
        message: str,
        details: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        result = {
            "schema_version": SCHEMA_SUBPLAN_RESULT,
            "subplan_id": subplan["subplan_id"],
            "backend": subplan["backend"],
            "operation": subplan["operation"],
            "planning_state": subplan["planning_state"],
            "execution_state": "executed" if subplan["planning_state"] == "ready" else "not_executed",
            "status": status,
            "result": None,
            "claims": [],
            "citations": [],
            "failure_code": failure_code,
            "errors": [{"failure_code": failure_code, "message": message, "details": dict(details or {})}],
            "warnings": [],
            "missing_fact_requests": [],
            "trace": {"pipeline_version": PIPELINE_VERSION, "failure_code": failure_code},
        }
        validate_subplan_result(result)
        return result

    def _terminal(
        self,
        *,
        request_id: str,
        status: str,
        failure_code: str,
        message: str,
        details: Mapping[str, Any],
        analysis: Mapping[str, Any] | None,
        scope: Mapping[str, Any] | None,
        authorization_set: Mapping[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        error = {"failure_code": failure_code, "message": message, "details": dict(details)}
        trace = finalize_trace({
            "request_id": request_id,
            "question_analysis": deepcopy(analysis),
            "scope_plan": deepcopy(scope),
            "composition_plan": None,
            "composition_decision": {
                "status": status,
                "failure_code": failure_code,
                "backend_call_count": 0,
                "static_subplan_count": 0,
            },
            "subplan_traces": [],
            "numeric_authorization_set": deepcopy(dict(authorization_set)),
            "composer": {"status": "not_invoked"},
            "artifact_fingerprints": self._artifact_fingerprints(None),
        })
        answer = {
            "schema_version": SCHEMA_ANSWER,
            "request_id": request_id,
            "status": status,
            "composition_pattern_id": None,
            "document_scope": deepcopy(scope),
            "subplans": [],
            "answer_text": "",
            "result": None,
            "citations": [],
            "trace": {"qa_trace_hash": trace["trace_hash"]},
            "errors": [error],
            "warnings": [],
            "missing_fact_requests": [],
        }
        validate_qa_answer(answer)
        return answer, trace

    def _artifact_fingerprints(self, plan: Mapping[str, Any] | None) -> dict[str, Any]:
        result = {
            "pipeline_version": PIPELINE_VERSION,
            "fact_repository": self.fact_repository.repository_fingerprint,
            "metadata_repository": self.metadata_repository.repository_fingerprint,
            "fallback_candidate_index": self.fallback_candidates.repository_fingerprint,
        }
        if plan is not None:
            result.update({
                "pattern_registry_semantic": plan["registry_semantic_sha256"],
                "pattern_registry_file": plan["registry_file_sha256"],
            })
        if self.evidence_executor is not None:
            provider = getattr(self.evidence_executor, "provider", None)
            fingerprint = getattr(provider, "provider_fingerprint", None)
            if isinstance(fingerprint, str):
                result["evidence_provider"] = fingerprint
        return result

    def _runtime(self, backend_calls: int) -> dict[str, Any]:
        return {
            "pipeline_version": PIPELINE_VERSION,
            "backend_call_count": backend_calls,
            "evidence_enabled": self.evidence_executor is not None,
        }


def build_default_pipeline(
    *,
    evidence_enabled: bool = True,
    device: str = "auto",
    model_cache: str | Path | None = None,
) -> tuple[Phase8Pipeline, A2RAGWarmWorkerTransport | None]:
    """Build the production pipeline and return its closable warm transport."""

    transport: A2RAGWarmWorkerTransport | None = None
    evidence_executor: EvidenceExecutor | None = None
    if evidence_enabled:
        transport = A2RAGWarmWorkerTransport(device=device, model_cache=model_cache)
        provider = DocumentScopedEvidenceProvider(transport)
        evidence_executor = EvidenceExecutor(provider)
    return Phase8Pipeline(
        evidence_executor=evidence_executor,
    ), transport


def _authorization_subset(
    items: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
    allowed_subplan_ids: set[str] | None,
) -> dict[str, Any]:
    selected = [
        deepcopy(dict(item)) for item in items
        if allowed_subplan_ids is None or item["source_subplan_id"] in allowed_subplan_ids
    ]
    selected.sort(key=lambda row: (row["source_subplan_id"], row["source_result_row"], row["authorization_id"]))
    result = {
        "schema_version": SCHEMA_NUMERIC_AUTHORIZATION_SET,
        "items": selected,
        "set_fingerprint": "pending",
    }
    result["set_fingerprint"] = semantic_sha256({
        "schema_version": result["schema_version"], "items": selected,
    })
    validate_numeric_authorization_set(result)
    return result


__all__ = ["PIPELINE_VERSION", "Phase8Pipeline", "PipelineRun", "build_default_pipeline"]
