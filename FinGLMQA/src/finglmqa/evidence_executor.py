"""Document-scoped evidence execution for Phase 8.

The executor is deliberately narrower than a general RAG answer generator.  A
composition plan has already frozen exactly one document.  This module only
retrieves inside that document, builds extractive draft claims, authorizes every
financial number, and constructs citations whose scope is identical to the
SubPlan scope.

Provider or generator output is untrusted boundary data.  A company/document
scope violation therefore fails closed with ``PROVENANCE_VALIDATION_FAILED``;
an unsupported or unauthorized numeric rendering only removes the affected
claim.  Chunks alone are never a usable evidence result.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
import hashlib
import re
import unicodedata
from pathlib import PurePosixPath
from typing import Any, Mapping, Sequence

from .contracts import (
    SCHEMA_SUBPLAN_RESULT,
    semantic_sha256,
    validate_numeric_authorization_set,
    validate_subplan,
    validate_subplan_result,
)
from .errors import status_for_blocked_plan
from .metric_catalog import MetricCatalog
from .ports import EvidenceProviderPort, GeneratorPort


EVIDENCE_EXECUTOR_VERSION = "phase8-evidence-executor-v3"
EVIDENCE_PROVIDER_RESULT_SCHEMA = "finglmqa.phase8.evidence_provider_result.v1"
CLAIM_BUILDER_SCHEMA = "finglmqa.phase8.claim_builder_request.v1"

_STOCK_CODE_RE = re.compile(r"^[0-9]{6}$")
_SCORE_RE = re.compile(r"^-?[0-9]+\.[0-9]{8}$")
_NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9])"
    r"[+-]?(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]+)?"
    r"(?:[%％]|(?:万|亿|千|百)?(?:亿元|万元|元|股|人|次|倍))?"
    r"(?![A-Za-z0-9])"
)
_YEAR_RE = re.compile(r"(?<![0-9])((?:19|20|21)[0-9]{2})(?=\s*年)")
_SENTENCE_RE = re.compile(r"[^。！？!?；;\n]+[。！？!?；;]?")
_CHECKBOX_PREFIX_RE = re.compile(
    r"^(?:[√✓✔☑□☐■●○]\s*(?:适用|不适用)\s*){1,2}",
)
_PURE_CHECKBOX_RE = re.compile(
    r"^(?:[√✓✔☑□☐■●○]\s*(?:适用|不适用)\s*)+$",
)
_NEGATIVE_CHECKBOX_RE = re.compile(
    r"^(?:(?:[□☐]\s*(?:本年度|本年)?\s*适用\s*[√✓✔☑]\s*(?:本年度|本年)?\s*不适用)|"
    r"(?:[√✓✔☑]\s*(?:本年度|本年)?\s*不适用\s*[□☐]\s*(?:本年度|本年)?\s*适用))$",
)
_STANDALONE_NEGATIVE_RE = re.compile(r"^(不适用|无|否)$")
_TOPIC_NEGATIVE_TEXT_RE = re.compile(
    r"(?:不适用|未发生|不存在|没有|并无|未涉及|未受到|未被|无需|无(?:重大|相关|其他|需|应))"
)
_UNIT_ONLY_RE = re.compile(
    r"^(?:单位|金额单位|币种)\s*[:：]?\s*(?:人民币)?\s*(?:元|万元|亿元|股|人|次|倍)?\s*$",
)
_DIRECTORY_LINE_RE = re.compile(
    r"^第[一二三四五六七八九十百0-9]+节\s*[^。！？!?]{0,40}(?:\.{2,}|…{2,}|\s+[0-9]{1,3})\s*$",
)
_NUMBERED_HEADING_RE = re.compile(
    r"^(?:(?:第[一二三四五六七八九十百0-9]+[章节])|(?:[一二三四五六七八九十0-9]+[、.)）]))"
    r"[^。！？!?；;]{0,32}(?:情况表?|说明|分析|概述|目录|事项|差异)$",
)
_DISCLAIMER_RES = (
    re.compile(r"(?:董事会|监事会).{0,30}保证.{0,20}(?:真实|准确|完整)"),
    re.compile(r"不存在虚假记载、?误导性陈述或重大遗漏"),
    re.compile(r"前瞻性陈述.{0,50}不构成.{0,20}(?:承诺|保证)"),
    re.compile(r"(?:敬请|请)投资者.{0,20}注意.{0,10}风险"),
    re.compile(r"(?:仅供参考|不构成投资建议|不承担.{0,12}责任)"),
)

# Question-side triggers and claim-side expressions are intentionally finite.
# They provide deterministic synonym coverage without turning the executor into
# an open-ended semantic model.
_RELEVANCE_CONCEPTS = (
    (("原因", "为何", "为什么", "影响因素"),
     ("由于", "受", "导致", "影响", "得益于", "主要系", "带动", "驱动", "优化", "推进")),
    (("营业收入", "营收", "收入", "业绩"),
     ("营业收入", "营收", "收入", "销售", "产品", "业务", "经营", "渠道", "市场", "订单", "销量", "价格")),
    (("风险", "不利因素", "挑战"),
     ("风险", "不确定", "波动", "压力", "挑战", "不利", "面临", "下降", "竞争", "争夺", "加剧", "冲击", "负面", "疫情")),
    (("研发", "研究开发", "技术创新", "专利"),
     ("研发", "研究", "开发", "技术", "创新", "专利", "科研")),
    (("重大资产", "股权出售", "资产出售", "股权转让"),
     ("重大资产", "资产", "股权", "出售", "转让", "过户")),
    (("主营业务", "业务模式", "经营模式"),
     ("主营", "业务", "经营", "产品", "服务", "生产", "销售")),
    (("客户", "供应商", "集中度"),
     ("客户", "供应商", "集中", "采购", "销售")),
    (("员工", "人员", "人才", "职工"),
     ("员工", "人员", "人才", "职工", "雇员", "队伍")),
    (("环境", "环保", "排污", "污染"),
     ("环境", "环保", "排放", "排污", "污染", "绿色", "节能")),
    (("未来", "战略", "规划", "发展计划"),
     ("未来", "战略", "规划", "计划", "发展", "目标")),
    (("审计", "会计师事务所", "审计意见"),
     ("审计", "会计师事务所", "意见", "保留意见", "聘任", "解聘")),
    (("分红", "利润分配", "股利"),
     ("分红", "利润分配", "股利", "现金红利", "派发")),
    (("诉讼", "仲裁"), ("诉讼", "仲裁")),
    (("退市", "暂停上市", "终止上市"), ("退市", "暂停上市", "终止上市")),
    (("处罚", "整改", "监管措施", "纪律处分"), ("处罚", "整改", "监管措施", "纪律处分")),
    (("破产", "重整"), ("破产", "重整")),
    (("重大合同", "合同履行"), ("重大合同", "合同履行")),
)
_NEGATIVE_HEADING_TOPICS = (
    (("诉讼", "仲裁"), ("诉讼", "仲裁")),
    (("退市", "暂停上市", "终止上市"), ("退市", "暂停上市", "终止上市")),
    (("处罚", "整改", "监管措施", "纪律处分"), ("处罚", "整改", "监管措施", "纪律处分")),
    (("破产", "重整"), ("破产", "重整")),
    (("重大合同", "合同履行"), ("重大合同", "合同履行")),
)
_QUESTION_GENERIC_PHRASES = (
    "股份有限公司", "有限责任公司", "年度报告", "年报", "报告期内", "本公司",
    "概述一下", "概述", "介绍一下", "介绍", "请问", "请说明", "说明一下",
    "针对", "根据", "分别", "相关", "主要", "具体", "情况", "是什么", "有哪些",
    "怎么样", "如何", "一下", "公司",
)
_QUESTION_TERM_STOP = {
    "股份", "有限", "报告", "年度", "年报", "公司", "本公", "情况", "概述", "介绍",
    "说明", "什么", "哪些", "如何", "一下", "针对", "根据", "分别", "相关", "主要",
    "具体", "报告期", "期内", "是否", "为何", "为什么", "原因",
}

_PROVIDER_FIELDS = {
    "schema_version",
    "status",
    "document_id",
    "company",
    "stock_code",
    "report_year",
    "retrieval_method",
    "provider_fingerprint",
    "chunks",
}
_CIRCLED_ENUMERATION_RE = re.compile(r"[\u2460-\u2473\u3251-\u325f\u32b1-\u32bf]")
_CHUNK_FIELDS = {
    "chunk_id",
    "document_chunk_ordinal",
    "score",
    "document_id",
    "company",
    "stock_code",
    "report_year",
    "section_path",
    "semantic_tags",
    "line_range",
    "source_markdown",
    "content",
}


class EvidenceBoundaryError(ValueError):
    """Invalid provider/generator data that cannot safely become evidence."""

    def __init__(self, message: str, *, provenance_failure: bool = False) -> None:
        super().__init__(message)
        self.provenance_failure = provenance_failure


class EvidenceExecutor:
    """Execute a ready, single-document evidence SubPlan.

    ``generator`` is optional.  Without it, the executor uses the deterministic
    extractive builder.  An injected generator still cannot widen scope,
    fabricate a citation, paraphrase beyond the cited chunk, or bypass numeric
    authorization.
    """

    def __init__(
        self,
        provider: EvidenceProviderPort,
        *,
        generator: GeneratorPort | None = None,
        metric_catalog: MetricCatalog | None = None,
    ) -> None:
        if not callable(getattr(provider, "retrieve", None)):
            raise TypeError("provider must implement EvidenceProviderPort.retrieve")
        if generator is not None and not callable(getattr(generator, "generate_claims", None)):
            raise TypeError("generator must implement GeneratorPort.generate_claims")
        self.provider = provider
        self.generator = generator
        self.metric_catalog = metric_catalog or MetricCatalog()

    def execute(
        self,
        subplan: Mapping[str, Any],
        numeric_authorization_set: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Return one contract-valid evidence ``SubPlanResult``."""

        try:
            validated_subplan = validate_subplan(dict(subplan))
        except Exception as exc:
            return self._failure(
                subplan,
                status="error",
                code="INVALID_REQUEST",
                message="EvidenceExecutor received an invalid SubPlan",
                details={"exception_type": type(exc).__name__},
            )

        if validated_subplan["backend"] != "evidence":
            return self._failure(
                validated_subplan,
                status="error",
                code="INVALID_REQUEST",
                message="EvidenceExecutor can execute only evidence SubPlans",
            )
        if validated_subplan["planning_state"] == "blocked":
            failure = validated_subplan["planning_failure"]
            return self._failure(
                validated_subplan,
                status=status_for_blocked_plan(failure["failure_code"]),
                code=failure["failure_code"],
                message=failure["message"],
                details=failure["details"],
                execution_state="not_executed",
                planning_state="blocked",
            )

        try:
            authorization_set = validate_numeric_authorization_set(dict(numeric_authorization_set))
        except Exception as exc:
            return self._failure(
                validated_subplan,
                status="blocked",
                code="PROVENANCE_VALIDATION_FAILED",
                message="NumericAuthorizationSet failed validation",
                details={"exception_type": type(exc).__name__},
            )

        request = {
            "document_id": validated_subplan["payload"]["document_id"],
            "question": validated_subplan["payload"]["question"],
            "top_k": validated_subplan["payload"]["top_k"],
        }
        try:
            raw_provider_result = self.provider.retrieve(request)
            provider_result = self._validate_provider_result(
                raw_provider_result,
                subplan=validated_subplan,
                request=request,
                authorizations=authorization_set["items"],
            )
        except EvidenceBoundaryError as exc:
            code = "PROVENANCE_VALIDATION_FAILED" if exc.provenance_failure else "EVIDENCE_UNAVAILABLE"
            return self._failure(
                validated_subplan,
                status="blocked" if exc.provenance_failure else "not_found",
                code=code,
                message=str(exc),
                trace={
                    "evidence_executor_version": EVIDENCE_EXECUTOR_VERSION,
                    "authorization_set_fingerprint": authorization_set["set_fingerprint"],
                    "provider_status": "rejected",
                },
            )
        except Exception as exc:
            return self._failure(
                validated_subplan,
                status="not_found",
                code="EVIDENCE_UNAVAILABLE",
                message="EvidenceProviderPort failed closed",
                details={"exception_type": type(exc).__name__},
                trace={
                    "evidence_executor_version": EVIDENCE_EXECUTOR_VERSION,
                    "authorization_set_fingerprint": authorization_set["set_fingerprint"],
                    "provider_status": "failed",
                },
            )

        chunks = provider_result["chunks"]
        if not chunks:
            return self._failure(
                validated_subplan,
                status="not_found",
                code="EVIDENCE_UNAVAILABLE",
                message="Document-scoped retrieval returned no chunks",
                trace=self._trace(provider_result, authorization_set, [], [], "none"),
            )

        generator_mode = (
            "injected" if self.generator is not None else "deterministic_question_aware_extractive"
        )
        try:
            drafts = self._draft_claims(validated_subplan, provider_result, authorization_set)
        except EvidenceBoundaryError as exc:
            code = "PROVENANCE_VALIDATION_FAILED" if exc.provenance_failure else "GENERATOR_UNAVAILABLE"
            return self._failure(
                validated_subplan,
                status="blocked",
                code=code,
                message=str(exc),
                trace=self._trace(provider_result, authorization_set, [], [], generator_mode),
            )
        except Exception as exc:
            return self._failure(
                validated_subplan,
                status="blocked",
                code="GENERATOR_UNAVAILABLE",
                message="GeneratorPort failed closed",
                details={"exception_type": type(exc).__name__},
                trace=self._trace(provider_result, authorization_set, [], [], generator_mode),
            )

        chunk_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
        claims: list[dict[str, Any]] = []
        citations_by_id: dict[str, dict[str, Any]] = {}
        rejected: list[dict[str, Any]] = []
        try:
            ordered_drafts = self._canonical_drafts(drafts, chunk_by_id)
            for draft_ordinal, draft in enumerate(ordered_drafts):
                decision = self._gate_claim(
                    draft,
                    subplan=validated_subplan,
                    provider_result=provider_result,
                    chunk_by_id=chunk_by_id,
                    authorizations=authorization_set["items"],
                )
                if decision["rejection_reason"] is not None:
                    rejected.append({
                        "draft_ordinal": draft_ordinal,
                        "reason": decision["rejection_reason"],
                        # Rejected renderings are untrusted input.  Preserve
                        # the deterministic decision, not the token values.
                        "financial_token_count": len(decision["financial_tokens"]),
                    })
                    continue
                citation_ids: list[str] = []
                for chunk_id in draft["evidence_chunk_ids"]:
                    citation = self._citation(validated_subplan, chunk_by_id[chunk_id])
                    citations_by_id[citation["citation_id"]] = citation
                    citation_ids.append(citation["citation_id"])
                authorization_ids = decision["authorization_ids"]
                claim_id = "claim_" + semantic_sha256({
                    "subplan_id": validated_subplan["subplan_id"],
                    "text": draft["text"],
                    "evidence_chunk_ids": draft["evidence_chunk_ids"],
                    "numeric_authorization_ids": authorization_ids,
                })[:20]
                claims.append({
                    "claim_id": claim_id,
                    "text": draft["text"],
                    "entity_key": validated_subplan["entity_key"],
                    "company": provider_result["company"],
                    "document_id": provider_result["document_id"],
                    "citation_ids": citation_ids,
                    "numeric_authorization_ids": authorization_ids,
                })
        except EvidenceBoundaryError as exc:
            return self._failure(
                validated_subplan,
                status="blocked",
                code="PROVENANCE_VALIDATION_FAILED" if exc.provenance_failure else "EVIDENCE_UNAVAILABLE",
                message=str(exc),
                trace=self._trace(provider_result, authorization_set, [], rejected, generator_mode),
            )

        if not claims:
            return self._failure(
                validated_subplan,
                status="not_found",
                code="EVIDENCE_UNAVAILABLE",
                message="No retrieved draft claim passed the relevance, numeric, and citation gates",
                details={"rejected_claim_count": len(rejected)},
                trace=self._trace(provider_result, authorization_set, [], rejected, generator_mode),
            )

        citations = sorted(
            citations_by_id.values(),
            key=lambda row: (
                row["provenance"]["document_chunk_ordinal"],
                row["provenance"]["evidence_chunk_id"],
                row["citation_id"],
            ),
        )
        accepted_chunk_ids = {
            citation["provenance"]["evidence_chunk_id"] for citation in citations
        }
        safe_chunks = [
            self._safe_chunk_projection(chunk)
            for chunk in chunks
            if chunk["chunk_id"] in accepted_chunk_ids
        ]
        result = {
            "schema_version": SCHEMA_SUBPLAN_RESULT,
            "subplan_id": validated_subplan["subplan_id"],
            "backend": "evidence",
            "operation": validated_subplan["operation"],
            "planning_state": "ready",
            "execution_state": "executed",
            "status": "ok",
            "result": {
                "document_id": provider_result["document_id"],
                "company": provider_result["company"],
                "stock_code": provider_result["stock_code"],
                "report_year": provider_result["report_year"],
                "retrieval_method": provider_result["retrieval_method"],
                "provider_fingerprint": provider_result["provider_fingerprint"],
                # Raw retrieval content is an internal, untrusted boundary
                # value.  Official results expose only accepted citation
                # metadata and a content hash.
                "chunks": safe_chunks,
                "claim_count": len(claims),
            },
            "claims": claims,
            "citations": citations,
            "failure_code": None,
            "errors": [],
            "warnings": (
                [{
                    "warning_code": "EVIDENCE_CLAIMS_FILTERED",
                    "message": "One or more draft claims failed relevance, numeric, or citation gates.",
                    "rejected_claim_count": len(rejected),
                }]
                if rejected else []
            ),
            "missing_fact_requests": [],
            "trace": self._trace(provider_result, authorization_set, claims, rejected, generator_mode),
        }
        validate_subplan_result(result)
        return result

    @staticmethod
    def _safe_chunk_projection(chunk: Mapping[str, Any]) -> dict[str, Any]:
        """Return citation-safe metadata without retrieved prose or numbers."""

        return {
            "chunk_id": chunk["chunk_id"],
            "document_chunk_ordinal": chunk["document_chunk_ordinal"],
            "score": chunk["score"],
            "document_id": chunk["document_id"],
            "company": chunk["company"],
            "stock_code": chunk["stock_code"],
            "report_year": chunk["report_year"],
            "section_path": list(chunk["section_path"]),
            "semantic_tags": list(chunk["semantic_tags"]),
            "line_range": list(chunk["line_range"]),
            "source_markdown": chunk["source_markdown"],
            "content_sha256": hashlib.sha256(chunk["content"].encode("utf-8")).hexdigest(),
        }

    def _validate_provider_result(
        self,
        value: Any,
        *,
        subplan: Mapping[str, Any],
        request: Mapping[str, Any],
        authorizations: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        if not isinstance(value, dict) or set(value) != _PROVIDER_FIELDS:
            raise EvidenceBoundaryError("EvidenceProviderPort returned an invalid result shape")
        if value["schema_version"] != EVIDENCE_PROVIDER_RESULT_SCHEMA or value["status"] != "ok":
            raise EvidenceBoundaryError("EvidenceProviderPort did not return a successful frozen response")
        expected_document = request["document_id"]
        if value["document_id"] != expected_document:
            raise EvidenceBoundaryError(
                "Provider response escaped the frozen document scope",
                provenance_failure=True,
            )
        for field in ("company", "stock_code", "retrieval_method", "provider_fingerprint"):
            if not isinstance(value[field], str) or not value[field].strip():
                raise EvidenceBoundaryError(f"Provider response has invalid {field}")
        if not _STOCK_CODE_RE.fullmatch(value["stock_code"]):
            raise EvidenceBoundaryError("Provider response has invalid stock_code")
        if not isinstance(value["report_year"], int) or isinstance(value["report_year"], bool):
            raise EvidenceBoundaryError("Provider response has invalid report_year")
        entity_key = subplan["entity_key"]
        if isinstance(entity_key, str) and _STOCK_CODE_RE.fullmatch(entity_key) and entity_key != value["stock_code"]:
            raise EvidenceBoundaryError(
                "Provider company identity does not match the evidence SubPlan",
                provenance_failure=True,
            )
        scoped_auth_companies = {
            item["company"]
            for item in authorizations
            if item["document_id"] == expected_document and item["entity_key"] == entity_key
        }
        if scoped_auth_companies and value["company"] not in scoped_auth_companies:
            raise EvidenceBoundaryError(
                "Provider company identity conflicts with structured authorization",
                provenance_failure=True,
            )

        chunks = value["chunks"]
        if not isinstance(chunks, list) or len(chunks) > request["top_k"]:
            raise EvidenceBoundaryError("Provider chunks violate the requested top_k")
        checked_chunks: list[dict[str, Any]] = []
        seen: set[str] = set()
        for raw_chunk in chunks:
            chunk = self._validate_chunk(raw_chunk)
            if chunk["chunk_id"] in seen:
                raise EvidenceBoundaryError("Provider returned duplicate chunk IDs")
            seen.add(chunk["chunk_id"])
            if (
                chunk["document_id"] != expected_document
                or chunk["company"] != value["company"]
                or chunk["stock_code"] != value["stock_code"]
                or chunk["report_year"] != value["report_year"]
            ):
                raise EvidenceBoundaryError(
                    "A provider chunk crossed the frozen company/document scope",
                    provenance_failure=True,
                )
            checked_chunks.append(chunk)
        expected_order = sorted(
            checked_chunks,
            key=lambda row: (-Decimal(row["score"]), row["document_chunk_ordinal"], row["chunk_id"]),
        )
        if checked_chunks != expected_order:
            raise EvidenceBoundaryError("Provider chunks are not in deterministic rank order")
        result = dict(value)
        result["chunks"] = checked_chunks
        return result

    @staticmethod
    def _validate_chunk(value: Any) -> dict[str, Any]:
        if not isinstance(value, dict) or set(value) != _CHUNK_FIELDS:
            raise EvidenceBoundaryError("Provider returned an invalid chunk shape")
        for field in ("chunk_id", "document_id", "company", "stock_code", "source_markdown", "content"):
            if not isinstance(value[field], str) or not value[field].strip():
                raise EvidenceBoundaryError(f"Provider chunk has invalid {field}")
        if not _STOCK_CODE_RE.fullmatch(value["stock_code"]):
            raise EvidenceBoundaryError("Provider chunk has invalid stock_code")
        if (
            not isinstance(value["document_chunk_ordinal"], int)
            or isinstance(value["document_chunk_ordinal"], bool)
            or value["document_chunk_ordinal"] < 1
        ):
            raise EvidenceBoundaryError("Provider chunk has invalid document_chunk_ordinal")
        if not isinstance(value["score"], str) or not _SCORE_RE.fullmatch(value["score"]):
            raise EvidenceBoundaryError("Provider score must be an 8-decimal string")
        try:
            score = Decimal(value["score"])
        except InvalidOperation as exc:
            raise EvidenceBoundaryError("Provider score is not finite Decimal text") from exc
        if not score.is_finite():
            raise EvidenceBoundaryError("Provider score is not finite")
        if not isinstance(value["report_year"], int) or isinstance(value["report_year"], bool):
            raise EvidenceBoundaryError("Provider chunk has invalid report_year")
        for field in ("section_path", "semantic_tags"):
            if not isinstance(value[field], list) or not all(
                isinstance(item, str) and item.strip() for item in value[field]
            ):
                raise EvidenceBoundaryError(f"Provider chunk has invalid {field}")
        line_range = value["line_range"]
        if (
            not isinstance(line_range, list)
            or len(line_range) != 2
            or any(not isinstance(item, int) or isinstance(item, bool) or item < 1 for item in line_range)
            or line_range[1] < line_range[0]
        ):
            raise EvidenceBoundaryError("Provider chunk has invalid line_range")
        source = PurePosixPath(value["source_markdown"])
        if source.is_absolute() or ".." in source.parts or "\\" in value["source_markdown"]:
            raise EvidenceBoundaryError("Provider chunk source_markdown must be workspace-relative")
        return dict(value)

    def _draft_claims(
        self,
        subplan: Mapping[str, Any],
        provider_result: Mapping[str, Any],
        authorization_set: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        if self.generator is None:
            return self._extractive_claims(subplan, provider_result, authorization_set)
        request = {
            "schema_version": CLAIM_BUILDER_SCHEMA,
            "subplan_id": subplan["subplan_id"],
            "question": subplan["payload"]["question"],
            "entity_key": subplan["entity_key"],
            "document_id": provider_result["document_id"],
            "company": provider_result["company"],
            "stock_code": provider_result["stock_code"],
            "report_year": provider_result["report_year"],
            "chunks": provider_result["chunks"],
            "numeric_authorization_set": authorization_set,
        }
        response = self.generator.generate_claims(request)
        if not isinstance(response, dict) or set(response) != {"claims"} or not isinstance(response["claims"], list):
            raise EvidenceBoundaryError("GeneratorPort returned an invalid claim response")
        return [dict(row) if isinstance(row, Mapping) else row for row in response["claims"]]

    def _extractive_claims(
        self,
        subplan: Mapping[str, Any],
        provider_result: Mapping[str, Any],
        authorization_set: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        """Select at most one usable, question-relevant sentence per chunk.

        Candidate ranking is lexical and deterministic.  The final choice is
        also previewed through the same fail-closed gate used after canonical
        ordering, allowing a later safe sentence to win when a higher-ranked
        sentence contains an unauthorized financial rendering.
        """

        drafts: list[dict[str, Any]] = []
        chunk_by_id = {chunk["chunk_id"]: chunk for chunk in provider_result["chunks"]}
        for chunk_ordinal, chunk in enumerate(provider_result["chunks"]):
            candidates: list[tuple[tuple[Any, ...], str]] = []
            for sentence_ordinal, match in enumerate(_SENTENCE_RE.finditer(chunk["content"])):
                candidate = self._clean_sentence(match.group(0))
                quality = self._claim_quality(
                    candidate,
                    question=subplan["payload"]["question"],
                    chunks=[chunk],
                    company=provider_result["company"],
                    stock_code=provider_result["stock_code"],
                )
                if quality["rejection_reason"] is not None:
                    continue
                rank = (
                    -quality["direct_score"],
                    -quality["concept_hits"],
                    -quality["section_score"],
                    sentence_ordinal,
                    self._normalize_prose(candidate),
                )
                candidates.append((rank, candidate))

            ranked = sorted(candidates, key=lambda row: row[0])
            first_relevant: dict[str, Any] | None = None
            selected: dict[str, Any] | None = None
            for _, candidate in ranked:
                draft = {
                    "text": candidate,
                    "evidence_chunk_ids": [chunk["chunk_id"]],
                    "entity_key": subplan["entity_key"],
                    "company": provider_result["company"],
                    "stock_code": provider_result["stock_code"],
                    "document_id": provider_result["document_id"],
                }
                first_relevant = first_relevant or draft
                decision = self._gate_claim(
                    draft,
                    subplan=subplan,
                    provider_result=provider_result,
                    chunk_by_id=chunk_by_id,
                    authorizations=authorization_set["items"],
                )
                if decision["rejection_reason"] is None:
                    selected = draft
                    break
            # Preserve one deterministic rejection in the official trace when
            # relevant prose exists but every candidate fails a safety gate.
            if selected is not None:
                drafts.append(selected)
            elif first_relevant is not None:
                drafts.append(first_relevant)
            elif chunk_ordinal == 0:
                negative = self._top_heading_negative_text(
                    question=subplan["payload"]["question"],
                    chunk=chunk,
                )
                if negative is not None:
                    drafts.append({
                        "text": negative,
                        "evidence_chunk_ids": [chunk["chunk_id"]],
                        "entity_key": subplan["entity_key"],
                        "company": provider_result["company"],
                        "stock_code": provider_result["stock_code"],
                        "document_id": provider_result["document_id"],
                    })
        return drafts

    @staticmethod
    def _clean_sentence(value: str) -> str:
        candidate = re.sub(r"^\s{0,3}(?:#{1,6}|[-*+])\s*", "", value).strip()
        candidate = _CHECKBOX_PREFIX_RE.sub("", candidate).strip()
        return candidate

    @staticmethod
    def _clean_heading(value: str) -> str:
        normalized = EvidenceExecutor._normalize_prose(value)
        return re.sub(
            r"^(?:(?:\([一二三四五六七八九十0-9]+\))|"
            r"(?:[一二三四五六七八九十0-9]+[、.)）])|"
            r"(?:第[一二三四五六七八九十0-9]+[章节]))\s*",
            "",
            normalized,
        ).strip()

    @staticmethod
    def _top_heading_negative_text(
        *, question: str, chunk: Mapping[str, Any]
    ) -> str | None:
        """Select an original negative marker under the top chunk's own heading.

        A checkbox in a later/nested section cannot qualify: the first
        non-empty content line must itself be the explicit negative marker.
        The heading authorizes relevance only and is never copied into claim
        text.
        """

        heading = EvidenceExecutor._clean_heading(chunk["section_path"][-1])
        if not EvidenceExecutor._negative_topic_heading_matches(question, heading):
            return None
        first_line = next(
            (
                EvidenceExecutor._normalize_prose(line)
                for line in chunk["content"].splitlines()
                if EvidenceExecutor._normalize_prose(line)
            ),
            "",
        )
        compact = re.sub(r"\s+", "", first_line)
        if _NEGATIVE_CHECKBOX_RE.fullmatch(compact):
            return first_line
        else:
            standalone = _STANDALONE_NEGATIVE_RE.fullmatch(compact)
            if standalone is None:
                return None
            return first_line

    @staticmethod
    def _negative_topic_heading_matches(question: str, heading: str) -> bool:
        normalized_question = EvidenceExecutor._normalize_prose(question)
        normalized_heading = EvidenceExecutor._normalize_prose(heading)
        return any(
            any(term in normalized_question for term in question_terms)
            and any(term in normalized_heading for term in heading_terms)
            for question_terms, heading_terms in _NEGATIVE_HEADING_TOPICS
        )

    @staticmethod
    def _is_scoped_topic_negative(text: str, question: str) -> bool:
        normalized_question = EvidenceExecutor._normalize_prose(question)
        question_has_topic = any(
            any(term in normalized_question for term in question_terms)
            for question_terms, _ in _NEGATIVE_HEADING_TOPICS
        )
        normalized_text = EvidenceExecutor._normalize_prose(text)
        return question_has_topic and (
            _TOPIC_NEGATIVE_TEXT_RE.search(normalized_text) is not None
            or _STANDALONE_NEGATIVE_RE.fullmatch(normalized_text) is not None
        )

    @staticmethod
    def _is_verified_top_heading_negative(
        draft: Mapping[str, Any],
        *,
        subplan: Mapping[str, Any],
        provider_result: Mapping[str, Any],
    ) -> bool:
        if len(draft["evidence_chunk_ids"]) != 1 or not provider_result["chunks"]:
            return False
        top_chunk = provider_result["chunks"][0]
        if draft["evidence_chunk_ids"][0] != top_chunk["chunk_id"]:
            return False
        expected = EvidenceExecutor._top_heading_negative_text(
            question=subplan["payload"]["question"], chunk=top_chunk
        )
        return expected is not None and draft["text"] == expected

    @staticmethod
    def _question_terms(question: str, *, company: str, stock_code: str) -> set[str]:
        normalized = unicodedata.normalize("NFKC", question).lower()
        normalized = normalized.replace(company.lower(), "").replace(stock_code, "")
        normalized = re.sub(r"(?<![0-9])(?:19|20|21)[0-9]{2}\s*年?", "", normalized)
        for phrase in _QUESTION_GENERIC_PHRASES:
            normalized = normalized.replace(phrase, "")
        terms: set[str] = set()
        for run in re.findall(r"[\u3400-\u9fffA-Za-z]+", normalized):
            for width in range(2, min(4, len(run)) + 1):
                for start in range(len(run) - width + 1):
                    term = run[start:start + width]
                    if term not in _QUESTION_TERM_STOP:
                        terms.add(term)
        return terms

    @staticmethod
    def _candidate_is_insufficient(text: str, chunks: Sequence[Mapping[str, Any]]) -> bool:
        normalized = EvidenceExecutor._normalize_prose(text)
        compact = re.sub(r"\s+", "", normalized)
        if not compact or len(re.findall(r"[\u3400-\u9fffA-Za-z0-9]", compact)) < 6:
            return True
        if _PURE_CHECKBOX_RE.fullmatch(compact) or _UNIT_ONLY_RE.fullmatch(compact):
            return True
        if any("目录" in heading for chunk in chunks for heading in chunk["section_path"][-1:]):
            return True
        if _DIRECTORY_LINE_RE.fullmatch(normalized) or _NUMBERED_HEADING_RE.fullmatch(normalized):
            return True
        if re.search(r"(?:\.{4,}|…{3,})", normalized):
            return True
        return any(pattern.search(normalized) for pattern in _DISCLAIMER_RES)

    @staticmethod
    def _claim_quality(
        text: str,
        *,
        question: str,
        chunks: Sequence[Mapping[str, Any]],
        company: str,
        stock_code: str,
    ) -> dict[str, Any]:
        if EvidenceExecutor._candidate_is_insufficient(text, chunks):
            return {
                "rejection_reason": "claim_insufficient_or_boilerplate",
                "direct_score": 0,
                "concept_hits": 0,
                "section_score": 0,
            }
        normalized_text = EvidenceExecutor._normalize_prose(text).lower()
        normalized_question = EvidenceExecutor._normalize_prose(question).lower()
        section_text = " ".join(
            heading for chunk in chunks for heading in chunk["section_path"]
        ).lower()
        terms = EvidenceExecutor._question_terms(
            normalized_question, company=company, stock_code=stock_code
        )
        direct_score = sum(len(term) for term in terms if term in normalized_text)
        section_score = sum(len(term) for term in terms if term in section_text)
        concept_hits = 0
        for question_triggers, claim_terms in _RELEVANCE_CONCEPTS:
            if any(term in normalized_question for term in question_triggers) and any(
                term in normalized_text for term in claim_terms
            ):
                concept_hits += 1
        return {
            "rejection_reason": (
                None if direct_score > 0 or concept_hits > 0 else "claim_not_question_relevant"
            ),
            "direct_score": direct_score,
            "concept_hits": concept_hits,
            "section_score": section_score,
        }

    @staticmethod
    def _canonical_drafts(
        drafts: Any, chunk_by_id: Mapping[str, Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        if not isinstance(drafts, list):
            raise EvidenceBoundaryError("Claim builder output must be an array")
        normalized: list[dict[str, Any]] = []
        allowed_fields = {
            "text", "evidence_chunk_ids", "entity_key", "company", "stock_code", "document_id",
            "canonical_metric", "metric_year", "formula_id",
        }
        for draft in drafts:
            if not isinstance(draft, dict) or set(draft) - allowed_fields:
                raise EvidenceBoundaryError("Claim builder emitted an invalid draft claim")
            text = draft.get("text")
            chunk_ids = draft.get("evidence_chunk_ids")
            if not isinstance(text, str) or not text.strip():
                continue
            if (
                not isinstance(chunk_ids, list)
                or not chunk_ids
                or any(not isinstance(item, str) or not item for item in chunk_ids)
                or len(chunk_ids) != len(set(chunk_ids))
            ):
                raise EvidenceBoundaryError(
                    "A draft claim has no valid evidence citation",
                    provenance_failure=True,
                )
            unknown = sorted(set(chunk_ids) - set(chunk_by_id))
            if unknown:
                raise EvidenceBoundaryError(
                    "A draft claim cited a chunk outside the frozen retrieval result",
                    provenance_failure=True,
                )
            normalized.append({**draft, "text": text.strip(), "evidence_chunk_ids": list(chunk_ids)})
        normalized.sort(
            key=lambda row: (
                min(chunk_by_id[item]["document_chunk_ordinal"] for item in row["evidence_chunk_ids"]),
                tuple(row["evidence_chunk_ids"]),
                row["text"],
            )
        )
        deduplicated: list[dict[str, Any]] = []
        seen: set[tuple[str, tuple[str, ...]]] = set()
        for row in normalized:
            key = (row["text"], tuple(row["evidence_chunk_ids"]))
            if key not in seen:
                seen.add(key)
                deduplicated.append(row)
        return deduplicated

    def _gate_claim(
        self,
        draft: Mapping[str, Any],
        *,
        subplan: Mapping[str, Any],
        provider_result: Mapping[str, Any],
        chunk_by_id: Mapping[str, Mapping[str, Any]],
        authorizations: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        expected_scope = {
            "entity_key": subplan["entity_key"],
            "company": provider_result["company"],
            "stock_code": provider_result["stock_code"],
            "document_id": provider_result["document_id"],
        }
        for field, expected in expected_scope.items():
            if field in draft and draft[field] is not None and draft[field] != expected:
                raise EvidenceBoundaryError(
                    f"A draft claim crossed the frozen {field} scope",
                    provenance_failure=True,
                )

        normalized_claim = self._normalize_prose(draft["text"])
        extractively_supported = any(
            normalized_claim in self._normalize_prose(chunk_by_id[chunk_id]["content"])
            for chunk_id in draft["evidence_chunk_ids"]
        )
        if not extractively_supported:
            return {
                "rejection_reason": "claim_not_extractively_supported",
                "financial_tokens": [],
                "authorization_ids": [],
            }

        verified_heading_negative = self._is_verified_top_heading_negative(
            draft, subplan=subplan, provider_result=provider_result
        )

        if self._is_scoped_topic_negative(
            draft["text"], subplan["payload"]["question"]
        ):
            top_chunk = provider_result["chunks"][0]
            negative_scope_valid = (
                len(draft["evidence_chunk_ids"]) == 1
                and draft["evidence_chunk_ids"][0] == top_chunk["chunk_id"]
                and self._negative_topic_heading_matches(
                    subplan["payload"]["question"], top_chunk["section_path"][-1]
                )
            )
            if not negative_scope_valid:
                return {
                    "rejection_reason": "negative_claim_heading_scope_not_verified",
                    "financial_tokens": [],
                    "authorization_ids": [],
                }

        quality = self._claim_quality(
            draft["text"],
            question=subplan["payload"]["question"],
            chunks=[chunk_by_id[chunk_id] for chunk_id in draft["evidence_chunk_ids"]],
            company=provider_result["company"],
            stock_code=provider_result["stock_code"],
        )
        if quality["rejection_reason"] is not None and not verified_heading_negative:
            return {
                "rejection_reason": quality["rejection_reason"],
                "financial_tokens": [],
                "authorization_ids": [],
            }

        financial_tokens = self._financial_tokens(draft["text"], provider_result["stock_code"])
        claim_measure = self._claim_measure_identity(draft) if financial_tokens else None
        if financial_tokens and claim_measure is None:
            return {
                "rejection_reason": "financial_claim_measure_or_year_not_unique",
                "financial_tokens": financial_tokens,
                "authorization_ids": [],
            }
        authorization_ids: list[str] = []
        for token in financial_tokens:
            matches = self._matching_authorizations(
                token,
                claim_measure=claim_measure,
                subplan=subplan,
                provider_result=provider_result,
                authorizations=authorizations,
            )
            if len(matches) != 1:
                return {
                    "rejection_reason": "financial_number_not_uniquely_authorized",
                    "financial_tokens": financial_tokens,
                    "authorization_ids": [],
                }
            authorization_ids.append(matches[0]["authorization_id"])
        return {
            "rejection_reason": None,
            "financial_tokens": financial_tokens,
            "authorization_ids": sorted(set(authorization_ids)),
        }

    def _claim_measure_identity(self, draft: Mapping[str, Any]) -> dict[str, Any] | None:
        """Infer one metric/formula and one year from the supported claim text.

        Generator annotations are untrusted hints.  They may narrow nothing and
        must agree with the catalog identity inferred from the extractive text.
        Claims containing financial numbers but no unique identity fail closed.
        """

        text = draft["text"]
        formula_mentions = self.metric_catalog.find_formula_mentions(text)
        formula_ids = sorted({row["formula"].formula_id for row in formula_mentions})
        formula_spans = [tuple(row["span"]) for row in formula_mentions]
        metric_mentions = self.metric_catalog.find_metric_mentions(text, formula_spans)
        metric_ids = sorted({
            row["candidates"][0]
            for row in metric_mentions
            if row["status"] == "unique" and len(row["candidates"]) == 1
        })
        if any(row["status"] != "unique" for row in metric_mentions):
            return None
        years = sorted({int(year) for year in _YEAR_RE.findall(text)})
        if len(years) != 1:
            return None
        if len(formula_ids) == 1 and not metric_ids:
            identity = {"kind": "formula", "formula_id": formula_ids[0], "metric_year": years[0]}
            if draft.get("canonical_metric") is not None:
                return None
            if draft.get("formula_id") is not None and draft["formula_id"] != formula_ids[0]:
                return None
        elif not formula_ids and len(metric_ids) == 1:
            identity = {"kind": "canonical_metric", "canonical_metric": metric_ids[0], "metric_year": years[0]}
            if draft.get("formula_id") is not None:
                return None
            if draft.get("canonical_metric") is not None and draft["canonical_metric"] != metric_ids[0]:
                return None
        else:
            return None
        if draft.get("metric_year") is not None and draft["metric_year"] != years[0]:
            return None
        return identity

    @staticmethod
    def _matching_authorizations(
        token: str,
        *,
        claim_measure: Mapping[str, Any],
        subplan: Mapping[str, Any],
        provider_result: Mapping[str, Any],
        authorizations: Sequence[Mapping[str, Any]],
    ) -> list[Mapping[str, Any]]:
        token_key = EvidenceExecutor._normalize_rendering(token)
        matches: list[Mapping[str, Any]] = []
        for item in authorizations:
            if (
                item["entity_key"] != subplan["entity_key"]
                or item["company"] != provider_result["company"]
                or item["document_id"] != provider_result["document_id"]
            ):
                continue
            if token_key not in {EvidenceExecutor._normalize_rendering(row) for row in item["allowed_renderings"]}:
                continue
            measure = item["measure"]
            if measure["kind"] == "canonical_fact":
                if (
                    claim_measure["kind"] != "canonical_metric"
                    or claim_measure["canonical_metric"] != measure["canonical_metric"]
                    or claim_measure["metric_year"] != measure["metric_year"]
                ):
                    continue
            elif measure["kind"] == "formula_result":
                if (
                    claim_measure["kind"] != "formula"
                    or claim_measure["formula_id"] != measure["formula_id"]
                    or claim_measure["metric_year"] != measure["target_year"]
                ):
                    continue
            else:
                if (
                    claim_measure["kind"] != "canonical_metric"
                    or claim_measure["canonical_metric"] != measure["canonical_metric"]
                    or claim_measure["metric_year"] != measure["metric_year"]
                ):
                    continue
            matches.append(item)
        return matches

    @staticmethod
    def _financial_tokens(text: str, stock_code: str) -> list[str]:
        result: list[str] = []
        # Annual reports frequently enumerate reasons with ①…㊿.  NFKC turns
        # those glyphs into ordinary decimal digits (and can turn ``③2018``
        # into ``32018``), which must not become an unauthorized financial
        # value.  Remove only the dedicated enumeration code points before
        # applying the otherwise useful compatibility normalization.
        normalized = unicodedata.normalize(
            "NFKC", _CIRCLED_ENUMERATION_RE.sub("", text)
        )
        for match in _NUMBER_RE.finditer(normalized):
            token = match.group(0)
            digits = token.lstrip("+-").replace(",", "")
            if EvidenceExecutor._is_nonfinancial_number(
                normalized, match.start(), match.end(), token, digits, stock_code
            ):
                continue
            result.append(token)
        return result

    @staticmethod
    def _is_nonfinancial_number(
        text: str,
        start: int,
        end: int,
        token: str,
        digits: str,
        stock_code: str,
    ) -> bool:
        plain = re.sub(r"[^0-9]", "", digits)
        prefix = text[max(0, start - 5):start]
        suffix = text[end:end + 5]
        if plain == stock_code and token.lstrip("+-") == stock_code:
            return True
        integer_part = digits.split(".", 1)[0]
        if integer_part.isdigit() and len(integer_part) == 4:
            year = int(integer_part)
            if 1900 <= year <= 2199 and re.match(r"\s*年", suffix):
                return True
        if re.search(r"第\s*$", prefix) and re.match(r"\s*(?:章|节|条|项|款|名|位|期|届|次)", suffix):
            return True
        # Chinese annual-report prose commonly uses inline list ordinals such
        # as ``风险：1、...`` and full-width parenthesized ordinals such as
        # ``（1）采购模式``.  They are structural markers, not financial
        # values.  Keep this deliberately narrow: only short integers without
        # signs/decimals/commas qualify, and the surrounding punctuation must
        # form an enumeration marker.  Currency, ratios and years therefore
        # continue through the strict numeric-authorization gate.
        is_short_unsigned_integer = (
            token == plain and plain.isdigit() and 1 <= len(plain) <= 2
        )
        if is_short_unsigned_integer and re.match(r"\s*、", suffix):
            return True
        if (
            is_short_unsigned_integer
            and re.search(r"[（(]\s*$", prefix)
            and re.match(r"\s*[）)]", suffix)
        ):
            return True
        if (start == 0 or text[start - 1] == "\n") and re.match(r"\s*[、.)）]", suffix):
            return True
        if re.match(r"\s*(?:月|日|号)", suffix):
            return True
        if re.search(r"(?:序号|章节|附注|排名)\s*$", prefix):
            return True
        return False

    @staticmethod
    def _citation(subplan: Mapping[str, Any], chunk: Mapping[str, Any]) -> dict[str, Any]:
        citation_id = "cite_evidence_" + semantic_sha256({
            "subplan_id": subplan["subplan_id"],
            "evidence_chunk_id": chunk["chunk_id"],
        })[:20]
        return {
            "citation_id": citation_id,
            "citation_kind": "evidence",
            "subplan_id": subplan["subplan_id"],
            "entity_key": subplan["entity_key"],
            "document_id": chunk["document_id"],
            "source_citation_ids": [],
            "provenance": {
                "evidence_chunk_id": chunk["chunk_id"],
                "document_chunk_ordinal": chunk["document_chunk_ordinal"],
                "score": chunk["score"],
                "section_path": chunk["section_path"],
                "semantic_tags": chunk["semantic_tags"],
                "line_range": chunk["line_range"],
                "source_markdown": chunk["source_markdown"],
                "content_sha256": hashlib.sha256(chunk["content"].encode("utf-8")).hexdigest(),
            },
        }

    @staticmethod
    def _trace(
        provider_result: Mapping[str, Any],
        authorization_set: Mapping[str, Any],
        claims: Sequence[Mapping[str, Any]],
        rejected: Sequence[Mapping[str, Any]],
        generator_mode: str,
    ) -> dict[str, Any]:
        return {
            "evidence_executor_version": EVIDENCE_EXECUTOR_VERSION,
            "provider_fingerprint": provider_result["provider_fingerprint"],
            "retrieval_method": provider_result["retrieval_method"],
            "retrieved_chunk_ids": [row["chunk_id"] for row in provider_result["chunks"]],
            "authorization_set_fingerprint": authorization_set["set_fingerprint"],
            "generator_mode": generator_mode,
            "accepted_claim_ids": [row["claim_id"] for row in claims],
            "rejected_claims": [dict(row) for row in rejected],
        }

    @staticmethod
    def _normalize_prose(value: str) -> str:
        return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value)).strip()

    @staticmethod
    def _normalize_rendering(value: str) -> str:
        return re.sub(r"[\s,]", "", unicodedata.normalize("NFKC", value))

    @staticmethod
    def _base_result(
        subplan: Mapping[str, Any],
        *,
        planning_state: str = "ready",
        execution_state: str = "executed",
    ) -> dict[str, Any]:
        subplan_id = subplan.get("subplan_id") if isinstance(subplan, Mapping) else None
        operation = subplan.get("operation") if isinstance(subplan, Mapping) else None
        return {
            "schema_version": SCHEMA_SUBPLAN_RESULT,
            "subplan_id": str(subplan_id or "invalid_subplan"),
            "backend": "evidence",
            "operation": str(operation or "document_retrieval"),
            "planning_state": planning_state,
            "execution_state": execution_state,
            "status": "error",
            "result": None,
            "claims": [],
            "citations": [],
            "failure_code": None,
            "errors": [],
            "warnings": [],
            "missing_fact_requests": [],
            "trace": {"evidence_executor_version": EVIDENCE_EXECUTOR_VERSION},
        }

    def _failure(
        self,
        subplan: Mapping[str, Any],
        *,
        status: str,
        code: str,
        message: str,
        details: Mapping[str, Any] | None = None,
        trace: Mapping[str, Any] | None = None,
        execution_state: str = "executed",
        planning_state: str = "ready",
    ) -> dict[str, Any]:
        result = self._base_result(
            subplan,
            planning_state=planning_state,
            execution_state=execution_state,
        )
        result.update({
            "status": status,
            "failure_code": code,
            "errors": [{"failure_code": code, "message": message, "details": dict(details or {})}],
            "trace": dict(trace or {"evidence_executor_version": EVIDENCE_EXECUTOR_VERSION}),
        })
        validate_subplan_result(result)
        return result


__all__ = [
    "CLAIM_BUILDER_SCHEMA",
    "EVIDENCE_EXECUTOR_VERSION",
    "EVIDENCE_PROVIDER_RESULT_SCHEMA",
    "EvidenceBoundaryError",
    "EvidenceExecutor",
]
