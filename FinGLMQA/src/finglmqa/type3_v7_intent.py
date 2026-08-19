"""Conservative intent matching for Type 3 v7 title paths.

This module is deliberately independent of retrieval and answer generation.  It
contains no corpus identities, report years, benchmark examples, or mutable
state.  Callers can therefore use it as a deterministic gate before accepting
a title-path candidate.

The gate is intentionally narrow.  An unrecognised phrase, a phrase that names
more than one supported intent, or an ambiguous title path is not accepted.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
import unicodedata
from typing import Iterable, Sequence


class Type3Intent(str, Enum):
    """Supported, deliberately disjoint annual-report intents."""

    CORPORATE_RESTRUCTURING = "corporate_restructuring"
    BANKRUPTCY_REORGANIZATION = "bankruptcy_reorganization"
    SHAREHOLDER_RELATIONSHIP = "shareholder_relationship"
    RELATED_PARTY_TRANSACTION = "related_party_transaction"
    FIXED_ASSETS = "fixed_assets"
    ASSET_OR_EQUITY_DISPOSAL = "asset_or_equity_disposal"
    AUDIT_OPINION = "audit_opinion"
    AUDITOR_ENGAGEMENT = "auditor_engagement"


class DecisionStatus(str, Enum):
    """Outcome of classifying a question or title path."""

    RESOLVED = "resolved"
    AMBIGUOUS = "ambiguous"
    UNKNOWN = "unknown"


@dataclass(frozen=True, order=True)
class IntentHit:
    """A transparent reason why an intent was considered.

    ``part_index`` is the zero-based component in a title path.  Questions are
    treated as a one-component path and therefore use index zero as well.
    ``marker`` is a generic language marker, not copied benchmark metadata.
    """

    intent: Type3Intent
    part_index: int
    marker: str


@dataclass(frozen=True)
class IntentDecision:
    """Fail-closed classification result."""

    status: DecisionStatus
    intent: Type3Intent | None
    candidates: tuple[Type3Intent, ...]
    hits: tuple[IntentHit, ...]
    normalized_parts: tuple[str, ...]
    reason: str

    @property
    def resolved(self) -> bool:
        return self.status is DecisionStatus.RESOLVED


@dataclass(frozen=True)
class TitlePathMatch:
    """Result of gating a title path against a question."""

    accepted: bool
    question: IntentDecision
    title_path: IntentDecision
    reason: str


_INTENT_ORDER = tuple(Type3Intent)
_SPACE_RE = re.compile(r"\s+")
_NON_WORD_RE = re.compile(r"[\s\W_]+", flags=re.UNICODE)


def _normalize(value: str) -> str:
    return _SPACE_RE.sub(" ", unicodedata.normalize("NFKC", value)).strip()


def _compact(value: str) -> str:
    return _NON_WORD_RE.sub("", _normalize(value).lower())


def normalize_title_path(title_path: str | Sequence[str]) -> tuple[str, ...]:
    """Return a canonical title path without modifying the caller's value.

    A string is one title component; it is not split on punctuation because
    slashes and dashes can legitimately occur inside annual-report headings.
    Empty components are discarded.  Invalid component types are rejected
    instead of being silently stringified.
    """

    if isinstance(title_path, str):
        values: Iterable[str] = (title_path,)
    elif isinstance(title_path, Sequence):
        values = title_path
    else:
        raise TypeError("title_path must be a string or a sequence of strings")

    result: list[str] = []
    for value in values:
        if not isinstance(value, str):
            raise TypeError("every title_path component must be a string")
        normalized = _normalize(value)
        if normalized:
            result.append(normalized)
    return tuple(result)


def _contains_any(text: str, markers: Sequence[str]) -> tuple[str, ...]:
    return tuple(marker for marker in markers if _compact(marker) in text)


def _corporate_restructuring_hits(text: str) -> tuple[str, ...]:
    explicit = _contains_any(
        text,
        (
            "重大资产重组",
            "资产重组",
            "债务重组",
            "业务重组",
            "并购重组",
            "公司重组",
            "企业重组",
            "重组方案",
            "重组计划",
            "重组进展",
            "重组事项",
        ),
    )
    if explicit:
        return explicit
    # Bare ``重组`` is conventional corporate-restructuring wording.  Do not
    # apply that fallback to colloquial bankruptcy wording such as ``破产重组``.
    if "重组" in text and not any(
        marker in text for marker in ("破产", "司法", "法院", "预重整", "重整程序")
    ):
        return ("重组",)
    return ()


def _bankruptcy_reorganization_hits(text: str) -> tuple[str, ...]:
    return _contains_any(
        text,
        (
            "破产重整",
            "破产重组",
            "司法重整",
            "预重整",
            "重整程序",
            "重整计划",
            "重整投资人",
            "申请重整",
            "进入重整",
            "法院裁定重整",
        ),
    )


def _shareholder_relationship_hits(text: str) -> tuple[str, ...]:
    direct = _contains_any(
        text,
        (
            "股东关系",
            "股东关联关系",
            "股东之间关联关系",
            "股东之间的关联关系",
            "股东相互间关联关系",
            "股东相互间的关联关系",
            "控股股东与实际控制人关系",
            "控股股东与实际控制人之间的关系",
            "与控股股东之间的关系",
        ),
    )
    if direct:
        return direct
    if "关联关系" in text and any(
        marker in text for marker in ("股东", "实际控制人")
    ):
        return ("股东+关联关系",)
    return ()


def _related_party_transaction_hits(text: str) -> tuple[str, ...]:
    return _contains_any(
        text,
        (
            "关联交易",
            "关联方交易",
            "日常关联",
            "关联采购",
            "关联销售",
            "关联租赁",
            "关联担保",
            "关联方资金往来",
            "与关联方的资金往来",
        ),
    )


def _fixed_asset_hits(text: str) -> tuple[str, ...]:
    return ("固定资产",) if "固定资产" in text else ()


def _asset_or_equity_disposal_hits(text: str) -> tuple[str, ...]:
    hits = list(
        _contains_any(
            text,
            (
                "出售资产",
                "出售重大资产",
                "资产出售",
                "处置资产",
                "资产处置",
                "转让资产",
                "资产转让",
                "出售股权",
                "股权出售",
                "处置股权",
                "股权处置",
                "转让股权",
                "股权转让",
                "购买或出售重大资产",
                "购买和出售重大资产",
                "资产和股权出售",
                "资产及股权出售",
            ),
        )
    )
    if "股权" in text and "交易" in text:
        hits.append("股权+交易")
    return tuple(dict.fromkeys(hits))


def _audit_opinion_hits(text: str) -> tuple[str, ...]:
    hits = list(
        _contains_any(
            text,
            (
                "审计意见",
                "审计结论",
                "审计报告意见",
                "审计报告类型",
                "审计报告的类型",
                "无保留意见",
                "保留意见",
                "否定意见",
                "无法表示意见",
                "非标准审计意见",
                "标准审计意见",
            ),
        )
    )
    # ``审计报告`` is a useful title-path container for an explicit
    # question-side audit-opinion intent.  It is still kept distinct from
    # headings about appointing the firm that performs the audit.
    if text.endswith("审计报告") or "财务报表审计报告" in text:
        hits.append("审计报告")
    return tuple(dict.fromkeys(hits))


def _auditor_engagement_hits(text: str) -> tuple[str, ...]:
    hits = list(
        _contains_any(
            text,
            (
                "聘任会计师事务所",
                "聘请会计师事务所",
                "聘用会计师事务所",
                "续聘会计师事务所",
                "选聘会计师事务所",
                "改聘会计师事务所",
                "解聘会计师事务所",
                "更换会计师事务所",
                "变更会计师事务所",
                "会计师事务所聘任",
                "会计师事务所聘用",
                "会计师事务所变更",
                "审计机构聘任",
                "聘任审计机构",
                "审计机构变更",
            ),
        )
    )
    if "会计师事务所" in text and any(
        action in text
        for action in ("聘任", "聘请", "聘用", "续聘", "选聘", "改聘", "解聘", "更换", "变更")
    ):
        hits.append("会计师事务所+聘任动作")
    return tuple(dict.fromkeys(hits))


_SCANNERS = (
    (Type3Intent.CORPORATE_RESTRUCTURING, _corporate_restructuring_hits),
    (Type3Intent.BANKRUPTCY_REORGANIZATION, _bankruptcy_reorganization_hits),
    (Type3Intent.SHAREHOLDER_RELATIONSHIP, _shareholder_relationship_hits),
    (Type3Intent.RELATED_PARTY_TRANSACTION, _related_party_transaction_hits),
    (Type3Intent.FIXED_ASSETS, _fixed_asset_hits),
    (Type3Intent.ASSET_OR_EQUITY_DISPOSAL, _asset_or_equity_disposal_hits),
    (Type3Intent.AUDIT_OPINION, _audit_opinion_hits),
    (Type3Intent.AUDITOR_ENGAGEMENT, _auditor_engagement_hits),
)


def _classify_parts(parts: tuple[str, ...]) -> IntentDecision:
    hits: list[IntentHit] = []
    for part_index, part in enumerate(parts):
        compact = _compact(part)
        if not compact:
            continue
        for intent, scanner in _SCANNERS:
            hits.extend(
                IntentHit(intent=intent, part_index=part_index, marker=marker)
                for marker in scanner(compact)
            )

    # A scanner can report both a specific phrase and its contained generic
    # phrase.  De-duplicate without losing the documented deterministic order.
    unique_hits = tuple(dict.fromkeys(hits))
    observed = {hit.intent for hit in unique_hits}
    candidates = tuple(intent for intent in _INTENT_ORDER if intent in observed)
    if len(candidates) == 1:
        return IntentDecision(
            status=DecisionStatus.RESOLVED,
            intent=candidates[0],
            candidates=candidates,
            hits=unique_hits,
            normalized_parts=parts,
            reason="one_supported_intent",
        )
    if candidates:
        return IntentDecision(
            status=DecisionStatus.AMBIGUOUS,
            intent=None,
            candidates=candidates,
            hits=unique_hits,
            normalized_parts=parts,
            reason="multiple_supported_intents",
        )
    return IntentDecision(
        status=DecisionStatus.UNKNOWN,
        intent=None,
        candidates=(),
        hits=(),
        normalized_parts=parts,
        reason="no_supported_intent",
    )


def classify_intent(text: str) -> IntentDecision:
    """Classify one question-like string using only generic language rules."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    normalized = _normalize(text)
    return _classify_parts((normalized,) if normalized else ())


def classify_question_intent(question: str) -> IntentDecision:
    """Named question-side entry point for integration readability."""

    return classify_intent(question)


def classify_title_path(title_path: str | Sequence[str]) -> IntentDecision:
    """Classify all components of a hierarchical title path together.

    Signals from different components are deliberately not allowed to override
    each other.  If a parent and child point at different supported intents,
    the whole path is ambiguous and therefore fails closed.
    """

    return _classify_parts(normalize_title_path(title_path))


def match_title_path(
    question: str,
    title_path: str | Sequence[str],
) -> TitlePathMatch:
    """Accept ``title_path`` only when both sides resolve to the same intent."""

    question_decision = classify_question_intent(question)
    path_decision = classify_title_path(title_path)
    if not question_decision.resolved:
        return TitlePathMatch(
            accepted=False,
            question=question_decision,
            title_path=path_decision,
            reason=f"question_{question_decision.status.value}",
        )
    if not path_decision.resolved:
        return TitlePathMatch(
            accepted=False,
            question=question_decision,
            title_path=path_decision,
            reason=f"title_path_{path_decision.status.value}",
        )
    if question_decision.intent is not path_decision.intent:
        return TitlePathMatch(
            accepted=False,
            question=question_decision,
            title_path=path_decision,
            reason="intent_mismatch",
        )
    return TitlePathMatch(
        accepted=True,
        question=question_decision,
        title_path=path_decision,
        reason="intent_match",
    )


__all__ = [
    "DecisionStatus",
    "IntentDecision",
    "IntentHit",
    "TitlePathMatch",
    "Type3Intent",
    "classify_intent",
    "classify_question_intent",
    "classify_title_path",
    "match_title_path",
    "normalize_title_path",
]
