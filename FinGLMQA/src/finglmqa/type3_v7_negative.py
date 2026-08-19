"""Pure, deterministic negative-evidence auditing for the Type 3 v7 experiment.

This module deliberately does not read files or questions.  Its caller supplies
the already-resolved source lines, heading paths, and semantic topic groups.
The audit answers one narrow question: does the parsed document contain
positive *event* evidence that makes negative fallback wording unsafe?

Keyword presence alone is not positive event evidence.  Annual reports repeat
section titles, laws, policies, and semantically adjacent terms in many places.
Conversely, a concrete event sentence or a positive applicability checkbox is
always a veto.  A negative applicability checkbox supports only a statement
about that exact leaf section; it never proves that the company had no event.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
import unicodedata
from typing import Any, Iterable, Sequence


NEGATIVE_AUDIT_SCHEMA = "finglmqa.experimental.negative_evidence_audit.v2"
SUPPORTED_NEGATIVE_TOPIC_GROUPS = (
    "bankruptcy",
    "delisting",
    "litigation",
    "penalty",
)

_SELECTED = frozenset("√✓✔☑■")
_UNSELECTED = frozenset("□☐○◯")
_CHECKBOX_OPTION_RE = re.compile(
    r"([□☐○◯■☑√✓✔])\s*((?:本年度|本年)?\s*(?:适用|不适用))"
)
_MARKDOWN_HEADING_RE = re.compile(r"^#{1,6}\s+(.+?)\s*$")

# These markers describe a rule, process, or hypothetical condition rather
# than an event.  A concrete event marker below takes precedence when both
# occur in the same sentence (for example, "根据处罚决定书，公司被罚款……").
_POLICY_MARKER_RE = re.compile(
    r"(?:制度|政策|办法|规则|规定|规范|指引|准则|流程|标准|原则|机制|"
    r"管理要求|适用条件|触发条件|认定标准|法律法规|上市规则|员工奖惩)"
)
_POLICY_VERB_RE = re.compile(
    r"(?:制定|建立|修订|完善|明确|规范|依据|按照|遵循|适用于|应当|可以|"
    r"不得|若|如发生|一旦|可能导致|视为)"
)
_NEGATIVE_STATEMENT_RE = re.compile(
    r"(?:未发生|不存在|没有|并无|未涉及|未受到|未被|无需|无相关|不涉及)"
)


def _normalize(value: Any) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(value))).strip()


def _compact(value: Any) -> str:
    return re.sub(r"[\s\W_]+", "", _normalize(value).lower(), flags=re.UNICODE)


def _stable_unique(values: Iterable[str]) -> tuple[str, ...]:
    """Return normalized unique values in a caller-order-independent order."""

    normalized = {_normalize(value) for value in values if _normalize(value)}
    return tuple(sorted(normalized, key=lambda value: (_compact(value), value)))


def _semantic_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class ConcernRule:
    """Semantic rule for one reusable annual-report concern.

    Patterns must describe concrete events, not the mere occurrence of an
    alias.  ``semantic_exclusions`` prevent related concepts from borrowing a
    shorter alias (for example, asset restructuring is not bankruptcy
    reorganization).
    """

    group_id: str
    aliases: tuple[str, ...]
    heading_aliases: tuple[str, ...]
    positive_patterns: tuple[tuple[str, str], ...]
    event_override_patterns: tuple[str, ...]
    semantic_exclusions: tuple[str, ...] = ()


_CONCERN_RULES: tuple[ConcernRule, ...] = (
    ConcernRule(
        group_id="bankruptcy",
        aliases=("破产重整", "破产清算", "破产", "重整"),
        heading_aliases=("破产重整", "破产清算", "破产事项", "破产情况"),
        positive_patterns=(
            (
                "bankruptcy_court_or_creditor_action",
                r"(?:法院|债权人).{0,28}(?:裁定|受理|申请|指定).{0,16}"
                r"(?:破产|重整|清算|管理人)",
            ),
            (
                "bankruptcy_proceeding",
                r"(?:破产|重整|破产清算).{0,16}(?:申请|程序|计划|管理人|裁定|"
                r"受理|执行|债权申报)",
            ),
            (
                "bankruptcy_proceeding",
                r"(?:进入|启动|实施|完成).{0,16}(?:破产|重整|破产清算)(?:程序|计划)?",
            ),
        ),
        event_override_patterns=(
            r"法院.{0,24}(?:裁定|受理)",
            r"(?:进入|启动|实施).{0,12}(?:破产|重整|破产清算)",
            r"(?:破产|重整|破产清算)(?:申请|程序|计划|管理人)",
        ),
        semantic_exclusions=(
            "重大资产重组", "资产重组", "债务重组", "并购重组", "重组上市",
        ),
    ),
    ConcernRule(
        group_id="delisting",
        aliases=("暂停上市和终止上市", "终止上市", "暂停上市", "退市"),
        heading_aliases=("暂停上市和终止上市", "终止上市", "暂停上市", "退市情况"),
        positive_patterns=(
            (
                "listing_status_action",
                r"(?:股票|证券|公司).{0,24}(?:被|将|已|决定|申请).{0,16}"
                r"(?:终止上市|暂停上市|退市)",
            ),
            (
                "listing_status_decision",
                r"(?:收到|作出|出具).{0,24}(?:终止上市|暂停上市|退市).{0,16}"
                r"(?:决定|通知|告知)",
            ),
            ("delisting_period", r"进入.{0,8}退市整理期|退市整理期.{0,12}(?:交易|届满)"),
        ),
        event_override_patterns=(
            r"(?:收到|作出|出具).{0,24}(?:终止上市|暂停上市|退市)",
            r"(?:被|将|已).{0,12}(?:终止上市|暂停上市|退市)",
            r"退市整理期",
        ),
        semantic_exclusions=("产品退市", "药品退市", "车型退市", "商品退市"),
    ),
    ConcernRule(
        group_id="penalty",
        aliases=("处罚及整改", "行政处罚", "监管措施", "纪律处分", "处罚", "整改"),
        heading_aliases=("处罚及整改", "行政处罚", "监管措施", "纪律处分", "受处罚"),
        positive_patterns=(
            (
                "penalty_received",
                r"(?:收到|接到|获悉).{0,28}(?:行政处罚|监管措施|纪律处分|"
                r"警示函|监管函).{0,16}(?:决定|通知|决定书|措施|函)?",
            ),
            (
                "penalty_imposed",
                r"(?:被|受到|遭到|给予).{0,20}(?:行政处罚|监管措施|纪律处分|"
                r"警告|罚款|没收)",
            ),
            (
                "regulator_action",
                r"(?:证监会|监管机构|证券交易所|交易所).{0,32}"
                r"(?:作出|采取|给予|出具).{0,16}(?:行政处罚|监管措施|纪律处分|"
                r"警告|罚款|警示函|监管函)",
            ),
            ("monetary_penalty", r"(?:处以|决定|合计)?罚款.{0,12}\d[\d,.]*\s*(?:元|万元)"),
            (
                "penalty_decision",
                r"(?:行政处罚|纪律处分|监管措施)(?:决定书|决定|通知书).{0,20}"
                r"(?:编号|文号|\d{4})",
            ),
        ),
        event_override_patterns=(
            r"(?:收到|接到).{0,24}(?:决定书|警示函|监管函)",
            r"(?:被|受到|遭到|给予).{0,16}(?:处罚|处分|警告|罚款|监管措施)",
            r"罚款.{0,10}\d",
            r"(?:证监会|监管机构|交易所).{0,24}(?:作出|采取|出具)",
        ),
        semantic_exclusions=("员工处罚", "供应商处罚", "违约处罚", "内部处罚"),
    ),
    ConcernRule(
        group_id="litigation",
        aliases=("重大诉讼仲裁", "重大诉讼", "重大仲裁", "诉讼", "仲裁"),
        heading_aliases=("重大诉讼仲裁", "重大诉讼", "重大仲裁", "诉讼仲裁事项"),
        positive_patterns=(
            (
                "litigation_filed_or_received",
                r"(?:提起|收到|涉及|卷入|受理|审理).{0,20}(?:诉讼|仲裁|传票|"
                r"起诉状|仲裁申请)",
            ),
            (
                "court_or_arbitration_action",
                r"(?:法院|仲裁委员会|仲裁院).{0,32}(?:立案|受理|判决|裁决|开庭)",
            ),
            (
                "litigation_case_detail",
                r"(?:诉讼|仲裁).{0,16}(?:案件|案号|金额|进展|判决|裁决|执行)",
            ),
        ),
        event_override_patterns=(
            r"(?:提起|收到|涉及|受理).{0,16}(?:诉讼|仲裁|传票|起诉状)",
            r"(?:法院|仲裁委员会|仲裁院).{0,24}(?:立案|受理|判决|裁决)",
            r"(?:诉讼|仲裁)(?:案件|案号|金额|进展)",
        ),
        semantic_exclusions=("诉讼制度", "仲裁规则", "争议解决条款"),
    ),
)

_RULE_BY_ID = {rule.group_id: rule for rule in _CONCERN_RULES}


@dataclass(frozen=True)
class TargetSection:
    section_id: str
    topic_groups: tuple[str, ...]
    heading: str
    heading_path: tuple[str, ...]
    line_range: tuple[int, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "section_id": self.section_id,
            "topic_groups": list(self.topic_groups),
            "heading": self.heading,
            "heading_path": list(self.heading_path),
            "line_range": list(self.line_range),
        }


@dataclass(frozen=True)
class NegativeEvidenceFinding:
    finding_id: str
    kind: str
    reason_code: str
    topic_groups: tuple[str, ...]
    matched_aliases: tuple[str, ...]
    heading_path: tuple[str, ...]
    line_range: tuple[int, int]
    text: str

    def sort_key(self) -> tuple[Any, ...]:
        kind_order = {
            "positive_event": 0,
            "checkbox_negative": 1,
            "incidental_mention": 2,
        }
        return (
            self.line_range,
            kind_order.get(self.kind, 99),
            self.reason_code,
            self.finding_id,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "finding_id": self.finding_id,
            "kind": self.kind,
            "reason_code": self.reason_code,
            "topic_groups": list(self.topic_groups),
            "matched_aliases": list(self.matched_aliases),
            "heading_path": list(self.heading_path),
            "line_range": list(self.line_range),
            "text": self.text,
        }


@dataclass(frozen=True)
class NegativeEvidenceAudit:
    audit_id: str
    schema_version: str
    document_id: str
    source_markdown: str
    document_sha256: str
    line_count: int
    channels_scanned: tuple[str, ...]
    topic_groups: tuple[str, ...]
    searched_aliases: tuple[str, ...]
    target_sections: tuple[TargetSection, ...]
    positive_events: tuple[NegativeEvidenceFinding, ...]
    scoped_checkbox_negatives: tuple[NegativeEvidenceFinding, ...]
    incidental_mentions: tuple[NegativeEvidenceFinding, ...]
    decision: str
    safe_claim_scope: str | None
    complete_scan: bool

    @property
    def allows_negative_wording(self) -> bool:
        return self.safe_claim_scope is not None and not self.positive_events

    def to_dict(self) -> dict[str, Any]:
        return {
            "audit_id": self.audit_id,
            "schema_version": self.schema_version,
            "document_id": self.document_id,
            "source_markdown": self.source_markdown,
            "document_sha256": self.document_sha256,
            "line_count": self.line_count,
            "channels_scanned": list(self.channels_scanned),
            "topic_groups": list(self.topic_groups),
            "searched_aliases": list(self.searched_aliases),
            "target_sections": [section.to_dict() for section in self.target_sections],
            "positive_events": [finding.to_dict() for finding in self.positive_events],
            "scoped_checkbox_negatives": [
                finding.to_dict() for finding in self.scoped_checkbox_negatives
            ],
            "incidental_mentions": [
                finding.to_dict() for finding in self.incidental_mentions
            ],
            "decision": self.decision,
            "safe_claim_scope": self.safe_claim_scope,
            "complete_scan": self.complete_scan,
        }


def _rule_heading_match(title: str, rule: ConcernRule) -> bool:
    compact = _compact(title)
    if not compact:
        return False
    excluded = any(_compact(value) in compact for value in rule.semantic_exclusions)
    # A hard bankruptcy anchor overrides a related restructuring phrase, as in
    # "破产重整与重大资产重组".  Other groups do not have this ambiguity.
    if excluded and not (rule.group_id == "bankruptcy" and "破产" in compact):
        return False
    return any(_compact(alias) in compact for alias in rule.heading_aliases)


def _semantic_collision(text: str, rule: ConcernRule) -> bool:
    compact = _compact(text)
    excluded = any(_compact(value) in compact for value in rule.semantic_exclusions)
    if rule.group_id == "bankruptcy" and "破产" in compact:
        return False
    return excluded


def _line_title(line: str, heading_path: Sequence[str]) -> str | None:
    if not heading_path:
        return None
    normalized = _normalize(line)
    match = _MARKDOWN_HEADING_RE.match(normalized)
    candidate = _normalize(match.group(1) if match else normalized)
    leaf = _normalize(heading_path[-1])
    return leaf if candidate == leaf else None


def _matched_aliases(text: str, aliases: Sequence[str]) -> tuple[str, ...]:
    compact = _compact(text)
    return tuple(alias for alias in aliases if _compact(alias) in compact)


def _checkbox_state(text: str) -> str | None:
    options = _CHECKBOX_OPTION_RE.findall(_normalize(text))
    if len(options) != 2:
        return None
    selected = [_compact(label) for symbol, label in options if symbol in _SELECTED]
    unselected = [_compact(label) for symbol, label in options if symbol in _UNSELECTED]
    if len(selected) != 1 or len(unselected) != 1:
        return None
    if selected[0] == "不适用" and unselected[0] == "适用":
        return "negative"
    if selected[0] == "适用" and unselected[0] == "不适用":
        return "positive"
    return None


def _finding(
    *,
    document_id: str,
    kind: str,
    reason_code: str,
    groups: Iterable[str],
    aliases: Iterable[str],
    heading_path: Sequence[str],
    line_number: int,
    text: str,
) -> NegativeEvidenceFinding:
    payload = {
        "document_id": document_id,
        "kind": kind,
        "reason_code": reason_code,
        "topic_groups": sorted(set(groups)),
        "matched_aliases": list(_stable_unique(aliases)),
        "heading_path": list(map(_normalize, heading_path)),
        "line_range": [line_number, line_number],
        "text": _normalize(text),
    }
    return NegativeEvidenceFinding(
        finding_id="v7-negative-finding-" + _semantic_hash(payload)[:24],
        kind=kind,
        reason_code=reason_code,
        topic_groups=tuple(payload["topic_groups"]),
        matched_aliases=tuple(payload["matched_aliases"]),
        heading_path=tuple(payload["heading_path"]),
        line_range=(line_number, line_number),
        text=payload["text"],
    )


def _compile_rules(topic_groups: Sequence[str]) -> tuple[ConcernRule, ...]:
    return tuple(_RULE_BY_ID[group_id] for group_id in topic_groups if group_id in _RULE_BY_ID)


def _target_sections(
    *,
    document_id: str,
    lines: Sequence[str],
    heading_paths: Sequence[tuple[str, ...]],
    rules: Sequence[ConcernRule],
) -> tuple[TargetSection, ...]:
    result: list[TargetSection] = []
    for index, (line, path) in enumerate(zip(lines, heading_paths)):
        title = _line_title(line, path)
        if title is None:
            continue
        groups = tuple(rule.group_id for rule in rules if _rule_heading_match(title, rule))
        if not groups:
            continue
        # Keep the broad target root once.  A matching child remains part of
        # this section but cannot acquire broader negative-checkbox scope.
        if any(
            _rule_heading_match(ancestor, rule)
            for ancestor in path[:-1]
            for rule in rules
        ):
            continue
        end = index
        for follower in range(index + 1, len(lines)):
            follower_path = heading_paths[follower]
            if tuple(follower_path[: len(path)]) != path:
                break
            end = follower
        payload = {
            "document_id": document_id,
            "topic_groups": sorted(groups),
            "heading": title,
            "heading_path": list(path),
            "line_range": [index + 1, end + 1],
        }
        result.append(TargetSection(
            section_id="v7-target-section-" + _semantic_hash(payload)[:24],
            topic_groups=tuple(payload["topic_groups"]),
            heading=title,
            heading_path=path,
            line_range=(index + 1, end + 1),
        ))
    return tuple(sorted(result, key=lambda section: (section.line_range, section.section_id)))


def _rules_in_path(path: Sequence[str], rules: Sequence[ConcernRule]) -> tuple[ConcernRule, ...]:
    return tuple(
        rule
        for rule in rules
        if any(_rule_heading_match(title, rule) for title in path)
    )


def _positive_reason(text: str, rule: ConcernRule) -> str | None:
    # Classify clauses independently so a negated mention cannot be revived by
    # an event-shaped substring (``未受到行政处罚`` contains ``受到行政处罚``),
    # while ``未处罚母公司，但子公司收到处罚决定`` still exposes its positive
    # second clause.
    clauses = tuple(
        clause.strip()
        for clause in re.split(r"[，,。；;！？!?]|(?:但是|不过|然而|但)", text)
        if clause.strip()
    ) or (text,)
    for clause in clauses:
        if _semantic_collision(clause, rule) or _NEGATIVE_STATEMENT_RE.search(clause):
            continue
        policy_like = bool(
            _POLICY_MARKER_RE.search(clause) or _POLICY_VERB_RE.search(clause)
        )
        has_event_override = any(
            re.search(pattern, clause) for pattern in rule.event_override_patterns
        )
        if policy_like and not has_event_override:
            continue
        for reason_code, pattern in rule.positive_patterns:
            if re.search(pattern, clause):
                return reason_code
    return None


def audit_negative_evidence(
    *,
    document_id: str,
    source_markdown: str,
    document_sha256: str,
    lines: Sequence[str],
    heading_paths: Sequence[Sequence[str]],
    topic_groups: Iterable[str],
    searched_aliases: Iterable[str] = (),
) -> NegativeEvidenceAudit:
    """Audit parsed Markdown without using a question or an expected answer.

    The only supported negative concerns are reusable annual-report semantic
    groups.  An unsupported group produces ``decision == "unsupported_topic"``
    and never authorizes negative wording.
    """

    normalized_lines = tuple(str(line) for line in lines)
    normalized_paths = tuple(tuple(_normalize(value) for value in path) for path in heading_paths)
    if len(normalized_lines) != len(normalized_paths):
        raise ValueError("lines and heading_paths must have identical lengths")

    groups = tuple(sorted({_normalize(value) for value in topic_groups if _normalize(value)}))
    rules = _compile_rules(groups)
    canonical_aliases = [alias for rule in rules for alias in rule.aliases]
    aliases = _stable_unique((*canonical_aliases, *searched_aliases))
    sections = _target_sections(
        document_id=document_id,
        lines=normalized_lines,
        heading_paths=normalized_paths,
        rules=rules,
    )

    positive: list[NegativeEvidenceFinding] = []
    checkbox_negative: list[NegativeEvidenceFinding] = []
    incidental: list[NegativeEvidenceFinding] = []

    for index, (raw_line, path) in enumerate(zip(normalized_lines, normalized_paths), start=1):
        text = _normalize(raw_line)
        if not text:
            continue
        title = _line_title(raw_line, path)
        scoped_rules = _rules_in_path(path, rules)
        leaf_rules = tuple(rule for rule in rules if path and _rule_heading_match(path[-1], rule))
        matched = _matched_aliases(text, aliases)
        checkbox_state = _checkbox_state(text)

        if checkbox_state == "positive" and scoped_rules:
            positive.append(_finding(
                document_id=document_id,
                kind="positive_event",
                reason_code="positive_applicability_checkbox",
                groups=(rule.group_id for rule in scoped_rules),
                aliases=matched,
                heading_path=path,
                line_number=index,
                text=text,
            ))
            continue

        if checkbox_state == "negative" and leaf_rules:
            # A matching ancestor plus a matching leaf denotes a narrower
            # subtopic.  Its checkbox cannot negate the broad requested topic.
            ancestor_rules = _rules_in_path(path[:-1], rules)
            if not ancestor_rules:
                checkbox_negative.append(_finding(
                    document_id=document_id,
                    kind="checkbox_negative",
                    reason_code="leaf_section_not_applicable",
                    groups=(rule.group_id for rule in leaf_rules),
                    aliases=matched,
                    heading_path=path,
                    line_number=index,
                    text=text,
                ))
                continue

        if title is not None:
            if matched or leaf_rules:
                incidental.append(_finding(
                    document_id=document_id,
                    kind="incidental_mention",
                    reason_code="section_title_only",
                    groups=(rule.group_id for rule in leaf_rules),
                    aliases=matched,
                    heading_path=path,
                    line_number=index,
                    text=text,
                ))
            continue

        positive_reasons: list[tuple[ConcernRule, str]] = []
        for rule in rules:
            reason = _positive_reason(text, rule)
            if reason is not None:
                positive_reasons.append((rule, reason))
        if positive_reasons:
            # One finding per semantic group preserves the exact veto reason
            # and yields deterministic ordering for multi-topic questions.
            for rule, reason in positive_reasons:
                positive.append(_finding(
                    document_id=document_id,
                    kind="positive_event",
                    reason_code=reason,
                    groups=(rule.group_id,),
                    aliases=matched,
                    heading_path=path,
                    line_number=index,
                    text=text,
                ))
            continue

        if matched:
            collision_groups = tuple(
                rule.group_id for rule in rules if _semantic_collision(text, rule)
            )
            if collision_groups:
                reason_code = "semantic_collision"
                finding_groups = collision_groups
            elif _POLICY_MARKER_RE.search(text) or _POLICY_VERB_RE.search(text):
                reason_code = "policy_description"
                finding_groups = tuple(rule.group_id for rule in scoped_rules)
            elif _NEGATIVE_STATEMENT_RE.search(text):
                reason_code = "explicit_negative_statement"
                finding_groups = tuple(rule.group_id for rule in scoped_rules)
            else:
                reason_code = "related_term_only"
                finding_groups = tuple(rule.group_id for rule in scoped_rules)
            incidental.append(_finding(
                document_id=document_id,
                kind="incidental_mention",
                reason_code=reason_code,
                groups=finding_groups,
                aliases=matched,
                heading_path=path,
                line_number=index,
                text=text,
            ))

    positive_tuple = tuple(sorted(positive, key=NegativeEvidenceFinding.sort_key))
    checkbox_tuple = tuple(sorted(checkbox_negative, key=NegativeEvidenceFinding.sort_key))
    incidental_tuple = tuple(sorted(incidental, key=NegativeEvidenceFinding.sort_key))

    unsupported = set(groups).difference(rule.group_id for rule in rules)
    if not groups or unsupported:
        decision = "unsupported_topic"
        safe_scope = None
    elif positive_tuple:
        decision = "positive_event_found"
        safe_scope = None
    elif checkbox_tuple:
        decision = "scoped_checkbox_negative"
        safe_scope = "target_section"
    else:
        decision = "no_positive_event_found"
        safe_scope = "document_event_search"

    base = {
        "schema_version": NEGATIVE_AUDIT_SCHEMA,
        "document_id": str(document_id),
        "source_markdown": str(source_markdown),
        "document_sha256": str(document_sha256),
        "line_count": len(normalized_lines),
        "channels_scanned": [
            "markdown_headings",
            "markdown_body",
            "embedded_html_tables",
            "target_section_headings",
            "applicability_controls",
            "event_predicates",
            "policy_and_collision_filters",
        ],
        "topic_groups": list(groups),
        "searched_aliases": list(aliases),
        "target_sections": [section.to_dict() for section in sections],
        "positive_events": [finding.to_dict() for finding in positive_tuple],
        "scoped_checkbox_negatives": [finding.to_dict() for finding in checkbox_tuple],
        "incidental_mentions": [finding.to_dict() for finding in incidental_tuple],
        "decision": decision,
        "safe_claim_scope": safe_scope,
        "complete_scan": True,
    }
    return NegativeEvidenceAudit(
        audit_id="v7-negative-audit-" + _semantic_hash(base)[:24],
        schema_version=NEGATIVE_AUDIT_SCHEMA,
        document_id=str(document_id),
        source_markdown=str(source_markdown),
        document_sha256=str(document_sha256),
        line_count=len(normalized_lines),
        channels_scanned=tuple(base["channels_scanned"]),
        topic_groups=groups,
        searched_aliases=aliases,
        target_sections=sections,
        positive_events=positive_tuple,
        scoped_checkbox_negatives=checkbox_tuple,
        incidental_mentions=incidental_tuple,
        decision=decision,
        safe_claim_scope=safe_scope,
        complete_scan=True,
    )


def recommended_negative_wording(
    audit: NegativeEvidenceAudit,
    *,
    concern: str | None = None,
) -> str | None:
    """Return claim-safe Chinese wording, or ``None`` when the audit vetoes it.

    The wording intentionally speaks about a report section or the search
    result.  It never converts missing disclosure into a claim that the company
    did not experience an event.
    """

    if not audit.allows_negative_wording:
        return None
    if audit.safe_claim_scope == "target_section":
        finding = audit.scoped_checkbox_negatives[0]
        heading = finding.heading_path[-1] if finding.heading_path else "相关事项"
        return f"“{heading}”栏明确勾选“不适用”，该栏未披露相关事件。"
    label = _normalize(concern or (audit.searched_aliases[0] if audit.searched_aliases else "相关事项"))
    return f"未检索到与“{label}”相关的事件披露。"


__all__ = [
    "ConcernRule",
    "NEGATIVE_AUDIT_SCHEMA",
    "SUPPORTED_NEGATIVE_TOPIC_GROUPS",
    "NegativeEvidenceAudit",
    "NegativeEvidenceFinding",
    "TargetSection",
    "audit_negative_evidence",
    "recommended_negative_wording",
]
