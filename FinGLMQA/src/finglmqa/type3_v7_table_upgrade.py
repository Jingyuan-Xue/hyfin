"""Pure, fail-closed decisions for upgrading weak Type 3 text answers.

The v7 retriever deliberately treats a recovered table row as evidence rather
than as an answer.  This module is the small policy boundary between those
audited rows and an existing narrative answer.  It has no filesystem access,
does not call a model, and only examines the explicit question, answer, scope,
citations, row semantics, and numeric authorizations supplied by its caller.

``decide_table_upgrade`` requires a ``row_semantics`` array on every usable
table group.  Each row must name its document and fragment, its heading path,
row label, column labels, rendered text, and the numeric authorization ids used
by that rendered text.  Requiring this projection prevents a broad table hit
from silently adding rows that do not answer the question.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
from pathlib import PurePosixPath
import re
import unicodedata
from typing import Any, Iterable, Mapping, Sequence

from .contracts import semantic_sha256


TABLE_NUMERIC_AUTHORIZATION_SCHEMA = (
    "finglmqa.experimental.table_numeric_authorization.v1"
)
TABLE_UPGRADE_POLICY_VERSION = "type3-v7-table-upgrade-v1"
MAX_UPGRADE_ROWS = 12
MAX_UPGRADE_GROUPS = 3
MAX_UPGRADE_CHARS = 1200

_AUTHORIZATION_FIELDS = frozenset({
    "schema_version",
    "authorization_id",
    "document_id",
    "company",
    "report_year",
    "metric_year",
    "table_id",
    "table_content_sha256",
    "fragment_id",
    "cell_coordinate",
    "column_label",
    "raw_value",
    "raw_value_sha256",
    "normalized_value",
    "normalized_unit",
    "source_markdown",
    "source_line_range",
    "allowed_renderings",
})
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_DECIMAL_RE = re.compile(r"^[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?[%％]?$")
_NUMBER_RE = re.compile(
    r"(?<![\d,.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[%％])?(?![\d,.])"
)
_YEAR_RE = re.compile(r"(?<!\d)((?:19|20|21)\d{2})\s*年?")
_SAFE_UNITS = frozenset({
    "%", "元", "万元", "亿元", "千元", "人民币元", "人民币万元",
    "人民币千元", "人民币亿元", "股", "万股", "人", "吨", "万吨",
    "万吨/日", "平方米", "万平方米", "万件", "万辆", "双", "美元",
    "澳元", "兆瓦", "万千瓦时", "万套/万件", "mg/L",
})

_UNCERTAIN_MARKERS = (
    "无法确定", "无法判断", "无法回答", "不能确定", "不清楚", "不详",
    "未知", "信息不足", "资料不足", "证据不足", "未找到", "未检索到",
    "未能找到", "暂无数据", "暂无相关", "没有找到", "未提供具体",
    "未披露具体", "无法从", "无法得出",
)
_GENERIC_NON_ANSWER_MARKERS = (
    "请参阅年报", "详见年报", "相关情况如下", "年报中有相关披露",
    "公司进行了披露", "具体情况请查阅", "以年报为准",
)
_DEFINITIVE_SHORT_MARKERS = (
    "不适用", "无重大变化", "不存在", "未发生", "无此事项", "没有此事项",
)

# These are domain vocabulary groups, not benchmark cases.  Specialty table
# intents win over the broad financial intent so, for example, an R&D question
# mentioning operating revenue remains an R&D question.
_INTENT_ALIASES: Mapping[str, tuple[str, ...]] = {
    "customer": (
        "主要客户", "前五大客户", "前五名客户", "客户集中度", "销售客户", "客户",
    ),
    "supplier": (
        "主要供应商", "前五大供应商", "前五名供应商", "供应商集中度", "采购供应商",
        "供应商",
    ),
    "research_development": (
        "研发投入", "研发费用", "研发人员", "研发项目", "研究开发", "研发",
    ),
    "major_contract": (
        "重大合同及其履行", "重大合同", "重要合同", "合同履行", "重大协议",
    ),
    "equity_asset_sale": (
        "出售股权", "股权出售", "股权转让", "转让股权", "资产出售", "出售资产",
        "资产处置", "处置资产", "出售子公司", "处置子公司", "产权转让",
    ),
    "related_party_transaction": (
        "重大关联交易", "日常关联交易", "关联方交易", "关联交易",
    ),
}

_FINANCIAL_METRICS: Mapping[str, tuple[str, ...]] = {
    "revenue": ("营业收入", "主营业务收入", "销售收入"),
    "operating_cost": ("营业成本", "主营业务成本", "销售成本"),
    "net_profit": ("归母净利润", "归属于母公司股东的净利润", "净利润"),
    "total_assets": ("资产总额", "总资产", "资产及负债状况", "资产负债状况"),
    "total_liabilities": ("负债总额", "总负债", "资产及负债状况", "资产负债状况"),
    "net_assets": ("净资产", "所有者权益", "股东权益"),
    "operating_cash_flow": ("经营活动产生的现金流量净额", "经营现金流", "现金流量净额"),
    "gross_margin": ("毛利率", "毛利"),
    "return_on_equity": ("净资产收益率", "加权平均净资产收益率"),
    "earnings_per_share": ("每股收益", "基本每股收益", "稀释每股收益"),
    "receivables": ("应收账款", "应收款项"),
    "inventory": ("存货",),
    "expense": ("销售费用", "管理费用", "财务费用"),
}
_GENERIC_FINANCIAL_ALIASES = (
    "主要会计数据", "财务指标", "财务数据", "经营指标", "利润表", "资产负债表",
    "现金流量表",
)

_ROW_VALUE_ALIASES: Mapping[str, tuple[str, ...]] = {
    "customer": (
        "客户名称", "客户", "单位名称", "销售额", "销售金额", "销售收入", "占比",
        "比例", "排名",
    ),
    "supplier": (
        "供应商名称", "供应商", "单位名称", "采购额", "采购金额", "采购成本", "占比",
        "比例", "排名",
    ),
    "research_development": (
        "投入金额", "研发投入", "研发费用", "费用化", "资本化", "占营业收入比例",
        "研发人员", "人员数量", "项目名称", "进展", "目的",
    ),
    "major_contract": (
        "合同名称", "合同标的", "交易对方", "合同对方", "签约方", "金额", "期限",
        "履行", "进度", "项目名称", "协议",
    ),
    "equity_asset_sale": (
        "交易对方", "受让方", "购买方", "标的", "股权", "资产", "转让", "出售",
        "处置", "价格", "金额", "对价", "收益", "完成", "进展",
    ),
    "related_party_transaction": (
        "关联人", "关联方", "交易类型", "交易内容", "交易金额", "预计金额",
        "实际发生金额", "关联关系", "定价原则", "结算方式", "交易对方",
    ),
}


def _normalize(value: Any) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(value))).strip()


def _compact(value: Any) -> str:
    return re.sub(r"[\s\W_]+", "", _normalize(value).lower(), flags=re.UNICODE)


def _contains_any(value: str, aliases: Iterable[str]) -> bool:
    compact = _compact(value)
    return any(_compact(alias) in compact for alias in aliases)


def _intent_ids(question: str) -> tuple[str, ...]:
    specialty = tuple(
        intent
        for intent, aliases in _INTENT_ALIASES.items()
        if _contains_any(question, aliases)
    )
    if specialty:
        return specialty
    if _contains_any(question, _GENERIC_FINANCIAL_ALIASES) or _question_metrics(question):
        return ("financial_metric",)
    return ()


def _question_metrics(question: str) -> tuple[str, ...]:
    return tuple(
        metric
        for metric, aliases in _FINANCIAL_METRICS.items()
        if _contains_any(question, aliases)
    )


def _material_length(value: str) -> int:
    text = re.sub(r"[\s\W_]+", "", _normalize(value), flags=re.UNICODE)
    return len(text)


def _citation_document_ids(citations: Sequence[Mapping[str, Any]]) -> set[str]:
    result: set[str] = set()
    for citation in citations:
        if not isinstance(citation, Mapping):
            continue
        document_id = citation.get("document_id")
        if isinstance(document_id, str) and document_id.strip():
            result.add(document_id.strip())
    return result


@dataclass(frozen=True)
class TextAnswerConfidence:
    """Deterministic confidence classification for an existing text answer."""

    low_confidence: bool
    reason_codes: tuple[str, ...]
    intent_ids: tuple[str, ...]
    material_length: int
    prefer_replacement: bool


def assess_text_answer_confidence(
    *,
    question: str,
    answer_text: str,
    status: str | None = None,
    citations: Sequence[Mapping[str, Any]] = (),
    document_id: str | None = None,
) -> TextAnswerConfidence:
    """Classify a text answer without consulting any benchmark annotation.

    A missing citation alone is not enough to replace a substantive answer.
    It is only a corroborating signal for a short or semantically empty answer.
    """

    question_text = _normalize(question)
    answer = _normalize(answer_text)
    intents = _intent_ids(question_text)
    length = _material_length(answer)
    reasons: set[str] = set()
    replacement_reasons: set[str] = set()

    if not answer:
        reasons.add("empty_answer")
        replacement_reasons.add("empty_answer")
    if status is not None and status not in {"ok", "success"}:
        reasons.add("non_success_status")
        replacement_reasons.add("non_success_status")
    if answer and _contains_any(answer, _UNCERTAIN_MARKERS):
        reasons.add("explicit_uncertainty")
        replacement_reasons.add("explicit_uncertainty")
    if answer and _contains_any(answer, _GENERIC_NON_ANSWER_MARKERS):
        reasons.add("generic_non_answer")
        replacement_reasons.add("generic_non_answer")

    compact_question = _compact(question_text)
    compact_answer = _compact(answer)
    if compact_answer and compact_question:
        shorter, longer = sorted((compact_answer, compact_question), key=len)
        if len(shorter) >= 6 and (shorter in longer or longer in shorter):
            reasons.add("question_echo")
            replacement_reasons.add("question_echo")

    definitive_short = _contains_any(answer, _DEFINITIVE_SHORT_MARKERS)
    if answer and length <= 18 and not definitive_short:
        reasons.add("terse_answer")

    if intents and answer:
        answer_intents = set(_intent_ids(answer))
        if not set(intents).intersection(answer_intents):
            reasons.add("missing_intent_anchor")
            # Once a document-scoped table group independently satisfies the
            # requested intent, retaining an unrelated narrative would keep
            # the very retrieval error this policy is meant to repair.
            replacement_reasons.add("missing_intent_anchor")
        if "financial_metric" in intents:
            asked_metrics = set(_question_metrics(question_text))
            answer_metrics = set(_question_metrics(answer))
            has_number = bool(_NUMBER_RE.search(answer))
            if (asked_metrics and not asked_metrics.intersection(answer_metrics)) or not has_number:
                reasons.add("missing_metric_value")
                replacement_reasons.add("missing_metric_value")

    citation_docs = _citation_document_ids(citations)
    if document_id and citation_docs and citation_docs != {document_id}:
        reasons.add("citation_scope_mismatch")
        replacement_reasons.add("citation_scope_mismatch")
    if not citations and ({"terse_answer", "missing_intent_anchor"} & reasons):
        reasons.add("uncited_weak_answer")

    # A short, explicitly definitive answer with an in-scope citation is not
    # weak merely because it has few characters.
    if definitive_short and citations and not replacement_reasons:
        reasons.discard("terse_answer")
        reasons.discard("missing_intent_anchor")
        reasons.discard("uncited_weak_answer")

    low_signal_reasons = {
        "empty_answer", "non_success_status", "explicit_uncertainty",
        "generic_non_answer", "question_echo", "terse_answer",
        "missing_intent_anchor", "missing_metric_value", "citation_scope_mismatch",
        "uncited_weak_answer",
    }
    low = bool(reasons.intersection(low_signal_reasons))
    return TextAnswerConfidence(
        low_confidence=low,
        reason_codes=tuple(sorted(reasons)),
        intent_ids=intents,
        material_length=length,
        prefer_replacement=bool(replacement_reasons),
    )


def is_low_confidence_text_answer(
    *,
    question: str,
    answer_text: str,
    status: str | None = None,
    citations: Sequence[Mapping[str, Any]] = (),
    document_id: str | None = None,
) -> bool:
    """Boolean convenience wrapper around ``assess_text_answer_confidence``."""

    return assess_text_answer_confidence(
        question=question,
        answer_text=answer_text,
        status=status,
        citations=citations,
        document_id=document_id,
    ).low_confidence


@dataclass(frozen=True)
class TableNumericAuthorization:
    """Validated authorization for rendering one audited numeric table cell."""

    authorization_id: str
    document_id: str
    company: str
    report_year: int
    metric_year: int
    table_id: str
    table_content_sha256: str
    fragment_id: str
    cell_coordinate: tuple[int, int]
    column_label: str
    raw_value: str
    raw_value_sha256: str
    normalized_value: str
    normalized_unit: str
    source_markdown: str
    source_line_range: tuple[int, int]
    allowed_renderings: tuple[str, ...]

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        expected_document_id: str | None = None,
    ) -> "TableNumericAuthorization":
        """Validate a v7 mapping or raise ``ValueError`` without partial use."""

        if not isinstance(value, Mapping) or set(value) != _AUTHORIZATION_FIELDS:
            raise ValueError("numeric authorization fields differ from v1")
        if value.get("schema_version") != TABLE_NUMERIC_AUTHORIZATION_SCHEMA:
            raise ValueError("unsupported numeric authorization schema")
        for field in (
            "authorization_id", "document_id", "company", "table_id", "fragment_id",
            "column_label", "raw_value", "normalized_value", "normalized_unit",
            "source_markdown",
        ):
            if not isinstance(value.get(field), str) or not str(value[field]).strip():
                raise ValueError(f"numeric authorization {field} must be non-empty")
        document_id = str(value["document_id"])
        if expected_document_id is not None and document_id != expected_document_id:
            raise ValueError("numeric authorization document scope mismatch")
        for field in ("report_year", "metric_year"):
            number = value.get(field)
            if isinstance(number, bool) or not isinstance(number, int) or not 1900 <= number <= 2199:
                raise ValueError(f"numeric authorization {field} must be a plausible year")
        if value["normalized_unit"] not in _SAFE_UNITS:
            raise ValueError("numeric authorization unit is not allowed")
        if not isinstance(value.get("table_content_sha256"), str) or not _SHA256_RE.fullmatch(
            str(value["table_content_sha256"])
        ):
            raise ValueError("numeric authorization table hash is invalid")
        if not isinstance(value.get("raw_value_sha256"), str) or value["raw_value_sha256"] != hashlib.sha256(
            str(value["raw_value"]).encode("utf-8")
        ).hexdigest():
            raise ValueError("numeric authorization raw value hash mismatch")

        coordinate = value.get("cell_coordinate")
        if (
            not isinstance(coordinate, list)
            or len(coordinate) != 2
            or any(isinstance(part, bool) or not isinstance(part, int) or part < 0 for part in coordinate)
        ):
            raise ValueError("numeric authorization coordinate is invalid")
        line_range = value.get("source_line_range")
        if (
            not isinstance(line_range, list)
            or len(line_range) != 2
            or any(isinstance(part, bool) or not isinstance(part, int) or part < 1 for part in line_range)
            or line_range[0] > line_range[1]
        ):
            raise ValueError("numeric authorization source line range is invalid")
        source_path = PurePosixPath(str(value["source_markdown"]))
        if source_path.is_absolute() or ".." in source_path.parts:
            raise ValueError("numeric authorization source path is not portable")

        compact_raw = re.sub(r"\s+", "", _normalize(value["raw_value"]))
        if not _DECIMAL_RE.fullmatch(compact_raw):
            raise ValueError("numeric authorization raw value is not decimal")
        try:
            decimal_value = Decimal(compact_raw.rstrip("%％").replace(",", ""))
        except InvalidOperation as exc:
            raise ValueError("numeric authorization raw value is invalid") from exc
        if not decimal_value.is_finite() or format(decimal_value, "f") != value["normalized_value"]:
            raise ValueError("numeric authorization normalized value mismatch")

        allowed = value.get("allowed_renderings")
        if (
            not isinstance(allowed, list)
            or not allowed
            or any(not isinstance(item, str) or not item for item in allowed)
            or allowed != sorted(set(allowed))
        ):
            raise ValueError("numeric authorization allowed renderings are invalid")
        expected_renderings = [compact_raw]
        unit = str(value["normalized_unit"])
        if unit == "%" and not compact_raw.endswith(("%", "％")):
            expected_renderings.append(f"{compact_raw}%")
        elif unit != "%" and not compact_raw.endswith(unit):
            expected_renderings.append(f"{compact_raw}{unit}")
        if allowed != sorted(set(expected_renderings)):
            raise ValueError("numeric authorization allowed renderings are not canonical")

        label_years = {int(year) for year in _YEAR_RE.findall(str(value["column_label"]))}
        metric_year = int(value["metric_year"])
        report_year = int(value["report_year"])
        if label_years:
            if label_years != {metric_year}:
                raise ValueError("numeric authorization metric year conflicts with column")
        else:
            compact_label = _compact(value["column_label"])
            if any(marker in compact_label for marker in ("上年", "上期", "期初")):
                expected_metric_year = report_year - 1
            elif any(
                marker in compact_label
                for marker in (
                    "本年", "本期", "期末", "年度", "发生额", "变动", "金额", "数"
                )
            ):
                expected_metric_year = report_year
            else:
                raise ValueError("numeric authorization column has no metric year semantics")
            if metric_year != expected_metric_year:
                raise ValueError("numeric authorization relative metric year mismatch")
        unsigned = dict(value)
        authorization_id = str(unsigned.pop("authorization_id"))
        if authorization_id != "v7-table-auth-" + semantic_sha256(unsigned)[:24]:
            raise ValueError("numeric authorization id mismatch")

        return cls(
            authorization_id=authorization_id,
            document_id=document_id,
            company=str(value["company"]),
            report_year=report_year,
            metric_year=metric_year,
            table_id=str(value["table_id"]),
            table_content_sha256=str(value["table_content_sha256"]),
            fragment_id=str(value["fragment_id"]),
            cell_coordinate=(int(coordinate[0]), int(coordinate[1])),
            column_label=str(value["column_label"]),
            raw_value=str(value["raw_value"]),
            raw_value_sha256=str(value["raw_value_sha256"]),
            normalized_value=str(value["normalized_value"]),
            normalized_unit=unit,
            source_markdown=str(value["source_markdown"]),
            source_line_range=(int(line_range[0]), int(line_range[1])),
            allowed_renderings=tuple(allowed),
        )

    def as_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": TABLE_NUMERIC_AUTHORIZATION_SCHEMA,
            "authorization_id": self.authorization_id,
            "document_id": self.document_id,
            "company": self.company,
            "report_year": self.report_year,
            "metric_year": self.metric_year,
            "table_id": self.table_id,
            "table_content_sha256": self.table_content_sha256,
            "fragment_id": self.fragment_id,
            "cell_coordinate": list(self.cell_coordinate),
            "column_label": self.column_label,
            "raw_value": self.raw_value,
            "raw_value_sha256": self.raw_value_sha256,
            "normalized_value": self.normalized_value,
            "normalized_unit": self.normalized_unit,
            "source_markdown": self.source_markdown,
            "source_line_range": list(self.source_line_range),
            "allowed_renderings": list(self.allowed_renderings),
        }


@dataclass(frozen=True)
class _SemanticRow:
    document_id: str
    fragment_id: str
    heading_path: tuple[str, ...]
    row_label: str
    column_labels: tuple[str, ...]
    rendered_text: str
    authorization_ids: tuple[str, ...]
    source_ordinal: int
    table_id: str


@dataclass(frozen=True)
class _SelectedGroup:
    group_id: str
    text: str
    citation_ids: tuple[str, ...]
    authorizations: tuple[TableNumericAuthorization, ...]
    intent_ids: tuple[str, ...]
    source_ordinal: int
    row_count: int


@dataclass(frozen=True)
class TableUpgradeDecision:
    """A deterministic keep/supplement/replace result ready for integration."""

    action: str
    answer_text: str
    selected_group_ids: tuple[str, ...]
    selected_citation_ids: tuple[str, ...]
    numeric_authorizations: tuple[TableNumericAuthorization, ...]
    reason_codes: tuple[str, ...]
    confidence: TextAnswerConfidence
    policy_version: str = TABLE_UPGRADE_POLICY_VERSION

    @property
    def applied(self) -> bool:
        return self.action in {"supplement", "replace"}

    def numeric_authorization_mappings(self) -> list[dict[str, Any]]:
        return [row.as_mapping() for row in self.numeric_authorizations]


def _parse_semantic_row(value: Any, *, fallback_ordinal: int) -> _SemanticRow | None:
    if not isinstance(value, Mapping):
        return None
    document_id = value.get("document_id")
    fragment_id = value.get("fragment_id", value.get("candidate_id"))
    heading_path = value.get("heading_path")
    column_labels = value.get("column_labels")
    authorization_ids = value.get("numeric_authorization_ids", ())
    source_ordinal = value.get("source_ordinal", fallback_ordinal)
    table_id = value.get("table_id")
    if (
        not isinstance(document_id, str) or not document_id.strip()
        or not isinstance(fragment_id, str) or not fragment_id.strip()
        or not isinstance(heading_path, (list, tuple))
        or any(not isinstance(item, str) or not item.strip() for item in heading_path)
        or not isinstance(column_labels, (list, tuple))
        or any(not isinstance(item, str) or not item.strip() for item in column_labels)
        or not isinstance(authorization_ids, (list, tuple))
        or any(not isinstance(item, str) or not item.strip() for item in authorization_ids)
        or len(set(authorization_ids)) != len(authorization_ids)
        or isinstance(source_ordinal, bool) or not isinstance(source_ordinal, int) or source_ordinal < 0
        or not isinstance(table_id, str) or not table_id.strip()
    ):
        return None
    row_label = value.get("row_label")
    rendered_text = value.get("rendered_text")
    if not isinstance(row_label, str) or not row_label.strip():
        return None
    if not isinstance(rendered_text, str) or not rendered_text.strip():
        return None
    return _SemanticRow(
        document_id=document_id.strip(),
        fragment_id=fragment_id.strip(),
        heading_path=tuple(_normalize(item) for item in heading_path),
        row_label=_normalize(row_label),
        column_labels=tuple(_normalize(item) for item in column_labels),
        rendered_text=_normalize(rendered_text),
        authorization_ids=tuple(authorization_ids),
        source_ordinal=source_ordinal,
        table_id=table_id.strip(),
    )


def _heading_matches(intent: str, heading_text: str, row_text: str, question: str) -> bool:
    if intent == "financial_metric":
        asked = _question_metrics(question)
        if asked:
            aliases = tuple(
                alias for metric in asked for alias in _FINANCIAL_METRICS[metric]
            )
            return _contains_any(f"{heading_text} {row_text}", aliases)
        return _contains_any(heading_text, _GENERIC_FINANCIAL_ALIASES) or any(
            _contains_any(row_text, aliases) for aliases in _FINANCIAL_METRICS.values()
        )
    # Some extracted tables inherit only a leaf column label as their rendered
    # heading.  The audited row semantics still carry an explicit customer /
    # supplier / contract label, so accept either surface while retaining the
    # stricter per-intent row-field count below.
    return _contains_any(f"{heading_text} {row_text}", _INTENT_ALIASES[intent])


def _row_matches(intent: str, row: _SemanticRow, question: str) -> bool:
    semantic_text = " ".join((row.row_label, *row.column_labels))
    heading_text = " ".join(row.heading_path)
    if not _heading_matches(intent, heading_text, semantic_text, question):
        return False
    if intent == "financial_metric":
        asked = _question_metrics(question)
        composite_balance_question = _contains_any(
            question, ("资产及负债状况", "资产负债状况")
        )
        composite_balance_heading = _contains_any(
            heading_text, ("资产及负债状况", "资产负债状况")
        )
        if composite_balance_question and composite_balance_heading:
            return bool(row.authorization_ids)
        if asked and not any(
            _contains_any(semantic_text, _FINANCIAL_METRICS[metric]) for metric in asked
        ):
            return False
        return bool(row.authorization_ids)
    value_hits = sum(
        1 for alias in _ROW_VALUE_ALIASES[intent] if _contains_any(semantic_text, (alias,))
    )
    # Customer and supplier identity/value tables need two semantic fields;
    # other specialty concerns can be anchored by one specific detail column.
    required = 2 if intent in {"customer", "supplier"} else 1
    return value_hits >= required


def _number_is_authorized(
    text: str,
    match: re.Match[str],
    authorizations: Sequence[TableNumericAuthorization],
    *,
    report_year: int,
) -> bool:
    token = _normalize(match.group(0)).replace("％", "%")
    if token == str(report_year):
        suffix = text[match.end() : match.end() + 2]
        if suffix.startswith("年"):
            return True
    for authorization in authorizations:
        for rendering in authorization.allowed_renderings:
            normalized_rendering = _normalize(rendering).replace("％", "%")
            start = 0
            while True:
                position = text.find(normalized_rendering, start)
                if position < 0:
                    break
                end = position + len(normalized_rendering)
                if position <= match.start() and match.end() <= end:
                    rendered_numbers = {
                        _normalize(item.group(0)).replace("％", "%")
                        for item in _NUMBER_RE.finditer(normalized_rendering)
                    }
                    if token in rendered_numbers:
                        return True
                start = position + 1
    return False


def _safe_numeric_text(
    text: str,
    authorizations: Sequence[TableNumericAuthorization],
    *,
    report_year: int,
) -> bool:
    return all(
        _number_is_authorized(
            text,
            match,
            authorizations,
            report_year=report_year,
        )
        for match in _NUMBER_RE.finditer(text)
    )


def _valid_group_document_ids(group: Mapping[str, Any]) -> set[str] | None:
    ids: set[str] = set()
    citations = group.get("citations")
    rows = group.get("row_semantics")
    authorizations = group.get("numeric_authorizations", ())
    if not isinstance(citations, (list, tuple)) or not citations:
        return None
    if not isinstance(rows, (list, tuple)) or not rows:
        return None
    if not isinstance(authorizations, (list, tuple)):
        return None
    for collection in (citations, rows, authorizations):
        for value in collection:
            if not isinstance(value, Mapping):
                return None
            document_id = value.get("document_id")
            if not isinstance(document_id, str) or not document_id.strip():
                return None
            ids.add(document_id.strip())
    return ids


def _select_group(
    group: Mapping[str, Any],
    *,
    document_id: str,
    question: str,
    intents: tuple[str, ...],
    report_year: int,
) -> _SelectedGroup | None:
    if group.get("source_kind") != "table":
        return None
    group_id = group.get("group_id")
    heading = group.get("heading")
    if not isinstance(group_id, str) or not group_id.strip():
        return None
    if not isinstance(heading, str) or not heading.strip():
        return None
    if _valid_group_document_ids(group) != {document_id}:
        return None

    citations = group["citations"]
    citation_by_fragment: dict[str, Mapping[str, Any]] = {}
    for citation in citations:
        candidate_id = citation.get("candidate_id")
        citation_id = citation.get("citation_id")
        if (
            not isinstance(candidate_id, str) or not candidate_id.strip()
            or not isinstance(citation_id, str) or not citation_id.strip()
            or candidate_id in citation_by_fragment
        ):
            return None
        citation_by_fragment[candidate_id] = citation

    raw_authorizations = group.get("numeric_authorizations", ())
    authorization_by_id: dict[str, TableNumericAuthorization] = {}
    for raw in raw_authorizations:
        raw_id = raw.get("authorization_id")
        if not isinstance(raw_id, str):
            continue
        if raw_id in authorization_by_id:
            return None
        try:
            parsed = TableNumericAuthorization.from_mapping(
                raw, expected_document_id=document_id
            )
        except ValueError:
            # One malformed or ambiguous cell must not poison independent
            # audited rows in the same extracted table.  A row that depends
            # on this cell is skipped below because its authorization id is
            # absent; no unvalidated number can become renderable.
            continue
        if parsed.report_year != report_year:
            return None
        authorization_by_id[parsed.authorization_id] = parsed

    parsed_rows: list[_SemanticRow] = []
    for ordinal, raw_row in enumerate(group["row_semantics"]):
        row = _parse_semantic_row(raw_row, fallback_ordinal=ordinal)
        if row is None or row.document_id != document_id:
            return None
        if row.fragment_id not in citation_by_fragment:
            return None
        parsed_rows.append(row)
    if len({row.fragment_id for row in parsed_rows}) != len(parsed_rows):
        return None

    selected: list[tuple[_SemanticRow, tuple[str, ...], tuple[TableNumericAuthorization, ...]]] = []
    for row in sorted(parsed_rows, key=lambda item: (item.source_ordinal, item.fragment_id)):
        matched_intents = tuple(
            intent for intent in intents if _row_matches(intent, row, question)
        )
        if not matched_intents:
            continue
        try:
            row_authorizations = tuple(
                authorization_by_id[authorization_id]
                for authorization_id in row.authorization_ids
            )
        except KeyError:
            continue
        if any(auth.fragment_id != row.fragment_id for auth in row_authorizations):
            continue
        if any(auth.table_id != row.table_id for auth in row_authorizations):
            continue
        if any(auth.column_label not in row.column_labels for auth in row_authorizations):
            continue
        if len({auth.cell_coordinate for auth in row_authorizations}) != len(row_authorizations):
            continue
        table_hashes: dict[str, set[str]] = {}
        for auth in row_authorizations:
            table_hashes.setdefault(auth.table_id, set()).add(auth.table_content_sha256)
        if any(len(hashes) != 1 for hashes in table_hashes.values()):
            continue
        citation = citation_by_fragment[row.fragment_id]
        citation_path = citation.get("source_markdown")
        citation_range = citation.get("line_range")
        if any(
            auth.source_markdown != citation_path
            or list(auth.source_line_range) != citation_range
            for auth in row_authorizations
        ):
            continue
        if not _safe_numeric_text(
            row.rendered_text,
            row_authorizations,
            report_year=report_year,
        ):
            continue
        selected.append((row, matched_intents, row_authorizations))
        if len(selected) >= MAX_UPGRADE_ROWS:
            break
    if not selected:
        return None

    # A numeric financial answer without a used authorization is never emitted.
    if "financial_metric" in intents and not any(auths for _, _, auths in selected):
        return None
    selected_auth = {
        auth.authorization_id: auth
        for _, _, authorizations in selected
        for auth in authorizations
    }
    rows_text = "；".join(row.rendered_text for row, _, _ in selected)
    rendered = f"{_normalize(heading)}：{rows_text}"
    authorizations = tuple(selected_auth[key] for key in sorted(selected_auth))
    if not _safe_numeric_text(rendered, authorizations, report_year=report_year):
        return None
    citation_ids = tuple(sorted({
        str(citation_by_fragment[row.fragment_id]["citation_id"])
        for row, _, _ in selected
    }))
    matched = tuple(dict.fromkeys(
        intent for _, row_intents, _ in selected for intent in row_intents
    ))
    return _SelectedGroup(
        group_id=group_id.strip(),
        text=rendered,
        citation_ids=citation_ids,
        authorizations=authorizations,
        intent_ids=matched,
        source_ordinal=min(row.source_ordinal for row, _, _ in selected),
        row_count=len(selected),
    )


def decide_table_upgrade(
    *,
    document_id: str,
    report_year: int,
    question: str,
    answer_text: str,
    table_groups: Sequence[Mapping[str, Any]],
    status: str | None = None,
    citations: Sequence[Mapping[str, Any]] = (),
) -> TableUpgradeDecision:
    """Decide whether authorized table rows may supplement or replace text.

    The function returns ``keep`` for unsupported intents, confident answers,
    ambiguous document scope, malformed row projections, and unauthorized
    numeric text.  Group and row ordering is derived from explicit stable keys,
    never from input order.
    """

    confidence = assess_text_answer_confidence(
        question=question,
        answer_text=answer_text,
        status=status,
        citations=citations,
        document_id=document_id,
    )
    base_answer = _normalize(answer_text)
    reasons = set(confidence.reason_codes)
    if not isinstance(document_id, str) or not document_id.strip():
        reasons.add("missing_document_scope")
    if isinstance(report_year, bool) or not isinstance(report_year, int) or not 1900 <= report_year <= 2199:
        reasons.add("invalid_report_year")
    if not confidence.intent_ids:
        reasons.add("unsupported_table_intent")
    if not confidence.low_confidence:
        reasons.add("base_answer_confident")
    if (
        "missing_document_scope" in reasons
        or "invalid_report_year" in reasons
        or "unsupported_table_intent" in reasons
        or "base_answer_confident" in reasons
    ):
        return TableUpgradeDecision(
            action="keep",
            answer_text=base_answer,
            selected_group_ids=(),
            selected_citation_ids=(),
            numeric_authorizations=(),
            reason_codes=tuple(sorted(reasons)),
            confidence=confidence,
        )

    selected_groups = [
        selected
        for group in table_groups
        if isinstance(group, Mapping)
        for selected in (
            _select_group(
                group,
                document_id=document_id,
                question=question,
                intents=confidence.intent_ids,
                report_year=report_year,
            ),
        )
        if selected is not None
    ]
    selected_groups.sort(key=lambda row: (row.source_ordinal, row.group_id))

    # Prefer groups that add a previously uncovered requested intent, then add
    # other aligned groups only while within the bounded answer surface.
    chosen: list[_SelectedGroup] = []
    covered: set[str] = set()
    used_chars = 0
    for group in selected_groups:
        adds_coverage = bool(set(group.intent_ids) - covered)
        if not adds_coverage and chosen:
            continue
        extra = len(group.text) + (1 if chosen else 0)
        if used_chars + extra > MAX_UPGRADE_CHARS:
            continue
        chosen.append(group)
        covered.update(group.intent_ids)
        used_chars += extra
        if len(chosen) >= MAX_UPGRADE_GROUPS or covered == set(confidence.intent_ids):
            break
    if not chosen:
        reasons.add("no_safe_aligned_table_evidence")
        return TableUpgradeDecision(
            action="keep",
            answer_text=base_answer,
            selected_group_ids=(),
            selected_citation_ids=(),
            numeric_authorizations=(),
            reason_codes=tuple(sorted(reasons)),
            confidence=confidence,
        )

    table_text = "\n".join(group.text for group in chosen)
    if confidence.prefer_replacement or not base_answer:
        action = "replace"
        upgraded_answer = table_text
    else:
        action = "supplement"
        upgraded_answer = "\n".join((base_answer, table_text))
    authorizations = {
        auth.authorization_id: auth
        for group in chosen
        for auth in group.authorizations
    }
    reasons.add(f"table_{action}_authorized")
    return TableUpgradeDecision(
        action=action,
        answer_text=upgraded_answer,
        selected_group_ids=tuple(group.group_id for group in chosen),
        selected_citation_ids=tuple(sorted({
            citation_id for group in chosen for citation_id in group.citation_ids
        })),
        numeric_authorizations=tuple(
            authorizations[key] for key in sorted(authorizations)
        ),
        reason_codes=tuple(sorted(reasons)),
        confidence=confidence,
    )


def can_use_table_evidence(**kwargs: Any) -> bool:
    """Boolean convenience wrapper for integrations that only need a gate."""

    return decide_table_upgrade(**kwargs).applied


__all__ = [
    "MAX_UPGRADE_CHARS",
    "MAX_UPGRADE_GROUPS",
    "MAX_UPGRADE_ROWS",
    "TABLE_NUMERIC_AUTHORIZATION_SCHEMA",
    "TABLE_UPGRADE_POLICY_VERSION",
    "TableNumericAuthorization",
    "TableUpgradeDecision",
    "TextAnswerConfidence",
    "assess_text_answer_confidence",
    "can_use_table_evidence",
    "decide_table_upgrade",
    "is_low_confidence_text_answer",
]
