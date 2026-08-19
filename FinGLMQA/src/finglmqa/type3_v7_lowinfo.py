"""Deterministic low-information boundary for Type 3 evidence.

This module is intentionally downstream of retrieval.  It can reject or
reorder the already-frozen A2RAG top-k, but it cannot retrieve another chunk,
change document scope, rewrite evidence, or render a number.  The title-path
compatibility decision is supplied by the caller so this generic boundary does
not duplicate the stricter Type 3 intent policy.

Only question text, candidate text, title paths, and ordinary evidence-chunk
identity fields are inspected.  There are no company, year, or benchmark-case
exceptions.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata
from typing import Any, Mapping, Sequence


LOWINFO_VERSION = "type3-v7-lowinfo-v1"

_YEAR_RE = re.compile(r"(?<!\d)(?:19|20|21)\d{2}\s*年?")
_REDIRECT_RE = re.compile(
    r"(?:具体)?(?:请参见|请参阅|详见|参见|请见)(?:本报告|报告|本节|下文|上文|"
    r"第[一二三四五六七八九十百零〇\d]+[章节部分项]|附注|附件|相关章节|有关章节)?"
)
_CHECKBOX_OPTION_RE = re.compile(
    r"([□☐○◯■☑√✓✔])\s*((?:本年度|本年)?\s*(?:适用|不适用))"
)
_SELECTED = frozenset("■☑√✓✔")
_UNSELECTED = frozenset("□☐○◯")
_STANDALONE_NEGATIVE_RE = re.compile(
    r"^(?:不适用|无|否|未发生|不存在|没有|并无|无需|未涉及|未受到)$"
)
_UNIT_VALUE_RE = re.compile(
    r"^(?:(?:人民币|美元|港币|欧元)?(?:元|万元|亿元|千元|百万元|股|万股|人|"
    r"万人|户|家|次|倍|%|％|平方米|万平方米|吨|万吨|千瓦时|万千瓦时)|"
    r"人民币|美元|港币|欧元)$"
)
_MARKDOWN_SEPARATOR_RE = re.compile(r"^:?-{3,}:?$")
_DOCUMENT_TITLE_RE = re.compile(
    r"^(?:\S{2,50}(?:股份有限公司|有限责任公司|集团有限公司|公司)\s*)?"
    r"(?:(?:19|20|21)\d{2}\s*年?)?(?:半年度|年度|季度)?报告(?:摘要|全文)?$"
)
_HEADING_PREFIX_RE = re.compile(
    r"^(?:第[一二三四五六七八九十百零〇\d]+[章节部分篇]|"
    r"[一二三四五六七八九十百零〇\d]+[、.)）]|"
    r"[（(][一二三四五六七八九十百零〇\d]+[）)])\s*\S"
)
_FACT_VERB_RE = re.compile(
    r"(?:为|是|达到|达成|增长|下降|增加|减少|实现|完成|获得|拥有|持有|发生|"
    r"未发生|存在|不存在|包括|构成|占比|同比|环比|续聘|聘任|解聘|出具|发表|"
    r"支付|收到|投入|生产|销售|签订|终止|转让|处置|设立|成立|导致|带来|"
    r"影响|面临)"
)
_EXPLICIT_FACT_PHRASE_RE = re.compile(
    r"(?:标准无保留意见|无保留意见|保留意见|否定意见|无法表示意见|"
    r"未发现重大缺陷|无重大诉讼|无重大仲裁|不构成重大影响)"
)
_LABELED_NUMBER_RE = re.compile(
    r"(?:[\u3400-\u9fffA-Za-z]{2,}.{0,18}\d|\d.{0,12}[\u3400-\u9fffA-Za-z]{2,})"
)
_HEADER_CELL_RE = re.compile(
    r"^(?:序号|项目|项目名称|科目|名称|类别|类型|指标|年份|年度|日期|本期|"
    r"上期|本年|上年|期初|期末|本期数|上期数|本年数|上年数|本期金额|"
    r"上期金额|金额|数量|比例|占比|同比|变动比例|备注|说明|合计)$"
)
_HEADING_END_RE = re.compile(
    r"(?:情况|事项|概况|说明|分析|讨论|意见|报告|信息|构成|结构|制度|"
    r"管理|治理|业务|模式|风险|计划|战略|展望|目录|指标|数据|变化)$"
)
_BARE_TOPIC_HEADING_RE = re.compile(
    r"(?:诉讼仲裁|重大诉讼|重大仲裁|关联交易|现金流量|营业收入|研发投入|"
    r"审计意见|固定资产|核心竞争力|行业地位)$"
)
_GENERIC_REDIRECT_SUBJECT_RE = re.compile(
    r"^(?:有关|相关|上述|该等|具体|其他)?(?:情况|事项|内容|信息|详情|说明)?"
    r"(?:可)?$"
)

_BOILERPLATE_RES = (
    re.compile(r"(?:董事会|监事会).{0,35}(?:保证|承诺).{0,30}(?:真实|准确|完整)"),
    re.compile(r"不存在虚假记载、?误导性陈述或重大遗漏"),
    re.compile(r"(?:全体董事|全体监事).{0,25}(?:异议|保证)"),
    re.compile(r"公司严格按照.{0,40}(?:公司法|证券法).{0,45}(?:规范运作|治理)"),
    re.compile(r"(?:不断|持续)(?:建立健全|健全|完善).{0,25}(?:治理结构|内控制度|内部控制制度)"),
    re.compile(r"依法依规.{0,20}履行信息披露义务"),
    re.compile(r"(?:股东大会|董事会|监事会).{0,35}(?:各司其职|规范运作|依法运作)"),
    re.compile(r"按照.{0,30}审计准则.{0,30}(?:执行|开展|实施)(?:了)?审计工作"),
    re.compile(r"(?:已获取|获取的)审计证据.{0,25}(?:充分|适当)"),
    re.compile(r"管理层负责.{0,35}(?:编制|财务报表|内部控制)"),
    re.compile(r"(?:仅供参考|不构成投资建议|敬请投资者注意投资风险)"),
)

# Stable annual-report concepts bridge ordinary synonyms.  These are generic
# taxonomy entries rather than document- or question-specific exceptions.
_CONCERN_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("business", ("主营业务", "主要业务", "经营模式", "业务模式", "产品服务")),
    ("revenue", ("营业收入", "营收", "收入", "销售收入")),
    ("profit", ("净利润", "利润", "盈利", "亏损")),
    ("cash_flow", ("现金流量", "现金流", "经营活动现金", "投资活动现金", "筹资活动现金")),
    ("research", ("研发", "研究开发", "技术创新", "专利")),
    ("employee", ("员工", "职工", "人员构成", "人才")),
    ("customer", ("客户", "客户集中度")),
    ("supplier", ("供应商", "供应商集中度")),
    ("audit_opinion", ("审计意见", "审计报告", "无保留意见", "保留意见")),
    ("auditor", ("会计师事务所", "审计机构", "续聘", "聘任审计")),
    ("governance", ("公司治理", "治理结构", "董事会", "监事会", "股东大会")),
    ("internal_control", ("内部控制", "内控", "重大缺陷")),
    ("litigation", ("诉讼", "仲裁")),
    ("penalty", ("处罚", "整改", "监管措施", "纪律处分")),
    ("delisting", ("退市", "暂停上市", "终止上市")),
    ("bankruptcy", ("破产", "重整")),
    ("contract", ("重大合同", "重要合同", "合同履行")),
    ("related_party", ("关联交易", "关联方交易")),
    ("asset", ("资产", "固定资产", "资产变动", "资产处置")),
    ("liability", ("负债", "偿债", "债务")),
    ("dividend", ("分红", "利润分配", "现金红利", "股利")),
    ("risk", ("风险", "不利因素", "挑战", "风险管控")),
    ("environment", ("环境保护", "环保", "排污", "污染", "碳排放")),
    ("strategy", ("发展战略", "发展规划", "未来计划", "经营计划")),
)

_QUESTION_GENERIC = (
    "股份有限公司", "有限责任公司", "年度报告", "年报", "报告期内", "报告期",
    "本年度", "请问", "请说明", "说明一下", "请介绍", "介绍一下", "概述一下",
    "概述", "介绍", "说明", "有关", "相关", "具体", "主要", "情况", "事项",
    "是什么", "有哪些", "怎么样", "如何", "是否", "公司",
)
_TERM_STOP = frozenset(
    {"股份", "有限", "年度", "报告", "年报", "公司", "本年", "情况", "事项",
     "介绍", "说明", "概述", "相关", "有关", "具体", "主要", "什么", "哪些",
     "如何", "是否", "请问", "报告期", "期内"}
)


@dataclass(frozen=True)
class InformationDecision:
    """Stable classification of one answer or evidence candidate."""

    usable: bool
    reason: str
    concern_score: int
    matched_concerns: tuple[str, ...]

    @property
    def low_information(self) -> bool:
        return not self.usable

    def as_dict(self) -> dict[str, Any]:
        return {
            "usable": self.usable,
            "low_information": self.low_information,
            "reason": self.reason,
            "concern_score": self.concern_score,
            "matched_concerns": list(self.matched_concerns),
        }


@dataclass(frozen=True)
class RankedFrozenChunk:
    """An unchanged retrieved chunk plus deterministic ranking metadata."""

    chunk_id: str
    original_rank: int
    concern_score: int
    matched_concerns: tuple[str, ...]
    chunk: Mapping[str, Any]


def _normalize(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("text values must be strings")
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value)).strip()


def _compact(value: str) -> str:
    return re.sub(r"[\s\W_]+", "", _normalize(value).lower(), flags=re.UNICODE)


def _title_tuple(title_path: Sequence[str]) -> tuple[str, ...]:
    if isinstance(title_path, (str, bytes)):
        raise TypeError("title_path must be a sequence of strings")
    result: list[str] = []
    for part in title_path:
        if not isinstance(part, str):
            raise TypeError("title_path components must be strings")
        normalized = _normalize(part)
        if normalized:
            result.append(normalized)
    return tuple(result)


def _question_terms(question: str) -> tuple[str, ...]:
    value = _YEAR_RE.sub("", _normalize(question).lower())
    for phrase in _QUESTION_GENERIC:
        value = value.replace(phrase, "")
    # Function particles otherwise create accidental bigram matches such as
    # ``地位的`` matching ``单位的`` in an unrelated accounting paragraph.
    value = re.sub(r"[的了着和与及并]", "", value)
    terms: set[str] = set()
    for run in re.findall(r"[\u3400-\u9fffA-Za-z]+", value):
        if re.fullmatch(r"[A-Za-z]+", run):
            if len(run) >= 2:
                terms.add(run.lower())
            continue
        for width in range(2, min(4, len(run)) + 1):
            for start in range(len(run) - width + 1):
                term = run[start : start + width]
                if term not in _TERM_STOP:
                    terms.add(term)
    return tuple(sorted(terms))


def _matched_groups(value: str) -> set[str]:
    compact = _compact(value)
    return {
        group
        for group, aliases in _CONCERN_GROUPS
        if any(_compact(alias) in compact for alias in aliases)
    }


def _concern_match(
    question: str, text: str, title_path: Sequence[str]
) -> tuple[int, tuple[str, ...]]:
    question_groups = _matched_groups(question)
    text_groups = _matched_groups(text)
    title_text = " ".join(title_path)
    title_groups = _matched_groups(title_text)
    matched = question_groups.intersection(text_groups.union(title_groups))
    terms = _question_terms(question)
    compact_text = _compact(text)
    compact_title = _compact(title_text)
    text_term_score = sum(len(term) for term in terms if _compact(term) in compact_text)
    title_term_score = sum(len(term) for term in terms if _compact(term) in compact_title)
    score = len(matched) * 100 + text_term_score * 3 + title_term_score
    return score, tuple(sorted(matched))


def _cells(value: str) -> tuple[str, ...]:
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    cells: list[str] = []
    for line in lines:
        if "|" not in line:
            continue
        cells.extend(part.strip() for part in line.strip("|").split("|") if part.strip())
    return tuple(cells)


def _is_unit_only(value: str) -> bool:
    compact = _normalize(value).strip("| ")
    direct = re.sub(r"^(?:单位|金额单位|计量单位|币种)\s*[:：]?\s*", "", compact)
    if direct != compact and _UNIT_VALUE_RE.fullmatch(direct):
        return True
    cells = _cells(value)
    if not cells:
        return _UNIT_VALUE_RE.fullmatch(compact) is not None
    normalized_cells = tuple(_normalize(cell).rstrip(":：") for cell in cells)
    has_unit = any(_UNIT_VALUE_RE.fullmatch(cell) for cell in normalized_cells)
    return has_unit and all(
        _UNIT_VALUE_RE.fullmatch(cell) is not None
        or cell in {"单位", "金额单位", "计量单位", "币种"}
        or _MARKDOWN_SEPARATOR_RE.fullmatch(cell) is not None
        for cell in normalized_cells
    )


def _is_table_header(value: str) -> bool:
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    pipe_rows = [
        tuple(cell.strip() for cell in line.strip("|").split("|") if cell.strip())
        for line in lines
        if "|" in line
    ]
    separator_indexes = [
        index
        for index, row in enumerate(pipe_rows)
        if row
        and all(
            _MARKDOWN_SEPARATOR_RE.fullmatch(cell.strip()) is not None
            for cell in row
        )
    ]
    if separator_indexes:
        separator_index = separator_indexes[0]
        data_rows = [
            row
            for index, row in enumerate(pipe_rows)
            if index > separator_index and row
        ]
        if data_rows:
            return False
        header_rows = [
            row
            for index, row in enumerate(pipe_rows)
            if index < separator_index and row
        ]
        return not header_rows or all(
            _HEADER_CELL_RE.fullmatch(_normalize(cell)) is not None
            for row in header_rows
            for cell in row
        )
    cells = tuple(
        cell for cell in _cells(value) if _MARKDOWN_SEPARATOR_RE.fullmatch(cell) is None
    )
    if len(cells) >= 2:
        return all(_HEADER_CELL_RE.fullmatch(_normalize(cell)) is not None for cell in cells)
    words = tuple(part for part in re.split(r"[\s、,/]+", _normalize(value)) if part)
    return len(words) >= 2 and all(_HEADER_CELL_RE.fullmatch(word) is not None for word in words)


def _has_explicit_fact(value: str) -> bool:
    normalized = _normalize(value)
    if _EXPLICIT_FACT_PHRASE_RE.search(normalized):
        return True
    if _LABELED_NUMBER_RE.search(normalized):
        return True
    alnum_count = len(re.findall(r"[\u3400-\u9fffA-Za-z0-9]", normalized))
    return alnum_count >= 5 and _FACT_VERB_RE.search(normalized) is not None


def _is_heading_or_title(value: str) -> bool:
    normalized = _normalize(value)
    if _DOCUMENT_TITLE_RE.fullmatch(normalized):
        return True
    if (
        "是否" in normalized
        and len(re.findall(r"[\u3400-\u9fffA-Za-z0-9]", normalized)) <= 60
        and not any(mark in normalized for mark in ("：", ":", "√", "✓", "✔", "☑", "□", "☐"))
        and not any(mark in normalized for mark in ("。", "！", "？", ";", "；"))
    ):
        return True
    if re.fullmatch(r"#{1,6}\s+[^\n]+", value.strip()):
        return not _has_explicit_fact(re.sub(r"^#{1,6}\s+", "", normalized))
    if (
        len(re.findall(r"[\u3400-\u9fffA-Za-z0-9]", normalized)) <= 45
        and not any(mark in normalized for mark in ("。", "！", "？", "；", ";"))
        and _HEADING_END_RE.search(normalized) is not None
        and not any(subject in normalized for subject in ("公司", "董事会", "股东大会", "报告期"))
        and _LABELED_NUMBER_RE.search(normalized) is None
    ):
        return True
    if _has_explicit_fact(normalized):
        return False
    if any(mark in normalized for mark in ("。", "！", "？", "；", ";")):
        return False
    alnum_count = len(re.findall(r"[\u3400-\u9fffA-Za-z0-9]", normalized))
    return alnum_count <= 45 and (
        _HEADING_PREFIX_RE.match(normalized) is not None
        or _HEADING_END_RE.search(normalized) is not None
        or _BARE_TOPIC_HEADING_RE.search(normalized) is not None
    )


def _is_redirect_only(value: str, question: str, title_path: Sequence[str]) -> bool:
    normalized = _normalize(value)
    if _REDIRECT_RE.search(normalized) is None:
        return False
    residuals: list[str] = []
    for clause in re.split(r"[。！？；;,，]", normalized):
        clause = clause.strip()
        if not clause:
            continue
        marker = _REDIRECT_RE.search(clause)
        residuals.append(clause[: marker.start()].strip() if marker else clause)
    for residual in residuals:
        if not residual or _GENERIC_REDIRECT_SUBJECT_RE.fullmatch(residual):
            continue
        # A relevant heading cannot turn a generic ``详见`` lead-in into a
        # substantive fact; relevance must be present in the residual text.
        concern_score, _ = _concern_match(question, residual, ())
        if (
            concern_score > 0
            and _has_explicit_fact(residual)
            and len(re.findall(r"[\u3400-\u9fffA-Za-z0-9]", residual)) >= 6
        ):
            return False
    return True


def _selected_negative_checkbox(value: str) -> bool:
    options = _CHECKBOX_OPTION_RE.findall(_normalize(value))
    if len(options) != 2:
        return False
    selected = [
        re.sub(r"^(?:本年度|本年)", "", _compact(label))
        for symbol, label in options
        if symbol in _SELECTED
    ]
    unselected = [
        re.sub(r"^(?:本年度|本年)", "", _compact(label))
        for symbol, label in options
        if symbol in _UNSELECTED
    ]
    return selected == ["不适用"] and unselected == ["适用"]


def classify_candidate(
    text: str,
    *,
    question: str,
    title_path: Sequence[str] = (),
    scoped_checkbox_negative: bool = False,
) -> InformationDecision:
    """Classify candidate text without modifying or augmenting its claims.

    A bare negative marker becomes usable only when its title path shares a
    concern with the question, or when the caller has already verified the
    checkbox's leaf-section scope and passes ``scoped_checkbox_negative``.
    """

    normalized = _normalize(text)
    path = _title_tuple(title_path)
    score, matched = _concern_match(question, normalized, path)
    if not normalized:
        return InformationDecision(False, "empty", score, matched)

    compact = _compact(normalized)
    negative_marker = _selected_negative_checkbox(normalized) or (
        _STANDALONE_NEGATIVE_RE.fullmatch(compact) is not None
    )
    if negative_marker and (scoped_checkbox_negative or score > 0):
        return InformationDecision(True, "scoped_checkbox_negative", score, matched)

    if _is_redirect_only(normalized, question, path):
        return InformationDecision(False, "redirect_only", score, matched)
    if _is_unit_only(text):
        return InformationDecision(False, "unit_only", score, matched)
    if _is_table_header(text):
        return InformationDecision(False, "table_header", score, matched)
    if _is_heading_or_title(text):
        return InformationDecision(False, "heading_or_document_title", score, matched)
    if any(pattern.search(normalized) for pattern in _BOILERPLATE_RES):
        return InformationDecision(False, "audit_or_governance_boilerplate", score, matched)

    if _has_explicit_fact(normalized):
        return InformationDecision(True, "explicit_fact", score, matched)
    alnum_count = len(re.findall(r"[\u3400-\u9fffA-Za-z0-9]", normalized))
    if alnum_count < 12 and score <= 0:
        return InformationDecision(False, "short_without_question_concern", score, matched)
    if negative_marker:
        return InformationDecision(False, "unscoped_negative", score, matched)
    return InformationDecision(True, "substantive", score, matched)


def classify_answer(
    answer: str,
    *,
    question: str,
    title_paths: Sequence[Sequence[str]] = (),
) -> InformationDecision:
    """Classify a rendered answer using only its cited title-path context."""

    flattened: list[str] = []
    if isinstance(title_paths, (str, bytes)):
        raise TypeError("title_paths must be a sequence of title paths")
    for path in title_paths:
        flattened.extend(_title_tuple(path))
    return classify_candidate(answer, question=question, title_path=flattened)


def is_low_information(
    text: str,
    *,
    question: str,
    title_path: Sequence[str] = (),
    scoped_checkbox_negative: bool = False,
) -> bool:
    """Convenience predicate backed by :func:`classify_candidate`."""

    return classify_candidate(
        text,
        question=question,
        title_path=title_path,
        scoped_checkbox_negative=scoped_checkbox_negative,
    ).low_information


def _chunk_id(chunk: Mapping[str, Any]) -> str:
    for field in ("chunk_id", "evidence_chunk_id", "a2rag_chunk_id"):
        value = chunk.get(field)
        if isinstance(value, str) and value:
            return value
    raise ValueError("every frozen chunk must have a stable chunk identity")


def rank_frozen_chunks(
    retrieved_top_k: Sequence[Mapping[str, Any]],
    *,
    question: str,
    title_path_compatibility: Mapping[str, bool],
    expected_document_id: str | None = None,
) -> tuple[RankedFrozenChunk, ...]:
    """Filter and stably rerank an already-frozen, single-document top-k.

    Compatibility is keyed by stable chunk identity and must be present for
    every input.  ``False`` fails closed.  Returned ``chunk`` values are the
    exact input mappings: the function neither edits text nor constructs a
    claim, so financial renderings cannot be introduced here.
    """

    if isinstance(retrieved_top_k, (str, bytes)):
        raise TypeError("retrieved_top_k must be a sequence of chunk mappings")
    if not isinstance(title_path_compatibility, Mapping):
        raise TypeError("title_path_compatibility must be a mapping")
    if expected_document_id is not None and (
        not isinstance(expected_document_id, str) or not expected_document_id
    ):
        raise ValueError("expected_document_id must be a non-empty string")

    rows: list[tuple[tuple[Any, ...], RankedFrozenChunk]] = []
    seen: set[str] = set()
    frozen_document_id = expected_document_id
    for ordinal, chunk in enumerate(retrieved_top_k):
        if not isinstance(chunk, Mapping):
            raise TypeError("frozen chunks must be mappings")
        chunk_id = _chunk_id(chunk)
        if chunk_id in seen:
            raise ValueError("frozen top-k contains duplicate chunk identities")
        seen.add(chunk_id)
        if chunk_id not in title_path_compatibility:
            raise ValueError(f"missing title-path compatibility for chunk {chunk_id}")
        compatible = title_path_compatibility[chunk_id]
        if type(compatible) is not bool:
            raise TypeError("title-path compatibility values must be bool")

        document_id = chunk.get("document_id")
        if not isinstance(document_id, str) or not document_id:
            raise ValueError("every frozen chunk must have a document_id")
        if frozen_document_id is None:
            frozen_document_id = document_id
        if document_id != frozen_document_id:
            raise ValueError("frozen top-k crosses document scope")

        content = chunk.get("content")
        if not isinstance(content, str):
            raise TypeError("every frozen chunk must have string content")
        raw_path = chunk.get("section_path", ())
        path = _title_tuple(raw_path)
        decision = classify_candidate(content, question=question, title_path=path)
        if not compatible or decision.low_information:
            continue

        content_score, content_groups = _concern_match(question, content, ())
        path_score, path_groups = _concern_match(question, "", path)
        matched = tuple(sorted(set(content_groups).union(path_groups)))
        concern_score = content_score * 3 + path_score
        ranked = RankedFrozenChunk(
            chunk_id=chunk_id,
            original_rank=ordinal,
            concern_score=concern_score,
            matched_concerns=matched,
            chunk=chunk,
        )
        rows.append(((-content_score, -path_score, ordinal, chunk_id), ranked))

    rows.sort(key=lambda row: row[0])
    return tuple(row[1] for row in rows)


def rerank_frozen_chunks(
    retrieved_top_k: Sequence[Mapping[str, Any]],
    *,
    question: str,
    title_path_compatibility: Mapping[str, bool],
    expected_document_id: str | None = None,
    max_chunks: int | None = None,
) -> tuple[Mapping[str, Any], ...]:
    """Return unchanged usable chunks in stable question-concern order."""

    if max_chunks is not None and (type(max_chunks) is not int or max_chunks < 0):
        raise ValueError("max_chunks must be a non-negative integer or None")
    ranked = rank_frozen_chunks(
        retrieved_top_k,
        question=question,
        title_path_compatibility=title_path_compatibility,
        expected_document_id=expected_document_id,
    )
    if max_chunks is not None:
        ranked = ranked[:max_chunks]
    return tuple(row.chunk for row in ranked)


__all__ = [
    "InformationDecision",
    "LOWINFO_VERSION",
    "RankedFrozenChunk",
    "classify_answer",
    "classify_candidate",
    "is_low_information",
    "rank_frozen_chunks",
    "rerank_frozen_chunks",
]
