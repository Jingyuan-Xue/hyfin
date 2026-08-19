"""Opt-in deterministic Type 3-1 evidence enhancement experiment.

The frozen Phase 8/10 pipeline remains the source of the baseline answer.  This
module may add only document-scoped evidence recovered from explicit Markdown
applicability controls, audited table fragments, or a complete-document
absence audit.  It deliberately has no GeneratorPort and never reads benchmark
answers, prompt annotations, or scoring keywords.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from typing import Any, Iterable, Mapping, Sequence

from .contracts import semantic_sha256
from .table_evidence import TableEvidenceIndex, canonical_json


TYPE3_V7_VERSION = "type3-v7-deterministic-evidence-v1"
MAX_TEXT_CANDIDATES = 50
MAX_TABLE_CANDIDATES = 50
MAX_CLAIM_GROUPS = 5
MAX_TABLE_ROWS_PER_GROUP = 12
MAX_ANSWER_CHARS = 1500
MAX_FRAME_SOURCE_CHARS = 175

STAGE_FEATURES: tuple[tuple[str, frozenset[str]], ...] = (
    ("baseline", frozenset()),
    ("checkbox", frozenset({"checkbox"})),
    ("table_text", frozenset({"checkbox", "table"})),
    ("table_numeric", frozenset({"checkbox", "table", "numeric"})),
    ("faceted_frame", frozenset({"checkbox", "table", "numeric", "frame"})),
    (
        "document_absence",
        frozenset({"checkbox", "table", "numeric", "frame", "absence"}),
    ),
    ("full", frozenset({"checkbox", "table", "numeric", "frame", "absence"})),
)

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_SECTION_HEADING_RE = re.compile(r"^第[^\s]{1,12}节(?:\s|$)")
_CHINESE_ORDINAL_HEADING_RE = re.compile(
    r"^[一二三四五六七八九十百零〇]{1,5}[、.]\s*\S"
)
_CHINESE_PAREN_HEADING_RE = re.compile(
    r"^[（(][一二三四五六七八九十百零〇]{1,5}[）)]\s*\S"
)
_ARABIC_ORDINAL_HEADING_RE = re.compile(r"^\d{1,3}[、.]\s*\S")
_ARABIC_PAREN_HEADING_RE = re.compile(r"^[（(]\d{1,3}[）)]\s*\S")
_SELECTED = frozenset("√✓✔☑■")
_UNSELECTED = frozenset("□☐○◯")
_CHECKBOX_OPTION_RE = re.compile(
    r"([□☐○◯■☑√✓✔])\s*((?:本年度|本年)?\s*(?:适用|不适用))"
)
_YEAR_RE = re.compile(r"(?<!\d)((?:19|20|21)\d{2})\s*年?")
_DATE_VALUE_RE = re.compile(
    r"^(?:(?:19|20|21)\d{2}(?:年|[-/.])\d{1,2}(?:(?:月|[-/.])\d{1,2}日?)?|"
    r"(?:19|20|21)\d{2}年)$"
)
_DECIMAL_RE = re.compile(r"^[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?[%％]?$")
_NUMERIC_LIKE_RE = re.compile(r"^[+\-\d\s,.%％/]+$")
_HEADER_ROW_LABELS = frozenset(
    {"项目", "项目名称", "科目", "事项概述", "协议", "主要会计数据", "col1"}
)
_GENERIC_PHRASES = (
    "股份有限公司", "有限责任公司", "年度报告", "年报", "报告期内", "本公司",
    "概述一下", "概述", "介绍一下", "介绍", "请问", "请说明", "说明一下",
    "针对", "根据", "分别", "相关", "主要", "具体", "情况", "详情", "是什么",
    "有哪些", "怎么样", "如何", "一下", "公司", "数据", "简要", "分析",
)
_TERM_STOP = frozenset(
    {"股份", "有限", "报告", "年度", "年报", "公司", "情况", "概述", "介绍", "说明",
     "什么", "哪些", "如何", "一下", "针对", "根据", "分别", "相关", "主要", "具体",
     "报告期", "期内", "是否", "详情", "数据", "简要", "分析"}
)

# These are annual-report concern synonyms, not benchmark cases.  They may
# bridge a question wording (for example ``退市``) to a conventional section
# title (``暂停上市和终止上市``), but never name a company, year, or answer.
_CONCERN_GROUPS: tuple[tuple[str, tuple[str, ...], bool, bool], ...] = (
    ("bankruptcy", ("破产重整", "破产", "重整"), True, False),
    ("delisting", ("暂停上市和终止上市", "终止上市", "暂停上市", "退市"), True, False),
    ("penalty", ("处罚及整改", "行政处罚", "监管措施", "纪律处分", "处罚", "整改"), True, False),
    ("litigation", ("重大诉讼仲裁", "重大诉讼", "重大仲裁", "诉讼", "仲裁"), True, False),
    ("major_contract", ("重大合同及其履行", "重大合同", "重要合同", "合同履行"), True, True),
    ("related_transaction", ("重大关联交易", "日常关联交易", "关联交易", "关联方交易"), False, True),
    ("asset_liability", ("资产及负债状况", "资产负债状况", "资产构成重大变动", "负债构成重大变动"), False, True),
    ("cash_flow", ("现金流量", "现金流", "经营活动", "投资活动", "筹资活动"), False, True),
    ("customer", ("主要客户", "客户集中度", "客户"), False, True),
    ("supplier", ("主要供应商", "供应商集中度", "供应商"), False, True),
    ("research", ("研发投入", "研发人员", "研究开发", "研发"), False, True),
    ("employee", ("员工构成", "员工情况", "职工人数", "人员构成", "员工", "职工"), False, True),
    ("audit", ("关键审计事项", "审计意见", "会计师事务所", "审计"), False, False),
    ("environment", ("环境信息", "环境保护", "排污", "环保"), False, True),
    ("social", ("社会责任", "扶贫", "公益"), False, True),
    ("management", ("董事监事高级管理人员", "董监高", "管理人员", "董事", "监事"), False, True),
    ("business", ("主营业务", "主要业务", "经营模式", "业务模式"), False, False),
    ("competitiveness", ("核心竞争力", "核心优势", "竞争优势"), False, False),
    ("risk", ("风险因素", "经营风险", "主要风险", "风险及对策"), False, False),
)


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(value))).strip()


def _compact(value: str) -> str:
    return re.sub(r"[\s\W_]+", "", _normalize(value).lower(), flags=re.UNICODE)


def _question_terms(question: str, company: str = "", stock_code: str = "") -> set[str]:
    value = _normalize(question).lower()
    for item in (company, stock_code):
        if item:
            value = value.replace(_normalize(item).lower(), "")
    value = _YEAR_RE.sub("", value)
    for phrase in _GENERIC_PHRASES:
        value = value.replace(phrase, "")
    terms: set[str] = set()
    for run in re.findall(r"[\u3400-\u9fffA-Za-z]+", value):
        for width in range(2, min(4, len(run)) + 1):
            for start in range(len(run) - width + 1):
                term = run[start : start + width]
                if term not in _TERM_STOP:
                    terms.add(term)
    return terms


def _matched_groups(value: str) -> set[str]:
    compact = _compact(value)
    result = {
        group_id
        for group_id, aliases, _, _ in _CONCERN_GROUPS
        if any(_compact(alias) in compact for alias in aliases)
    }
    # Valuation sections often contain ``预计未来现金流量现值``.  That is a
    # discounted-cash-flow assumption, not the annual-report cash-flow
    # statement or cash-flow analysis requested by a broad cash-flow question.
    if "cash_flow" in result and any(
        marker in compact for marker in ("商誉", "减值测试", "未来现金流量现值")
    ):
        result.remove("cash_flow")
    return result


def _question_groups(question: str) -> set[str]:
    return _matched_groups(question)


def _group_aliases(group_ids: Iterable[str]) -> tuple[str, ...]:
    wanted = set(group_ids)
    return tuple(
        alias
        for group_id, aliases, _, _ in _CONCERN_GROUPS
        if group_id in wanted
        for alias in aliases
    )


def _negative_capable(group_ids: Iterable[str]) -> bool:
    wanted = set(group_ids)
    return any(group_id in wanted and negative for group_id, _, negative, _ in _CONCERN_GROUPS)


def _table_capable(group_ids: Iterable[str]) -> bool:
    wanted = set(group_ids)
    return any(group_id in wanted and table for group_id, _, _, table in _CONCERN_GROUPS)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _outline_level(value: str, markdown_level: int | None = None) -> int | None:
    """Infer annual-report hierarchy despite flattened Markdown headings."""

    title = _normalize(value)
    if not title or len(title) > 80:
        return None
    if title.startswith(("|", "<", "□", "√", "✓", "✔", "☑")):
        return None
    if any(mark in title for mark in ("。", "；", ";")):
        return None
    if _SECTION_HEADING_RE.match(title):
        return 1
    if _CHINESE_ORDINAL_HEADING_RE.match(title):
        return 2
    if _CHINESE_PAREN_HEADING_RE.match(title):
        return 3
    if _ARABIC_ORDINAL_HEADING_RE.match(title):
        return 4
    if _ARABIC_PAREN_HEADING_RE.match(title):
        return 5
    if markdown_level is not None:
        return 10 + markdown_level
    return None


@dataclass(frozen=True)
class EvidenceCandidate:
    candidate_id: str
    source_kind: str
    document_id: str
    company: str
    report_year: int
    heading_path: tuple[str, ...]
    source_markdown: str
    line_range: tuple[int, int]
    raw_text: str
    source_ordinal: int
    topic_groups: tuple[str, ...]
    heading_group_hits: int
    topic_anchor_score: int
    source_payload: Mapping[str, Any]

    def sort_key(self) -> tuple[Any, ...]:
        return (
            -self.heading_group_hits,
            -self.topic_anchor_score,
            self.source_ordinal,
            self.candidate_id,
        )


@dataclass(frozen=True)
class _SourceDocument:
    path: Path
    portable_path: str
    lines: tuple[str, ...]
    heading_paths: tuple[tuple[str, ...], ...]
    sha256: str

    def heading_path(self, line_number: int) -> tuple[str, ...]:
        if not self.heading_paths:
            return ()
        index = max(0, min(line_number - 1, len(self.heading_paths) - 1))
        return self.heading_paths[index]


class DocumentAuditIndex:
    """Lazy, hash-recorded source Markdown view for headings and absence scans."""

    def __init__(
        self,
        *,
        root: str | Path,
        document_map_path: str | Path | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        map_path = (
            Path(document_map_path).resolve()
            if document_map_path is not None
            else self.root / "data/indexes/a2rag_index/document_chunk_map.jsonl"
        )
        self._sources: dict[str, str] = {}
        with map_path.open("r", encoding="utf-8") as handle:
            for raw in handle:
                if not raw.strip():
                    continue
                row = json.loads(raw)
                document_id = str(row.get("document_id") or "")
                source = str(row.get("source_markdown") or "")
                if not document_id or not source or document_id in self._sources:
                    raise ValueError("document chunk map has an invalid source identity")
                self._sources[document_id] = source
        self._cache: dict[str, _SourceDocument] = {}

    def source(self, document_id: str) -> _SourceDocument:
        cached = self._cache.get(document_id)
        if cached is not None:
            return cached
        portable = self._sources.get(document_id)
        if portable is None:
            raise KeyError(f"unknown document_id: {document_id}")
        path = Path(portable)
        if path.is_absolute():
            try:
                portable = path.resolve().relative_to(self.root).as_posix()
            except ValueError as exc:
                raise ValueError("document source is outside the workspace") from exc
        else:
            path = self.root / path
        if not path.is_file():
            raise FileNotFoundError(path)
        lines = tuple(path.read_text(encoding="utf-8").splitlines())
        stack: list[tuple[int, str]] = []
        paths: list[tuple[str, ...]] = []
        for line in lines:
            match = _HEADING_RE.match(line)
            title = _normalize(match.group(2)) if match else _normalize(line)
            level = _outline_level(title, len(match.group(1)) if match else None)
            if level is not None:
                stack = [item for item in stack if item[0] < level]
                stack.append((level, title))
            paths.append(tuple(item[1] for item in stack))
        document = _SourceDocument(
            path=path,
            portable_path=portable,
            lines=lines,
            heading_paths=tuple(paths),
            sha256=_sha256_file(path),
        )
        self._cache[document_id] = document
        return document

    def scoped_negative_candidates(
        self,
        *,
        document_id: str,
        company: str,
        report_year: int,
        question: str,
    ) -> list[EvidenceCandidate]:
        groups = _question_groups(question)
        if not groups or not _negative_capable(groups):
            return []
        document = self.source(document_id)
        result: list[EvidenceCandidate] = []
        for index, line in enumerate(document.lines):
            options = _CHECKBOX_OPTION_RE.findall(_normalize(line))
            if len(options) != 2:
                continue
            selected = [label for symbol, label in options if symbol in _SELECTED]
            unselected = [label for symbol, label in options if symbol in _UNSELECTED]
            if len(selected) != 1 or len(unselected) != 1:
                continue
            if _compact(selected[0]) != "不适用" or _compact(unselected[0]) != "适用":
                continue
            heading_path = document.heading_path(index + 1)
            # A selected control applies only to its leaf section.  An
            # ``不适用`` marker under ``租赁情况`` must not negate its ancestor
            # ``重大合同及其履行情况`` section.
            heading_groups = _matched_groups(heading_path[-1] if heading_path else "")
            group_hits = len(groups.intersection(heading_groups))
            if not group_hits:
                continue
            # If both an ancestor and the leaf match the concern, the leaf is
            # a narrower subtopic (for example ``其他重大合同`` beneath
            # ``重大合同及其履行情况``).  Its control cannot negate the broad
            # ancestor requested by the user.
            if any(
                groups.intersection(_matched_groups(title))
                for title in heading_path[:-1]
            ):
                continue
            raw_text = _normalize(line)
            candidate_id = "v7-neg-" + semantic_sha256(
                [TYPE3_V7_VERSION, document_id, index + 1, raw_text]
            )[:24]
            result.append(EvidenceCandidate(
                candidate_id=candidate_id,
                source_kind="text_checkbox_negative",
                document_id=document_id,
                company=company,
                report_year=report_year,
                heading_path=heading_path,
                source_markdown=document.portable_path,
                line_range=(index + 1, index + 1),
                raw_text=raw_text,
                source_ordinal=index + 1,
                topic_groups=tuple(sorted(heading_groups)),
                heading_group_hits=group_hits,
                topic_anchor_score=group_hits,
                source_payload={"document_sha256": document.sha256},
            ))
        result.sort(key=EvidenceCandidate.sort_key)
        return result[:MAX_TEXT_CANDIDATES]

    def absence_audit(
        self,
        *,
        document_id: str,
        question: str,
    ) -> dict[str, Any] | None:
        groups = _question_groups(question)
        if not groups:
            return None
        aliases = tuple(dict.fromkeys(_group_aliases(groups)))
        document = self.source(document_id)
        normalized_source = _compact("\n".join(document.lines))
        hit_counts = {
            alias: normalized_source.count(_compact(alias))
            for alias in aliases
            if _compact(alias)
        }
        if any(hit_counts.values()):
            return None
        audit = {
            "schema_version": "finglmqa.experimental.document_absence_audit.v1",
            "document_id": document_id,
            "source_markdown": document.portable_path,
            "document_sha256": document.sha256,
            "line_count": len(document.lines),
            "channels_scanned": ["markdown_headings", "markdown_body", "embedded_html_tables"],
            "topic_groups": sorted(groups),
            "searched_aliases": list(aliases),
            "alias_hit_counts": hit_counts,
            "complete_scan": True,
        }
        audit["audit_id"] = "v7-absence-" + semantic_sha256(audit)[:24]
        return audit


def _is_header_fragment(fragment: Mapping[str, Any]) -> bool:
    label = _compact(str(fragment.get("row_label") or ""))
    values = [_compact(str(value)) for value in fragment.get("raw_cell_values") or []]
    labels = [_compact(str(value)) for value in fragment.get("column_labels") or []]
    if label in {_compact(value) for value in _HEADER_ROW_LABELS}:
        return not values or values == labels or all(value in labels for value in values)
    return bool(values and labels and values == labels)


class V7TableRetriever:
    """Deterministic document-scoped ranking over audited table fragments."""

    def __init__(self, index: TableEvidenceIndex, source_index: DocumentAuditIndex) -> None:
        self.index = index
        self.source_index = source_index
        self._document_cache: dict[str, tuple[dict[str, Any], ...]] = {}
        self._search_cache: dict[tuple[str, str, str], tuple[EvidenceCandidate, ...]] = {}

    def _document_rows(self, document_id: str) -> tuple[dict[str, Any], ...]:
        if document_id not in self._document_cache:
            self._document_cache[document_id] = tuple(self.index.iter_document(document_id))
        return self._document_cache[document_id]

    def search(
        self,
        *,
        document_id: str,
        company: str,
        stock_code: str,
        report_year: int,
        question: str,
    ) -> list[EvidenceCandidate]:
        key = (document_id, question, company)
        cached = self._search_cache.get(key)
        if cached is not None:
            return list(cached)
        question_groups = _question_groups(question)
        if question_groups and not _table_capable(question_groups):
            self._search_cache[key] = ()
            return []
        question_terms = _question_terms(question, company=company, stock_code=stock_code)
        source = self.source_index.source(document_id)
        candidates: list[EvidenceCandidate] = []
        for fragment in self._document_rows(document_id):
            if fragment["fragment_kind"] != "table_row" or _is_header_fragment(fragment):
                continue
            line_number = int(fragment["source_line_range"][0])
            source_path = source.heading_path(line_number)
            existing_path = tuple(str(value) for value in fragment["section_path"] if str(value).strip())
            heading_path = tuple(dict.fromkeys((*source_path, *existing_path)))
            heading_text = " ".join(
                [*heading_path, str(fragment["caption"]), *map(str, fragment["header_path"])]
            )
            content_text = " ".join(
                [str(fragment["row_label"]), *map(str, fragment["column_labels"]), str(fragment["content"])]
            )
            heading_groups = _matched_groups(heading_text)
            content_groups = _matched_groups(content_text)
            group_hits = len(question_groups.intersection(heading_groups))
            content_group_hits = len(question_groups.intersection(content_groups))
            normalized_heading = _compact(heading_text)
            normalized_content = _compact(content_text)
            heading_term_score = sum(len(term) for term in question_terms if _compact(term) in normalized_heading)
            content_term_score = sum(len(term) for term in question_terms if _compact(term) in normalized_content)
            topic_anchor_score = group_hits * 100 + content_group_hits * 25 + heading_term_score * 3 + content_term_score
            if topic_anchor_score <= 0:
                continue
            candidate_id = str(fragment["fragment_id"])
            candidates.append(EvidenceCandidate(
                candidate_id=candidate_id,
                source_kind="table_row",
                document_id=document_id,
                company=company,
                report_year=report_year,
                heading_path=heading_path,
                source_markdown=str(fragment["source_markdown"]),
                line_range=tuple(fragment["source_line_range"]),
                raw_text=str(fragment["content"]),
                source_ordinal=int(fragment["table_index"]) * 100000 + int(fragment["row_index"]),
                topic_groups=tuple(sorted(heading_groups.union(content_groups))),
                heading_group_hits=group_hits,
                topic_anchor_score=topic_anchor_score,
                source_payload=dict(fragment),
            ))
        # When the Markdown hierarchy exposes an exact concern section, rows
        # that match only an incidental word in content cannot compete with
        # that section.  The content fallback remains available for malformed
        # documents whose source has no matching heading at all.
        if question_groups and any(row.heading_group_hits for row in candidates):
            candidates = [row for row in candidates if row.heading_group_hits]
        candidates.sort(key=EvidenceCandidate.sort_key)
        selected = tuple(candidates[:MAX_TABLE_CANDIDATES])
        self._search_cache[key] = selected
        return list(selected)


def _unit_from_fragment(fragment: Mapping[str, Any], column_label: str, raw_value: str) -> str | None:
    label = _compact(column_label)
    if raw_value.rstrip().endswith(("%", "％")) or any(token in label for token in ("比例", "比率", "率%", "率％")):
        return "%"
    source = fragment.get("unit_source")
    if not isinstance(source, Mapping):
        return None
    value = _normalize(str(source.get("value") or ""))
    match = re.search(r"[:：]\s*(?:人民币)?\s*([^\s币]+)", value)
    return match.group(1) if match else None


def _year_from_column(report_year: int, column_label: str) -> int | None:
    years = {int(value) for value in _YEAR_RE.findall(column_label)}
    if len(years) == 1:
        return next(iter(years))
    compact = _compact(column_label)
    if any(value in compact for value in ("上年", "上期", "期初")):
        return report_year - 1
    if any(value in compact for value in ("本年", "本期", "期末", "发生额", "变动", "金额", "数")):
        return report_year
    return None


def _numeric_authorization(
    candidate: EvidenceCandidate,
    *,
    column_ordinal: int,
    column_label: str,
    raw_value: str,
) -> dict[str, Any] | None:
    compact_value = re.sub(r"\s+", "", _normalize(raw_value))
    if _DATE_VALUE_RE.fullmatch(compact_value) or not _DECIMAL_RE.fullmatch(compact_value):
        return None
    fragment = candidate.source_payload
    unit = _unit_from_fragment(fragment, column_label, compact_value)
    metric_year = _year_from_column(candidate.report_year, column_label)
    if unit is None or metric_year is None:
        return None
    decimal_text = compact_value.rstrip("%％").replace(",", "")
    try:
        value = Decimal(decimal_text)
    except InvalidOperation:
        return None
    if not value.is_finite():
        return None
    coordinates = fragment.get("cell_coordinates") or []
    coordinate = coordinates[column_ordinal] if column_ordinal < len(coordinates) else None
    if not isinstance(coordinate, list) or len(coordinate) != 2:
        return None
    normalized_value = format(value, "f")
    allowed_renderings = [compact_value]
    if unit == "%" and not compact_value.endswith(("%", "％")):
        allowed_renderings.append(f"{compact_value}%")
    elif unit != "%" and not compact_value.endswith(unit):
        allowed_renderings.append(f"{compact_value}{unit}")
    authorization = {
        "schema_version": "finglmqa.experimental.table_numeric_authorization.v1",
        "document_id": candidate.document_id,
        "company": candidate.company,
        "report_year": candidate.report_year,
        "metric_year": metric_year,
        "table_id": str(fragment["table_id"]),
        "table_content_sha256": str(fragment["provenance"]["table_content_sha256"]),
        "fragment_id": candidate.candidate_id,
        "cell_coordinate": list(coordinate),
        "column_label": column_label,
        "raw_value": raw_value,
        "raw_value_sha256": hashlib.sha256(raw_value.encode("utf-8")).hexdigest(),
        "normalized_value": normalized_value,
        "normalized_unit": unit,
        "source_markdown": candidate.source_markdown,
        "source_line_range": list(candidate.line_range),
        "allowed_renderings": sorted(set(allowed_renderings)),
    }
    authorization["authorization_id"] = "v7-table-auth-" + semantic_sha256(authorization)[:24]
    return authorization


def _text_cell(value: str) -> bool:
    compact = re.sub(r"\s+", "", _normalize(value))
    return (
        bool(compact)
        and not _DECIMAL_RE.fullmatch(compact)
        and not _DATE_VALUE_RE.fullmatch(compact)
        and not _NUMERIC_LIKE_RE.fullmatch(compact)
    )


def _preferred_cell_ordinals(fragment: Mapping[str, Any], question: str) -> list[int]:
    labels = [str(value) for value in fragment.get("column_labels") or []]
    values = [str(value) for value in fragment.get("raw_cell_values") or []]
    row_label = _normalize(str(fragment.get("row_label") or ""))
    terms = _question_terms(question)
    scored: list[tuple[tuple[int, int, int], int]] = []
    for ordinal, (label, value) in enumerate(zip(labels, values)):
        if _normalize(value) == row_label:
            continue
        compact_label = _compact(label)
        direct = sum(len(term) for term in terms if _compact(term) in compact_label)
        descriptive = int(any(token in compact_label for token in ("说明", "原因", "概述", "内容", "类型", "关系", "是否")))
        current = int(
            any(token in compact_label for token in ("本期", "期末", "发生额", "变动比例", "金额"))
            or bool(_YEAR_RE.search(label))
        )
        text_priority = int(_text_cell(value))
        scored.append(((-direct, -descriptive - text_priority, -current, ordinal), ordinal))
    scored.sort(key=lambda item: item[0])
    return [ordinal for _, ordinal in scored[:4]]


def _render_table_row(
    candidate: EvidenceCandidate,
    *,
    question: str,
    include_numeric: bool,
) -> tuple[str, list[dict[str, Any]]]:
    fragment = candidate.source_payload
    label = _normalize(str(fragment.get("row_label") or ""))
    labels = [str(value) for value in fragment.get("column_labels") or []]
    values = [str(value) for value in fragment.get("raw_cell_values") or []]
    parts: list[str] = []
    authorizations: list[dict[str, Any]] = []
    for ordinal in _preferred_cell_ordinals(fragment, question):
        if ordinal >= len(values):
            continue
        column_label = _normalize(labels[ordinal]) if ordinal < len(labels) else ""
        raw_value = _normalize(values[ordinal])
        if not raw_value or raw_value == label:
            continue
        if _DATE_VALUE_RE.fullmatch(re.sub(r"\s+", "", raw_value)):
            parts.append(f"{column_label}={raw_value}" if column_label else raw_value)
            continue
        authorization = _numeric_authorization(
            candidate,
            column_ordinal=ordinal,
            column_label=column_label,
            raw_value=values[ordinal],
        )
        if authorization is not None:
            if not include_numeric:
                continue
            authorizations.append(authorization)
            renderings = authorization["allowed_renderings"]
            rendered = next(
                (value for value in renderings if value.endswith(authorization["normalized_unit"])),
                renderings[0],
            )
            parts.append(f"{column_label}={rendered}" if column_label else rendered)
        elif _text_cell(raw_value):
            parts.append(f"{column_label}={raw_value}" if column_label else raw_value)
    if not label:
        label = _normalize(str(fragment.get("caption") or ""))
    rendered = f"{label}：" + "；".join(parts) if parts else label
    return rendered.strip("：； "), authorizations


def _candidate_citation(candidate: EvidenceCandidate) -> dict[str, Any]:
    payload = {
        "source_kind": candidate.source_kind,
        "candidate_id": candidate.candidate_id,
        "document_id": candidate.document_id,
        "source_markdown": candidate.source_markdown,
        "line_range": list(candidate.line_range),
        "heading_path": list(candidate.heading_path),
        "content_sha256": hashlib.sha256(candidate.raw_text.encode("utf-8")).hexdigest(),
    }
    payload["citation_id"] = "v7-cite-" + semantic_sha256(payload)[:24]
    return payload


class Type3V7Enhancer:
    """Enhance a frozen v4 answer without replacing safe baseline claims."""

    def __init__(self, *, root: str | Path, table_index: TableEvidenceIndex) -> None:
        self.root = Path(root).resolve()
        self.source_index = DocumentAuditIndex(root=self.root)
        self.table_retriever = V7TableRetriever(table_index, self.source_index)

    @staticmethod
    def _base_heading(answer: Mapping[str, Any], question: str) -> str | None:
        groups = _question_groups(question)
        ranked: list[tuple[int, str]] = []
        for citation in answer.get("citations") or []:
            provenance = citation.get("provenance") if isinstance(citation, Mapping) else None
            path = provenance.get("section_path") if isinstance(provenance, Mapping) else None
            if not isinstance(path, list):
                continue
            for heading in path:
                if not isinstance(heading, str) or not heading.strip():
                    continue
                hits = len(groups.intersection(_matched_groups(heading)))
                if hits:
                    ranked.append((-hits, _normalize(heading)))
        return sorted(ranked)[0][1] if ranked else None

    @staticmethod
    def _frame_answer(
        answer_text: str,
        *,
        company: str,
        report_year: int,
        heading: str | None,
    ) -> str:
        if not answer_text or heading is None:
            return answer_text
        prefix = f"根据{report_year}年{company}年报“{heading}”披露："
        compact_head = _compact(answer_text[:120])
        if _compact(company) in compact_head and str(report_year) in answer_text[:120]:
            return answer_text
        if answer_text.startswith(prefix):
            return answer_text
        return prefix + answer_text

    def _table_groups(
        self,
        scope: Mapping[str, Any],
        *,
        include_numeric: bool,
    ) -> list[dict[str, Any]]:
        candidates = self.table_retriever.search(
            document_id=scope["document_id"],
            company=scope["company"],
            stock_code=scope["stock_code"],
            report_year=scope["report_year"],
            question=scope["question"],
        )
        if not candidates:
            return []
        by_table: dict[str, list[EvidenceCandidate]] = defaultdict(list)
        for candidate in candidates:
            by_table[str(candidate.source_payload["table_id"])].append(candidate)
        table_order = sorted(
            by_table,
            key=lambda table_id: (by_table[table_id][0].sort_key(), table_id),
        )
        result: list[dict[str, Any]] = []
        for table_id in table_order[:2]:
            rows = sorted(by_table[table_id], key=EvidenceCandidate.sort_key)
            rendered_rows: list[str] = []
            authorizations: list[dict[str, Any]] = []
            citations: list[dict[str, Any]] = []
            used = 0
            for candidate in rows:
                row_text, row_authorizations = _render_table_row(
                    candidate,
                    question=scope["question"],
                    include_numeric=include_numeric,
                )
                if not row_text or row_text in rendered_rows:
                    continue
                if used + len(row_text) > 1100 and rendered_rows:
                    break
                rendered_rows.append(row_text)
                authorizations.extend(row_authorizations)
                citations.append(_candidate_citation(candidate))
                used += len(row_text)
                if len(rendered_rows) >= MAX_TABLE_ROWS_PER_GROUP:
                    break
            if not rendered_rows:
                continue
            heading = next(
                (value for value in reversed(rows[0].heading_path) if _matched_groups(value)),
                rows[0].heading_path[-1] if rows[0].heading_path else "表格披露",
            )
            text = f"{heading}：" + "；".join(rendered_rows)
            result.append({
                "group_id": "v7-table-group-" + semantic_sha256([table_id, text])[:20],
                "source_kind": "table",
                "text": text,
                "citations": citations,
                "numeric_authorizations": sorted(
                    {row["authorization_id"]: row for row in authorizations}.values(),
                    key=lambda row: row["authorization_id"],
                ),
                "candidate_ids": [row.candidate_id for row in rows[:len(rendered_rows)]],
                "heading": heading,
            })
        return result

    def prepare(
        self,
        *,
        scope: Mapping[str, Any],
        base_answer: Mapping[str, Any],
        base_trace: Mapping[str, Any],
    ) -> dict[str, Any]:
        negative = self.source_index.scoped_negative_candidates(
            document_id=scope["document_id"],
            company=scope["company"],
            report_year=scope["report_year"],
            question=scope["question"],
        )
        table_text = self._table_groups(scope, include_numeric=False)
        table_numeric = self._table_groups(scope, include_numeric=True)
        absence = self.source_index.absence_audit(
            document_id=scope["document_id"], question=scope["question"]
        )
        return {
            "scope": dict(scope),
            "base_answer": dict(base_answer),
            "base_trace_hash": base_trace.get("trace_hash"),
            "base_heading": self._base_heading(base_answer, scope["question"]),
            "negative_candidates": negative,
            "table_text_groups": table_text,
            "table_numeric_groups": table_numeric,
            "absence_audit": absence,
        }

    def materialize(self, prepared: Mapping[str, Any], features: frozenset[str]) -> dict[str, Any]:
        scope = prepared["scope"]
        base = prepared["base_answer"]
        answer_text = str(base.get("answer_text") or "").strip()
        citations = list(base.get("citations") or [])
        selected_groups: list[dict[str, Any]] = []
        numeric_authorizations: list[dict[str, Any]] = []
        absence_projection: dict[str, Any] | None = None

        if not answer_text and "checkbox" in features and prepared["negative_candidates"]:
            candidate = prepared["negative_candidates"][0]
            heading = candidate.heading_path[-1] if candidate.heading_path else "相关事项"
            answer_text = (
                f"{scope['report_year']}年{scope['company']}年报“{heading}”栏明确勾选“√不适用”。"
            )
            citation = _candidate_citation(candidate)
            citations.append(citation)
            selected_groups.append({
                "group_id": candidate.candidate_id,
                "source_kind": candidate.source_kind,
                "text": answer_text,
                "citations": [citation],
                "numeric_authorizations": [],
            })

        if "table" in features:
            groups = (
                prepared["table_numeric_groups"]
                if "numeric" in features
                else prepared["table_text_groups"]
            )
            # The experimental table surface repairs a genuine coverage hole;
            # it never appends table cells to a usable v4 narrative.  Besides
            # preserving the frozen claim set, this prevents a mixed text cell
            # with unscoped numbers from bypassing cell-level authorization.
            if not answer_text:
                room = MAX_CLAIM_GROUPS - len(selected_groups)
                for group in groups[:max(0, room)]:
                    if len(answer_text) + len(group["text"]) + 1 > MAX_ANSWER_CHARS:
                        break
                    answer_text = "\n".join(value for value in (answer_text, group["text"]) if value)
                    citations.extend(group["citations"])
                    numeric_authorizations.extend(group["numeric_authorizations"])
                    selected_groups.append(group)

        if not answer_text and "absence" in features and prepared["absence_audit"] is not None:
            audit = dict(prepared["absence_audit"])
            groups = audit["topic_groups"]
            aliases = _group_aliases(groups)
            concern = aliases[0] if aliases else "相关事项"
            answer_text = (
                f"在已解析的{scope['report_year']}年{scope['company']}年报中，"
                f"未检索到与“{concern}”相关的披露。"
            )
            absence_projection = audit
            citations.append({
                "citation_id": "v7-cite-absence-" + semantic_sha256(audit)[:20],
                "source_kind": "document_absence_audit",
                "candidate_id": audit["audit_id"],
                "document_id": audit["document_id"],
                "source_markdown": audit["source_markdown"],
                "line_range": [1, audit["line_count"]],
                "heading_path": [],
                "content_sha256": audit["document_sha256"],
            })
            selected_groups.append({
                "group_id": audit["audit_id"],
                "source_kind": "document_absence",
                "text": answer_text,
                "citations": [citations[-1]],
                "numeric_authorizations": [],
            })

        if (
            "frame" in features
            and answer_text
            and not selected_groups
            and len(_normalize(answer_text)) <= MAX_FRAME_SOURCE_CHARS
        ):
            answer_text = self._frame_answer(
                answer_text,
                company=scope["company"],
                report_year=scope["report_year"],
                heading=prepared["base_heading"],
            )

        # A final stable de-duplication cannot merge citations across scope;
        # every retained record already names this one frozen document.
        citation_map = {
            str(row.get("citation_id")): row
            for row in citations
            if isinstance(row, Mapping) and row.get("citation_id")
        }
        citations = [citation_map[key] for key in sorted(citation_map)]
        numeric_map = {row["authorization_id"]: row for row in numeric_authorizations}
        numeric_authorizations = [numeric_map[key] for key in sorted(numeric_map)]
        status = "ok" if answer_text else "not_found"
        trace = {
            "schema_version": "finglmqa.experimental.type3_v7_trace.v1",
            "profile_version": TYPE3_V7_VERSION,
            "case_id": scope["case_id"],
            "document_id": scope["document_id"],
            "features": sorted(features),
            "base_trace_hash": prepared["base_trace_hash"],
            "candidate_counts": {
                "checkbox": len(prepared["negative_candidates"]),
                "table_text_groups": len(prepared["table_text_groups"]),
                "table_numeric_groups": len(prepared["table_numeric_groups"]),
            },
            "selected_group_ids": [row["group_id"] for row in selected_groups],
            "selected_groups": [
                {
                    "group_id": row["group_id"],
                    "source_kind": row["source_kind"],
                    "text": row["text"],
                    "citation_ids": [
                        citation["citation_id"] for citation in row["citations"]
                    ],
                    "numeric_authorization_ids": [
                        authorization["authorization_id"]
                        for authorization in row["numeric_authorizations"]
                    ],
                }
                for row in selected_groups
            ],
            "numeric_authorizations": numeric_authorizations,
            "document_absence_audit": absence_projection,
            "generative_llm_used": False,
        }
        trace["trace_hash"] = semantic_sha256(trace)
        return {
            "answer": answer_text,
            "status": status,
            "errors": [] if answer_text else [{"failure_code": "EVIDENCE_UNAVAILABLE"}],
            "warnings": [],
            "citations": citations,
            "trace": trace,
        }


def canonical_line(value: Any) -> bytes:
    return (canonical_json(value) + "\n").encode("utf-8")


__all__ = [
    "DocumentAuditIndex",
    "EvidenceCandidate",
    "MAX_ANSWER_CHARS",
    "MAX_CLAIM_GROUPS",
    "MAX_FRAME_SOURCE_CHARS",
    "MAX_TABLE_CANDIDATES",
    "MAX_TABLE_ROWS_PER_GROUP",
    "MAX_TEXT_CANDIDATES",
    "STAGE_FEATURES",
    "TYPE3_V7_VERSION",
    "Type3V7Enhancer",
    "V7TableRetriever",
    "canonical_line",
]
