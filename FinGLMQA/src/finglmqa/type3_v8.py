"""Opt-in Type 3-1 v8 evidence repair profile.

The profile composes the four generic v7 follow-up policies without changing
the frozen Phase 8/10 service path.  It consumes only the question, the
already-resolved document scope, frozen A2RAG top-k chunk identities, audited
table fragments, and source Markdown.  Benchmark annotations and reference
answers are never inputs to this module.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

from .contracts import semantic_sha256
from .evidence_executor import EvidenceExecutor
from .table_evidence import TableEvidenceIndex
from .type3_v7 import (
    DocumentAuditIndex,
    EvidenceCandidate,
    _DATE_VALUE_RE,
    _DECIMAL_RE,
    _NUMERIC_LIKE_RE,
    _candidate_citation,
    _compact,
    _group_aliases,
    _is_header_fragment,
    _matched_groups,
    _normalize,
    _numeric_authorization,
    _preferred_cell_ordinals,
    _question_groups,
    _text_cell,
    _unit_from_fragment,
    _year_from_column,
)
from .type3_v7_intent import DecisionStatus, classify_question_intent, match_title_path
from .type3_v7_lowinfo import classify_answer, classify_candidate, rank_frozen_chunks
from .type3_v7_negative import (
    SUPPORTED_NEGATIVE_TOPIC_GROUPS,
    audit_negative_evidence,
    recommended_negative_wording,
)
from .type3_v7_table_upgrade import decide_table_upgrade


TYPE3_V8_VERSION = "type3-v8-deterministic-evidence-repair-v1"
TYPE3_V8_STAGES: tuple[tuple[str, frozenset[str]], ...] = (
    ("v7_baseline", frozenset()),
    ("intent_lowinfo", frozenset({"text"})),
    ("table_upgrade", frozenset({"text", "table"})),
    ("full", frozenset({"text", "table", "negative"})),
)
MAX_TEXT_REPAIR_CHUNKS = 3
MAX_TEXT_REPAIR_CLAIMS = 3
MAX_V8_ANSWER_CHARS = 1500
_ANY_NUMBER_RE = re.compile(r"(?<![A-Za-z])\d+(?:[,.]\d+)*(?![A-Za-z])")
_CURRENT_PERIOD_RE = re.compile(r"(?:本年|本年度|本期|报告期|本报告期|当年|年度)")
_PREVIOUS_PERIOD_RE = re.compile(r"(?:上年|上年度|上期|期初|去年)")
_SENTENCE_SPLIT_RE = re.compile(r"[^。！？!?；;]+[。！？!?；;]?")
_CHECKBOX_OPTION_RE = re.compile(
    r"([□☐○◯■☑√✓✔])\s*((?:本年度|本年)?\s*(?:适用|不适用|是|否))"
)
_SELECTED_CHECKBOX = frozenset("■☑√✓✔")
_UNSELECTED_CHECKBOX = frozenset("□☐○◯")
_RANKED_PARTY_DISCLOSURE_RE = re.compile(
    r"前五名(?P<party>客户|供应商)(?P<amount_label>销售额|采购额)"
    r"(?P<amount>[-+]?\d[\d,.]*)(?P<unit>亿元|万元|千元|元)"
    r"[，,]?占年度(?P<total_kind>销售|采购)总额(?P<ratio>[-+]?\d[\d,.]*)[%％]"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class FrozenEvidenceCorpus:
    """Read-only lookup for Phase 7 chunks and their document ordinals."""

    def __init__(
        self,
        *,
        chunk_path: str | Path,
        document_map_path: str | Path,
    ) -> None:
        self.chunk_path = Path(chunk_path).resolve()
        self.document_map_path = Path(document_map_path).resolve()
        self._chunks: dict[str, dict[str, Any]] = {}
        self._ordinals: dict[str, dict[str, int]] = {}
        self._document_ids: dict[str, tuple[str, ...]] = {}
        for line in self.chunk_path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            row = json.loads(line)
            chunk_id = str(row["evidence_chunk_id"])
            if chunk_id in self._chunks:
                raise RuntimeError(f"duplicate evidence chunk: {chunk_id}")
            self._chunks[chunk_id] = row
        for line in self.document_map_path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            row = json.loads(line)
            document_id = str(row["document_id"])
            ids = [str(value) for value in row["chunk_ids"]]
            self._ordinals[document_id] = {chunk_id: index for index, chunk_id in enumerate(ids)}
            self._document_ids[document_id] = tuple(ids)

    def rows(self, chunk_ids: Sequence[str], *, document_id: str) -> list[dict[str, Any]]:
        ordinals = self._ordinals.get(document_id)
        if ordinals is None:
            return []
        result: list[dict[str, Any]] = []
        for chunk_id in chunk_ids:
            row = self._chunks.get(chunk_id)
            if row is None or row.get("document_id") != document_id or chunk_id not in ordinals:
                continue
            result.append(dict(row))
        return result

    def document_rows(self, document_id: str) -> list[dict[str, Any]]:
        return self.rows(self._document_ids.get(document_id, ()), document_id=document_id)

    def provider_chunk(
        self,
        row: Mapping[str, Any],
        *,
        score: str,
    ) -> dict[str, Any]:
        if row.get("_source_fallback") is True:
            return {
                "chunk_id": str(row["evidence_chunk_id"]),
                "document_chunk_ordinal": int(row["source_line_number"]),
                "score": score,
                "document_id": str(row["document_id"]),
                "company": str(row["company_full"]),
                "stock_code": str(row["stock_code"]),
                "report_year": int(row["report_year"]),
                "section_path": list(row["section_path"]),
                "semantic_tags": [],
                "line_range": [int(row["source_line_number"]), int(row["source_line_number"])],
                "source_markdown": str(row["source_markdown"]),
                "content": str(row["content"]),
            }
        document_id = str(row["document_id"])
        chunk_id = str(row["evidence_chunk_id"])
        return {
            "chunk_id": chunk_id,
            "document_chunk_ordinal": self._ordinals[document_id][chunk_id],
            "score": score,
            "document_id": document_id,
            "company": str(row["company_full"]),
            "stock_code": str(row["stock_code"]),
            "report_year": int(row["report_year"]),
            "section_path": list(row["section_path"]),
            "semantic_tags": list(row["semantic_tags"]),
            "line_range": list(row["line_range"]),
            "source_markdown": str(row["source_markdown"]),
            "content": str(row["content"]),
        }


class _StaticEvidenceProvider:
    def __init__(self, result: Mapping[str, Any]) -> None:
        self.result = dict(result)

    def retrieve(self, request: Mapping[str, Any]) -> dict[str, Any]:
        if request["document_id"] != self.result["document_id"]:
            raise ValueError("static provider document scope mismatch")
        return dict(self.result)


def _citation_paths(citations: Sequence[Mapping[str, Any]]) -> tuple[tuple[str, ...], ...]:
    result: list[tuple[str, ...]] = []
    for citation in citations:
        if not isinstance(citation, Mapping):
            continue
        path = citation.get("heading_path")
        if not isinstance(path, list):
            provenance = citation.get("provenance")
            path = provenance.get("section_path") if isinstance(provenance, Mapping) else None
        if isinstance(path, list) and all(isinstance(value, str) for value in path):
            result.append(tuple(value for value in path if value.strip()))
    return tuple(result)


def _intent_mismatch(
    question: str,
    title_paths: Sequence[Sequence[str]],
) -> tuple[bool, str]:
    decision = classify_question_intent(question)
    if decision.status is DecisionStatus.UNKNOWN:
        groups = _question_groups(question)
        if not groups:
            return False, "unsupported_disambiguation_intent"
        if not title_paths:
            return True, "answer_has_no_concern_title_path"
        if any(
            groups.intersection(_matched_groups(" ".join(path)))
            for path in title_paths
        ):
            return False, "answer_concern_title_path_compatible"
        return True, "answer_concern_title_path_mismatch"
    if decision.status is DecisionStatus.AMBIGUOUS:
        return True, "question_intent_ambiguous"
    if not title_paths:
        return True, "answer_has_no_title_path"
    if any(match_title_path(question, path).accepted for path in title_paths):
        return False, "answer_title_path_compatible"
    return True, "answer_title_path_mismatch"


def _enhanced_numeric_authorization(
    candidate: EvidenceCandidate,
    *,
    column_ordinal: int,
    column_label: str,
    raw_value: str,
) -> dict[str, Any] | None:
    standard = _numeric_authorization(
        candidate,
        column_ordinal=column_ordinal,
        column_label=column_label,
        raw_value=raw_value,
    )
    if standard is not None:
        return standard
    compact_value = re.sub(r"\s+", "", _normalize(raw_value))
    if _DATE_VALUE_RE.fullmatch(compact_value) or not _DECIMAL_RE.fullmatch(compact_value):
        return None
    normalized_label = _normalize(column_label)
    if _PREVIOUS_PERIOD_RE.search(normalized_label) or not _CURRENT_PERIOD_RE.search(normalized_label):
        return None
    fragment = candidate.source_payload
    unit = _unit_from_fragment(fragment, column_label, compact_value)
    if unit is None:
        return None
    coordinates = fragment.get("cell_coordinates") or []
    coordinate = coordinates[column_ordinal] if column_ordinal < len(coordinates) else None
    if not isinstance(coordinate, list) or len(coordinate) != 2:
        return None
    try:
        decimal_value = Decimal(compact_value.rstrip("%％").replace(",", ""))
    except InvalidOperation:
        return None
    if not decimal_value.is_finite():
        return None
    allowed = [compact_value]
    if unit == "%" and not compact_value.endswith(("%", "％")):
        allowed.append(f"{compact_value}%")
    elif unit != "%" and not compact_value.endswith(unit):
        allowed.append(f"{compact_value}{unit}")
    authorization = {
        "schema_version": "finglmqa.experimental.table_numeric_authorization.v1",
        "document_id": candidate.document_id,
        "company": candidate.company,
        "report_year": candidate.report_year,
        "metric_year": candidate.report_year,
        "table_id": str(fragment["table_id"]),
        "table_content_sha256": str(fragment["provenance"]["table_content_sha256"]),
        "fragment_id": candidate.candidate_id,
        "cell_coordinate": list(coordinate),
        "column_label": column_label,
        "raw_value": raw_value,
        "raw_value_sha256": hashlib.sha256(raw_value.encode("utf-8")).hexdigest(),
        "normalized_value": format(decimal_value, "f"),
        "normalized_unit": unit,
        "source_markdown": candidate.source_markdown,
        "source_line_range": list(candidate.line_range),
        "allowed_renderings": sorted(set(allowed)),
    }
    authorization["authorization_id"] = "v7-table-auth-" + semantic_sha256(authorization)[:24]
    return authorization


def _render_table_candidate(
    candidate: EvidenceCandidate,
    *,
    question: str,
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
        raw = values[ordinal]
        value = _normalize(raw)
        if not value or value == label or _DATE_VALUE_RE.fullmatch(re.sub(r"\s+", "", value)):
            continue
        authorization = _enhanced_numeric_authorization(
            candidate,
            column_ordinal=ordinal,
            column_label=column_label,
            raw_value=raw,
        )
        if authorization is not None:
            authorizations.append(authorization)
            rendering = next(
                (
                    item for item in authorization["allowed_renderings"]
                    if item.endswith(authorization["normalized_unit"])
                ),
                authorization["allowed_renderings"][0],
            )
            parts.append(f"{column_label}={rendering}" if column_label else rendering)
        elif _text_cell(value) and not _ANY_NUMBER_RE.search(value):
            parts.append(f"{column_label}={value}" if column_label else value)
    if not parts:
        return "", []
    prefix = "" if _NUMERIC_LIKE_RE.fullmatch(label) else f"{label}："
    return (prefix + "；".join(parts)).strip("：； "), authorizations


class V8TableSurface:
    """Create row-semantic table groups for the fail-closed upgrade policy."""

    def __init__(self, index: TableEvidenceIndex, source_index: DocumentAuditIndex) -> None:
        self.index = index
        self.source_index = source_index
        self._document_cache: dict[str, tuple[dict[str, Any], ...]] = {}

    def _rows(self, document_id: str) -> tuple[dict[str, Any], ...]:
        if document_id not in self._document_cache:
            self._document_cache[document_id] = tuple(self.index.iter_document(document_id))
        return self._document_cache[document_id]

    def groups(self, scope: Mapping[str, Any]) -> list[dict[str, Any]]:
        document_id = str(scope["document_id"])
        source = self.source_index.source(document_id)
        grouped: dict[str, list[tuple[EvidenceCandidate, str, list[dict[str, Any]]]]] = defaultdict(list)
        for fragment in self._rows(document_id):
            if fragment.get("fragment_kind") != "table_row" or _is_header_fragment(fragment):
                continue
            line_number = int(fragment["source_line_range"][0])
            source_path = source.heading_path(line_number)
            existing_path = tuple(
                str(value) for value in fragment.get("section_path") or [] if str(value).strip()
            )
            extra_path = tuple(
                str(value)
                for value in (fragment.get("caption"), *(fragment.get("header_path") or []))
                if str(value or "").strip()
            )
            heading_path = tuple(dict.fromkeys((*source_path, *existing_path, *extra_path)))
            candidate = EvidenceCandidate(
                candidate_id=str(fragment["fragment_id"]),
                source_kind="table_row",
                document_id=document_id,
                company=str(scope["company"]),
                report_year=int(scope["report_year"]),
                heading_path=heading_path,
                source_markdown=str(fragment["source_markdown"]),
                line_range=tuple(fragment["source_line_range"]),
                raw_text=str(fragment["content"]),
                source_ordinal=int(fragment["table_index"]) * 100000 + int(fragment["row_index"]),
                topic_groups=tuple(sorted(_matched_groups(" ".join(heading_path)))),
                heading_group_hits=0,
                topic_anchor_score=0,
                source_payload=dict(fragment),
            )
            rendered, authorizations = _render_table_candidate(
                candidate,
                question=str(scope["question"]),
            )
            if rendered:
                grouped[str(fragment["table_id"])].append((candidate, rendered, authorizations))

        result: list[dict[str, Any]] = []
        for table_id, rows in grouped.items():
            rows.sort(key=lambda item: (item[0].source_ordinal, item[0].candidate_id))
            heading = next(
                (
                    value
                    for value in reversed(rows[0][0].heading_path)
                    if value and "年度报告" not in _compact(value)
                ),
                rows[0][0].heading_path[-1] if rows[0][0].heading_path else "表格披露",
            )
            citations = [_candidate_citation(candidate) for candidate, _, _ in rows]
            auth_map = {
                authorization["authorization_id"]: authorization
                for _, _, authorizations in rows
                for authorization in authorizations
            }
            row_semantics = [
                {
                    "document_id": document_id,
                    "fragment_id": candidate.candidate_id,
                    "table_id": table_id,
                    "heading_path": list(candidate.heading_path),
                    "row_label": str(candidate.source_payload.get("row_label") or "项目"),
                    "column_labels": [
                        str(value) for value in candidate.source_payload.get("column_labels") or []
                    ],
                    "rendered_text": rendered,
                    "numeric_authorization_ids": [
                        authorization["authorization_id"] for authorization in authorizations
                    ],
                    "source_ordinal": candidate.source_ordinal,
                }
                for candidate, rendered, authorizations in rows
            ]
            result.append({
                "group_id": "v8-table-group-" + semantic_sha256([document_id, table_id])[:20],
                "source_kind": "table",
                "heading": heading,
                "text": f"{heading}：" + "；".join(rendered for _, rendered, _ in rows[:12]),
                "citations": citations,
                "numeric_authorizations": [auth_map[key] for key in sorted(auth_map)],
                "row_semantics": row_semantics,
            })
        result.sort(
            key=lambda group: (
                min(row["source_ordinal"] for row in group["row_semantics"]),
                group["group_id"],
            )
        )
        return result


@dataclass(frozen=True)
class _TextRepair:
    answer: str
    citations: tuple[dict[str, Any], ...]
    chunk_ids: tuple[str, ...]
    executor_trace: Mapping[str, Any]


@dataclass(frozen=True)
class _RankedPartyDisclosure:
    answer: str
    citation: Mapping[str, Any]
    numeric_authorizations: tuple[Mapping[str, Any], ...]
    disclosure_id: str


_SOURCE_CONTRACT_EVENT_RE = re.compile(
    r"(?:(?:签订|订立|签署).{0,80}(?:合同|协议)|(?:合同|协议).{0,80}(?:签订|订立|签署))"
)


def _parse_ranked_party_disclosure(
    text: str,
    *,
    expected_group: str,
    document_id: str,
    company: str,
    report_year: int,
    source_markdown: str,
    line_number: int,
    heading_path: Sequence[str],
) -> _RankedPartyDisclosure | None:
    """Authorize the standard top-five customer/supplier narrative line.

    Some annual reports render this disclosure as ordinary Markdown rather
    than a table.  The authorization is deliberately source-line-specific and
    permits only the two literal values captured from that line.
    """

    compact_text = re.sub(r"\s+", "", _normalize(text))
    match = _RANKED_PARTY_DISCLOSURE_RE.search(compact_text)
    if match is None:
        return None
    party = match.group("party")
    group = "customer" if party == "客户" else "supplier"
    if group != expected_group:
        return None
    if (party == "客户") != (match.group("amount_label") == "销售额"):
        return None
    if (party == "客户") != (match.group("total_kind") == "销售"):
        return None
    amount = match.group("amount")
    unit = match.group("unit")
    ratio = match.group("ratio") + "%"
    answer = (
        f"根据{report_year}年{company}年报，前五名{party}"
        f"{match.group('amount_label')}为{amount}{unit}，"
        f"占年度{match.group('total_kind')}总额{ratio}。"
    )
    source_hash = hashlib.sha256(_normalize(text).encode("utf-8")).hexdigest()
    base = {
        "schema_version": "finglmqa.experimental.source_numeric_authorization.v1",
        "document_id": document_id,
        "company": company,
        "report_year": report_year,
        "source_markdown": source_markdown,
        "source_line_range": [line_number, line_number],
        "source_line_sha256": source_hash,
        "heading_path": list(heading_path),
        "concern_group": group,
    }
    authorizations: list[dict[str, Any]] = []
    for role, raw_value, normalized_unit, rendering in (
        ("top_five_amount", amount, unit, f"{amount}{unit}"),
        ("annual_total_ratio", ratio, "%", ratio),
    ):
        authorization = {
            **base,
            "value_role": role,
            "raw_value": raw_value,
            "normalized_unit": normalized_unit,
            "allowed_renderings": [rendering],
        }
        authorization["authorization_id"] = (
            "v8-source-auth-" + semantic_sha256(authorization)[:24]
        )
        authorizations.append(authorization)
    disclosure_payload = {
        "answer": answer,
        "document_id": document_id,
        "source_line_range": [line_number, line_number],
        "authorization_ids": [row["authorization_id"] for row in authorizations],
    }
    disclosure_id = "v8-source-disclosure-" + semantic_sha256(disclosure_payload)[:24]
    citation = {
        "citation_id": "v8-cite-" + disclosure_id[-20:],
        "source_kind": "source_numeric_disclosure",
        "candidate_id": disclosure_id,
        "document_id": document_id,
        "source_markdown": source_markdown,
        "line_range": [line_number, line_number],
        "heading_path": list(heading_path),
        "content_sha256": source_hash,
    }
    return _RankedPartyDisclosure(
        answer=answer,
        citation=citation,
        numeric_authorizations=tuple(authorizations),
        disclosure_id=disclosure_id,
    )


def _parse_major_contract_disclosure(
    text: str,
    *,
    source_line_text: str | None = None,
    document_id: str,
    company: str,
    report_year: int,
    source_markdown: str,
    line_number: int,
    heading_path: Sequence[str],
) -> _RankedPartyDisclosure | None:
    normalized = _normalize(text)
    compact = _compact(normalized)
    if not normalized or not _SOURCE_CONTRACT_EVENT_RE.search(normalized):
        return None
    if str(report_year) not in normalized:
        return None
    if any(marker in compact for marker in ("详见", "参见", "请查阅")) and len(compact) < 60:
        return None
    source_line = _normalize(source_line_text if source_line_text is not None else text)
    source_hash = hashlib.sha256(source_line.encode("utf-8")).hexdigest()
    base = {
        "schema_version": "finglmqa.experimental.source_numeric_authorization.v1",
        "document_id": document_id,
        "company": company,
        "report_year": report_year,
        "source_markdown": source_markdown,
        "source_line_range": [line_number, line_number],
        "source_line_sha256": source_hash,
        "heading_path": list(heading_path),
        "concern_group": "major_contract",
    }
    authorizations: list[dict[str, Any]] = []
    for ordinal, match in enumerate(_ANY_NUMBER_RE.finditer(normalized)):
        raw = match.group(0)
        authorization = {
            **base,
            "value_role": f"literal_{ordinal}",
            "raw_value": raw,
            "normalized_unit": "source_literal",
            "allowed_renderings": [raw],
            "source_character_range": [match.start(), match.end()],
        }
        authorization["authorization_id"] = (
            "v8-source-auth-" + semantic_sha256(authorization)[:24]
        )
        authorizations.append(authorization)
    answer = f"根据{report_year}年{company}年报，{normalized}"
    disclosure_payload = {
        "answer": answer,
        "document_id": document_id,
        "source_line_range": [line_number, line_number],
        "authorization_ids": [row["authorization_id"] for row in authorizations],
    }
    disclosure_id = "v8-source-disclosure-" + semantic_sha256(disclosure_payload)[:24]
    citation = {
        "citation_id": "v8-cite-" + disclosure_id[-20:],
        "source_kind": "source_numeric_disclosure",
        "candidate_id": disclosure_id,
        "document_id": document_id,
        "source_markdown": source_markdown,
        "line_range": [line_number, line_number],
        "heading_path": list(heading_path),
        "content_sha256": source_hash,
    }
    return _RankedPartyDisclosure(
        answer=answer,
        citation=citation,
        numeric_authorizations=tuple(authorizations),
        disclosure_id=disclosure_id,
    )


class Type3V8Enhancer:
    def __init__(
        self,
        *,
        root: str | Path,
        table_index: TableEvidenceIndex,
        evidence_corpus: FrozenEvidenceCorpus,
    ) -> None:
        self.root = Path(root).resolve()
        self.source_index = DocumentAuditIndex(root=self.root)
        self.table_surface = V8TableSurface(table_index, self.source_index)
        self.evidence_corpus = evidence_corpus

    @staticmethod
    def _retrieved_chunk_ids(base_trace: Mapping[str, Any]) -> list[str]:
        result: list[str] = []
        for row in base_trace.get("subplan_traces") or []:
            trace = row.get("trace") if isinstance(row, Mapping) else None
            ids = trace.get("retrieved_chunk_ids") if isinstance(trace, Mapping) else None
            if isinstance(ids, list):
                result.extend(str(value) for value in ids if isinstance(value, str))
        return list(dict.fromkeys(result))

    @staticmethod
    def _ranking_question(scope: Mapping[str, Any]) -> str:
        question = str(scope["question"])
        for value in (scope.get("company"), scope.get("stock_code")):
            if isinstance(value, str) and value:
                question = question.replace(value, "")
        return re.sub(r"(?<!\d)(?:19|20|21)\d{2}\s*年?", "", question)

    def _source_fallback_rows(
        self,
        *,
        scope: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        question = str(scope["question"])
        decision = classify_question_intent(question)
        question_groups = _question_groups(question)
        source = self.source_index.source(str(scope["document_id"]))
        result: list[dict[str, Any]] = []
        for line_number, (line, path) in enumerate(
            zip(source.lines, source.heading_paths), start=1
        ):
            text = _normalize(line)
            if not text or text.startswith(("#", "<table", "</table")):
                continue
            if decision.status is DecisionStatus.RESOLVED:
                compatible = match_title_path(question, path).accepted
            else:
                compatible = bool(question_groups.intersection(_matched_groups(" ".join(path))))
            if not compatible:
                continue
            sentences = [
                _normalize(match.group(0))
                for match in _SENTENCE_SPLIT_RE.finditer(text)
                if _normalize(match.group(0))
            ] or [text]
            for sentence_ordinal, sentence in enumerate(sentences):
                payload = {
                    "document_id": str(scope["document_id"]),
                    "source_line_number": line_number,
                    "sentence_ordinal": sentence_ordinal,
                    "section_path": list(path),
                    "content": sentence,
                }
                result.append({
                    "_source_fallback": True,
                    "evidence_chunk_id": "v8-source-" + semantic_sha256(payload)[:24],
                    "document_id": str(scope["document_id"]),
                    "company_full": str(scope["company"]),
                    "stock_code": str(scope["stock_code"]),
                    "report_year": int(scope["report_year"]),
                    "section_path": list(path),
                    "source_markdown": source.portable_path,
                    "source_line_number": line_number,
                    "content": sentence,
                })
        return result

    def _ranked_party_disclosure(
        self,
        *,
        scope: Mapping[str, Any],
    ) -> _RankedPartyDisclosure | None:
        groups = _question_groups(str(scope["question"]))
        expected = [group for group in ("customer", "supplier") if group in groups]
        if len(expected) != 1:
            return None
        source = self.source_index.source(str(scope["document_id"]))
        candidates: list[_RankedPartyDisclosure] = []
        for line_number, (line, path) in enumerate(
            zip(source.lines, source.heading_paths), start=1
        ):
            if expected[0] not in _matched_groups(" ".join(path)):
                continue
            parsed = _parse_ranked_party_disclosure(
                line,
                expected_group=expected[0],
                document_id=str(scope["document_id"]),
                company=str(scope["company"]),
                report_year=int(scope["report_year"]),
                source_markdown=source.portable_path,
                line_number=line_number,
                heading_path=path,
            )
            if parsed is not None:
                candidates.append(parsed)
        if not candidates:
            return None
        candidates.sort(
            key=lambda row: (
                row.citation["line_range"],
                row.disclosure_id,
            )
        )
        return candidates[0]

    def _major_contract_disclosure(
        self,
        *,
        scope: Mapping[str, Any],
    ) -> _RankedPartyDisclosure | None:
        if "major_contract" not in _question_groups(str(scope["question"])):
            return None
        source = self.source_index.source(str(scope["document_id"]))
        candidates: list[_RankedPartyDisclosure] = []
        for line_number, (line, path) in enumerate(
            zip(source.lines, source.heading_paths), start=1
        ):
            if "major_contract" not in _matched_groups(" ".join(path)):
                continue
            for sentence_match in _SENTENCE_SPLIT_RE.finditer(_normalize(line)):
                parsed = _parse_major_contract_disclosure(
                    sentence_match.group(0),
                    source_line_text=line,
                    document_id=str(scope["document_id"]),
                    company=str(scope["company"]),
                    report_year=int(scope["report_year"]),
                    source_markdown=source.portable_path,
                    line_number=line_number,
                    heading_path=path,
                )
                if parsed is not None:
                    candidates.append(parsed)
        if not candidates:
            return None
        candidates.sort(
            key=lambda row: (row.citation["line_range"], row.disclosure_id)
        )
        return candidates[0]

    def _generic_scoped_checkbox(
        self,
        *,
        scope: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        groups = _question_groups(str(scope["question"]))
        if not groups:
            return None
        source = self.source_index.source(str(scope["document_id"]))
        candidates: list[dict[str, Any]] = []
        for line_number, (line, path) in enumerate(
            zip(source.lines, source.heading_paths), start=1
        ):
            options = _CHECKBOX_OPTION_RE.findall(_normalize(line))
            if len(options) != 2 or not path:
                continue
            selected = [
                _compact(label).removeprefix("本年度").removeprefix("本年")
                for symbol, label in options if symbol in _SELECTED_CHECKBOX
            ]
            unselected = [
                _compact(label).removeprefix("本年度").removeprefix("本年")
                for symbol, label in options if symbol in _UNSELECTED_CHECKBOX
            ]
            if (selected, unselected) not in ((["不适用"], ["适用"]), (["否"], ["是"])):
                continue
            # A negative child row must not negate a broader parent concern
            # (for example, a non-applicable fair-value subsection cannot
            # answer a question about the whole asset/liability position).
            matched = groups.intersection(_matched_groups(path[-1]))
            if not matched:
                continue
            payload = {
                "document_id": str(scope["document_id"]),
                "heading_path": list(path),
                "line_range": [line_number, line_number],
                "text": _normalize(line),
                "selected_label": selected[0],
                "topic_groups": sorted(matched),
            }
            payload["candidate_id"] = "v8-checkbox-" + semantic_sha256(payload)[:24]
            candidates.append(payload)
        if not candidates:
            return None
        candidates.sort(
            key=lambda row: (
                -len(row["topic_groups"]),
                -len(row["heading_path"]),
                row["line_range"],
                row["candidate_id"],
            )
        )
        selected = candidates[0]
        selected["source_markdown"] = source.portable_path
        selected["document_sha256"] = source.sha256
        return selected

    def _text_repair(
        self,
        *,
        scope: Mapping[str, Any],
        base_trace: Mapping[str, Any],
    ) -> _TextRepair | None:
        frozen_rows = self.evidence_corpus.rows(
            self._retrieved_chunk_ids(base_trace),
            document_id=str(scope["document_id"]),
        )
        if not frozen_rows:
            return None
        question = str(scope["question"])
        question_intent = classify_question_intent(question)
        # A wrong top-k cannot be repaired by merely reordering it.  The
        # fallback remains inside the already-resolved document and reuses the
        # same immutable Phase 7 allow-list; exact title/concern compatibility
        # still has to pass before a chunk reaches EvidenceExecutor.
        document_rows = self.evidence_corpus.document_rows(str(scope["document_id"]))
        source_rows = self._source_fallback_rows(scope=scope)
        by_id = {
            str(row["evidence_chunk_id"]): row
            for row in (*frozen_rows, *document_rows, *source_rows)
        }
        chunk_rows = list(by_id.values())
        compatibility: dict[str, bool] = {}
        generic_groups = _question_groups(question)
        has_generic_heading_match = bool(generic_groups) and any(
            generic_groups.intersection(_matched_groups(" ".join(row.get("section_path") or [])))
            for row in chunk_rows
        )
        for row in chunk_rows:
            chunk_id = str(row["evidence_chunk_id"])
            if question_intent.status is DecisionStatus.RESOLVED:
                compatibility[chunk_id] = match_title_path(
                    question, row.get("section_path") or []
                ).accepted
            elif question_intent.status is DecisionStatus.AMBIGUOUS:
                compatibility[chunk_id] = False
            else:
                compatibility[chunk_id] = (
                    bool(generic_groups.intersection(
                        _matched_groups(" ".join(row.get("section_path") or []))
                    ))
                    if has_generic_heading_match else True
                )
        ranked_rows = rank_frozen_chunks(
            chunk_rows,
            question=self._ranking_question(scope),
            title_path_compatibility=compatibility,
            expected_document_id=str(scope["document_id"]),
        )
        report_year = int(scope["report_year"])

        def year_priority(row: Any) -> int:
            years = {
                int(value)
                for value in re.findall(
                    r"(?<!\d)((?:19|20|21)\d{2})\s*年",
                    str(row.chunk.get("content") or ""),
                )
            }
            if report_year in years:
                return 0
            return 1 if not years else 2

        usable_ranked = [row for row in ranked_rows if row.concern_score > 0]
        usable_ranked.sort(
            key=lambda row: (
                -row.concern_score,
                bool(EvidenceExecutor._financial_tokens(
                    str(row.chunk.get("content") or ""), str(scope["stock_code"])
                )),
                year_priority(row),
                row.original_rank,
                row.chunk_id,
            )
        )
        ranked = tuple(row.chunk for row in usable_ranked[:MAX_TEXT_REPAIR_CHUNKS])
        if not ranked:
            return None
        provider_chunks = [
            self.evidence_corpus.provider_chunk(
                row,
                score=f"{1 - ordinal * 0.01:.8f}",
            )
            for ordinal, row in enumerate(ranked)
        ]
        provider_result = {
            "schema_version": "finglmqa.phase8.evidence_provider_result.v1",
            "status": "ok",
            "document_id": str(scope["document_id"]),
            "company": str(scope["company"]),
            "stock_code": str(scope["stock_code"]),
            "report_year": int(scope["report_year"]),
            "retrieval_method": "phase7_frozen_topk_title_lowinfo_rerank",
            "provider_fingerprint": semantic_sha256([
                str(scope["document_id"]),
                [row["chunk_id"] for row in provider_chunks],
            ]),
            "chunks": provider_chunks,
        }
        subplans = base_trace.get("composition_plan", {}).get("subplans", [])
        if not isinstance(subplans, list) or len(subplans) != 1:
            return None
        authorization_set = base_trace.get("numeric_authorization_set")
        if not isinstance(authorization_set, Mapping):
            return None
        result = EvidenceExecutor(_StaticEvidenceProvider(provider_result)).execute(
            subplans[0], authorization_set
        )
        if result.get("status") != "ok" or not result.get("claims"):
            return None
        rank_by_chunk = {
            row["chunk_id"]: ordinal for ordinal, row in enumerate(provider_chunks)
        }
        citation_by_id = {
            citation["citation_id"]: citation for citation in result.get("citations") or []
        }

        def claim_rank(claim: Mapping[str, Any]) -> tuple[Any, ...]:
            ids = [
                citation_by_id[citation_id]["provenance"]["evidence_chunk_id"]
                for citation_id in claim.get("citation_ids") or []
                if citation_id in citation_by_id
            ]
            return (
                min((rank_by_chunk.get(chunk_id, 999) for chunk_id in ids), default=999),
                str(claim.get("claim_id") or ""),
            )

        ordered_claims = sorted(result["claims"], key=claim_rank)
        claims: list[dict[str, Any]] = []
        seen_text: set[str] = set()
        for claim in ordered_claims:
            text = str(claim.get("text") or "").strip()
            claim_paths = [
                citation_by_id[citation_id]["provenance"]["section_path"]
                for citation_id in claim.get("citation_ids") or []
                if citation_id in citation_by_id
            ]
            flattened_path = tuple(
                value for path in claim_paths for value in path if isinstance(value, str)
            )
            information = classify_candidate(
                text,
                question=self._ranking_question(scope),
                title_path=flattened_path,
            )
            text_key = _compact(text)
            if information.low_information or not text_key or text_key in seen_text:
                continue
            seen_text.add(text_key)
            claims.append(claim)
            if len(claims) >= MAX_TEXT_REPAIR_CLAIMS:
                break
        answer = "\n".join(str(claim["text"]).strip() for claim in claims if claim.get("text"))
        if not answer or len(answer) > MAX_V8_ANSWER_CHARS:
            return None
        used_citation_ids = {
            citation_id for claim in claims for citation_id in claim.get("citation_ids") or []
        }
        citations = tuple(
            citation for citation in result["citations"] if citation["citation_id"] in used_citation_ids
        )
        decision = classify_answer(
            answer,
            question=question,
            title_paths=_citation_paths(citations),
        )
        if decision.low_information:
            return None
        return _TextRepair(
            answer=answer,
            citations=citations,
            chunk_ids=tuple(row["chunk_id"] for row in provider_chunks),
            executor_trace=dict(result.get("trace") or {}),
        )

    def prepare(
        self,
        *,
        scope: Mapping[str, Any],
        base_answer: Mapping[str, Any],
        base_trace: Mapping[str, Any],
    ) -> dict[str, Any]:
        citations = [
            dict(value) for value in base_answer.get("citations") or [] if isinstance(value, Mapping)
        ]
        title_paths = _citation_paths(citations)
        lowinfo = classify_answer(
            str(base_answer.get("answer_text") or ""),
            question=str(scope["question"]),
            title_paths=title_paths,
        )
        mismatch, mismatch_reason = _intent_mismatch(str(scope["question"]), title_paths)
        table_groups = self.table_surface.groups(scope)
        table_decision = decide_table_upgrade(
            document_id=str(scope["document_id"]),
            report_year=int(scope["report_year"]),
            question=str(scope["question"]),
            answer_text=str(base_answer.get("answer_text") or ""),
            table_groups=table_groups,
            status=str(base_answer.get("status") or ""),
            citations=citations,
        )
        negative_audit = None
        groups = sorted(_question_groups(str(scope["question"])))
        supported = tuple(group for group in groups if group in SUPPORTED_NEGATIVE_TOPIC_GROUPS)
        if supported:
            source = self.source_index.source(str(scope["document_id"]))
            negative_audit = audit_negative_evidence(
                document_id=str(scope["document_id"]),
                source_markdown=source.portable_path,
                document_sha256=source.sha256,
                lines=source.lines,
                heading_paths=source.heading_paths,
                topic_groups=supported,
                searched_aliases=_group_aliases(supported),
            )
        return {
            "scope": dict(scope),
            "base_answer": dict(base_answer),
            "base_trace_hash": base_trace.get("trace_hash"),
            "base_lowinfo": lowinfo.as_dict(),
            "base_intent_mismatch": mismatch,
            "base_intent_reason": mismatch_reason,
            "table_decision": table_decision,
            "table_groups": table_groups,
            "negative_audit": negative_audit,
            "generic_scoped_checkbox": self._generic_scoped_checkbox(scope=scope),
            "ranked_party_disclosure": (
                self._ranked_party_disclosure(scope=scope)
                or self._major_contract_disclosure(scope=scope)
            ),
            "text_repair": self._text_repair(scope=scope, base_trace=base_trace),
        }

    def materialize(
        self,
        prepared: Mapping[str, Any],
        features: frozenset[str],
    ) -> dict[str, Any]:
        scope = prepared["scope"]
        base = prepared["base_answer"]
        answer = str(base.get("answer_text") or "").strip()
        citations = [dict(value) for value in base.get("citations") or []]
        status = "ok" if answer else "not_found"
        errors: list[dict[str, Any]] = [] if answer else [{"failure_code": "EVIDENCE_UNAVAILABLE"}]
        numeric_authorizations: list[dict[str, Any]] = []
        selected_source = "v7_baseline"
        selected_group_ids: list[str] = []
        reasons = [prepared["base_intent_reason"], prepared["base_lowinfo"]["reason"]]
        negative_projection = None
        text_projection = None

        audit = prepared.get("negative_audit")
        if "negative" in features and audit is not None:
            wording = recommended_negative_wording(audit)
            if wording is not None:
                answer = f"根据{scope['report_year']}年{scope['company']}年报，{wording}"
                if audit.safe_claim_scope == "target_section":
                    finding = audit.scoped_checkbox_negatives[0]
                    line_range = list(finding.line_range)
                    heading_path = list(finding.heading_path)
                    content_hash = hashlib.sha256(finding.text.encode("utf-8")).hexdigest()
                else:
                    line_range = [1, audit.line_count]
                    heading_path = []
                    content_hash = audit.document_sha256
                citation = {
                    "citation_id": "v8-cite-negative-" + audit.audit_id[-20:],
                    "source_kind": "negative_evidence_audit",
                    "candidate_id": audit.audit_id,
                    "document_id": audit.document_id,
                    "source_markdown": audit.source_markdown,
                    "line_range": line_range,
                    "heading_path": heading_path,
                    "content_sha256": content_hash,
                }
                citations = [citation]
                status = "ok"
                errors = []
                selected_source = "negative_evidence"
                selected_group_ids = [audit.audit_id]
                negative_projection = audit.to_dict()
                reasons.append(audit.decision)

        if selected_source == "v7_baseline" and "table" in features:
            decision = prepared["table_decision"]
            if decision.applied:
                answer = decision.answer_text
                selected_ids = set(decision.selected_citation_ids)
                citations = [
                    citation
                    for group in prepared["table_groups"]
                    for citation in group["citations"]
                    if citation["citation_id"] in selected_ids
                ]
                citation_map = {citation["citation_id"]: citation for citation in citations}
                citations = [citation_map[key] for key in sorted(citation_map)]
                numeric_authorizations = decision.numeric_authorization_mappings()
                status = "ok"
                errors = []
                selected_source = "table_upgrade"
                selected_group_ids = list(decision.selected_group_ids)
                reasons.extend(decision.reason_codes)

        party_disclosure = prepared.get("ranked_party_disclosure")
        if (
            selected_source == "v7_baseline"
            and "table" in features
            and party_disclosure is not None
            and (
                prepared["base_lowinfo"]["low_information"]
                or prepared["base_intent_mismatch"]
            )
        ):
            answer = party_disclosure.answer
            citations = [dict(party_disclosure.citation)]
            numeric_authorizations = [
                dict(value) for value in party_disclosure.numeric_authorizations
            ]
            status = "ok"
            errors = []
            selected_source = "source_numeric_disclosure"
            selected_group_ids = [party_disclosure.disclosure_id]
            reasons.append("source_line_numeric_disclosure_authorized")

        checkbox = prepared.get("generic_scoped_checkbox")
        if (
            selected_source == "v7_baseline"
            and "negative" in features
            and checkbox is not None
            and (
                prepared["base_lowinfo"]["low_information"]
                or prepared["base_intent_mismatch"]
            )
            and prepared.get("text_repair") is None
        ):
            heading = checkbox["heading_path"][-1]
            if checkbox["selected_label"] == "否":
                answer = (
                    f"根据{scope['report_year']}年{scope['company']}年报，"
                    f"“{heading}”项明确勾选“否”。"
                )
            else:
                answer = (
                    f"根据{scope['report_year']}年{scope['company']}年报，"
                    f"“{heading}”栏明确勾选“不适用”，该栏未披露相关事项。"
                )
            citation = {
                "citation_id": "v8-cite-" + checkbox["candidate_id"][-20:],
                "source_kind": "scoped_checkbox_negative",
                "candidate_id": checkbox["candidate_id"],
                "document_id": scope["document_id"],
                "source_markdown": checkbox["source_markdown"],
                "line_range": checkbox["line_range"],
                "heading_path": checkbox["heading_path"],
                "content_sha256": hashlib.sha256(checkbox["text"].encode("utf-8")).hexdigest(),
            }
            citations = [citation]
            status = "ok"
            errors = []
            selected_source = "scoped_checkbox_negative"
            selected_group_ids = [checkbox["candidate_id"]]
            reasons.append("generic_scoped_checkbox_negative")

        base_rejected = bool(
            prepared["base_lowinfo"]["low_information"] or prepared["base_intent_mismatch"]
        )
        if selected_source == "v7_baseline" and "text" in features and base_rejected:
            repair = prepared.get("text_repair")
            if repair is not None:
                answer = repair.answer
                citations = list(repair.citations)
                status = "ok"
                errors = []
                selected_source = "reranked_frozen_text"
                selected_group_ids = list(repair.chunk_ids)
                text_projection = dict(repair.executor_trace)
                reasons.append("base_replaced_by_reranked_text")
            else:
                answer = ""
                citations = []
                status = "not_found"
                errors = [{"failure_code": "EVIDENCE_UNAVAILABLE"}]
                selected_source = "low_information_rejected"
                reasons.append("no_safe_replacement")

        trace = {
            "schema_version": "finglmqa.experimental.type3_v8_trace.v1",
            "profile_version": TYPE3_V8_VERSION,
            "case_id": scope["case_id"],
            "document_id": scope["document_id"],
            "features": sorted(features),
            "base_trace_hash": prepared["base_trace_hash"],
            "base_lowinfo": prepared["base_lowinfo"],
            "base_intent_mismatch": prepared["base_intent_mismatch"],
            "base_intent_reason": prepared["base_intent_reason"],
            "selected_source": selected_source,
            "selected_group_ids": selected_group_ids,
            "decision_reasons": sorted(set(reasons)),
            "numeric_authorizations": numeric_authorizations,
            "negative_evidence_audit": negative_projection,
            "text_executor_trace": text_projection,
            "generative_llm_used": False,
        }
        trace["trace_hash"] = semantic_sha256(trace)
        return {
            "answer": answer,
            "status": status,
            "errors": errors,
            "warnings": [],
            "citations": citations,
            "trace": trace,
        }


__all__ = [
    "FrozenEvidenceCorpus",
    "MAX_TEXT_REPAIR_CHUNKS",
    "TYPE3_V8_STAGES",
    "TYPE3_V8_VERSION",
    "Type3V8Enhancer",
    "V8TableSurface",
]
