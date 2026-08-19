"""Deterministic, provenance-first evidence fusion for Type 3 QA.

This module is deliberately model-free.  It accepts only evidence already
scoped to one corpus/document, projects every item to a common schema, fuses
integer RRF signals, and composes an answer-safe evidence packet.  A model may
inspect the packet later, but it cannot create evidence or numeric literals.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
import re
from typing import Any, Iterable, Mapping, Sequence

from finglmqa.type3_tabgr_retriever import (
    Type3TabGRCandidate,
    numeric_fragments,
    safe_numeric_projection,
)


FUSION_SCHEMA = "finglmqa.type3.a2rag_tabgr.evidence_candidate.v1"
TRACE_SCHEMA = "finglmqa.type3.a2rag_tabgr.semantic_trace.v1"
PLANNER_VERSION = "type3-deterministic-facet-planner-v1"
FUSION_VERSION = "type3-a2rag-tabgr-fusion-v1"
COMPOSER_VERSION = "type3-programmatic-answer-safe-composer-v1"
RRF_SCALE = 1_000_000_000
RRF_K = 60

_YEAR_RE = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
_CLAUSE_SPLIT_RE = re.compile(r"[，。；;！？?!]+")
_METRIC_SUFFIX_RE = re.compile(
    r"(?:情况|原因|变化|变动|分析|说明|如何|是什么|有哪些|多少|是否|请|简要|"
    r"根据|年度报告|年报|公司|该公司|本公司|的)+$"
)
_QUESTION_PREFIX_RE = re.compile(
    r"^(?:根据|请|结合|从|简要|详细|分析|说明|介绍|概述|回答|查询)+"
)
_NUMBER_RE = re.compile(r"(?<![A-Za-z])[-+]?\d[\d,，. ]*(?:%|％)?")
_ALLOWED_INPUT_FIELDS = frozenset({"corpus_id", "question_id", "question", "document_id"})


class Type3FusionError(ValueError):
    """Raised when evidence or runtime inputs violate a fail-closed contract."""


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def semantic_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def validate_generator_input(value: Mapping[str, Any]) -> dict[str, str]:
    """Accept the exact online QA boundary and reject annotation-like fields."""

    keys = set(value)
    if keys != _ALLOWED_INPUT_FIELDS:
        raise Type3FusionError(
            "generator boundary forbids benchmark annotations or extra fields; "
            f"fields must be exactly {sorted(_ALLOWED_INPUT_FIELDS)!r}"
        )
    result = {key: value.get(key) for key in sorted(_ALLOWED_INPUT_FIELDS)}
    if any(not isinstance(item, str) or not item.strip() for item in result.values()):
        raise Type3FusionError("all generator input fields must be non-empty strings")
    return {key: str(item).strip() for key, item in result.items()}


@dataclass(frozen=True)
class Facet:
    facet_id: str
    query: str
    kind: str
    ordinal: int

    def as_mapping(self) -> dict[str, Any]:
        return {
            "facet_id": self.facet_id,
            "query": self.query,
            "kind": self.kind,
            "ordinal": self.ordinal,
        }


def _clean_clause(value: str) -> str:
    text = " ".join(value.split()).strip(" ：:，,。；;！？?!（）()[]【】")
    text = _QUESTION_PREFIX_RE.sub("", text)
    return _METRIC_SUFFIX_RE.sub("", text).strip(" ：:，,。；;！？?!")


def plan_facets(question: str, *, max_facets: int = 6) -> list[Facet]:
    """Create one to six deterministic query facets from the question only."""

    if not isinstance(question, str) or not question.strip():
        raise Type3FusionError("question must be non-empty")
    if not isinstance(max_facets, int) or not 1 <= max_facets <= 6:
        raise Type3FusionError("max_facets must be in [1, 6]")
    normalized = " ".join(question.split())
    proposals: list[tuple[str, str]] = [("question", normalized)]
    clauses = [_clean_clause(value) for value in _CLAUSE_SPLIT_RE.split(normalized)]
    clauses = [value for value in clauses if len(value) >= 2 and value != normalized]
    for clause in clauses:
        proposals.append(("clause", clause))

    years = list(dict.fromkeys(_YEAR_RE.findall(normalized)))
    without_boilerplate = re.sub(
        r"(?:根据|来自|年度报告|年报|请|简要|详细|分析|说明|介绍|概述|回答|查询|"
        r"该公司|本公司|公司|股份有限公司|有限责任公司)",
        " ",
        normalized,
    )
    metric = _clean_clause(_YEAR_RE.sub(" ", without_boilerplate))
    if len(metric) >= 2:
        proposals.append(("metric", metric))
        for year in years[:2]:
            proposals.append(("metric_year", f"{metric} {year}"))

    if any(term in normalized for term in ("原因", "为何", "为什么", "变动", "变化")):
        proposals.append(("causal", f"{metric or normalized} 原因 影响因素"))
    if any(term in normalized for term in ("增长", "下降", "增加", "减少", "同比", "变动")):
        proposals.append(("comparison", f"{metric or normalized} 本期 上期 同比 变动"))
    if any(term in normalized for term in ("核心竞争力", "优势", "竞争")):
        proposals.append(("topic", "核心竞争力 技术 品牌 渠道 研发 人才"))
    if any(term in normalized for term in ("风险", "不利", "困难")):
        proposals.append(("topic", f"{metric or normalized} 风险 不利因素 应对措施"))

    facets: list[Facet] = []
    seen: set[str] = set()
    for kind, raw in proposals:
        query = " ".join(raw.split()).strip()
        fingerprint = re.sub(r"\s+", "", query).lower()
        if not query or fingerprint in seen:
            continue
        seen.add(fingerprint)
        ordinal = len(facets)
        facets.append(
            Facet(
                facet_id=f"facet-{ordinal + 1}-{sha256_text(query)[:10]}",
                query=query,
                kind=kind,
                ordinal=ordinal,
            )
        )
        if len(facets) >= max_facets:
            break
    if not facets:
        raise Type3FusionError("facet planner produced no query")
    return facets


@dataclass(frozen=True)
class RankSignal:
    facet_id: str
    channel: str
    rank: int
    weight: int

    def as_mapping(self) -> dict[str, Any]:
        return {
            "facet_id": self.facet_id,
            "channel": self.channel,
            "rank": self.rank,
            "weight": self.weight,
        }


@dataclass(frozen=True)
class EvidenceCandidate:
    candidate_id: str
    corpus_id: str
    document_id: str
    route: str
    evidence_type: str
    display_text: str
    answer_safe_text: str
    source_markdown: str
    line_range: tuple[int, int]
    char_range: tuple[int, int] | None
    byte_range: tuple[int, int] | None
    source_sha256: str
    content_sha256: str
    table_sha256: str | None
    cell_coordinates: tuple[tuple[int, int], ...]
    origin_cell_hashes: tuple[str, ...]
    heading_path: tuple[str, ...]
    table_id: str | None
    adjacent_table_ids: tuple[str, ...]
    row_path: tuple[str, ...]
    semantic_states: Mapping[str, Any]
    numeric_authorizations: tuple[Mapping[str, Any], ...]
    unauthorized_numeric_values: tuple[str, ...]
    rank_signals: tuple[RankSignal, ...] = ()
    complement_ids: tuple[str, ...] = ()
    fusion_score: int = 0
    conflict_status: str = "clear"
    answer_eligible: bool = True

    def core_fingerprint(self) -> str:
        return semantic_sha256(
            {
                "candidate_id": self.candidate_id,
                "corpus_id": self.corpus_id,
                "document_id": self.document_id,
                "route": self.route,
                "evidence_type": self.evidence_type,
                "display_text": self.display_text,
                "source_markdown": self.source_markdown,
                "line_range": list(self.line_range),
                "char_range": list(self.char_range) if self.char_range else None,
                "byte_range": list(self.byte_range) if self.byte_range else None,
                "source_sha256": self.source_sha256,
                "content_sha256": self.content_sha256,
                "table_sha256": self.table_sha256,
                "cell_coordinates": [list(value) for value in self.cell_coordinates],
                "origin_cell_hashes": list(self.origin_cell_hashes),
                "table_id": self.table_id,
            }
        )

    def as_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": FUSION_SCHEMA,
            "candidate_id": self.candidate_id,
            "corpus_id": self.corpus_id,
            "document_id": self.document_id,
            "route": self.route,
            "evidence_type": self.evidence_type,
            "display_text": self.display_text,
            "answer_safe_text": self.answer_safe_text,
            "source_markdown": self.source_markdown,
            "line_range": list(self.line_range),
            "char_range": list(self.char_range) if self.char_range else None,
            "byte_range": list(self.byte_range) if self.byte_range else None,
            "source_sha256": self.source_sha256,
            "content_sha256": self.content_sha256,
            "table_sha256": self.table_sha256,
            "cell_coordinates": [list(value) for value in self.cell_coordinates],
            "origin_cell_hashes": list(self.origin_cell_hashes),
            "heading_path": list(self.heading_path),
            "table_id": self.table_id,
            "adjacent_table_ids": list(self.adjacent_table_ids),
            "row_path": list(self.row_path),
            "semantic_states": dict(self.semantic_states),
            "numeric_authorizations": [dict(value) for value in self.numeric_authorizations],
            "unauthorized_numeric_values": list(self.unauthorized_numeric_values),
            "rank_signals": [value.as_mapping() for value in self.rank_signals],
            "complement_ids": list(self.complement_ids),
            "fusion_score": str(self.fusion_score),
            "conflict_status": self.conflict_status,
            "answer_eligible": self.answer_eligible,
        }


def _dynamic_text_authorizations(atom: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    content = str(atom.get("content") or "")
    rows = []
    for literal in sorted(set(numeric_fragments(content))):
        unsigned = {
            "schema_version": "finglmqa.type3.a2rag.literal_authorization.v1",
            "corpus_id": str(atom["corpus_id"]),
            "document_id": str(atom["document_id"]),
            "atom_id": str(atom["atom_id"]),
            "source_markdown": str(atom["source_markdown"]),
            "source_sha256": str(atom["source_sha256"]),
            "content_sha256": str(atom["content_sha256"]),
            "literal": literal,
            "literal_sha256": sha256_text(literal),
            "authorization_rule": "literal_occurs_verbatim_in_exact_atom",
        }
        if literal not in content:
            raise Type3FusionError("text numeric authorization is not literal in atom")
        rows.append(
            {
                **unsigned,
                "authorization_id": "t3a2-lit-" + semantic_sha256(unsigned)[:24],
                "allowed_renderings": [literal],
            }
        )
    return tuple(rows)


def candidate_from_atom(
    atom: Mapping[str, Any],
    *,
    facet_id: str,
    channel: str,
    rank: int,
    weight: int,
) -> EvidenceCandidate:
    content = str(atom.get("content") or "")
    if not content:
        raise Type3FusionError("empty text atom")
    if sha256_text(content) != atom.get("content_sha256"):
        raise Type3FusionError("text atom content hash mismatch")
    authorizations = _dynamic_text_authorizations(atom)
    return EvidenceCandidate(
        candidate_id=str(atom["atom_id"]),
        corpus_id=str(atom["corpus_id"]),
        document_id=str(atom["document_id"]),
        route="text",
        evidence_type=f"text_{atom.get('atom_kind') or 'atom'}",
        display_text=content,
        answer_safe_text=content,
        source_markdown=str(atom["source_markdown"]),
        line_range=tuple(int(value) for value in atom["line_range"]),
        char_range=tuple(int(value) for value in atom["char_range"]),
        byte_range=tuple(int(value) for value in atom["byte_range"]),
        source_sha256=str(atom["source_sha256"]),
        content_sha256=str(atom["content_sha256"]),
        table_sha256=None,
        cell_coordinates=(),
        origin_cell_hashes=(),
        heading_path=tuple(str(value) for value in atom.get("heading_path") or ()),
        table_id=None,
        adjacent_table_ids=tuple(str(value) for value in atom.get("adjacent_table_ids") or ()),
        row_path=(),
        semantic_states={},
        numeric_authorizations=authorizations,
        unauthorized_numeric_values=(),
        rank_signals=(RankSignal(facet_id, channel, rank, weight),),
    )


def _contains_conflict(value: object) -> bool:
    if isinstance(value, Mapping):
        if value.get("status") == "conflict":
            return True
        return any(_contains_conflict(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_conflict(item) for item in value)
    return False


def _validate_table_authorizations(
    candidate: Type3TabGRCandidate,
    *,
    no_fact_join: bool,
) -> tuple[tuple[Mapping[str, Any], ...], str, tuple[str, ...], str]:
    display = candidate.display_text
    conflict = _contains_conflict(candidate.semantic_states)
    accepted: list[Mapping[str, Any]] = []
    seen_ids: dict[str, str] = {}
    if not no_fact_join and not conflict:
        for raw in candidate.numeric_authorizations:
            authorization = dict(raw)
            if (
                authorization.get("corpus_id") != candidate.corpus_id
                or authorization.get("document_id") != candidate.document_id
                or authorization.get("table_id") != candidate.table_id
                or authorization.get("source_markdown") != candidate.source_markdown
                or list(authorization.get("table_line_range") or ()) != list(candidate.line_range)
            ):
                raise Type3FusionError("table authorization provenance mismatch")
            allowed = authorization.get("allowed_renderings")
            if not isinstance(allowed, list) or not allowed:
                raise Type3FusionError("table authorization has no exact rendering")
            if any(not isinstance(value, str) or value not in display for value in allowed):
                raise Type3FusionError("table authorization rendering is absent from evidence")
            authorization_id = str(authorization.get("authorization_id") or "")
            fingerprint = semantic_sha256(authorization)
            previous = seen_ids.setdefault(authorization_id, fingerprint)
            if not authorization_id or previous != fingerprint:
                raise Type3FusionError("conflicting table authorization id")
            accepted.append(authorization)
    allowed_renderings = sorted(
        {
            rendering
            for authorization in accepted
            for rendering in authorization["allowed_renderings"]
        },
        key=lambda value: (-len(value), value),
    )
    answer_safe_text = safe_numeric_projection(display, allowed_renderings)
    remaining = tuple(
        literal for literal in numeric_fragments(answer_safe_text) if literal not in allowed_renderings
    )
    if remaining:
        raise Type3FusionError("unauthorized numeric literal survived table projection")
    unauthorized = tuple(
        literal for literal in numeric_fragments(display) if literal not in allowed_renderings
    )
    return (
        tuple(accepted),
        answer_safe_text,
        unauthorized,
        "semantic_conflict_redacted" if conflict else "clear",
    )


def candidate_from_table(
    candidate: Type3TabGRCandidate,
    *,
    provenance: Mapping[str, Any],
    facet_id: str,
    channel: str,
    rank: int,
    weight: int,
    no_fact_join: bool = False,
) -> EvidenceCandidate:
    source_sha256 = str(provenance.get("source_sha256") or "")
    table_sha256 = str(provenance.get("table_sha256") or "")
    source_markdown = str(provenance.get("source_markdown") or "")
    raw_coordinates = provenance.get("cell_coordinates")
    raw_origin_hashes = provenance.get("origin_cell_hashes")
    if len(source_sha256) != 64 or len(table_sha256) != 64:
        raise Type3FusionError("table evidence lacks source/table SHA-256 binding")
    if source_markdown != candidate.source_markdown:
        raise Type3FusionError("table source path differs from corpus provenance")
    if provenance.get("table_id") != candidate.table_id:
        raise Type3FusionError("table id differs from rich row provenance")
    if list(provenance.get("table_line_range") or ()) != list(candidate.line_range):
        raise Type3FusionError("table line range differs from rich row provenance")
    if not isinstance(raw_coordinates, (list, tuple)) or not raw_coordinates:
        raise Type3FusionError("table row evidence lacks cell coordinates")
    if not isinstance(raw_origin_hashes, (list, tuple)) or not raw_origin_hashes:
        raise Type3FusionError("table row evidence lacks origin-cell hashes")
    cell_coordinates = tuple(
        tuple(int(value) for value in coordinate)
        for coordinate in raw_coordinates
    )
    if any(len(value) != 2 or min(value) < 0 for value in cell_coordinates):
        raise Type3FusionError("invalid table cell coordinate")
    origin_cell_hashes = tuple(str(value) for value in raw_origin_hashes)
    if any(len(value) != 64 for value in origin_cell_hashes):
        raise Type3FusionError("invalid origin-cell hash")
    authorizations, safe_text, unauthorized, conflict_status = _validate_table_authorizations(
        candidate, no_fact_join=no_fact_join
    )
    for authorization in authorizations:
        if authorization.get("table_sha256") != table_sha256:
            raise Type3FusionError("table authorization hash differs from row provenance")
        raw_coordinate = authorization.get("cell_coordinate")
        if (
            not isinstance(raw_coordinate, list)
            or tuple(int(value) for value in raw_coordinate) not in cell_coordinates
        ):
            raise Type3FusionError("table authorization cell is outside row provenance")
    return EvidenceCandidate(
        candidate_id=candidate.evidence_id,
        corpus_id=candidate.corpus_id,
        document_id=candidate.document_id,
        route="table",
        evidence_type=candidate.evidence_type,
        display_text=candidate.display_text,
        answer_safe_text=safe_text,
        source_markdown=candidate.source_markdown,
        line_range=candidate.line_range,
        char_range=None,
        byte_range=None,
        source_sha256=source_sha256,
        content_sha256=sha256_text(candidate.display_text),
        table_sha256=table_sha256,
        cell_coordinates=cell_coordinates,
        origin_cell_hashes=origin_cell_hashes,
        heading_path=candidate.heading_path,
        table_id=candidate.table_id,
        adjacent_table_ids=(),
        row_path=candidate.row_path,
        semantic_states=dict(candidate.semantic_states),
        numeric_authorizations=authorizations,
        unauthorized_numeric_values=unauthorized,
        rank_signals=(RankSignal(facet_id, channel, rank, weight),),
        conflict_status=conflict_status,
    )


def merge_candidates(values: Iterable[EvidenceCandidate]) -> list[EvidenceCandidate]:
    """Deduplicate evidence IDs while requiring identical core provenance."""

    merged: dict[str, EvidenceCandidate] = {}
    for candidate in values:
        previous = merged.get(candidate.candidate_id)
        if previous is None:
            merged[candidate.candidate_id] = candidate
            continue
        if previous.core_fingerprint() != candidate.core_fingerprint():
            raise Type3FusionError(f"candidate provenance conflict: {candidate.candidate_id}")
        signals = {
            (value.facet_id, value.channel, value.rank, value.weight): value
            for value in (*previous.rank_signals, *candidate.rank_signals)
        }
        authorizations: dict[str, Mapping[str, Any]] = {}
        authorization_hashes: dict[str, str] = {}
        for authorization in (*previous.numeric_authorizations, *candidate.numeric_authorizations):
            authorization_id = str(authorization["authorization_id"])
            fingerprint = semantic_sha256(dict(authorization))
            prior = authorization_hashes.setdefault(authorization_id, fingerprint)
            if prior != fingerprint:
                raise Type3FusionError(
                    f"conflicting authorization provenance: {authorization_id}"
                )
            authorizations[authorization_id] = authorization
        merged[candidate.candidate_id] = replace(
            previous,
            numeric_authorizations=tuple(authorizations[key] for key in sorted(authorizations)),
            rank_signals=tuple(
                signals[key]
                for key in sorted(signals, key=lambda item: (item[0], item[1], item[2], item[3]))
            ),
            conflict_status=(
                previous.conflict_status
                if previous.conflict_status == candidate.conflict_status
                else "merged_conflict_redacted"
            ),
        )
    return list(merged.values())


def _rrf_score(candidate: EvidenceCandidate) -> int:
    return sum(
        signal.weight * (RRF_SCALE // (RRF_K + signal.rank))
        for signal in candidate.rank_signals
    )


def _apply_adjacency(
    candidates: Sequence[EvidenceCandidate],
) -> list[EvidenceCandidate]:
    table_by_id: dict[str, list[str]] = {}
    for candidate in candidates:
        if candidate.route == "table" and candidate.table_id:
            table_by_id.setdefault(candidate.table_id, []).append(candidate.candidate_id)
    complemented: list[EvidenceCandidate] = []
    for candidate in candidates:
        complement_ids: set[str] = set()
        if candidate.route == "text":
            for table_id in candidate.adjacent_table_ids:
                complement_ids.update(table_by_id.get(table_id, ()))
        elif candidate.table_id:
            for other in candidates:
                if candidate.table_id in other.adjacent_table_ids:
                    complement_ids.add(other.candidate_id)
        boost = len(complement_ids) * (RRF_SCALE // 500)
        complemented.append(
            replace(
                candidate,
                complement_ids=tuple(sorted(complement_ids)),
                fusion_score=_rrf_score(candidate) + boost,
            )
        )
    return complemented


def _rank_fused(candidates: Iterable[EvidenceCandidate]) -> list[EvidenceCandidate]:
    return sorted(
        candidates,
        key=lambda value: (
            -value.fusion_score,
            0 if value.route == "table" else 1,
            value.candidate_id,
        ),
    )


def fuse_evidence(
    candidates: Iterable[EvidenceCandidate],
    *,
    facets: Sequence[Facet],
    max_candidates: int = 18,
    route_quota: bool = True,
    adjacency: bool = True,
) -> list[EvidenceCandidate]:
    """Fuse stable integer RRF signals with per-facet/per-route reservations."""

    if not 1 <= max_candidates <= 48:
        raise Type3FusionError("max_candidates must be in [1, 48]")
    merged = merge_candidates(candidates)
    if not merged:
        return []
    scored = (
        _apply_adjacency(merged)
        if adjacency
        else [replace(value, fusion_score=_rrf_score(value)) for value in merged]
    )
    ranked = _rank_fused(scored)
    if not route_quota:
        return ranked[:max_candidates]

    reserved: list[EvidenceCandidate] = []
    reserved_ids: set[str] = set()
    available_routes = tuple(
        route for route in ("text", "table") if any(value.route == route for value in ranked)
    )
    for facet in facets:
        for route in available_routes:
            match = next(
                (
                    value
                    for value in ranked
                    if value.candidate_id not in reserved_ids
                    and value.route == route
                    and any(signal.facet_id == facet.facet_id for signal in value.rank_signals)
                ),
                None,
            )
            if match is not None and len(reserved) < max_candidates:
                reserved.append(match)
                reserved_ids.add(match.candidate_id)
    for candidate in ranked:
        if len(reserved) >= max_candidates:
            break
        if candidate.candidate_id not in reserved_ids:
            reserved.append(candidate)
            reserved_ids.add(candidate.candidate_id)
    return reserved


def compose_answer_safe_packet(
    candidates: Sequence[EvidenceCandidate],
    *,
    max_items: int = 8,
) -> dict[str, Any]:
    """Compose deterministic evidence text without inventing numeric literals."""

    selected = list(candidates[:max_items])
    texts = [value.answer_safe_text.strip() for value in selected if value.answer_safe_text.strip()]
    answer_safe_text = "\n\n－ ".join(["以下为可核验的原文证据：", *texts])
    citations = [
        {
            "candidate_id": value.candidate_id,
            "route": value.route,
            "source_markdown": value.source_markdown,
            "line_range": list(value.line_range),
            "char_range": list(value.char_range) if value.char_range else None,
            "byte_range": list(value.byte_range) if value.byte_range else None,
            "table_id": value.table_id,
            "source_sha256": value.source_sha256,
            "table_sha256": value.table_sha256,
            "cell_coordinates": [list(item) for item in value.cell_coordinates],
            "origin_cell_hashes": list(value.origin_cell_hashes),
            "authorization_ids": [
                str(item["authorization_id"]) for item in value.numeric_authorizations
            ],
        }
        for value in selected
    ]
    allowed_literals = {
        rendering
        for value in selected
        for authorization in value.numeric_authorizations
        for rendering in authorization.get("allowed_renderings") or ()
    }
    unsupported = [
        literal
        for literal in numeric_fragments(answer_safe_text)
        if literal not in allowed_literals
    ]
    if unsupported:
        raise Type3FusionError(
            f"composer introduced unsupported numeric literals: {sorted(set(unsupported))!r}"
        )
    return {
        "composer_version": COMPOSER_VERSION,
        "answer_safe_text": answer_safe_text,
        "citations": citations,
        "selected_candidate_ids": [value.candidate_id for value in selected],
        "unsupported_numeric_literals": [],
    }


def evaluate_shadow_id_selector(
    deterministic_candidates: Sequence[EvidenceCandidate],
    response: object,
    *,
    max_selected: int = 8,
    timed_out: bool = False,
) -> dict[str, Any]:
    """Validate an optional Qwen ID-only response without affecting formal output."""

    fallback = [value.candidate_id for value in deterministic_candidates[:max_selected]]
    if timed_out:
        return {
            "status": "fallback_timeout",
            "selected_ids": fallback,
            "affected_semantic_trace": False,
        }
    try:
        parsed = json.loads(response) if isinstance(response, str) else response
    except json.JSONDecodeError:
        parsed = None
    if not isinstance(parsed, Mapping) or set(parsed) != {"selected_ids"}:
        return {
            "status": "fallback_free_text_or_schema",
            "selected_ids": fallback,
            "affected_semantic_trace": False,
        }
    selected = parsed.get("selected_ids")
    if (
        not isinstance(selected, list)
        or len(selected) > max_selected
        or any(not isinstance(value, str) for value in selected)
        or len(set(selected)) != len(selected)
    ):
        return {
            "status": "fallback_invalid_cardinality",
            "selected_ids": fallback,
            "affected_semantic_trace": False,
        }
    allowed = {value.candidate_id for value in deterministic_candidates}
    if any(value not in allowed for value in selected):
        return {
            "status": "fallback_unknown_id",
            "selected_ids": fallback,
            "affected_semantic_trace": False,
        }
    return {
        "status": "shadow_valid",
        "selected_ids": selected,
        "affected_semantic_trace": False,
    }


__all__ = [
    "COMPOSER_VERSION",
    "EvidenceCandidate",
    "FUSION_SCHEMA",
    "FUSION_VERSION",
    "Facet",
    "PLANNER_VERSION",
    "RankSignal",
    "TRACE_SCHEMA",
    "Type3FusionError",
    "canonical_json_bytes",
    "candidate_from_atom",
    "candidate_from_table",
    "compose_answer_safe_packet",
    "evaluate_shadow_id_selector",
    "fuse_evidence",
    "merge_candidates",
    "plan_facets",
    "semantic_sha256",
    "sha256_text",
    "validate_generator_input",
]
