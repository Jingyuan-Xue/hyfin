"""Hybrid recall and coverage-planning Qwen experiment for Type 3.

This opt-in v10 profile keeps the v9 security boundary: Qwen may plan
document-scoped retrieval and return supplied fragment identifiers, but it may
not author answer text, numbers, or citations.  The two changes are deliberately
isolated here so the frozen v8/v9 implementations remain byte-for-byte intact:

* a larger, facet-balanced union of dense and document-local sparse recall;
* an explicit core/supporting facet plan followed by per-facet coverage output.

Benchmark answers, keywords, case IDs, company-specific rules and year-specific
rules are not runtime inputs.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from .contracts import semantic_sha256
from .type3_qwen36_faceted_v9 import (
    EvidenceFragment,
    FacetedV9Error,
    SELECTOR_SEEDS,
    SourceLexicalIndex,
    Type3Qwen36FacetedV9,
    _fragment_id,
    _generation_body,
    _lexical_score,
    _LOW_INFO_RE,
    _response_content,
    _split_fragments,
    compact_text,
    normalize_text,
)


PROFILE_VERSION = "type3-qwen36-hybrid-coverage-v10"
PROMPT_VERSION = "type3-qwen36-core-coverage-planner-v2"
RESULT_SCHEMA = "finglmqa.experimental.type3_qwen36_coverage_v10.result.v1"

MAX_FACETS = 6
MAX_CANDIDATES = 36
MAX_DENSE_CHUNKS_PER_FACET = 15
MAX_DENSE_FRAGMENTS_PER_FACET = 12
MAX_SPARSE_FRAGMENTS_PER_FACET = 12
MAX_NEIGHBORS_PER_FACET = 4
MAX_IDS_PER_FACET = 3

PLAN_SYSTEM_PROMPT = (
    "你是通用年报问答的检索与覆盖规划器。把问题拆成1到6个不重复的关注面。"
    "简单单一问题只生成1到3个关注面；只有问题明确并列询问多个事项时才可生成4到6个，"
    "supporting关注面最多2个。label必须是关注面的简短主题名称，不能填写core或supporting。"
    "每个关注面必须标为core或supporting：直接回答用户问题不可缺少的是core，"
    "解释、背景或延伸材料才是supporting。公司名、年份、‘根据年报’不得单独成为关注面。"
    "query要使用年报中可能出现的规范表述，允许对问题概念作通用同义改写，但不得猜测"
    "任何公司事实、数值、结论、人物或引用。并列询问的客户与供应商、三类现金流、"
    "董监高变动、处罚与整改等必须拆开覆盖。"
)

COVERAGE_SYSTEM_PROMPT = (
    "你是年报证据覆盖规划器。程序已给出唯一年报内的候选原文和不可删除的基线答案。"
    "你只能逐关注面返回候选fragment_id，不能写答案、数字、理由或引用。"
    "对每个关注面必须且只能给出一个状态："
    "evidence表示候选中存在直接、完整且能新增信息的证据；baseline表示基线已充分覆盖；"
    "not_found表示候选与基线都未可靠覆盖。"
    "core关注面优先，evidence时选择1到3个最小充分且互补的片段；"
    "不要选择只含标题、目录、单位、模板、截断句、泛泛战略或仅与主题擦边的片段。"
    "同一fragment可覆盖多个关注面，但不得创建ID。"
)

PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["facets"],
    "properties": {
        "facets": {
            "type": "array",
            "minItems": 1,
            "maxItems": MAX_FACETS,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["facet_id", "label", "query", "priority"],
                "properties": {
                    "facet_id": {"type": "string", "pattern": "^f[1-6]$"},
                    "label": {"type": "string", "minLength": 2, "maxLength": 40},
                    "query": {"type": "string", "minLength": 2, "maxLength": 120},
                    "priority": {"type": "string", "enum": ["core", "supporting"]},
                },
            },
        }
    },
}

COVERAGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["facet_coverage"],
    "properties": {
        "facet_coverage": {
            "type": "array",
            "minItems": 1,
            "maxItems": MAX_FACETS,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["facet_id", "status", "fragment_ids"],
                "properties": {
                    "facet_id": {"type": "string", "pattern": "^f[1-6]$"},
                    "status": {
                        "type": "string",
                        "enum": ["evidence", "baseline", "not_found"],
                    },
                    "fragment_ids": {
                        "type": "array",
                        "maxItems": MAX_IDS_PER_FACET,
                        "items": {"type": "string", "minLength": 1},
                    },
                },
            },
        }
    },
}

PROMPT_CONTRACT_HASH = semantic_sha256({
    "profile_version": PROFILE_VERSION,
    "prompt_version": PROMPT_VERSION,
    "plan_system_prompt": PLAN_SYSTEM_PROMPT,
    "coverage_system_prompt": COVERAGE_SYSTEM_PROMPT,
    "plan_schema": PLAN_SCHEMA,
    "coverage_schema": COVERAGE_SCHEMA,
    "selector_seeds": SELECTOR_SEEDS,
    "max_candidates": MAX_CANDIDATES,
    "dense_chunks_per_facet": MAX_DENSE_CHUNKS_PER_FACET,
    "sparse_fragments_per_facet": MAX_SPARSE_FRAGMENTS_PER_FACET,
})

_CJK_RUN_RE = re.compile(r"[\u3400-\u9fff]+")
_ASCII_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_SPARSE_STOP = frozenset({
    "根据", "年度", "报告", "年报", "公司", "情况", "什么", "哪些", "为何",
    "为什么", "如何", "请问", "简要", "分析", "概述", "介绍", "说明", "主要",
    "相关", "报告期", "本年度", "该公司", "是否", "以及", "及其",
})


def _sparse_tokens(value: str) -> list[str]:
    text = normalize_text(value).lower()
    tokens = list(_ASCII_TOKEN_RE.findall(text))
    for run in _CJK_RUN_RE.findall(text):
        tokens.extend(run[index : index + 2] for index in range(max(0, len(run) - 1)))
        tokens.extend(run[index : index + 3] for index in range(max(0, len(run) - 2)))
    return [token for token in tokens if token and token not in _SPARSE_STOP]


class SourceSparseIndex(SourceLexicalIndex):
    """Document-local BM25-ish sentence recall with source-neighbour expansion."""

    def search_hybrid(
        self,
        document_id: str,
        query: str,
        *,
        limit: int = MAX_SPARSE_FRAGMENTS_PER_FACET,
        neighbor_limit: int = MAX_NEIGHBORS_PER_FACET,
    ) -> tuple[list[tuple[EvidenceFragment, float]], list[tuple[EvidenceFragment, float]]]:
        rows = list(self.fragments(document_id))
        query_tokens = _sparse_tokens(query)
        if not rows or not query_tokens:
            return [], []
        query_counts = Counter(query_tokens)
        tokenized = [Counter(_sparse_tokens(" ".join((*row.heading_path, row.text)))) for row in rows]
        document_frequency: Counter[str] = Counter()
        wanted = set(query_counts)
        for counts in tokenized:
            document_frequency.update(wanted.intersection(counts))
        average_length = sum(sum(counts.values()) for counts in tokenized) / max(1, len(tokenized))
        scored: list[tuple[EvidenceFragment, float]] = []
        normalized_query = compact_text(query)
        for row, counts in zip(rows, tokenized):
            row_length = max(1, sum(counts.values()))
            score = 0.0
            for token, query_frequency in query_counts.items():
                frequency = counts.get(token, 0)
                if not frequency:
                    continue
                inverse_frequency = math.log(
                    1.0 + (len(rows) - document_frequency[token] + 0.5)
                    / (document_frequency[token] + 0.5)
                )
                denominator = frequency + 1.2 * (
                    0.25 + 0.75 * row_length / max(1.0, average_length)
                )
                score += query_frequency * inverse_frequency * frequency * 2.2 / denominator
            if not score:
                continue
            heading_key = compact_text(" ".join(row.heading_path))
            text_key = compact_text(row.text)
            if normalized_query and normalized_query in text_key:
                score += 2.0
            if normalized_query and normalized_query in heading_key:
                score += 1.25
            score += 0.35 * _lexical_score(query, row.text, row.heading_path)
            scored.append((row, round(score, 8)))
        scored.sort(key=lambda item: (-item[1], item[0].line_range, item[0].fragment_id))
        primary = scored[:limit]

        by_line: dict[int, list[EvidenceFragment]] = defaultdict(list)
        for row in rows:
            by_line[row.line_range[0]].append(row)
        primary_ids = {row.fragment_id for row, _ in primary}
        neighbours: list[tuple[EvidenceFragment, float]] = []
        neighbour_ids: set[str] = set()
        for row, score in primary[:4]:
            for line_number in (row.line_range[0] - 1, row.line_range[1] + 1):
                for candidate in by_line.get(line_number, []):
                    if (
                        candidate.fragment_id in primary_ids
                        or candidate.fragment_id in neighbour_ids
                        or candidate.heading_path != row.heading_path
                    ):
                        continue
                    neighbour_ids.add(candidate.fragment_id)
                    neighbours.append((candidate, round(score * 0.42, 8)))
        neighbours.sort(
            key=lambda item: (-item[1], item[0].line_range, item[0].fragment_id)
        )
        return primary, neighbours[:neighbor_limit]


class Type3Qwen36CoverageV10(Type3Qwen36FacetedV9):
    """Expanded hybrid recall plus strict per-facet coverage planning."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        root = Path(self.source_index.audit.root)
        self.source_index = SourceSparseIndex(root)

    def _plan(self, question: str) -> tuple[list[dict[str, str]], str]:
        body = _generation_body(
            model=self.model,
            messages=[
                {"role": "system", "content": PLAN_SYSTEM_PROMPT},
                {"role": "user", "content": f"问题：{question}"},
            ],
            schema_name="annual_report_core_coverage_facets",
            schema=PLAN_SCHEMA,
            seed=0,
        )
        try:
            value = _response_content(self.client.complete(body))
            facets = value.get("facets")
            if not isinstance(facets, list) or not 1 <= len(facets) <= MAX_FACETS:
                raise FacetedV9Error("facet count is invalid")
            checked: list[dict[str, str]] = []
            seen_queries: set[str] = set()
            for ordinal, row in enumerate(facets, 1):
                if not isinstance(row, Mapping) or set(row) != {
                    "facet_id", "label", "query", "priority"
                }:
                    raise FacetedV9Error("coverage facet shape is invalid")
                facet_id = str(row["facet_id"])
                label = normalize_text(str(row["label"]))
                query = normalize_text(str(row["query"]))
                priority = str(row["priority"])
                query_key = compact_text(query)
                if (
                    facet_id != f"f{ordinal}"
                    or not 2 <= len(label) <= 40
                    or not 2 <= len(query) <= 120
                    or priority not in {"core", "supporting"}
                    or not query_key
                    or query_key in seen_queries
                ):
                    raise FacetedV9Error("coverage facet value is invalid")
                seen_queries.add(query_key)
                checked.append({
                    "facet_id": facet_id,
                    "label": label,
                    "query": query,
                    "priority": priority,
                })
            if not any(row["priority"] == "core" for row in checked):
                checked[0]["priority"] = "core"
            return checked, "qwen_core_coverage_planned"
        except Exception:
            return [{
                "facet_id": "f1",
                "label": "问题核心信息",
                "query": normalize_text(question)[:120],
                "priority": "core",
            }], "question_core_fallback"

    def _candidates(
        self,
        *,
        document_id: str,
        facets: Sequence[Mapping[str, str]],
    ) -> tuple[list[EvidenceFragment], list[dict[str, Any]]]:
        merged: dict[str, EvidenceFragment] = {}
        retrieval_trace: list[dict[str, Any]] = []
        facet_ranked_ids: dict[str, list[str]] = {}

        for facet in facets:
            facet_id = str(facet["facet_id"])
            query = str(facet["query"])
            dense_result = self.retriever.retrieve_for_document(
                document_id, query, top_k=MAX_DENSE_CHUNKS_PER_FACET
            )
            chunks = dense_result.get("chunks") if isinstance(dense_result, Mapping) else None
            if not isinstance(chunks, list):
                chunks = []
            dense_pool: list[tuple[EvidenceFragment, float, int]] = []
            dense_ids: list[str] = []
            for fallback_rank, chunk in enumerate(chunks, 1):
                if not isinstance(chunk, Mapping) or chunk.get("document_id") != document_id:
                    continue
                content = str(chunk.get("content") or "")
                source_markdown = str(chunk.get("source_markdown") or "")
                line_range = chunk.get("line_range") or []
                if (
                    not content
                    or not source_markdown
                    or not isinstance(line_range, list)
                    or len(line_range) != 2
                ):
                    continue
                rank = int(chunk.get("rank") or fallback_rank)
                source_id = str(chunk.get("evidence_chunk_id") or "") or None
                if source_id:
                    dense_ids.append(source_id)
                heading_path = tuple(str(value) for value in chunk.get("section_path") or [])
                for text in _split_fragments(content):
                    if len(compact_text(text)) < 8 or _LOW_INFO_RE.fullmatch(text):
                        continue
                    lexical = _lexical_score(query, text, heading_path)
                    # RRF-like dense contribution plus direct lexical/heading agreement.
                    score = 0.65 / (5.0 + rank) + lexical
                    dense_pool.append((EvidenceFragment(
                        fragment_id=_fragment_id(document_id, source_markdown, line_range, text),
                        document_id=document_id,
                        text=text,
                        source_markdown=source_markdown,
                        line_range=(int(line_range[0]), int(line_range[1])),
                        heading_path=heading_path,
                        source_kind="expanded_dense_evidence_fragment",
                        source_content=content,
                        source_evidence_id=source_id,
                    ), round(score, 8), rank))
            dense_pool.sort(
                key=lambda item: (-item[1], item[2], item[0].line_range, item[0].fragment_id)
            )
            channel_ids: list[str] = []
            for candidate, score, rank in dense_pool[:MAX_DENSE_FRAGMENTS_PER_FACET]:
                self._merge_candidate(
                    merged, candidate, facet_id=facet_id, score=score, rank=rank
                )
                channel_ids.append(candidate.fragment_id)

            sparse, neighbours = self.source_index.search_hybrid(document_id, query)
            sparse_ids: list[str] = []
            for sparse_rank, (candidate, score) in enumerate(sparse, 1):
                # Normalize BM25 into a bounded range while preserving order.
                normalized_score = score / (score + 4.0)
                self._merge_candidate(
                    merged,
                    candidate,
                    facet_id=facet_id,
                    score=round(normalized_score, 8),
                    rank=100 + sparse_rank,
                )
                sparse_ids.append(candidate.fragment_id)
                channel_ids.append(candidate.fragment_id)
            neighbour_ids: list[str] = []
            for neighbour_rank, (candidate, score) in enumerate(neighbours, 1):
                normalized_score = score / (score + 4.0)
                self._merge_candidate(
                    merged,
                    candidate,
                    facet_id=facet_id,
                    score=round(normalized_score, 8),
                    rank=200 + neighbour_rank,
                )
                neighbour_ids.append(candidate.fragment_id)
                channel_ids.append(candidate.fragment_id)
            facet_ranked_ids[facet_id] = list(dict.fromkeys(channel_ids))
            retrieval_trace.append({
                "facet_id": facet_id,
                "label": str(facet.get("label") or ""),
                "priority": str(facet.get("priority") or "core"),
                "query": query,
                "dense_retrieved_chunk_ids": dense_ids,
                "dense_retrieved_chunk_count": len(chunks),
                "sparse_fragment_ids": sparse_ids,
                "neighbour_fragment_ids": neighbour_ids,
                "retrieval_channels": ["bge_m3_dense", "document_bm25", "source_neighbour"],
            })

        # Round-robin first prevents one broad facet from consuming the prompt.
        ordered_ids: list[str] = []
        seen: set[str] = set()
        for position in range(MAX_DENSE_FRAGMENTS_PER_FACET + MAX_SPARSE_FRAGMENTS_PER_FACET):
            for facet in facets:
                ids = facet_ranked_ids.get(str(facet["facet_id"]), [])
                if position < len(ids) and ids[position] in merged and ids[position] not in seen:
                    ordered_ids.append(ids[position])
                    seen.add(ids[position])
                    if len(ordered_ids) >= MAX_CANDIDATES:
                        break
            if len(ordered_ids) >= MAX_CANDIDATES:
                break
        if len(ordered_ids) < MAX_CANDIDATES:
            remainder = sorted(
                (row for key, row in merged.items() if key not in seen),
                key=lambda row: (
                    -sum(row.facet_scores.values()),
                    -len(row.facet_scores),
                    -row.best_score,
                    row.line_range,
                    row.fragment_id,
                ),
            )
            ordered_ids.extend(row.fragment_id for row in remainder[:MAX_CANDIDATES - len(ordered_ids)])
        return [merged[fragment_id] for fragment_id in ordered_ids], retrieval_trace

    def _select_once(
        self,
        *,
        question: str,
        baseline_answer: str,
        facets: Sequence[Mapping[str, str]],
        candidates: Sequence[EvidenceFragment],
        seed: int,
    ) -> list[dict[str, Any]]:
        allowed_ids = {row.fragment_id for row in candidates}
        facet_ids = [str(row["facet_id"]) for row in facets]
        payload = {
            "question": question,
            "baseline_answer": baseline_answer,
            "facets": [dict(row) for row in facets],
            "candidates": [{
                **row.public_prompt_row(),
                "source_kind": row.source_kind,
                "retrieved_for_facets": sorted(row.facet_scores),
            } for row in candidates],
        }
        body = _generation_body(
            model=self.model,
            messages=[
                {"role": "system", "content": COVERAGE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                },
            ],
            schema_name="annual_report_facet_coverage",
            schema=COVERAGE_SCHEMA,
            seed=seed,
        )
        value = _response_content(self.client.complete(body))
        coverage = value.get("facet_coverage")
        if not isinstance(coverage, list) or len(coverage) != len(facet_ids):
            raise FacetedV9Error("coverage row count is invalid")
        by_fragment: dict[str, list[str]] = {}
        seen_facets: set[str] = set()
        for row in coverage:
            if not isinstance(row, Mapping) or set(row) != {
                "facet_id", "status", "fragment_ids"
            }:
                raise FacetedV9Error("coverage row shape is invalid")
            facet_id = str(row["facet_id"])
            status = str(row["status"])
            fragment_ids = row["fragment_ids"]
            if (
                facet_id not in facet_ids
                or facet_id in seen_facets
                or status not in {"evidence", "baseline", "not_found"}
                or not isinstance(fragment_ids, list)
                or len(fragment_ids) > MAX_IDS_PER_FACET
                or (status == "evidence" and not fragment_ids)
                or (status != "evidence" and bool(fragment_ids))
            ):
                raise FacetedV9Error("coverage status/IDs are invalid")
            seen_facets.add(facet_id)
            for fragment_id_value in fragment_ids:
                fragment_id = str(fragment_id_value)
                if fragment_id not in allowed_ids:
                    raise FacetedV9Error("coverage crossed the supplied ID boundary")
                by_fragment.setdefault(fragment_id, []).append(facet_id)
        if seen_facets != set(facet_ids):
            raise FacetedV9Error("coverage omitted a facet")
        return [
            {"fragment_id": fragment_id, "facet_ids": list(dict.fromkeys(ids))}
            for fragment_id, ids in by_fragment.items()
        ]

    @staticmethod
    def _compose(selected: Sequence[EvidenceFragment], baseline_answer: str) -> str:
        baseline = baseline_answer.strip()
        baseline_key = compact_text(baseline)
        parts = [
            row.text for row in selected
            if row.text.strip()
            and (not baseline_key or compact_text(row.text) not in baseline_key)
        ]
        if baseline:
            parts.append(baseline)
        return "\n".join(parts)

    def answer(self, **kwargs: Any) -> dict[str, Any]:
        result = super().answer(**kwargs)
        result["schema_version"] = RESULT_SCHEMA
        result["profile_version"] = PROFILE_VERSION
        core_ids = {
            str(row["facet_id"])
            for row in result["facets"]
            if str(row.get("priority") or "core") == "core"
        }
        selected_ids = set(result["selected_fragment_ids"])
        selected_candidate_facets: set[str] = set()
        for run in result["selector_runs"]:
            if run.get("status") != "ok":
                continue
            for selection in run.get("selections") or []:
                if selection.get("fragment_id") in selected_ids:
                    selected_candidate_facets.update(str(value) for value in selection.get("facet_ids") or [])
        result["coverage_report"] = {
            "core_facet_ids": sorted(core_ids),
            "selected_core_facet_ids": sorted(core_ids.intersection(selected_candidate_facets)),
            "selected_facet_ids": sorted(selected_candidate_facets),
            "core_selection_coverage_ratio": round(
                len(core_ids.intersection(selected_candidate_facets)) / max(1, len(core_ids)), 8
            ),
            "candidate_count": result["candidate_count"],
            "retrieval_channels": ["bge_m3_dense", "document_bm25", "source_neighbour"],
        }
        result["result_fingerprint"] = semantic_sha256({
            "schema_version": result["schema_version"],
            "profile_version": result["profile_version"],
            "case_id": result["case_id"],
            "question": result["question"],
            "document_id": result["document_id"],
            "answer": result["answer"],
            "citations": result["citations"],
            "status": result["status"],
            "facets": result["facets"],
            "selected_fragment_ids": result["selected_fragment_ids"],
            "gate_report": result["gate_report"],
        })
        return result


__all__ = [
    "COVERAGE_SCHEMA",
    "MAX_CANDIDATES",
    "MAX_DENSE_CHUNKS_PER_FACET",
    "MAX_FACETS",
    "PLAN_SCHEMA",
    "PROFILE_VERSION",
    "PROMPT_CONTRACT_HASH",
    "PROMPT_VERSION",
    "RESULT_SCHEMA",
    "SELECTOR_SEEDS",
    "SourceSparseIndex",
    "Type3Qwen36CoverageV10",
]
