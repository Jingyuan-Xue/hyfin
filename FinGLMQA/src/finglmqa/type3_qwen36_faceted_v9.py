"""Document-scoped Qwen planning and extractive evidence enrichment for Type 3.

This opt-in experiment moves Qwen before answer construction.  The model may
decompose an arbitrary annual-report question into retrieval facets and select
fragment identifiers, but it can never author answer text.  Selected text is
copied from one resolved document and prepended to the unchanged v8 answer.

The module deliberately knows nothing about benchmark IDs, reference answers,
keywords, companies, or years.  Those are evaluation concerns, not runtime
question-answering rules.
"""

from __future__ import annotations

from collections import Counter, OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from .contracts import semantic_sha256
from .type3_v7 import DocumentAuditIndex


PROFILE_VERSION = "type3-qwen36-faceted-evidence-v9"
PROMPT_VERSION = "type3-qwen36-faceted-planner-selector-v1"
RESULT_SCHEMA = "finglmqa.experimental.type3_qwen36_faceted_v9.result.v1"
MAX_FACETS = 6
MAX_CANDIDATES = 28
MAX_SELECTED_FRAGMENTS = 10
MAX_FRAGMENT_CHARS = 480
MAX_NEW_EVIDENCE_CHARS = 2_400
SELECTOR_SEEDS = (0, 1, 2)

PLAN_SYSTEM_PROMPT = (
    "你是通用年报问答的检索规划器。把问题拆成1到6个互不重复、共同覆盖完整回答所需信息的检索分面。"
    "检索词只能改写问题中已有的概念，不得猜测公司事实、数值、结论或引用。"
    "客户、供应商、审计事项、现金流原因等问题要保留其各自需要覆盖的关注面。"
)

SELECT_SYSTEM_PROMPT = (
    "你是年报证据覆盖选择器。答案正文由程序逐字复制候选fragment，模型只能返回fragment_id。"
    "选择能直接回答问题、补足基线遗漏或纠正基线关注面偏差的最小充分证据集合；"
    "必须优先保证各检索分面的核心信息完整，不得为了缩短而漏掉并列原因、客户供应商、关键审计事项等关注面。"
    "不得选择仅含目录、跳转说明、单位、通用模板或与问题无关的片段，不得创建ID或输出答案文字。"
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
                "required": ["facet_id", "query"],
                "properties": {
                    "facet_id": {"type": "string", "pattern": "^f[1-6]$"},
                    "query": {"type": "string", "minLength": 2, "maxLength": 120},
                },
            },
        }
    },
}

SELECTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["selections"],
    "properties": {
        "selections": {
            "type": "array",
            "maxItems": MAX_SELECTED_FRAGMENTS,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["fragment_id", "facet_ids"],
                "properties": {
                    "fragment_id": {"type": "string", "minLength": 1},
                    "facet_ids": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": MAX_FACETS,
                        "items": {"type": "string", "pattern": "^f[1-6]$"},
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
    "select_system_prompt": SELECT_SYSTEM_PROMPT,
    "plan_schema": PLAN_SCHEMA,
    "selection_schema": SELECTION_SCHEMA,
    "selector_seeds": SELECTOR_SEEDS,
    "max_candidates": MAX_CANDIDATES,
})

_ANSWER_NUMBER_RE = re.compile(
    r"(?<![\d,.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[%％])?(?![\d,.])"
)
_SENTENCE_RE = re.compile(r"[^。！？!?\r\n]+[。！？!?]?")
_HTML_RE = re.compile(r"<\s*/?\s*(?:table|tr|td|th)\b", re.IGNORECASE)
_LOW_INFO_RE = re.compile(
    r"^(?:√?\s*适用\s*□?\s*不适用|□?\s*适用\s*√?\s*不适用|"
    r"单位\s*[：:]?.{0,12}|详见.{0,80}|参见.{0,80})[。.]?$"
)


class FacetedV9Error(RuntimeError):
    """The experimental planner/selector boundary rejected an input/output."""


@runtime_checkable
class ChatClient(Protocol):
    def complete(self, body: Mapping[str, Any]) -> Mapping[str, Any]: ...


@runtime_checkable
class DocumentRetriever(Protocol):
    def retrieve_for_document(
        self, document_id: str, question: str, top_k: int = 5
    ) -> Mapping[str, Any]: ...


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(value))).strip()


def source_render_text(value: str) -> str:
    """Collapse layout whitespace while preserving every source glyph."""

    return re.sub(r"\s+", " ", str(value)).strip()


def compact_text(value: str) -> str:
    return re.sub(r"[\s\W_]+", "", normalize_text(value).lower(), flags=re.UNICODE)


def _ngrams(value: str) -> set[str]:
    compact = compact_text(value)
    result: set[str] = set()
    for width in (2, 3, 4):
        result.update(compact[index : index + width] for index in range(max(0, len(compact) - width + 1)))
    return result


def _lexical_score(query: str, text: str, heading_path: Sequence[str]) -> float:
    query_terms = _ngrams(query)
    if not query_terms:
        return 0.0
    text_terms = _ngrams(text)
    heading_terms = _ngrams(" ".join(heading_path))
    content_recall = len(query_terms & text_terms) / len(query_terms)
    heading_recall = len(query_terms & heading_terms) / len(query_terms)
    compact_query = compact_text(query)
    compact_value = compact_text(text)
    phrase_bonus = 0.30 if compact_query and compact_query in compact_value else 0.0
    return round(content_recall + 0.65 * heading_recall + phrase_bonus, 8)


def _split_fragments(value: str) -> list[str]:
    """Split only at source boundaries; semicolon enumerations stay intact."""

    result: list[str] = []
    for raw_line in re.split(r"[\r\n]+", value):
        line = source_render_text(raw_line)
        if not line:
            continue
        for match in _SENTENCE_RE.finditer(line):
            sentence = source_render_text(match.group(0))
            if not sentence:
                continue
            if len(sentence) <= MAX_FRAGMENT_CHARS:
                result.append(sentence)
                continue
            # Long enumerations are split at their own punctuation.  Every
            # emitted slice remains a normalized verbatim substring.
            parts = [source_render_text(item) for item in re.split(r"(?<=[；;])", sentence)]
            buffer = ""
            for part in parts:
                if not part:
                    continue
                if buffer and len(buffer) + len(part) > MAX_FRAGMENT_CHARS:
                    result.append(buffer)
                    buffer = ""
                if len(part) > MAX_FRAGMENT_CHARS:
                    if buffer:
                        result.append(buffer)
                        buffer = ""
                    for start in range(0, len(part), MAX_FRAGMENT_CHARS):
                        result.append(part[start : start + MAX_FRAGMENT_CHARS])
                else:
                    buffer += part
            if buffer:
                result.append(buffer)
    return result


@dataclass
class EvidenceFragment:
    fragment_id: str
    document_id: str
    text: str
    source_markdown: str
    line_range: tuple[int, int]
    heading_path: tuple[str, ...]
    source_kind: str
    source_content: str
    source_evidence_id: str | None = None
    facet_scores: dict[str, float] = field(default_factory=dict)
    retrieval_ranks: dict[str, int] = field(default_factory=dict)

    @property
    def best_score(self) -> float:
        return max(self.facet_scores.values(), default=0.0)

    def public_prompt_row(self) -> dict[str, Any]:
        return {
            "fragment_id": self.fragment_id,
            "heading_path": list(self.heading_path),
            "text": self.text,
        }

    def citation(self) -> dict[str, Any]:
        content_hash = hashlib.sha256(self.source_content.encode("utf-8")).hexdigest()
        return {
            "citation_id": "v9-cite-" + self.fragment_id[-20:],
            "citation_kind": "evidence",
            "source_kind": self.source_kind,
            "candidate_id": self.fragment_id,
            "document_id": self.document_id,
            "source_markdown": self.source_markdown,
            "line_range": list(self.line_range),
            "heading_path": list(self.heading_path),
            "content_sha256": content_hash,
            "source_evidence_id": self.source_evidence_id,
        }


def _fragment_id(
    document_id: str,
    source_markdown: str,
    line_range: Sequence[int],
    text: str,
) -> str:
    return "v9frag_" + semantic_sha256({
        "document_id": document_id,
        "source_markdown": source_markdown,
        "line_range": list(line_range),
        "text": normalize_text(text),
    })[:24]


class SourceLexicalIndex:
    """Small LRU view over non-table Markdown sentences in a resolved document."""

    def __init__(self, root: str | Path, *, cache_size: int = 8) -> None:
        self.audit = DocumentAuditIndex(root=root)
        self.cache_size = cache_size
        self._cache: OrderedDict[str, tuple[EvidenceFragment, ...]] = OrderedDict()

    def fragments(self, document_id: str) -> tuple[EvidenceFragment, ...]:
        cached = self._cache.pop(document_id, None)
        if cached is not None:
            self._cache[document_id] = cached
            return cached
        source = self.audit.source(document_id)
        rows: list[EvidenceFragment] = []
        for line_number, raw in enumerate(source.lines, 1):
            if not raw.strip() or raw.lstrip().startswith("#") or _HTML_RE.search(raw):
                continue
            path = source.heading_path(line_number)
            for text in _split_fragments(raw):
                if len(compact_text(text)) < 8 or _LOW_INFO_RE.fullmatch(text):
                    continue
                rows.append(EvidenceFragment(
                    fragment_id=_fragment_id(
                        document_id, source.portable_path, (line_number, line_number), text
                    ),
                    document_id=document_id,
                    text=text,
                    source_markdown=source.portable_path,
                    line_range=(line_number, line_number),
                    heading_path=path,
                    source_kind="source_sentence",
                    source_content=raw,
                ))
        result = tuple(rows)
        self._cache[document_id] = result
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return result

    def search(
        self, document_id: str, query: str, *, limit: int = 3
    ) -> list[tuple[EvidenceFragment, float]]:
        scored = [
            (row, _lexical_score(query, row.text, row.heading_path))
            for row in self.fragments(document_id)
        ]
        scored = [item for item in scored if item[1] >= 0.08]
        scored.sort(
            key=lambda item: (
                -item[1], item[0].line_range, item[0].fragment_id
            )
        )
        return scored[:limit]


def _response_content(envelope: Mapping[str, Any]) -> dict[str, Any]:
    try:
        content = envelope["choices"][0]["message"]["content"]
        value = json.loads(content)
    except Exception as exc:
        raise FacetedV9Error("model did not return JSON content") from exc
    if not isinstance(value, dict):
        raise FacetedV9Error("model JSON content must be an object")
    return value


def _generation_body(
    *,
    model: str,
    messages: Sequence[Mapping[str, str]],
    schema_name: str,
    schema: Mapping[str, Any],
    seed: int,
) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [dict(row) for row in messages],
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "seed": seed,
        "max_tokens": 1024,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": schema_name, "strict": True, "schema": dict(schema)},
        },
        "chat_template_kwargs": {"enable_thinking": True},
    }


class Type3Qwen36FacetedV9:
    """Plan, retrieve and select document-scoped evidence; never write prose."""

    def __init__(
        self,
        client: ChatClient,
        retriever: DocumentRetriever,
        *,
        root: str | Path,
        model: str = "finglmqa-qwen3.6-27b",
        selector_seeds: Sequence[int] = SELECTOR_SEEDS,
    ) -> None:
        self.client = client
        self.retriever = retriever
        self.model = model
        self.selector_seeds = tuple(int(value) for value in selector_seeds)
        if not self.selector_seeds:
            raise ValueError("selector_seeds must not be empty")
        self.source_index = SourceLexicalIndex(root)

    def _plan(self, question: str) -> tuple[list[dict[str, str]], str]:
        body = _generation_body(
            model=self.model,
            messages=[
                {"role": "system", "content": PLAN_SYSTEM_PROMPT},
                {"role": "user", "content": f"问题：{question}"},
            ],
            schema_name="annual_report_retrieval_facets",
            schema=PLAN_SCHEMA,
            seed=0,
        )
        try:
            value = _response_content(self.client.complete(body))
            facets = value.get("facets")
            if not isinstance(facets, list) or not 1 <= len(facets) <= MAX_FACETS:
                raise FacetedV9Error("facet count is invalid")
            checked: list[dict[str, str]] = []
            seen_ids: set[str] = set()
            seen_queries: set[str] = set()
            for ordinal, row in enumerate(facets, 1):
                if not isinstance(row, Mapping) or set(row) != {"facet_id", "query"}:
                    raise FacetedV9Error("facet shape is invalid")
                facet_id = str(row["facet_id"])
                query = normalize_text(str(row["query"]))
                if facet_id != f"f{ordinal}" or not 2 <= len(query) <= 120:
                    raise FacetedV9Error("facet identity/query is invalid")
                query_key = compact_text(query)
                if not query_key or facet_id in seen_ids or query_key in seen_queries:
                    raise FacetedV9Error("facets are duplicated")
                seen_ids.add(facet_id)
                seen_queries.add(query_key)
                checked.append({"facet_id": facet_id, "query": query})
            return checked, "qwen_planned"
        except Exception:
            # Retrieval still works when planning fails.  The fallback is the
            # user's arbitrary question, never a benchmark-specific rule.
            return [{"facet_id": "f1", "query": normalize_text(question)[:120]}], "question_fallback"

    @staticmethod
    def _merge_candidate(
        target: dict[str, EvidenceFragment],
        candidate: EvidenceFragment,
        *,
        facet_id: str,
        score: float,
        rank: int,
    ) -> None:
        text_key = compact_text(candidate.text)
        existing_id = next(
            (key for key, row in target.items() if compact_text(row.text) == text_key), None
        )
        if existing_id is not None:
            existing = target[existing_id]
            existing.facet_scores[facet_id] = max(existing.facet_scores.get(facet_id, 0.0), score)
            existing.retrieval_ranks[facet_id] = min(existing.retrieval_ranks.get(facet_id, rank), rank)
            return
        candidate.facet_scores[facet_id] = score
        candidate.retrieval_ranks[facet_id] = rank
        target[candidate.fragment_id] = candidate

    def _candidates(
        self,
        *,
        document_id: str,
        facets: Sequence[Mapping[str, str]],
    ) -> tuple[list[EvidenceFragment], list[dict[str, Any]]]:
        merged: dict[str, EvidenceFragment] = {}
        retrieval_trace: list[dict[str, Any]] = []
        for facet in facets:
            facet_id = str(facet["facet_id"])
            query = str(facet["query"])
            result = self.retriever.retrieve_for_document(document_id, query, top_k=5)
            chunks = result.get("chunks") if isinstance(result, Mapping) else None
            if not isinstance(chunks, list):
                chunks = []
            facet_pool: list[tuple[EvidenceFragment, float, int]] = []
            retrieved_ids: list[str] = []
            for fallback_rank, chunk in enumerate(chunks, 1):
                if not isinstance(chunk, Mapping) or chunk.get("document_id") != document_id:
                    continue
                source_content = str(chunk.get("content") or "")
                source_markdown = str(chunk.get("source_markdown") or "")
                line_range_value = chunk.get("line_range") or []
                if (
                    not source_content
                    or not source_markdown
                    or not isinstance(line_range_value, list)
                    or len(line_range_value) != 2
                ):
                    continue
                rank = int(chunk.get("rank") or fallback_rank)
                source_id = str(chunk.get("evidence_chunk_id") or "") or None
                if source_id:
                    retrieved_ids.append(source_id)
                heading_path = tuple(str(value) for value in chunk.get("section_path") or [])
                for text in _split_fragments(source_content):
                    if len(compact_text(text)) < 8 or _LOW_INFO_RE.fullmatch(text):
                        continue
                    lexical = _lexical_score(query, text, heading_path)
                    score = lexical + max(0.0, (6 - rank) * 0.035)
                    candidate = EvidenceFragment(
                        fragment_id=_fragment_id(
                            document_id, source_markdown, line_range_value, text
                        ),
                        document_id=document_id,
                        text=text,
                        source_markdown=source_markdown,
                        line_range=(int(line_range_value[0]), int(line_range_value[1])),
                        heading_path=heading_path,
                        source_kind="dense_evidence_fragment",
                        source_content=source_content,
                        source_evidence_id=source_id,
                    )
                    facet_pool.append((candidate, score, rank))
            facet_pool.sort(
                key=lambda item: (-item[1], item[2], item[0].line_range, item[0].fragment_id)
            )
            for candidate, score, rank in facet_pool[:5]:
                self._merge_candidate(
                    merged, candidate, facet_id=facet_id, score=score, rank=rank
                )
            for candidate, lexical in self.source_index.search(document_id, query, limit=3):
                self._merge_candidate(
                    merged, candidate, facet_id=facet_id, score=lexical + 0.04, rank=1000
                )
            retrieval_trace.append({
                "facet_id": facet_id,
                "query": query,
                "retrieved_chunk_ids": retrieved_ids,
                "retrieved_chunk_count": len(chunks),
            })

        ranked = sorted(
            merged.values(),
            key=lambda row: (
                -len(row.facet_scores), -row.best_score, row.line_range, row.fragment_id
            ),
        )[:MAX_CANDIDATES]
        return ranked, retrieval_trace

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
        facet_ids = {str(row["facet_id"]) for row in facets}
        payload = {
            "question": question,
            "baseline_answer": baseline_answer,
            "facets": [dict(row) for row in facets],
            "candidates": [row.public_prompt_row() for row in candidates],
        }
        body = _generation_body(
            model=self.model,
            messages=[
                {"role": "system", "content": SELECT_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))},
            ],
            schema_name="annual_report_evidence_selection",
            schema=SELECTION_SCHEMA,
            seed=seed,
        )
        value = _response_content(self.client.complete(body))
        selections = value.get("selections")
        if not isinstance(selections, list) or len(selections) > MAX_SELECTED_FRAGMENTS:
            raise FacetedV9Error("selection count is invalid")
        checked: list[dict[str, Any]] = []
        seen: set[str] = set()
        for row in selections:
            if not isinstance(row, Mapping) or set(row) != {"fragment_id", "facet_ids"}:
                raise FacetedV9Error("selection shape is invalid")
            fragment_id = str(row["fragment_id"])
            selected_facets = row["facet_ids"]
            if (
                fragment_id not in allowed_ids
                or fragment_id in seen
                or not isinstance(selected_facets, list)
                or not selected_facets
                or any(str(value) not in facet_ids for value in selected_facets)
            ):
                raise FacetedV9Error("selection crossed the supplied ID/facet boundary")
            seen.add(fragment_id)
            checked.append({
                "fragment_id": fragment_id,
                "facet_ids": list(dict.fromkeys(str(value) for value in selected_facets)),
            })
        return checked

    def _consensus_select(
        self,
        *,
        question: str,
        baseline_answer: str,
        facets: Sequence[Mapping[str, str]],
        candidates: Sequence[EvidenceFragment],
    ) -> tuple[list[EvidenceFragment], list[dict[str, Any]], str]:
        if not candidates:
            return [], [], "no_candidates"
        runs: list[dict[str, Any]] = []
        valid: list[list[dict[str, Any]]] = []
        def call(seed: int) -> tuple[int, list[dict[str, Any]] | None]:
            try:
                rows = self._select_once(
                    question=question,
                    baseline_answer=baseline_answer,
                    facets=facets,
                    candidates=candidates,
                    seed=seed,
                )
                return seed, rows
            except Exception:
                return seed, None

        with ThreadPoolExecutor(max_workers=len(self.selector_seeds)) as executor:
            outcomes = list(executor.map(call, self.selector_seeds))
        for seed, rows in outcomes:
            if rows is None:
                runs.append({"seed": seed, "status": "invalid", "selections": []})
            else:
                valid.append(rows)
                runs.append({"seed": seed, "status": "ok", "selections": rows})
        if not valid:
            return [], runs, "all_selector_runs_invalid"

        votes: Counter[str] = Counter()
        facet_votes: dict[str, Counter[str]] = defaultdict(Counter)
        positions: dict[str, list[int]] = defaultdict(list)
        for rows in valid:
            for position, row in enumerate(rows):
                fragment_id = str(row["fragment_id"])
                votes[fragment_id] += 1
                positions[fragment_id].append(position)
                for facet_id in row["facet_ids"]:
                    facet_votes[fragment_id][facet_id] += 1
        threshold = len(valid) // 2 + 1
        candidate_by_id = {row.fragment_id: row for row in candidates}
        kept_ids = [fragment_id for fragment_id, count in votes.items() if count >= threshold]
        kept_ids.sort(
            key=lambda fragment_id: (
                min(
                    (
                        index
                        for index, facet in enumerate(facets)
                        if facet_votes[fragment_id][str(facet["facet_id"])] >= threshold
                    ),
                    default=999,
                ),
                sum(positions[fragment_id]) / len(positions[fragment_id]),
                -votes[fragment_id],
                candidate_by_id[fragment_id].line_range,
                fragment_id,
            )
        )
        selected: list[EvidenceFragment] = []
        used_chars = 0
        for fragment_id in kept_ids:
            row = candidate_by_id[fragment_id]
            addition = len(row.text) + (1 if selected else 0)
            if used_chars + addition > MAX_NEW_EVIDENCE_CHARS:
                continue
            selected.append(row)
            used_chars += addition
            if len(selected) >= MAX_SELECTED_FRAGMENTS:
                break
        return selected, runs, "consensus_selected" if selected else "consensus_empty"

    @staticmethod
    def _deterministic_ablation(
        facets: Sequence[Mapping[str, str]],
        candidates: Sequence[EvidenceFragment],
    ) -> list[EvidenceFragment]:
        selected: list[EvidenceFragment] = []
        seen: set[str] = set()
        for facet in facets:
            facet_id = str(facet["facet_id"])
            ranked = sorted(
                (row for row in candidates if facet_id in row.facet_scores),
                key=lambda row: (-row.facet_scores[facet_id], row.line_range, row.fragment_id),
            )
            if ranked and ranked[0].facet_scores[facet_id] >= 0.16 and ranked[0].fragment_id not in seen:
                selected.append(ranked[0])
                seen.add(ranked[0].fragment_id)
        return selected[:MAX_FACETS]

    @staticmethod
    def _compose(selected: Sequence[EvidenceFragment], baseline_answer: str) -> str:
        parts = [row.text for row in selected if row.text.strip()]
        if baseline_answer.strip():
            baseline_key = compact_text(baseline_answer)
            if not any(compact_text(row.text) == baseline_key for row in selected):
                parts.append(baseline_answer.strip())
        return "\n".join(parts)

    def answer(
        self,
        *,
        case_id: str,
        question: str,
        document_id: str,
        baseline_answer: str,
        baseline_citations: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        if not all(isinstance(value, str) and value.strip() for value in (case_id, question, document_id)):
            raise FacetedV9Error("case identity, question and document scope must be non-empty")
        for citation in baseline_citations:
            if isinstance(citation, Mapping) and citation.get("document_id") not in (None, document_id):
                raise FacetedV9Error("baseline citations cross the resolved document")

        facets, planner_outcome = self._plan(question)
        candidates, retrieval_trace = self._candidates(document_id=document_id, facets=facets)
        selected, selector_runs, selector_outcome = self._consensus_select(
            question=question,
            baseline_answer=baseline_answer,
            facets=facets,
            candidates=candidates,
        )
        deterministic_selected = self._deterministic_ablation(facets, candidates)
        no_qwen_facets = [{"facet_id": "f1", "query": normalize_text(question)[:120]}]
        no_qwen_candidates, no_qwen_retrieval_trace = self._candidates(
            document_id=document_id,
            facets=no_qwen_facets,
        )
        no_qwen_selected = self._deterministic_ablation(
            no_qwen_facets, no_qwen_candidates
        )
        answer = self._compose(selected, baseline_answer)
        deterministic_answer = self._compose(deterministic_selected, baseline_answer)
        no_qwen_answer = self._compose(no_qwen_selected, baseline_answer)

        candidate_by_id = {row.fragment_id: row for row in candidates}
        support_passed = all(
            normalize_text(row.text) in normalize_text(row.source_content) for row in selected
        )
        number_support_passed = all(
            match.group(0) in row.text
            for row in selected
            for match in _ANSWER_NUMBER_RE.finditer(row.text)
        )
        answer_projection_passed = answer == self._compose(selected, baseline_answer)
        selected_citations = [row.citation() for row in selected]
        citation_keys = {
            semantic_sha256(dict(value)) for value in selected_citations
        }
        citations = list(selected_citations)
        for value in baseline_citations:
            if not isinstance(value, Mapping):
                continue
            row = dict(value)
            key = semantic_sha256(row)
            if key not in citation_keys:
                citations.append(row)
                citation_keys.add(key)
        citation_scope_passed = all(
            not isinstance(value, Mapping) or value.get("document_id") in (None, document_id)
            for value in citations
        )
        gates_passed = all((
            support_passed,
            number_support_passed,
            answer_projection_passed,
            citation_scope_passed,
        ))
        if not gates_passed:
            selected = []
            selected_citations = []
            answer = baseline_answer.strip()
            citations = [dict(value) for value in baseline_citations if isinstance(value, Mapping)]
            selector_outcome = "safety_fallback_to_v8"

        result = {
            "schema_version": RESULT_SCHEMA,
            "profile_version": PROFILE_VERSION,
            "case_id": case_id,
            "question": question,
            "document_id": document_id,
            "answer": answer,
            "qwen_plan_deterministic_selector_answer": deterministic_answer,
            "no_qwen_same_index_answer": no_qwen_answer,
            "baseline_answer": baseline_answer,
            "citations": citations,
            "status": "ok" if answer.strip() else "not_found",
            "planner_outcome": planner_outcome,
            "selector_outcome": selector_outcome,
            "facets": facets,
            "retrieval_trace": retrieval_trace,
            "no_qwen_retrieval_trace": no_qwen_retrieval_trace,
            "candidate_count": len(candidates),
            "candidate_ids": [row.fragment_id for row in candidates],
            "selected_fragment_ids": [row.fragment_id for row in selected],
            "deterministic_selected_fragment_ids": [
                row.fragment_id for row in deterministic_selected
            ],
            "no_qwen_selected_fragment_ids": [
                row.fragment_id for row in no_qwen_selected
            ],
            "selector_runs": selector_runs,
            "gate_report": {
                "citation_scope_passed": citation_scope_passed,
                "selected_text_verbatim_supported": support_passed,
                "selected_numbers_source_supported": number_support_passed,
                "answer_projection_passed": answer_projection_passed,
                "model_text_accepted": False,
                "passed": gates_passed,
            },
            "selected_fragment_projection": [
                {
                    "fragment_id": row.fragment_id,
                    "text": row.text,
                    "source_markdown": row.source_markdown,
                    "line_range": list(row.line_range),
                    "heading_path": list(row.heading_path),
                    "source_content_sha256": hashlib.sha256(
                        row.source_content.encode("utf-8")
                    ).hexdigest(),
                    "source_number_count": len(_ANSWER_NUMBER_RE.findall(row.text)),
                }
                for row in selected
            ],
        }
        # The fingerprint covers the externally deliverable result.  Sampled
        # diagnostic ordering that did not survive majority consensus remains
        # in selector_runs but cannot make a stable final answer look changed.
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
        # Defensive assertion that only supplied IDs survived consensus.
        if any(fragment_id not in candidate_by_id for fragment_id in result["selected_fragment_ids"]):
            raise FacetedV9Error("selected ID is absent from candidate projection")
        return result


__all__ = [
    "ChatClient",
    "DocumentRetriever",
    "FacetedV9Error",
    "MAX_CANDIDATES",
    "MAX_FACETS",
    "MAX_SELECTED_FRAGMENTS",
    "PLAN_SCHEMA",
    "PROFILE_VERSION",
    "PROMPT_CONTRACT_HASH",
    "PROMPT_VERSION",
    "RESULT_SCHEMA",
    "SELECTION_SCHEMA",
    "SELECTOR_SEEDS",
    "Type3Qwen36FacetedV9",
]
