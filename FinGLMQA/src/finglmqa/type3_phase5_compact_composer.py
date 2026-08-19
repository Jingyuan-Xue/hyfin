"""Deterministic Phase 5 answer composition over frozen Type 3 evidence.

The composer is annotation-free.  It may use a frozen, already-audited legacy
answer as a safety baseline and append at most one question-relevant sentence
selected from the Phase 4 text route.  Benchmark answers, keywords and scores
are not inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping, Sequence

from finglmqa.type3_evidence_fusion import semantic_sha256
from finglmqa.type3_tabgr_retriever import numeric_fragments


PROFILE_VERSION = "type3-a2rag-tabgr-compact-baseline-v2"
COMPOSER_VERSION = "type3-phase5-compact-composer-v2"
OUTPUT_SCHEMA = "finglmqa.type3.phase5.compact_answer.v2"
TRACE_SCHEMA = "finglmqa.type3.phase5.compact_trace.v2"
MASK = "[未经授权数值]"
MIN_RECALL = 0.08

_STOP_WORDS = (
    "根据",
    "年度报告",
    "年报",
    "公司",
    "该公司",
    "本公司",
    "请",
    "简要",
    "分析",
    "情况",
    "什么",
    "哪些",
    "如何",
    "是否",
    "报告期",
    "相关",
    "主要",
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?；;])|\n+")
_DIGIT_RE = re.compile(r"\d")
_CHECKBOX_RE = re.compile(
    r"^[√□☑■●○\s]*(?:适用|不适用)(?:\s*[√□☑■●○]\s*(?:适用|不适用))*$"
)


class Phase5ComposerError(ValueError):
    """Raised when a frozen evidence packet violates the Phase 5 contract."""


def compact_text(value: str) -> str:
    return re.sub(r"\s+", "", value or "")


def _semantic_text(value: str) -> str:
    text = compact_text(value)
    for stop_word in _STOP_WORDS:
        text = text.replace(stop_word, "")
    return text


def character_bigrams(value: str) -> frozenset[str]:
    text = _semantic_text(value)
    return frozenset(
        text[index : index + 2]
        for index in range(max(0, len(text) - 1))
        if not _DIGIT_RE.search(text[index : index + 2])
    )


def _is_low_information(value: str) -> bool:
    compact = compact_text(value)
    if not 12 <= len(compact) <= 420:
        return True
    if compact in {"无", "其他说明", "其他说明："}:
        return True
    if "常用词语释义" in compact:
        return True
    if _CHECKBOX_RE.fullmatch(value.strip()):
        return True
    if compact.count("适用") >= 2 and len(compact) < 40:
        return True
    return False


def split_full_sentences(value: str) -> tuple[str, ...]:
    """Return bounded, complete clauses without masked table placeholders."""

    projected = value.replace(MASK, "")
    sentences: list[str] = []
    for raw in _SENTENCE_SPLIT_RE.split(projected):
        sentence = " ".join(raw.split()).strip(" －-；;，,：:")
        if not sentence or _is_low_information(sentence):
            continue
        sentences.append(sentence)
    return tuple(sentences)


@dataclass(frozen=True)
class RankedSentence:
    candidate_id: str
    evidence_position: int
    text: str
    score: float
    question_recall: float
    sentence_precision: float
    rank_signal: float
    evidence: Mapping[str, Any]

    def rank_key(self) -> tuple[Any, ...]:
        return (
            -self.score,
            -self.question_recall,
            self.evidence_position,
            self.candidate_id,
            self.text,
        )

    def as_trace(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "evidence_position": self.evidence_position,
            "text": self.text,
            "ranking_value": format(self.score, ".12f"),
            "question_recall": format(self.question_recall, ".12f"),
            "sentence_precision": format(self.sentence_precision, ".12f"),
            "rank_signal": format(self.rank_signal, ".12f"),
            "heading_path": list(self.evidence.get("heading_path") or ()),
        }


def _rank_signal(evidence: Mapping[str, Any]) -> float:
    values: list[float] = []
    for signal in evidence.get("rank_signals") or ():
        try:
            rank = int(signal["rank"])
        except (KeyError, TypeError, ValueError):
            continue
        if rank >= 1:
            values.append(1.0 / (10.0 + rank))
    return max(values, default=0.0)


def rank_text_sentences(
    question: str,
    evidence: Sequence[Mapping[str, Any]],
) -> tuple[RankedSentence, ...]:
    question_grams = character_bigrams(question)
    ranked: list[RankedSentence] = []
    for position, item in enumerate(evidence):
        if item.get("route") != "text":
            continue
        candidate_id = str(item.get("candidate_id") or "")
        if not candidate_id:
            raise Phase5ComposerError("text evidence lacks candidate_id")
        signal = _rank_signal(item)
        for sentence in split_full_sentences(str(item.get("answer_safe_text") or "")):
            sentence_grams = character_bigrams(sentence)
            overlap = len(question_grams.intersection(sentence_grams))
            recall = overlap / max(1, len(question_grams))
            precision = overlap / max(1, len(sentence_grams))
            score = (
                recall
                + 0.30 * precision
                + 0.10 * signal
                - 0.00015 * max(0, len(sentence) - 240)
            )
            ranked.append(
                RankedSentence(
                    candidate_id=candidate_id,
                    evidence_position=position,
                    text=sentence,
                    score=score,
                    question_recall=recall,
                    sentence_precision=precision,
                    rank_signal=signal,
                    evidence=item,
                )
            )
    ranked.sort(key=RankedSentence.rank_key)
    deduplicated: list[RankedSentence] = []
    seen: list[str] = []
    for candidate in ranked:
        key = compact_text(candidate.text)
        if any(key in previous or previous in key for previous in seen):
            continue
        deduplicated.append(candidate)
        seen.append(key)
    return tuple(deduplicated)


def _citation_from_evidence(evidence: Mapping[str, Any]) -> dict[str, Any]:
    corpus_id = str(evidence.get("corpus_id") or "")
    document_id = str(evidence.get("document_id") or "")
    if not corpus_id or not document_id:
        raise Phase5ComposerError(
            "selected text evidence lacks corpus_id or document_id"
        )
    return {
        "citation_kind": "phase5_a2rag_text",
        "corpus_id": corpus_id,
        "document_id": document_id,
        "candidate_id": str(evidence["candidate_id"]),
        "route": "text",
        "source_markdown": str(evidence["source_markdown"]),
        "line_range": list(evidence["line_range"]),
        "char_range": (
            list(evidence["char_range"]) if evidence.get("char_range") is not None else None
        ),
        "byte_range": (
            list(evidence["byte_range"]) if evidence.get("byte_range") is not None else None
        ),
        "source_sha256": str(evidence["source_sha256"]),
        "content_sha256": str(evidence["content_sha256"]),
        "authorization_ids": [
            str(value["authorization_id"])
            for value in evidence.get("numeric_authorizations") or ()
        ],
    }


def _validate_selected_numeric_literals(candidate: RankedSentence) -> None:
    allowed = {
        rendering
        for authorization in candidate.evidence.get("numeric_authorizations") or ()
        for rendering in authorization.get("allowed_renderings") or ()
    }
    unsupported = [
        literal for literal in numeric_fragments(candidate.text) if literal not in allowed
    ]
    if unsupported:
        raise Phase5ComposerError(
            f"selected sentence has unsupported numeric literals: {sorted(set(unsupported))!r}"
        )


def compose_compact_answer(
    *,
    question: str,
    evidence: Sequence[Mapping[str, Any]],
    legacy_answer: str = "",
    legacy_citations: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Compose a deterministic compact answer without benchmark annotations."""

    if not isinstance(question, str) or not question.strip():
        raise Phase5ComposerError("question must be non-empty")
    ranked = rank_text_sentences(question, evidence)
    selected = next(
        (candidate for candidate in ranked if candidate.question_recall >= MIN_RECALL),
        None,
    )
    mode = "legacy_only"
    if selected is None and not legacy_answer.strip() and ranked:
        selected = ranked[0]
        mode = "retrieval_only_fallback"
    elif selected is not None and legacy_answer.strip():
        mode = "legacy_plus_a2rag"
    elif selected is not None:
        mode = "retrieval_only"

    answer_parts = [legacy_answer.strip()] if legacy_answer.strip() else []
    selected_ids: list[str] = []
    citations = [dict(value) for value in legacy_citations]
    selected_authorizations: list[dict[str, Any]] = []
    if selected is not None:
        _validate_selected_numeric_literals(selected)
        answer_parts.append(selected.text)
        citations.append(_citation_from_evidence(selected.evidence))
        selected_ids.append(selected.candidate_id)
        selected_authorizations.extend(
            dict(value) for value in selected.evidence.get("numeric_authorizations") or ()
        )

    answer = "\n".join(value for value in answer_parts if value)
    if MASK in answer:
        raise Phase5ComposerError("masked numeric placeholder reached final answer")
    trace_unsigned = {
        "schema_version": TRACE_SCHEMA,
        "profile_version": PROFILE_VERSION,
        "composer_version": COMPOSER_VERSION,
        "semantic_input": {"question": question},
        "mode": mode,
        "minimum_question_recall": format(MIN_RECALL, ".2f"),
        "ranked_sentence_count": len(ranked),
        "top_ranked_sentences": [value.as_trace() for value in ranked[:5]],
        "selected_candidate_ids": selected_ids,
        "selected_numeric_authorizations": selected_authorizations,
        "legacy_answer_sha256": semantic_sha256(legacy_answer),
    }
    trace = {
        **trace_unsigned,
        "semantic_trace_sha256": semantic_sha256(trace_unsigned),
    }
    return {
        "schema_version": OUTPUT_SCHEMA,
        "profile_version": PROFILE_VERSION,
        "answer_safe_text": answer,
        "citations": citations,
        "selected_candidate_ids": selected_ids,
        "semantic_trace": trace,
    }


__all__ = [
    "COMPOSER_VERSION",
    "MASK",
    "MIN_RECALL",
    "OUTPUT_SCHEMA",
    "PROFILE_VERSION",
    "Phase5ComposerError",
    "character_bigrams",
    "compact_text",
    "compose_compact_answer",
    "rank_text_sentences",
    "split_full_sentences",
]
