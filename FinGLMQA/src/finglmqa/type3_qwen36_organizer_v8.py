"""Evidence-bounded Qwen organization of frozen Type 3 v8 answers.

This experimental module is intentionally not imported by the Phase 8/10
service.  Qwen is given only a question and deterministic segments cut from
the already-authorized v8 answer.  It may return segment identifiers in a new
order; model-authored prose, citations, and numbers are never accepted.

The post-generation boundary independently checks citation scope, normalized
verbatim support in the authorized v8 answer, and numeric authorization.  A
failure produces an empty answer (fail closed), never the unchecked baseline.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import PurePosixPath
import re
import unicodedata
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from .contracts import semantic_sha256
from .type3_v7_table_upgrade import TableNumericAuthorization


RESULT_SCHEMA = "finglmqa.experimental.type3_qwen36_organization_v8.result.v1"
PROMPT_VERSION = "type3-v8-authorized-segment-organizer-v2"
STRUCTURED_OUTPUT_VERSION = "type3-v8-segment-selection-v2"
MAX_SELECTED_SEGMENTS = 8
MAX_SOURCE_SEGMENTS = 64
MAX_ANSWER_CHARS = 5_000
MAX_QUESTION_CHARS = 2_000

SYSTEM_PROMPT = (
    "你是答案证据片段的组织器。只按相关性选择和排序给定segment_id，最多8条。"
    "不得输出或改写任何答案文字，不得创建segment_id，不得加入数字、事实、解释或引用。"
    "证据不足时返回空selected_segment_ids。"
)

SELECTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["selected_segment_ids"],
    "properties": {
        "selected_segment_ids": {
            "type": "array",
            "maxItems": MAX_SELECTED_SEGMENTS,
            "items": {"type": "string", "minLength": 1},
        },
    },
}

PROMPT_CONTRACT_HASH = semantic_sha256({
    "prompt_version": PROMPT_VERSION,
    "system_prompt": SYSTEM_PROMPT,
    "selection_schema": SELECTION_SCHEMA,
    "max_source_segments": MAX_SOURCE_SEGMENTS,
})

_NUMBER_RE = re.compile(
    r"(?<![\d,.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[%％])?(?![\d,.])"
)
_YEAR_WITH_SUFFIX_RE = re.compile(r"(?<!\d)((?:19|20|21)\d{2})\s*年")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_AUTH_FIELDS = frozenset({
    "schema_version", "authorization_id", "document_id", "company",
    "report_year", "source_markdown", "source_line_range",
    "source_line_sha256", "heading_path", "concern_group", "value_role",
    "raw_value", "normalized_unit", "allowed_renderings",
})


class OrganizerError(RuntimeError):
    """Base error for the isolated organizer boundary."""


class OrganizerInputError(OrganizerError):
    """The supplied v8 projection is malformed or crosses evidence scope."""


class OrganizerOutputError(OrganizerError):
    """The local model did not return the frozen selection shape."""


@runtime_checkable
class ChatClient(Protocol):
    def complete(self, body: Mapping[str, Any]) -> Mapping[str, Any]: ...


class OpenAICompatibleClient:
    """Minimal local HTTP client; it never persists prompts or responses."""

    def __init__(self, base_url: str, *, timeout_seconds: float = 130.0) -> None:
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("base_url must be non-empty")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = float(timeout_seconds)

    def complete(self, body: Mapping[str, Any]) -> Mapping[str, Any]:
        import httpx

        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(f"{self.base_url}/chat/completions", json=dict(body))
            response.raise_for_status()
            value = response.json()
        if not isinstance(value, Mapping):
            raise OrganizerOutputError("chat completion envelope must be an object")
        return value


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value)).strip()


@dataclass(frozen=True)
class ValidatedNumericGrant:
    authorization_id: str
    allowed_renderings: tuple[str, ...]


def split_authorized_answer(answer: str) -> list[dict[str, str]]:
    """Cut one authorized answer with a generic, case-independent rule."""

    if not isinstance(answer, str):
        raise OrganizerInputError("authorized answer must be a string")
    if len(answer) > MAX_ANSWER_CHARS:
        raise OrganizerInputError("authorized answer exceeds the frozen character limit")
    parts: list[str] = []
    for line in re.split(r"[\r\n]+", answer):
        normalized_line = normalize_text(line)
        if not normalized_line:
            continue
        # Keep terminators on their source span.  Semicolons are useful for
        # long audited table renderings while commas are never split because
        # they may be thousands separators.
        matches = re.findall(r".*?(?:[。！？!?；;]|$)", normalized_line)
        parts.extend(part for part in (normalize_text(item) for item in matches) if part)
    if len(parts) > MAX_SOURCE_SEGMENTS:
        raise OrganizerInputError("authorized answer yields too many source segments")
    return [
        {
            "segment_id": "v8seg_" + semantic_sha256({"ordinal": ordinal, "text": text})[:20],
            "text": text,
        }
        for ordinal, text in enumerate(parts)
    ]


def _citation_projection(value: Any) -> tuple[list[dict[str, Any]], str | None]:
    if not isinstance(value, list):
        raise OrganizerInputError("citations must be an array")
    checked: list[dict[str, Any]] = []
    document_ids: set[str] = set()
    citation_ids: set[str] = set()
    for citation in value:
        if not isinstance(citation, Mapping):
            raise OrganizerInputError("citation must be an object")
        row = copy.deepcopy(dict(citation))
        citation_id = row.get("citation_id")
        document_id = row.get("document_id")
        if not isinstance(citation_id, str) or not citation_id.strip():
            raise OrganizerInputError("citation_id must be non-empty")
        if citation_id in citation_ids:
            raise OrganizerInputError("citation IDs must be unique")
        if not isinstance(document_id, str) or not document_id.strip():
            raise OrganizerInputError("citation document_id must be non-empty")
        citation_ids.add(citation_id)
        document_ids.add(document_id)

        provenance = row.get("provenance")
        source = provenance if isinstance(provenance, Mapping) else row
        source_markdown = source.get("source_markdown")
        line_range = source.get("line_range")
        if not isinstance(source_markdown, str) or not source_markdown.strip():
            raise OrganizerInputError("citation source_markdown must be non-empty")
        path = PurePosixPath(source_markdown)
        if path.is_absolute() or ".." in path.parts or "\\" in source_markdown:
            raise OrganizerInputError("citation source_markdown is not portable")
        if (
            not isinstance(line_range, list)
            or len(line_range) != 2
            or any(isinstance(item, bool) or not isinstance(item, int) or item < 1 for item in line_range)
            or line_range[1] < line_range[0]
        ):
            raise OrganizerInputError("citation line_range is invalid")
        content_hash = source.get("content_sha256")
        if not isinstance(content_hash, str) or not _SHA256_RE.fullmatch(content_hash):
            raise OrganizerInputError("citation content hash is invalid")
        checked.append(row)
    if len(document_ids) > 1:
        raise OrganizerInputError("citations cross document scope")
    return checked, next(iter(document_ids), None)


def _authorization_projection(
    values: Any,
    *,
    document_id: str | None,
    citations: Sequence[Mapping[str, Any]],
) -> list[ValidatedNumericGrant]:
    if not isinstance(values, list):
        raise OrganizerInputError("numeric_authorizations must be an array")
    if values and document_id is None:
        raise OrganizerInputError("numeric authorizations require citation scope")
    result: list[ValidatedNumericGrant] = []
    seen: set[str] = set()
    for value in values:
        schema = value.get("schema_version") if isinstance(value, Mapping) else None
        if schema == "finglmqa.experimental.table_numeric_authorization.v1":
            try:
                parsed = TableNumericAuthorization.from_mapping(
                    value,
                    expected_document_id=document_id,
                )
            except (TypeError, ValueError) as exc:
                raise OrganizerInputError("table numeric authorization failed frozen validation") from exc
            candidate_ids = {
                row.get("candidate_id") for row in citations
                if isinstance(row, Mapping) and row.get("document_id") == document_id
            }
            if parsed.fragment_id not in candidate_ids:
                raise OrganizerInputError("table numeric authorization lacks an in-scope citation")
            grant = ValidatedNumericGrant(
                parsed.authorization_id,
                tuple(parsed.allowed_renderings),
            )
        elif schema == "finglmqa.experimental.source_numeric_authorization.v1":
            grant = _validate_source_numeric_authorization(
                value,
                document_id=document_id,
                citations=citations,
            )
        else:
            raise OrganizerInputError("numeric authorization schema is unsupported")
        if grant.authorization_id in seen:
            raise OrganizerInputError("numeric authorization IDs must be unique")
        seen.add(grant.authorization_id)
        result.append(grant)
    return result


def _validate_source_numeric_authorization(
    value: Mapping[str, Any],
    *,
    document_id: str | None,
    citations: Sequence[Mapping[str, Any]],
) -> ValidatedNumericGrant:
    fields = set(value)
    if fields not in (_SOURCE_AUTH_FIELDS, _SOURCE_AUTH_FIELDS | {"source_character_range"}):
        raise OrganizerInputError("source numeric authorization fields differ from v1")
    if value.get("document_id") != document_id:
        raise OrganizerInputError("source numeric authorization document scope mismatch")
    for field in (
        "authorization_id", "company", "source_markdown", "source_line_sha256",
        "concern_group", "value_role", "raw_value", "normalized_unit",
    ):
        if not isinstance(value.get(field), str) or not str(value[field]).strip():
            raise OrganizerInputError(f"source numeric authorization {field} is invalid")
    report_year = value.get("report_year")
    if isinstance(report_year, bool) or not isinstance(report_year, int) or not 1900 <= report_year <= 2199:
        raise OrganizerInputError("source numeric authorization report_year is invalid")
    path = PurePosixPath(str(value["source_markdown"]))
    if path.is_absolute() or ".." in path.parts or "\\" in str(value["source_markdown"]):
        raise OrganizerInputError("source numeric authorization path is not portable")
    line_range = value.get("source_line_range")
    if (
        not isinstance(line_range, list)
        or len(line_range) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) or item < 1 for item in line_range)
        or line_range[1] < line_range[0]
    ):
        raise OrganizerInputError("source numeric authorization line range is invalid")
    if not _SHA256_RE.fullmatch(str(value["source_line_sha256"])):
        raise OrganizerInputError("source numeric authorization source hash is invalid")
    heading = value.get("heading_path")
    if not isinstance(heading, list) or any(not isinstance(item, str) for item in heading):
        raise OrganizerInputError("source numeric authorization heading_path is invalid")
    allowed = value.get("allowed_renderings")
    if (
        not isinstance(allowed, list)
        or not allowed
        or any(not isinstance(item, str) or not item for item in allowed)
        or allowed != sorted(set(allowed))
        or not any(normalize_text(str(value["raw_value"])) in normalize_text(item) for item in allowed)
    ):
        raise OrganizerInputError("source numeric authorization renderings are invalid")
    character_range = value.get("source_character_range")
    if character_range is not None and (
        not isinstance(character_range, list)
        or len(character_range) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in character_range)
        or character_range[1] <= character_range[0]
    ):
        raise OrganizerInputError("source numeric authorization character range is invalid")
    linked = any(
        citation.get("document_id") == document_id
        and citation.get("source_markdown") == value["source_markdown"]
        and citation.get("line_range") == line_range
        and citation.get("content_sha256") == value["source_line_sha256"]
        for citation in citations
    )
    if not linked:
        raise OrganizerInputError("source numeric authorization lacks an exact source citation")
    unsigned = dict(value)
    authorization_id = str(unsigned.pop("authorization_id"))
    if authorization_id != "v8-source-auth-" + semantic_sha256(unsigned)[:24]:
        raise OrganizerInputError("source numeric authorization ID mismatch")
    return ValidatedNumericGrant(authorization_id, tuple(allowed))


def _number_is_authorized(
    text: str,
    match: re.Match[str],
    *,
    question: str,
    authorizations: Sequence[ValidatedNumericGrant],
) -> bool:
    token = normalize_text(match.group(0)).replace("％", "%")
    contextual_years = {item.group(1) for item in _YEAR_WITH_SUFFIX_RE.finditer(question)}
    suffix = text[match.end() : match.end() + 2]
    if token in contextual_years and suffix.lstrip().startswith("年"):
        return True
    for authorization in authorizations:
        for rendering in authorization.allowed_renderings:
            candidate = normalize_text(rendering).replace("％", "%")
            start = 0
            while True:
                position = text.find(candidate, start)
                if position < 0:
                    break
                end = position + len(candidate)
                if position <= match.start() and match.end() <= end:
                    candidate_numbers = {
                        normalize_text(item.group(0)).replace("％", "%")
                        for item in _NUMBER_RE.finditer(candidate)
                    }
                    if token in candidate_numbers:
                        return True
                start = position + 1
    return False


def numeric_text_is_authorized(
    text: str,
    *,
    question: str,
    authorizations: Sequence[ValidatedNumericGrant],
) -> bool:
    return all(
        _number_is_authorized(
            text,
            match,
            question=question,
            authorizations=authorizations,
        )
        for match in _NUMBER_RE.finditer(text)
    )


class Type3Qwen36OrganizerV8:
    """Select authorized v8 segments and enforce every gate after Qwen."""

    def __init__(
        self,
        client: ChatClient,
        *,
        model: str = "finglmqa-qwen3.6-27b",
    ) -> None:
        if not isinstance(client, ChatClient):
            raise TypeError("client must implement complete")
        if not isinstance(model, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]+", model.strip()):
            raise ValueError("model must be a portable served name")
        self.client = client
        self.model = model.strip()

    @property
    def model_config(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "temperature": 0,
            "top_p": 1,
            "seed": 0,
            "max_tokens": 256,
            "enable_thinking": False,
            "max_selected_segments": MAX_SELECTED_SEGMENTS,
            "prompt_version": PROMPT_VERSION,
            "prompt_contract_hash": PROMPT_CONTRACT_HASH,
            "structured_output_version": STRUCTURED_OUTPUT_VERSION,
        }

    @staticmethod
    def _messages(question: str, segments: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
        evidence = "\n".join(
            json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            for row in segments
        )
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"问题：{question}\n已授权证据片段（JSONL）：\n{evidence}",
            },
        ]

    def request_body(
        self,
        *,
        question: str,
        segments: Sequence[Mapping[str, str]],
    ) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": self._messages(question, segments),
            "temperature": 0,
            "top_p": 1,
            "seed": 0,
            "max_tokens": 256,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "type3_v8_segment_selection",
                    "strict": True,
                    "schema": SELECTION_SCHEMA,
                },
            },
            "chat_template_kwargs": {"enable_thinking": False},
        }

    @staticmethod
    def _parse_selection(value: Any, allowed_ids: set[str]) -> list[str]:
        try:
            if not isinstance(value, Mapping):
                raise TypeError
            choices = value["choices"]
            if not isinstance(choices, list) or len(choices) != 1:
                raise TypeError
            content = choices[0]["message"]["content"]
            if not isinstance(content, str):
                raise TypeError
            parsed = json.loads(content)
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise OrganizerOutputError("model response envelope is invalid") from exc
        if not isinstance(parsed, dict) or set(parsed) != {"selected_segment_ids"}:
            raise OrganizerOutputError("model selection fields are invalid")
        selected = parsed["selected_segment_ids"]
        if (
            not isinstance(selected, list)
            or len(selected) > MAX_SELECTED_SEGMENTS
            or any(not isinstance(item, str) for item in selected)
            or len(selected) != len(set(selected))
            or any(item not in allowed_ids for item in selected)
        ):
            raise OrganizerOutputError("model selected outside the authorized segment set")
        return selected

    def organize(
        self,
        *,
        case_id: str,
        question: str,
        authorized_answer: str,
        citations: list[dict[str, Any]],
        numeric_authorizations: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not isinstance(case_id, str) or not case_id.strip():
            raise OrganizerInputError("case_id must be non-empty")
        if not isinstance(question, str) or not question.strip() or len(question) > MAX_QUESTION_CHARS:
            raise OrganizerInputError("question is invalid")
        checked_citations, document_id = _citation_projection(citations)
        checked_authorizations = _authorization_projection(
            numeric_authorizations,
            document_id=document_id,
            citations=checked_citations,
        )
        segments = split_authorized_answer(authorized_answer)
        if segments and not checked_citations:
            raise OrganizerInputError("non-empty authorized answer requires citations")

        selected_ids: list[str] = []
        generator_error: str | None = None
        if segments:
            try:
                raw = self.client.complete(self.request_body(question=question, segments=segments))
                selected_ids = self._parse_selection(
                    raw,
                    {row["segment_id"] for row in segments},
                )
            except Exception:
                generator_error = "GENERATOR_INVALID_OUTPUT"

        by_id = {row["segment_id"]: row for row in segments}
        accepted: list[dict[str, str]] = []
        rejected: list[dict[str, str]] = []
        normalized_source = normalize_text(authorized_answer)
        if generator_error is None:
            for segment_id in selected_ids:
                text = by_id[segment_id]["text"]
                if normalize_text(text) not in normalized_source:
                    rejected.append({"segment_id": segment_id, "reason": "NOT_IN_AUTHORIZED_ANSWER"})
                    continue
                if not numeric_text_is_authorized(
                    text,
                    question=question,
                    authorizations=checked_authorizations,
                ):
                    rejected.append({"segment_id": segment_id, "reason": "UNAUTHORIZED_NUMBER"})
                    continue
                accepted.append({"segment_id": segment_id, "text": text})

        answer = "\n".join(row["text"] for row in accepted)
        final_numeric_pass = numeric_text_is_authorized(
            answer,
            question=question,
            authorizations=checked_authorizations,
        )
        final_support_pass = all(normalize_text(row["text"]) in normalized_source for row in accepted)
        final_citations = copy.deepcopy(checked_citations) if answer else []

        if generator_error is not None:
            status = "error"
            outcome = "generator_invalid_output"
        elif answer:
            status = "ok"
            outcome = "organized"
        elif not segments:
            status = "not_found"
            outcome = "v8_not_found"
        elif selected_ids and rejected:
            status = "not_found"
            outcome = "selection_rejected_by_gate"
        else:
            status = "not_found"
            outcome = "generator_refused"

        source_snapshot = {
            "authorized_answer_sha256": hashlib.sha256(authorized_answer.encode("utf-8")).hexdigest(),
            "citation_projection_sha256": semantic_sha256(checked_citations),
            "numeric_authorization_projection_sha256": semantic_sha256(numeric_authorizations),
            "segment_count": len(segments),
            "document_id": document_id,
        }
        result: dict[str, Any] = {
            "schema_version": RESULT_SCHEMA,
            "case_id": case_id,
            "question": question,
            "status": status,
            "generator_outcome": outcome,
            "answer": answer,
            "citations": final_citations,
            "selected_segment_ids": [row["segment_id"] for row in accepted],
            "rejected_selections": rejected,
            "gate_report": {
                "citation_scope_passed": True,
                "authorized_answer_support_passed": final_support_pass,
                "numeric_authorization_passed": final_numeric_pass,
                "model_text_accepted": False,
            },
            "source_snapshot": source_snapshot,
            "model_config": self.model_config,
            "result_fingerprint": "",
        }
        result["result_fingerprint"] = semantic_sha256({
            key: value for key, value in result.items() if key != "result_fingerprint"
        })
        return result


__all__ = [
    "ChatClient",
    "MAX_SELECTED_SEGMENTS",
    "OpenAICompatibleClient",
    "OrganizerError",
    "OrganizerInputError",
    "OrganizerOutputError",
    "PROMPT_CONTRACT_HASH",
    "PROMPT_VERSION",
    "RESULT_SCHEMA",
    "SELECTION_SCHEMA",
    "SYSTEM_PROMPT",
    "Type3Qwen36OrganizerV8",
    "normalize_text",
    "numeric_text_is_authorized",
    "split_authorized_answer",
]
