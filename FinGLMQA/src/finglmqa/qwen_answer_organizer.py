"""Experimental Qwen answer organization over document-scoped evidence.

This module is deliberately separate from the frozen Phase 8/10 pipeline.  A
local Qwen model may only *order and select* normalized verbatim substrings
from evidence that has already been retrieved for one annual report.  It
cannot paraphrase, join fragments, invent citations, or authorize table
numbers.  The final answer is assembled deterministically after every proposal
has passed the local safety boundary.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import PurePosixPath
import re
import unicodedata
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from .contracts import semantic_sha256


RESULT_SCHEMA = "finglmqa.experimental.table_qwen_result.v1"
PROMPT_VERSION = "table-qwen-extractive-organizer-v1"
STRUCTURED_OUTPUT_SCHEMA_VERSION = "table-qwen-claims-v1"
MAX_CLAIMS = 4
MAX_PROMPT_CONTENT_CHARS = 2_000
MAX_PROMPT_TOTAL_CHARS = 24_000

SOURCE_KINDS = {"a2rag_text", "mixed_narrative", "table_row"}
# V1 intentionally has no provenance-bearing numeric authorization input.
# A future contract may add verifiable authorization IDs; accepting a bare
# caller assertion here would allow a fabricated number through the gate.
NUMERIC_AUTHORIZATIONS = {"not_authorized_for_answer"}
_ARABIC_NUMBER_RE = re.compile(r"[0-9]")

CLAIM_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["claims"],
    "properties": {
        "claims": {
            "type": "array",
            "maxItems": MAX_CLAIMS,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["text", "evidence_id"],
                "properties": {
                    "text": {"type": "string", "minLength": 1},
                    "evidence_id": {"type": "string", "minLength": 1},
                },
            },
        },
    },
}

_EVIDENCE_FIELDS = {
    "evidence_id",
    "source_kind",
    "document_id",
    "company",
    "stock_code",
    "report_year",
    "content",
    "source_markdown",
    "line_range",
    "numeric_authorization",
}


class QwenAnswerOrganizerError(RuntimeError):
    """Base error for invalid experimental input or model output."""


class QwenAnswerInputError(QwenAnswerOrganizerError):
    """The caller attempted to widen scope or supplied malformed evidence."""


class QwenAnswerOutputError(QwenAnswerOrganizerError):
    """The model/client did not return the frozen structured output shape."""


@runtime_checkable
class ChatCompletionClient(Protocol):
    """Small injectable boundary used by the real HTTP and fake clients."""

    def complete(self, body: Mapping[str, Any]) -> Mapping[str, Any]: ...


class OpenAICompatibleChatClient:
    """OpenAI-compatible client that never logs request or prompt content."""

    def __init__(self, base_url: str, *, timeout_seconds: float = 130.0) -> None:
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("base_url must be a non-empty string")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = float(timeout_seconds)

    def complete(self, body: Mapping[str, Any]) -> Mapping[str, Any]:
        # Keep pure contract/gate tests independent of the Phase 10 runtime;
        # the actual local-model runner uses .venv-phase10, which pins httpx.
        import httpx

        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(f"{self.base_url}/chat/completions", json=dict(body))
            response.raise_for_status()
            value = response.json()
        if not isinstance(value, Mapping):
            raise QwenAnswerOutputError("chat completion response must be an object")
        return value


def normalize_prose(value: str) -> str:
    """Normalize only compatibility characters and whitespace.

    The same operation is applied to the proposal and its one cited source.
    This permits harmless Markdown/newline differences while still requiring
    a literal contiguous substring and forbidding cross-fragment composition.
    """

    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value)).strip()


def _content_sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _identity(value: Mapping[str, Any]) -> tuple[str, str, str, int]:
    return (
        str(value["document_id"]),
        str(value["company"]),
        str(value["stock_code"]),
        int(value["report_year"]),
    )


def validate_evidence(
    value: Any,
    *,
    expected_identity: tuple[str, str, str, int],
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _EVIDENCE_FIELDS:
        raise QwenAnswerInputError("evidence fields differ from the experimental v1 boundary")
    row = dict(value)
    for field in (
        "evidence_id", "source_kind", "document_id", "company", "stock_code",
        "content", "source_markdown", "numeric_authorization",
    ):
        if not isinstance(row[field], str) or not row[field].strip():
            raise QwenAnswerInputError(f"evidence {field} must be a non-empty string")
    if row["source_kind"] not in SOURCE_KINDS:
        raise QwenAnswerInputError("evidence source_kind is unsupported")
    if row["numeric_authorization"] not in NUMERIC_AUTHORIZATIONS:
        raise QwenAnswerInputError("evidence numeric_authorization is unsupported")
    if row["source_kind"] in {"mixed_narrative", "table_row"} and (
        row["numeric_authorization"] != "not_authorized_for_answer"
    ):
        raise QwenAnswerInputError("table evidence cannot authorize answer numbers")
    if isinstance(row["report_year"], bool) or not isinstance(row["report_year"], int):
        raise QwenAnswerInputError("evidence report_year must be an integer")
    if not re.fullmatch(r"[0-9]{6}", row["stock_code"]):
        raise QwenAnswerInputError("evidence stock_code must contain six digits")
    source = PurePosixPath(row["source_markdown"])
    if source.is_absolute() or ".." in source.parts or "\\" in row["source_markdown"]:
        raise QwenAnswerInputError("evidence source_markdown must be workspace-relative")
    line_range = row["line_range"]
    if (
        not isinstance(line_range, list)
        or len(line_range) != 2
        or any(isinstance(part, bool) or not isinstance(part, int) or part < 1 for part in line_range)
        or line_range[1] < line_range[0]
    ):
        raise QwenAnswerInputError("evidence line_range must be an ascending positive pair")
    if _identity(row) != expected_identity:
        raise QwenAnswerInputError("evidence crossed the unique company/document/year scope")
    if not normalize_prose(row["content"]):
        raise QwenAnswerInputError("evidence content is empty after normalization")
    return row


class QwenAnswerOrganizer:
    """Call Qwen once, gate its proposals, then compose an extractive answer."""

    def __init__(
        self,
        client: ChatCompletionClient,
        *,
        model: str = "finglmqa-qwen3.6-27b",
    ) -> None:
        if not isinstance(client, ChatCompletionClient):
            raise TypeError("client must implement ChatCompletionClient.complete")
        if (
            not isinstance(model, str)
            or not re.fullmatch(r"[A-Za-z0-9_.:-]+", model.strip())
        ):
            raise ValueError("model must be a portable served model name")
        self.client = client
        self.model = model.strip()

    @property
    def model_config(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "temperature": 0,
            "top_p": 1,
            "seed": 0,
            "enable_thinking": False,
            "max_tokens": 768,
            "max_claims": MAX_CLAIMS,
            "prompt_version": PROMPT_VERSION,
            "structured_output_schema_version": STRUCTURED_OUTPUT_SCHEMA_VERSION,
        }

    @staticmethod
    def _prompt_evidence(evidence: Sequence[Mapping[str, Any]]) -> str:
        blocks: list[str] = []
        per_source_limit = min(
            MAX_PROMPT_CONTENT_CHARS,
            max(1, MAX_PROMPT_TOTAL_CHARS // max(1, len(evidence))),
        )
        for row in evidence:
            content = normalize_prose(str(row["content"]))
            # Every allow-listed evidence ID is represented in the prompt;
            # equal per-source caps prevent late table fragments from being
            # silently omitted when an early Markdown chunk is unusually long.
            content = content[:per_source_limit]
            blocks.append(json.dumps({
                "evidence_id": row["evidence_id"],
                "source_kind": row["source_kind"],
                "content": content,
            }, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        return "\n".join(blocks)

    @classmethod
    def _messages(
        cls,
        *,
        question: str,
        evidence: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": (
                    "你是逐字抽取式证据组织器。按最能回答问题的顺序返回最多4条claim。"
                    "每条claim只能绑定一个给定evidence_id，text必须从该证据content连续逐字复制；"
                    "不得改写、概括、跨证据拼接、补充连接句或常识。没有直接证据时返回空claims。"
                ),
            },
            {
                "role": "user",
                "content": f"问题：{question}\n候选证据（JSONL）：\n{cls._prompt_evidence(evidence)}",
            },
        ]

    def _request_body(
        self,
        *,
        question: str,
        evidence: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": self._messages(question=question, evidence=evidence),
            "temperature": 0,
            "top_p": 1,
            "seed": 0,
            "max_tokens": 768,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "table_qwen_extractive_claims",
                    "strict": True,
                    "schema": CLAIM_SCHEMA,
                },
            },
            "chat_template_kwargs": {"enable_thinking": False},
        }

    @staticmethod
    def _parse_response(value: Any, allowed_ids: set[str]) -> list[dict[str, str]]:
        try:
            if not isinstance(value, Mapping):
                raise TypeError
            choices = value["choices"]
            if not isinstance(choices, list) or len(choices) != 1:
                raise TypeError
            message = choices[0]["message"]
            content = message["content"]
            if not isinstance(content, str):
                raise TypeError
            parsed = json.loads(content)
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise QwenAnswerOutputError("Qwen response envelope or JSON is invalid") from exc
        if not isinstance(parsed, dict) or set(parsed) != {"claims"}:
            raise QwenAnswerOutputError("Qwen structured response fields are invalid")
        claims = parsed["claims"]
        if not isinstance(claims, list) or len(claims) > MAX_CLAIMS:
            raise QwenAnswerOutputError("Qwen claim count is invalid")
        checked: list[dict[str, str]] = []
        for row in claims:
            if not isinstance(row, dict) or set(row) != {"text", "evidence_id"}:
                raise QwenAnswerOutputError("Qwen claim fields are invalid")
            text = row["text"]
            evidence_id = row["evidence_id"]
            if not isinstance(text, str) or not normalize_prose(text):
                raise QwenAnswerOutputError("Qwen claim text is invalid")
            if not isinstance(evidence_id, str) or evidence_id not in allowed_ids:
                raise QwenAnswerOutputError("Qwen cited evidence outside the supplied allow-list")
            checked.append({"text": text, "evidence_id": evidence_id})
        return checked

    @staticmethod
    def _citation(row: Mapping[str, Any]) -> dict[str, Any]:
        identity = {
            "evidence_id": row["evidence_id"],
            "source_kind": row["source_kind"],
            "document_id": row["document_id"],
            "company": row["company"],
            "stock_code": row["stock_code"],
            "report_year": row["report_year"],
            "source_markdown": row["source_markdown"],
            "line_range": list(row["line_range"]),
            "content_sha256": _content_sha256(row["content"]),
        }
        return {
            "citation_id": "tqcite_" + semantic_sha256(identity)[:24],
            **identity,
        }

    @staticmethod
    def _source_snapshot(row: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "evidence_id": row["evidence_id"],
            "source_kind": row["source_kind"],
            "document_id": row["document_id"],
            "company": row["company"],
            "stock_code": row["stock_code"],
            "report_year": row["report_year"],
            "content": row["content"],
            "content_sha256": _content_sha256(row["content"]),
            "source_markdown": row["source_markdown"],
            "line_range": list(row["line_range"]),
            "numeric_authorization": row["numeric_authorization"],
        }

    def organize(
        self,
        *,
        case_id: str,
        question: str,
        document_id: str,
        company: str,
        stock_code: str,
        report_year: int,
        evidence: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        if not isinstance(case_id, str) or not case_id.strip():
            raise QwenAnswerInputError("case_id must be a non-empty string")
        for name, value in (("question", question), ("document_id", document_id), ("company", company)):
            if not isinstance(value, str) or not value.strip():
                raise QwenAnswerInputError(f"{name} must be a non-empty string")
        if not isinstance(stock_code, str) or not re.fullmatch(r"[0-9]{6}", stock_code):
            raise QwenAnswerInputError("stock_code must contain six digits")
        if isinstance(report_year, bool) or not isinstance(report_year, int):
            raise QwenAnswerInputError("report_year must be an integer")
        if not isinstance(evidence, Sequence) or isinstance(evidence, (str, bytes)):
            raise QwenAnswerInputError("evidence must be an array")

        expected_identity = (document_id, company, stock_code, report_year)
        checked = [validate_evidence(row, expected_identity=expected_identity) for row in evidence]
        evidence_ids = [row["evidence_id"] for row in checked]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise QwenAnswerInputError("evidence IDs must be unique")

        snapshots = [self._source_snapshot(row) for row in checked]
        authorization_snapshot = [
            {
                "evidence_id": row["evidence_id"],
                "numeric_authorization": row["numeric_authorization"],
            }
            for row in checked
        ]
        by_id = {row["evidence_id"]: row for row in checked}
        proposals: list[dict[str, str]] = []
        output_error: QwenAnswerOutputError | None = None
        try:
            raw = self.client.complete(self._request_body(question=question, evidence=checked))
            proposals = self._parse_response(raw, set(by_id))
        except Exception as exc:
            output_error = exc if isinstance(exc, QwenAnswerOutputError) else QwenAnswerOutputError(
                "Qwen completion failed closed"
            )

        accepted: list[dict[str, Any]] = []
        citations: list[dict[str, Any]] = []
        rejections: list[dict[str, Any]] = []
        seen_evidence_ids: set[str] = set()
        if output_error is None:
            for proposal_ordinal, proposal in enumerate(proposals):
                source = by_id[proposal["evidence_id"]]
                text = normalize_prose(proposal["text"])
                source_text = normalize_prose(source["content"])
                reason: str | None = None
                if text not in source_text:
                    reason = "NOT_NORMALIZED_VERBATIM_SUBSTRING"
                elif source["source_kind"] == "table_row":
                    reason = "TABLE_ROW_CLAIM_FORBIDDEN"
                elif (
                    _ARABIC_NUMBER_RE.search(text) is not None
                    and source["numeric_authorization"] != "authorized_for_answer"
                ):
                    reason = "UNAUTHORIZED_NUMERIC_CLAIM"
                elif source["evidence_id"] in seen_evidence_ids:
                    reason = "DUPLICATE_CLAIM"
                if reason is not None:
                    rejections.append({
                        "proposal_ordinal": proposal_ordinal,
                        "evidence_id": source["evidence_id"],
                        "reason": reason,
                    })
                    continue
                seen_evidence_ids.add(source["evidence_id"])
                citation = self._citation(source)
                citations.append(citation)
                accepted.append({
                    "ordinal": len(accepted),
                    "text": text,
                    "evidence_id": source["evidence_id"],
                    "citation_id": citation["citation_id"],
                    "document_id": source["document_id"],
                    "company": source["company"],
                    "stock_code": source["stock_code"],
                    "report_year": source["report_year"],
                    "source_kind": source["source_kind"],
                })

        if output_error is not None:
            status = "error"
            outcome = "generator_invalid_output"
        elif accepted:
            status = "ok"
            outcome = "accepted"
        elif proposals:
            status = "not_found"
            outcome = "generator_rejected_by_gate"
        else:
            status = "not_found"
            outcome = "generator_refused"

        result = {
            "schema_version": RESULT_SCHEMA,
            "case_id": case_id,
            "question": question,
            "scope": {
                "document_id": document_id,
                "company": company,
                "stock_code": stock_code,
                "report_year": report_year,
            },
            "status": status,
            "generator_outcome": outcome,
            "answer": "\n".join(row["text"] for row in accepted),
            "accepted_claim_projection": accepted,
            "citation_projection": citations,
            "rejections": rejections,
            "source_snapshot": snapshots,
            "authorization_snapshot": authorization_snapshot,
            "model_config": self.model_config,
            "result_fingerprint": "",
        }
        result["result_fingerprint"] = semantic_sha256({
            key: value for key, value in result.items() if key != "result_fingerprint"
        })
        return result


__all__ = [
    "CLAIM_SCHEMA",
    "ChatCompletionClient",
    "MAX_CLAIMS",
    "OpenAICompatibleChatClient",
    "PROMPT_VERSION",
    "QwenAnswerInputError",
    "QwenAnswerOrganizer",
    "QwenAnswerOrganizerError",
    "QwenAnswerOutputError",
    "RESULT_SCHEMA",
    "normalize_prose",
    "validate_evidence",
]
