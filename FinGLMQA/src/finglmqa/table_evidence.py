"""Experimental, document-scoped table evidence fragments.

This module intentionally sits outside the frozen Phase 8 pipeline.  It turns
audited Phase 3/4 records into small row or narrative fragments and provides a
deterministic lexical retrieval boundary.  A fragment is evidence *input*, not
an answer: table numbers remain explicitly unauthorized until a later table
provenance/authorization contract validates them.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_EVEN
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


SCHEMA_VERSION = "finglmqa.experimental.table_evidence_fragment.v1"
BUILDER_VERSION = "table-evidence-index-experimental-v1"
MAX_TOP_K = 20
_SCORE_QUANTUM = Decimal("0.00000001")
_CJK_RE = re.compile(r"[\u3400-\u9fff]+")
_WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_TRIVIAL_NARRATIVE_RE = re.compile(
    r"^(?:单位[：:]?.*|币种[：:]?.*|[□√]?\s*适用\s*[□√]?\s*不适用|是|否)$"
)
_UNIT_HINT_RE = re.compile(r"^(?:单位|金额单位)\s*[：:]\s*(.+)$")
_SAFE_UNITS = {
    "元", "万元", "亿元", "千元", "人民币元", "人民币万元", "人民币千元",
    "人民币亿元", "股", "万股", "人", "吨", "万吨", "万吨/日", "平方米",
    "万平方米", "万件", "万辆", "双", "美元", "澳元", "兆瓦", "万千瓦时",
    "万套/万件", "mg/L",
}


class TableEvidenceError(RuntimeError):
    """Fail-closed table evidence build or retrieval error."""


def canonical_json(value: Any) -> str:
    """Return canonical JSON bytes-as-text without a trailing newline."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def portable_source_path(value: Any, root: Path) -> str:
    """Make a source path portable without accepting an untraceable path."""

    if not isinstance(value, str) or not value.strip():
        raise TableEvidenceError("source_markdown must be a non-empty string")
    path = Path(value)
    if not path.is_absolute():
        portable = path.as_posix()
    else:
        try:
            portable = path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            mirrored = root / "refs/source_markdown" / path.name
            if not mirrored.is_file():
                raise TableEvidenceError(f"source Markdown is outside the workspace: {path.name}")
            portable = mirrored.relative_to(root).as_posix()
    if Path(portable).is_absolute() or portable.startswith("../"):
        raise TableEvidenceError("source path was not made portable")
    return portable


def safe_unit_source(value: Any) -> dict[str, str] | None:
    """Accept only a small auditable subset of Phase 4 unit hints.

    In particular, malformed extraction such as ``币种：人`` is not promoted
    into provenance.  The original, trimmed hint is retained when accepted so
    a validator can compare it byte-for-byte with the table corpus.
    """

    if not isinstance(value, str) or not value.strip():
        return None
    raw = value.strip()
    compact = re.sub(r"\s+", "", raw)
    match = _UNIT_HINT_RE.fullmatch(compact)
    if not match or match.group(1) not in _SAFE_UNITS:
        return None
    return {"kind": "table_unit_hint", "value": raw}


def fragment_identity(
    *,
    fragment_kind: str,
    document_id: str,
    table_id: str,
    row_or_block: int | str,
) -> str:
    identity = [SCHEMA_VERSION, fragment_kind, document_id, table_id, row_or_block]
    return "tef-" + sha256_text(canonical_json(identity))[:32]


def seal_fragment(fragment: Mapping[str, Any]) -> dict[str, Any]:
    """Add a deterministic full-record digest.

    The digest covers every field except itself.  Consumers can therefore
    detect edits to content, identity, or provenance with one calculation.
    """

    row = dict(fragment)
    row.pop("fragment_sha256", None)
    row["fragment_sha256"] = sha256_text(canonical_json(row))
    return row


def validate_fragment(fragment: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "builder_version",
        "fragment_id",
        "fragment_kind",
        "document_id",
        "company_full",
        "company_name",
        "stock_code",
        "report_year",
        "table_id",
        "table_index",
        "section_path",
        "caption",
        "header_path",
        "row_index",
        "row_label",
        "column_labels",
        "cell_coordinates",
        "raw_cell_values",
        "raw_value_sha256",
        "year_source",
        "unit_source",
        "source_markdown",
        "source_line_range",
        "source_content_sha256",
        "source_block_id",
        "content",
        "retrieval_text",
        "provenance",
        "fragment_sha256",
    }
    if set(fragment) != required:
        missing = sorted(required - set(fragment))
        extra = sorted(set(fragment) - required)
        raise TableEvidenceError(f"fragment fields differ from v1 (missing={missing}, extra={extra})")
    if fragment["schema_version"] != SCHEMA_VERSION:
        raise TableEvidenceError("table evidence schema version mismatch")
    if fragment["builder_version"] != BUILDER_VERSION:
        raise TableEvidenceError("table evidence builder version mismatch")
    if fragment["fragment_kind"] not in {"table_row", "mixed_narrative"}:
        raise TableEvidenceError("unsupported table evidence fragment_kind")
    for name in ("fragment_id", "document_id", "table_id", "company_full", "source_markdown"):
        if not isinstance(fragment[name], str) or not fragment[name].strip():
            raise TableEvidenceError(f"{name} must be a non-empty string")
    if Path(str(fragment["source_markdown"])).is_absolute():
        raise TableEvidenceError("source_markdown must be portable")
    if isinstance(fragment["report_year"], bool) or not isinstance(fragment["report_year"], int):
        raise TableEvidenceError("report_year must be an integer")
    if isinstance(fragment["table_index"], bool) or not isinstance(fragment["table_index"], int):
        raise TableEvidenceError("table_index must be an integer")
    line_range = fragment["source_line_range"]
    if (
        not isinstance(line_range, list)
        or len(line_range) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in line_range)
        or line_range[0] > line_range[1]
    ):
        raise TableEvidenceError("source_line_range must be an ascending positive pair")
    coordinates = fragment["cell_coordinates"]
    if not isinstance(coordinates, list) or any(
        not isinstance(value, list)
        or len(value) != 2
        or any(isinstance(part, bool) or not isinstance(part, int) or part < 0 for part in value)
        for value in coordinates
    ):
        raise TableEvidenceError("cell_coordinates must be [row,col] integer pairs")
    if coordinates != sorted(coordinates) or len({tuple(value) for value in coordinates}) != len(coordinates):
        raise TableEvidenceError("cell_coordinates must be unique and sorted")
    values = fragment["raw_cell_values"]
    if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
        raise TableEvidenceError("raw_cell_values must be an array of strings")
    if fragment["raw_value_sha256"] != sha256_text(canonical_json(values)):
        raise TableEvidenceError("raw_value_sha256 mismatch")
    if fragment["fragment_sha256"] != sha256_text(
        canonical_json({key: value for key, value in fragment.items() if key != "fragment_sha256"})
    ):
        raise TableEvidenceError("fragment_sha256 mismatch")
    provenance = fragment["provenance"]
    if not isinstance(provenance, Mapping):
        raise TableEvidenceError("provenance must be an object")
    expected_provenance = {
        "source_kind",
        "source_schema_version",
        "source_record_ids",
        "table_content_sha256",
        "text_was_separated_from_table",
        "numeric_authorization",
    }
    if set(provenance) != expected_provenance:
        raise TableEvidenceError("provenance fields differ from v1")
    if provenance["numeric_authorization"] != "not_authorized_for_answer":
        raise TableEvidenceError("table fragments must not authorize answer numbers")
    if fragment["fragment_kind"] == "table_row":
        if not coordinates or len(coordinates) != len(values):
            raise TableEvidenceError("table_row coordinates and values must be non-empty and aligned")
        if fragment["source_block_id"] is not None:
            raise TableEvidenceError("table_row must not declare source_block_id")
    else:
        if coordinates or values or fragment["row_index"] is not None:
            raise TableEvidenceError("mixed_narrative must be separated from table cells")
        if not isinstance(fragment["source_block_id"], str) or not fragment["source_block_id"]:
            raise TableEvidenceError("mixed_narrative requires source_block_id")


def fragment_sort_key(fragment: Mapping[str, Any]) -> tuple[Any, ...]:
    kind_order = 0 if fragment["fragment_kind"] == "table_row" else 1
    row_or_line = (
        int(fragment["row_index"])
        if fragment["row_index"] is not None
        else int(fragment["source_line_range"][0])
    )
    return (
        str(fragment["document_id"]),
        int(fragment["table_index"]),
        str(fragment["table_id"]),
        kind_order,
        row_or_line,
        str(fragment["fragment_id"]),
    )


def meaningful_narrative(text: str, section_path: Sequence[str]) -> bool:
    normalized = re.sub(r"\s+", " ", text).strip()
    material = re.sub(r"[\s\W_]+", "", normalized)
    if len(material) < 20 or _TRIVIAL_NARRATIVE_RE.fullmatch(normalized):
        return False
    # A text block that is only the current Markdown heading is navigation,
    # not recovered narrative evidence.
    if section_path and normalized == str(section_path[-1]).strip():
        return False
    return "<table" not in normalized.lower() and "</table" not in normalized.lower()


def _normalize_search_text(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value).lower()).strip()


def _tokens(value: str) -> set[str]:
    text = _normalize_search_text(value)
    result = set(_WORD_RE.findall(text))
    for run in _CJK_RE.findall(text):
        result.update(run[index : index + 2] for index in range(max(0, len(run) - 1)))
        if len(run) <= 4:
            result.add(run)
    return {token for token in result if token}


@dataclass(frozen=True)
class _DocumentRange:
    start: int
    end: int


class TableEvidenceIndex:
    """Read-only, single-document lexical index over canonical JSONL.

    Construction records byte ranges only; it does not load the potentially
    large row corpus.  A query reads exactly one document range and cannot
    widen scope to another annual report.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise TableEvidenceError(f"table evidence index does not exist: {self.path}")
        self._ranges = self._scan_ranges()

    @property
    def document_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._ranges))

    def _scan_ranges(self) -> dict[str, _DocumentRange]:
        ranges: dict[str, _DocumentRange] = {}
        last_document: str | None = None
        last_key: tuple[Any, ...] | None = None
        with self.path.open("rb") as handle:
            while True:
                start = handle.tell()
                raw = handle.readline()
                if not raw:
                    break
                if not raw.strip():
                    raise TableEvidenceError("blank lines are forbidden in canonical table evidence JSONL")
                try:
                    row = json.loads(raw)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise TableEvidenceError("invalid table evidence JSONL") from exc
                if not isinstance(row, dict):
                    raise TableEvidenceError("table evidence JSONL rows must be objects")
                validate_fragment(row)
                expected = (canonical_json(row) + "\n").encode("utf-8")
                if raw != expected:
                    raise TableEvidenceError("table evidence JSONL is not canonical")
                key = fragment_sort_key(row)
                if last_key is not None and key <= last_key:
                    raise TableEvidenceError("table evidence fragments are not strictly sorted")
                last_key = key
                document_id = str(row["document_id"])
                end = handle.tell()
                if document_id != last_document:
                    if document_id in ranges:
                        raise TableEvidenceError("document ranges are non-contiguous")
                    ranges[document_id] = _DocumentRange(start=start, end=end)
                    last_document = document_id
                else:
                    ranges[document_id] = _DocumentRange(start=ranges[document_id].start, end=end)
        if not ranges:
            raise TableEvidenceError("table evidence index is empty")
        return ranges

    def iter_document(self, document_id: str) -> Iterator[dict[str, Any]]:
        if not isinstance(document_id, str) or not document_id.strip():
            raise TableEvidenceError("document_id is required")
        span = self._ranges.get(document_id)
        if span is None:
            raise TableEvidenceError("document_id is outside the table evidence allow-list")
        with self.path.open("rb") as handle:
            handle.seek(span.start)
            while handle.tell() < span.end:
                row = json.loads(handle.readline())
                if row["document_id"] != document_id:
                    raise TableEvidenceError("document range isolation failed")
                yield row

    def search(
        self,
        *,
        document_id: str,
        question: str,
        top_k: int = 5,
        fragment_kinds: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        if not isinstance(question, str) or not question.strip():
            raise TableEvidenceError("question must be a non-empty string")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= MAX_TOP_K:
            raise TableEvidenceError(f"top_k must be between 1 and {MAX_TOP_K}")
        kinds = set(fragment_kinds or ("table_row", "mixed_narrative"))
        if not kinds or not kinds.issubset({"table_row", "mixed_narrative"}):
            raise TableEvidenceError("fragment_kinds contains an unsupported value")

        candidates = [row for row in self.iter_document(document_id) if row["fragment_kind"] in kinds]
        query_tokens = _tokens(question)
        if not query_tokens or not candidates:
            return []
        token_rows = [_tokens(str(row["retrieval_text"])) for row in candidates]
        document_frequency: Counter[str] = Counter()
        for row_tokens in token_rows:
            document_frequency.update(query_tokens.intersection(row_tokens))
        total = len(candidates)
        normalized_question = _normalize_search_text(question)
        ranked: list[tuple[Decimal, tuple[Any, ...], dict[str, Any]]] = []
        for row, row_tokens in zip(candidates, token_rows):
            overlap = query_tokens.intersection(row_tokens)
            if not overlap:
                continue
            weighted = sum(math.log((total + 1) / (document_frequency[token] + 1)) + 1 for token in overlap)
            score = Decimal(str(weighted / max(1, len(query_tokens))))
            for value in (row["row_label"], row["caption"], *row["header_path"]):
                normalized_value = _normalize_search_text(str(value))
                if len(normalized_value) >= 2 and normalized_value in normalized_question:
                    score += Decimal("0.10000000")
            if row["fragment_kind"] == "mixed_narrative":
                score += Decimal("0.01000000")
            score = score.quantize(_SCORE_QUANTUM, rounding=ROUND_HALF_EVEN)
            result = dict(row)
            result["score"] = format(score, ".8f")
            ranked.append((score, fragment_sort_key(row), result))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return [item[2] for item in ranked[:top_k]]


def write_canonical_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> tuple[int, str]:
    """Atomically write validated, already-sorted fragments."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    digest = hashlib.sha256()
    previous_key: tuple[Any, ...] | None = None
    with temporary.open("wb") as handle:
        for source in rows:
            row = dict(source)
            validate_fragment(row)
            key = fragment_sort_key(row)
            if previous_key is not None and key <= previous_key:
                raise TableEvidenceError("builder output is not strictly sorted")
            previous_key = key
            raw = (canonical_json(row) + "\n").encode("utf-8")
            handle.write(raw)
            digest.update(raw)
            count += 1
    if not count:
        temporary.unlink(missing_ok=True)
        raise TableEvidenceError("refusing to publish an empty table evidence index")
    temporary.replace(path)
    return count, digest.hexdigest()
