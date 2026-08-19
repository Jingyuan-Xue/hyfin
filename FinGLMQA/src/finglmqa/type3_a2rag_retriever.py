"""Corpus-scoped Type 3 text parsing, indexing, and document-first retrieval.

The source Markdown is the sole evidence authority.  This module deliberately
does not accept benchmark annotations: builders consume a corpus profile plus
Phase 3 table ranges, while the retriever consumes only a question and an
explicit ``document_id``.
"""

from __future__ import annotations

import bisect
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np


ATOM_SCHEMA = "finglmqa.type3.a2rag.text_atom.v1"
UNIT_SCHEMA = "finglmqa.type3.a2rag.retrieval_unit.v1"
INDEX_SCHEMA = "finglmqa.type3.a2rag.index_manifest.v1"
BUILDER_VERSION = "type3-a2rag-text-v1"
EMBEDDING_DIMENSION = 1024

_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
_LIST_RE = re.compile(
    r"^\s*(?:[-+*][ \t]+|\d{1,3}[.)、][ \t]*|[（(]?[一二三四五六七八九十]+[)）、.．][ \t]*)"
)
_WORD_RE = re.compile(r"[A-Za-z]+(?:[-_.][A-Za-z0-9]+)*|\d+(?:\.\d+)?|[\u3400-\u9fff]")
_TABLE_OPEN_RE = re.compile(r"<\s*table\b", re.IGNORECASE)
_TABLE_CLOSE_RE = re.compile(r"<\s*/\s*table\s*>", re.IGNORECASE)
_DISPLAY_MATH_RE = re.compile(r"^\s*\$\$[\s\S]*\$\$\s*$")


class Type3A2RAGError(RuntimeError):
    """Fail-closed error for Type 3 A2RAG artifacts or retrieval."""


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def semantic_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8", errors="strict") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise Type3A2RAGError(f"JSON object required: {path}")
    return value


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="strict") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise Type3A2RAGError(f"JSON object required: {path}:{line_number}")
            yield value


def _stable_id(prefix: str, corpus_id: str, document_id: str, payload: Mapping[str, Any]) -> str:
    digest = semantic_sha256(
        {"corpus_id": corpus_id, "document_id": document_id, **dict(payload)}
    )[:28]
    return f"{prefix}:{corpus_id}:{digest}"


def shard_id(document_id: str) -> str:
    return hashlib.sha256(document_id.encode("utf-8")).hexdigest()[:24]


def line_starts(source: str) -> list[int]:
    starts = [0]
    starts.extend(match.end() for match in re.finditer("\n", source))
    return starts


def line_range_for(starts: Sequence[int], start: int, end: int) -> list[int]:
    if end <= start:
        line = bisect.bisect_right(starts, start)
        return [line, line]
    return [bisect.bisect_right(starts, start), bisect.bisect_right(starts, end - 1)]


def byte_offsets_for(source: str, offsets: Iterable[int]) -> dict[int, int]:
    """Map selected Python character offsets to exact UTF-8 byte offsets."""

    wanted = sorted(set(offsets))
    if not wanted or wanted[0] < 0 or wanted[-1] > len(source):
        raise Type3A2RAGError("character offset outside source")
    result: dict[int, int] = {}
    previous_char = 0
    previous_byte = 0
    for offset in wanted:
        previous_byte += len(source[previous_char:offset].encode("utf-8"))
        result[offset] = previous_byte
        previous_char = offset
    return result


@dataclass(frozen=True)
class TableRange:
    table_id: str
    start: int
    end: int
    caption: str
    line_range: tuple[int, int]


def validate_table_ranges(
    source: str,
    rows: Iterable[Mapping[str, Any]],
    *,
    document_id: str,
) -> list[TableRange]:
    """Validate complete Phase 3 ranges against the current strict UTF-8 source."""

    ranges: list[TableRange] = []
    starts = line_starts(source)
    seen_ids: set[str] = set()
    for ordinal, row in enumerate(rows, 1):
        if row.get("document_id") != document_id:
            raise Type3A2RAGError(f"table document mismatch: {document_id}:{ordinal}")
        table_id = str(row.get("table_id") or "")
        if not table_id or table_id in seen_ids:
            raise Type3A2RAGError(f"invalid/duplicate table_id: {document_id}:{table_id}")
        seen_ids.add(table_id)
        char_range = row.get("char_range")
        if (
            not isinstance(char_range, list)
            or len(char_range) != 2
            or not all(isinstance(value, int) for value in char_range)
        ):
            raise Type3A2RAGError(f"invalid table char_range: {table_id}")
        start, end = char_range
        if start < 0 or end <= start or end > len(source):
            raise Type3A2RAGError(f"table char_range outside source: {table_id}")
        raw = row.get("raw_markdown")
        if not isinstance(raw, str) or source[start:end] != raw:
            raise Type3A2RAGError(f"table range/raw_markdown mismatch: {table_id}")
        if not _TABLE_OPEN_RE.match(raw) or not _TABLE_CLOSE_RE.search(raw):
            raise Type3A2RAGError(f"table range lacks complete HTML table: {table_id}")
        declared_lines = row.get("line_range")
        if not (
            isinstance(declared_lines, list)
            and len(declared_lines) == 2
            and all(isinstance(value, int) for value in declared_lines)
        ):
            raise Type3A2RAGError(f"invalid table line_range: {table_id}")
        actual_lines = line_range_for(starts, start, end)
        if actual_lines != declared_lines:
            raise Type3A2RAGError(
                f"table line_range differs: {table_id}: {declared_lines}!={actual_lines}"
            )
        ranges.append(
            TableRange(
                table_id=table_id,
                start=start,
                end=end,
                caption=str(row.get("caption") or "").strip(),
                line_range=(declared_lines[0], declared_lines[1]),
            )
        )
    ranges.sort(key=lambda value: (value.start, value.end, value.table_id))
    for previous, current in zip(ranges, ranges[1:]):
        if previous.end > current.start:
            raise Type3A2RAGError(
                f"overlapping table ranges: {previous.table_id}, {current.table_id}"
            )

    # Every HTML table opener/closer must be inside exactly one declared range.
    for pattern, label in ((_TABLE_OPEN_RE, "opener"), (_TABLE_CLOSE_RE, "closer")):
        for match in pattern.finditer(source):
            covering = sum(value.start <= match.start() < value.end for value in ranges)
            if covering != 1:
                raise Type3A2RAGError(
                    f"uncovered/ambiguous table {label} at char {match.start()} in {document_id}"
                )
    return ranges


def _iter_lines(source: str, start: int, end: int) -> Iterator[tuple[int, int, str]]:
    cursor = start
    while cursor < end:
        newline = source.find("\n", cursor, end)
        line_end = end if newline < 0 else newline
        yield cursor, line_end, source[cursor:line_end]
        cursor = end if newline < 0 else newline + 1


def _heading_value(line: str) -> tuple[int, str] | None:
    match = _HEADING_RE.match(line)
    if not match:
        return None
    value = match.group(2).strip()
    return len(match.group(1)), value


def _with_heading(stack: list[tuple[int, str]], level: int, value: str) -> list[tuple[int, str]]:
    kept = [row for row in stack if row[0] < level]
    kept.append((level, value))
    return kept


def build_text_atoms(
    *,
    corpus_id: str,
    document_id: str,
    source_markdown: str,
    source_sha256: str,
    source: str,
    tables: Sequence[TableRange],
) -> list[dict[str, Any]]:
    """Rebuild exact heading/paragraph/list atoms outside complete table ranges."""

    starts = line_starts(source)
    heading_stack: list[tuple[int, str]] = []
    provisional: list[dict[str, Any]] = []
    segment_index = 0

    def emit(kind: str, start: int, end: int, heading_path: Sequence[str]) -> None:
        if end <= start or not source[start:end].strip():
            return
        content = source[start:end]
        atom_id = _stable_id(
            "a2atom",
            corpus_id,
            document_id,
            {
                "kind": kind,
                "char_range": [start, end],
                "content_sha256": sha256_text(content),
            },
        )
        provisional.append(
            {
                "schema_version": ATOM_SCHEMA,
                "builder_version": BUILDER_VERSION,
                "atom_id": atom_id,
                "corpus_id": corpus_id,
                "document_id": document_id,
                "atom_kind": kind,
                "segment_index": segment_index,
                "heading_path": list(heading_path),
                "content": content,
                "content_sha256": sha256_text(content),
                "line_range": line_range_for(starts, start, end),
                "char_range": [start, end],
                "source_markdown": source_markdown,
                "source_sha256": source_sha256,
                "adjacent_table_ids": [],
                "table_adjacency": [],
            }
        )

    boundaries: list[tuple[int, int, TableRange | None]] = []
    cursor = 0
    for table in tables:
        boundaries.append((cursor, table.start, None))
        boundaries.append((table.start, table.end, table))
        cursor = table.end
    boundaries.append((cursor, len(source), None))

    for start, end, table in boundaries:
        if table is not None:
            segment_index += 1
            continue
        pending_kind: str | None = None
        pending_start = 0
        pending_end = 0
        pending_heading: list[str] = []

        def flush() -> None:
            nonlocal pending_kind, pending_start, pending_end, pending_heading
            if pending_kind is not None:
                emit(pending_kind, pending_start, pending_end, pending_heading)
            pending_kind = None
            pending_heading = []

        for line_start, line_end, line in _iter_lines(source, start, end):
            if not line.strip():
                flush()
                continue
            heading = _heading_value(line)
            if heading is not None:
                flush()
                level, value = heading
                heading_stack = _with_heading(heading_stack, level, value)
                emit("heading", line_start, line_end, [item[1] for item in heading_stack])
                segment_index += 1
                continue
            kind = "list" if _LIST_RE.match(line) else "paragraph"
            current_heading = [item[1] for item in heading_stack]
            if kind == "list" and pending_kind == "list" and pending_heading == current_heading:
                pending_end = line_end
                continue
            # Consecutive prose lines between blank lines are one Markdown paragraph.
            if kind == "paragraph" and pending_kind == "paragraph" and pending_heading == current_heading:
                pending_end = line_end
                continue
            flush()
            pending_kind = kind
            pending_start = line_start
            pending_end = line_end
            pending_heading = current_heading
        flush()

    provisional.sort(key=lambda row: (row["char_range"][0], row["char_range"][1], row["atom_id"]))
    table_positions = {table.table_id: table for table in tables}
    # Attach the closest narrative atom on either side, stopping at a heading or table.
    for table in tables:
        before = [
            row for row in provisional
            if row["char_range"][1] <= table.start and row["atom_kind"] != "heading"
        ]
        after = [
            row for row in provisional
            if row["char_range"][0] >= table.end and row["atom_kind"] != "heading"
        ]
        chosen: list[tuple[dict[str, Any], str, int]] = []
        if before:
            row = before[-1]
            intervening = [other for other in tables if row["char_range"][1] <= other.start < table.start]
            headings = [
                other for other in provisional
                if row["char_range"][1] <= other["char_range"][0] < table.start
                and other["atom_kind"] == "heading"
            ]
            if not intervening and not headings:
                chosen.append((row, "before", table.start - row["char_range"][1]))
        if after:
            row = after[0]
            intervening = [other for other in tables if table.end < other.end <= row["char_range"][0]]
            headings = [
                other for other in provisional
                if table.end <= other["char_range"][0] < row["char_range"][0]
                and other["atom_kind"] == "heading"
            ]
            if not intervening and not headings:
                chosen.append((row, "after", row["char_range"][0] - table.end))
        for row, relation, distance in chosen:
            row["adjacent_table_ids"].append(table.table_id)
            row["table_adjacency"].append(
                {
                    "table_id": table.table_id,
                    "relation": relation,
                    "char_distance": distance,
                    "caption": table.caption,
                }
            )

    offsets = [value for row in provisional for value in row["char_range"]]
    byte_map = byte_offsets_for(source, offsets)
    for row in provisional:
        start, end = row["char_range"]
        row["byte_range"] = [byte_map[start], byte_map[end]]
        row["adjacent_table_ids"] = sorted(set(row["adjacent_table_ids"]))
        row["table_adjacency"].sort(key=lambda value: (value["table_id"], value["relation"]))
        if row["content"] != source[start:end]:
            raise Type3A2RAGError(f"atom source alignment failed: {row['atom_id']}")
        for table_id in row["adjacent_table_ids"]:
            if table_id not in table_positions:
                raise Type3A2RAGError(f"unknown adjacent table: {table_id}")
    return provisional


def build_retrieval_units(
    *,
    corpus_id: str,
    document_id: str,
    source_markdown: str,
    source_sha256: str,
    source: str,
    atoms: Sequence[Mapping[str, Any]],
    target_chars: int = 900,
    max_chars: int = 1800,
) -> list[dict[str, Any]]:
    """Pack whole atoms without crossing a heading boundary or table barrier."""

    if target_chars < 1 or max_chars < target_chars:
        raise Type3A2RAGError("invalid unit sizing")
    units: list[dict[str, Any]] = []
    groups: list[list[Mapping[str, Any]]] = []
    current: list[Mapping[str, Any]] = []

    def flush() -> None:
        nonlocal current
        if current:
            groups.append(current)
        current = []

    for atom in atoms:
        if atom["atom_kind"] == "heading":
            flush()
            # Headings remain exact, persistent atoms and are injected into the
            # retrieval text of following narrative units.  They are not
            # standalone answer evidence/dense units: annual reports contain
            # many layout headings and indexing each one would both swamp
            # narrative recall and exceed the fail-closed 150k unit budget.
            continue
        if current:
            first = current[0]
            crosses = (
                atom["segment_index"] != first["segment_index"]
                or atom["heading_path"] != first["heading_path"]
            )
            proposed = atom["char_range"][1] - first["char_range"][0]
            if crosses or (proposed > max_chars and len(current) > 0):
                flush()
        current.append(atom)
        span = current[-1]["char_range"][1] - current[0]["char_range"][0]
        if span >= target_chars:
            flush()
    flush()

    starts = line_starts(source)
    offsets = [
        value
        for group in groups
        for value in (group[0]["char_range"][0], group[-1]["char_range"][1])
    ]
    byte_map = byte_offsets_for(source, offsets)
    for ordinal, group in enumerate(groups):
        start = int(group[0]["char_range"][0])
        end = int(group[-1]["char_range"][1])
        content = source[start:end]
        atom_ids = [str(row["atom_id"]) for row in group]
        kinds = sorted({str(row["atom_kind"]) for row in group})
        heading_path = list(group[0]["heading_path"])
        adjacent = sorted({value for row in group for value in row["adjacent_table_ids"]})
        adjacency = sorted(
            {
                (str(value["table_id"]), str(value["relation"]), str(value.get("caption") or ""))
                for row in group
                for value in row["table_adjacency"]
            }
        )
        unit_id = _stable_id(
            "a2unit",
            corpus_id,
            document_id,
            {
                "char_range": [start, end],
                "atom_ids": atom_ids,
                "content_sha256": sha256_text(content),
            },
        )
        units.append(
            {
                "schema_version": UNIT_SCHEMA,
                "builder_version": BUILDER_VERSION,
                "unit_id": unit_id,
                "corpus_id": corpus_id,
                "document_id": document_id,
                "document_unit_ordinal": ordinal,
                "unit_kind": "heading" if kinds == ["heading"] else "narrative",
                "content_char_count": end - start,
                "oversize_single_atom": (end - start > max_chars and len(group) == 1),
                "atom_kinds": kinds,
                "atom_ids": atom_ids,
                "segment_index": group[0]["segment_index"],
                "heading_path": heading_path,
                "content": content,
                "content_sha256": sha256_text(content),
                "line_range": line_range_for(starts, start, end),
                "char_range": [start, end],
                "byte_range": [byte_map[start], byte_map[end]],
                "source_markdown": source_markdown,
                "source_sha256": source_sha256,
                "adjacent_table_ids": adjacent,
                "table_adjacency": [
                    {"table_id": table_id, "relation": relation, "caption": caption}
                    for table_id, relation, caption in adjacency
                ],
            }
        )
    if len({row["unit_id"] for row in units}) != len(units):
        raise Type3A2RAGError(f"duplicate retrieval unit ID: {document_id}")
    return units


def heading_text(unit: Mapping[str, Any]) -> str:
    return " > ".join(str(value) for value in unit.get("heading_path") or [] if str(value).strip())


def adjacency_text(unit: Mapping[str, Any]) -> str:
    captions = [
        str(value.get("caption") or "").strip()
        for value in unit.get("table_adjacency") or []
        if str(value.get("caption") or "").strip()
    ]
    return " ".join(dict.fromkeys(captions))


def retrieval_text(unit: Mapping[str, Any], *, include_heading: bool) -> str:
    pieces: list[str] = []
    if include_heading and heading_text(unit):
        pieces.append(f"章节：{heading_text(unit)}")
    pieces.append(str(unit["content"]))
    return "\n".join(pieces)


def tokenize(value: str) -> list[str]:
    """Deterministic mixed Chinese-character/bigram and Latin tokenization."""

    raw = [match.group(0).lower() for match in _WORD_RE.finditer(value)]
    tokens: list[str] = []
    chinese_run: list[str] = []

    def flush_chinese() -> None:
        nonlocal chinese_run
        if not chinese_run:
            return
        tokens.extend(chinese_run)
        tokens.extend(
            chinese_run[index] + chinese_run[index + 1]
            for index in range(len(chinese_run) - 1)
        )
        chinese_run = []

    for value_token in raw:
        if len(value_token) == 1 and "\u3400" <= value_token <= "\u9fff":
            chinese_run.append(value_token)
        else:
            flush_chinese()
            tokens.append(value_token)
    flush_chinese()
    return tokens


def is_formula_only_block(value: str) -> bool:
    """Identify MinerU display-math/OCR arrays that are not narrative evidence."""

    if not _DISPLAY_MATH_RE.fullmatch(value):
        return False
    lowered = value.lower()
    latex_commands = len(re.findall(r"\\[a-z]+", lowered))
    han_characters = len(re.findall(r"[\u3400-\u9fff]", value))
    return latex_commands >= 20 and han_characters < 20


def build_bm25_shard(units: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    postings: dict[str, dict[str, list[list[int]]]] = {
        "body": defaultdict(list),
        "heading": defaultdict(list),
        "adjacency": defaultdict(list),
    }
    lengths = {"body": [], "heading": [], "adjacency": []}
    for ordinal, unit in enumerate(units):
        fields = {
            "body": str(unit["content"]),
            "heading": heading_text(unit),
            "adjacency": adjacency_text(unit),
        }
        for field, text in fields.items():
            counts = Counter(tokenize(text))
            lengths[field].append(sum(counts.values()))
            for token, frequency in sorted(counts.items()):
                postings[field][token].append([ordinal, frequency])
    return {
        "schema_version": "finglmqa.type3.a2rag.bm25_shard.v1",
        "unit_count": len(units),
        "unit_ids": [unit["unit_id"] for unit in units],
        "field_lengths": lengths,
        "postings": {
            field: dict(sorted(values.items())) for field, values in postings.items()
        },
        "tokenizer": "latin-word+han-unigram+han-bigram-v1",
    }


def _bm25_scores(
    shard: Mapping[str, Any],
    query: str,
    *,
    include_heading: bool,
    include_adjacency: bool,
) -> np.ndarray:
    count = int(shard["unit_count"])
    scores = np.zeros(count, dtype=np.float32)
    query_terms = Counter(tokenize(query))
    selected_fields = [("body", 1.0)]
    if include_heading:
        selected_fields.append(("heading", 1.25))
    if include_adjacency:
        selected_fields.append(("adjacency", 0.55))
    lengths = np.zeros(count, dtype=np.float32)
    for field, weight in selected_fields:
        lengths += np.asarray(shard["field_lengths"][field], dtype=np.float32) * weight
    average = float(np.mean(lengths)) if count else 1.0
    if average <= 0:
        average = 1.0
    k1, b = 1.2, 0.75
    for term, query_frequency in query_terms.items():
        combined: dict[int, float] = defaultdict(float)
        for field, weight in selected_fields:
            for ordinal, frequency in shard["postings"][field].get(term, []):
                combined[int(ordinal)] += float(frequency) * weight
        document_frequency = len(combined)
        if not document_frequency:
            continue
        inverse = math.log(1.0 + (count - document_frequency + 0.5) / (document_frequency + 0.5))
        for ordinal, frequency in combined.items():
            denominator = frequency + k1 * (1.0 - b + b * float(lengths[ordinal]) / average)
            scores[ordinal] += float(query_frequency) * inverse * frequency * (k1 + 1.0) / denominator
    return scores


def _rank(scores: np.ndarray) -> list[int]:
    return sorted(range(len(scores)), key=lambda index: (-float(scores[index]), index))


class Type3A2RAGRetriever:
    """Document-sharded dense/BM25 retriever with mandatory prefiltering."""

    def __init__(self, index_dir: Path | str, *, model_path: Path | str | None = None) -> None:
        self.index_dir = Path(index_dir)
        self.manifest = read_json(self.index_dir / "index_manifest.json")
        if self.manifest.get("schema_version") != INDEX_SCHEMA:
            raise Type3A2RAGError("unsupported A2RAG index manifest")
        documents = self.manifest.get("documents")
        if not isinstance(documents, list):
            raise Type3A2RAGError("manifest documents must be a list")
        self.documents = {str(row["document_id"]): dict(row) for row in documents}
        if len(self.documents) != len(documents):
            raise Type3A2RAGError("duplicate document_id in A2RAG manifest")
        configured = model_path or self.manifest["embedding_model"]["local_path"]
        self.model_path = Path(configured)
        self._model: Any | None = None
        self._shard_cache: OrderedDict[
            str, tuple[list[dict[str, Any]], dict[str, Any], np.ndarray, np.ndarray]
        ] = OrderedDict()

    def _load_shard(
        self, document_id: str
    ) -> tuple[list[dict[str, Any]], dict[str, Any], np.ndarray, np.ndarray]:
        cached = self._shard_cache.get(document_id)
        if cached is not None:
            self._shard_cache.move_to_end(document_id)
            return cached
        document = self.documents[document_id]
        shard_dir = self.index_dir / str(document["shard_path"])
        units = list(read_jsonl(shard_dir / "units.jsonl"))
        if len(units) != int(document["unit_count"]):
            raise Type3A2RAGError(f"unit shard count mismatch: {document_id}")
        bm25 = read_json(shard_dir / "bm25.json")
        context = np.load(shard_dir / "dense_context.npy", mmap_mode="r")
        content = np.load(shard_dir / "dense_content.npy", mmap_mode="r")
        expected = (len(units), EMBEDDING_DIMENSION)
        if context.shape != expected or content.shape != expected:
            raise Type3A2RAGError(f"dense shard shape mismatch: {document_id}")
        value = (units, bm25, context, content)
        self._shard_cache[document_id] = value
        self._shard_cache.move_to_end(document_id)
        while len(self._shard_cache) > 4:
            self._shard_cache.popitem(last=False)
        return value

    def _load_model(self) -> Any:
        if self._model is None:
            if not self.model_path.is_dir():
                raise Type3A2RAGError(f"local embedding model missing: {self.model_path}")
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.model_path.as_posix(), device="cuda", local_files_only=True
            )
            self._model.max_seq_length = 8192
            dimension = int(self._model.get_sentence_embedding_dimension() or 0)
            if dimension != EMBEDDING_DIMENSION:
                raise Type3A2RAGError(f"embedding dimension differs: {dimension}")
        return self._model

    def encode_queries(self, values: Sequence[str], *, batch_size: int = 8) -> np.ndarray:
        vectors = self._load_model().encode(
            list(values),
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        result = np.asarray(vectors, dtype=np.float32)
        if result.ndim != 2 or result.shape[1] != EMBEDDING_DIMENSION:
            raise Type3A2RAGError(f"invalid query vector shape: {result.shape}")
        return result

    def retrieve(
        self,
        question: str,
        *,
        document_id: str,
        top_k: int = 15,
        mode: str = "hybrid",
        include_heading: bool = True,
        include_adjacency: bool = True,
        query_vector: np.ndarray | Sequence[float] | None = None,
    ) -> dict[str, Any]:
        if not isinstance(question, str) or not question.strip():
            raise Type3A2RAGError("question must be non-empty")
        if document_id not in self.documents:
            raise KeyError(f"unknown document_id: {document_id}")
        if mode not in {"dense", "sparse", "hybrid"}:
            raise Type3A2RAGError(f"unsupported retrieval mode: {mode}")
        if not isinstance(top_k, int) or top_k < 1 or top_k > 100:
            raise Type3A2RAGError("top_k must be in [1, 100]")

        # Security property: resolve one document shard before reading/scoring candidates.
        units, bm25, context_matrix, content_matrix = self._load_shard(document_id)

        dense_scores: np.ndarray | None = None
        sparse_scores: np.ndarray | None = None
        if mode in {"dense", "hybrid"}:
            matrix = context_matrix if include_heading else content_matrix
            if query_vector is None:
                query = self.encode_queries([question])[0]
            else:
                query = np.asarray(query_vector, dtype=np.float32)
                if query.shape != (EMBEDDING_DIMENSION,):
                    raise Type3A2RAGError(f"query vector must have shape ({EMBEDDING_DIMENSION},)")
                norm = float(np.linalg.norm(query))
                if not np.isfinite(norm) or norm <= 0:
                    raise Type3A2RAGError("query vector must be finite and non-zero")
                query = query / norm
            dense_scores = np.asarray(matrix @ query, dtype=np.float32)
        if mode in {"sparse", "hybrid"}:
            sparse_scores = _bm25_scores(
                bm25,
                question,
                include_heading=include_heading,
                include_adjacency=include_adjacency,
            )

        if mode == "dense":
            assert dense_scores is not None
            order = _rank(dense_scores)
            combined = dense_scores
        elif mode == "sparse":
            assert sparse_scores is not None
            order = _rank(sparse_scores)
            combined = sparse_scores
        else:
            assert dense_scores is not None and sparse_scores is not None
            dense_order = _rank(dense_scores)
            sparse_order = _rank(sparse_scores)
            combined = np.zeros(len(units), dtype=np.float32)
            for rank, ordinal in enumerate(dense_order, 1):
                combined[ordinal] += 1.0 / (60.0 + rank)
            for rank, ordinal in enumerate(sparse_order, 1):
                combined[ordinal] += 1.0 / (60.0 + rank)
            order = _rank(combined)

        formula_only_ordinals = {
            ordinal for ordinal, unit in enumerate(units)
            if is_formula_only_block(str(unit["content"]))
        }
        order = [ordinal for ordinal in order if ordinal not in formula_only_ordinals]

        candidates = []
        for rank, ordinal in enumerate(order[: min(top_k, len(order))], 1):
            unit = units[ordinal]
            candidates.append(
                {
                    "rank": rank,
                    "score": format(float(combined[ordinal]), ".8f"),
                    "dense_score": (
                        format(float(dense_scores[ordinal]), ".8f")
                        if dense_scores is not None else None
                    ),
                    "sparse_score": (
                        format(float(sparse_scores[ordinal]), ".8f")
                        if sparse_scores is not None else None
                    ),
                    "content_truncated": False,
                    "requires_atom_projection": True,
                    **unit,
                }
            )
        return {
            "schema_version": "finglmqa.type3.a2rag.retrieval_result.v1",
            "corpus_id": self.manifest["corpus_id"],
            "document_id": document_id,
            "prefilter_applied_before_scoring": True,
            "candidate_document_count": 1,
            "candidate_unit_count": len(units),
            "formula_only_units_filtered": len(formula_only_ordinals),
            "mode": mode,
            "include_heading": include_heading,
            "include_adjacency": include_adjacency,
            "top_k": top_k,
            "candidates": candidates,
        }

    def retrieve_many(
        self,
        questions: Sequence[str],
        *,
        document_id: str,
        top_k: int = 15,
        mode: str = "hybrid",
        include_heading: bool = True,
        include_adjacency: bool = True,
    ) -> list[dict[str, Any]]:
        """Batch facet queries while loading/scoping the document shard once."""

        if document_id not in self.documents:
            raise KeyError(f"unknown document_id: {document_id}")
        values = list(questions)
        if not values or any(not isinstance(value, str) or not value.strip() for value in values):
            raise Type3A2RAGError("questions must be a non-empty sequence of strings")
        self._load_shard(document_id)
        vectors: np.ndarray | None = None
        if mode in {"dense", "hybrid"}:
            vectors = self.encode_queries(values, batch_size=8)
        return [
            self.retrieve(
                question,
                document_id=document_id,
                top_k=top_k,
                mode=mode,
                include_heading=include_heading,
                include_adjacency=include_adjacency,
                query_vector=(vectors[index] if vectors is not None else None),
            )
            for index, question in enumerate(values)
        ]
