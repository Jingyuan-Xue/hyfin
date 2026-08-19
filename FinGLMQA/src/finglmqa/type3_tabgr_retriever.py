"""Corpus-scoped Type 3 table evidence and lexical retrieval.

The builder deliberately reconstructs Phase 3 matrices from ``cell_spans``
instead of trusting a flattened header.  Retrieval is performed only after an
explicit document prefilter and never reparses source Markdown/HTML.
"""

from __future__ import annotations

from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
from typing import Any, Iterable, Mapping, Sequence


TABGR_V2_BUILDER_VERSION = "type3-tabgr-table-evidence-v2"
TABGR_V2_RETRIEVER_VERSION = "type3-tabgr-lexical-retriever-v2"
TABGR_V2_TABLE_SCHEMA = "finglmqa.type3.tabgr.structured_table.v2"
TABGR_V2_ROW_SCHEMA = "finglmqa.type3.tabgr.table_row_evidence.v2"
TABGR_V2_AUTH_SCHEMA = "finglmqa.type3.tabgr.numeric_authorization.v1"
TABGR_RUNTIME_SHA256 = "7d193807d5f74b3281c8bd52c0d6da76f1f149cd5e92c4c82b47de4b8708d316"

_YEAR_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")
_NUMBER_RE = re.compile(r"(?<![A-Za-z])[-+]?\d[\d,，. ]*(?:%|％)?")
_ASCII_RE = re.compile(r"[A-Za-z][A-Za-z0-9_.:/-]*|\d{4}年?|\d+(?:\.\d+)?%?")
_CJK_RUN_RE = re.compile(r"[\u3400-\u9fff]+")
_HEADER_TERMS = (
    "项目", "类别", "名称", "类型", "指标", "本期", "上期", "期末", "期初",
    "本年", "上年", "年度", "年月", "日期", "金额", "比例", "单位", "合计",
    "账面", "原值", "净值", "数量", "占比", "变动", "原因", "内容",
)
_UNIT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("亿元", re.compile(r"亿元")),
    ("万元", re.compile(r"万元")),
    ("千元", re.compile(r"千元")),
    ("人民币元", re.compile(r"人民币\s*元")),
    ("美元", re.compile(r"美元")),
    ("港元", re.compile(r"港元")),
    ("元/股", re.compile(r"元\s*[/／]\s*股")),
    ("元", re.compile(r"(?<!万)(?<!千)(?<!亿)(?<!人民)(?<!美元)(?<!港)元")),
    ("%", re.compile(r"[%％]|百分比")),
    ("人", re.compile(r"单位\s*[:：]?\s*人(?:\b|次|数|员)?")),
    ("股", re.compile(r"单位\s*[:：]?\s*股")),
)


class Type3TabGRError(ValueError):
    """Raised when a table artifact violates a fail-closed contract."""


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def semantic_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize_text(value: object) -> str:
    return " ".join(str(value or "").split())


def lexical_tokens(value: object) -> tuple[str, ...]:
    """Return deterministic ASCII tokens and CJK uni/bi-grams."""

    text = normalize_text(value).lower()
    tokens: set[str] = {match.group(0) for match in _ASCII_RE.finditer(text)}
    for run in _CJK_RUN_RE.findall(text):
        if len(run) == 1:
            tokens.add(run)
        else:
            tokens.update(run[index : index + 2] for index in range(len(run) - 1))
            if len(run) <= 8:
                tokens.add(run)
    return tuple(sorted(token for token in tokens if token))


@dataclass(frozen=True)
class OriginCell:
    source_row: int
    source_col: int
    origin_row: int
    origin_col: int
    rowspan: int
    colspan: int
    tag: str
    text: str
    cell_hash: str

    def as_mapping(self) -> dict[str, Any]:
        return {
            "source_coordinate": [self.source_row, self.source_col],
            "origin_coordinate": [self.origin_row, self.origin_col],
            "rowspan": self.rowspan,
            "colspan": self.colspan,
            "tag": self.tag,
            "text": self.text,
            "cell_hash": self.cell_hash,
        }


@dataclass(frozen=True)
class ReconstructedGrid:
    matrix: tuple[tuple[str, ...], ...]
    origin_cells: tuple[OriginCell, ...]
    origin_by_coordinate: Mapping[tuple[int, int], OriginCell]
    parser_overwrite_count: int


def _positive_integer(value: object, path: str) -> int:
    if isinstance(value, bool):
        raise Type3TabGRError(f"{path} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise Type3TabGRError(f"{path} must be a positive integer") from exc
    if result < 1:
        raise Type3TabGRError(f"{path} must be a positive integer")
    return result


def reconstruct_origin_grid(
    cell_spans: object,
    expected_matrix: object,
    *,
    table_id: str,
) -> ReconstructedGrid:
    """Rebuild Phase 3's exact rowspan/colspan expansion and verify matrix.

    Phase 3's parser records ``source_col`` as the ordinal HTML cell, not the
    final grid column.  Its historical expansion can overwrite a carried empty
    rowspan when a later colspan begins before it; reproducing that behavior is
    required for a truthful matrix check and is recorded in the audit counter.
    """

    if not isinstance(cell_spans, list) or not isinstance(expected_matrix, list):
        raise Type3TabGRError("cell_spans and matrix must be arrays")
    rows: dict[int, list[tuple[int, int, Mapping[str, Any]]]] = defaultdict(list)
    for position, raw in enumerate(cell_spans):
        if not isinstance(raw, Mapping):
            raise Type3TabGRError(f"cell_spans[{position}] must be an object")
        source_row = int(raw.get("source_row", -1))
        source_col = int(raw.get("source_col", -1))
        if source_row < 0 or source_col < 0:
            raise Type3TabGRError("cell source coordinates must be nonnegative")
        rows[source_row].append((source_col, position, raw))
    if rows and sorted(rows) != list(range(max(rows) + 1)):
        raise Type3TabGRError("cell source rows are not contiguous")

    matrix: list[list[str]] = []
    origins: list[OriginCell] = []
    origin_map: dict[tuple[int, int], OriginCell] = {}
    pending: dict[int, tuple[int, str, OriginCell]] = {}
    overwrite_count = 0
    for source_row in range(max(rows, default=-1) + 1):
        row: list[str | None] = []
        next_pending: dict[int, tuple[int, str, OriginCell]] = {}
        col = 0

        def set_cell(column: int, value: str, origin: OriginCell) -> None:
            nonlocal overwrite_count
            while len(row) <= column:
                row.append(None)
            if row[column] is not None:
                overwrite_count += 1
            row[column] = value
            origin_map[(source_row, column)] = origin

        for source_col, _, raw in sorted(rows[source_row]):
            while col in pending:
                remaining, text, origin = pending[col]
                set_cell(col, text, origin)
                if remaining > 1:
                    next_pending[col] = (remaining - 1, text, origin)
                col += 1
            text = normalize_text(raw.get("text"))
            rowspan = _positive_integer(raw.get("rowspan", 1), "cell.rowspan")
            colspan = _positive_integer(raw.get("colspan", 1), "cell.colspan")
            unsigned = {
                "table_id": table_id,
                "source_coordinate": [source_row, source_col],
                "origin_coordinate": [source_row, col],
                "rowspan": rowspan,
                "colspan": colspan,
                "tag": normalize_text(raw.get("tag") or "td").lower(),
                "text": text,
            }
            origin = OriginCell(
                source_row=source_row,
                source_col=source_col,
                origin_row=source_row,
                origin_col=col,
                rowspan=rowspan,
                colspan=colspan,
                tag=unsigned["tag"],
                text=text,
                cell_hash=semantic_sha256(unsigned),
            )
            origins.append(origin)
            for span_col in range(col, col + colspan):
                set_cell(span_col, text, origin)
                if rowspan > 1:
                    next_pending[span_col] = (rowspan - 1, text, origin)
            col += colspan
        for pending_col in sorted(pending):
            if pending_col < col:
                continue
            remaining, text, origin = pending[pending_col]
            set_cell(pending_col, text, origin)
            if remaining > 1:
                next_pending[pending_col] = (remaining - 1, text, origin)
            col = pending_col + 1
        matrix.append(["" if value is None else value for value in row])
        pending = next_pending

    while pending:
        source_row = len(matrix)
        row: list[str | None] = []
        next_pending = {}
        for pending_col, (remaining, text, origin) in sorted(pending.items()):
            while len(row) <= pending_col:
                row.append(None)
            row[pending_col] = text
            origin_map[(source_row, pending_col)] = origin
            if remaining > 1:
                next_pending[pending_col] = (remaining - 1, text, origin)
        matrix.append(["" if value is None else value for value in row])
        pending = next_pending

    width = max((len(row) for row in matrix), default=0)
    matrix = [row + [""] * (width - len(row)) for row in matrix]
    expected: list[list[str]] = []
    for index, raw_row in enumerate(expected_matrix):
        if not isinstance(raw_row, list):
            raise Type3TabGRError(f"matrix[{index}] must be an array")
        expected.append([normalize_text(value) for value in raw_row])
    expected_width = max((len(row) for row in expected), default=0)
    expected = [row + [""] * (expected_width - len(row)) for row in expected]
    if matrix != expected:
        raise Type3TabGRError("reconstructed cell_spans do not equal Phase 3 matrix")
    return ReconstructedGrid(
        matrix=tuple(tuple(row) for row in matrix),
        origin_cells=tuple(origins),
        origin_by_coordinate=origin_map,
        parser_overwrite_count=overwrite_count,
    )


def _numeric_ratio(row: Sequence[str]) -> float:
    values = [value for value in row if value]
    return (sum(bool(_NUMBER_RE.fullmatch(value)) for value in values) / len(values)) if values else 0.0


def _header_like(row: Sequence[str]) -> bool:
    values = [value for value in row if value]
    if not values:
        return False
    joined = " ".join(values)
    keyword_count = sum(term in joined for term in _HEADER_TERMS)
    short_ratio = sum(len(value) <= 18 for value in values) / len(values)
    year_ratio = sum(bool(_YEAR_RE.search(value)) for value in values) / len(values)
    if year_ratio >= 0.5:
        return True
    if _numeric_ratio(row) > 0.34:
        return False
    return keyword_count > 0 or (not row[0] and short_ratio >= 0.75)


def infer_header_bands(grid: ReconstructedGrid) -> dict[str, Any]:
    matrix = [list(row) for row in grid.matrix]
    if not matrix:
        raise Type3TabGRError("cannot infer headers from an empty matrix")
    header_rows = [0]
    structural = any(
        cell.origin_row == 0 and (cell.colspan > 1 or cell.rowspan > 1 or cell.tag == "th")
        for cell in grid.origin_cells
    )
    for row_index in range(1, min(5, len(matrix))):
        carried = any(
            cell.origin_row < row_index < cell.origin_row + cell.rowspan
            and cell.origin_row in header_rows
            for cell in grid.origin_cells
        )
        grouped_previous = any(
            cell.origin_row == row_index - 1 and cell.colspan > 1 for cell in grid.origin_cells
        )
        if _header_like(matrix[row_index]) and (structural or carried or grouped_previous):
            header_rows.append(row_index)
            structural = True
        else:
            break

    reference_values = {normalize_text(value) for row in header_rows for value in matrix[row] if value}
    embedded: list[int] = []
    for row_index in range(header_rows[-1] + 1, len(matrix)):
        values = {normalize_text(value) for value in matrix[row_index] if value}
        if not values or not _header_like(matrix[row_index]):
            continue
        similarity = len(values & reference_values) / max(1, min(len(values), len(reference_values)))
        if similarity >= 0.6:
            embedded.append(row_index)
    return {
        "initial_header_rows": header_rows,
        "embedded_header_resets": embedded,
        "header_strategy": "active_multilevel_bands_v2",
    }


def flatten_headers(matrix: Sequence[Sequence[str]], header_rows: Sequence[int]) -> list[str]:
    width = max((len(row) for row in matrix), default=0)
    labels: list[str] = []
    for column in range(width):
        parts: list[str] = []
        for row_index in header_rows:
            if row_index >= len(matrix) or column >= len(matrix[row_index]):
                continue
            value = normalize_text(matrix[row_index][column])
            if value and (not parts or value != parts[-1]):
                parts.append(value)
        labels.append(" / ".join(parts) or f"第{column + 1}列")
    return labels


def infer_data_start_column(
    matrix: Sequence[Sequence[str]],
    *,
    header_rows: Sequence[int],
    embedded_resets: Sequence[int],
) -> int:
    width = max((len(row) for row in matrix), default=0)
    if width <= 1:
        return width
    excluded = set(header_rows) | set(embedded_resets)
    rows = [row for index, row in enumerate(matrix) if index not in excluded]
    labels = flatten_headers(matrix, header_rows)
    for column in range(1, width):
        values = [normalize_text(row[column]) for row in rows if column < len(row) and normalize_text(row[column])]
        numeric = sum(bool(_NUMBER_RE.fullmatch(value)) for value in values)
        if values and numeric / len(values) >= 0.2:
            return column
        if any(term in labels[column] for term in ("年", "期", "金额", "比例", "数量", "余额", "发生额")):
            return column
    return 1


def state(value: str | None, *, source: str | None = None, conflict: Sequence[str] = ()) -> dict[str, Any]:
    if conflict:
        return {"status": "conflict", "value": None, "source": source, "candidates": sorted(set(conflict))}
    if value:
        return {"status": "resolved", "value": value, "source": source, "candidates": [value]}
    return {"status": "unknown", "value": None, "source": None, "candidates": []}


def infer_unit_state(*values: object) -> dict[str, Any]:
    candidates: list[str] = []
    source_parts: list[str] = []
    for index, raw in enumerate(values):
        text = normalize_text(raw)
        if not text:
            continue
        found = [label for label, pattern in _UNIT_PATTERNS if pattern.search(text)]
        # Composite currency/share units are authoritative over generic
        # substring matches.  This avoids 人民币元->人+元 and 美元->美元+元.
        if "人民币元" in found:
            found = [label for label in found if label not in {"人", "元"}]
        if "美元" in found or "港元" in found or "元/股" in found:
            found = [label for label in found if label != "元"]
        if "元/股" in found:
            found = [label for label in found if label != "股"]
        if found:
            candidates.extend(found)
            source_parts.append(f"field_{index}")
    unique = sorted(set(candidates))
    if len(unique) == 1:
        return state(unique[0], source="+".join(source_parts))
    if len(unique) > 1:
        return state(None, source="+".join(source_parts), conflict=unique)
    return state(None)


def infer_period_state(column_header: str, report_year: int) -> dict[str, Any]:
    years = sorted(set(int(value) for value in _YEAR_RE.findall(column_header)))
    if len(years) == 1:
        return state(str(years[0]), source="column_header")
    if len(years) > 1:
        return state(None, source="column_header", conflict=[str(year) for year in years])
    if re.search(r"本期|本年|本年度|期末|年末", column_header):
        return state(str(report_year), source="relative_column_header")
    if re.search(r"上期|上年|上年度|期初|年初", column_header):
        return state(str(report_year - 1), source="relative_column_header")
    return state(None)


def infer_scope_state(*values: object) -> dict[str, Any]:
    text = " ".join(normalize_text(value) for value in values if normalize_text(value))
    candidates: list[str] = []
    if re.search(r"合并(?:口径|报表|资产负债表|利润表|现金流量表)?", text):
        candidates.append("consolidated")
    if re.search(r"母公司(?:口径|报表|资产负债表|利润表|现金流量表)?", text):
        candidates.append("parent_company")
    candidates = sorted(set(candidates))
    if len(candidates) == 1:
        return state(candidates[0], source="caption_section_or_header")
    if len(candidates) > 1:
        return state(None, source="caption_section_or_header", conflict=candidates)
    return state(None)


def numeric_fragments(value: str) -> list[str]:
    return [normalize_text(match.group(0)) for match in _NUMBER_RE.finditer(value)]


def redact_unauthorized(value: str, *, authorized: bool) -> str:
    if authorized or not numeric_fragments(value):
        return value
    return _NUMBER_RE.sub("[未经授权数值]", value)


def safe_numeric_projection(value: str, allowed_renderings: Iterable[str]) -> str:
    """Redact every numeric literal not exactly present in the authorization set."""

    allowed = {normalize_text(item) for item in allowed_renderings if normalize_text(item)}

    def replace(match: re.Match[str]) -> str:
        literal = normalize_text(match.group(0))
        return match.group(0) if literal in allowed else "[未经授权数值]"

    return _NUMBER_RE.sub(replace, value)


def build_fact_authorization(
    fact: Mapping[str, Any],
    *,
    corpus_id: str,
    raw_value: str,
    source_markdown: str,
    table_sha256: str,
    table_line_range: Sequence[int],
) -> dict[str, Any]:
    unsigned = {
        "schema_version": TABGR_V2_AUTH_SCHEMA,
        "corpus_id": corpus_id,
        "fact_id": str(fact["fact_id"]),
        "document_id": str(fact["document_id"]),
        "table_id": str(fact["source_table_id"]),
        "table_sha256": table_sha256,
        "source_markdown": source_markdown,
        "table_line_range": [int(value) for value in table_line_range],
        "cell_coordinate": [int(fact["source_row_index"]), int(fact["source_col_index"])],
        "raw_value": raw_value,
        "raw_value_sha256": sha256_text(raw_value),
        "canonical_metric": str(fact["canonical_metric"]),
        "metric_year": int(fact["metric_year"]),
        "normalized_value": str(fact["normalized_value"]),
        "normalized_unit": str(fact["normalized_unit"]),
        "selection_status": str(fact["selection_status"]),
        "source_candidate_id": str(fact["source_candidate_id"]),
        # The table route authorizes only the exact source rendering.  Unit
        # conversions remain the responsibility of Phase 8 fact/formula paths.
        "allowed_renderings": [raw_value],
    }
    return {**unsigned, "authorization_id": "t3tabgr-auth-" + semantic_sha256(unsigned)[:24]}


def _safe_relative_path(value: object) -> str:
    text = normalize_text(value)
    pure = PurePosixPath(text)
    if not text or pure.is_absolute() or ".." in pure.parts or pure.as_posix() != text:
        raise Type3TabGRError("index shard path is not a safe relative POSIX path")
    return text


@dataclass(frozen=True)
class Type3TabGRCandidate:
    evidence_id: str
    corpus_id: str
    document_id: str
    evidence_type: str
    heading_path: tuple[str, ...]
    display_text: str
    answer_safe_text: str
    source_markdown: str
    line_range: tuple[int, int]
    table_id: str
    numeric_authorizations: tuple[Mapping[str, Any], ...]
    unauthorized_numeric_values: tuple[str, ...]
    retrieval_channel: str
    retrieval_score: float
    row_path: tuple[str, ...]
    semantic_states: Mapping[str, Any]

    def as_mapping(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "corpus_id": self.corpus_id,
            "document_id": self.document_id,
            "evidence_type": self.evidence_type,
            "heading_path": list(self.heading_path),
            "display_text": self.display_text,
            "answer_safe_text": self.answer_safe_text,
            "source_markdown": self.source_markdown,
            "line_range": list(self.line_range),
            "table_id": self.table_id,
            "numeric_authorizations": [dict(value) for value in self.numeric_authorizations],
            "unauthorized_numeric_values": list(self.unauthorized_numeric_values),
            "retrieval_channel": self.retrieval_channel,
            "retrieval_score": format(self.retrieval_score, ".8f"),
            "row_path": list(self.row_path),
            "semantic_states": dict(self.semantic_states),
        }


class Type3TabGRRetriever:
    """A deterministic document-prefiltered lexical table-row retriever."""

    def __init__(self, index_dir: str | Path, *, expected_corpus_id: str | None = None) -> None:
        self.index_dir = Path(index_dir)
        manifest_path = self.index_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != "finglmqa.type3.tabgr.lexical_index_manifest.v2":
            raise Type3TabGRError("unsupported TabGR v2 index manifest")
        self.corpus_id = str(manifest.get("corpus_id") or "")
        if expected_corpus_id is not None and self.corpus_id != expected_corpus_id:
            raise Type3TabGRError("TabGR index corpus_id mismatch")
        if manifest.get("tabgr_runtime_sha256") != TABGR_RUNTIME_SHA256:
            raise Type3TabGRError("TabGR runtime hash pin mismatch")
        document_rows: list[Mapping[str, Any]] = []
        with (self.index_dir / "document_manifest.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    value = json.loads(line)
                    if not isinstance(value, Mapping):
                        raise Type3TabGRError("document manifest row must be an object")
                    document_rows.append(value)
        self._documents: dict[str, tuple[Path, str]] = {}
        for row in document_rows:
            document_id = str(row.get("document_id") or "")
            relative = _safe_relative_path(row.get("shard_path"))
            shard = self.index_dir / relative
            expected_hash = str(row.get("shard_sha256") or "")
            if not document_id or document_id in self._documents:
                raise Type3TabGRError("document manifest ids must be non-empty and unique")
            self._documents[document_id] = (shard, expected_hash)
        if len(self._documents) != int(manifest.get("document_count", -1)):
            raise Type3TabGRError("document manifest count mismatch")
        self._cache: OrderedDict[str, tuple[list[dict[str, Any]], dict[str, int]]] = OrderedDict()
        self._cache_limit = 6
        self._variant_cache: OrderedDict[
            tuple[str, str, tuple[str, ...]],
            tuple[
                tuple[dict[str, Any], ...],
                tuple[frozenset[str], ...],
                dict[str, int],
            ],
        ] = OrderedDict()
        self._variant_cache_limit = 12

    @property
    def document_ids(self) -> frozenset[str]:
        return frozenset(self._documents)

    def _load_document(self, document_id: str) -> tuple[list[dict[str, Any]], dict[str, int]]:
        cached = self._cache.get(document_id)
        if cached is not None:
            self._cache.move_to_end(document_id)
            return cached
        selected = self._documents.get(document_id)
        if selected is None:
            raise Type3TabGRError("document_id is outside the explicitly indexed corpus")
        shard, expected_hash = selected
        digest = hashlib.sha256()
        records: list[dict[str, Any]] = []
        document_frequency: Counter[str] = Counter()
        with shard.open("rb") as handle:
            for raw_line in handle:
                digest.update(raw_line)
                if not raw_line.strip():
                    raise Type3TabGRError("blank row in document shard")
                row = json.loads(raw_line)
                if row.get("document_id") != document_id or row.get("corpus_id") != self.corpus_id:
                    raise Type3TabGRError("cross-document or cross-corpus row in shard")
                records.append(row)
                for token in set(lexical_tokens(row.get("search_text"))):
                    document_frequency[token] += 1
        if digest.hexdigest() != expected_hash:
            raise Type3TabGRError("document shard hash mismatch")
        result = (records, dict(document_frequency))
        self._cache[document_id] = result
        self._cache.move_to_end(document_id)
        while len(self._cache) > self._cache_limit:
            evicted_document_id, _ = self._cache.popitem(last=False)
            for key in tuple(self._variant_cache):
                if key[0] == evicted_document_id:
                    del self._variant_cache[key]
        return result

    @staticmethod
    def _variant_text(record: Mapping[str, Any], ablations: frozenset[str]) -> str:
        if "legacy_flattened_a0" in ablations:
            return str(record.get("legacy_flat_text") or record.get("search_text") or "")
        parts = [
            str(record.get("base_search_text") or ""),
            str(record.get("value_search_text") or ""),
            str(record.get("own_row_header_text") or ""),
        ]
        if "no_multilevel_headers" in ablations:
            parts.append(str(record.get("single_header_text") or ""))
        else:
            parts.append(str(record.get("multilevel_header_text") or ""))
        if "no_hierarchical_rows" not in ablations:
            parts.append(str(record.get("hierarchical_row_text") or ""))
        if "no_unit_period_scope" not in ablations:
            parts.append(str(record.get("semantic_search_text") or ""))
        return " ".join(part for part in parts if part)

    @staticmethod
    def _variant_signature(ablations: frozenset[str]) -> tuple[str, ...]:
        """Return the minimal signature that can change lexical row text."""

        if "legacy_flattened_a0" in ablations:
            return ("legacy_flattened_a0",)
        return tuple(sorted(
            ablation
            for ablation in ablations
            if ablation in {
                "no_multilevel_headers",
                "no_hierarchical_rows",
                "no_unit_period_scope",
            }
        ))

    def _load_variant_tokens(
        self,
        document_id: str,
        records: Sequence[dict[str, Any]],
        *,
        record_type: str,
        ablations: frozenset[str],
    ) -> tuple[
        tuple[dict[str, Any], ...],
        tuple[frozenset[str], ...],
        dict[str, int],
    ]:
        """Cache exact variant tokens/DFs within the document LRU boundary."""

        key = (document_id, record_type, self._variant_signature(ablations))
        cached = self._variant_cache.get(key)
        if cached is not None:
            self._variant_cache.move_to_end(key)
            return cached
        eligible = tuple(row for row in records if row.get("record_type") == record_type)
        token_sets = tuple(
            frozenset(lexical_tokens(self._variant_text(row, ablations)))
            for row in eligible
        )
        document_frequency: Counter[str] = Counter()
        for terms in token_sets:
            document_frequency.update(terms)
        result = (eligible, token_sets, dict(document_frequency))
        self._variant_cache[key] = result
        self._variant_cache.move_to_end(key)
        while len(self._variant_cache) > self._variant_cache_limit:
            self._variant_cache.popitem(last=False)
        return result

    def retrieve(
        self,
        query: str,
        *,
        document_id: str,
        top_k: int = 12,
        ablations: Iterable[str] = (),
        ppr_scores: Mapping[str, float] | None = None,
        ppr_source_sha256: str | None = None,
        ppr_binding_sha256: str | None = None,
    ) -> list[Type3TabGRCandidate]:
        """Retrieve within exactly one document.

        Optional PPR scores are accepted only with the pinned TabGR source hash;
        callers can therefore run query-specific PPR on a lexical shortlist
        without allowing unpinned runtime behavior into fusion.
        """

        if not isinstance(query, str) or not query.strip():
            raise Type3TabGRError("query must be a non-empty string")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 100:
            raise Type3TabGRError("top_k must be between 1 and 100")
        ablation_set = frozenset(ablations)
        allowed = {
            "legacy_flattened_a0", "no_multilevel_headers", "no_hierarchical_rows",
            "no_unit_period_scope", "no_tabgr_ppr", "no_fact_join", "table_level",
        }
        unknown = ablation_set - allowed
        if unknown:
            raise Type3TabGRError(f"unknown ablations: {sorted(unknown)!r}")
        if ppr_scores is not None:
            if ppr_source_sha256 != TABGR_RUNTIME_SHA256:
                raise Type3TabGRError("PPR shortlist is not bound to the pinned TabGR runtime")
            expected_ppr_binding = semantic_sha256({
                "runtime_sha256": ppr_source_sha256,
                "corpus_id": self.corpus_id,
                "document_id": document_id,
                "query_sha256": sha256_text(normalize_text(query)),
                "scores": [
                    [str(key), format(float(value), ".8f")]
                    for key, value in sorted(ppr_scores.items())
                ],
            })
            if ppr_binding_sha256 != expected_ppr_binding:
                raise Type3TabGRError("PPR scores lack a query/candidate-set binding")

        # This lookup is the document prefilter.  No shard is opened before it.
        records, _ = self._load_document(document_id)
        query_terms = Counter(lexical_tokens(query))
        if not query_terms:
            return []
        record_type = "table" if "table_level" in ablation_set else "table_row"
        eligible, token_sets, dfs = self._load_variant_tokens(
            document_id,
            records,
            record_type=record_type,
            ablations=ablation_set,
        )
        total = max(1, len(eligible))
        scored: list[tuple[float, str, Mapping[str, Any]]] = []
        for row, terms in zip(eligible, token_sets):
            score = 0.0
            for term, query_count in query_terms.items():
                if term not in terms:
                    continue
                idf = math.log(1.0 + (total - dfs[term] + 0.5) / (dfs[term] + 0.5))
                score += idf * query_count
            evidence_id = str(row["evidence_id"])
            if ppr_scores is not None and "no_tabgr_ppr" not in ablation_set:
                score += 0.25 * max(0.0, float(ppr_scores.get(evidence_id, 0.0)))
            if score > 0:
                scored.append((score, evidence_id, row))
        scored.sort(key=lambda item: (-item[0], item[1]))

        candidates: list[Type3TabGRCandidate] = []
        for score, _, row in scored[:top_k]:
            no_fact_join = "no_fact_join" in ablation_set
            authorizations = () if no_fact_join else tuple(row.get("numeric_authorizations") or ())
            display_text = str(row.get("display_text") or "")
            if no_fact_join:
                answer_safe_text = redact_unauthorized(display_text, authorized=False)
                unauthorized_values = tuple(numeric_fragments(display_text))
            else:
                answer_safe_text = str(row.get("answer_safe_text") or "")
                unauthorized_values = tuple(
                    str(value) for value in row.get("unauthorized_numeric_values") or ()
                )
            candidates.append(Type3TabGRCandidate(
                evidence_id=str(row["evidence_id"]),
                corpus_id=self.corpus_id,
                document_id=document_id,
                evidence_type="table_row" if record_type == "table_row" else "table",
                heading_path=tuple(str(value) for value in row.get("heading_path") or ()),
                display_text=display_text,
                answer_safe_text=answer_safe_text,
                source_markdown=str(row.get("source_markdown") or ""),
                line_range=tuple(int(value) for value in row["table_line_range"]),
                table_id=str(row["table_id"]),
                numeric_authorizations=authorizations,
                unauthorized_numeric_values=unauthorized_values,
                retrieval_channel=(
                    "tabgr_ppr_lexical_v2"
                    if ppr_scores is not None and "no_tabgr_ppr" not in ablation_set
                    else "tabgr_lexical_v2"
                ),
                retrieval_score=score,
                row_path=tuple(str(value) for value in row.get("row_path") or ()),
                semantic_states=dict(row.get("semantic_states") or {}),
            ))
        return candidates


__all__ = [
    "OriginCell", "ReconstructedGrid", "TABGR_RUNTIME_SHA256",
    "TABGR_V2_AUTH_SCHEMA", "TABGR_V2_BUILDER_VERSION", "TABGR_V2_ROW_SCHEMA",
    "TABGR_V2_TABLE_SCHEMA", "Type3TabGRCandidate", "Type3TabGRError",
    "Type3TabGRRetriever", "build_fact_authorization", "canonical_json_bytes",
    "flatten_headers", "infer_data_start_column", "infer_header_bands",
    "infer_period_state", "infer_scope_state", "infer_unit_state", "lexical_tokens",
    "normalize_text", "numeric_fragments", "reconstruct_origin_grid",
    "redact_unauthorized", "semantic_sha256", "sha256_text", "state",
    "safe_numeric_projection",
]
