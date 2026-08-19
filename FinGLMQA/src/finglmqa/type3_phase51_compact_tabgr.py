"""Fail-closed Compact TabGR cell-fact projection for Type 3 Phase 5.1.

This module consumes only a sanitized question, Phase 4 table evidence, and
the exact rich row hydrated from the frozen TabGR document shard.  It never
renders a row-wide ``answer_safe_text`` value.  Final claims are composed only
from exact row labels plus one or two individually authorized cells.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping, Sequence

from finglmqa.type3_tabgr_retriever import (
    TABGR_RUNTIME_SHA256,
    TABGR_V2_AUTH_SCHEMA,
    TABGR_V2_BUILDER_VERSION,
    TABGR_V2_ROW_SCHEMA,
    canonical_json_bytes,
    lexical_tokens,
    infer_unit_state,
    normalize_text,
    numeric_fragments,
    semantic_sha256,
    sha256_text,
)


PROFILE_VERSION = "compact-tabgr-cell-fact-v1"
COMPOSER_VERSION = "type3-phase51-compact-tabgr-v1"
TRACE_SCHEMA = "finglmqa.type3.phase51.compact_tabgr_trace.v1"
MASK = "[未经授权数值]"
MAX_CLAIMS = 2
MAX_CELLS_PER_CLAIM = 2
MAX_CLAIM_CHARACTERS = 160
MAX_TOTAL_CHARACTERS = 260

_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_YEAR_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")
_DIRECT_TERMS = (
    "多少", "金额", "数量", "比例", "占比", "余额", "净额", "收入", "利润",
    "资产", "负债", "现金流量", "研发投入", "人数", "财务指标", "会计数据",
)
_COMPARISON_TERMS = (
    "同比", "较上年", "增长", "下降", "增加", "减少", "分别", "对比", "比较",
)
_BLOCKED_TERMS = (
    "原因", "为何", "为什么", "措施", "风险", "战略", "规划", "整改",
    "履行情况", "是否存在", "有无",
)
_EXPLICIT_VALUE_ASK_TERMS = (
    "多少", "金额", "数量", "比例", "占比", "幅度", "分别",
)
_AUTH_FIELDS = frozenset(
    {
        "schema_version",
        "corpus_id",
        "fact_id",
        "document_id",
        "table_id",
        "table_sha256",
        "source_markdown",
        "table_line_range",
        "cell_coordinate",
        "raw_value",
        "raw_value_sha256",
        "canonical_metric",
        "metric_year",
        "normalized_value",
        "normalized_unit",
        "selection_status",
        "source_candidate_id",
        "allowed_renderings",
        "authorization_id",
    }
)
_ROW_PROJECTION_FIELDS = (
    "schema_version",
    "builder_version",
    "record_type",
    "evidence_id",
    "corpus_id",
    "document_id",
    "table_id",
    "table_index",
    "row_index",
    "heading_path",
    "active_header_rows",
    "flattened_column_headers",
    "row_path",
    "cells",
    "source_markdown",
    "table_line_range",
    "table_sha256",
    "numeric_authorizations",
    "semantic_states",
)


class CompactTabGRError(ValueError):
    """Raised when a Compact TabGR input violates the frozen contract."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _safe_relative_path(value: object) -> str:
    text = normalize_text(value)
    pure = PurePosixPath(text)
    if not text or pure.is_absolute() or ".." in pure.parts or pure.as_posix() != text:
        raise CompactTabGRError("TabGR shard path is not a safe relative POSIX path")
    return text


def _load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CompactTabGRError(f"expected JSON object: {path}")
    return value


def hydrate_candidate(
    index_dir: str | Path,
    *,
    candidate_id: str,
    corpus_id: str,
    document_id: str,
) -> dict[str, Any]:
    """Hydrate one exact row by evidence id from its bound document shard.

    The helper validates the index and document-manifest hash chain, opens only
    the requested document shard, verifies the complete shard hash, and returns
    a deliberately narrow rich-row projection.  Row-wide display/answer text is
    not projected.
    """

    if not candidate_id or not corpus_id or not document_id:
        raise CompactTabGRError("candidate/corpus/document identity must be non-empty")
    root = Path(index_dir).resolve()
    manifest_path = root / "manifest.json"
    document_manifest_path = root / "document_manifest.jsonl"
    manifest = _load_json_object(manifest_path)
    if (
        manifest.get("schema_version")
        != "finglmqa.type3.tabgr.lexical_index_manifest.v2"
        or manifest.get("builder_version") != TABGR_V2_BUILDER_VERSION
        or manifest.get("tabgr_runtime_sha256") != TABGR_RUNTIME_SHA256
        or manifest.get("corpus_id") != corpus_id
        or manifest.get("document_prefilter_required") is not True
        or manifest.get("online_source_table_reparse_allowed") is not False
    ):
        raise CompactTabGRError("TabGR index manifest contract differs")
    if (
        manifest.get("document_manifest_sha256")
        != _sha256_file(document_manifest_path)
    ):
        raise CompactTabGRError("TabGR document manifest hash differs")

    selected: dict[str, Any] | None = None
    seen_documents: set[str] = set()
    with document_manifest_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise CompactTabGRError(
                    f"blank TabGR document manifest row at {line_number}"
                )
            row = json.loads(line)
            if not isinstance(row, dict):
                raise CompactTabGRError("TabGR document manifest row is not an object")
            row_document_id = str(row.get("document_id") or "")
            if not row_document_id or row_document_id in seen_documents:
                raise CompactTabGRError("TabGR document manifest ids are invalid")
            seen_documents.add(row_document_id)
            if row_document_id == document_id:
                selected = row
    if len(seen_documents) != int(manifest.get("document_count", -1)):
        raise CompactTabGRError("TabGR document manifest count differs")
    if selected is None:
        raise CompactTabGRError("document_id is outside the frozen TabGR index")
    if selected.get("schema_version") != "finglmqa.type3.tabgr.document_shard.v2":
        raise CompactTabGRError("TabGR document shard manifest schema differs")

    relative = _safe_relative_path(selected.get("shard_path"))
    shard_path = (root / relative).resolve()
    if not shard_path.is_relative_to(root):
        raise CompactTabGRError("TabGR shard escapes the frozen index directory")
    expected_shard_hash = str(selected.get("shard_sha256") or "")
    if not _HEX64_RE.fullmatch(expected_shard_hash):
        raise CompactTabGRError("TabGR shard manifest hash is invalid")

    digest = hashlib.sha256()
    found: list[dict[str, Any]] = []
    record_count = 0
    with shard_path.open("rb") as handle:
        for raw_line in handle:
            digest.update(raw_line)
            if not raw_line.strip():
                raise CompactTabGRError("blank record in TabGR document shard")
            row = json.loads(raw_line)
            if not isinstance(row, dict):
                raise CompactTabGRError("TabGR document shard row is not an object")
            record_count += 1
            if row.get("corpus_id") != corpus_id or row.get("document_id") != document_id:
                raise CompactTabGRError("cross-corpus/document row in TabGR shard")
            if row.get("evidence_id") == candidate_id:
                found.append(row)
    if digest.hexdigest() != expected_shard_hash:
        raise CompactTabGRError("TabGR document shard hash differs")
    if record_count != int(selected.get("record_count", -1)):
        raise CompactTabGRError("TabGR document shard record count differs")
    if len(found) != 1:
        raise CompactTabGRError("candidate_id did not hydrate to exactly one row")
    rich_row = found[0]
    if (
        rich_row.get("schema_version") != TABGR_V2_ROW_SCHEMA
        or rich_row.get("builder_version") != TABGR_V2_BUILDER_VERSION
        or rich_row.get("record_type") != "table_row"
    ):
        raise CompactTabGRError("candidate_id does not identify a rich table row")
    projection = {key: rich_row.get(key) for key in _ROW_PROJECTION_FIELDS}
    projection_hash = semantic_sha256(projection)
    return {
        "row": projection,
        "hydration": {
            "candidate_id": candidate_id,
            "document_shard_path": relative,
            "document_shard_sha256": expected_shard_hash,
            "rich_row_projection_sha256": projection_hash,
        },
    }


def question_route(question: str) -> str:
    """Classify table intent using only the sanitized question text."""

    if not isinstance(question, str) or not question.strip():
        raise CompactTabGRError("question must be non-empty")
    has_direct = any(term in question for term in _DIRECT_TERMS)
    has_comparison = any(term in question for term in _COMPARISON_TERMS)
    has_block = any(term in question for term in _BLOCKED_TERMS)
    has_explicit_value_ask = any(
        term in question for term in _EXPLICIT_VALUE_ASK_TERMS
    )
    if has_block and not has_explicit_value_ask:
        return "table_blocked"
    if has_block:
        return "table_optional"
    if has_comparison:
        return "table_comparison"
    if has_direct:
        return "table_direct"
    return "table_blocked"


def _semantic_tokens(value: object) -> frozenset[str]:
    return frozenset(
        token for token in lexical_tokens(value)
        if not token.isdigit() and not _YEAR_RE.fullmatch(token.rstrip("年"))
    )


def _overlap(question: str, *values: object) -> tuple[int, float, float]:
    question_terms = _semantic_tokens(question)
    value_terms = frozenset().union(*(_semantic_tokens(value) for value in values))
    common = len(question_terms.intersection(value_terms))
    return (
        common,
        common / max(1, len(value_terms)),
        common / max(1, len(question_terms)),
    )


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


def _resolved_state(value: object, *, allow_unknown: bool = False) -> str | None:
    if not isinstance(value, Mapping):
        raise CompactTabGRError("cell semantic state is not an object")
    status = value.get("status")
    if status == "unknown" and allow_unknown:
        return None
    if status != "resolved" or not isinstance(value.get("value"), str):
        raise CompactTabGRError("cell semantic state is not resolved")
    return str(value["value"])


def _cell_coordinate(value: object, *, field: str) -> tuple[int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or not all(
            isinstance(item, int) and not isinstance(item, bool) and item >= 0
            for item in value
        )
    ):
        raise CompactTabGRError(f"{field} is not a valid cell coordinate")
    return int(value[0]), int(value[1])


def _validated_column_headers(row: Mapping[str, Any]) -> tuple[str, ...]:
    headers = row.get("flattened_column_headers")
    row_index = row.get("row_index")
    cells = row.get("cells")
    if (
        not isinstance(headers, list)
        or not headers
        or any(not isinstance(value, str) or not normalize_text(value) for value in headers)
        or not isinstance(row_index, int)
        or isinstance(row_index, bool)
        or row_index < 0
        or not isinstance(cells, list)
        or not cells
    ):
        raise CompactTabGRError("rich row header/cell structure is invalid")
    normalized_headers = tuple(normalize_text(value) for value in headers)
    seen_coordinates: set[tuple[int, int]] = set()
    for cell in cells:
        if not isinstance(cell, Mapping):
            raise CompactTabGRError("rich row cell is not an object")
        coordinate = _cell_coordinate(cell.get("coordinate"), field="cell.coordinate")
        if coordinate in seen_coordinates or coordinate[0] != row_index:
            raise CompactTabGRError("rich row cell coordinate/row binding differs")
        seen_coordinates.add(coordinate)
        column = coordinate[1]
        if column >= len(normalized_headers):
            raise CompactTabGRError("rich row cell column is outside flattened headers")
        if normalize_text(cell.get("column_header")) != normalized_headers[column]:
            raise CompactTabGRError("cell column_header differs from flattened header")
    return normalized_headers


@dataclass(frozen=True)
class RowLabelCell:
    coordinate: tuple[int, int]
    origin_coordinate: tuple[int, int]
    origin_cell_hash: str
    raw_value: str
    raw_value_sha256: str

    def as_mapping(self) -> dict[str, Any]:
        return {
            "coordinate": list(self.coordinate),
            "origin_coordinate": list(self.origin_coordinate),
            "origin_cell_hash": self.origin_cell_hash,
            "raw_value": self.raw_value,
            "raw_value_sha256": self.raw_value_sha256,
        }


def _exact_row_label(row: Mapping[str, Any]) -> RowLabelCell:
    row_path = row.get("row_path")
    if not isinstance(row_path, list):
        raise CompactTabGRError("rich row row_path is not an array")
    labels = [normalize_text(value) for value in row_path if normalize_text(value)]
    if not labels:
        raise CompactTabGRError("rich row lacks an exact source row label")
    label = labels[-1]
    if (
        MASK in label
        or numeric_fragments(label)
        or re.fullmatch(r"第\s*\d+\s*列", label)
    ):
        raise CompactTabGRError("row label would introduce an unauthorized number")
    row_index = int(row["row_index"])
    matches: dict[tuple[Any, ...], RowLabelCell] = {}
    for cell in row["cells"]:
        if (
            not isinstance(cell, Mapping)
            or cell.get("numeric_status") != "not_numeric"
            or normalize_text(cell.get("raw_value")) != label
        ):
            continue
        coordinate = _cell_coordinate(cell.get("coordinate"), field="label.coordinate")
        origin_coordinate = _cell_coordinate(
            cell.get("origin_coordinate"),
            field="label.origin_coordinate",
        )
        raw_value = str(cell.get("raw_value") or "")
        raw_hash = str(cell.get("raw_value_sha256") or "")
        origin_hash = str(cell.get("origin_cell_hash") or "")
        if (
            coordinate[0] != row_index
            or origin_coordinate[0] > coordinate[0]
            or origin_coordinate[1] > coordinate[1]
            or raw_hash != sha256_text(raw_value)
            or not _HEX64_RE.fullmatch(origin_hash)
        ):
            raise CompactTabGRError("row label cell provenance is invalid")
        provenance = RowLabelCell(
            coordinate=coordinate,
            origin_coordinate=origin_coordinate,
            origin_cell_hash=origin_hash,
            raw_value=raw_value,
            raw_value_sha256=raw_hash,
        )
        matches[
            (
                provenance.origin_coordinate,
                provenance.origin_cell_hash,
                provenance.raw_value_sha256,
            )
        ] = provenance
    if len(matches) != 1:
        raise CompactTabGRError(
            "terminal row_path does not bind one exact nonnumeric source cell"
        )
    return next(iter(matches.values()))


def _evidence_binding(
    evidence: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    corpus_id: str,
    document_id: str,
) -> None:
    if evidence.get("route") != "table" or evidence.get("evidence_type") != "table_row":
        raise CompactTabGRError("Phase 4 candidate is not table-row evidence")
    exact_pairs = (
        ("candidate_id", "evidence_id"),
        ("corpus_id", "corpus_id"),
        ("document_id", "document_id"),
        ("source_markdown", "source_markdown"),
        ("table_id", "table_id"),
        ("table_sha256", "table_sha256"),
    )
    for evidence_key, row_key in exact_pairs:
        if evidence.get(evidence_key) != row.get(row_key):
            raise CompactTabGRError(f"candidate binding differs at {evidence_key}")
    if evidence.get("corpus_id") != corpus_id or evidence.get("document_id") != document_id:
        raise CompactTabGRError("candidate crosses the requested corpus/document")
    if (
        row.get("schema_version") != TABGR_V2_ROW_SCHEMA
        or row.get("builder_version") != TABGR_V2_BUILDER_VERSION
        or row.get("record_type") != "table_row"
    ):
        raise CompactTabGRError("hydrated row schema/version differs")
    if list(evidence.get("line_range") or ()) != list(row.get("table_line_range") or ()):
        raise CompactTabGRError("candidate table line range differs")
    if not _HEX64_RE.fullmatch(str(evidence.get("source_sha256") or "")):
        raise CompactTabGRError("candidate source hash is incomplete")
    cells = row.get("cells")
    if not isinstance(cells, list) or not cells:
        raise CompactTabGRError("hydrated row lacks cells")
    coordinates = [cell.get("coordinate") for cell in cells if isinstance(cell, Mapping)]
    origin_hashes = [
        cell.get("origin_cell_hash") for cell in cells if isinstance(cell, Mapping)
    ]
    if list(evidence.get("cell_coordinates") or ()) != coordinates:
        raise CompactTabGRError("candidate cell-coordinate binding differs")
    if list(evidence.get("origin_cell_hashes") or ()) != origin_hashes:
        raise CompactTabGRError("candidate origin-cell hash binding differs")
    if list(evidence.get("row_path") or ()) != list(row.get("row_path") or ()):
        raise CompactTabGRError("candidate row-path binding differs")
    if list(evidence.get("heading_path") or ()) != list(
        row.get("heading_path") or ()
    ):
        raise CompactTabGRError("candidate heading-path binding differs")
    if list(evidence.get("numeric_authorizations") or ()) != list(
        row.get("numeric_authorizations") or ()
    ):
        raise CompactTabGRError("candidate NumericAuthorization binding differs")


@dataclass(frozen=True)
class AuthorizedCell:
    candidate_id: str
    row_label: str
    row_label_cell: RowLabelCell
    row_index: int
    table_index: int
    coordinate: tuple[int, int]
    origin_coordinate: tuple[int, int]
    origin_cell_hash: str
    column_header: str
    raw_value: str
    raw_value_sha256: str
    normalized_value: str
    unit: str
    period: int
    scope: str | None
    authorization: Mapping[str, Any]
    evidence: Mapping[str, Any]
    row: Mapping[str, Any]
    projection_sha256: str

    def semantic_conflict_key(self) -> tuple[Any, ...]:
        return (
            normalize_text(self.row_label),
            self.period,
            normalize_text(self.unit),
            self.scope,
        )


def _authorized_cells(
    evidence: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    corpus_id: str,
    document_id: str,
    projection_sha256: str,
) -> tuple[AuthorizedCell, ...]:
    _evidence_binding(
        evidence,
        row,
        corpus_id=corpus_id,
        document_id=document_id,
    )
    flattened_headers = _validated_column_headers(row)
    label_cell = _exact_row_label(row)
    label = normalize_text(label_cell.raw_value)
    semantic_states = row.get("semantic_states")
    if not isinstance(semantic_states, Mapping):
        raise CompactTabGRError("rich row semantic_states is not an object")
    raw_authorizations = row.get("numeric_authorizations")
    if not isinstance(raw_authorizations, list):
        raise CompactTabGRError("row numeric authorizations are not an array")
    by_id: dict[str, Mapping[str, Any]] = {}
    for authorization in raw_authorizations:
        if not isinstance(authorization, Mapping) or set(authorization) != _AUTH_FIELDS:
            raise CompactTabGRError("NumericAuthorization shape differs")
        authorization_id = str(authorization.get("authorization_id") or "")
        if not authorization_id or authorization_id in by_id:
            raise CompactTabGRError("NumericAuthorization ids are invalid")
        by_id[authorization_id] = authorization

    result: list[AuthorizedCell] = []
    for cell in row["cells"]:
        if not isinstance(cell, Mapping) or cell.get("numeric_status") != "authorized":
            continue
        authorization_ids = cell.get("authorization_ids")
        if not isinstance(authorization_ids, list) or len(authorization_ids) != 1:
            raise CompactTabGRError("authorized cell must bind exactly one authorization")
        authorization = by_id.get(str(authorization_ids[0]))
        if authorization is None:
            raise CompactTabGRError("authorized cell references a missing authorization")
        coordinate_tuple = _cell_coordinate(
            cell.get("coordinate"),
            field="authorized_cell.coordinate",
        )
        origin_coordinate_tuple = _cell_coordinate(
            cell.get("origin_coordinate"),
            field="authorized_cell.origin_coordinate",
        )
        coordinate = list(coordinate_tuple)
        origin_coordinate = list(origin_coordinate_tuple)
        column = coordinate_tuple[1]
        if (
            coordinate_tuple[0] != int(row["row_index"])
            or column >= len(flattened_headers)
            or normalize_text(cell.get("column_header"))
            != flattened_headers[column]
        ):
            raise CompactTabGRError("authorized cell header/coordinate binding differs")
        raw_value = str(cell.get("raw_value") or "")
        raw_hash = str(cell.get("raw_value_sha256") or "")
        origin_hash = str(cell.get("origin_cell_hash") or "")
        unit = _resolved_state(cell.get("unit"))
        period_text = _resolved_state(cell.get("period"))
        scope = _resolved_state(cell.get("accounting_scope"), allow_unknown=True)
        for state_field, cell_state in (
            ("unit_by_column", cell.get("unit")),
            ("period_by_column", cell.get("period")),
            ("accounting_scope_by_column", cell.get("accounting_scope")),
        ):
            by_column = semantic_states.get(state_field)
            if (
                not isinstance(by_column, Mapping)
                or by_column.get(str(column)) != cell_state
            ):
                raise CompactTabGRError(
                    f"authorized cell {state_field} binding differs"
                )
        header = flattened_headers[column]
        header_years = sorted({int(value) for value in _YEAR_RE.findall(header)})
        if header_years and (
            len(header_years) != 1
            or header_years[0] != int(period_text)
        ):
            raise CompactTabGRError("authorized cell header/period binding differs")
        header_unit = infer_unit_state(header)
        if (
            header_unit.get("status") == "conflict"
            or (
                header_unit.get("status") == "resolved"
                and header_unit.get("value") != unit
            )
        ):
            raise CompactTabGRError("authorized cell header/unit binding differs")
        if (
            not raw_value
            or raw_hash != sha256_text(raw_value)
            or not _HEX64_RE.fullmatch(origin_hash)
            or normalize_text(raw_value) not in numeric_fragments(raw_value)
        ):
            raise CompactTabGRError("authorized cell value/hash is invalid")
        expected = {
            "schema_version": TABGR_V2_AUTH_SCHEMA,
            "corpus_id": corpus_id,
            "document_id": document_id,
            "table_id": row["table_id"],
            "table_sha256": row["table_sha256"],
            "source_markdown": row["source_markdown"],
            "table_line_range": list(row["table_line_range"]),
            "cell_coordinate": coordinate,
            "raw_value": raw_value,
            "raw_value_sha256": raw_hash,
        }
        for key, expected_value in expected.items():
            if authorization.get(key) != expected_value:
                raise CompactTabGRError(
                    f"NumericAuthorization field binding differs at {key}"
                )
        if (
            authorization.get("selection_status")
            not in {"selected_single_value", "resolved_by_confidence"}
            or authorization.get("allowed_renderings") != [raw_value]
            or authorization.get("normalized_unit") != unit
            or not isinstance(authorization.get("metric_year"), int)
            or str(authorization["metric_year"]) != period_text
            or not normalize_text(authorization.get("canonical_metric"))
            or not normalize_text(authorization.get("normalized_value"))
            or not normalize_text(authorization.get("source_candidate_id"))
        ):
            raise CompactTabGRError("NumericAuthorization semantic binding differs")
        result.append(
            AuthorizedCell(
                candidate_id=str(evidence["candidate_id"]),
                row_label=label,
                row_label_cell=label_cell,
                row_index=int(row["row_index"]),
                table_index=int(row["table_index"]),
                coordinate=coordinate_tuple,
                origin_coordinate=origin_coordinate_tuple,
                origin_cell_hash=origin_hash,
                column_header=normalize_text(cell.get("column_header")),
                raw_value=raw_value,
                raw_value_sha256=raw_hash,
                normalized_value=str(authorization["normalized_value"]),
                unit=str(unit),
                period=int(authorization["metric_year"]),
                scope=scope,
                authorization=dict(authorization),
                evidence=evidence,
                row=row,
                projection_sha256=projection_sha256,
            )
        )
    return tuple(result)


def _question_years(question: str) -> tuple[int, ...]:
    return tuple(sorted({int(value) for value in _YEAR_RE.findall(question)}))


def _scope_required(question: str) -> str | None:
    consolidated = "合并" in question
    parent = "母公司" in question
    if consolidated and parent:
        return "conflict"
    if consolidated:
        return "consolidated"
    if parent:
        return "parent_company"
    return None


def _period_phrase(cell: AuthorizedCell, question: str) -> str:
    header = cell.column_header
    if any(term in header for term in ("本期", "本年", "本年度", "期末", "年末")):
        return "本期"
    if any(term in header for term in ("上期", "上年", "上年度", "期初", "年初")):
        return "上期"
    years = _question_years(question)
    if years and cell.period == max(years):
        return "本期"
    if years and cell.period == max(years) - 1:
        return "上期"
    return ""


def _render_value(cell: AuthorizedCell) -> str:
    raw = cell.raw_value
    if cell.unit == "%" and raw.endswith(("%", "％")):
        return raw
    if cell.unit and raw.endswith(cell.unit):
        return raw
    return raw + cell.unit


@dataclass(frozen=True)
class CompactClaim:
    text: str
    cells: tuple[AuthorizedCell, ...]
    metric_overlap_count: int
    metric_precision: float
    question_recall: float
    period_match: int
    unit_match: int
    scope_match: int
    rank_signal: float

    def source_key(self) -> tuple[Any, ...]:
        first = self.cells[0]
        return (
            str(first.row["table_sha256"]),
            first.row_index,
            tuple(cell.coordinate for cell in self.cells),
        )

    def semantic_key(self) -> tuple[Any, ...]:
        first = self.cells[0]
        return (
            normalize_text(first.row_label),
            tuple(cell.period for cell in self.cells),
            normalize_text(first.unit),
            first.scope,
            tuple(cell.normalized_value for cell in self.cells),
        )

    def rank_key(self) -> tuple[Any, ...]:
        return (
            -self.metric_overlap_count,
            -self.metric_precision,
            -self.question_recall,
            -self.period_match,
            -self.unit_match,
            -self.scope_match,
            -self.rank_signal,
            len(self.text),
            self.cells[0].candidate_id,
            tuple(cell.coordinate for cell in self.cells),
        )

    def as_mapping(self) -> dict[str, Any]:
        first = self.cells[0]
        row_label_cell = first.row_label_cell.as_mapping()
        selected_cells = [
            {
                "coordinate": list(cell.coordinate),
                "origin_coordinate": list(cell.origin_coordinate),
                "origin_cell_hash": cell.origin_cell_hash,
                "raw_value": cell.raw_value,
                "raw_value_sha256": cell.raw_value_sha256,
                "column_header": cell.column_header,
                "unit": cell.unit,
                "period": str(cell.period),
                "accounting_scope": cell.scope,
                "numeric_authorization": dict(cell.authorization),
            }
            for cell in self.cells
        ]
        return {
            "claim_kind": (
                "compact_tabgr_comparison" if len(self.cells) == 2
                else "compact_tabgr_single_value"
            ),
            "text": self.text,
            "candidate_id": first.candidate_id,
            "corpus_id": str(first.row["corpus_id"]),
            "document_id": str(first.row["document_id"]),
            "source_markdown": str(first.row["source_markdown"]),
            "source_sha256": str(first.evidence["source_sha256"]),
            "table_id": str(first.row["table_id"]),
            "table_index": first.table_index,
            "table_sha256": str(first.row["table_sha256"]),
            "table_line_range": list(first.row["table_line_range"]),
            "heading_path": list(first.row.get("heading_path") or ()),
            "row_index": first.row_index,
            "row_path": list(first.row.get("row_path") or ()),
            "row_label_cell": row_label_cell,
            "selected_cells": selected_cells,
            "hydrated_projection_sha256": first.projection_sha256,
            "claim_sha256": semantic_sha256(
                {
                    "text": self.text,
                    "candidate_id": first.candidate_id,
                    "row_label_cell": row_label_cell,
                    "selected_cells": selected_cells,
                }
            ),
        }


def _validate_claim_text(claim: CompactClaim) -> None:
    if not claim.text or len(claim.text) > MAX_CLAIM_CHARACTERS:
        raise CompactTabGRError("Compact TabGR claim length differs")
    if (
        MASK in claim.text
        or not 1 <= len(claim.cells) <= MAX_CELLS_PER_CLAIM
    ):
        raise CompactTabGRError("Compact TabGR claim is masked or over-wide")
    allowed = {
        rendering
        for cell in claim.cells
        for rendering in cell.authorization["allowed_renderings"]
    }
    unsupported = [
        value for value in numeric_fragments(claim.text) if value not in allowed
    ]
    if unsupported:
        raise CompactTabGRError("claim introduces an unauthorized numeric literal")


def _claim_score(
    question: str,
    cells: tuple[AuthorizedCell, ...],
    text: str,
) -> CompactClaim:
    first = cells[0]
    canonical_metrics = [cell.authorization["canonical_metric"] for cell in cells]
    common, precision, recall = _overlap(
        question,
        first.row_label,
        *canonical_metrics,
    )
    years = set(_question_years(question))
    period_match = sum(cell.period in years for cell in cells)
    if len(cells) == 2 and any(term in question for term in _COMPARISON_TERMS):
        period_match += 1
    unit_match = int(bool(first.unit and first.unit in question))
    required_scope = _scope_required(question)
    scope_match = int(
        required_scope is None
        or (
            required_scope != "conflict"
            and all(cell.scope == required_scope for cell in cells)
        )
    )
    return CompactClaim(
        text=text,
        cells=cells,
        metric_overlap_count=common,
        metric_precision=precision,
        question_recall=recall,
        period_match=period_match,
        unit_match=unit_match,
        scope_match=scope_match,
        rank_signal=_rank_signal(first.evidence),
    )


def _single_claim(question: str, cell: AuthorizedCell) -> CompactClaim:
    phrase = _period_phrase(cell, question)
    text = f"{cell.row_label}{phrase}为{_render_value(cell)}。"
    claim = _claim_score(question, (cell,), text)
    _validate_claim_text(claim)
    return claim


def _comparison_claim(
    question: str,
    current: AuthorizedCell,
    previous: AuthorizedCell,
) -> CompactClaim:
    text = (
        f"{current.row_label}本期为{_render_value(current)}，"
        f"上期为{_render_value(previous)}。"
    )
    claim = _claim_score(question, (current, previous), text)
    _validate_claim_text(claim)
    return claim


def _compatible_pair(
    first: AuthorizedCell,
    second: AuthorizedCell,
    question: str,
) -> bool:
    if (
        first.candidate_id != second.candidate_id
        or first.row_index != second.row_index
        or first.unit != second.unit
        or first.scope != second.scope
        or first.period == second.period
        or first.authorization["canonical_metric"]
        != second.authorization["canonical_metric"]
    ):
        return False
    current, previous = sorted((first, second), key=lambda value: -value.period)
    years = _question_years(question)
    relative = any(term in question for term in _COMPARISON_TERMS)
    return (
        current.period == previous.period + 1
        and (
            relative
            or (
                len(years) >= 2
                and current.period in years
                and previous.period in years
            )
        )
    )


def _base_numeric_values(base_answer: str) -> frozenset[str]:
    return frozenset(
        re.sub(r"[,，\s]", "", value)
        for value in numeric_fragments(base_answer)
    )


def compose_compact_claims(
    *,
    question: str,
    corpus_id: str,
    document_id: str,
    candidates: Sequence[Mapping[str, Any]],
    base_answer: str = "",
    enable_route_gate: bool = True,
    enable_complementarity: bool = True,
) -> dict[str, Any]:
    """Select and render zero to two authorized Compact TabGR claims.

    Each candidate mapping must contain ``evidence`` and either a rich ``row``
    or the mapping returned by :func:`hydrate_candidate`.
    """

    route = question_route(question)
    rejected: list[dict[str, str]] = []
    cells: list[AuthorizedCell] = []
    seen_candidate_ids: set[str] = set()
    for packet in candidates:
        evidence = packet.get("evidence")
        row = packet.get("row")
        hydration = packet.get("hydration")
        if not isinstance(evidence, Mapping) or not isinstance(row, Mapping):
            rejected.append({"candidate_id": "", "reason": "invalid_candidate_packet"})
            continue
        candidate_id = str(evidence.get("candidate_id") or "")
        if not candidate_id or candidate_id in seen_candidate_ids:
            rejected.append({"candidate_id": candidate_id, "reason": "duplicate_candidate"})
            continue
        seen_candidate_ids.add(candidate_id)
        projection = {key: row.get(key) for key in _ROW_PROJECTION_FIELDS}
        projection_sha256 = semantic_sha256(projection)
        if hydration is not None:
            if (
                not isinstance(hydration, Mapping)
                or hydration.get("candidate_id") != candidate_id
                or hydration.get("rich_row_projection_sha256")
                != projection_sha256
                or not _HEX64_RE.fullmatch(
                    str(hydration.get("document_shard_sha256") or "")
                )
            ):
                rejected.append(
                    {"candidate_id": candidate_id, "reason": "hydration_binding_failed"}
                )
                continue
        try:
            selected_cells = _authorized_cells(
                evidence,
                row,
                corpus_id=corpus_id,
                document_id=document_id,
                projection_sha256=projection_sha256,
            )
        except (CompactTabGRError, KeyError, TypeError, ValueError) as exc:
            rejected.append(
                {"candidate_id": candidate_id, "reason": str(exc)}
            )
            continue
        if not selected_cells:
            rejected.append(
                {"candidate_id": candidate_id, "reason": "no_authorized_cells"}
            )
            continue
        cells.extend(selected_cells)

    # A conflicting value for the same metric/period/unit/scope fails closed
    # for that semantic cell regardless of the ablation settings.
    values_by_key: dict[tuple[Any, ...], set[str]] = {}
    for cell in cells:
        values_by_key.setdefault(cell.semantic_conflict_key(), set()).add(
            cell.normalized_value
        )
    conflict_keys = {
        key for key, values in values_by_key.items() if len(values) > 1
    }
    if conflict_keys:
        cells = [
            cell for cell in cells
            if cell.semantic_conflict_key() not in conflict_keys
        ]

    required_scope = _scope_required(question)
    eligible_cells: list[AuthorizedCell] = []
    for cell in cells:
        common, _, _ = _overlap(
            question,
            cell.row_label,
            cell.authorization["canonical_metric"],
        )
        if common == 0:
            continue
        if required_scope == "conflict":
            continue
        if required_scope is not None and cell.scope != required_scope:
            continue
        years = _question_years(question)
        if years and cell.period not in years:
            comparison_allowed = (
                not enable_route_gate
                or route in {"table_comparison", "table_optional"}
            )
            if not comparison_allowed or cell.period != max(years) - 1:
                continue
        eligible_cells.append(cell)

    claims: list[CompactClaim] = []
    route_allows_comparison = route in {"table_comparison", "table_optional"}
    if not enable_route_gate or route_allows_comparison:
        by_candidate: dict[str, list[AuthorizedCell]] = {}
        for cell in eligible_cells:
            by_candidate.setdefault(cell.candidate_id, []).append(cell)
        for candidate_cells in by_candidate.values():
            ordered = sorted(
                candidate_cells,
                key=lambda value: (-value.period, value.coordinate),
            )
            for first_index, first in enumerate(ordered):
                for second in ordered[first_index + 1 :]:
                    if _compatible_pair(first, second, question):
                        current, previous = sorted(
                            (first, second), key=lambda value: -value.period
                        )
                        try:
                            claims.append(
                                _comparison_claim(question, current, previous)
                            )
                        except CompactTabGRError:
                            pass
    if not enable_route_gate or route != "table_blocked":
        for cell in eligible_cells:
            try:
                claims.append(_single_claim(question, cell))
            except CompactTabGRError:
                pass

    claims.sort(key=CompactClaim.rank_key)
    deduplicated: list[CompactClaim] = []
    source_keys: set[tuple[Any, ...]] = set()
    semantic_keys: set[tuple[Any, ...]] = set()
    covered_cells: set[tuple[str, tuple[int, int]]] = set()
    base_values = _base_numeric_values(base_answer)
    for claim in claims:
        if claim.metric_overlap_count == 0 or claim.source_key() in source_keys:
            continue
        if enable_complementarity and claim.semantic_key() in semantic_keys:
            continue
        claim_cells = {
            (cell.candidate_id, cell.coordinate) for cell in claim.cells
        }
        if enable_complementarity and claim_cells.intersection(covered_cells):
            continue
        if enable_complementarity and base_answer:
            normalized = {
                re.sub(r"[,，\s]", "", cell.normalized_value)
                for cell in claim.cells
            }
            raw_values = {
                re.sub(r"[,，\s]", "", cell.raw_value)
                for cell in claim.cells
            }
            if normalized.union(raw_values).intersection(base_values):
                continue
        source_keys.add(claim.source_key())
        semantic_keys.add(claim.semantic_key())
        covered_cells.update(claim_cells)
        deduplicated.append(claim)

    selected: list[CompactClaim] = []
    total_characters = 0
    for claim in deduplicated:
        separator = 1 if selected else 0
        if total_characters + separator + len(claim.text) > MAX_TOTAL_CHARACTERS:
            continue
        selected.append(claim)
        total_characters += separator + len(claim.text)
        if len(selected) == MAX_CLAIMS:
            break
    append_text = "\n".join(claim.text for claim in selected)
    if len(selected) > MAX_CLAIMS or len(append_text) > MAX_TOTAL_CHARACTERS:
        raise CompactTabGRError("Compact TabGR output bounds differ")

    selected_mappings = [claim.as_mapping() for claim in selected]
    trace_unsigned = {
        "schema_version": TRACE_SCHEMA,
        "profile_version": PROFILE_VERSION,
        "composer_version": COMPOSER_VERSION,
        "semantic_input": {"question": question},
        "route": route,
        "route_gate_enabled": enable_route_gate,
        "complementarity_enabled": enable_complementarity,
        "input_candidate_count": len(candidates),
        "authorized_cell_count": len(cells),
        "eligible_cell_count": len(eligible_cells),
        "ranked_claim_count": len(claims),
        "selected_claim_count": len(selected),
        "selected_candidate_ids": [
            claim.cells[0].candidate_id for claim in selected
        ],
        "selected_claim_sha256": [
            value["claim_sha256"] for value in selected_mappings
        ],
        "selected_row_label_cells": [
            value["row_label_cell"] for value in selected_mappings
        ],
        "rejected_candidates": rejected,
        "base_answer_sha256": semantic_sha256(base_answer),
        "append_characters": len(append_text),
    }
    trace = {
        **trace_unsigned,
        "semantic_trace_sha256": semantic_sha256(trace_unsigned),
    }
    return {
        "profile_version": PROFILE_VERSION,
        "append_text": append_text,
        "claims": selected_mappings,
        "selected_candidate_ids": trace_unsigned["selected_candidate_ids"],
        "semantic_trace": trace,
    }


__all__ = [
    "COMPOSER_VERSION",
    "CompactTabGRError",
    "MASK",
    "MAX_CELLS_PER_CLAIM",
    "MAX_CLAIMS",
    "MAX_CLAIM_CHARACTERS",
    "MAX_TOTAL_CHARACTERS",
    "PROFILE_VERSION",
    "TRACE_SCHEMA",
    "compose_compact_claims",
    "hydrate_candidate",
    "question_route",
]
