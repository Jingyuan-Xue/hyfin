"""Phase B.1 Compact TabGR composer with exact literal-cell authorization.

The module is deliberately separate from the frozen Phase B v3 composer.  It
accepts hydrated, document-bound TabGR rows plus the matching frozen
``TabGRCellLiteralAuthorization`` records.  A valid ``NumericAuthorization``
always has precedence for its exact cell; a literal authorization can only
surface the unchanged source-cell literal and can never supply a calculation
or normalized fact.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re
from typing import Any, Mapping, Sequence

from finglmqa import type3_phase51_compact_tabgr as _v3
from finglmqa.type3_tabgr_retriever import (
    infer_unit_state,
    normalize_text,
    numeric_fragments,
    semantic_sha256,
    sha256_text,
)


PROFILE_VERSION = "compact-tabgr-cell-literal-v1"
COMPOSER_VERSION = "type3-phase51-b1-compact-v1"
TRACE_SCHEMA = "finglmqa.type3.phase51.b1.compact_trace.v1"
AUTHORIZATION_MODE_EXISTING_ONLY = "existing_only"
AUTHORIZATION_MODE_EXISTING_PLUS_LITERAL = "existing_plus_literal"
AUTHORIZATION_KIND_NUMERIC = "numeric_authorization"
AUTHORIZATION_KIND_LITERAL = "source_cell_exact_literal"
MASK = _v3.MASK
MAX_CLAIMS = _v3.MAX_CLAIMS
MAX_CELLS_PER_CLAIM = _v3.MAX_CELLS_PER_CLAIM
MAX_CLAIM_CHARACTERS = _v3.MAX_CLAIM_CHARACTERS
MAX_TOTAL_CHARACTERS = _v3.MAX_TOTAL_CHARACTERS

_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_FALLBACK_HEADER_RE = re.compile(r"第\s*\d+\s*列")
_LITERAL_SCHEMA = "finglmqa.type3.tabgr.cell_literal_authorization.v1"
_COMPARISON_TERMS = (
    "同比",
    "较上年",
    "增长",
    "下降",
    "增加",
    "减少",
    "分别",
    "对比",
    "比较",
)
_LITERAL_NUMERIC_ASK_TERMS = (
    "多少",
    "金额",
    "数量",
    "比例",
    "占比",
    "分别",
    "幅度",
    "数值",
    "是多少",
    "为多少",
)
_CURRENT_PERIOD_TERMS = ("本期", "本年", "本年度", "期末", "年末")
_PREVIOUS_PERIOD_TERMS = ("上期", "上年", "上年度", "期初", "年初")
_PERCENT_UNITS = frozenset({"%", "％", "百分比", "百分数"})

_AUTH_FIELDS = {
    "schema_version",
    "authorization_kind",
    "authorization_id",
    "corpus_id",
    "document_id",
    "row_evidence_id",
    "source_binding",
    "row_binding",
    "header_binding",
    "cell_binding",
    "semantic_context",
    "grant",
    "source_cell_binding_sha256",
    "semantic_context_sha256",
    "authorization_sha256",
}
_SOURCE_FIELDS = {
    "source_markdown",
    "source_sha256",
    "table_id",
    "table_index",
    "table_sha256",
    "table_line_range",
    "structured_table_projection_sha256",
    "row_evidence_projection_sha256",
}
_ROW_FIELDS = {"row_index", "row_path", "row_path_sha256", "row_label_cell"}
_HEADER_FIELDS = {
    "column_index",
    "active_header_rows",
    "flattened_column_header",
    "flattened_column_header_sha256",
    "header_source_cells",
    "fallback_used",
}
_CELL_FIELDS = {
    "coordinate",
    "origin_coordinate",
    "origin_cell_hash",
    "origin_rowspan",
    "origin_colspan",
    "source_cell_literal",
    "source_cell_literal_sha256",
    "numeric_fragments",
}
_SOURCE_CELL_FIELDS = {
    "coordinate",
    "origin_coordinate",
    "origin_cell_hash",
    "source_cell_literal",
    "source_cell_literal_sha256",
}
_SEMANTIC_FIELDS = {"period", "unit", "accounting_scope"}
_STATE_FIELDS = {"status", "value", "source", "candidates"}
_GRANT_FIELDS = {
    "literal_role",
    "canonical_fact_status",
    "allowed_renderings",
    "normalization_allowed",
    "rounding_allowed",
    "arithmetic_allowed",
    "unit_conversion_allowed",
    "percentage_conversion_allowed",
    "whole_row_rendering_allowed",
    "mask_rendering_allowed",
}
_OUTPUT_FIELDS = {
    "profile_version",
    "authorization_mode",
    "render_suppressed",
    "append_text",
    "claims",
    "citations",
    "would_render_append_text",
    "would_render_claims",
    "would_render_citations",
    "selected_candidate_ids",
    "semantic_trace",
}
_CLAIM_FIELDS = {
    "claim_kind",
    "authorization_kind",
    "text",
    "candidate_id",
    "corpus_id",
    "document_id",
    "source_markdown",
    "source_sha256",
    "table_id",
    "table_index",
    "table_sha256",
    "table_line_range",
    "heading_path",
    "row_index",
    "row_path",
    "row_label_cell",
    "selected_cells",
    "hydrated_projection_sha256",
    "claim_sha256",
}
_OUTPUT_ROW_LABEL_FIELDS = {
    "coordinate",
    "origin_coordinate",
    "origin_cell_hash",
    "raw_value",
    "raw_value_sha256",
}
_SELECTED_CELL_FIELDS = {
    "authorization_kind",
    "authorization_id",
    "authorization_sha256",
    "coordinate",
    "origin_coordinate",
    "origin_cell_hash",
    "raw_value",
    "raw_value_sha256",
    "column_header",
    "unit",
    "period",
    "accounting_scope",
    "authorization",
}
_CITATION_FIELDS = {
    "citation_kind",
    "claim_sha256",
    "authorization_kind",
    "candidate_id",
    "corpus_id",
    "document_id",
    "source_markdown",
    "source_sha256",
    "table_id",
    "table_index",
    "table_sha256",
    "table_line_range",
    "row_index",
    "row_path",
    "row_label_cell",
    "selected_cells",
    "hydrated_projection_sha256",
    "citation_sha256",
}
_TRACE_FIELDS = {
    "schema_version",
    "profile_version",
    "composer_version",
    "semantic_input",
    "authorization_mode",
    "render_suppressed",
    "route",
    "route_gate_enabled",
    "complementarity_enabled",
    "input_candidate_count",
    "existing_authorized_cell_count",
    "literal_authorized_cell_count",
    "eligible_cell_count",
    "ranked_claim_count",
    "would_render_claim_count",
    "rendered_claim_count",
    "selected_candidate_ids",
    "selected_authorization_kinds",
    "selected_claim_sha256",
    "selected_citation_sha256",
    "selected_authorizations",
    "would_render_append_text_sha256",
    "rejected_candidates",
    "base_answer_sha256",
    "append_characters",
    "semantic_trace_sha256",
}
_TRACE_AUTHORIZATION_FIELDS = {
    "candidate_id",
    "claim_sha256",
    "authorization_kind",
    "cells",
}


class Phase51B1CompactError(ValueError):
    """Raised when a Phase B.1 input or output fails the frozen contract."""


def _closed(value: object, fields: set[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise Phase51B1CompactError(f"{label} is not a closed object")
    return value


def _coordinate(value: object, *, label: str) -> tuple[int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in value
        )
    ):
        raise Phase51B1CompactError(f"{label} is not a valid coordinate")
    return int(value[0]), int(value[1])


def _line_range(value: object, *, label: str) -> list[int]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 1
            for item in value
        )
        or value[0] > value[1]
    ):
        raise Phase51B1CompactError(f"{label} is not a valid line range")
    return [int(value[0]), int(value[1])]


def _resolved_state(value: object, *, allow_unknown: bool = False) -> str | None:
    state = _closed(value, _STATE_FIELDS, label="semantic state")
    if state.get("status") == "unknown" and allow_unknown:
        if (
            state.get("value") is not None
            or state.get("source") is not None
            or state.get("candidates") != []
        ):
            raise Phase51B1CompactError("unknown semantic state differs")
        return None
    candidates = state.get("candidates")
    if (
        state.get("status") != "resolved"
        or not isinstance(state.get("value"), str)
        or not state["value"]
        or not isinstance(state.get("source"), str)
        or not state["source"]
        or not isinstance(candidates, list)
        or not candidates
        or len(candidates) != len(set(candidates))
        or any(not isinstance(item, str) or not item for item in candidates)
        or state["value"] not in candidates
    ):
        raise Phase51B1CompactError("resolved semantic state differs")
    return str(state["value"])


def _has_percent_sign(value: str) -> bool:
    return "%" in value or "％" in value


def _is_percent_unit(value: str) -> bool:
    return normalize_text(value) in _PERCENT_UNITS


def _validate_literal_unit_pair(raw_value: str, unit: str) -> None:
    if _has_percent_sign(raw_value) and not _is_percent_unit(unit):
        raise Phase51B1CompactError(
            "percent literal conflicts with non-percent unit"
        )


def _validate_source_cell(
    value: object,
    *,
    label: str,
    required_row: int | None = None,
    required_column: int | None = None,
) -> dict[str, Any]:
    cell = _closed(value, _SOURCE_CELL_FIELDS, label=label)
    coordinate = _coordinate(cell.get("coordinate"), label=f"{label}.coordinate")
    origin = _coordinate(
        cell.get("origin_coordinate"),
        label=f"{label}.origin_coordinate",
    )
    literal = cell.get("source_cell_literal")
    if (
        not isinstance(literal, str)
        or not literal
        or cell.get("source_cell_literal_sha256") != sha256_text(literal)
        or not _HEX64_RE.fullmatch(str(cell.get("origin_cell_hash") or ""))
        or origin[0] > coordinate[0]
        or origin[1] > coordinate[1]
        or (required_row is not None and coordinate[0] != required_row)
        or (required_column is not None and coordinate[1] != required_column)
    ):
        raise Phase51B1CompactError(f"{label} provenance differs")
    return {
        "coordinate": list(coordinate),
        "origin_coordinate": list(origin),
        "origin_cell_hash": str(cell["origin_cell_hash"]),
        "source_cell_literal": literal,
        "source_cell_literal_sha256": str(
            cell["source_cell_literal_sha256"]
        ),
    }


def _literal_hash_chain(record: Mapping[str, Any]) -> tuple[str, str, str]:
    source_hash = semantic_sha256(
        {
            "corpus_id": record["corpus_id"],
            "document_id": record["document_id"],
            "row_evidence_id": record["row_evidence_id"],
            "source_binding": record["source_binding"],
            "row_index": record["row_binding"]["row_index"],
            "cell_binding": record["cell_binding"],
        }
    )
    semantic_hash = semantic_sha256(
        {
            "row_path": record["row_binding"]["row_path"],
            "row_label_cell": record["row_binding"]["row_label_cell"],
            "header_binding": record["header_binding"],
            "semantic_context": record["semantic_context"],
        }
    )
    payload = {
        key: value
        for key, value in record.items()
        if key not in {"authorization_id", "authorization_sha256"}
    }
    payload["source_cell_binding_sha256"] = source_hash
    payload["semantic_context_sha256"] = semantic_hash
    authorization_hash = semantic_sha256(payload)
    return source_hash, semantic_hash, authorization_hash


def _validate_literal_shape_and_hash(record: Mapping[str, Any]) -> None:
    _closed(record, _AUTH_FIELDS, label="literal authorization")
    source = _closed(
        record.get("source_binding"),
        _SOURCE_FIELDS,
        label="literal source_binding",
    )
    row_binding = _closed(
        record.get("row_binding"),
        _ROW_FIELDS,
        label="literal row_binding",
    )
    header = _closed(
        record.get("header_binding"),
        _HEADER_FIELDS,
        label="literal header_binding",
    )
    cell = _closed(
        record.get("cell_binding"),
        _CELL_FIELDS,
        label="literal cell_binding",
    )
    semantic = _closed(
        record.get("semantic_context"),
        _SEMANTIC_FIELDS,
        label="literal semantic_context",
    )
    grant = _closed(
        record.get("grant"),
        _GRANT_FIELDS,
        label="literal grant",
    )
    if (
        record.get("schema_version") != _LITERAL_SCHEMA
        or record.get("authorization_kind") != AUTHORIZATION_KIND_LITERAL
        or not isinstance(record.get("corpus_id"), str)
        or not record["corpus_id"]
        or not isinstance(record.get("document_id"), str)
        or not record["document_id"]
        or not isinstance(record.get("row_evidence_id"), str)
        or not record["row_evidence_id"]
    ):
        raise Phase51B1CompactError("literal authorization identity differs")
    for key in (
        "source_sha256",
        "table_sha256",
        "structured_table_projection_sha256",
        "row_evidence_projection_sha256",
    ):
        if not _HEX64_RE.fullmatch(str(source.get(key) or "")):
            raise Phase51B1CompactError(f"literal source hash differs at {key}")
    _line_range(source.get("table_line_range"), label="table_line_range")
    if (
        not isinstance(source.get("table_index"), int)
        or isinstance(source.get("table_index"), bool)
        or source["table_index"] < 1
    ):
        raise Phase51B1CompactError("literal table_index differs")
    if (
        not isinstance(row_binding.get("row_index"), int)
        or isinstance(row_binding.get("row_index"), bool)
        or row_binding["row_index"] < 0
        or not isinstance(row_binding.get("row_path"), list)
        or not row_binding["row_path"]
        or any(not isinstance(item, str) or not item for item in row_binding["row_path"])
        or row_binding.get("row_path_sha256")
        != semantic_sha256(row_binding["row_path"])
    ):
        raise Phase51B1CompactError("literal row binding differs")
    _validate_source_cell(
        row_binding.get("row_label_cell"),
        label="literal row_label_cell",
        required_row=int(row_binding["row_index"]),
    )
    column = header.get("column_index")
    active_rows = header.get("active_header_rows")
    if (
        not isinstance(column, int)
        or isinstance(column, bool)
        or column < 0
        or not isinstance(active_rows, list)
        or not active_rows
        or len(active_rows) != len(set(active_rows))
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in active_rows
        )
        or not isinstance(header.get("flattened_column_header"), str)
        or not header["flattened_column_header"]
        or header.get("flattened_column_header_sha256")
        != sha256_text(header["flattened_column_header"])
        or header.get("fallback_used") is not False
        or _FALLBACK_HEADER_RE.fullmatch(header["flattened_column_header"])
        or not isinstance(header.get("header_source_cells"), list)
        or not header["header_source_cells"]
    ):
        raise Phase51B1CompactError("literal header binding differs")
    header_cells = [
        _validate_source_cell(
            value,
            label="literal header_source_cell",
            required_column=column,
        )
        for value in header["header_source_cells"]
    ]
    header_coordinates = [tuple(value["coordinate"]) for value in header_cells]
    if (
        len(header_coordinates) != len(set(header_coordinates))
        or [value[0] for value in header_coordinates]
        != sorted(value[0] for value in header_coordinates)
        or any(value[0] not in active_rows for value in header_coordinates)
    ):
        raise Phase51B1CompactError("literal header source-cell order differs")
    coordinate = _coordinate(cell.get("coordinate"), label="literal coordinate")
    origin = _coordinate(
        cell.get("origin_coordinate"),
        label="literal origin_coordinate",
    )
    literal = cell.get("source_cell_literal")
    fragments = cell.get("numeric_fragments")
    if (
        coordinate[1] != column
        or origin[0] > coordinate[0]
        or origin[1] > coordinate[1]
        or not _HEX64_RE.fullmatch(str(cell.get("origin_cell_hash") or ""))
        or cell.get("origin_rowspan") != 1
        or cell.get("origin_colspan") != 1
        or not isinstance(literal, str)
        or not literal
        or len(literal) > 96
        or MASK in literal
        or cell.get("source_cell_literal_sha256") != sha256_text(literal)
        or not isinstance(fragments, list)
        or len(fragments) != 1
        or fragments != numeric_fragments(literal)
        or normalize_text(literal) != fragments[0]
    ):
        raise Phase51B1CompactError("literal cell binding differs")
    _resolved_state(semantic.get("period"))
    unit = _resolved_state(semantic.get("unit"))
    _resolved_state(semantic.get("accounting_scope"), allow_unknown=True)
    if unit is None or MASK in unit or numeric_fragments(unit):
        raise Phase51B1CompactError("literal unit is not render-safe")
    _validate_literal_unit_pair(literal, unit)
    expected_grant = {
        "literal_role": "answer_surface_only",
        "canonical_fact_status": "not_a_canonical_fact",
        "allowed_renderings": [literal],
        "normalization_allowed": False,
        "rounding_allowed": False,
        "arithmetic_allowed": False,
        "unit_conversion_allowed": False,
        "percentage_conversion_allowed": False,
        "whole_row_rendering_allowed": False,
        "mask_rendering_allowed": False,
    }
    if dict(grant) != expected_grant:
        raise Phase51B1CompactError("literal grant differs")
    source_hash, semantic_hash, authorization_hash = _literal_hash_chain(record)
    if (
        record.get("source_cell_binding_sha256") != source_hash
        or record.get("semantic_context_sha256") != semantic_hash
        or record.get("authorization_sha256") != authorization_hash
        or record.get("authorization_id")
        != "t3tabgr-lit-" + authorization_hash[:24]
    ):
        raise Phase51B1CompactError("literal authorization hash chain differs")


@dataclass(frozen=True)
class B1AuthorizedCell:
    authorization_kind: str
    authorization_id: str
    authorization_sha256: str
    candidate_id: str
    row_label: str
    row_label_cell: _v3.RowLabelCell
    row_index: int
    table_index: int
    coordinate: tuple[int, int]
    origin_coordinate: tuple[int, int]
    origin_cell_hash: str
    column_header: str
    raw_value: str
    raw_value_sha256: str
    unit: str
    period: int
    scope: str | None
    semantic_value: str
    metric_context: tuple[str, ...]
    authorization: Mapping[str, Any]
    evidence: Mapping[str, Any]
    row: Mapping[str, Any]
    projection_sha256: str

    def cell_key(self) -> tuple[str, tuple[int, int]]:
        return self.candidate_id, self.coordinate

    def semantic_conflict_key(self) -> tuple[Any, ...]:
        return (
            normalize_text(self.row_label),
            self.period,
            normalize_text(self.unit),
            self.scope,
            normalize_text(self.column_header),
        )

    def provenance_mapping(self) -> dict[str, Any]:
        return {
            "authorization_kind": self.authorization_kind,
            "authorization_id": self.authorization_id,
            "authorization_sha256": self.authorization_sha256,
            "coordinate": list(self.coordinate),
            "origin_coordinate": list(self.origin_coordinate),
            "origin_cell_hash": self.origin_cell_hash,
            "raw_value": self.raw_value,
            "raw_value_sha256": self.raw_value_sha256,
            "column_header": self.column_header,
            "unit": self.unit,
            "period": str(self.period),
            "accounting_scope": self.scope,
            "authorization": dict(self.authorization),
        }


def _row_cell(row: Mapping[str, Any], coordinate: tuple[int, int]) -> Mapping[str, Any]:
    matches = [
        cell
        for cell in row.get("cells") or ()
        if isinstance(cell, Mapping)
        and _coordinate(cell.get("coordinate"), label="row cell coordinate")
        == coordinate
    ]
    if len(matches) != 1:
        raise Phase51B1CompactError("literal coordinate does not bind one row cell")
    return matches[0]


def _row_label_source_mapping(label: _v3.RowLabelCell) -> dict[str, Any]:
    return {
        "coordinate": list(label.coordinate),
        "origin_coordinate": list(label.origin_coordinate),
        "origin_cell_hash": label.origin_cell_hash,
        "source_cell_literal": label.raw_value,
        "source_cell_literal_sha256": label.raw_value_sha256,
    }


def validate_literal_authorization(
    record: Mapping[str, Any],
    *,
    evidence: Mapping[str, Any],
    row: Mapping[str, Any],
    corpus_id: str,
    document_id: str,
    authoritative_record: Mapping[str, Any] | None = None,
    authoritative_row_evidence_sha256: str | None = None,
) -> B1AuthorizedCell:
    """Validate one frozen literal record against its authoritative rich row.

    ``authoritative_record`` is optional for normal package consumption.  An
    independent validator should pass the separately reloaded package record;
    this makes attacker rehashing of a mutated working copy insufficient.
    """

    if authoritative_record is not None and dict(record) != dict(authoritative_record):
        raise Phase51B1CompactError("literal record differs from authoritative shard")
    _validate_literal_shape_and_hash(record)
    try:
        _v3._evidence_binding(  # type: ignore[attr-defined]
            evidence,
            row,
            corpus_id=corpus_id,
            document_id=document_id,
        )
        headers = _v3._validated_column_headers(row)  # type: ignore[attr-defined]
        label_cell = _v3._exact_row_label(row)  # type: ignore[attr-defined]
    except (ValueError, KeyError, TypeError) as exc:
        raise Phase51B1CompactError(str(exc)) from exc
    if (
        record["corpus_id"] != corpus_id
        or record["document_id"] != document_id
        or record["row_evidence_id"] != row.get("evidence_id")
    ):
        raise Phase51B1CompactError("literal corpus/document/row identity differs")
    source = record["source_binding"]
    safe_projection_sha256 = semantic_sha256(row)
    bound_row_evidence_sha256 = (
        authoritative_row_evidence_sha256 or safe_projection_sha256
    )
    if not _HEX64_RE.fullmatch(bound_row_evidence_sha256):
        raise Phase51B1CompactError(
            "authoritative row-evidence projection hash differs"
        )
    source_expected = {
        "source_markdown": row.get("source_markdown"),
        "source_sha256": evidence.get("source_sha256"),
        "table_id": row.get("table_id"),
        "table_index": row.get("table_index"),
        "table_sha256": row.get("table_sha256"),
        "table_line_range": list(row.get("table_line_range") or ()),
        "row_evidence_projection_sha256": bound_row_evidence_sha256,
    }
    if any(source.get(key) != value for key, value in source_expected.items()):
        raise Phase51B1CompactError("literal source/table projection binding differs")
    row_binding = record["row_binding"]
    if (
        row_binding["row_index"] != row.get("row_index")
        or row_binding["row_path"] != list(row.get("row_path") or ())
        or row_binding["row_label_cell"] != _row_label_source_mapping(label_cell)
    ):
        raise Phase51B1CompactError("literal row-label binding differs")
    header = record["header_binding"]
    coordinate = _coordinate(
        record["cell_binding"]["coordinate"],
        label="literal coordinate",
    )
    column = coordinate[1]
    if (
        column >= len(headers)
        or header["column_index"] != column
        or header["active_header_rows"] != list(row.get("active_header_rows") or ())
        or header["flattened_column_header"] != headers[column]
    ):
        raise Phase51B1CompactError("literal flattened-header binding differs")
    cell = _row_cell(row, coordinate)
    cell_binding = record["cell_binding"]
    if (
        cell.get("numeric_status") != "unauthorized"
        or cell.get("authorization_ids") != []
        or list(cell.get("origin_coordinate") or ())
        != cell_binding["origin_coordinate"]
        or cell.get("origin_cell_hash") != cell_binding["origin_cell_hash"]
        or cell.get("raw_value") != cell_binding["source_cell_literal"]
        or cell.get("raw_value_sha256")
        != cell_binding["source_cell_literal_sha256"]
        or cell.get("column_header") != header["flattened_column_header"]
    ):
        raise Phase51B1CompactError("literal row-cell binding differs")
    semantic = record["semantic_context"]
    semantic_states = row.get("semantic_states")
    if not isinstance(semantic_states, Mapping):
        raise Phase51B1CompactError("literal row semantic_states differs")
    for field, state in (
        ("period_by_column", semantic["period"]),
        ("unit_by_column", semantic["unit"]),
        ("accounting_scope_by_column", semantic["accounting_scope"]),
    ):
        by_column = semantic_states.get(field)
        if (
            not isinstance(by_column, Mapping)
            or by_column.get(str(column)) != state
        ):
            raise Phase51B1CompactError("literal semantic column binding differs")
    if (
        cell.get("period") != semantic["period"]
        or cell.get("unit") != semantic["unit"]
        or cell.get("accounting_scope") != semantic["accounting_scope"]
    ):
        raise Phase51B1CompactError("literal cell semantic binding differs")
    period_text = _resolved_state(semantic["period"])
    unit = _resolved_state(semantic["unit"])
    scope = _resolved_state(semantic["accounting_scope"], allow_unknown=True)
    if period_text is None or not period_text.isdigit() or unit is None:
        raise Phase51B1CompactError("literal period/unit differs")
    header_years = sorted(
        {int(value) for value in re.findall(r"(?<!\d)((?:19|20)\d{2})(?!\d)", headers[column])}
    )
    if header_years and (
        len(header_years) != 1 or header_years[0] != int(period_text)
    ):
        raise Phase51B1CompactError("literal header/period binding differs")
    header_unit = infer_unit_state(headers[column])
    if (
        header_unit.get("status") == "conflict"
        or (
            header_unit.get("status") == "resolved"
            and header_unit.get("value") != unit
        )
    ):
        raise Phase51B1CompactError("literal header/unit binding differs")
    literal = str(cell_binding["source_cell_literal"])
    return B1AuthorizedCell(
        authorization_kind=AUTHORIZATION_KIND_LITERAL,
        authorization_id=str(record["authorization_id"]),
        authorization_sha256=str(record["authorization_sha256"]),
        candidate_id=str(evidence["candidate_id"]),
        row_label=label_cell.raw_value,
        row_label_cell=label_cell,
        row_index=int(row["row_index"]),
        table_index=int(row["table_index"]),
        coordinate=coordinate,
        origin_coordinate=_coordinate(
            cell_binding["origin_coordinate"],
            label="literal origin_coordinate",
        ),
        origin_cell_hash=str(cell_binding["origin_cell_hash"]),
        column_header=str(header["flattened_column_header"]),
        raw_value=literal,
        raw_value_sha256=str(cell_binding["source_cell_literal_sha256"]),
        unit=unit,
        period=int(period_text),
        scope=scope,
        semantic_value=literal,
        metric_context=(
            label_cell.raw_value,
            str(header["flattened_column_header"]),
            *(str(value) for value in row.get("heading_path") or ()),
        ),
        authorization=dict(record),
        evidence=evidence,
        row=row,
        projection_sha256=safe_projection_sha256,
    )


def _numeric_cell(value: _v3.AuthorizedCell) -> B1AuthorizedCell:
    authorization = dict(value.authorization)
    _validate_literal_unit_pair(value.raw_value, value.unit)
    return B1AuthorizedCell(
        authorization_kind=AUTHORIZATION_KIND_NUMERIC,
        authorization_id=str(authorization["authorization_id"]),
        authorization_sha256=semantic_sha256(authorization),
        candidate_id=value.candidate_id,
        row_label=value.row_label,
        row_label_cell=value.row_label_cell,
        row_index=value.row_index,
        table_index=value.table_index,
        coordinate=value.coordinate,
        origin_coordinate=value.origin_coordinate,
        origin_cell_hash=value.origin_cell_hash,
        column_header=value.column_header,
        raw_value=value.raw_value,
        raw_value_sha256=value.raw_value_sha256,
        unit=value.unit,
        period=value.period,
        scope=value.scope,
        semantic_value=value.normalized_value,
        metric_context=(
            value.row_label,
            str(authorization["canonical_metric"]),
        ),
        authorization=authorization,
        evidence=value.evidence,
        row=value.row,
        projection_sha256=value.projection_sha256,
    )


def _literal_targets_numeric_cell(
    record: Mapping[str, Any],
    *,
    numeric_by_coordinate: Mapping[tuple[int, int], B1AuthorizedCell],
    row: Mapping[str, Any],
    evidence: Mapping[str, Any],
    corpus_id: str,
    document_id: str,
) -> bool:
    """Ignore a redundant exact literal record only when it matches strong auth."""

    _validate_literal_shape_and_hash(record)
    coordinate = _coordinate(
        record["cell_binding"]["coordinate"],
        label="literal coordinate",
    )
    numeric = numeric_by_coordinate.get(coordinate)
    if numeric is None:
        return False
    source = record["source_binding"]
    cell = record["cell_binding"]
    if (
        record["corpus_id"] != corpus_id
        or record["document_id"] != document_id
        or record["row_evidence_id"] != row.get("evidence_id")
        or source["source_markdown"] != row.get("source_markdown")
        or source["source_sha256"] != evidence.get("source_sha256")
        or source["table_id"] != row.get("table_id")
        or source["table_sha256"] != row.get("table_sha256")
        or cell["source_cell_literal"] != numeric.raw_value
        or cell["source_cell_literal_sha256"] != numeric.raw_value_sha256
        or cell["origin_coordinate"] != list(numeric.origin_coordinate)
        or cell["origin_cell_hash"] != numeric.origin_cell_hash
    ):
        raise Phase51B1CompactError(
            "literal record conflicts with stronger NumericAuthorization"
        )
    return True


def _question_scope(question: str) -> str | None:
    try:
        return _v3._scope_required(question)  # type: ignore[attr-defined]
    except AttributeError:
        consolidated = "合并" in question
        parent = "母公司" in question
        if consolidated and parent:
            return "conflict"
        if consolidated:
            return "consolidated"
        if parent:
            return "parent_company"
        return None


def _question_years(question: str) -> tuple[int, ...]:
    return tuple(
        sorted(
            {
                int(value)
                for value in re.findall(
                    r"(?<!\d)((?:19|20)\d{2})(?!\d)",
                    question,
                )
            }
        )
    )


def _literal_numeric_ask(question: str) -> bool:
    return any(term in question for term in _LITERAL_NUMERIC_ASK_TERMS)


def _explicit_comparison(question: str) -> bool:
    return (
        any(term in question for term in _COMPARISON_TERMS)
        or len(_question_years(question)) >= 2
    )


def _period_specific_single_cells(
    question: str,
    cells: Sequence[B1AuthorizedCell],
) -> tuple[B1AuthorizedCell, ...]:
    """Resolve non-comparison single-cell ambiguity within one exact row."""

    if len(cells) <= 1 or _explicit_comparison(question):
        return tuple(cells)
    years = _question_years(question)
    if years:
        matches = [cell for cell in cells if cell.period in years]
        return tuple(matches) if len(matches) == 1 else ()
    asks_current = any(term in question for term in _CURRENT_PERIOD_TERMS)
    asks_previous = any(term in question for term in _PREVIOUS_PERIOD_TERMS)
    if asks_current == asks_previous:
        return ()
    terms = _CURRENT_PERIOD_TERMS if asks_current else _PREVIOUS_PERIOD_TERMS
    matches = [
        cell
        for cell in cells
        if any(term in cell.column_header for term in terms)
    ]
    return tuple(matches) if len(matches) == 1 else ()


def _semantic_overlap(question: str, cell: B1AuthorizedCell) -> tuple[int, float, float]:
    return _v3._overlap(  # type: ignore[attr-defined]
        question,
        cell.row_label,
        *cell.metric_context,
    )


def _rank_signal(evidence: Mapping[str, Any]) -> float:
    return _v3._rank_signal(evidence)  # type: ignore[attr-defined]


def _period_phrase(cell: B1AuthorizedCell, question: str) -> str:
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


def _render_raw_value(raw: str, unit: str) -> str:
    _validate_literal_unit_pair(raw, unit)
    if _has_percent_sign(raw):
        return raw
    if unit and raw.endswith(unit):
        return raw
    return raw + unit


def _render_value(cell: B1AuthorizedCell) -> str:
    return _render_raw_value(cell.raw_value, cell.unit)


@dataclass(frozen=True)
class B1Claim:
    text: str
    cells: tuple[B1AuthorizedCell, ...]
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
            tuple(
                re.sub(r"[,，\s]", "", cell.semantic_value)
                for cell in self.cells
            ),
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
        selected_cells = [cell.provenance_mapping() for cell in self.cells]
        unsigned = {
            "claim_kind": (
                "compact_tabgr_comparison"
                if len(self.cells) == 2
                else "compact_tabgr_single_value"
            ),
            "authorization_kind": first.authorization_kind,
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
        }
        return {
            **unsigned,
            "claim_sha256": semantic_sha256(unsigned),
        }


def _validate_claim_text(claim: B1Claim) -> None:
    if (
        not claim.text
        or len(claim.text) > MAX_CLAIM_CHARACTERS
        or MASK in claim.text
        or not 1 <= len(claim.cells) <= MAX_CELLS_PER_CLAIM
        or len({cell.authorization_kind for cell in claim.cells}) != 1
    ):
        raise Phase51B1CompactError("compact claim shape/length differs")
    expected_fragments: list[str] = []
    expected_literals: Counter[str] = Counter()
    for cell in claim.cells:
        allowed = cell.authorization.get("allowed_renderings")
        if cell.authorization_kind == AUTHORIZATION_KIND_LITERAL:
            allowed = cell.authorization["grant"]["allowed_renderings"]
        if allowed != [cell.raw_value]:
            raise Phase51B1CompactError("claim changes an authorized literal")
        expected_literals[cell.raw_value] += 1
        expected_fragments.extend(
            numeric_fragments(_render_raw_value(cell.raw_value, cell.unit))
        )
    if any(
        claim.text.count(literal) != count
        for literal, count in expected_literals.items()
    ):
        raise Phase51B1CompactError("claim changes an authorized literal")
    if Counter(numeric_fragments(claim.text)) != Counter(expected_fragments):
        raise Phase51B1CompactError("claim numeric fragments differ from source cells")


def _claim_score(
    question: str,
    cells: tuple[B1AuthorizedCell, ...],
    text: str,
) -> B1Claim:
    common, precision, recall = _semantic_overlap(question, cells[0])
    years = set(_question_years(question))
    period_match = sum(cell.period in years for cell in cells)
    if len(cells) == 2 and any(term in question for term in _COMPARISON_TERMS):
        period_match += 1
    unit_match = int(bool(cells[0].unit and cells[0].unit in question))
    required_scope = _question_scope(question)
    scope_match = int(
        required_scope is None
        or (
            required_scope != "conflict"
            and all(cell.scope == required_scope for cell in cells)
        )
    )
    return B1Claim(
        text=text,
        cells=cells,
        metric_overlap_count=common,
        metric_precision=precision,
        question_recall=recall,
        period_match=period_match,
        unit_match=unit_match,
        scope_match=scope_match,
        rank_signal=_rank_signal(cells[0].evidence),
    )


def _single_claim(question: str, cell: B1AuthorizedCell) -> B1Claim:
    claim = _claim_score(
        question,
        (cell,),
        f"{cell.row_label}{_period_phrase(cell, question)}为{_render_value(cell)}。",
    )
    _validate_claim_text(claim)
    return claim


def _comparison_claim(
    question: str,
    current: B1AuthorizedCell,
    previous: B1AuthorizedCell,
) -> B1Claim:
    claim = _claim_score(
        question,
        (current, previous),
        (
            f"{current.row_label}本期为{_render_value(current)}，"
            f"上期为{_render_value(previous)}。"
        ),
    )
    _validate_claim_text(claim)
    return claim


def _compatible_pair(
    first: B1AuthorizedCell,
    second: B1AuthorizedCell,
    question: str,
) -> bool:
    if (
        first.authorization_kind != second.authorization_kind
        or first.candidate_id != second.candidate_id
        or first.row_index != second.row_index
        or first.row.get("table_id") != second.row.get("table_id")
        or first.row_label != second.row_label
        or first.unit != second.unit
        or first.scope != second.scope
        or first.period == second.period
    ):
        return False
    if (
        first.authorization_kind == AUTHORIZATION_KIND_NUMERIC
        and first.authorization.get("canonical_metric")
        != second.authorization.get("canonical_metric")
    ):
        return False
    current, previous = sorted((first, second), key=lambda value: -value.period)
    years = _question_years(question)
    explicit_comparison = any(term in question for term in _COMPARISON_TERMS) or (
        len(years) >= 2
        and current.period in years
        and previous.period in years
    )
    return current.period == previous.period + 1 and explicit_comparison


def _citation(claim: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = {
        "citation_kind": "compact_tabgr_cell_claim",
        "claim_sha256": claim["claim_sha256"],
        "authorization_kind": claim["authorization_kind"],
        "candidate_id": claim["candidate_id"],
        "corpus_id": claim["corpus_id"],
        "document_id": claim["document_id"],
        "source_markdown": claim["source_markdown"],
        "source_sha256": claim["source_sha256"],
        "table_id": claim["table_id"],
        "table_index": claim["table_index"],
        "table_sha256": claim["table_sha256"],
        "table_line_range": claim["table_line_range"],
        "row_index": claim["row_index"],
        "row_path": claim["row_path"],
        "row_label_cell": claim["row_label_cell"],
        "selected_cells": claim["selected_cells"],
        "hydrated_projection_sha256": claim["hydrated_projection_sha256"],
    }
    return {**unsigned, "citation_sha256": semantic_sha256(unsigned)}


def _base_numeric_values(base_answer: str) -> frozenset[str]:
    return frozenset(
        re.sub(r"[,，\s]", "", value)
        for value in numeric_fragments(base_answer)
    )


def _compose_impl(
    *,
    question: str,
    corpus_id: str,
    document_id: str,
    candidates: Sequence[Mapping[str, Any]],
    base_answer: str,
    authorization_mode: str,
    enable_route_gate: bool,
    enable_complementarity: bool,
    suppress_rendering: bool,
) -> dict[str, Any]:
    if authorization_mode not in {
        AUTHORIZATION_MODE_EXISTING_ONLY,
        AUTHORIZATION_MODE_EXISTING_PLUS_LITERAL,
    }:
        raise Phase51B1CompactError("authorization_mode differs")
    route = _v3.question_route(question)
    rejected: list[dict[str, str]] = []
    cells: list[B1AuthorizedCell] = []
    seen_candidate_ids: set[str] = set()
    existing_count = 0
    literal_count = 0
    for packet in candidates:
        evidence = packet.get("evidence")
        row = packet.get("row")
        hydration = packet.get("hydration")
        candidate_id = (
            str(evidence.get("candidate_id") or "")
            if isinstance(evidence, Mapping)
            else ""
        )
        if (
            not isinstance(evidence, Mapping)
            or not isinstance(row, Mapping)
            or not candidate_id
            or candidate_id in seen_candidate_ids
        ):
            rejected.append(
                {"candidate_id": candidate_id, "reason": "invalid_or_duplicate_candidate"}
            )
            continue
        seen_candidate_ids.add(candidate_id)
        projection_sha256 = semantic_sha256(row)
        if hydration is not None and (
            not isinstance(hydration, Mapping)
            or hydration.get("candidate_id") != candidate_id
            or hydration.get("rich_row_projection_sha256") != projection_sha256
            or not _HEX64_RE.fullmatch(
                str(hydration.get("document_shard_sha256") or "")
            )
        ):
            rejected.append(
                {"candidate_id": candidate_id, "reason": "hydration_binding_failed"}
            )
            continue
        try:
            numeric_cells = tuple(
                _numeric_cell(value)
                for value in _v3._authorized_cells(  # type: ignore[attr-defined]
                    evidence,
                    row,
                    corpus_id=corpus_id,
                    document_id=document_id,
                    projection_sha256=projection_sha256,
                )
            )
            numeric_by_coordinate = {
                value.coordinate: value for value in numeric_cells
            }
            selected_cells = list(numeric_cells)
            if authorization_mode == AUTHORIZATION_MODE_EXISTING_PLUS_LITERAL:
                raw_literal_records = packet.get("literal_authorizations") or []
                if not isinstance(raw_literal_records, list):
                    raise Phase51B1CompactError(
                        "literal_authorizations is not an array"
                    )
                seen_literal_ids: set[str] = set()
                authoritative_row_hash = packet.get(
                    "authoritative_row_evidence_sha256"
                )
                if authoritative_row_hash is None and isinstance(
                    hydration,
                    Mapping,
                ):
                    authoritative_row_hash = hydration.get(
                        "authoritative_row_evidence_sha256"
                    )
                for record in raw_literal_records:
                    if not isinstance(record, Mapping):
                        raise Phase51B1CompactError(
                            "literal authorization is not an object"
                        )
                    authorization_id = str(record.get("authorization_id") or "")
                    if (
                        not authorization_id
                        or authorization_id in seen_literal_ids
                    ):
                        raise Phase51B1CompactError(
                            "literal authorization ids are invalid"
                        )
                    seen_literal_ids.add(authorization_id)
                    if _literal_targets_numeric_cell(
                        record,
                        numeric_by_coordinate=numeric_by_coordinate,
                        row=row,
                        evidence=evidence,
                        corpus_id=corpus_id,
                        document_id=document_id,
                    ):
                        continue
                    selected_cells.append(
                        validate_literal_authorization(
                            record,
                            evidence=evidence,
                            row=row,
                            corpus_id=corpus_id,
                            document_id=document_id,
                            authoritative_row_evidence_sha256=(
                                str(authoritative_row_hash)
                                if authoritative_row_hash is not None
                                else None
                            ),
                        )
                    )
            if not selected_cells:
                rejected.append(
                    {"candidate_id": candidate_id, "reason": "no_authorized_cells"}
                )
                continue
            existing_count += sum(
                value.authorization_kind == AUTHORIZATION_KIND_NUMERIC
                for value in selected_cells
            )
            literal_count += sum(
                value.authorization_kind == AUTHORIZATION_KIND_LITERAL
                for value in selected_cells
            )
            cells.extend(selected_cells)
        except (ValueError, KeyError, TypeError) as exc:
            rejected.append({"candidate_id": candidate_id, "reason": str(exc)})

    values_by_key: dict[tuple[Any, ...], set[str]] = {}
    for cell in cells:
        values_by_key.setdefault(cell.semantic_conflict_key(), set()).add(
            cell.semantic_value
        )
    conflicts = {
        key for key, values in values_by_key.items() if len(values) > 1
    }
    if conflicts:
        cells = [
            cell for cell in cells if cell.semantic_conflict_key() not in conflicts
        ]

    required_scope = _question_scope(question)
    years = _question_years(question)
    eligible_cells: list[B1AuthorizedCell] = []
    for cell in cells:
        common, _, _ = _semantic_overlap(question, cell)
        if common == 0 or required_scope == "conflict":
            continue
        if (
            cell.authorization_kind == AUTHORIZATION_KIND_LITERAL
            and not _literal_numeric_ask(question)
        ):
            continue
        if required_scope is not None and cell.scope != required_scope:
            continue
        if years and cell.period not in years:
            comparison_allowed = (
                not enable_route_gate
                or route in {"table_comparison", "table_optional"}
            )
            if not comparison_allowed or cell.period != max(years) - 1:
                continue
        eligible_cells.append(cell)

    claims: list[B1Claim] = []
    route_allows_comparison = route in {"table_comparison", "table_optional"}
    if not enable_route_gate or route_allows_comparison:
        by_candidate: dict[str, list[B1AuthorizedCell]] = {}
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
                            (first, second),
                            key=lambda value: -value.period,
                        )
                        try:
                            claims.append(
                                _comparison_claim(question, current, previous)
                            )
                        except Phase51B1CompactError:
                            pass
    single_cells: list[B1AuthorizedCell] = []
    by_exact_row: dict[tuple[str, str], list[B1AuthorizedCell]] = {}
    for cell in eligible_cells:
        by_exact_row.setdefault(
            (cell.candidate_id, normalize_text(cell.row_label)),
            [],
        ).append(cell)
    for row_cells in by_exact_row.values():
        single_cells.extend(_period_specific_single_cells(question, row_cells))
    if not enable_route_gate or route != "table_blocked":
        for cell in single_cells:
            try:
                claims.append(_single_claim(question, cell))
            except Phase51B1CompactError:
                pass

    claims.sort(key=B1Claim.rank_key)
    deduplicated: list[B1Claim] = []
    source_keys: set[tuple[Any, ...]] = set()
    semantic_keys: set[tuple[Any, ...]] = set()
    covered_cells: set[tuple[str, tuple[int, int]]] = set()
    covered_rows: set[tuple[str, str]] = set()
    base_values = _base_numeric_values(base_answer)
    for claim in claims:
        row_key = (
            claim.cells[0].candidate_id,
            normalize_text(claim.cells[0].row_label),
        )
        if (
            claim.metric_overlap_count == 0
            or claim.source_key() in source_keys
            or row_key in covered_rows
        ):
            continue
        if enable_complementarity and claim.semantic_key() in semantic_keys:
            continue
        claim_cells = {cell.cell_key() for cell in claim.cells}
        if enable_complementarity and claim_cells.intersection(covered_cells):
            continue
        if enable_complementarity and base_answer:
            values = {
                re.sub(r"[,，\s]", "", cell.semantic_value)
                for cell in claim.cells
            }.union(
                {
                    re.sub(r"[,，\s]", "", cell.raw_value)
                    for cell in claim.cells
                }
            )
            if values.intersection(base_values):
                continue
        source_keys.add(claim.source_key())
        semantic_keys.add(claim.semantic_key())
        covered_cells.update(claim_cells)
        covered_rows.add(row_key)
        deduplicated.append(claim)

    selected: list[B1Claim] = []
    total_characters = 0
    for claim in deduplicated:
        separator = 1 if selected else 0
        if total_characters + separator + len(claim.text) > MAX_TOTAL_CHARACTERS:
            continue
        selected.append(claim)
        total_characters += separator + len(claim.text)
        if len(selected) == MAX_CLAIMS:
            break
    would_render_claims = [claim.as_mapping() for claim in selected]
    would_render_citations = [_citation(claim) for claim in would_render_claims]
    would_render_append_text = "\n".join(claim.text for claim in selected)
    rendered_claims = [] if suppress_rendering else would_render_claims
    rendered_citations = [] if suppress_rendering else would_render_citations
    append_text = "" if suppress_rendering else would_render_append_text

    selected_authorizations = [
        {
            "candidate_id": claim["candidate_id"],
            "claim_sha256": claim["claim_sha256"],
            "authorization_kind": claim["authorization_kind"],
            "cells": claim["selected_cells"],
        }
        for claim in would_render_claims
    ]
    trace_unsigned = {
        "schema_version": TRACE_SCHEMA,
        "profile_version": PROFILE_VERSION,
        "composer_version": COMPOSER_VERSION,
        "semantic_input": {"question": question},
        "authorization_mode": authorization_mode,
        "render_suppressed": suppress_rendering,
        "route": route,
        "route_gate_enabled": enable_route_gate,
        "complementarity_enabled": enable_complementarity,
        "input_candidate_count": len(candidates),
        "existing_authorized_cell_count": existing_count,
        "literal_authorized_cell_count": literal_count,
        "eligible_cell_count": len(eligible_cells),
        "ranked_claim_count": len(claims),
        "would_render_claim_count": len(would_render_claims),
        "rendered_claim_count": len(rendered_claims),
        "selected_candidate_ids": [
            claim["candidate_id"] for claim in would_render_claims
        ],
        "selected_authorization_kinds": [
            claim["authorization_kind"] for claim in would_render_claims
        ],
        "selected_claim_sha256": [
            claim["claim_sha256"] for claim in would_render_claims
        ],
        "selected_citation_sha256": [
            citation["citation_sha256"] for citation in would_render_citations
        ],
        "selected_authorizations": selected_authorizations,
        "would_render_append_text_sha256": semantic_sha256(
            would_render_append_text
        ),
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
        "authorization_mode": authorization_mode,
        "render_suppressed": suppress_rendering,
        "append_text": append_text,
        "claims": rendered_claims,
        "citations": rendered_citations,
        "would_render_append_text": would_render_append_text,
        "would_render_claims": would_render_claims,
        "would_render_citations": would_render_citations,
        "selected_candidate_ids": trace_unsigned["selected_candidate_ids"],
        "semantic_trace": trace,
    }


def _validate_output_structure(output: Mapping[str, Any]) -> None:
    _closed(output, _OUTPUT_FIELDS, label="Phase B.1 output")
    if output.get("profile_version") != PROFILE_VERSION:
        raise Phase51B1CompactError("output profile version differs")
    suppressed = output.get("render_suppressed")
    if not isinstance(suppressed, bool):
        raise Phase51B1CompactError("output suppression flag differs")
    claims = output.get("claims")
    citations = output.get("citations")
    would_claims = output.get("would_render_claims")
    would_citations = output.get("would_render_citations")
    if any(
        not isinstance(value, list)
        for value in (claims, citations, would_claims, would_citations)
    ):
        raise Phase51B1CompactError("output claims/citations differ")
    if (
        len(would_claims) > MAX_CLAIMS
        or len(would_claims) != len(would_citations)
        or len(output.get("would_render_append_text") or "")
        > MAX_TOTAL_CHARACTERS
        or output.get("would_render_append_text")
        != "\n".join(str(claim.get("text") or "") for claim in would_claims)
        or output.get("selected_candidate_ids")
        != [claim.get("candidate_id") for claim in would_claims]
    ):
        raise Phase51B1CompactError("would-render output bounds differ")
    if suppressed:
        if output.get("append_text") != "" or claims or citations:
            raise Phase51B1CompactError("suppressed output rendered a claim")
    elif (
        claims != would_claims
        or citations != would_citations
        or output.get("append_text") != output.get("would_render_append_text")
    ):
        raise Phase51B1CompactError("rendered output differs from selection")
    for index, claim in enumerate(would_claims):
        _closed(claim, _CLAIM_FIELDS, label="output claim")
        _closed(
            claim.get("row_label_cell"),
            _OUTPUT_ROW_LABEL_FIELDS,
            label="output row_label_cell",
        )
        unsigned = {key: value for key, value in claim.items() if key != "claim_sha256"}
        if (
            claim.get("claim_sha256") != semantic_sha256(unsigned)
            or len(str(claim.get("text") or "")) > MAX_CLAIM_CHARACTERS
            or MASK in str(claim.get("text") or "")
        ):
            raise Phase51B1CompactError("output claim hash/text differs")
        selected_cells = claim.get("selected_cells")
        if (
            not isinstance(selected_cells, list)
            or not 1 <= len(selected_cells) <= MAX_CELLS_PER_CLAIM
            or any(
                cell.get("authorization_kind") != claim.get("authorization_kind")
                for cell in selected_cells
                if isinstance(cell, Mapping)
            )
        ):
            raise Phase51B1CompactError("output selected cells differ")
        expected_fragments: list[str] = []
        expected_literals: Counter[str] = Counter()
        for cell in selected_cells:
            if not isinstance(cell, Mapping):
                raise Phase51B1CompactError("output selected cell is invalid")
            _closed(cell, _SELECTED_CELL_FIELDS, label="output selected cell")
            raw = str(cell.get("raw_value") or "")
            authorization = cell.get("authorization")
            if (
                cell.get("raw_value_sha256") != sha256_text(raw)
                or cell.get("authorization_kind")
                not in {AUTHORIZATION_KIND_NUMERIC, AUTHORIZATION_KIND_LITERAL}
                or not _HEX64_RE.fullmatch(
                    str(cell.get("authorization_sha256") or "")
                )
                or not isinstance(authorization, Mapping)
                or cell.get("authorization_id")
                != authorization.get("authorization_id")
            ):
                raise Phase51B1CompactError("output cell provenance differs")
            if cell["authorization_kind"] == AUTHORIZATION_KIND_LITERAL:
                _validate_literal_shape_and_hash(authorization)
                if (
                    cell["authorization_sha256"]
                    != authorization["authorization_sha256"]
                    or authorization["cell_binding"]["source_cell_literal"] != raw
                ):
                    raise Phase51B1CompactError(
                        "output literal authorization closure differs"
                    )
            elif (
                set(authorization) != _v3._AUTH_FIELDS  # type: ignore[attr-defined]
                or cell["authorization_sha256"] != semantic_sha256(authorization)
                or authorization.get("allowed_renderings") != [raw]
            ):
                raise Phase51B1CompactError(
                    "output NumericAuthorization closure differs"
                )
            expected_literals[raw] += 1
            expected_fragments.extend(
                numeric_fragments(
                    _render_raw_value(raw, str(cell.get("unit") or ""))
                )
            )
        if any(
            claim["text"].count(literal) != count
            for literal, count in expected_literals.items()
        ):
            raise Phase51B1CompactError("output literal occurrences differ")
        if Counter(numeric_fragments(claim["text"])) != Counter(expected_fragments):
            raise Phase51B1CompactError("output numeric fragments differ")
        citation = would_citations[index]
        _closed(citation, _CITATION_FIELDS, label="output citation")
        citation_unsigned = {
            key: value for key, value in citation.items()
            if key != "citation_sha256"
        }
        if (
            citation.get("citation_sha256") != semantic_sha256(citation_unsigned)
            or citation != _citation(claim)
        ):
            raise Phase51B1CompactError("output citation differs")
    trace = output.get("semantic_trace")
    if not isinstance(trace, Mapping):
        raise Phase51B1CompactError("output trace differs")
    _closed(trace, _TRACE_FIELDS, label="output trace")
    _closed(
        trace.get("semantic_input"),
        {"question"},
        label="output trace semantic_input",
    )
    selected_authorizations = trace.get("selected_authorizations")
    if not isinstance(selected_authorizations, list):
        raise Phase51B1CompactError("output trace authorizations differ")
    for value in selected_authorizations:
        record = _closed(
            value,
            _TRACE_AUTHORIZATION_FIELDS,
            label="output trace authorization",
        )
        cells = record.get("cells")
        if not isinstance(cells, list):
            raise Phase51B1CompactError("output trace authorization cells differ")
        for cell in cells:
            _closed(
                cell,
                _SELECTED_CELL_FIELDS,
                label="output trace selected cell",
            )
    trace_unsigned = {
        key: value for key, value in trace.items()
        if key != "semantic_trace_sha256"
    }
    if (
        trace.get("semantic_trace_sha256") != semantic_sha256(trace_unsigned)
        or trace.get("profile_version") != PROFILE_VERSION
        or trace.get("composer_version") != COMPOSER_VERSION
        or trace.get("authorization_mode") != output.get("authorization_mode")
        or trace.get("selected_claim_sha256")
        != [claim["claim_sha256"] for claim in would_claims]
        or trace.get("selected_citation_sha256")
        != [citation["citation_sha256"] for citation in would_citations]
        or trace.get("selected_authorization_kinds")
        != [claim["authorization_kind"] for claim in would_claims]
        or selected_authorizations
        != [
            {
                "candidate_id": claim["candidate_id"],
                "claim_sha256": claim["claim_sha256"],
                "authorization_kind": claim["authorization_kind"],
                "cells": claim["selected_cells"],
            }
            for claim in would_claims
        ]
        or trace.get("selected_candidate_ids")
        != output.get("selected_candidate_ids")
        or trace.get("would_render_append_text_sha256")
        != semantic_sha256(output.get("would_render_append_text"))
        or trace.get("would_render_claim_count") != len(would_claims)
        or trace.get("rendered_claim_count") != len(claims)
        or trace.get("append_characters") != len(str(output.get("append_text") or ""))
        or trace.get("render_suppressed") is not suppressed
    ):
        raise Phase51B1CompactError("output trace closure differs")


def compose_phase51_b1_claims(
    *,
    question: str,
    corpus_id: str,
    document_id: str,
    candidates: Sequence[Mapping[str, Any]],
    base_answer: str = "",
    authorization_mode: str = AUTHORIZATION_MODE_EXISTING_PLUS_LITERAL,
    enable_route_gate: bool = True,
    enable_complementarity: bool = True,
    suppress_rendering: bool = False,
) -> dict[str, Any]:
    """Select at most two compact claims and optionally suppress only rendering."""

    output = _compose_impl(
        question=question,
        corpus_id=corpus_id,
        document_id=document_id,
        candidates=candidates,
        base_answer=base_answer,
        authorization_mode=authorization_mode,
        enable_route_gate=enable_route_gate,
        enable_complementarity=enable_complementarity,
        suppress_rendering=suppress_rendering,
    )
    _validate_output_structure(output)
    return output


def validate_phase51_b1_output(
    output: Mapping[str, Any],
    *,
    question: str,
    corpus_id: str,
    document_id: str,
    candidates: Sequence[Mapping[str, Any]],
    base_answer: str = "",
    authorization_mode: str = AUTHORIZATION_MODE_EXISTING_PLUS_LITERAL,
    enable_route_gate: bool = True,
    enable_complementarity: bool = True,
    suppress_rendering: bool = False,
) -> None:
    """Authoritatively validate output by recomposing from frozen inputs."""

    _validate_output_structure(output)
    expected = _compose_impl(
        question=question,
        corpus_id=corpus_id,
        document_id=document_id,
        candidates=candidates,
        base_answer=base_answer,
        authorization_mode=authorization_mode,
        enable_route_gate=enable_route_gate,
        enable_complementarity=enable_complementarity,
        suppress_rendering=suppress_rendering,
    )
    if dict(output) != expected:
        raise Phase51B1CompactError("output differs from authoritative recomposition")


__all__ = [
    "AUTHORIZATION_KIND_LITERAL",
    "AUTHORIZATION_KIND_NUMERIC",
    "AUTHORIZATION_MODE_EXISTING_ONLY",
    "AUTHORIZATION_MODE_EXISTING_PLUS_LITERAL",
    "COMPOSER_VERSION",
    "MASK",
    "MAX_CELLS_PER_CLAIM",
    "MAX_CLAIMS",
    "MAX_CLAIM_CHARACTERS",
    "MAX_TOTAL_CHARACTERS",
    "PROFILE_VERSION",
    "Phase51B1CompactError",
    "TRACE_SCHEMA",
    "compose_phase51_b1_claims",
    "validate_literal_authorization",
    "validate_phase51_b1_output",
]
