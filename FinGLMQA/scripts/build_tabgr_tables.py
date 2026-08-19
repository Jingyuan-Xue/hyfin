#!/usr/bin/env python3
"""Build Phase 4 TabGR-compatible table artifacts from Phase 3 table blocks."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO


SCHEMA_TABGR_TABLE = "finglmqa.phase4.tabgr_table_corpus.v1"
SCHEMA_TABLE_CELL = "finglmqa.phase4.table_cell.v1"
SCHEMA_TABGR_INDEX = "finglmqa.phase4.tabgr_table_index.v1"
SCHEMA_REPORT = "finglmqa.phase4.tabgr_parse_report.v1"
ADAPTER_VERSION = "phase4-tabgr-structurer-v1"

EXPECTED_INPUT_TABLE_BLOCKS = 43569
EXPECTED_READY_TABLES = 43540
EXPECTED_NON_READY_TABLES = 29

CELL_ROLES = {"header", "row_header", "data"}

TABGR_TABLE_REQUIRED_FIELDS = [
    "schema_version",
    "document_id",
    "table_id",
    "table_index",
    "header",
    "rows",
    "matrix",
    "matrix_text",
    "edge_list",
    "source_markdown",
    "line_range",
    "caption",
    "nearby_text",
    "unit_hint",
    "section_path",
    "raw_markdown_sha1",
    "metadata",
    "stats",
    "tabgr_ready",
]

TABLE_CELL_REQUIRED_FIELDS = [
    "schema_version",
    "document_id",
    "table_id",
    "row_index",
    "col_index",
    "cell_role",
    "column_label",
    "row_label",
    "raw_value",
    "source_markdown",
    "line_range",
    "unit_hint",
    "section_path",
    "metadata",
]

TABGR_INDEX_REQUIRED_FIELDS = [
    "schema_version",
    "document_id",
    "table_id",
    "table_index",
    "stock_code",
    "stock_symbol",
    "stock_name",
    "company_full",
    "report_year",
    "source_markdown",
    "resolved_source_path",
    "line_range",
    "raw_markdown_sha1",
    "row_count",
    "column_count",
    "non_empty_cell_count",
    "header_cell_count",
    "row_header_cell_count",
    "data_cell_count",
    "edge_count",
    "tabgr_ready",
    "parse_status",
    "failure_reason",
]


def workspace_root() -> Path:
    return Path(__file__).resolve().parents[1]


def rel(path: Path, root: Path) -> str:
    try:
        return path.resolve(strict=False).relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
        fh.write("\n")


def open_jsonl(path: Path) -> TextIO:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("w", encoding="utf-8")


def write_jsonl_row(fh: TextIO, row: dict[str, Any]) -> None:
    json.dump(row, fh, ensure_ascii=False, separators=(",", ":"))
    fh.write("\n")


def short_snippet(value: object, limit: int = 500) -> str:
    text = "" if value is None else str(value)
    return re.sub(r"\s+", " ", text).strip()[:limit]


def json_type_name(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int) and not isinstance(value, bool):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def normalize_line_range(value: object) -> list[int | None]:
    if isinstance(value, list) and len(value) >= 2:
        start, end = value[0], value[1]
    elif isinstance(value, tuple) and len(value) >= 2:
        start, end = value[0], value[1]
    else:
        return [None, None]

    def to_int(item: object) -> int | None:
        if isinstance(item, bool):
            return None
        if isinstance(item, int):
            return item
        try:
            return int(str(item))
        except (TypeError, ValueError):
            return None

    return [to_int(start), to_int(end)]


def normalize_matrix(value: object) -> tuple[list[list[str]], list[str]]:
    errors: list[str] = []
    if not isinstance(value, list):
        return [], [f"matrix_not_array:{json_type_name(value)}"]

    matrix: list[list[str]] = []
    for row_index, row in enumerate(value):
        if not isinstance(row, list):
            errors.append(f"matrix_row_{row_index}_not_array:{json_type_name(row)}")
            continue
        matrix.append([normalize_text(cell) for cell in row])

    width = max((len(row) for row in matrix), default=0)
    if width:
        matrix = [row + [""] * (width - len(row)) for row in matrix]
    return matrix, errors


def matrix_text(matrix: list[list[str]]) -> str:
    return "\n".join("\t".join(row) for row in matrix)


def non_empty_cells(matrix: list[list[str]]) -> int:
    return sum(1 for row in matrix for value in row if value)


def header_labels(matrix: list[list[str]]) -> list[str]:
    if not matrix:
        return []
    return [value if value else f"Col{index + 1}" for index, value in enumerate(matrix[0])]


def first_non_empty_cell(row: list[str]) -> tuple[int | None, str]:
    for index, value in enumerate(row):
        if value:
            return index, value
    return None, ""


def fiscal_year(report_year: object) -> str | None:
    year = normalize_text(report_year)
    match = re.search(r"(?:19|20)\d{2}", year)
    return f"FY{match.group(0)}" if match else None


def table_metadata(block: dict[str, Any]) -> dict[str, Any]:
    source_metadata = block.get("metadata") if isinstance(block.get("metadata"), dict) else {}
    metadata = dict(source_metadata)
    metadata.update({
        "doc_id": block.get("document_id"),
        "document_id": block.get("document_id"),
        "table_id": block.get("table_id"),
        "table_index": block.get("table_index"),
        "ticker": block.get("stock_symbol") or source_metadata.get("ticker"),
        "stock_code": block.get("stock_code") or source_metadata.get("stock_code"),
        "stock_symbol": block.get("stock_symbol"),
        "stock_name": block.get("stock_name") or source_metadata.get("stock_name"),
        "company": block.get("company_full") or source_metadata.get("company"),
        "company_full": block.get("company_full") or source_metadata.get("company_full"),
        "report_year": block.get("report_year") or source_metadata.get("report_year"),
        "source_markdown": block.get("source_markdown") or source_metadata.get("source_markdown"),
        "raw_markdown_sha1": block.get("raw_markdown_sha1"),
    })
    fy = fiscal_year(metadata.get("report_year"))
    if fy:
        metadata["fiscal_year"] = fy
    return {key: value for key, value in metadata.items() if value is not None}


def cell_metadata(block: dict[str, Any]) -> dict[str, Any]:
    metadata = {
        "doc_id": block.get("document_id"),
        "document_id": block.get("document_id"),
        "table_id": block.get("table_id"),
        "table_index": block.get("table_index"),
        "stock_code": block.get("stock_code"),
        "stock_symbol": block.get("stock_symbol"),
        "stock_name": block.get("stock_name"),
        "company_full": block.get("company_full"),
        "report_year": block.get("report_year"),
        "raw_markdown_sha1": block.get("raw_markdown_sha1"),
    }
    fy = fiscal_year(block.get("report_year"))
    if fy:
        metadata["fiscal_year"] = fy
    return {key: value for key, value in metadata.items() if value is not None}


def build_structured_table(block: dict[str, Any], matrix: list[list[str]]) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, int]]:
    labels = header_labels(matrix)
    rows = matrix[1:] if len(matrix) > 1 else []
    edge_list: list[str] = []
    cell_rows: list[dict[str, Any]] = []
    role_counts: Counter[str] = Counter()

    source_markdown = block.get("source_markdown")
    line_range = normalize_line_range(block.get("line_range"))
    section_path = block.get("section_path") if isinstance(block.get("section_path"), list) else []
    metadata = cell_metadata(block)

    for row_index, row in enumerate(matrix):
        first_col_index, first_value = first_non_empty_cell(row)
        for col_index, value in enumerate(row):
            if not value:
                continue
            if row_index == 0:
                role = "header"
                row_label = ""
            elif col_index == first_col_index:
                role = "row_header"
                row_label = value
            else:
                role = "data"
                row_label = first_value
            role_counts[role] += 1
            cell_rows.append({
                "schema_version": SCHEMA_TABLE_CELL,
                "adapter_version": ADAPTER_VERSION,
                "document_id": block.get("document_id"),
                "table_id": block.get("table_id"),
                "table_index": block.get("table_index"),
                "row_index": row_index,
                "col_index": col_index,
                "cell_role": role,
                "column_label": labels[col_index] if col_index < len(labels) else f"Col{col_index + 1}",
                "row_label": row_label,
                "raw_value": value,
                "source_markdown": source_markdown,
                "line_range": line_range,
                "unit_hint": block.get("unit_hint"),
                "section_path": section_path,
                "metadata": metadata,
            })
            if row_index > 0:
                edge_list.append(
                    f"(row{row_index}; {labels[col_index] if col_index < len(labels) else f'Col{col_index + 1}'}; {value})"
                )

    source_title = block.get("source_title") or f"{block.get('stock_symbol') or block.get('stock_code') or block.get('document_id')} table {block.get('table_index')}"
    summary_lines = [f"## Table evidence: {source_title}", *edge_list]
    stats = {
        "matrix_rows": len(matrix),
        "matrix_cols": max((len(row) for row in matrix), default=0),
        "row_count": len(rows),
        "column_count": len(labels),
        "non_empty_cell_count": sum(role_counts.values()),
        "header_cell_count": role_counts["header"],
        "row_header_cell_count": role_counts["row_header"],
        "data_cell_count": role_counts["data"],
        "edge_count": len(edge_list),
        "source_rows": int((block.get("stats") or {}).get("source_rows") or 0)
        if isinstance(block.get("stats"), dict)
        else 0,
    }

    corpus_record = {
        "schema_version": SCHEMA_TABGR_TABLE,
        "adapter_version": ADAPTER_VERSION,
        "document_id": block.get("document_id"),
        "doc_id": block.get("document_id"),
        "table_id": block.get("table_id"),
        "table_index": block.get("table_index"),
        "stock_code": block.get("stock_code"),
        "stock_symbol": block.get("stock_symbol"),
        "stock_name": block.get("stock_name"),
        "company_full": block.get("company_full"),
        "report_year": block.get("report_year"),
        "header": labels,
        "rows": rows,
        "matrix": matrix,
        "matrix_text": matrix_text(matrix),
        "edge_list": edge_list,
        "summary_md": "\n".join(summary_lines),
        "table": {
            "name": block.get("table_id"),
            "header": labels,
            "rows": rows,
        },
        "table_caption": block.get("caption") or source_title,
        "source_title": source_title,
        "source_markdown": block.get("source_markdown"),
        "resolved_source_path": block.get("resolved_source_path"),
        "line_range": line_range,
        "caption": block.get("caption"),
        "nearby_text": block.get("nearby_text") or "",
        "unit_hint": block.get("unit_hint"),
        "section_path": section_path,
        "raw_markdown_sha1": block.get("raw_markdown_sha1"),
        "fingerprint": block.get("raw_markdown_sha1"),
        "content_hash": block.get("content_hash"),
        "metadata": table_metadata(block),
        "stats": stats,
        "parse_status": block.get("parse_status"),
        "failure_reason": None,
        "tabgr_ready": True,
    }
    return corpus_record, cell_rows, stats


def ready_failure_reason(block: dict[str, Any], matrix: list[list[str]], matrix_errors: list[str]) -> str | None:
    parse_status = normalize_text(block.get("parse_status")) or "unknown"
    if parse_status != "ok":
        return normalize_text(block.get("failure_reason")) or parse_status
    if matrix_errors:
        return "invalid_matrix:" + ",".join(matrix_errors[:5])
    if not matrix:
        return "invalid_matrix:empty_matrix"
    if not matrix[0]:
        return "invalid_matrix:empty_header_row"
    if non_empty_cells(matrix) == 0:
        return "invalid_matrix:no_non_empty_cells"
    return None


def build_index_record(
    block: dict[str, Any],
    *,
    matrix: list[list[str]],
    stats: dict[str, int] | None,
    tabgr_ready: bool,
    failure_reason: str | None,
) -> dict[str, Any]:
    source_stats = stats or {}
    row_count = int(source_stats.get("row_count", max(len(matrix) - 1, 0)))
    column_count = int(source_stats.get("column_count", max((len(row) for row in matrix), default=0)))
    non_empty_count = int(source_stats.get("non_empty_cell_count", non_empty_cells(matrix) if matrix else 0))

    return {
        "schema_version": SCHEMA_TABGR_INDEX,
        "adapter_version": ADAPTER_VERSION,
        "document_id": block.get("document_id"),
        "doc_id": block.get("document_id"),
        "table_id": block.get("table_id"),
        "table_index": block.get("table_index"),
        "stock_code": block.get("stock_code"),
        "stock_symbol": block.get("stock_symbol"),
        "stock_name": block.get("stock_name"),
        "company_full": block.get("company_full"),
        "report_year": block.get("report_year"),
        "fiscal_year": fiscal_year(block.get("report_year")),
        "source_markdown": block.get("source_markdown"),
        "resolved_source_path": block.get("resolved_source_path"),
        "line_range": normalize_line_range(block.get("line_range")),
        "section_path": block.get("section_path") if isinstance(block.get("section_path"), list) else [],
        "caption": block.get("caption"),
        "unit_hint": block.get("unit_hint"),
        "raw_markdown_sha1": block.get("raw_markdown_sha1"),
        "content_hash": block.get("content_hash"),
        "row_count": row_count,
        "column_count": column_count,
        "non_empty_cell_count": non_empty_count,
        "header_cell_count": int(source_stats.get("header_cell_count", 0)),
        "row_header_cell_count": int(source_stats.get("row_header_cell_count", 0)),
        "data_cell_count": int(source_stats.get("data_cell_count", 0)),
        "edge_count": int(source_stats.get("edge_count", 0)),
        "tabgr_ready": tabgr_ready,
        "parse_status": block.get("parse_status") or "unknown",
        "failure_reason": failure_reason,
        "metadata": table_metadata(block),
    }


def failure_sample(block: dict[str, Any], failure_reason: str | None) -> dict[str, Any]:
    return {
        "document_id": block.get("document_id"),
        "table_id": block.get("table_id"),
        "table_index": block.get("table_index"),
        "parse_status": block.get("parse_status") or "unknown",
        "failure_reason": failure_reason,
        "line_range": normalize_line_range(block.get("line_range")),
        "raw_markdown_sha1": block.get("raw_markdown_sha1"),
        "caption": block.get("caption"),
        "snippet": short_snippet(block.get("raw_markdown")),
    }


def validate_required(record: dict[str, Any], required: list[str]) -> list[str]:
    return [field for field in required if field not in record]


def validate_table_record(record: dict[str, Any]) -> list[str]:
    errors = [f"missing:{field}" for field in validate_required(record, TABGR_TABLE_REQUIRED_FIELDS)]
    if not isinstance(record.get("document_id"), str) or not record.get("document_id"):
        errors.append("document_id_not_string")
    if not isinstance(record.get("table_id"), str) or not record.get("table_id"):
        errors.append("table_id_not_string")
    if not isinstance(record.get("table_index"), int):
        errors.append("table_index_not_integer")
    if not isinstance(record.get("header"), list):
        errors.append("header_not_array")
    if not isinstance(record.get("rows"), list):
        errors.append("rows_not_array")
    if not isinstance(record.get("matrix"), list):
        errors.append("matrix_not_array")
    if not isinstance(record.get("edge_list"), list):
        errors.append("edge_list_not_array")
    if not isinstance(record.get("metadata"), dict):
        errors.append("metadata_not_object")
    if not isinstance(record.get("stats"), dict):
        errors.append("stats_not_object")
    if record.get("tabgr_ready") is not True:
        errors.append("tabgr_ready_not_true")

    header = record.get("header") if isinstance(record.get("header"), list) else []
    rows = record.get("rows") if isinstance(record.get("rows"), list) else []
    matrix = record.get("matrix") if isinstance(record.get("matrix"), list) else []
    if matrix:
        width = len(matrix[0]) if isinstance(matrix[0], list) else None
        if width != len(header):
            errors.append("header_width_mismatch")
        if len(matrix) != len(rows) + 1:
            errors.append("matrix_rows_mismatch")
        for row in matrix:
            if not isinstance(row, list):
                errors.append("matrix_row_not_array")
                break
            if width is not None and len(row) != width:
                errors.append("matrix_not_rectangular")
                break
    return errors


def validate_cell_record(record: dict[str, Any]) -> list[str]:
    errors = [f"missing:{field}" for field in validate_required(record, TABLE_CELL_REQUIRED_FIELDS)]
    if not isinstance(record.get("document_id"), str) or not record.get("document_id"):
        errors.append("document_id_not_string")
    if not isinstance(record.get("table_id"), str) or not record.get("table_id"):
        errors.append("table_id_not_string")
    if not isinstance(record.get("row_index"), int) or record.get("row_index", -1) < 0:
        errors.append("row_index_not_nonnegative_integer")
    if not isinstance(record.get("col_index"), int) or record.get("col_index", -1) < 0:
        errors.append("col_index_not_nonnegative_integer")
    if record.get("cell_role") not in CELL_ROLES:
        errors.append("invalid_cell_role")
    if not isinstance(record.get("column_label"), str) or not record.get("column_label"):
        errors.append("column_label_not_string")
    if not isinstance(record.get("row_label"), str):
        errors.append("row_label_not_string")
    if not isinstance(record.get("raw_value"), str) or not record.get("raw_value"):
        errors.append("raw_value_not_string")
    if not isinstance(record.get("metadata"), dict):
        errors.append("metadata_not_object")
    return errors


def validate_index_record(record: dict[str, Any]) -> list[str]:
    errors = [f"missing:{field}" for field in validate_required(record, TABGR_INDEX_REQUIRED_FIELDS)]
    for field in [
        "table_index",
        "row_count",
        "column_count",
        "non_empty_cell_count",
        "header_cell_count",
        "row_header_cell_count",
        "data_cell_count",
        "edge_count",
    ]:
        if not isinstance(record.get(field), int) or record.get(field, -1) < 0:
            errors.append(f"{field}_not_nonnegative_integer")
    if not isinstance(record.get("document_id"), str) or not record.get("document_id"):
        errors.append("document_id_not_string")
    if not isinstance(record.get("table_id"), str) or not record.get("table_id"):
        errors.append("table_id_not_string")
    if not isinstance(record.get("tabgr_ready"), bool):
        errors.append("tabgr_ready_not_boolean")
    if record.get("tabgr_ready"):
        expected_edges = int(record.get("row_header_cell_count") or 0) + int(record.get("data_cell_count") or 0)
        if int(record.get("edge_count") or 0) != expected_edges:
            errors.append("edge_count_invariant_failed")
        expected_non_empty = (
            int(record.get("header_cell_count") or 0)
            + int(record.get("row_header_cell_count") or 0)
            + int(record.get("data_cell_count") or 0)
        )
        if int(record.get("non_empty_cell_count") or 0) != expected_non_empty:
            errors.append("non_empty_count_invariant_failed")
    return errors


def schema_for_table() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": SCHEMA_TABGR_TABLE,
        "type": "object",
        "required": TABGR_TABLE_REQUIRED_FIELDS,
        "additionalProperties": True,
        "properties": {
            "schema_version": {"const": SCHEMA_TABGR_TABLE},
            "document_id": {"type": "string"},
            "table_id": {"type": "string"},
            "table_index": {"type": "integer", "minimum": 1},
            "header": {"type": "array", "items": {"type": "string"}},
            "rows": {"type": "array", "items": {"type": "array", "items": {"type": "string"}}},
            "matrix": {"type": "array", "items": {"type": "array", "items": {"type": "string"}}},
            "matrix_text": {"type": "string"},
            "edge_list": {"type": "array", "items": {"type": "string"}},
            "source_markdown": {"type": ["string", "null"]},
            "line_range": {"type": "array", "prefixItems": [{"type": ["integer", "null"]}, {"type": ["integer", "null"]}]},
            "caption": {"type": ["string", "null"]},
            "nearby_text": {"type": "string"},
            "unit_hint": {"type": ["string", "null"]},
            "section_path": {"type": "array", "items": {"type": "string"}},
            "raw_markdown_sha1": {"type": "string"},
            "metadata": {"type": "object"},
            "stats": {"type": "object"},
            "tabgr_ready": {"type": "boolean"},
        },
    }


def schema_for_cell() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": SCHEMA_TABLE_CELL,
        "type": "object",
        "required": TABLE_CELL_REQUIRED_FIELDS,
        "additionalProperties": True,
        "properties": {
            "schema_version": {"const": SCHEMA_TABLE_CELL},
            "document_id": {"type": "string"},
            "table_id": {"type": "string"},
            "row_index": {"type": "integer", "minimum": 0},
            "col_index": {"type": "integer", "minimum": 0},
            "cell_role": {"enum": sorted(CELL_ROLES)},
            "column_label": {"type": "string"},
            "row_label": {"type": "string"},
            "raw_value": {"type": "string"},
            "source_markdown": {"type": ["string", "null"]},
            "line_range": {"type": "array", "prefixItems": [{"type": ["integer", "null"]}, {"type": ["integer", "null"]}]},
            "unit_hint": {"type": ["string", "null"]},
            "section_path": {"type": "array", "items": {"type": "string"}},
            "metadata": {"type": "object"},
        },
    }


def schema_for_index() -> dict[str, Any]:
    count_schema = {"type": "integer", "minimum": 0}
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": SCHEMA_TABGR_INDEX,
        "type": "object",
        "required": TABGR_INDEX_REQUIRED_FIELDS,
        "additionalProperties": True,
        "properties": {
            "schema_version": {"const": SCHEMA_TABGR_INDEX},
            "document_id": {"type": "string"},
            "table_id": {"type": "string"},
            "table_index": {"type": "integer", "minimum": 1},
            "stock_code": {"type": ["string", "null"]},
            "stock_symbol": {"type": ["string", "null"]},
            "stock_name": {"type": ["string", "null"]},
            "company_full": {"type": ["string", "null"]},
            "report_year": {"type": ["string", "null"]},
            "source_markdown": {"type": ["string", "null"]},
            "resolved_source_path": {"type": ["string", "null"]},
            "line_range": {"type": "array", "prefixItems": [{"type": ["integer", "null"]}, {"type": ["integer", "null"]}]},
            "raw_markdown_sha1": {"type": "string"},
            "row_count": count_schema,
            "column_count": count_schema,
            "non_empty_cell_count": count_schema,
            "header_cell_count": count_schema,
            "row_header_cell_count": count_schema,
            "data_cell_count": count_schema,
            "edge_count": count_schema,
            "tabgr_ready": {"type": "boolean"},
            "parse_status": {"type": "string"},
            "failure_reason": {"type": ["string", "null"]},
        },
    }


def write_schemas(schema_dir: Path) -> dict[str, str]:
    outputs = {
        "tabgr_table_corpus_schema": schema_dir / "tabgr_table_corpus.schema.json",
        "table_cells_schema": schema_dir / "table_cells.schema.json",
        "tabgr_table_index_schema": schema_dir / "tabgr_table_index.schema.json",
    }
    write_json(outputs["tabgr_table_corpus_schema"], schema_for_table())
    write_json(outputs["table_cells_schema"], schema_for_cell())
    write_json(outputs["tabgr_table_index_schema"], schema_for_index())
    return {key: path.as_posix() for key, path in outputs.items()}


def has_merged_cells(block: dict[str, Any]) -> bool:
    spans = block.get("cell_spans")
    if not isinstance(spans, list):
        return False
    for cell in spans:
        if not isinstance(cell, dict):
            continue
        try:
            rowspan = int(cell.get("rowspan") or 1)
            colspan = int(cell.get("colspan") or 1)
        except (TypeError, ValueError):
            continue
        if rowspan > 1 or colspan > 1:
            return True
    return False


def year_labels(labels: list[str]) -> list[str]:
    years: list[str] = []
    seen: set[str] = set()
    for label in labels:
        for match in re.findall(r"(?<!\d)(?:201\d|202[0-6])(?!\d)", label):
            if match not in seen:
                seen.add(match)
                years.append(match)
    return years


def has_financial_context(block: dict[str, Any]) -> bool:
    parts: list[str] = []
    for key in ("caption", "nearby_text"):
        value = block.get(key)
        if value:
            parts.append(str(value))
    section_path = block.get("section_path")
    if isinstance(section_path, list):
        parts.extend(str(item) for item in section_path)
    semantic_tags = block.get("semantic_tags")
    if isinstance(semantic_tags, list):
        parts.extend(str(item) for item in semantic_tags)
    haystack = "\n".join(parts)
    return any(
        term in haystack
        for term in ("财务", "会计", "资产", "利润", "现金", "收入", "营业", "主要会计数据", "主要财务指标", "financial")
    )


def phase_report_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    validation = report["validation"]
    outputs = report["outputs"]
    lines = [
        "# Phase 4 Report",
        "",
        "## Date",
        report["generated_at_utc"],
        "",
        "## Environment",
        f"- Workspace: `{report['workspace_root']}`",
        f"- Adapter: `{ADAPTER_VERSION}`",
        "",
        "## Inputs",
    ]
    for label, path in report["inputs"].items():
        lines.append(f"- {label}: `{path}`")
    lines.extend(["", "## Generated Artifacts"])
    for label, path in outputs.items():
        lines.append(f"- {label}: `{path}`")
    lines.extend([
        "",
        "## Verification Results",
        f"- Input table blocks: {summary['input_table_blocks']}",
        f"- Ready tables: {summary['ready_tables']}",
        f"- Non-ready tables: {summary['non_ready_tables']}",
        f"- Corpus JSONL rows: {summary['corpus_rows']}",
        f"- Cell JSONL rows: {summary['cell_rows']}",
        f"- Index JSONL rows: {summary['index_rows']}",
        f"- Failure sample rows: {summary['failure_sample_rows']}",
        f"- Parse status counts: `{summary['parse_status_counts']}`",
        f"- Failure reason counts: `{summary['failure_reason_counts']}`",
        f"- Edge count: {summary['edge_count']}",
        f"- Header cells: {summary['header_cell_count']}",
        f"- Row-header cells: {summary['row_header_cell_count']}",
        f"- Data cells: {summary['data_cell_count']}",
        f"- Edge invariant failures: {validation['edge_invariant_failure_count']}",
        f"- Validation error count: {validation['error_count']}",
        "",
        "## Spot Checks",
    ])
    for key, sample in report["spot_checks"].items():
        lines.append(f"- {key}: `{sample}`")
    lines.extend(["", "## Issues Encountered"])
    if validation["errors"]:
        for item in validation["errors"][:20]:
            lines.append(f"- {item}")
    else:
        lines.append("- None.")
    lines.extend(["", "## Decision", "- continue", ""])
    return "\n".join(lines)


def main() -> int:
    root = workspace_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=root / "data/corpus_package/table_blocks.jsonl")
    parser.add_argument("--output-dir", type=Path, default=root / "data/corpus_package")
    parser.add_argument("--index-dir", type=Path, default=root / "data/indexes")
    parser.add_argument("--schema-dir", type=Path, default=root / "data/schemas")
    parser.add_argument("--run-dir", type=Path, default=root / "runs/phase_04")
    parser.add_argument("--strict-counts", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    generated_at = datetime.now(UTC).isoformat(timespec="seconds")
    corpus_path = args.output_dir / "tabgr_table_corpus.jsonl"
    cells_path = args.output_dir / "table_cells.jsonl"
    index_path = args.index_dir / "tabgr_table_index.jsonl"
    reports_dir = args.run_dir / "reports"
    failure_samples_path = reports_dir / "tabgr_failure_samples.jsonl"
    parse_report_path = args.run_dir / "tabgr_parse_report.json"
    phase_report_path = args.run_dir / "phase_04_report.md"

    schema_outputs = write_schemas(args.schema_dir)

    counters: Counter[str] = Counter()
    parse_status_counts: Counter[str] = Counter()
    failure_reason_counts: Counter[str] = Counter()
    cell_role_counts: Counter[str] = Counter()
    validation_errors: list[str] = []
    edge_invariant_failures: list[dict[str, Any]] = []

    all_failure_samples: list[dict[str, Any]] = []
    sampled_failures_by_reason: dict[str, list[dict[str, Any]]] = defaultdict(list)

    first_table_sample: dict[str, Any] | None = None
    first_ready_sample: dict[str, Any] | None = None
    merged_cell_sample: dict[str, Any] | None = None
    year_column_sample: dict[str, Any] | None = None
    non_ready_sample: dict[str, Any] | None = None

    with (
        args.input.open("r", encoding="utf-8") as input_fh,
        open_jsonl(corpus_path) as corpus_fh,
        open_jsonl(cells_path) as cells_fh,
        open_jsonl(index_path) as index_fh,
    ):
        for line_no, line in enumerate(input_fh, start=1):
            if not line.strip():
                continue
            counters["input_table_blocks"] += 1
            try:
                block = json.loads(line)
            except json.JSONDecodeError as exc:
                validation_errors.append(f"line {line_no}: invalid_json:{exc}")
                continue
            if not isinstance(block, dict):
                validation_errors.append(f"line {line_no}: record_not_object:{json_type_name(block)}")
                continue

            parse_status = normalize_text(block.get("parse_status")) or "unknown"
            parse_status_counts[parse_status] += 1
            matrix, matrix_errors = normalize_matrix(block.get("matrix"))
            failure_reason = ready_failure_reason(block, matrix, matrix_errors)
            tabgr_ready = failure_reason is None

            if first_table_sample is None:
                first_table_sample = {
                    "table_id": block.get("table_id"),
                    "raw_markdown_sha1": block.get("raw_markdown_sha1"),
                    "parse_status": parse_status,
                }

            if tabgr_ready:
                counters["ready_tables"] += 1
                corpus_record, cell_rows, stats = build_structured_table(block, matrix)
                table_errors = validate_table_record(corpus_record)
                if table_errors:
                    validation_errors.append(
                        f"{block.get('table_id')}: table_validation:{','.join(table_errors[:10])}"
                    )
                write_jsonl_row(corpus_fh, corpus_record)
                counters["corpus_rows"] += 1

                for cell_row in cell_rows:
                    cell_errors = validate_cell_record(cell_row)
                    if cell_errors:
                        validation_errors.append(
                            f"{block.get('table_id')}[{cell_row.get('row_index')},{cell_row.get('col_index')}]: cell_validation:{','.join(cell_errors[:10])}"
                        )
                    write_jsonl_row(cells_fh, cell_row)
                    counters["cell_rows"] += 1
                    cell_role_counts[str(cell_row["cell_role"])] += 1

                expected_edges = stats["row_header_cell_count"] + stats["data_cell_count"]
                if stats["edge_count"] != expected_edges:
                    failure = {
                        "table_id": block.get("table_id"),
                        "edge_count": stats["edge_count"],
                        "expected_edges": expected_edges,
                    }
                    if len(edge_invariant_failures) < 50:
                        edge_invariant_failures.append(failure)
                    validation_errors.append(f"{block.get('table_id')}: edge_count_invariant_failed")

                counters["edge_count"] += stats["edge_count"]
                counters["header_cell_count"] += stats["header_cell_count"]
                counters["row_header_cell_count"] += stats["row_header_cell_count"]
                counters["data_cell_count"] += stats["data_cell_count"]

                if first_ready_sample is None:
                    first_ready_sample = {
                        "table_id": block.get("table_id"),
                        "raw_markdown_sha1": block.get("raw_markdown_sha1"),
                        "edge_count": stats["edge_count"],
                        "first_edge": corpus_record["edge_list"][0] if corpus_record["edge_list"] else None,
                    }

                if merged_cell_sample is None and has_merged_cells(block):
                    merged_cell_sample = {
                        "table_id": block.get("table_id"),
                        "raw_markdown_sha1": block.get("raw_markdown_sha1"),
                        "matrix_rows": stats["matrix_rows"],
                        "matrix_cols": stats["matrix_cols"],
                        "header": corpus_record["header"][:8],
                    }

                labels_years = year_labels(corpus_record["header"])
                if year_column_sample is None and labels_years and has_financial_context(block):
                    year_column_sample = {
                        "table_id": block.get("table_id"),
                        "years": labels_years,
                        "header": corpus_record["header"][:12],
                    }
            else:
                counters["non_ready_tables"] += 1
                failure_reason_counts[failure_reason or "unknown"] += 1
                sample = failure_sample(block, failure_reason)
                if non_ready_sample is None:
                    non_ready_sample = sample
                if len(all_failure_samples) < 200:
                    all_failure_samples.append(sample)
                if len(sampled_failures_by_reason[failure_reason or "unknown"]) < 50:
                    sampled_failures_by_reason[failure_reason or "unknown"].append(sample)
                stats = None

            index_record = build_index_record(
                block,
                matrix=matrix,
                stats=stats,
                tabgr_ready=tabgr_ready,
                failure_reason=failure_reason,
            )
            index_errors = validate_index_record(index_record)
            if index_errors:
                validation_errors.append(
                    f"{block.get('table_id')}: index_validation:{','.join(index_errors[:10])}"
                )
            write_jsonl_row(index_fh, index_record)
            counters["index_rows"] += 1

    if counters["non_ready_tables"] <= 200:
        failure_samples = all_failure_samples
    else:
        failure_samples = [
            sample
            for reason in sorted(sampled_failures_by_reason)
            for sample in sampled_failures_by_reason[reason]
        ]
    with open_jsonl(failure_samples_path) as failure_fh:
        for sample in failure_samples:
            write_jsonl_row(failure_fh, sample)
    counters["failure_sample_rows"] = len(failure_samples)

    if args.strict_counts:
        expected_checks = {
            "input_table_blocks": EXPECTED_INPUT_TABLE_BLOCKS,
            "ready_tables": EXPECTED_READY_TABLES,
            "non_ready_tables": EXPECTED_NON_READY_TABLES,
        }
        for key, expected in expected_checks.items():
            observed = counters[key]
            if observed != expected:
                validation_errors.append(f"{key}_mismatch: observed={observed} expected={expected}")

    if counters["corpus_rows"] != counters["ready_tables"]:
        validation_errors.append(
            f"corpus_rows_ready_mismatch: corpus_rows={counters['corpus_rows']} ready={counters['ready_tables']}"
        )
    if counters["index_rows"] != counters["input_table_blocks"]:
        validation_errors.append(
            f"index_rows_input_mismatch: index_rows={counters['index_rows']} input={counters['input_table_blocks']}"
        )
    if cell_role_counts and any(role not in CELL_ROLES for role in cell_role_counts):
        validation_errors.append(f"invalid_cell_roles_seen:{dict(cell_role_counts)}")

    first_expected = {
        "table_id": "A000026_飞亚达_2019年年度报告_table_0001_abd1f09db4",
        "raw_markdown_sha1": "abd1f09db4a4ddafebc2530d9bb5bc732330cde8",
    }
    if first_table_sample:
        for key, expected in first_expected.items():
            if first_table_sample.get(key) != expected:
                validation_errors.append(f"first_table_{key}_mismatch")

    summary = {
        "input_table_blocks": counters["input_table_blocks"],
        "ready_tables": counters["ready_tables"],
        "non_ready_tables": counters["non_ready_tables"],
        "corpus_rows": counters["corpus_rows"],
        "cell_rows": counters["cell_rows"],
        "index_rows": counters["index_rows"],
        "failure_sample_rows": counters["failure_sample_rows"],
        "parse_status_counts": dict(parse_status_counts),
        "failure_reason_counts": dict(failure_reason_counts),
        "cell_role_counts": dict(cell_role_counts),
        "edge_count": counters["edge_count"],
        "header_cell_count": counters["header_cell_count"],
        "row_header_cell_count": counters["row_header_cell_count"],
        "data_cell_count": counters["data_cell_count"],
    }
    outputs = {
        "tabgr_table_corpus": rel(corpus_path, root),
        "table_cells": rel(cells_path, root),
        "tabgr_table_index": rel(index_path, root),
        "tabgr_table_corpus_schema": rel(args.schema_dir / "tabgr_table_corpus.schema.json", root),
        "table_cells_schema": rel(args.schema_dir / "table_cells.schema.json", root),
        "tabgr_table_index_schema": rel(args.schema_dir / "tabgr_table_index.schema.json", root),
        "tabgr_parse_report": rel(parse_report_path, root),
        "phase_report": rel(phase_report_path, root),
        "tabgr_failure_samples": rel(failure_samples_path, root),
    }
    report = {
        "schema_version": SCHEMA_REPORT,
        "adapter_version": ADAPTER_VERSION,
        "generated_at_utc": generated_at,
        "command": " ".join(sys.argv),
        "workspace_root": root.as_posix(),
        "inputs": {
            "table_blocks": rel(args.input, root),
        },
        "outputs": outputs,
        "schemas": schema_outputs,
        "summary": summary,
        "validation": {
            "strict_counts": args.strict_counts,
            "expected_counts": {
                "input_table_blocks": EXPECTED_INPUT_TABLE_BLOCKS,
                "ready_tables": EXPECTED_READY_TABLES,
                "non_ready_tables": EXPECTED_NON_READY_TABLES,
            },
            "corpus_rows_equal_ready_tables": counters["corpus_rows"] == counters["ready_tables"],
            "index_rows_equal_input_table_blocks": counters["index_rows"] == counters["input_table_blocks"],
            "edge_invariant": "edge_count == row_header_cell_count + data_cell_count",
            "edge_invariant_failure_count": len(edge_invariant_failures),
            "edge_invariant_failure_samples": edge_invariant_failures,
            "error_count": len(validation_errors),
            "errors": validation_errors[:100],
        },
        "spot_checks": {
            "first_table_preserves_sha1_table_id": first_table_sample,
            "first_ready_table": first_ready_sample,
            "merged_cell_matrix_source_of_truth": merged_cell_sample,
            "year_columns_preserved": year_column_sample,
            "non_ready_audit_sample": non_ready_sample,
        },
    }
    write_json(parse_report_path, report)
    phase_report_path.parent.mkdir(parents=True, exist_ok=True)
    phase_report_path.write_text(phase_report_markdown(report), encoding="utf-8")

    print(json.dumps({"summary": summary, "validation": report["validation"], "outputs": outputs}, ensure_ascii=False, indent=2))
    return 1 if validation_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
