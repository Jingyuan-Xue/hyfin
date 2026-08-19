#!/usr/bin/env python3
"""Independent, fail-closed validator for the experimental table evidence index.

The validator intentionally does not import the index builder.  It joins every
fragment back to the frozen Phase 3/4 records so a builder bug cannot validate
itself by repeating its own assertions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from itertools import groupby
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "finglmqa.experimental.table_evidence_fragment.v1"
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
HTML_TABLE_RE = re.compile(r"<\s*/?\s*(?:table|thead|tbody|tfoot|tr|th|td)\b", re.I)
MARKDOWN_SEPARATOR_RE = re.compile(r"(?m)^\s*\|?(?:\s*:?-{3,}:?\s*\|){1,}\s*$")

REQUIRED_FIELDS = {
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
    "fragment_sha256",
    "provenance",
}

THEME_PATTERNS: dict[str, tuple[str, ...]] = {
    "customer_supplier": ("客户", "供应商", "采购", "销售客户"),
    "staff_personnel": ("员工", "职工", "人员构成", "教育程度", "专业构成"),
    "shareholder_governance": ("股东", "持股", "董事", "监事", "治理"),
    "dividend": ("分红", "股利", "现金红利", "利润分配"),
    "contract_related_party": ("合同", "关联方", "关联交易", "担保"),
    "assets_liabilities": ("资产", "负债", "应收", "应付", "存货"),
}

EXPECTED_EXCLUSION_COUNTS = {
    "dividend": 905,
    "financial_table": 9020,
    "governance_or_other_non_financial": 4826,
    "mixed_narrative": 44237,
    "shareholder": 1668,
    "staff_or_personnel": 1718,
}

ALLOWED_UNIT_VALUES = {
    "%",
    "人",
    "件",
    "万元",
    "万元/吨",
    "万件",
    "万套/万件",
    "万平方米",
    "万千瓦时",
    "万股",
    "万辆",
    "万吨",
    "万吨/日",
    "人民币元",
    "人民币千元",
    "人民币万元",
    "人民币亿元",
    "亿元",
    "兆瓦",
    "元",
    "元/股",
    "千元",
    "千瓦时",
    "吨",
    "平方米",
    "澳元",
    "美元",
    "股",
    "辆",
    "双",
    "mg/L",
}
UNIT_HINT_RE = re.compile(r"^(?:单位|金额单位)\s*[：:]\s*(.+?)\s*$")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def raw_values_sha256(values: list[str]) -> str:
    return sha256_text(canonical_json(values))


def fragment_payload(record: dict[str, Any]) -> dict[str, Any]:
    """Return the semantic payload covered by ``fragment_sha256``."""

    return {key: value for key, value in record.items() if key != "fragment_sha256"}


def computed_fragment_sha256(record: dict[str, Any]) -> str:
    return sha256_text(canonical_json(fragment_payload(record)))


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank JSONL line")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: JSONL row is not an object")
            yield value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def check_canonical_file(path: Path, rows: list[dict[str, Any]]) -> list[str]:
    expected = "".join(canonical_json(row) + "\n" for row in rows).encode("utf-8")
    return [] if path.read_bytes() == expected else ["index_not_canonical_jsonl"]


def _coord(value: Any) -> tuple[int, int] | None:
    if (
        isinstance(value, list)
        and len(value) == 2
        and all(isinstance(item, int) and not isinstance(item, bool) and item >= 0 for item in value)
    ):
        return value[0], value[1]
    return None


def _line_range(value: Any) -> tuple[int, int] | None:
    coord = _coord(value)
    if coord is None or coord[0] < 1 or coord[1] < coord[0]:
        return None
    return coord


def canonical_sort_key(record: dict[str, Any]) -> tuple[Any, ...]:
    coordinates = [_coord(value) for value in record.get("cell_coordinates", [])]
    coordinates = [value for value in coordinates if value is not None]
    line_range = _line_range(record.get("source_line_range")) or (0, 0)
    kind = record.get("fragment_kind")
    kind_order = {"table_row": 0, "mixed_narrative": 1}.get(kind, 99)
    row_or_line = record.get("row_index") if kind == "table_row" else line_range[0]
    if not isinstance(row_or_line, int):
        row_or_line = -1
    return (
        str(record.get("document_id", "")),
        int(record.get("table_index", -1)) if isinstance(record.get("table_index"), int) else -1,
        str(record.get("table_id", "")),
        kind_order,
        row_or_line,
        str(record.get("fragment_id", "")),
    )


def expected_fragment_id(record: dict[str, Any]) -> str:
    coordinates = [_coord(value) for value in record.get("cell_coordinates", [])]
    coordinates = [value for value in coordinates if value is not None]
    row_or_block: Any
    if record.get("fragment_kind") == "table_row":
        row_or_block = record.get("row_index")
    else:
        row_or_block = record.get("source_block_id")
    identity = [
        record.get("schema_version"),
        record.get("fragment_kind"),
        record.get("document_id"),
        record.get("table_id"),
        row_or_block,
    ]
    return "tef-" + sha256_text(canonical_json(identity))[:32]


def _source_descriptor_valid(value: Any, *, nullable: bool) -> bool:
    if value is None:
        return nullable
    if not isinstance(value, dict) or set(value) != {"kind", "value"}:
        return False
    kind = value.get("kind")
    source_value = value.get("value")
    return isinstance(kind, str) and bool(kind) and isinstance(source_value, str) and bool(source_value)


def _relative_source_valid(value: Any, root: Path) -> bool:
    if not isinstance(value, str) or not value:
        return False
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        return False
    # The audited refs/source_markdown entries are symlinks whose targets live
    # outside the repository; containment applies to the recorded path itself.
    return (root / relative).is_file()


def _relative_source_from_frozen(value: Any, root: Path) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        path = Path(value)
        if path.is_absolute():
            return path.relative_to(root.absolute()).as_posix()
        if ".." in path.parts:
            return None
        return path.as_posix()
    except (ValueError, OSError):
        return None


def _text_for_themes(record: dict[str, Any]) -> str:
    pieces: list[str] = []
    for key in ("caption", "row_label", "content", "retrieval_text"):
        value = record.get(key)
        if isinstance(value, str):
            pieces.append(value)
    for key in ("section_path", "header_path", "column_labels"):
        value = record.get(key)
        if isinstance(value, list):
            pieces.extend(str(item) for item in value)
    return "\n".join(pieces)


def _issue(issues: list[dict[str, Any]], code: str, record: dict[str, Any] | None = None, detail: str = "") -> None:
    item: dict[str, Any] = {"code": code}
    if record is not None:
        item["fragment_id"] = record.get("fragment_id")
    if detail:
        item["detail"] = detail
    issues.append(item)


def validate_records(
    records: list[dict[str, Any]],
    *,
    tables: dict[tuple[str, str], dict[str, Any]],
    cells: dict[tuple[str, str, int, int], dict[str, Any]],
    text_blocks: dict[tuple[str, str], dict[str, Any]],
    root: Path = ROOT,
    minimum_documents: int = 1,
    minimum_per_theme: int = 1,
    minimum_documents_per_theme: int = 1,
) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    ids: set[str] = set()
    themes: Counter[str] = Counter()
    theme_documents: dict[str, set[str]] = {theme: set() for theme in THEME_PATTERNS}
    kinds: Counter[str] = Counter()
    documents: set[str] = set()

    if not records:
        _issue(issues, "index_empty")
    if records != sorted(records, key=canonical_sort_key):
        _issue(issues, "index_order_not_canonical")

    for record in records:
        missing = sorted(REQUIRED_FIELDS - set(record))
        if missing:
            _issue(issues, "required_fields_missing", record, ",".join(missing))
            continue
        if record["schema_version"] != SCHEMA_VERSION:
            _issue(issues, "schema_version_invalid", record)
        fragment_id = record["fragment_id"]
        if not isinstance(fragment_id, str) or not fragment_id:
            _issue(issues, "fragment_id_invalid", record)
        elif fragment_id in ids:
            _issue(issues, "fragment_id_duplicate", record)
        else:
            ids.add(fragment_id)
            if fragment_id != expected_fragment_id(record):
                _issue(issues, "fragment_id_identity_mismatch", record)

        kind = record["fragment_kind"]
        if kind not in {"table_row", "mixed_narrative"}:
            _issue(issues, "fragment_kind_invalid", record)
            continue
        kinds[kind] += 1
        document_id = record["document_id"]
        table_id = record["table_id"]
        if isinstance(document_id, str) and document_id:
            documents.add(document_id)

        table = tables.get((document_id, table_id))
        if table is None:
            # A table ID found under another document is explicit cross-document poison.
            poisoned = any(key[1] == table_id for key in tables)
            _issue(issues, "cross_document_table_reference" if poisoned else "table_reference_missing", record)
            continue

        expected_identity = {
            "company_full": table.get("company_full") or table.get("metadata", {}).get("company_full"),
            "company_name": table.get("stock_name") or table.get("metadata", {}).get("stock_name"),
            "stock_code": table.get("stock_code") or table.get("metadata", {}).get("stock_code"),
            "report_year": int(table.get("report_year") or table.get("metadata", {}).get("report_year")),
            "table_index": table.get("table_index"),
        }
        for field, expected in expected_identity.items():
            if record[field] != expected:
                _issue(issues, "document_identity_mismatch", record, field)
        if kind == "table_row" and record["section_path"] != table.get("section_path", []):
            _issue(issues, "section_path_source_mismatch", record)
        expected_caption = str(table.get("caption") or table.get("table_caption") or "")
        if record["caption"] != expected_caption:
            _issue(issues, "caption_source_mismatch", record)
        if record["header_path"] != table.get("header", []):
            _issue(issues, "header_path_source_mismatch", record)
        expected_source = _relative_source_from_frozen(table.get("source_markdown"), root)
        if expected_source is not None and record["source_markdown"] != expected_source:
            _issue(issues, "source_markdown_mismatch", record)

        for field in ("section_path", "header_path", "column_labels"):
            value = record[field]
            if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                _issue(issues, f"{field}_invalid", record)
        if kind == "table_row" and not record["header_path"]:
            _issue(issues, "header_path_unresolved", record)
        if not _source_descriptor_valid(record["year_source"], nullable=False):
            _issue(issues, "year_source_unauditable", record)
        elif record["year_source"] != {"kind": "report_metadata", "value": str(record["report_year"])}:
            _issue(issues, "year_source_not_frozen_report_metadata", record)
        if not _source_descriptor_valid(record["unit_source"], nullable=True):
            _issue(issues, "unit_source_unauditable", record)
        elif record["unit_source"] is not None:
            unit_source = record["unit_source"]
            raw_hint = table.get("unit_hint")
            match = UNIT_HINT_RE.fullmatch(str(raw_hint).strip()) if raw_hint is not None else None
            if unit_source.get("kind") != "table_unit_hint":
                _issue(issues, "unit_source_kind_invalid", record)
            if unit_source.get("value") != (str(raw_hint).strip() if raw_hint is not None else None):
                _issue(issues, "unit_source_value_mismatch", record)
            normalized_unit = re.sub(r"\s+", "", match.group(1)) if match is not None else None
            if normalized_unit not in ALLOWED_UNIT_VALUES:
                _issue(issues, "unit_source_value_not_allowlisted", record)
        if not _relative_source_valid(record["source_markdown"], root):
            _issue(issues, "source_markdown_not_safe_relative_file", record)
        line_range = _line_range(record["source_line_range"])
        if line_range is None:
            _issue(issues, "source_line_range_invalid", record)
        for hash_field in ("raw_value_sha256", "source_content_sha256", "fragment_sha256"):
            if not isinstance(record[hash_field], str) or not HEX64_RE.fullmatch(record[hash_field]):
                _issue(issues, f"{hash_field}_invalid", record)
        if record["fragment_sha256"] != computed_fragment_sha256(record):
            _issue(issues, "fragment_sha256_mismatch", record)
        if not isinstance(record["content"], str) or not record["content"].strip():
            _issue(issues, "content_empty", record)
        if not isinstance(record["retrieval_text"], str) or not record["retrieval_text"].strip():
            _issue(issues, "retrieval_text_empty", record)
        provenance = record["provenance"]
        if not isinstance(provenance, dict) or not provenance:
            _issue(issues, "provenance_invalid", record)
            provenance = {}
        required_provenance = {
            "source_kind",
            "source_schema_version",
            "source_record_ids",
            "table_content_sha256",
            "text_was_separated_from_table",
            "numeric_authorization",
        }
        if not required_provenance.issubset(provenance):
            _issue(issues, "provenance_fields_missing", record)
        if provenance.get("numeric_authorization") != "not_authorized_for_answer":
            _issue(issues, "numeric_authorization_not_fail_closed", record)
        if not isinstance(provenance.get("source_record_ids"), list) or not all(
            isinstance(item, str) and item for item in provenance.get("source_record_ids", [])
        ):
            _issue(issues, "provenance_source_record_ids_invalid", record)
        if provenance.get("table_content_sha256") != table.get("content_hash"):
            _issue(issues, "provenance_table_hash_mismatch", record)

        values = record["raw_cell_values"]
        coordinates = record["cell_coordinates"]
        if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
            _issue(issues, "raw_cell_values_invalid", record)
            values = []
        parsed_coords = [_coord(item) for item in coordinates] if isinstance(coordinates, list) else []
        if not isinstance(coordinates, list) or any(item is None for item in parsed_coords):
            _issue(issues, "cell_coordinates_invalid", record)
            parsed_coords = []
        if record["raw_value_sha256"] != raw_values_sha256(values):
            _issue(issues, "raw_value_sha256_mismatch", record)

        if kind == "table_row":
            if (
                not isinstance(record["row_index"], int)
                or isinstance(record["row_index"], bool)
                or record["row_index"] < 0
            ):
                _issue(issues, "row_index_invalid", record)
            elif parsed_coords and any(coordinate[0] != record["row_index"] for coordinate in parsed_coords if coordinate):
                _issue(issues, "row_index_coordinate_mismatch", record)
            if not parsed_coords or len(parsed_coords) != len(values):
                _issue(issues, "coordinate_value_cardinality_invalid", record)
            if len(set(parsed_coords)) != len(parsed_coords) or parsed_coords != sorted(parsed_coords):
                _issue(issues, "coordinates_not_unique_sorted", record)
            source_cells: list[dict[str, Any]] = []
            for coordinate, raw_value in zip(parsed_coords, values):
                assert coordinate is not None
                cell = cells.get((document_id, table_id, coordinate[0], coordinate[1]))
                if cell is None:
                    _issue(issues, "cell_coordinate_missing", record, str(coordinate))
                elif cell.get("raw_value") != raw_value:
                    _issue(issues, "cell_value_mismatch", record, str(coordinate))
                else:
                    source_cells.append(cell)
            if len(source_cells) == len(parsed_coords):
                expected_column_labels = [str(cell.get("column_label", "")) for cell in source_cells]
                if record["column_labels"] != expected_column_labels:
                    _issue(issues, "column_labels_source_mismatch", record)
                source_row_labels = {
                    str(cell.get("row_label", "")).strip()
                    for cell in source_cells
                    if str(cell.get("row_label", "")).strip()
                }
                if len(source_row_labels) > 1:
                    _issue(issues, "source_row_label_conflict", record)
                expected_row_label = (
                    next(iter(source_row_labels))
                    if len(source_row_labels) == 1
                    else (values[0] if values and not source_row_labels else "")
                )
                if record["row_label"] != expected_row_label:
                    _issue(issues, "row_label_source_mismatch", record)
            if record["source_block_id"] is not None:
                _issue(issues, "table_row_source_block_forbidden", record)
            expected_record_ids = [
                f"{table_id}:{coordinate[0]}:{coordinate[1]}"
                for coordinate in parsed_coords
                if coordinate is not None
            ]
            if provenance.get("source_kind") != "phase4_table_cells":
                _issue(issues, "table_row_provenance_kind_invalid", record)
            if provenance.get("source_schema_version") != "finglmqa.phase4.table_cell.v1":
                _issue(issues, "table_row_provenance_schema_invalid", record)
            if provenance.get("source_record_ids") != expected_record_ids:
                _issue(issues, "table_row_provenance_ids_mismatch", record)
            if provenance.get("text_was_separated_from_table") is not False:
                _issue(issues, "table_row_separation_flag_invalid", record)
            source_hash = table.get("content_hash")
            if source_hash and record["source_content_sha256"] != source_hash:
                _issue(issues, "table_source_content_sha256_mismatch", record)
            source_ranges = [_line_range(cell.get("line_range")) for cell in source_cells]
            source_ranges = [value for value in source_ranges if value is not None]
            expected_range = (
                (min(value[0] for value in source_ranges), max(value[1] for value in source_ranges))
                if source_ranges
                else None
            )
            if line_range and expected_range and line_range != expected_range:
                _issue(issues, "table_source_line_range_mismatch", record)
        else:
            if record["row_index"] is not None:
                _issue(issues, "mixed_narrative_row_index_must_be_null", record)
            if parsed_coords or values:
                _issue(issues, "mixed_narrative_contains_cells", record)
            block_id = record["source_block_id"]
            block = text_blocks.get((document_id, block_id)) if isinstance(block_id, str) else None
            if block is None:
                poisoned = any(key[1] == block_id for key in text_blocks) if block_id else False
                _issue(issues, "cross_document_text_block_reference" if poisoned else "text_block_reference_missing", record)
            else:
                if table_id not in block.get("linked_table_ids", []):
                    _issue(issues, "mixed_narrative_table_not_linked", record)
                if record["content"] != block.get("text"):
                    _issue(issues, "mixed_narrative_source_text_mismatch", record)
                if record["source_content_sha256"] != block.get("text_hash"):
                    _issue(issues, "mixed_narrative_source_hash_mismatch", record)
                if line_range != _line_range(block.get("line_range")):
                    _issue(issues, "mixed_narrative_source_line_range_mismatch", record)
                if record["section_path"] != block.get("section_path", []):
                    _issue(issues, "mixed_narrative_section_path_mismatch", record)
                expected_block_source = _relative_source_from_frozen(block.get("source_markdown"), root)
                if expected_block_source is not None and record["source_markdown"] != expected_block_source:
                    _issue(issues, "mixed_narrative_source_markdown_mismatch", record)
            if provenance.get("source_kind") != "phase3_text_block":
                _issue(issues, "mixed_narrative_provenance_kind_invalid", record)
            if provenance.get("source_schema_version") != "finglmqa.phase3.text_block.v1":
                _issue(issues, "mixed_narrative_provenance_schema_invalid", record)
            if provenance.get("source_record_ids") != [block_id]:
                _issue(issues, "mixed_narrative_provenance_ids_mismatch", record)
            if provenance.get("text_was_separated_from_table") is not True:
                _issue(issues, "mixed_narrative_separation_flag_invalid", record)
            for field in ("content", "retrieval_text"):
                text = record[field]
                if isinstance(text, str) and (HTML_TABLE_RE.search(text) or MARKDOWN_SEPARATOR_RE.search(text)):
                    _issue(issues, "mixed_narrative_table_markup_leak", record, field)

        text = _text_for_themes(record)
        for theme, keywords in THEME_PATTERNS.items():
            if any(keyword in text for keyword in keywords):
                themes[theme] += 1
                if isinstance(document_id, str):
                    theme_documents[theme].add(document_id)

    if len(documents) < minimum_documents:
        _issue(issues, "document_coverage_below_minimum", detail=f"{len(documents)}<{minimum_documents}")
    for theme in THEME_PATTERNS:
        if themes[theme] < minimum_per_theme:
            _issue(issues, "theme_coverage_below_minimum", detail=f"{theme}:{themes[theme]}<{minimum_per_theme}")
        if len(theme_documents[theme]) < minimum_documents_per_theme:
            _issue(
                issues,
                "theme_document_coverage_below_minimum",
                detail=f"{theme}:{len(theme_documents[theme])}<{minimum_documents_per_theme}",
            )

    counts = Counter(item["code"] for item in issues)
    return {
        "schema_version": "finglmqa.table_evidence_validation_report.v1",
        "status": "pass" if not issues else "fail",
        "record_count": len(records),
        "document_count": len(documents),
        "fragment_kind_counts": dict(sorted(kinds.items())),
        "theme_counts": {theme: themes[theme] for theme in sorted(THEME_PATTERNS)},
        "theme_document_counts": {theme: len(theme_documents[theme]) for theme in sorted(THEME_PATTERNS)},
        "issue_count": len(issues),
        "issue_counts": dict(sorted(counts.items())),
        "issues": issues[:1000],
        "issues_truncated": len(issues) > 1000,
    }


def _selected_source_maps(
    records: list[dict[str, Any]],
    *,
    table_corpus_path: Path,
    table_cells_path: Path,
    text_blocks_path: Path,
) -> tuple[
    dict[tuple[str, str], dict[str, Any]],
    dict[tuple[str, str, int, int], dict[str, Any]],
    dict[tuple[str, str], dict[str, Any]],
]:
    selected_tables = {(row.get("document_id"), row.get("table_id")) for row in records}
    selected_table_ids = {table_id for _, table_id in selected_tables}
    selected_coords = {
        (row.get("document_id"), row.get("table_id"), coordinate[0], coordinate[1])
        for row in records
        for coordinate in row.get("cell_coordinates", [])
        if _coord(coordinate) is not None
    }
    selected_blocks = {
        (row.get("document_id"), row.get("source_block_id"))
        for row in records
        if row.get("source_block_id") is not None
    }
    selected_block_ids = {block_id for _, block_id in selected_blocks}
    tables = {
        (row.get("document_id"), row.get("table_id")): row
        for row in iter_jsonl(table_corpus_path)
        if (row.get("document_id"), row.get("table_id")) in selected_tables
        or row.get("table_id") in selected_table_ids
    }
    cells = {
        (row.get("document_id"), row.get("table_id"), row.get("row_index"), row.get("col_index")): row
        for row in iter_jsonl(table_cells_path)
        if (row.get("document_id"), row.get("table_id"), row.get("row_index"), row.get("col_index")) in selected_coords
    }
    text_blocks = {
        (row.get("document_id"), row.get("block_id") or row.get("text_block_id")): row
        for row in iter_jsonl(text_blocks_path)
        if (row.get("document_id"), row.get("block_id") or row.get("text_block_id")) in selected_blocks
        or (row.get("block_id") or row.get("text_block_id")) in selected_block_ids
    }
    return tables, cells, text_blocks


def _exclusion_summary(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    return {
        "record_count": len(rows),
        "classification_counts": dict(sorted(Counter(row.get("classification") for row in rows).items())),
        "document_count": len({row.get("document_id") for row in rows}),
    }


def _load_slim_tables(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    keep = {
        "schema_version",
        "document_id",
        "table_id",
        "table_index",
        "company_full",
        "stock_name",
        "stock_code",
        "report_year",
        "content_hash",
        "line_range",
        "unit_hint",
        "section_path",
        "caption",
        "table_caption",
        "header",
        "source_markdown",
        "metadata",
    }
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for row in iter_jsonl(path):
        key = (str(row.get("document_id") or ""), str(row.get("table_id") or ""))
        result[key] = {name: row.get(name) for name in keep}
    return result


def _load_linked_text_blocks(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    keep = {
        "schema_version",
        "document_id",
        "block_id",
        "text_block_id",
        "text",
        "text_hash",
        "line_range",
        "linked_table_ids",
        "section_path",
        "source_markdown",
    }
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for row in iter_jsonl(path):
        if not row.get("linked_table_ids"):
            continue
        block_id = str(row.get("block_id") or row.get("text_block_id") or "")
        key = (str(row.get("document_id") or ""), block_id)
        result[key] = {name: row.get(name) for name in keep}
    return result


def _cell_group_key(row: dict[str, Any]) -> tuple[str, int, str, int]:
    return (
        str(row.get("document_id") or ""),
        int(row.get("table_index", -1)),
        str(row.get("table_id") or ""),
        int(row.get("row_index", -1)),
    )


def _iter_cell_groups(path: Path) -> Iterable[tuple[tuple[str, int, str, int], list[dict[str, Any]]]]:
    for key, rows in groupby(iter_jsonl(path), key=_cell_group_key):
        yield key, list(rows)


def validate_index_file_streaming(
    index_path: Path,
    *,
    table_corpus_path: Path,
    table_cells_path: Path,
    text_blocks_path: Path,
    root: Path = ROOT,
    minimum_documents: int = 170,
    minimum_per_theme: int = 20,
    minimum_documents_per_theme: int = 5,
) -> dict[str, Any]:
    """Validate the full corpus without materializing ~460k fragments in RAM."""

    tables = _load_slim_tables(table_corpus_path)
    text_blocks = _load_linked_text_blocks(text_blocks_path)
    source_groups = iter(_iter_cell_groups(table_cells_path))
    current_group = next(source_groups, None)
    issue_counts: Counter[str] = Counter()
    issues: list[dict[str, Any]] = []
    record_count = 0
    kinds: Counter[str] = Counter()
    themes: Counter[str] = Counter()
    theme_documents: dict[str, set[str]] = {theme: set() for theme in THEME_PATTERNS}
    documents: set[str] = set()
    fragment_ids: set[str] = set()
    last_key: tuple[Any, ...] | None = None

    def add(code: str, fragment_id: Any = None, detail: str = "") -> None:
        issue_counts[code] += 1
        if len(issues) < 1000:
            item: dict[str, Any] = {"code": code}
            if fragment_id is not None:
                item["fragment_id"] = fragment_id
            if detail:
                item["detail"] = detail
            issues.append(item)

    with index_path.open("rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                add("blank_jsonl_line", detail=str(line_number))
                continue
            try:
                record = json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                add("invalid_jsonl", detail=f"{line_number}:{exc}")
                continue
            if not isinstance(record, dict):
                add("jsonl_row_not_object", detail=str(line_number))
                continue
            record_count += 1
            fragment_id = record.get("fragment_id")
            if raw != (canonical_json(record) + "\n").encode("utf-8"):
                add("index_not_canonical_jsonl", fragment_id, str(line_number))
            key = canonical_sort_key(record)
            if last_key is not None and key <= last_key:
                add("index_order_not_strictly_canonical", fragment_id, str(line_number))
            last_key = key
            if isinstance(fragment_id, str):
                if fragment_id in fragment_ids:
                    add("fragment_id_duplicate", fragment_id)
                fragment_ids.add(fragment_id)
            document_id = record.get("document_id")
            if isinstance(document_id, str):
                documents.add(document_id)
            kind = record.get("fragment_kind")
            if isinstance(kind, str):
                kinds[kind] += 1

            source_cells: dict[tuple[str, str, int, int], dict[str, Any]] = {}
            if kind == "table_row":
                expected_group_key = (
                    str(record.get("document_id") or ""),
                    int(record.get("table_index", -1)),
                    str(record.get("table_id") or ""),
                    int(record.get("row_index", -1)),
                )
                while current_group is not None and current_group[0] < expected_group_key:
                    add("source_cell_group_not_indexed", detail=str(current_group[0]))
                    current_group = next(source_groups, None)
                if current_group is None or current_group[0] != expected_group_key:
                    add("indexed_row_missing_source_cell_group", fragment_id, str(expected_group_key))
                else:
                    for cell in current_group[1]:
                        source_cells[
                            (
                                str(cell.get("document_id") or ""),
                                str(cell.get("table_id") or ""),
                                int(cell.get("row_index", -1)),
                                int(cell.get("col_index", -1)),
                            )
                        ] = cell
                    current_group = next(source_groups, None)

            subreport = validate_records(
                [record],
                tables=tables,
                cells=source_cells,
                text_blocks=text_blocks,
                root=root,
                minimum_documents=0,
                minimum_per_theme=0,
                minimum_documents_per_theme=0,
            )
            for code, count in subreport["issue_counts"].items():
                issue_counts[code] += count
            for item in subreport["issues"]:
                if len(issues) < 1000:
                    issues.append(item)
            for theme, count in subreport["theme_counts"].items():
                themes[theme] += count
                if count and isinstance(document_id, str):
                    theme_documents[theme].add(document_id)

    while current_group is not None:
        add("source_cell_group_not_indexed", detail=str(current_group[0]))
        current_group = next(source_groups, None)
    if record_count == 0:
        add("index_empty")
    if len(documents) < minimum_documents:
        add("document_coverage_below_minimum", detail=f"{len(documents)}<{minimum_documents}")
    for theme in THEME_PATTERNS:
        if themes[theme] < minimum_per_theme:
            add("theme_coverage_below_minimum", detail=f"{theme}:{themes[theme]}<{minimum_per_theme}")
        if len(theme_documents[theme]) < minimum_documents_per_theme:
            add(
                "theme_document_coverage_below_minimum",
                detail=f"{theme}:{len(theme_documents[theme])}<{minimum_documents_per_theme}",
            )
    total_issues = sum(issue_counts.values())
    return {
        "schema_version": "finglmqa.table_evidence_validation_report.v1",
        "status": "pass" if total_issues == 0 else "fail",
        "record_count": record_count,
        "document_count": len(documents),
        "fragment_kind_counts": dict(sorted(kinds.items())),
        "theme_counts": {theme: themes[theme] for theme in sorted(THEME_PATTERNS)},
        "theme_document_counts": {theme: len(theme_documents[theme]) for theme in sorted(THEME_PATTERNS)},
        "issue_count": total_issues,
        "issue_counts": dict(sorted(issue_counts.items())),
        "issues": issues,
        "issues_truncated": total_issues > len(issues),
    }


def write_canonical_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--table-corpus", type=Path, default=ROOT / "data/corpus_package/tabgr_table_corpus.jsonl")
    parser.add_argument("--table-cells", type=Path, default=ROOT / "data/corpus_package/table_cells.jsonl")
    parser.add_argument("--text-blocks", type=Path, default=ROOT / "data/corpus_package/text_blocks.jsonl")
    parser.add_argument(
        "--exclusions",
        type=Path,
        default=ROOT / "runs/phase_07/reports/table_exclusion_classification.jsonl",
    )
    parser.add_argument("--report", type=Path)
    parser.add_argument("--minimum-documents", type=int, default=170)
    parser.add_argument("--minimum-per-theme", type=int, default=20)
    parser.add_argument("--minimum-documents-per-theme", type=int, default=5)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = validate_index_file_streaming(
            args.index,
            table_corpus_path=args.table_corpus,
            table_cells_path=args.table_cells,
            text_blocks_path=args.text_blocks,
            minimum_documents=args.minimum_documents,
            minimum_per_theme=args.minimum_per_theme,
            minimum_documents_per_theme=args.minimum_documents_per_theme,
        )
        report["exclusion_baseline"] = _exclusion_summary(args.exclusions)
        if (
            report["exclusion_baseline"]["record_count"] != 62_374
            or report["exclusion_baseline"]["classification_counts"] != EXPECTED_EXCLUSION_COUNTS
        ):
            report["status"] = "fail"
            report["issue_count"] += 1
            report["issue_counts"]["exclusion_baseline_count_changed"] = 1
            report["issues"].append({"code": "exclusion_baseline_count_changed"})
    except (OSError, ValueError) as exc:
        report = {
            "schema_version": "finglmqa.table_evidence_validation_report.v1",
            "status": "fail",
            "issue_count": 1,
            "issue_counts": {"validator_input_error": 1},
            "issues": [{"code": "validator_input_error", "detail": str(exc)}],
        }
    if args.report:
        write_canonical_json(args.report, report)
    sys.stdout.write(canonical_json(report) + "\n")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
