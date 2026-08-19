#!/usr/bin/env python3
"""Build corpus-scoped TabGR v2 structured tables and row evidence.

The build is streaming: it retains selected Phase 6 facts and one table/document
shard at a time.  Source Markdown, legacy tables, indexes, facts, and runs are
read-only inputs; all outputs must be explicit corpus-scoped v2 paths.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import shutil
import sys
import tempfile
import time
from typing import Any, Iterable, Mapping, TextIO


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from finglmqa.type3_corpus_profile import (  # noqa: E402
    load_corpus_profile,
    sha256_file,
    source_snapshot,
)
from finglmqa.type3_tabgr_retriever import (  # noqa: E402
    TABGR_RUNTIME_SHA256,
    TABGR_V2_BUILDER_VERSION,
    TABGR_V2_ROW_SCHEMA,
    TABGR_V2_TABLE_SCHEMA,
    Type3TabGRError,
    build_fact_authorization,
    canonical_json_bytes,
    flatten_headers,
    infer_data_start_column,
    infer_header_bands,
    infer_period_state,
    infer_scope_state,
    infer_unit_state,
    normalize_text,
    numeric_fragments,
    reconstruct_origin_grid,
    redact_unauthorized,
    safe_numeric_projection,
    semantic_sha256,
    sha256_text,
)


MAX_PEAK_RSS_KIB = 8 * 1024 * 1024


def _atomic_text_writer(path: Path) -> tuple[TextIO, Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    )
    return handle, Path(handle.name)


def _finish_atomic(handle: TextIO, temporary: Path, target: Path) -> None:
    handle.flush()
    os.fsync(handle.fileno())
    handle.close()
    os.replace(temporary, target)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(canonical_json_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_row(handle: TextIO, value: Any) -> None:
    handle.write(canonical_json_bytes(value).decode("utf-8"))


def _safe_remove_dry_run(path: Path, *, run_dir: Path) -> None:
    resolved = path.resolve(strict=False)
    root = run_dir.resolve(strict=False)
    if resolved != root and root not in resolved.parents:
        raise RuntimeError("refusing to clean a dry-run path outside run_dir")
    if path.exists():
        shutil.rmtree(path)


def _verify_output_scope(path: Path, *, corpus_id: str, allow_dry_run: bool, run_dir: Path) -> None:
    resolved = path.resolve(strict=False)
    if allow_dry_run and run_dir.resolve(strict=False) in resolved.parents:
        return
    normalized = resolved.as_posix()
    if f"/type3/{corpus_id}/" not in normalized and not normalized.endswith(f"/type3/{corpus_id}"):
        raise RuntimeError(f"output is not corpus scoped: {path}")
    forbidden = (
        ROOT / "data/corpus_package/table_blocks.jsonl",
        ROOT / "data/corpus_package/tabgr_table_corpus.jsonl",
        ROOT / "data/indexes/tabgr_table_index.jsonl",
        ROOT / "data/facts/financial_facts.jsonl",
    )
    if any(resolved == value.resolve(strict=False) for value in forbidden):
        raise RuntimeError("refusing to overwrite a legacy artifact")


def _selected_facts(path: Path) -> tuple[dict[tuple[str, str, int, int, str], dict[str, Any]], list[dict[str, Any]]]:
    selected: dict[tuple[str, str, int, int, str], dict[str, Any]] = {}
    ordered: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise RuntimeError(f"blank financial fact row: {line_number}")
            fact = json.loads(line)
            if fact.get("is_selected") is not True:
                continue
            if fact.get("selection_status") not in {"selected_single_value", "resolved_by_confidence"}:
                raise RuntimeError("selected fact has an unsupported selection status")
            provenance = [
                row for row in fact.get("provenance") or ()
                if row.get("candidate_id") == fact.get("source_candidate_id")
            ]
            if len(provenance) != 1:
                raise RuntimeError("selected fact lacks exactly one selected provenance row")
            raw_value = str(provenance[0].get("raw_value") or "")
            key = (
                str(fact["document_id"]), str(fact["source_table_id"]),
                int(fact["source_row_index"]), int(fact["source_col_index"]), raw_value,
            )
            if key in selected:
                raise RuntimeError("selected facts duplicate an exact source coordinate/value")
            selected[key] = fact
            ordered.append(fact)
    ordered.sort(key=lambda row: str(row["fact_id"]))
    return selected, ordered


def _load_input_freeze(path: Path, *, corpus_id: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version", "corpus_id", "corpus_profile_sha256", "table_blocks_sha256",
        "financial_facts_sha256", "tabgr_runtime_sha256", "expected_input_tables",
        "expected_ready_tables", "expected_anomalies", "expected_selected_facts",
        "freeze_fingerprint",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise RuntimeError("TabGR input freeze fields differ")
    if value["schema_version"] != "finglmqa.type3.tabgr.input_freeze.v1":
        raise RuntimeError("TabGR input freeze schema is unsupported")
    if value["corpus_id"] != corpus_id:
        raise RuntimeError("TabGR input freeze corpus_id mismatch")
    expected_fingerprint = semantic_sha256({
        key: child for key, child in value.items() if key != "freeze_fingerprint"
    })
    if value["freeze_fingerprint"] != expected_fingerprint:
        raise RuntimeError("TabGR input freeze fingerprint mismatch")
    for field in (
        "corpus_profile_sha256", "table_blocks_sha256", "financial_facts_sha256",
        "tabgr_runtime_sha256",
    ):
        if not isinstance(value[field], str) or not re.fullmatch(r"[0-9a-f]{64}", value[field]):
            raise RuntimeError(f"TabGR input freeze {field} is not a SHA256")
    for field in (
        "expected_input_tables", "expected_ready_tables", "expected_anomalies",
        "expected_selected_facts",
    ):
        if isinstance(value[field], bool) or not isinstance(value[field], int) or value[field] < 0:
            raise RuntimeError(f"TabGR input freeze {field} is invalid")
    if value["expected_ready_tables"] + value["expected_anomalies"] != value["expected_input_tables"]:
        raise RuntimeError("TabGR input freeze table counts are inconsistent")
    return value


def _portable_source(profile: Mapping[str, Any], document: Mapping[str, Any]) -> str:
    return f"{profile['source_ref']}/{document['source_markdown']}"


def _line_range(block: Mapping[str, Any]) -> list[int]:
    value = block.get("line_range")
    if not isinstance(value, list) or len(value) != 2:
        raise Type3TabGRError("table line_range must contain two integers")
    result = [int(value[0]), int(value[1])]
    if result[0] < 1 or result[1] < result[0]:
        raise Type3TabGRError("table line_range is invalid")
    return result


def _verify_source_binding(
    block: Mapping[str, Any], *, source_text: str, source_sha256: str
) -> dict[str, Any]:
    raw = str(block.get("raw_markdown") or "")
    char_range = block.get("char_range")
    if not isinstance(char_range, list) or len(char_range) != 2:
        raise Type3TabGRError("table char_range must contain two integers")
    start, end = int(char_range[0]), int(char_range[1])
    if start < 0 or end <= start or end > len(source_text):
        raise Type3TabGRError("table char_range is invalid")
    if source_text[start:end] != raw:
        raise Type3TabGRError("raw Markdown does not match the profile source char range")
    raw_sha1 = hashlib.sha1(raw.encode("utf-8")).hexdigest()
    raw_sha256 = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    if raw_sha1 != block.get("raw_markdown_sha1"):
        raise Type3TabGRError("raw Markdown SHA1 differs")
    if raw_sha256 != block.get("content_hash"):
        raise Type3TabGRError("raw Markdown SHA256 differs")
    line_range = _line_range(block)
    before = source_text[:start]
    raw_end_offset = max(start, end - 1)
    actual_lines = [before.count("\n") + 1, source_text[:raw_end_offset].count("\n") + 1]
    if actual_lines != line_range:
        raise Type3TabGRError("table line range differs from the profile source")
    return {
        "status": "exact",
        "source_sha256": source_sha256,
        "raw_markdown_sha1": raw_sha1,
        "raw_markdown_sha256": raw_sha256,
        "char_range": [start, end],
        "line_range": line_range,
    }


def _resolved_state_text(value: Mapping[str, Any]) -> str:
    return str(value.get("value") or "") if value.get("status") == "resolved" else ""


def _combine_state(primary: Mapping[str, Any], fallback: Mapping[str, Any]) -> dict[str, Any]:
    if primary.get("status") != "unknown":
        return dict(primary)
    return dict(fallback)


def _row_hierarchy(
    row: list[str], *, data_start: int, category_path: list[str]
) -> tuple[list[str], list[str]]:
    leading = [normalize_text(value) for value in row[:data_start] if normalize_text(value)]
    nonempty = [normalize_text(value) for value in row if normalize_text(value)]
    data_values = [normalize_text(value) for value in row[data_start:] if normalize_text(value)]
    if len(nonempty) == 1 and leading and not numeric_fragments(leading[0]):
        value = leading[0]
        if re.match(r"^(其中|(?:[（(]?[一二三四五六七八九十0-9]+[）).、]))", value):
            category_path = [*category_path[:1], value]
        else:
            category_path = [value]
    row_path: list[str] = []
    for value in (*category_path, *leading):
        if value and (not row_path or value != row_path[-1]):
            row_path.append(value)
    if not row_path and data_values:
        row_path.append(data_values[0])
    return row_path, category_path


def _make_table(
    block: Mapping[str, Any],
    *,
    corpus_id: str,
    source_markdown: str,
    document: Mapping[str, Any],
    source_binding: Mapping[str, Any],
    fact_map: Mapping[tuple[str, str, int, int, str], Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    document_id = str(block["document_id"])
    table_id = str(block["table_id"])
    if document_id != document["document_id"]:
        raise Type3TabGRError("table document differs from corpus profile")
    if str(block.get("stock_code")) != str(document["stock_code"]):
        raise Type3TabGRError("table stock code differs from corpus profile")
    if int(block.get("report_year")) != int(document["report_year"]):
        raise Type3TabGRError("table report year differs from corpus profile")
    raw_sha1 = str(block.get("raw_markdown_sha1") or "")
    if len(raw_sha1) != 40 or table_id.rsplit("_", 1)[-1] != raw_sha1[:10]:
        raise Type3TabGRError("table id is not bound to raw Markdown SHA1")
    table_line_range = _line_range(block)
    grid = reconstruct_origin_grid(block.get("cell_spans"), block.get("matrix"), table_id=table_id)
    matrix = [list(row) for row in grid.matrix]
    if not matrix or not any(value for row in matrix for value in row):
        raise Type3TabGRError("ready table reconstructed as empty")

    table_sha256 = semantic_sha256({
        "corpus_id": corpus_id, "document_id": document_id, "table_id": table_id,
        "raw_markdown_sha1": raw_sha1, "matrix": matrix,
    })
    bands = infer_header_bands(grid)
    fact_rows = {
        key[2] for key in fact_map if key[0] == document_id and key[1] == table_id
    }
    if fact_rows:
        first_fact_row = min(fact_rows)
        bands["initial_header_rows"] = [
            row for row in bands["initial_header_rows"] if row < first_fact_row
        ] or [0]
        bands["embedded_header_resets"] = [
            row for row in bands["embedded_header_resets"] if row not in fact_rows
        ]
    initial_headers = list(bands["initial_header_rows"])
    embedded = set(bands["embedded_header_resets"])
    data_start = infer_data_start_column(
        matrix, header_rows=initial_headers, embedded_resets=sorted(embedded)
    )
    initial_flat = flatten_headers(matrix, initial_headers)
    legacy_headers = [normalize_text(value) or f"第{index + 1}列" for index, value in enumerate(matrix[0])]
    caption = normalize_text(block.get("caption"))
    section_path = [normalize_text(value) for value in block.get("section_path") or () if normalize_text(value)]
    nearby_text = normalize_text(block.get("nearby_text"))
    semantic_tags = sorted({normalize_text(value) for value in block.get("semantic_tags") or () if normalize_text(value)})
    report_year = int(document["report_year"])
    table_unit = infer_unit_state(block.get("unit_hint"), caption)
    table_scope = infer_scope_state(caption, " ".join(section_path), nearby_text)
    origin_cells = [cell.as_mapping() for cell in grid.origin_cells]
    origin_by_coordinate = grid.origin_by_coordinate

    rows: list[dict[str, Any]] = []
    ppr_cells: list[dict[str, Any]] = []
    ppr_edges: list[str] = []
    all_authorizations: list[dict[str, Any]] = []
    category_path: list[str] = []
    active_header_rows = list(initial_headers)
    flattened_headers = list(initial_flat)
    joined_fact_ids: set[str] = set()
    header_nonempty = sum(bool(value) for value in flattened_headers)
    raw_header_resolved = sum(
        any(normalize_text(matrix[row][column]) for row in initial_headers if column < len(matrix[row]))
        for column in range(len(initial_flat))
    )
    legacy_raw_header_resolved = sum(bool(normalize_text(value)) for value in matrix[0])
    for row_index, row in enumerate(matrix):
        if row_index in initial_headers:
            continue
        if row_index in embedded:
            active_header_rows = [row_index]
            flattened_headers = flatten_headers(matrix, active_header_rows)
            category_path = []
            header_nonempty += sum(bool(value) for value in flattened_headers)
            raw_header_resolved += sum(bool(normalize_text(value)) for value in matrix[row_index])
            legacy_raw_header_resolved += sum(bool(normalize_text(value)) for value in matrix[row_index])
            continue
        row_path, category_path = _row_hierarchy(row, data_start=data_start, category_path=category_path)
        if not any(normalize_text(value) for value in row):
            continue
        cells: list[dict[str, Any]] = []
        authorizations: list[dict[str, Any]] = []
        unauthorized: list[str] = []
        display_parts: list[str] = []
        period_states: dict[str, dict[str, Any]] = {}
        unit_states: dict[str, dict[str, Any]] = {}
        scope_states: dict[str, dict[str, Any]] = {}
        for col_index, raw_value in enumerate(row):
            value = normalize_text(raw_value)
            if not value:
                continue
            label = flattened_headers[col_index] if col_index < len(flattened_headers) else f"第{col_index + 1}列"
            origin = origin_by_coordinate.get((row_index, col_index))
            if origin is None:
                raise Type3TabGRError("matrix cell lacks an origin cell")
            fact = fact_map.get((document_id, table_id, row_index, col_index, value))
            authorization: dict[str, Any] | None = None
            if fact is not None:
                authorization = build_fact_authorization(
                    fact, corpus_id=corpus_id, raw_value=value,
                    source_markdown=source_markdown, table_sha256=table_sha256,
                    table_line_range=table_line_range,
                )
                authorizations.append(authorization)
                all_authorizations.append(authorization)
                joined_fact_ids.add(str(fact["fact_id"]))
            fragments = numeric_fragments(value) if col_index >= data_start else []
            if fragments and authorization is None:
                unauthorized.extend(fragments)
            explicit_unit = infer_unit_state(label)
            unit_state = _combine_state(explicit_unit, table_unit)
            period_state = infer_period_state(label, report_year)
            explicit_scope = infer_scope_state(label)
            scope_state = _combine_state(explicit_scope, table_scope)
            period_states[str(col_index)] = period_state
            unit_states[str(col_index)] = unit_state
            scope_states[str(col_index)] = scope_state
            cell = {
                "coordinate": [row_index, col_index],
                "origin_coordinate": [origin.origin_row, origin.origin_col],
                "origin_cell_hash": origin.cell_hash,
                "column_header": label,
                "raw_value": value,
                "raw_value_sha256": sha256_text(value),
                "numeric_status": (
                    "authorized" if authorization is not None else
                    "unauthorized" if fragments else "not_numeric"
                ),
                "authorization_ids": [authorization["authorization_id"]] if authorization else [],
                "unit": unit_state,
                "period": period_state,
                "accounting_scope": scope_state,
            }
            cells.append(cell)
            if col_index >= data_start:
                display_parts.append(f"{label}：{value}")
                ppr_edges.append(f"(row{row_index}; {label}; {value})")
                ppr_cells.append({
                    "row_index": row_index, "col_index": col_index,
                    "row_label": " / ".join(row_path), "column_label": label, "raw_value": value,
                })
        prefix = " / ".join(row_path)
        display_text = "；".join(([prefix] if prefix else []) + display_parts)
        allowed_renderings = {
            rendering
            for authorization in authorizations
            for rendering in authorization["allowed_renderings"]
        }
        answer_safe_text = safe_numeric_projection(display_text, allowed_renderings)
        unauthorized = [
            literal for literal in numeric_fragments(display_text)
            if normalize_text(literal) not in {normalize_text(value) for value in allowed_renderings}
        ]
        semantic_states = {
            "unit_by_column": unit_states,
            "period_by_column": period_states,
            "accounting_scope_by_column": scope_states,
            "fail_closed": any(
                child.get("status") != "resolved"
                for group in (unit_states, period_states, scope_states)
                for child in group.values()
            ),
        }
        evidence_unsigned = {
            "corpus_id": corpus_id, "document_id": document_id, "table_id": table_id,
            "row_index": row_index, "row_path": row_path, "cells": cells,
        }
        evidence_id = f"{corpus_id}:{table_id}:r{row_index:04d}:{semantic_sha256(evidence_unsigned)[:12]}"
        # Neutral base/value/row/header/semantic fields must remain disjoint so
        # the evaluator can perform real structural ablations without leakage.
        base_search = " ".join((caption, " ".join(section_path), nearby_text)).strip()
        value_search = " ".join(
            normalize_text(value) for value in row[data_start:] if normalize_text(value)
        )
        own_row_header = " ".join(
            normalize_text(value) for value in row[:data_start] if normalize_text(value)
        )
        semantic_search = " ".join(
            value for value in (
                *(_resolved_state_text(item) for item in unit_states.values()),
                *(_resolved_state_text(item) for item in period_states.values()),
                *(_resolved_state_text(item) for item in scope_states.values()),
            ) if value
        )
        row_record = {
            "schema_version": TABGR_V2_ROW_SCHEMA,
            "builder_version": TABGR_V2_BUILDER_VERSION,
            "record_type": "table_row",
            "evidence_id": evidence_id,
            "corpus_id": corpus_id,
            "document_id": document_id,
            "table_id": table_id,
            "table_index": int(block["table_index"]),
            "row_index": row_index,
            "heading_path": section_path,
            "active_header_rows": active_header_rows,
            "flattened_column_headers": flattened_headers,
            "row_path": row_path,
            "business_classification": {"semantic_tags": semantic_tags, "row_path": row_path},
            "cells": cells,
            "display_text": display_text,
            "answer_safe_text": answer_safe_text,
            "source_markdown": source_markdown,
            "table_line_range": table_line_range,
            "table_sha256": table_sha256,
            "numeric_authorizations": sorted(authorizations, key=lambda value: value["authorization_id"]),
            "unauthorized_numeric_values": sorted(set(unauthorized)),
            "semantic_states": semantic_states,
            "base_search_text": base_search,
            "value_search_text": value_search,
            "own_row_header_text": own_row_header,
            "single_header_text": " ".join(legacy_headers),
            "multilevel_header_text": " ".join(flattened_headers),
            "hierarchical_row_text": " ".join(row_path),
            "semantic_search_text": semantic_search,
            "legacy_flat_text": " ".join((*legacy_headers, *row)),
            "search_text": " ".join((
                base_search, value_search, own_row_header, " ".join(flattened_headers),
                " ".join(row_path), semantic_search,
            )).strip(),
        }
        rows.append(row_record)

    table_search = " ".join(
        (caption, " ".join(section_path), nearby_text, " ".join(initial_flat),
         " ".join(row["display_text"] for row in rows))
    )[:50_000]
    table_evidence_id = f"{corpus_id}:{table_id}:table:{table_sha256[:12]}"
    table_index_record = {
        "schema_version": TABGR_V2_ROW_SCHEMA,
        "builder_version": TABGR_V2_BUILDER_VERSION,
        "record_type": "table",
        "evidence_id": table_evidence_id,
        "corpus_id": corpus_id,
        "document_id": document_id,
        "table_id": table_id,
        "table_index": int(block["table_index"]),
        "heading_path": section_path,
        "row_path": [],
        "display_text": caption or f"表格 {int(block['table_index'])}",
        "answer_safe_text": safe_numeric_projection(
            caption or f"表格 {int(block['table_index'])}", ()
        ),
        "source_markdown": source_markdown,
        "table_line_range": table_line_range,
        "table_sha256": table_sha256,
        "numeric_authorizations": [],
        "unauthorized_numeric_values": [],
        "semantic_states": {"unit": table_unit, "accounting_scope": table_scope, "fail_closed": True},
        "base_search_text": " ".join((caption, " ".join(section_path), nearby_text)),
        "value_search_text": " ".join(row["value_search_text"] for row in rows)[:50_000],
        "own_row_header_text": " ".join(row["own_row_header_text"] for row in rows)[:20_000],
        "single_header_text": " ".join(legacy_headers),
        "multilevel_header_text": " ".join(initial_flat),
        "hierarchical_row_text": " ".join(" ".join(row["row_path"]) for row in rows),
        "semantic_search_text": " ".join((_resolved_state_text(table_unit), _resolved_state_text(table_scope))),
        "legacy_flat_text": " ".join((" ".join(legacy_headers), table_search)),
        "search_text": table_search,
    }
    structured = {
        "schema_version": TABGR_V2_TABLE_SCHEMA,
        "builder_version": TABGR_V2_BUILDER_VERSION,
        "corpus_id": corpus_id,
        "document_id": document_id,
        "table_id": table_id,
        "table_index": int(block["table_index"]),
        "source_markdown": source_markdown,
        "table_line_range": table_line_range,
        "raw_markdown_sha1": raw_sha1,
        "content_hash": str(block.get("content_hash") or ""),
        "source_binding": dict(source_binding),
        "table_sha256": table_sha256,
        "caption": caption,
        "nearby_text": nearby_text,
        "heading_path": section_path,
        "semantic_tags": semantic_tags,
        "matrix": matrix,
        "origin_cells": origin_cells,
        "span_validation": {
            "status": "matrix_exact",
            "parser_overwrite_count": grid.parser_overwrite_count,
            "matrix_sha256": semantic_sha256(matrix),
        },
        "header_bands": bands,
        "flattened_column_headers": initial_flat,
        "legacy_flattened_headers": legacy_headers,
        "data_start_column": data_start,
        "header_resolution": {
            "raw_resolved_columns": raw_header_resolved,
            "fallback_columns": max(0, len(initial_flat) + sum(len(matrix[row]) for row in embedded) - raw_header_resolved),
            "legacy_raw_resolved_columns": legacy_raw_header_resolved,
            "total_columns": len(initial_flat) + sum(len(matrix[row]) for row in embedded),
        },
        "unit": table_unit,
        "accounting_scope": table_scope,
        "ppr_graph": {
            "runtime_sha256": TABGR_RUNTIME_SHA256,
            "edge_list": ppr_edges,
            "cells": ppr_cells,
        },
        "row_evidence_ids": [row["evidence_id"] for row in rows],
        "parse_status": "ready",
        "failure_reason": None,
    }
    counts = {
        "row_evidence": len(rows),
        "origin_cells": len(origin_cells),
        "matrix_cells": sum(len(row) for row in matrix),
        "header_nonempty": header_nonempty,
        "header_raw_resolved": raw_header_resolved,
        "header_fallback": max(0, len(initial_flat) + sum(len(matrix[row]) for row in embedded) - raw_header_resolved),
        "legacy_header_raw_resolved": legacy_raw_header_resolved,
        "header_total": len(initial_flat) + sum(
            len(matrix[row]) for row in embedded if row < len(matrix)
        ),
        "numeric_authorizations": len(all_authorizations),
        "joined_facts": len(joined_fact_ids),
        "unauthorized_numeric_values": sum(len(row["unauthorized_numeric_values"]) for row in rows),
    }
    return structured, [table_index_record, *rows], all_authorizations, counts


class DocumentShardWriter:
    def __init__(self, index_dir: Path) -> None:
        self.index_dir = index_dir
        self.current_document: str | None = None
        self.handle: TextIO | None = None
        self.temporary: Path | None = None
        self.target: Path | None = None
        self.row_count = 0
        self.table_count = 0
        self.rows: list[dict[str, Any]] = []
        self.closed_documents: set[str] = set()

    def _close(self) -> None:
        if self.handle is None or self.temporary is None or self.target is None or self.current_document is None:
            return
        _finish_atomic(self.handle, self.temporary, self.target)
        self.rows.append({
            "schema_version": "finglmqa.type3.tabgr.document_shard.v2",
            "document_id": self.current_document,
            "shard_path": self.target.relative_to(self.index_dir).as_posix(),
            "shard_sha256": sha256_file(self.target),
            "record_count": self.row_count,
            "table_count": self.table_count,
        })
        self.closed_documents.add(self.current_document)
        self.handle = None
        self.temporary = None
        self.target = None

    def write(self, document_id: str, records: Iterable[Mapping[str, Any]]) -> None:
        if document_id != self.current_document:
            self._close()
            if document_id in self.closed_documents:
                raise RuntimeError("table blocks are not contiguous by document")
            name = hashlib.sha256(document_id.encode("utf-8")).hexdigest()[:20] + ".jsonl"
            self.target = self.index_dir / "documents" / name
            self.handle, self.temporary = _atomic_text_writer(self.target)
            self.current_document = document_id
            self.row_count = 0
            self.table_count = 0
        assert self.handle is not None
        rows = list(records)
        for row in rows:
            _write_row(self.handle, row)
            self.row_count += 1
        self.table_count += 1

    def finish(self) -> list[dict[str, Any]]:
        self._close()
        return self.rows


def build(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    profile = load_corpus_profile(args.corpus_profile)
    corpus_id = profile["corpus_id"]
    freeze = _load_input_freeze(args.expected_input_manifest, corpus_id=corpus_id)
    if profile["profile_sha256"] != freeze["corpus_profile_sha256"]:
        raise RuntimeError("corpus profile hash pin drifted")
    for path in (args.output_package, args.output_index, args.output_facts):
        _verify_output_scope(path, corpus_id=corpus_id, allow_dry_run=args.dry_run, run_dir=args.run_dir)
    if args.dry_run:
        for path in (args.output_package, args.output_index, args.output_facts):
            _safe_remove_dry_run(path, run_dir=args.run_dir)

    table_blocks_sha256 = sha256_file(args.table_blocks)
    facts_sha256 = sha256_file(args.facts)
    runtime_sha256 = sha256_file(args.tabgr_source)
    if table_blocks_sha256 != freeze["table_blocks_sha256"]:
        raise RuntimeError("Phase 3 table_blocks hash pin drifted")
    if facts_sha256 != freeze["financial_facts_sha256"]:
        raise RuntimeError("Phase 6 financial facts hash pin drifted")
    if runtime_sha256 != freeze["tabgr_runtime_sha256"] or runtime_sha256 != TABGR_RUNTIME_SHA256:
        raise RuntimeError("TabGR runtime hash pin drifted")
    before_sources = source_snapshot(profile, workspace_root=ROOT)
    documents = {row["document_id"]: row for row in profile["documents"]}
    selected_facts, ordered_facts = _selected_facts(args.facts)
    if len(ordered_facts) != freeze["expected_selected_facts"]:
        raise RuntimeError("selected fact count drifted")

    args.output_package.mkdir(parents=True, exist_ok=True)
    args.output_index.mkdir(parents=True, exist_ok=True)
    args.output_facts.mkdir(parents=True, exist_ok=True)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    structured_path = args.output_package / "structured_tables.jsonl"
    evidence_path = args.output_package / "table_row_evidence.jsonl"
    anomalies_path = args.output_package / "anomaly_audit.jsonl"
    auth_path = args.output_facts / "selected_fact_authorizations.jsonl"
    handles: list[tuple[TextIO, Path, Path]] = []
    for path in (structured_path, evidence_path, anomalies_path, auth_path):
        handle, temporary = _atomic_text_writer(path)
        handles.append((handle, temporary, path))
    structured_handle, evidence_handle, anomaly_handle, auth_handle = [row[0] for row in handles]
    shard_writer = DocumentShardWriter(args.output_index)

    counts: Counter[str] = Counter()
    joined_fact_ids: set[str] = set()
    selected_documents: set[str] = set()
    last_document_order = -1
    active_source_document: str | None = None
    active_source_text = ""
    active_source_sha256 = ""
    document_order = {row["document_id"]: index for index, row in enumerate(profile["documents"])}
    try:
        with args.table_blocks.open(encoding="utf-8") as input_handle:
            for line_number, line in enumerate(input_handle, 1):
                if not line.strip():
                    raise RuntimeError(f"blank table block row: {line_number}")
                block = json.loads(line)
                document_id = str(block.get("document_id") or "")
                if document_id not in documents:
                    raise RuntimeError("table block references a document outside corpus")
                order = document_order[document_id]
                if order < last_document_order:
                    raise RuntimeError("table blocks are not sorted by corpus document order")
                last_document_order = order
                if document_id not in selected_documents:
                    if args.max_documents and len(selected_documents) >= args.max_documents:
                        break
                    selected_documents.add(document_id)
                counts["input_tables"] += 1
                if active_source_document != document_id:
                    document = documents[document_id]
                    source_path = ROOT / profile["source_ref"] / document["source_markdown"]
                    active_source_text = source_path.read_text(encoding="utf-8")
                    active_source_sha256 = sha256_file(source_path)
                    if active_source_sha256 != document["source_sha256"]:
                        raise RuntimeError("active source hash differs from corpus profile")
                    active_source_document = document_id
                source_binding = _verify_source_binding(
                    block, source_text=active_source_text, source_sha256=active_source_sha256
                )
                status = str(block.get("parse_status") or "unknown")
                if status != "ok":
                    audit = {
                        "schema_version": "finglmqa.type3.tabgr.anomaly_audit.v2",
                        "corpus_id": corpus_id,
                        "document_id": document_id,
                        "table_id": str(block.get("table_id") or ""),
                        "table_index": int(block.get("table_index") or 0),
                        "parse_status": status,
                        "failure_reason": str(block.get("failure_reason") or status),
                        "raw_markdown_sha1": str(block.get("raw_markdown_sha1") or ""),
                        "content_hash": str(block.get("content_hash") or ""),
                        "table_line_range": _line_range(block),
                        "evidence_eligible": False,
                        "fixed_legacy_anomaly": True,
                    }
                    _write_row(anomaly_handle, audit)
                    counts["anomalies"] += 1
                    continue
                document = documents[document_id]
                structured, index_records, authorizations, table_counts = _make_table(
                    block, corpus_id=corpus_id,
                    source_markdown=_portable_source(profile, document),
                    document=document, source_binding=source_binding, fact_map=selected_facts,
                )
                _write_row(structured_handle, structured)
                for record in index_records[1:]:
                    _write_row(evidence_handle, record)
                shard_writer.write(document_id, index_records)
                for authorization in authorizations:
                    joined_fact_ids.add(str(authorization["fact_id"]))
                counts["ready_tables"] += 1
                counts.update(table_counts)
                if args.max_tables and counts["ready_tables"] >= args.max_tables:
                    break
        document_manifest_rows = shard_writer.finish()
        if not args.dry_run:
            if counts["input_tables"] != freeze["expected_input_tables"]:
                raise RuntimeError("input table count drifted")
            if (
                counts["ready_tables"] != freeze["expected_ready_tables"]
                or counts["anomalies"] != freeze["expected_anomalies"]
            ):
                raise RuntimeError("ready/anomaly table coverage is below the legacy gate")
            if len(selected_documents) != profile["document_count"]:
                raise RuntimeError("not every corpus document was built")
            if len(joined_fact_ids) != freeze["expected_selected_facts"]:
                missing = sorted({str(row["fact_id"]) for row in ordered_facts} - joined_fact_ids)
                raise RuntimeError(f"not every selected fact joined exactly: {missing[:5]!r}")
        authorization_by_id: dict[str, dict[str, Any]] = {}
        # The row stream may reference a fact only once; emit the exact joined
        # authorizations sorted independently for audit/reuse.
        for fact in ordered_facts:
            key_prefix = (
                str(fact["document_id"]), str(fact["source_table_id"]),
                int(fact["source_row_index"]), int(fact["source_col_index"]),
            )
            matches = [key for key in selected_facts if key[:4] == key_prefix]
            if len(matches) != 1:
                raise RuntimeError("selected fact exact key is ambiguous")
            raw_value = matches[0][4]
            # Find its table binding from the already joined authorization.
            # A compact map is populated from per-row authorizations below.
            # During dry runs, facts outside selected documents are omitted.
        # Re-read the compact evidence output only after its writer is closed;
        # authorization output itself is populated from collected row auths.
        # ``all_authorization_rows`` remains bounded by 4,189.
        all_authorization_rows: dict[str, dict[str, Any]] = {}
        # Authorizations were not retained globally above; reconstruct them by
        # scanning the small row-evidence temporary before atomic publication.
        evidence_handle.flush()
        with Path(handles[1][1]).open(encoding="utf-8") as temp_evidence:
            for line in temp_evidence:
                row = json.loads(line)
                for authorization in row.get("numeric_authorizations") or ():
                    all_authorization_rows[str(authorization["authorization_id"])] = authorization
        for authorization_id in sorted(all_authorization_rows):
            _write_row(auth_handle, all_authorization_rows[authorization_id])
        counts["authorization_artifacts"] = len(all_authorization_rows)
        if not args.dry_run and len(all_authorization_rows) != freeze["expected_selected_facts"]:
            raise RuntimeError("authorization artifact count differs from selected facts")
        for handle, temporary, target in handles:
            _finish_atomic(handle, temporary, target)
    except BaseException:
        for handle, temporary, _ in handles:
            try:
                handle.close()
            except Exception:
                pass
            temporary.unlink(missing_ok=True)
        raise

    document_manifest_path = args.output_index / "document_manifest.jsonl"
    document_manifest_handle, document_manifest_temp = _atomic_text_writer(document_manifest_path)
    for row in document_manifest_rows:
        _write_row(document_manifest_handle, row)
    _finish_atomic(document_manifest_handle, document_manifest_temp, document_manifest_path)

    artifact_hashes = {
        "structured_tables_sha256": sha256_file(structured_path),
        "table_row_evidence_sha256": sha256_file(evidence_path),
        "anomaly_audit_sha256": sha256_file(anomalies_path),
        "selected_fact_authorizations_sha256": sha256_file(auth_path),
        "document_manifest_sha256": sha256_file(document_manifest_path),
    }
    package_manifest = {
        "schema_version": "finglmqa.type3.tabgr.package_manifest.v2",
        "builder_version": TABGR_V2_BUILDER_VERSION,
        "corpus_id": corpus_id,
        "corpus_profile_sha256": profile["profile_sha256"],
        "input_freeze_sha256": sha256_file(args.expected_input_manifest),
        "table_blocks_sha256": table_blocks_sha256,
        "financial_facts_sha256": facts_sha256,
        "tabgr_runtime_sha256": runtime_sha256,
        "build_mode": "dry_run" if args.dry_run else "full",
        "counts": dict(sorted(counts.items())),
        "artifacts": artifact_hashes,
    }
    package_manifest["manifest_fingerprint"] = semantic_sha256(package_manifest)
    _write_json(args.output_package / "manifest.json", package_manifest)
    facts_manifest = {
        "schema_version": "finglmqa.type3.tabgr.fact_authorization_manifest.v1",
        "corpus_id": corpus_id,
        "corpus_profile_sha256": profile["profile_sha256"],
        "financial_facts_sha256": facts_sha256,
        "selected_fact_count": len(all_authorization_rows),
        "authorizations_sha256": artifact_hashes["selected_fact_authorizations_sha256"],
    }
    facts_manifest["manifest_fingerprint"] = semantic_sha256(facts_manifest)
    _write_json(args.output_facts / "manifest.json", facts_manifest)
    index_manifest = {
        "schema_version": "finglmqa.type3.tabgr.lexical_index_manifest.v2",
        "builder_version": TABGR_V2_BUILDER_VERSION,
        "corpus_id": corpus_id,
        "corpus_profile_sha256": profile["profile_sha256"],
        "tabgr_runtime_sha256": runtime_sha256,
        "document_count": len(document_manifest_rows),
        "record_count": sum(int(row["record_count"]) for row in document_manifest_rows),
        "table_count": sum(int(row["table_count"]) for row in document_manifest_rows),
        "document_manifest_sha256": artifact_hashes["document_manifest_sha256"],
        "row_evidence_sha256": artifact_hashes["table_row_evidence_sha256"],
        "document_prefilter_required": True,
        "online_source_table_reparse_allowed": False,
    }
    index_manifest["manifest_fingerprint"] = semantic_sha256(index_manifest)
    _write_json(args.output_index / "manifest.json", index_manifest)

    after_sources = source_snapshot(profile, workspace_root=ROOT)
    if before_sources != after_sources:
        raise RuntimeError("source Markdown changed during TabGR v2 build")
    peak_rss_kib = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if peak_rss_kib >= MAX_PEAK_RSS_KIB:
        raise RuntimeError("streaming build peak RSS exceeded 8 GiB")
    report = {
        "schema_version": "finglmqa.type3.tabgr.build_report.v2",
        "status": "passed",
        "corpus_id": corpus_id,
        "dry_run": bool(args.dry_run),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "peak_rss_kib": peak_rss_kib,
        "peak_rss_below_8gib": True,
        "source_unchanged": True,
        "counts": dict(sorted(counts.items())),
        "artifact_hashes": artifact_hashes,
        "package_manifest_fingerprint": package_manifest["manifest_fingerprint"],
        "index_manifest_fingerprint": index_manifest["manifest_fingerprint"],
        "stop_conditions": [],
    }
    _write_json(args.run_dir / "build_report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-profile", type=Path, default=ROOT / "data/corpus_package/type3/annual_reports_170_v1/corpus_manifest.json")
    parser.add_argument("--table-blocks", type=Path, default=ROOT / "data/corpus_package/table_blocks.jsonl")
    parser.add_argument("--facts", type=Path, default=ROOT / "data/facts/financial_facts.jsonl")
    parser.add_argument("--tabgr-source", type=Path, default=ROOT / "refs/tabgr_runtime/build_graphs/graph_to_text_triple_full.py")
    parser.add_argument("--expected-input-manifest", type=Path, default=ROOT / "data/schemas/type3/tabgr_annual_reports_170_v1_input_freeze.json")
    parser.add_argument("--output-package", type=Path, default=ROOT / "data/corpus_package/type3/annual_reports_170_v1/tabgr_table_v2")
    parser.add_argument("--output-index", type=Path, default=ROOT / "data/indexes/type3/annual_reports_170_v1/tabgr")
    parser.add_argument("--output-facts", type=Path, default=ROOT / "data/facts/type3/annual_reports_170_v1")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs/type3_a2rag_tabgr_experiment_v1/annual_reports_170_v1/phase_3")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-documents", type=int, default=0)
    parser.add_argument("--max-tables", type=int, default=0)
    args = parser.parse_args()
    if args.max_documents < 0 or args.max_tables < 0:
        parser.error("limits must be nonnegative")
    if (args.max_documents or args.max_tables) and not args.dry_run:
        parser.error("limits are allowed only with --dry-run")
    return args


if __name__ == "__main__":
    try:
        result = build(parse_args())
    except Exception as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        raise
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
