#!/usr/bin/env python3
"""Parse Phase 2 Markdown documents into text and table block packages."""

from __future__ import annotations

import argparse
import bisect
import hashlib
import html
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any


SCHEMA_PACKAGE_MANIFEST = "finglmqa.phase3.document_package_manifest.v1"
SCHEMA_METADATA = "finglmqa.phase3.document_metadata.v1"
SCHEMA_TEXT_BLOCK = "finglmqa.phase3.text_block.v1"
SCHEMA_TABLE_BLOCK = "finglmqa.phase3.table_block.v1"
SCHEMA_EXTRACTION_REPORT = "finglmqa.phase3.extraction_report.v1"
SCHEMA_COVERAGE_DIFF = "finglmqa.phase3.coverage_diff.v1"
PARSER_VERSION = "phase3-markdown-parser-v1"

TABLE_RE = re.compile(r"<table\b.*?</table\s*>", re.IGNORECASE | re.DOTALL)
TABLE_OPEN_RE = re.compile(r"<table\b", re.IGNORECASE)
TABLE_CLOSE_RE = re.compile(r"</table\s*>", re.IGNORECASE)
HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*$")
IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]+\)")
UNIT_RE = re.compile(
    r"(?:单位|金额单位|币种)\s*[:：]\s*([人民币\s]*[万千亿]?[元円]|万元|千元|亿元|元|%|％|股|人|元/股|人民币元)",
    re.IGNORECASE,
)
TAG_RULES = {
    "business": ("业务", "主营", "产品", "行业", "经营模式", "客户", "供应商"),
    "risk": ("风险", "不确定性", "可能面对", "应对措施"),
    "rd": ("研发", "技术", "专利", "创新", "研究开发"),
    "staff": ("员工", "人员", "薪酬", "职工", "教育程度"),
    "management_discussion": ("管理层讨论", "经营情况讨论与分析", "经营情况讨论", "管理层分析"),
    "financial": ("财务报告", "会计", "审计", "资产负债", "利润表", "现金流"),
    "governance": ("公司治理", "董事", "监事", "高级管理"),
    "shareholder": ("股东", "股份", "股本", "持股"),
}


def workspace_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
        fh.write("\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            json.dump(row, fh, ensure_ascii=False, sort_keys=True)
            fh.write("\n")


def rel(path: Path, root: Path) -> str:
    try:
        return path.resolve(strict=False).relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def safe_dir_name(value: str) -> str:
    return re.sub(r"[\\/:\0]+", "_", value).strip() or "unknown_document"


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def sha1_text(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8", errors="replace")).hexdigest()


def clean_cell_text(value: str) -> str:
    text = html.unescape(value)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_int_attr(value: str | None, default: int = 1) -> int:
    if value is None:
        return default
    try:
        parsed = int(str(value).strip())
    except ValueError:
        return default
    return max(1, parsed)


class TableHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[dict[str, Any]]] = []
        self.current_row: list[dict[str, Any]] | None = None
        self.current_cell: dict[str, Any] | None = None
        self.in_table = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        attr_map = {name.lower(): value for name, value in attrs}
        if tag == "table":
            self.in_table += 1
        elif tag == "tr" and self.in_table:
            self.current_row = []
        elif tag in {"td", "th"} and self.in_table:
            self.current_cell = {
                "tag": tag,
                "rowspan": parse_int_attr(attr_map.get("rowspan")),
                "colspan": parse_int_attr(attr_map.get("colspan")),
                "text_parts": [],
            }

    def handle_data(self, data: str) -> None:
        if self.current_cell is not None:
            self.current_cell["text_parts"].append(data)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"td", "th"} and self.current_cell is not None and self.current_row is not None:
            text = clean_cell_text("".join(self.current_cell.pop("text_parts", [])))
            self.current_cell["text"] = text
            self.current_row.append(self.current_cell)
            self.current_cell = None
        elif tag == "tr" and self.current_row is not None:
            if self.current_row:
                self.rows.append(self.current_row)
            self.current_row = None
        elif tag == "table" and self.in_table:
            self.in_table -= 1


def set_grid_cell(row: list[str | None], col: int, value: str) -> None:
    while len(row) <= col:
        row.append(None)
    row[col] = value


def expand_matrix(rows: list[list[dict[str, Any]]]) -> list[list[str]]:
    matrix: list[list[str]] = []
    pending: dict[int, tuple[int, str]] = {}
    for source_row in rows:
        row: list[str | None] = []
        next_pending: dict[int, tuple[int, str]] = {}
        col = 0

        def fill_pending_until(target_col: int | None = None) -> None:
            nonlocal col
            pending_cols = sorted(pending)
            for pending_col in pending_cols:
                if target_col is not None and pending_col >= target_col:
                    break
                if pending_col < col:
                    continue
                remaining, text = pending[pending_col]
                set_grid_cell(row, pending_col, text)
                if remaining > 1:
                    next_pending[pending_col] = (remaining - 1, text)
                col = pending_col + 1

        for cell in source_row:
            while col in pending:
                remaining, text = pending[col]
                set_grid_cell(row, col, text)
                if remaining > 1:
                    next_pending[col] = (remaining - 1, text)
                col += 1
            text = str(cell.get("text") or "")
            rowspan = int(cell.get("rowspan") or 1)
            colspan = int(cell.get("colspan") or 1)
            for span_col in range(col, col + colspan):
                set_grid_cell(row, span_col, text)
                if rowspan > 1:
                    next_pending[span_col] = (rowspan - 1, text)
            col += colspan
        fill_pending_until(None)
        matrix.append(["" if value is None else value for value in row])
        pending = next_pending

    while pending:
        row = []
        next_pending = {}
        for pending_col, (remaining, text) in sorted(pending.items()):
            set_grid_cell(row, pending_col, text)
            if remaining > 1:
                next_pending[pending_col] = (remaining - 1, text)
        matrix.append(["" if value is None else value for value in row])
        pending = next_pending
    width = max((len(row) for row in matrix), default=0)
    return [row + [""] * (width - len(row)) for row in matrix]


def parse_html_table(raw: str) -> dict[str, Any]:
    parser = TableHTMLParser()
    parser.feed(raw)
    parser.close()
    cell_spans: list[dict[str, Any]] = []
    for row_index, row in enumerate(parser.rows):
        for col_index, cell in enumerate(row):
            cell_spans.append({
                "source_row": row_index,
                "source_col": col_index,
                "rowspan": cell.get("rowspan", 1),
                "colspan": cell.get("colspan", 1),
                "tag": cell.get("tag"),
                "text": cell.get("text", ""),
            })
    matrix = expand_matrix(parser.rows)
    nonempty_cells = sum(1 for row in matrix for value in row if value)
    syntax_flags = []
    if re.search(r"<\s*(?:td|th|tr|table)\s*<", raw, flags=re.IGNORECASE):
        syntax_flags.append("malformed_tag")
    if not matrix:
        parse_status = "unsupported"
        failure_reason = "no_table_rows_parsed"
    elif nonempty_cells == 0:
        parse_status = "empty"
        failure_reason = "no_nonempty_cells"
    elif syntax_flags:
        parse_status = "malformed"
        failure_reason = ",".join(syntax_flags)
    else:
        parse_status = "ok"
        failure_reason = None
    return {
        "parse_status": parse_status,
        "failure_reason": failure_reason,
        "header": matrix[0] if matrix else [],
        "rows": matrix[1:] if len(matrix) > 1 else [],
        "matrix": matrix,
        "cell_spans": cell_spans,
        "stats": {
            "source_rows": len(parser.rows),
            "matrix_rows": len(matrix),
            "matrix_cols": max((len(row) for row in matrix), default=0),
            "nonempty_cells": nonempty_cells,
        },
    }


def strip_heading_marks(line: str) -> str:
    match = HEADING_RE.match(line)
    return match.group(2).strip() if match else line.strip()


def semantic_tags(*parts: str | list[str] | None) -> list[str]:
    text_parts: list[str] = []
    for part in parts:
        if part is None:
            continue
        if isinstance(part, list):
            text_parts.extend(str(item) for item in part)
        else:
            text_parts.append(str(part))
    haystack = "\n".join(text_parts)
    tags = [tag for tag, needles in TAG_RULES.items() if any(needle in haystack for needle in needles)]
    return tags


def line_starts(text: str) -> list[int]:
    starts = [0]
    for match in re.finditer(r"\n", text):
        starts.append(match.end())
    return starts


def line_for_offset(starts: list[int], offset: int) -> int:
    return bisect.bisect_right(starts, offset)  # 1-based line number


def line_spans(text: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    cursor = 0
    for line in text.splitlines(keepends=True):
        start = cursor
        end = start + len(line)
        spans.append((start, end, line))
        cursor = end
    if text and (not spans or spans[-1][1] < len(text)):
        spans.append((cursor, len(text), text[cursor:]))
    return spans


def sections_by_line(lines: list[str]) -> list[list[str]]:
    sections: list[list[str]] = [[] for _ in range(len(lines) + 1)]
    stack: list[str] = []
    for line_no, line in enumerate(lines, start=1):
        match = HEADING_RE.match(line.strip())
        if match:
            level = len(match.group(1))
            title = match.group(2).strip()
            stack = stack[: level - 1]
            stack.append(title)
        sections[line_no] = list(stack)
    return sections


def remove_table_ranges(line_start: int, line_end: int, line: str, table_ranges: list[tuple[int, int]]) -> str:
    pieces: list[str] = []
    cursor = line_start
    for table_start, table_end in table_ranges:
        if table_end <= line_start:
            continue
        if table_start >= line_end:
            break
        if table_start > cursor:
            pieces.append(line[cursor - line_start : table_start - line_start])
        cursor = max(cursor, min(table_end, line_end))
    if cursor < line_end:
        pieces.append(line[cursor - line_start :])
    return "".join(pieces)


def nearest_caption(clean_lines: list[str], start_line: int, window: int = 6) -> str | None:
    for line_no in range(start_line - 1, max(0, start_line - window) - 1, -1):
        stripped = clean_lines[line_no - 1].strip() if line_no - 1 < len(clean_lines) else ""
        if not stripped:
            continue
        if stripped.startswith("#"):
            return strip_heading_marks(stripped)
        return stripped[:300]
    return None


def nearby_text(clean_lines: list[str], start_line: int, end_line: int, window: int = 3) -> str:
    snippets: list[str] = []
    for line_no in range(max(1, start_line - window), start_line):
        stripped = clean_lines[line_no - 1].strip()
        if stripped:
            snippets.append(strip_heading_marks(stripped))
    for line_no in range(end_line + 1, min(len(clean_lines), end_line + window) + 1):
        stripped = clean_lines[line_no - 1].strip()
        if stripped:
            snippets.append(strip_heading_marks(stripped))
    return "\n".join(snippets)[:1200]


def unit_hint(*parts: str | None) -> str | None:
    text = "\n".join(part for part in parts if part)
    match = UNIT_RE.search(text[:5000])
    if match:
        return match.group(0).strip()
    loose = re.search(r"单位\s*[:：]\s*([^\s，。；;、<]{1,20})", text[:5000])
    return loose.group(0).strip() if loose else None


def malformed_table_samples(text: str, starts: list[int], matched_ranges: list[tuple[int, int]], limit: int = 20) -> list[dict[str, Any]]:
    covered = [False] * len(re.findall(TABLE_OPEN_RE, text))
    samples: list[dict[str, Any]] = []
    match_index = 0
    for open_match in TABLE_OPEN_RE.finditer(text):
        offset = open_match.start()
        is_covered = any(start <= offset < end for start, end in matched_ranges)
        if is_covered:
            covered[match_index] = True
        elif len(samples) < limit:
            line_no = line_for_offset(starts, offset)
            snippet = text[offset : offset + 500].replace("\n", "\\n")
            samples.append({"line": line_no, "reason": "table_open_without_matched_close", "snippet": snippet})
        match_index += 1
    return samples


def extract_tables(
    doc: dict[str, Any],
    text: str,
    lines: list[str],
    starts: list[int],
    sections: list[list[str]],
    clean_lines: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    matches = list(TABLE_RE.finditer(text))
    table_blocks: list[dict[str, Any]] = []
    parse_status_counts: Counter[str] = Counter()
    non_ok_samples: list[dict[str, Any]] = []
    for ordinal, match in enumerate(matches, start=1):
        raw = match.group(0)
        start_line = line_for_offset(starts, match.start())
        end_line = line_for_offset(starts, max(match.start(), match.end() - 1))
        raw_sha1 = sha1_text(raw)
        content_hash = sha256_text(raw)
        table_id = f"{doc['document_id']}_table_{ordinal:04d}_{raw_sha1[:10]}"
        section_path = sections[start_line] if start_line < len(sections) else []
        caption = nearest_caption(clean_lines, start_line)
        nearby = nearby_text(clean_lines, start_line, end_line)
        parsed = parse_html_table(raw)
        parse_status_counts[parsed["parse_status"]] += 1
        block = {
            "schema_version": SCHEMA_TABLE_BLOCK,
            "parser_version": PARSER_VERSION,
            "document_id": doc["document_id"],
            "table_id": table_id,
            "table_index": ordinal,
            "stock_code": doc.get("stock_code"),
            "stock_symbol": doc.get("stock_symbol"),
            "stock_name": doc.get("stock_name"),
            "company_full": doc.get("company_full"),
            "report_year": doc.get("report_year"),
            "source_markdown": doc.get("markdown_path"),
            "markdown_path": doc.get("markdown_path"),
            "resolved_source_path": doc.get("resolved_source_path"),
            "line_range": [start_line, end_line],
            "char_range": [match.start(), match.end()],
            "section_path": section_path,
            "caption": caption,
            "nearby_text": nearby,
            "unit_hint": unit_hint(caption, nearby, raw),
            "raw_format": "html_table",
            "raw_markdown": raw,
            "raw_markdown_sha1": raw_sha1,
            "content_hash": content_hash,
            "parse_status": parsed["parse_status"],
            "failure_reason": parsed["failure_reason"],
            "header": parsed["header"],
            "rows": parsed["rows"],
            "matrix": parsed["matrix"],
            "cell_spans": parsed["cell_spans"],
            "stats": parsed["stats"],
            "metadata": {
                "doc_id": doc["document_id"],
                "ticker": doc.get("stock_symbol"),
                "stock_code": doc.get("stock_code"),
                "stock_name": doc.get("stock_name"),
                "company_full": doc.get("company_full"),
                "report_year": doc.get("report_year"),
                "source_markdown": doc.get("markdown_path"),
            },
            "source_title": f"{doc.get('stock_symbol') or doc.get('stock_code')} 年报Markdown表格 {ordinal}",
            "semantic_tags": semantic_tags(section_path, caption, nearby),
        }
        table_blocks.append(block)
        if parsed["parse_status"] != "ok" and len(non_ok_samples) < 20:
            non_ok_samples.append({
                "table_id": table_id,
                "table_index": ordinal,
                "line_range": [start_line, end_line],
                "parse_status": parsed["parse_status"],
                "failure_reason": parsed["failure_reason"],
                "raw_markdown_sha1": raw_sha1,
                "snippet": raw[:500].replace("\n", "\\n"),
            })

    open_count = len(TABLE_OPEN_RE.findall(text))
    close_count = len(TABLE_CLOSE_RE.findall(text))
    matched_ranges = [(match.start(), match.end()) for match in matches]
    unmatched_count = max(open_count, close_count) - len(matches)
    unmatched_samples = malformed_table_samples(text, starts, matched_ranges)
    diagnostics = {
        "table_open_tag_count": open_count,
        "table_close_tag_count": close_count,
        "matched_table_count": len(matches),
        "parse_status_counts": dict(parse_status_counts),
        "malformed_or_unsupported_table_count": unmatched_count + sum(
            count for status, count in parse_status_counts.items() if status != "ok"
        ),
        "malformed_or_unsupported_samples": unmatched_samples + non_ok_samples,
    }
    return table_blocks, diagnostics


def flush_text_block(
    blocks: list[dict[str, Any]],
    doc: dict[str, Any],
    buffer: list[tuple[int, str, list[str]]],
    block_index: int,
) -> int:
    if not buffer:
        return block_index
    text = "\n".join(item[1].strip() for item in buffer if item[1].strip()).strip()
    if not text:
        buffer.clear()
        return block_index
    line_range = [buffer[0][0], buffer[-1][0]]
    section_path = buffer[0][2]
    block_index += 1
    block_id = f"{doc['document_id']}_text_{block_index:05d}"
    blocks.append({
        "schema_version": SCHEMA_TEXT_BLOCK,
        "parser_version": PARSER_VERSION,
        "document_id": doc["document_id"],
        "block_id": block_id,
        "text_block_id": block_id,
        "block_index": block_index,
        "stock_code": doc.get("stock_code"),
        "stock_symbol": doc.get("stock_symbol"),
        "stock_name": doc.get("stock_name"),
        "company_full": doc.get("company_full"),
        "report_year": doc.get("report_year"),
        "source_markdown": doc.get("markdown_path"),
        "resolved_source_path": doc.get("resolved_source_path"),
        "line_range": line_range,
        "section_path": section_path,
        "semantic_tags": semantic_tags(section_path, text),
        "text": text,
        "text_hash": sha256_text(text),
        "linked_table_ids": [],
    })
    buffer.clear()
    return block_index


def extract_text_blocks(
    doc: dict[str, Any],
    text: str,
    line_span_rows: list[tuple[int, int, str]],
    table_ranges: list[tuple[int, int]],
    sections: list[list[str]],
    max_chars: int,
) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    buffer: list[tuple[int, str, list[str]]] = []
    block_index = 0
    current_chars = 0
    previous_section: list[str] | None = None

    for line_no, (start, end, raw_line) in enumerate(line_span_rows, start=1):
        cleaned = remove_table_ranges(start, end, raw_line, table_ranges).strip()
        section_path = sections[line_no] if line_no < len(sections) else []
        is_heading = bool(HEADING_RE.match(cleaned))
        if previous_section is not None and section_path != previous_section:
            block_index = flush_text_block(blocks, doc, buffer, block_index)
            current_chars = 0
        previous_section = list(section_path)

        if not cleaned:
            block_index = flush_text_block(blocks, doc, buffer, block_index)
            current_chars = 0
            continue

        text_line = strip_heading_marks(cleaned) if is_heading else cleaned
        if buffer and current_chars + len(text_line) > max_chars:
            block_index = flush_text_block(blocks, doc, buffer, block_index)
            current_chars = 0
        buffer.append((line_no, text_line, list(section_path)))
        current_chars += len(text_line) + 1

    flush_text_block(blocks, doc, buffer, block_index)
    return blocks


def link_nearby_tables(text_blocks: list[dict[str, Any]], table_blocks: list[dict[str, Any]], max_line_gap: int = 3) -> None:
    for text_block in text_blocks:
        start, end = text_block["line_range"]
        linked: list[str] = []
        for table_block in table_blocks:
            table_start, table_end = table_block["line_range"]
            same_section = table_block.get("section_path") == text_block.get("section_path")
            gap = min(abs(table_start - end), abs(start - table_end))
            if same_section and gap <= max_line_gap:
                linked.append(table_block["table_id"])
        text_block["linked_table_ids"] = linked[:20]


def parse_document(doc: dict[str, Any], output_root: Path, root: Path, max_text_chars: int) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    source_path = Path(str(doc["markdown_path"]))
    text = source_path.read_text(encoding="utf-8", errors="replace")
    spans = line_spans(text)
    lines = [row[2] for row in spans]
    starts = line_starts(text)
    sections = sections_by_line(lines)
    matched_table_ranges = [(match.start(), match.end()) for match in TABLE_RE.finditer(text)]
    clean_lines = [
        remove_table_ranges(start, end, line, matched_table_ranges)
        for start, end, line in spans
    ]

    table_blocks, table_diagnostics = extract_tables(doc, text, lines, starts, sections, clean_lines)
    text_blocks = extract_text_blocks(doc, text, spans, matched_table_ranges, sections, max_text_chars)
    link_nearby_tables(text_blocks, table_blocks)

    package_dir = output_root / "documents" / safe_dir_name(str(doc["document_id"]))
    metadata_path = package_dir / "metadata.json"
    manifest_path = package_dir / "manifest.json"
    text_blocks_path = package_dir / "text_blocks.jsonl"
    table_blocks_path = package_dir / "table_blocks.jsonl"
    extraction_report_path = package_dir / "extraction_report.json"

    heading_count = sum(1 for line in lines if HEADING_RE.match(line.strip()))
    image_count = len(IMAGE_RE.findall(text))
    metadata = {
        "schema_version": SCHEMA_METADATA,
        "parser_version": PARSER_VERSION,
        "document": doc,
        "source_markdown": doc.get("markdown_path"),
        "resolved_source_path": doc.get("resolved_source_path"),
        "line_count": len(lines),
        "heading_count": heading_count,
        "image_count": image_count,
        "text_block_count": len(text_blocks),
        "table_block_count": len(table_blocks),
        "content_sha256": sha256_text(text),
    }
    extraction_report = {
        "schema_version": SCHEMA_EXTRACTION_REPORT,
        "parser_version": PARSER_VERSION,
        "document_id": doc["document_id"],
        "status": "parsed",
        "source_markdown": doc.get("markdown_path"),
        "resolved_source_path": doc.get("resolved_source_path"),
        "line_count": len(lines),
        "heading_count": heading_count,
        "image_count": image_count,
        "text_block_count": len(text_blocks),
        "table_block_count": len(table_blocks),
        **table_diagnostics,
        "warnings": list(doc.get("warnings") or []),
        "errors": [],
    }
    package_manifest = {
        "schema_version": SCHEMA_PACKAGE_MANIFEST,
        "parser_version": PARSER_VERSION,
        "document_id": doc["document_id"],
        "status": "parsed",
        "source_markdown": doc.get("markdown_path"),
        "resolved_source_path": doc.get("resolved_source_path"),
        "artifacts": {
            "manifest": rel(manifest_path, root),
            "metadata": rel(metadata_path, root),
            "text_blocks": rel(text_blocks_path, root),
            "table_blocks": rel(table_blocks_path, root),
            "extraction_report": rel(extraction_report_path, root),
        },
        "counts": {
            "text_blocks": len(text_blocks),
            "table_blocks": len(table_blocks),
            "malformed_or_unsupported_tables": table_diagnostics["malformed_or_unsupported_table_count"],
        },
    }

    write_json(metadata_path, metadata)
    write_jsonl(text_blocks_path, text_blocks)
    write_jsonl(table_blocks_path, table_blocks)
    write_json(extraction_report_path, extraction_report)
    write_json(manifest_path, package_manifest)
    return extraction_report, text_blocks, table_blocks


def load_old_table_baseline(path: Path) -> tuple[Counter[str], dict[str, Counter[str]], dict[str, list[dict[str, Any]]]]:
    counts: Counter[str] = Counter()
    fingerprints_by_doc: dict[str, Counter[str]] = defaultdict(Counter)
    by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            record = json.loads(line)
            doc_id = record.get("doc_id") or (record.get("metadata") or {}).get("doc_id")
            if not doc_id:
                continue
            counts[str(doc_id)] += 1
            fingerprint = record.get("fingerprint")
            if fingerprint:
                fingerprints_by_doc[str(doc_id)][str(fingerprint)] += 1
            if len(by_doc[str(doc_id)]) < 1000:
                by_doc[str(doc_id)].append({
                    "doc_id": doc_id,
                    "table_id": record.get("table_id"),
                    "table_index": record.get("table_index"),
                    "fingerprint": record.get("fingerprint"),
                    "source_title": record.get("source_title"),
                })
    return counts, fingerprints_by_doc, by_doc


def build_coverage_diff(
    docs: list[dict[str, Any]],
    table_blocks: list[dict[str, Any]],
    old_counts: Counter[str],
    old_fingerprints_by_doc: dict[str, Counter[str]],
    old_samples_by_doc: dict[str, list[dict[str, Any]]],
    root: Path,
    run_reports_dir: Path,
) -> dict[str, Any]:
    new_by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    new_fingerprints_by_doc: dict[str, Counter[str]] = defaultdict(Counter)
    for block in table_blocks:
        doc_id = str(block["document_id"])
        new_by_doc[doc_id].append(block)
        new_fingerprints_by_doc[doc_id][str(block["raw_markdown_sha1"])] += 1

    rows: list[dict[str, Any]] = []
    missing_in_old_samples: list[dict[str, Any]] = []
    old_without_current_samples: list[dict[str, Any]] = []
    fingerprint_matched_total = 0
    missing_in_old_total = 0
    old_without_current_total = 0
    for doc in docs:
        doc_id = str(doc["document_id"])
        new_count = len(new_by_doc.get(doc_id, []))
        old_count = int(old_counts.get(doc_id, 0))
        delta = new_count - old_count
        new_fingerprints = new_fingerprints_by_doc.get(doc_id, Counter())
        old_fingerprints = old_fingerprints_by_doc.get(doc_id, Counter())
        matched_count = sum(min(new_fingerprints[key], old_fingerprints[key]) for key in set(new_fingerprints) | set(old_fingerprints))
        missing_counter = new_fingerprints - old_fingerprints
        old_extra_counter = old_fingerprints - new_fingerprints
        missing_count = sum(missing_counter.values())
        old_extra_count = sum(old_extra_counter.values())
        fingerprint_matched_total += matched_count
        missing_in_old_total += missing_count
        old_without_current_total += old_extra_count
        if missing_count == 0 and old_extra_count == 0:
            status = "match"
        elif old_extra_count == 0 and missing_count > 0:
            status = "markdown_gt_old_jsonl"
        elif missing_count == 0 and old_extra_count > 0:
            status = "old_jsonl_gt_markdown"
        else:
            status = "fingerprint_mismatch"
        rows.append({
            "document_id": doc_id,
            "stock_code": doc.get("stock_code"),
            "stock_name": doc.get("stock_name"),
            "report_year": doc.get("report_year"),
            "markdown_table_blocks": new_count,
            "old_jsonl_table_records": old_count,
            "delta": delta,
            "fingerprint_matched": matched_count,
            "missing_in_old_jsonl": missing_count,
            "old_jsonl_without_current_markdown_match": old_extra_count,
            "status": status,
        })
        if missing_count > 0:
            remaining = Counter(missing_counter)
            for block in new_by_doc[doc_id]:
                fingerprint = str(block["raw_markdown_sha1"])
                if remaining[fingerprint] <= 0:
                    continue
                missing_in_old_samples.append({
                    "document_id": doc_id,
                    "table_id": block["table_id"],
                    "table_index": block["table_index"],
                    "line_range": block["line_range"],
                    "caption": block.get("caption"),
                    "parse_status": block.get("parse_status"),
                    "failure_reason": block.get("failure_reason"),
                    "raw_markdown_sha1": fingerprint,
                    "content_hash": block["content_hash"],
                    "snippet": block["raw_markdown"][:500].replace("\n", "\\n"),
                })
                remaining[fingerprint] -= 1
        if old_extra_count > 0:
            remaining_old = Counter(old_extra_counter)
            for sample in old_samples_by_doc.get(doc_id, []):
                fingerprint = str(sample.get("fingerprint") or "")
                if remaining_old[fingerprint] <= 0:
                    continue
                old_without_current_samples.append(sample)
                remaining_old[fingerprint] -= 1

    write_jsonl(run_reports_dir / "coverage_diff_by_document.jsonl", rows)
    write_jsonl(run_reports_dir / "missing_in_old_jsonl_samples.jsonl", missing_in_old_samples)
    write_jsonl(run_reports_dir / "old_jsonl_without_current_markdown_match_samples.jsonl", old_without_current_samples)

    markdown_table_blocks = sum(len(new_by_doc.get(str(doc["document_id"]), [])) for doc in docs)
    old_jsonl_table_records = sum(old_counts.values())
    diff = {
        "schema_version": SCHEMA_COVERAGE_DIFF,
        "parser_version": PARSER_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "baseline": "refs/combined_tables.jsonl",
        "current": "data/corpus_package/table_blocks.jsonl",
        "summary": {
            "documents": len(docs),
            "markdown_table_blocks": markdown_table_blocks,
            "old_jsonl_table_records": old_jsonl_table_records,
            "delta": markdown_table_blocks - old_jsonl_table_records,
            "fingerprint_matched_table_records": fingerprint_matched_total,
            "missing_in_old_jsonl_tables": missing_in_old_total,
            "old_jsonl_without_current_markdown_match_tables": old_without_current_total,
            "documents_exact_match": sum(1 for row in rows if row["status"] == "match"),
            "documents_markdown_gt_old_jsonl": sum(1 for row in rows if row["status"] == "markdown_gt_old_jsonl"),
            "documents_old_jsonl_gt_markdown": sum(1 for row in rows if row["status"] == "old_jsonl_gt_markdown"),
            "documents_with_fingerprint_mismatch": sum(1 for row in rows if row["status"] == "fingerprint_mismatch"),
        },
        "artifacts": {
            "by_document": rel(run_reports_dir / "coverage_diff_by_document.jsonl", root),
            "missing_in_old_jsonl_samples": rel(run_reports_dir / "missing_in_old_jsonl_samples.jsonl", root),
            "old_jsonl_without_current_markdown_match_samples": rel(run_reports_dir / "old_jsonl_without_current_markdown_match_samples.jsonl", root),
        },
        "rows_sample": rows[:20],
    }
    write_json(run_reports_dir / "coverage_diff.json", diff)
    return diff


def phase_report_markdown(report: dict[str, Any], coverage_diff: dict[str, Any]) -> str:
    summary = report["summary"]
    coverage = coverage_diff["summary"]
    lines = [
        "# Phase 3 Report",
        "",
        "## Date",
        report["generated_at_utc"],
        "",
        "## Environment",
        f"- Workspace: `{report['workspace_root']}`",
        f"- Parser: `{PARSER_VERSION}`",
        "",
        "## Inputs",
    ]
    for label, path in report["inputs"].items():
        lines.append(f"- {label}: `{path}`")
    lines.extend([
        "",
        "## Generated Artifacts",
    ])
    for label, path in report["outputs"].items():
        lines.append(f"- {label}: `{path}`")
    lines.extend([
        "",
        "## Verification Results",
        f"- Documents requested: {summary['documents_requested']}",
        f"- Documents parsed: {summary['documents_parsed']}",
        f"- Documents failed: {summary['documents_failed']}",
        f"- Text blocks: {summary['text_blocks']}",
        f"- Table blocks: {summary['table_blocks']}",
        f"- Malformed or unsupported tables: {summary['malformed_or_unsupported_tables']}",
        f"- Markdown table blocks: {coverage['markdown_table_blocks']}",
        f"- Old JSONL table records: {coverage['old_jsonl_table_records']}",
        f"- Coverage delta: {coverage['delta']}",
        f"- Fingerprint matched table records: {coverage['fingerprint_matched_table_records']}",
        f"- Missing in old JSONL tables: {coverage['missing_in_old_jsonl_tables']}",
        f"- Old JSONL without current Markdown match tables: {coverage['old_jsonl_without_current_markdown_match_tables']}",
        f"- Exact-match documents: {coverage['documents_exact_match']}",
        f"- Markdown > old JSONL documents: {coverage['documents_markdown_gt_old_jsonl']}",
        f"- Old JSONL > Markdown documents: {coverage['documents_old_jsonl_gt_markdown']}",
        f"- Fingerprint-mismatch documents: {coverage['documents_with_fingerprint_mismatch']}",
        "",
        "## Issues Encountered",
    ])
    if report["errors"]:
        for item in report["errors"][:20]:
            lines.append(f"- {item['document_id']}: {item['error']}")
    else:
        lines.append("- None.")
    lines.extend([
        "",
        "## Decision",
        "- continue",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    root = workspace_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=root / "data/corpus_package/corpus_manifest.json")
    parser.add_argument("--combined-tables", type=Path, default=root / "refs/combined_tables.jsonl")
    parser.add_argument("--output-dir", type=Path, default=root / "data/corpus_package")
    parser.add_argument("--run-dir", type=Path, default=root / "runs/phase_03")
    parser.add_argument("--max-text-chars", type=int, default=1800)
    args = parser.parse_args()

    generated_at = datetime.now(UTC).isoformat(timespec="seconds")
    manifest = load_json(args.manifest)
    docs = list(manifest.get("documents") or [])
    old_counts, old_fingerprints_by_doc, old_samples_by_doc = load_old_table_baseline(args.combined_tables)

    all_text_blocks: list[dict[str, Any]] = []
    all_table_blocks: list[dict[str, Any]] = []
    extraction_reports: list[dict[str, Any]] = []
    package_index: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    malformed_samples: list[dict[str, Any]] = []

    for doc in docs:
        try:
            extraction_report, text_blocks, table_blocks = parse_document(doc, args.output_dir, root, args.max_text_chars)
            extraction_reports.append(extraction_report)
            all_text_blocks.extend(text_blocks)
            all_table_blocks.extend(table_blocks)
            package_index.append({
                "document_id": doc["document_id"],
                "status": "parsed",
                "package_dir": rel(args.output_dir / "documents" / safe_dir_name(str(doc["document_id"])), root),
                "text_block_count": len(text_blocks),
                "table_block_count": len(table_blocks),
            })
            for sample in extraction_report.get("malformed_or_unsupported_samples", []):
                malformed_samples.append({"document_id": doc["document_id"], **sample})
        except Exception as exc:  # keep batch processing resilient
            errors.append({"document_id": doc.get("document_id"), "error": f"{type(exc).__name__}: {exc}"})
            package_index.append({
                "document_id": doc.get("document_id"),
                "status": "failed",
                "package_dir": None,
                "text_block_count": 0,
                "table_block_count": 0,
            })

    reports_dir = args.run_dir / "reports"
    outputs = {
        "document_packages": rel(args.output_dir / "documents", root),
        "document_package_index": rel(args.output_dir / "document_packages_index.jsonl", root),
        "text_blocks": rel(args.output_dir / "text_blocks.jsonl", root),
        "table_blocks": rel(args.output_dir / "table_blocks.jsonl", root),
        "extraction_reports": rel(args.run_dir / "reports/extraction_reports.jsonl", root),
        "coverage_diff": rel(args.run_dir / "reports/coverage_diff.json", root),
        "phase_report": rel(args.run_dir / "phase_03_report.md", root),
    }

    write_jsonl(args.output_dir / "text_blocks.jsonl", all_text_blocks)
    write_jsonl(args.output_dir / "table_blocks.jsonl", all_table_blocks)
    write_jsonl(args.output_dir / "document_packages_index.jsonl", package_index)
    write_jsonl(reports_dir / "extraction_reports.jsonl", extraction_reports)
    write_jsonl(reports_dir / "malformed_or_unsupported_table_samples.jsonl", malformed_samples)
    coverage_diff = build_coverage_diff(
        docs,
        all_table_blocks,
        old_counts,
        old_fingerprints_by_doc,
        old_samples_by_doc,
        root,
        reports_dir,
    )

    parse_status_counts: Counter[str] = Counter()
    malformed_or_unsupported_count = 0
    for extraction_report in extraction_reports:
        parse_status_counts.update(extraction_report.get("parse_status_counts") or {})
        malformed_or_unsupported_count += int(extraction_report.get("malformed_or_unsupported_table_count") or 0)

    summary = {
        "documents_requested": len(docs),
        "documents_parsed": len(extraction_reports),
        "documents_failed": len(errors),
        "text_blocks": len(all_text_blocks),
        "table_blocks": len(all_table_blocks),
        "parse_status_counts": dict(parse_status_counts),
        "malformed_or_unsupported_tables": malformed_or_unsupported_count,
        "malformed_or_unsupported_sample_count": len(malformed_samples),
    }
    report = {
        "schema_version": "finglmqa.phase3.run_report.v1",
        "parser_version": PARSER_VERSION,
        "generated_at_utc": generated_at,
        "command": " ".join(sys.argv),
        "workspace_root": root.as_posix(),
        "inputs": {
            "phase2_manifest": rel(args.manifest, root),
            "combined_tables_baseline": rel(args.combined_tables, root),
        },
        "outputs": outputs,
        "summary": summary,
        "errors": errors,
    }
    write_json(args.run_dir / "phase_03_run_report.json", report)
    (args.run_dir / "phase_03_report.md").parent.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "phase_03_report.md").write_text(phase_report_markdown(report, coverage_diff), encoding="utf-8")

    print(json.dumps({"summary": summary, "coverage": coverage_diff["summary"], "outputs": outputs}, ensure_ascii=False, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
