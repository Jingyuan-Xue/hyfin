#!/usr/bin/env python3
"""Build the experimental row-level and mixed-narrative table evidence index.

The builder is deliberately separate from the frozen Phase 7/8 artifacts.  It
reads Phase 3/4 source records, writes a new canonical JSONL index, and never
modifies an existing pipeline, schema, or manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, (ROOT / "src").as_posix())

from finglmqa.table_evidence import (  # noqa: E402
    BUILDER_VERSION,
    SCHEMA_VERSION,
    TableEvidenceError,
    canonical_json,
    fragment_identity,
    fragment_sort_key,
    meaningful_narrative,
    portable_source_path,
    safe_unit_source,
    seal_fragment,
    sha256_text,
    write_canonical_jsonl,
)


DEFAULT_EVIDENCE = ROOT / "data/corpus_package/evidence_chunks.jsonl"
DEFAULT_TABLE_CELLS = ROOT / "data/corpus_package/table_cells.jsonl"
DEFAULT_TABLE_CORPUS = ROOT / "data/corpus_package/tabgr_table_corpus.jsonl"
DEFAULT_DOCUMENTS_DIR = ROOT / "data/corpus_package/documents"
DEFAULT_EXCLUSION_AUDIT = ROOT / "runs/phase_07/reports/table_exclusion_audit.json"
DEFAULT_OUTPUT = ROOT / "runs/table_evidence_experiment/table_evidence_fragments.jsonl"
DEFAULT_REPORT = ROOT / "runs/table_evidence_experiment/build_report.json"
REPORT_SCHEMA = "finglmqa.experimental.table_evidence_build_report.v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TableEvidenceError(f"{path} must contain a JSON object")
    return value


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise TableEvidenceError(f"{path}:{line_number} is not valid JSON") from exc
            if not isinstance(row, dict):
                raise TableEvidenceError(f"{path}:{line_number} must contain an object")
            yield row


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_document_allow_list(evidence_path: Path) -> dict[str, dict[str, Any]]:
    documents: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(evidence_path):
        document_id = str(row.get("document_id") or "")
        if not document_id:
            raise TableEvidenceError("Phase 7 evidence row lacks document_id")
        metadata = {
            "document_id": document_id,
            "company_full": str(row.get("company_full") or ""),
            "company_name": str(row.get("company_name") or ""),
            "stock_code": str(row.get("stock_code") or ""),
            "report_year": int(row["report_year"]),
        }
        prior = documents.setdefault(document_id, metadata)
        if prior != metadata:
            raise TableEvidenceError(f"Phase 7 document identity conflict: {document_id}")
    if not documents:
        raise TableEvidenceError("Phase 7 evidence allow-list is empty")
    return documents


def load_table_metadata(
    path: Path,
    documents: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    tables: dict[str, dict[str, Any]] = {}
    previous: tuple[str, int, str] | None = None
    for row in iter_jsonl(path):
        document_id = str(row.get("document_id") or "")
        table_id = str(row.get("table_id") or "")
        table_index = row.get("table_index")
        if document_id not in documents:
            raise TableEvidenceError(f"table is outside Phase 7 document allow-list: {document_id}")
        if not table_id or isinstance(table_index, bool) or not isinstance(table_index, int):
            raise TableEvidenceError("table corpus identity is invalid")
        key = (document_id, table_index, table_id)
        if previous is not None and key <= previous:
            raise TableEvidenceError("table corpus is not strictly sorted")
        previous = key
        header = row.get("header") or []
        if not isinstance(header, list) or any(not isinstance(value, str) for value in header):
            raise TableEvidenceError(f"table header is invalid: {table_id}")
        content_hash = str(row.get("content_hash") or "")
        if len(content_hash) != 64:
            raise TableEvidenceError(f"table content hash is invalid: {table_id}")
        table_meta = {
            "document_id": document_id,
            "table_id": table_id,
            "table_index": table_index,
            "caption": str(row.get("caption") or row.get("table_caption") or ""),
            "header_path": [str(value) for value in header],
            "section_path": [str(value) for value in row.get("section_path") or []],
            "source_line_range": list(row.get("line_range") or []),
            "source_markdown": portable_source_path(row.get("source_markdown"), ROOT),
            "content_hash": content_hash,
            "unit_source": safe_unit_source(row.get("unit_hint")),
        }
        if table_id in tables:
            raise TableEvidenceError(f"duplicate table_id: {table_id}")
        tables[table_id] = table_meta
    if not tables:
        raise TableEvidenceError("table corpus is empty")
    return tables


def _year_source(year: int) -> dict[str, str]:
    return {"kind": "report_metadata", "value": str(year)}


def _row_content(
    caption: str,
    row_label: str,
    labels: Sequence[str],
    values: Sequence[str],
) -> str:
    pairs: list[str] = []
    for label, value in zip(labels, values):
        value = value.strip()
        if not value:
            continue
        pairs.append(f"{label.strip()}={value}" if label.strip() and label.strip() != value else value)
    prefix = "；".join(value for value in (caption.strip(), row_label.strip()) if value)
    body = "；".join(pairs)
    return "；".join(value for value in (prefix, body) if value)


def build_table_row_fragment(
    cells: Sequence[Mapping[str, Any]],
    *,
    table: Mapping[str, Any],
    document: Mapping[str, Any],
) -> dict[str, Any]:
    if not cells:
        raise TableEvidenceError("cannot build an empty table row fragment")
    ordered = sorted(cells, key=lambda value: int(value["col_index"]))
    first = ordered[0]
    row_index = int(first["row_index"])
    table_id = str(first["table_id"])
    coordinates = [[row_index, int(cell["col_index"])] for cell in ordered]
    if len({tuple(value) for value in coordinates}) != len(coordinates):
        raise TableEvidenceError(f"duplicate table cell coordinate: {table_id} row {row_index}")
    values = [str(cell.get("raw_value") or "") for cell in ordered]
    labels = [str(cell.get("column_label") or "") for cell in ordered]
    row_labels = [str(cell.get("row_label") or "").strip() for cell in ordered]
    distinct_row_labels = {value for value in row_labels if value}
    if len(distinct_row_labels) > 1:
        raise TableEvidenceError(f"table row has conflicting row labels: {table_id} row {row_index}")
    row_label = next(iter(distinct_row_labels), values[0].strip() if values else "")
    line_ranges = [cell.get("line_range") for cell in ordered]
    if any(
        not isinstance(value, list)
        or len(value) != 2
        or any(isinstance(part, bool) or not isinstance(part, int) or part < 1 for part in value)
        for value in line_ranges
    ):
        raise TableEvidenceError(f"table cell line range is invalid: {table_id} row {row_index}")
    line_range = [min(value[0] for value in line_ranges), max(value[1] for value in line_ranges)]
    sources = {portable_source_path(cell.get("source_markdown"), ROOT) for cell in ordered}
    if sources != {table["source_markdown"]}:
        raise TableEvidenceError(f"table cell source does not match table corpus: {table_id}")
    source_versions = {str(cell.get("schema_version") or "") for cell in ordered}
    if source_versions != {"finglmqa.phase4.table_cell.v1"}:
        raise TableEvidenceError(f"unexpected table cell schema: {table_id}")
    content = _row_content(str(table["caption"]), row_label, labels, values)
    if not content:
        raise TableEvidenceError(f"table row has no retrievable content: {table_id} row {row_index}")
    fragment = {
        "schema_version": SCHEMA_VERSION,
        "builder_version": BUILDER_VERSION,
        "fragment_id": fragment_identity(
            fragment_kind="table_row",
            document_id=str(document["document_id"]),
            table_id=table_id,
            row_or_block=row_index,
        ),
        "fragment_kind": "table_row",
        "document_id": str(document["document_id"]),
        "company_full": str(document["company_full"]),
        "company_name": str(document["company_name"]),
        "stock_code": str(document["stock_code"]),
        "report_year": int(document["report_year"]),
        "table_id": table_id,
        "table_index": int(table["table_index"]),
        "section_path": list(table["section_path"]),
        "caption": str(table["caption"]),
        "header_path": list(table["header_path"]),
        "row_index": row_index,
        "row_label": row_label,
        "column_labels": labels,
        "cell_coordinates": coordinates,
        "raw_cell_values": values,
        "raw_value_sha256": sha256_text(canonical_json(values)),
        "year_source": _year_source(int(document["report_year"])),
        "unit_source": table["unit_source"],
        "source_markdown": str(table["source_markdown"]),
        "source_line_range": line_range,
        "source_content_sha256": str(table["content_hash"]),
        "source_block_id": None,
        "content": content,
        "retrieval_text": "\n".join(
            value for value in [*table["section_path"], table["caption"], *table["header_path"], content] if value
        ),
        "provenance": {
            "source_kind": "phase4_table_cells",
            "source_schema_version": "finglmqa.phase4.table_cell.v1",
            "source_record_ids": [f"{table_id}:{row_index}:{coordinate[1]}" for coordinate in coordinates],
            "table_content_sha256": str(table["content_hash"]),
            "text_was_separated_from_table": False,
            "numeric_authorization": "not_authorized_for_answer",
        },
    }
    return seal_fragment(fragment)


def iter_table_row_fragments(
    cells_path: Path,
    tables: Mapping[str, Mapping[str, Any]],
    documents: Mapping[str, Mapping[str, Any]],
    diagnostics: Counter[str] | None = None,
) -> Iterator[dict[str, Any]]:
    group: list[dict[str, Any]] = []
    group_key: tuple[str, str, int] | None = None
    previous_cell_key: tuple[str, int, int, int] | None = None
    previous_skipped_table_id: str | None = None
    for cell in iter_jsonl(cells_path):
        document_id = str(cell.get("document_id") or "")
        table_id = str(cell.get("table_id") or "")
        table_index = cell.get("table_index")
        row_index = cell.get("row_index")
        col_index = cell.get("col_index")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in (table_index, row_index, col_index)):
            raise TableEvidenceError("table cell coordinate fields must be integers")
        cell_key = (document_id, table_index, row_index, col_index)
        if previous_cell_key is not None and cell_key <= previous_cell_key:
            raise TableEvidenceError("table cells are not strictly sorted")
        previous_cell_key = cell_key
        if document_id not in documents:
            raise TableEvidenceError(f"table cell document is outside the Phase 7 allow-list: {document_id}")
        if table_id not in tables:
            # Phase 4 table_cells retains a small number of parsed tables that
            # did not pass the stricter TabGR-ready corpus boundary.  Flush a
            # preceding ready row, then skip every cell from the non-ready
            # table.  Input sorting makes the per-table count deterministic.
            if group_key is not None:
                yield build_table_row_fragment(
                    group,
                    table=tables[group_key[1]],
                    document=documents[group_key[0]],
                )
                group = []
                group_key = None
            if diagnostics is not None:
                diagnostics["table_cells_missing_phase4_corpus"] += 1
                if table_id != previous_skipped_table_id:
                    diagnostics["tables_missing_phase4_corpus"] += 1
            previous_skipped_table_id = table_id
            continue
        if tables[table_id]["document_id"] != document_id or tables[table_id]["table_index"] != table_index:
            raise TableEvidenceError(f"table cell identity conflicts with table corpus: {table_id}")
        key = (document_id, table_id, row_index)
        if group_key is not None and key != group_key:
            yield build_table_row_fragment(group, table=tables[group_key[1]], document=documents[group_key[0]])
            group = []
        group_key = key
        group.append(cell)
    if group_key is not None:
        yield build_table_row_fragment(group, table=tables[group_key[1]], document=documents[group_key[0]])


def build_mixed_narrative_fragment(
    block: Mapping[str, Any],
    *,
    table: Mapping[str, Any],
    document: Mapping[str, Any],
) -> dict[str, Any]:
    text = str(block.get("text") or "").strip()
    block_id = str(block.get("text_block_id") or block.get("block_id") or "")
    source_hash = str(block.get("text_hash") or "")
    if not block_id or source_hash != sha256_text(text):
        raise TableEvidenceError(f"text block hash or identity is invalid: {block_id}")
    source_path = portable_source_path(block.get("source_markdown"), ROOT)
    if source_path != table["source_markdown"]:
        raise TableEvidenceError(f"text block and linked table source differ: {block_id}")
    line_range = list(block.get("line_range") or [])
    fragment = {
        "schema_version": SCHEMA_VERSION,
        "builder_version": BUILDER_VERSION,
        "fragment_id": fragment_identity(
            fragment_kind="mixed_narrative",
            document_id=str(document["document_id"]),
            table_id=str(table["table_id"]),
            row_or_block=block_id,
        ),
        "fragment_kind": "mixed_narrative",
        "document_id": str(document["document_id"]),
        "company_full": str(document["company_full"]),
        "company_name": str(document["company_name"]),
        "stock_code": str(document["stock_code"]),
        "report_year": int(document["report_year"]),
        "table_id": str(table["table_id"]),
        "table_index": int(table["table_index"]),
        "section_path": [str(value) for value in block.get("section_path") or []],
        "caption": str(table["caption"]),
        "header_path": list(table["header_path"]),
        "row_index": None,
        "row_label": "",
        "column_labels": [],
        "cell_coordinates": [],
        "raw_cell_values": [],
        "raw_value_sha256": sha256_text(canonical_json([])),
        "year_source": _year_source(int(document["report_year"])),
        "unit_source": None,
        "source_markdown": source_path,
        "source_line_range": line_range,
        "source_content_sha256": source_hash,
        "source_block_id": block_id,
        "content": text,
        "retrieval_text": "\n".join(
            value for value in [*block.get("section_path", []), table["caption"], text] if value
        ),
        "provenance": {
            "source_kind": "phase3_text_block",
            "source_schema_version": "finglmqa.phase3.text_block.v1",
            "source_record_ids": [block_id],
            "table_content_sha256": str(table["content_hash"]),
            "text_was_separated_from_table": True,
            "numeric_authorization": "not_authorized_for_answer",
        },
    }
    return seal_fragment(fragment)


def iter_mixed_narrative_fragments(
    documents_dir: Path,
    tables: Mapping[str, Mapping[str, Any]],
    documents: Mapping[str, Mapping[str, Any]],
    diagnostics: Counter[str] | None = None,
) -> Iterator[dict[str, Any]]:
    if not documents_dir.is_dir():
        raise TableEvidenceError(f"documents directory does not exist: {documents_dir}")
    paths = sorted(documents_dir.glob("*/text_blocks.jsonl"), key=lambda value: value.parent.name)
    seen_documents: set[str] = set()
    for path in paths:
        per_document: list[dict[str, Any]] = []
        path_document_id = path.parent.name
        if path_document_id not in documents:
            raise TableEvidenceError(f"text blocks are outside Phase 7 allow-list: {path_document_id}")
        seen_documents.add(path_document_id)
        for block in iter_jsonl(path):
            document_id = str(block.get("document_id") or "")
            if document_id != path_document_id:
                raise TableEvidenceError(f"text block document mismatch in {path}")
            if block.get("schema_version") != "finglmqa.phase3.text_block.v1":
                raise TableEvidenceError(f"unexpected text block schema in {path}")
            text = str(block.get("text") or "").strip()
            section_path = [str(value) for value in block.get("section_path") or []]
            linked_table_ids = block.get("linked_table_ids") or []
            if not isinstance(linked_table_ids, list) or any(not isinstance(value, str) for value in linked_table_ids):
                raise TableEvidenceError(f"linked_table_ids is invalid in {path}")
            if not linked_table_ids or not meaningful_narrative(text, section_path):
                continue
            known_table_ids: list[str] = []
            for table_id in set(linked_table_ids):
                table = tables.get(table_id)
                if table is None:
                    if diagnostics is not None:
                        diagnostics["linked_table_missing_phase4_corpus"] += 1
                    continue
                if table["document_id"] != document_id:
                    raise TableEvidenceError(f"text block links a cross-document table: {table_id}")
                known_table_ids.append(table_id)
            for table_id in sorted(known_table_ids, key=lambda value: (tables[value]["table_index"], value)):
                table = tables[table_id]
                per_document.append(
                    build_mixed_narrative_fragment(
                        block,
                        table=table,
                        document=documents[document_id],
                    )
                )
        per_document.sort(key=fragment_sort_key)
        for fragment in per_document:
            yield fragment
    if seen_documents != set(documents):
        missing = sorted(set(documents) - seen_documents)
        raise TableEvidenceError(f"text block corpus does not cover the allow-list: {missing[:5]}")


def merge_fragment_streams(*streams: Iterable[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    yield from heapq.merge(*streams, key=fragment_sort_key)


def build_index(
    *,
    evidence_path: Path,
    cells_path: Path,
    table_corpus_path: Path,
    documents_dir: Path,
    exclusion_audit_path: Path,
    output_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    for path in (evidence_path, cells_path, table_corpus_path, exclusion_audit_path):
        if not path.is_file():
            raise TableEvidenceError(f"required input does not exist: {path}")
    documents = load_document_allow_list(evidence_path)
    tables = load_table_metadata(table_corpus_path, documents)
    exclusion_audit = read_json_object(exclusion_audit_path)
    if exclusion_audit.get("schema_version") != "finglmqa.phase7.table_exclusion_audit.v1":
        raise TableEvidenceError("Phase 7 table exclusion audit schema mismatch")

    counts: Counter[str] = Counter()
    diagnostics: Counter[str] = Counter()

    def counted(rows: Iterable[dict[str, Any]]) -> Iterator[dict[str, Any]]:
        for row in rows:
            counts[str(row["fragment_kind"])] += 1
            counts[f"document:{row['document_id']}"] += 1
            yield row

    rows = counted(
        merge_fragment_streams(
            iter_table_row_fragments(cells_path, tables, documents, diagnostics),
            iter_mixed_narrative_fragments(documents_dir, tables, documents, diagnostics),
        )
    )
    fragment_count, output_sha256 = write_canonical_jsonl(output_path, rows)
    document_counts = {key.split(":", 1)[1]: value for key, value in counts.items() if key.startswith("document:")}
    if set(document_counts) != set(documents):
        raise TableEvidenceError("built index does not cover every allow-listed document")
    report = {
        "schema_version": REPORT_SCHEMA,
        "builder_version": BUILDER_VERSION,
        "fragment_schema_version": SCHEMA_VERSION,
        "inputs": {
            "evidence_chunks": evidence_path.relative_to(ROOT).as_posix(),
            "table_cells": cells_path.relative_to(ROOT).as_posix(),
            "table_corpus": table_corpus_path.relative_to(ROOT).as_posix(),
            "documents_dir": documents_dir.relative_to(ROOT).as_posix(),
            "phase7_exclusion_audit": exclusion_audit_path.relative_to(ROOT).as_posix(),
        },
        "input_sha256": {
            "evidence_chunks": sha256_file(evidence_path),
            "table_cells": sha256_file(cells_path),
            "table_corpus": sha256_file(table_corpus_path),
            "phase7_exclusion_audit": sha256_file(exclusion_audit_path),
        },
        "output": output_path.relative_to(ROOT).as_posix(),
        "output_sha256": output_sha256,
        "counts": {
            "documents": len(documents),
            "tables": len(tables),
            "fragments": fragment_count,
            "table_row": counts["table_row"],
            "mixed_narrative": counts["mixed_narrative"],
            "phase7_excluded_table_chunks": int(exclusion_audit.get("classified_chunks", 0)),
            "phase7_mixed_narrative_chunks": int(
                (exclusion_audit.get("classification_counts") or {}).get("mixed_narrative", 0)
            ),
            "linked_tables_missing_phase4_corpus": diagnostics["linked_table_missing_phase4_corpus"],
            "tables_missing_phase4_corpus": diagnostics["tables_missing_phase4_corpus"],
            "table_cells_missing_phase4_corpus": diagnostics["table_cells_missing_phase4_corpus"],
        },
        "validations": {
            "document_scoped": True,
            "canonical_jsonl": True,
            "deterministically_sorted": True,
            "raw_html_not_indexed": True,
            "mixed_narrative_separated_from_table_rows": True,
            "all_numeric_values_unauthorized_for_answer": True,
        },
        "limitations": [
            "Mixed narrative recovery uses exact Phase 3 text blocks linked within three source lines, not truncated Phase 7 audit previews.",
            "Retrieval is deterministic lexical ranking only; no dense table embeddings are built in this experiment.",
            "A table fragment cannot authorize a number or become an answer without a future table citation/value gate.",
            "Merged and multi-row headers retain the Phase 4 flattened labels and are not semantically repaired.",
            "Narrative links to Phase 3 tables absent from the Phase 4 ready corpus are counted and skipped.",
            "Phase 4 cells belonging to tables absent from the TabGR-ready corpus are counted and skipped as whole tables.",
        ],
    }
    atomic_write_json(report_path, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-chunks", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--table-cells", type=Path, default=DEFAULT_TABLE_CELLS)
    parser.add_argument("--table-corpus", type=Path, default=DEFAULT_TABLE_CORPUS)
    parser.add_argument("--documents-dir", type=Path, default=DEFAULT_DOCUMENTS_DIR)
    parser.add_argument("--exclusion-audit", type=Path, default=DEFAULT_EXCLUSION_AUDIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_index(
        evidence_path=args.evidence_chunks.resolve(),
        cells_path=args.table_cells.resolve(),
        table_corpus_path=args.table_corpus.resolve(),
        documents_dir=args.documents_dir.resolve(),
        exclusion_audit_path=args.exclusion_audit.resolve(),
        output_path=args.output.resolve(),
        report_path=args.report.resolve(),
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
