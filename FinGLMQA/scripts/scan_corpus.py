#!/usr/bin/env python3
"""Build the Phase 2 corpus manifest and company-year index."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


SCHEMA_MANIFEST = "finglmqa.phase2.corpus_manifest.v1"
SCHEMA_INDEX = "finglmqa.phase2.company_year_index.v1"
SCHEMA_REPORT = "finglmqa.phase2.corpus_scan_report.v1"
SCANNER_VERSION = "phase2-scanner-v1"


def workspace_root() -> Path:
    return Path(__file__).resolve().parents[1]


def rel(path: Path, root: Path) -> str:
    try:
        return path.resolve(strict=False).relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_aliases(values: list[Any]) -> list[str]:
    seen: set[str] = set()
    aliases: list[str] = []
    for value in values:
        if value is None:
            continue
        alias = str(value).strip()
        if not alias or alias in seen:
            continue
        seen.add(alias)
        aliases.append(alias)
    return aliases


def infer_stock_symbol(stock_code: str | None, aliases: list[str], file_name: str) -> str | None:
    for value in [file_name, *aliases]:
        match = re.search(r"A\d{6}", value)
        if match:
            return match.group(0)
    if stock_code and re.fullmatch(r"\d{6}", stock_code):
        return f"A{stock_code}"
    return None


def parse_file_name(file_name: str) -> dict[str, str | None]:
    stem = Path(file_name).stem
    match = re.match(r"^(A\d{6})_(\d{4})(?:_(.*))?$", stem)
    if not match:
        return {
            "stock_symbol": None,
            "stock_code": None,
            "report_year": None,
            "tail": stem,
            "stock_name": None,
            "report_title": stem,
        }

    stock_symbol = match.group(1)
    report_year = match.group(2)
    tail = match.group(3) or ""
    stock_name = None
    report_title = tail or stem

    tail_match = re.match(r"^A\d{6}_([^_]+)_(.+)$", tail)
    if tail_match:
        stock_name = tail_match.group(1)
        report_title = tail_match.group(2)

    return {
        "stock_symbol": stock_symbol,
        "stock_code": stock_symbol[1:],
        "report_year": report_year,
        "tail": tail or None,
        "stock_name": stock_name,
        "report_title": report_title,
    }


def markdown_header(path: Path, max_lines: int = 80) -> dict[str, str | None]:
    first_non_empty = None
    first_h1 = None
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for index, line in enumerate(fh):
                if index >= max_lines:
                    break
                stripped = line.strip()
                if stripped and first_non_empty is None:
                    first_non_empty = stripped
                if stripped.startswith("# ") and first_h1 is None:
                    first_h1 = stripped[2:].strip()
                if first_non_empty and first_h1:
                    break
    except OSError:
        pass
    return {
        "first_non_empty_line": first_non_empty,
        "first_h1": first_h1,
    }


def markdown_stats(path: Path) -> dict[str, int | None]:
    stats = {
        "line_count": None,
        "table_tag_count": None,
        "heading_count": None,
        "image_count": None,
    }
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return stats
    lines = text.splitlines()
    return {
        "line_count": len(lines),
        "table_tag_count": len(re.findall(r"<table\b", text, flags=re.IGNORECASE)),
        "heading_count": sum(1 for line in lines if re.match(r"^\s{0,3}#{1,6}\s+", line)),
        "image_count": len(re.findall(r"!\[[^\]]*\]\([^)]+\)", text)),
    }


def warning_details(warnings: list[str]) -> list[dict[str, str]]:
    details = []
    for warning in warnings:
        severity = "error" if warning.startswith("missing_") or warning.endswith("_missing") else "warning"
        details.append({"code": warning, "severity": severity})
    return details


def manifest_lookup(source_manifest: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    by_name: dict[str, dict[str, Any]] = {}
    by_resolved: dict[str, dict[str, Any]] = {}
    for record in source_manifest.get("selected_reports", []):
        if not isinstance(record, dict):
            continue
        for key in ("a2rag_doc", "source_markdown"):
            value = record.get(key)
            if not value:
                continue
            path = Path(str(value))
            by_name.setdefault(path.name, record)
            by_resolved.setdefault(path.resolve(strict=False).as_posix(), record)
    return by_name, by_resolved


def required_missing(record: dict[str, Any], fields: list[str]) -> list[str]:
    return [field for field in fields if not str(record.get(field) or "").strip()]


def build_documents(root: Path, source_markdown: Path, source_manifest: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_name, by_resolved = manifest_lookup(source_manifest)
    md_paths = sorted(path for path in source_markdown.iterdir() if path.suffix.lower() == ".md")

    documents: list[dict[str, Any]] = []
    markdown_matched_manifest_keys: set[str] = set()
    markdown_without_manifest: list[str] = []

    for md_path in md_paths:
        resolved = md_path.resolve(strict=False)
        record = by_resolved.get(resolved.as_posix()) or by_name.get(md_path.name) or {}
        if record:
            for key in ("a2rag_doc", "source_markdown"):
                if record.get(key):
                    markdown_matched_manifest_keys.add(Path(str(record[key])).resolve(strict=False).as_posix())
                    markdown_matched_manifest_keys.add(Path(str(record[key])).name)
        else:
            markdown_without_manifest.append(md_path.name)

        filename_meta = parse_file_name(md_path.name)
        aliases = normalize_aliases(record.get("aliases", []) if isinstance(record.get("aliases"), list) else [])
        stock_code = str(record.get("stock_code") or filename_meta["stock_code"] or "").strip()
        stock_symbol = infer_stock_symbol(stock_code, aliases, md_path.name)
        stock_name = str(record.get("stock_name") or filename_meta["stock_name"] or "").strip()
        report_year = str(record.get("report_year") or filename_meta["report_year"] or "").strip()
        company_full = str(record.get("company_full") or "").strip()
        document_id = str(record.get("doc_id") or md_path.stem).strip()
        report_title = str(record.get("doc_id") or filename_meta["report_title"] or md_path.stem).strip()
        header = markdown_header(resolved)
        file_stats = markdown_stats(resolved)

        aliases = normalize_aliases([
            *(aliases or []),
            stock_symbol,
            stock_code,
            stock_name,
            company_full,
        ])

        warnings: list[str] = []
        if not record:
            warnings.append("missing_manifest_record")
        for field in required_missing(
            {
                "document_id": document_id,
                "stock_code": stock_code,
                "stock_name": stock_name,
                "report_year": report_year,
                "markdown_path": md_path.as_posix(),
            },
            ["document_id", "stock_code", "stock_name", "report_year", "markdown_path"],
        ):
            warnings.append(f"missing_{field}")
        if filename_meta["stock_code"] and stock_code and filename_meta["stock_code"] != stock_code:
            warnings.append("filename_stock_code_mismatch")
        if filename_meta["report_year"] and report_year and filename_meta["report_year"] != report_year:
            warnings.append("filename_report_year_mismatch")
        if filename_meta["stock_name"] and stock_name and filename_meta["stock_name"] != stock_name:
            warnings.append("filename_stock_name_mismatch")
        if not resolved.exists():
            warnings.append("markdown_target_missing")
        if company_full and header["first_h1"] and company_full not in header["first_h1"] and header["first_h1"] not in company_full:
            warnings.append("markdown_h1_company_mismatch")

        document = {
            "document_id": document_id,
            "source_manifest_doc_id": record.get("doc_id"),
            "metadata_source": "source_manifest" if record else "filename_markdown_fallback",
            "stock_code": stock_code or None,
            "stock_symbol": stock_symbol,
            "stock_name": stock_name or None,
            "company_full": company_full or None,
            "report_year": report_year or None,
            "report_title": report_title,
            "aliases": aliases,
            "markdown_path": rel(md_path, root),
            "resolved_source_path": resolved.as_posix(),
            "source_markdown": record.get("source_markdown"),
            "source_a2rag_doc": record.get("a2rag_doc"),
            "tables_jsonl": record.get("tables_jsonl"),
            "content_sha256": sha256_file(resolved) if resolved.exists() else None,
            "file_size_bytes": resolved.stat().st_size if resolved.exists() else None,
            **file_stats,
            "filename_metadata": filename_meta,
            "markdown_header": header,
            "warnings": warnings,
            "warning_details": warning_details(warnings),
            "status": "valid" if not [w for w in warnings if w.startswith("missing_") or w.endswith("_missing")] else "invalid",
        }
        documents.append(document)

    manifest_without_markdown: list[dict[str, str | None]] = []
    for record in source_manifest.get("selected_reports", []):
        if not isinstance(record, dict):
            continue
        a2rag_doc = str(record.get("a2rag_doc") or "")
        source_doc = str(record.get("source_markdown") or "")
        keys = {
            Path(a2rag_doc).resolve(strict=False).as_posix() if a2rag_doc else "",
            Path(source_doc).resolve(strict=False).as_posix() if source_doc else "",
            Path(a2rag_doc).name if a2rag_doc else "",
            Path(source_doc).name if source_doc else "",
        }
        if not any(key in markdown_matched_manifest_keys for key in keys if key):
            manifest_without_markdown.append({
                "doc_id": record.get("doc_id"),
                "a2rag_doc": record.get("a2rag_doc"),
                "source_markdown": record.get("source_markdown"),
            })

    scan_gaps = {
        "markdown_without_manifest": markdown_without_manifest,
        "manifest_without_markdown": manifest_without_markdown,
    }
    return documents, scan_gaps


def duplicate_values(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    values = Counter(str(row.get(key) or "") for row in rows)
    duplicates = []
    for value, count in sorted(values.items()):
        if value and count > 1:
            duplicates.append({"value": value, "count": count})
    return duplicates


def build_index(documents: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[tuple[str | None, str | None], list[dict[str, Any]]] = defaultdict(list)
    for doc in documents:
        groups[(doc.get("stock_code"), doc.get("report_year"))].append(doc)

    alias_map: dict[str, set[str]] = defaultdict(set)
    for doc in documents:
        stock_code = doc.get("stock_code")
        for alias in doc.get("aliases") or []:
            if stock_code:
                alias_map[str(alias)].add(str(stock_code))

    alias_collisions = [
        {"alias": alias, "stock_codes": sorted(codes)}
        for alias, codes in sorted(alias_map.items())
        if len(codes) > 1
    ]

    rows: list[dict[str, Any]] = []
    ambiguous_groups: list[dict[str, Any]] = []
    for (stock_code, report_year), docs in sorted(groups.items(), key=lambda item: (str(item[0][0]), str(item[0][1]))):
        candidate_count = len(docs)
        if candidate_count > 1:
            ambiguous_groups.append({
                "stock_code": stock_code,
                "report_year": report_year,
                "document_ids": [doc["document_id"] for doc in docs],
            })
        for doc in docs:
            if doc.get("status") == "invalid" or not stock_code or not report_year:
                status = "invalid"
            elif candidate_count > 1:
                status = "ambiguous"
            else:
                status = "unique"
            rows.append({
                "schema_version": SCHEMA_INDEX,
                "stock_code": stock_code,
                "stock_symbol": doc.get("stock_symbol"),
                "stock_name": doc.get("stock_name"),
                "company_full": doc.get("company_full"),
                "report_year": report_year,
                "document_id": doc.get("document_id"),
                "aliases": doc.get("aliases"),
                "markdown_path": doc.get("markdown_path"),
                "resolved_source_path": doc.get("resolved_source_path"),
                "status": status,
                "candidate_count": candidate_count,
                "group_key": f"{stock_code}:{report_year}",
            })

    return rows, ambiguous_groups, alias_collisions


def report_markdown(report: dict[str, Any]) -> str:
    counts = report["counts"]
    lines = [
        "# Phase 2 Report",
        "",
        "## Date",
        report["generated_at_utc"],
        "",
        "## Environment",
        f"- Workspace: `{report['workspace_root']}`",
        f"- Source Markdown: `{report['inputs']['source_markdown_dir']}`",
        f"- Source Manifest: `{report['inputs']['source_manifest_path']}`",
        "",
        "## Generated Artifacts",
    ]
    for label, path in report["outputs"].items():
        lines.append(f"- {label}: `{path}`")
    lines.extend([
        "",
        "## Verification Results",
        f"- Markdown files found: {counts['markdown_files']}",
        f"- Source manifest selected reports: {counts['source_manifest_selected_reports']}",
        f"- Documents emitted: {counts['documents_emitted']}",
        f"- Unique resolver rows: {counts['unique_index_rows']}",
        f"- Ambiguous resolver rows: {counts['ambiguous_index_rows']}",
        f"- Invalid resolver rows: {counts['invalid_index_rows']}",
        f"- Alias collisions: {counts['alias_collisions']}",
        f"- Manifest entries without Markdown: {counts['manifest_without_markdown']}",
        f"- Markdown files without manifest record: {counts['markdown_without_manifest']}",
        "",
        "## Issues Encountered",
    ])
    if report["samples"]["warning_counts"]:
        for item in report["samples"]["warning_counts"]:
            lines.append(f"- {item['warning']}: {item['count']}")
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
    parser.add_argument("--source-markdown", type=Path, default=root / "refs/source_markdown")
    parser.add_argument("--source-manifest", type=Path, default=root / "refs/corpus_manifest.json")
    parser.add_argument("--output-dir", type=Path, default=root / "data/corpus_package")
    parser.add_argument("--run-dir", type=Path, default=root / "runs/phase_02")
    args = parser.parse_args()

    source_manifest = load_json(args.source_manifest)
    generated_at = datetime.now(UTC).isoformat(timespec="seconds")
    documents, gaps = build_documents(root, args.source_markdown, source_manifest)
    index_rows, ambiguous_groups, alias_collisions = build_index(documents)

    document_id_duplicates = duplicate_values(documents, "document_id")
    warning_counter = Counter(warning for doc in documents for warning in doc.get("warnings", []))

    manifest_out = args.output_dir / "corpus_manifest.json"
    index_out = args.output_dir / "company_year_index.jsonl"
    report_out = args.run_dir / "corpus_scan_report.json"
    phase_report_out = args.run_dir / "phase_02_report.md"

    manifest = {
        "schema_version": SCHEMA_MANIFEST,
        "scanner_version": SCANNER_VERSION,
        "generated_at_utc": generated_at,
        "command": " ".join(sys.argv),
        "workspace_root": root.as_posix(),
        "inputs": {
            "source_markdown_dir": rel(args.source_markdown, root),
            "source_manifest_path": rel(args.source_manifest, root),
            "source_profile_output_root": source_manifest.get("output_root"),
        },
        "summary": {
            "document_count": len(documents),
            "valid_document_count": sum(1 for doc in documents if doc.get("status") == "valid"),
            "invalid_document_count": sum(1 for doc in documents if doc.get("status") == "invalid"),
            "unique_document_id_count": len({doc.get("document_id") for doc in documents}),
        },
        "documents": documents,
    }

    counts = {
        "markdown_files": len(documents),
        "source_manifest_selected_reports": len(source_manifest.get("selected_reports", [])),
        "documents_emitted": len(documents),
        "valid_documents": manifest["summary"]["valid_document_count"],
        "invalid_documents": manifest["summary"]["invalid_document_count"],
        "unique_index_rows": sum(1 for row in index_rows if row["status"] == "unique"),
        "ambiguous_index_rows": sum(1 for row in index_rows if row["status"] == "ambiguous"),
        "invalid_index_rows": sum(1 for row in index_rows if row["status"] == "invalid"),
        "duplicate_document_ids": len(document_id_duplicates),
        "duplicate_company_year_groups": len(ambiguous_groups),
        "alias_collisions": len(alias_collisions),
        "manifest_without_markdown": len(gaps["manifest_without_markdown"]),
        "markdown_without_manifest": len(gaps["markdown_without_manifest"]),
    }

    report = {
        "schema_version": SCHEMA_REPORT,
        "scanner_version": SCANNER_VERSION,
        "generated_at_utc": generated_at,
        "command": " ".join(sys.argv),
        "workspace_root": root.as_posix(),
        "inputs": manifest["inputs"],
        "outputs": {
            "corpus_manifest": rel(manifest_out, root),
            "company_year_index": rel(index_out, root),
            "corpus_scan_report": rel(report_out, root),
            "phase_report": rel(phase_report_out, root),
        },
        "counts": counts,
        "duplicates": {
            "document_ids": document_id_duplicates,
            "company_year_groups": ambiguous_groups,
        },
        "alias_collisions": alias_collisions,
        "gaps": gaps,
        "samples": {
            "warning_counts": [
                {"warning": warning, "count": count}
                for warning, count in warning_counter.most_common()
            ],
            "warning_documents": [
                {
                    "document_id": doc["document_id"],
                    "markdown_path": doc["markdown_path"],
                    "warnings": doc["warnings"],
                }
                for doc in documents
                if doc.get("warnings")
            ][:50],
        },
    }

    write_json(manifest_out, manifest)
    write_jsonl(index_out, index_rows)
    write_json(report_out, report)
    phase_report_out.parent.mkdir(parents=True, exist_ok=True)
    phase_report_out.write_text(report_markdown(report), encoding="utf-8")

    print(json.dumps({"counts": counts, "outputs": report["outputs"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
