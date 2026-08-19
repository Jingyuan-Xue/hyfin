#!/usr/bin/env python3
"""Build the Phase 7 provenance layer over the reused A2RAG dense index.

Run this script with the existing A2RAG runtime Python because it already owns
the pandas/pyarrow dependencies used to inspect the dense-vector parquet:

    refs/a2rag_runtime/.venv/bin/python scripts/build_a2rag_text_index.py

The script never writes to the reused runtime or source index. It selects
non-table A2RAG chunks, aligns them to Markdown lines, emits the FinGLMQA
evidence corpus, and creates a read-only symlink to the existing vectors.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import os
import re
import subprocess
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import pyarrow.parquet as pq


BUILDER_VERSION = "phase7-a2rag-text-index-v2"
SCHEMA_EVIDENCE_CHUNK = "finglmqa.phase7.evidence_chunk.v1"
SCHEMA_INDEX_MANIFEST = "finglmqa.phase7.a2rag_index_manifest.v1"
SCHEMA_DOCUMENT_CHUNK_MAP = "finglmqa.phase7.document_chunk_map.v1"
SCHEMA_BUILD_REPORT = "finglmqa.phase7.build_report.v1"
A2RAG_LABEL = "qwen3.6-27b-local_BAAI_bge-m3"
TABLE_HTML_RE = re.compile(r"<\s*/?\s*(?:table|tr|td|th)\b", re.IGNORECASE)
TABLE_BLOCK_RE = re.compile(r"<\s*table\b.*?<\s*/\s*table\s*>", re.IGNORECASE | re.DOTALL)
HTML_TAG_RE = re.compile(r"<[^>]+>", re.DOTALL)
MARKDOWN_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s*(.*?)\s*#*\s*$")

TABLE_SUBJECT_RULES = (
    ("staff_or_personnel", ("员工", "职工", "人员", "薪酬", "教育程度", "专业构成", "培训")),
    ("dividend", ("分红", "利润分配", "股利", "派息", "红股", "转增")),
    ("shareholder", ("股东", "持股", "股份变动", "前十名", "股本结构")),
    (
        "financial_table",
        ("财务", "资产", "负债", "利润", "现金流", "收入", "成本", "费用", "应收", "应付", "会计", "审计"),
    ),
    (
        "governance_or_other_non_financial",
        ("治理", "董事", "监事", "高管", "会议", "委员会", "处罚", "诉讼", "承诺", "关联交易"),
    ),
)

TAG_RULES = {
    "business": ("业务", "主营", "产品", "行业", "经营模式", "客户", "供应商"),
    "risk": ("风险", "不确定性", "可能面对", "应对措施"),
    "rd": ("研发", "技术", "专利", "创新", "研究开发"),
    "staff": ("员工", "人员", "薪酬", "职工", "教育程度", "培训"),
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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no} must contain a JSON object")
            rows.append(row)
    return rows


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")
    tmp.replace(path)


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            json.dump(row, fh, ensure_ascii=False, sort_keys=True)
            fh.write("\n")
    tmp.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def relative_path(path: Path, root: Path) -> str:
    try:
        return path.absolute().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def parse_metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("A2RAG chunk metadata must be a JSON object")


def normalize_without_whitespace(value: str) -> str:
    return "".join(char for char in value if not char.isspace())


def normalize_heading(value: str) -> str:
    """Normalize only heading presentation, never body punctuation/content."""
    return normalize_without_whitespace(
        unicodedata.normalize("NFKC", value.strip().strip("#").strip())
    )


def normalized_source_with_offsets(value: str) -> tuple[str, list[int]]:
    chars: list[str] = []
    offsets: list[int] = []
    for offset, char in enumerate(value):
        if not char.isspace():
            chars.append(char)
            offsets.append(offset)
    return "".join(chars), offsets


def line_start_offsets(value: str) -> list[int]:
    starts = [0]
    starts.extend(match.end() for match in re.finditer(r"\n", value))
    return starts


def line_for_offset(starts: list[int], offset: int) -> int:
    return bisect.bisect_right(starts, offset)


@dataclass(frozen=True)
class MarkdownSection:
    heading: str
    normalized_heading: str
    level: int
    heading_line: int
    body_normalized_start: int
    section_normalized_end: int
    section_line_end: int


def parse_markdown_sections(
    source_text: str,
    source_offsets: list[int],
) -> tuple[list[MarkdownSection], dict[str, list[MarkdownSection]]]:
    """Parse Markdown headings and section bounds independently of chunk order."""
    raw_headings: list[dict[str, Any]] = []
    char_offset = 0
    source_lines = source_text.splitlines(keepends=True)
    for line_no, line in enumerate(source_lines, start=1):
        match = MARKDOWN_HEADING_RE.match(line.rstrip("\r\n"))
        if match:
            heading = match.group(2).strip().rstrip("#").strip()
            raw_headings.append({
                "heading": heading,
                "normalized_heading": normalize_heading(heading),
                "level": len(match.group(1)),
                "heading_line": line_no,
                "heading_normalized_start": bisect.bisect_left(source_offsets, char_offset),
                "body_normalized_start": bisect.bisect_left(source_offsets, char_offset + len(line)),
            })
        char_offset += len(line)

    sections: list[MarkdownSection] = []
    source_line_count = len(source_text.splitlines())
    for index, heading in enumerate(raw_headings):
        section_normalized_end = len(source_offsets)
        section_line_end = source_line_count
        for later in raw_headings[index + 1 :]:
            if int(later["level"]) <= int(heading["level"]):
                section_normalized_end = int(later["heading_normalized_start"])
                section_line_end = int(later["heading_line"]) - 1
                break
        sections.append(MarkdownSection(
            heading=str(heading["heading"]),
            normalized_heading=str(heading["normalized_heading"]),
            level=int(heading["level"]),
            heading_line=int(heading["heading_line"]),
            body_normalized_start=int(heading["body_normalized_start"]),
            section_normalized_end=section_normalized_end,
            section_line_end=section_line_end,
        ))

    by_heading: dict[str, list[MarkdownSection]] = defaultdict(list)
    for section in sections:
        by_heading[section.normalized_heading].append(section)
    return sections, dict(by_heading)


def find_all_exact(haystack: str, needle: str, start: int = 0, end: int | None = None) -> list[int]:
    if not needle:
        return []
    boundary = len(haystack) if end is None else end
    positions: list[int] = []
    cursor = start
    while cursor <= boundary - len(needle):
        position = haystack.find(needle, cursor, boundary)
        if position < 0:
            break
        positions.append(position)
        cursor = position + 1
    return positions


def align_chunk_to_section(
    normalized_source: str,
    normalized_chunk: str,
    heading_path: list[str],
    sections_by_heading: dict[str, list[MarkdownSection]],
    cursor: int,
) -> dict[str, Any]:
    """Return a unique content+section alignment, otherwise fail closed.

    The cursor is only a monotonicity guard. It never chooses among duplicate
    body matches, which prevents short boilerplate from being assigned to an
    arbitrary occurrence.
    """
    normalized_path = [normalize_heading(value) for value in heading_path if normalize_heading(value)]
    if not normalized_path:
        return {"status": "alignment_section_missing", "candidate_count": 0}
    target_heading = normalized_path[-1]
    compatible_sections = sections_by_heading.get(target_heading, [])
    compatible: list[tuple[int, MarkdownSection]] = []
    for section in compatible_sections:
        for position in find_all_exact(
            normalized_source,
            normalized_chunk,
            section.body_normalized_start,
            section.section_normalized_end,
        ):
            if position + len(normalized_chunk) <= section.section_normalized_end:
                compatible.append((position, section))
    compatible = sorted(set(compatible), key=lambda item: (item[0], item[1].heading_line))

    # A second, more conservative ambiguity lens catches presentation-only
    # variants (for example full-width punctuation) that NFKC makes identical.
    # It never supplies a position; it can only veto an otherwise exact match.
    nfkc_chunk = unicodedata.normalize("NFKC", normalized_chunk)
    nfkc_compatible_count = sum(
        unicodedata.normalize(
            "NFKC",
            normalized_source[section.body_normalized_start : section.section_normalized_end],
        ).count(nfkc_chunk)
        for section in compatible_sections
    )
    if nfkc_compatible_count > 1:
        return {
            "status": "alignment_ambiguous",
            "candidate_count": nfkc_compatible_count,
            "exact_candidate_count": len(compatible),
            "ambiguity_lens": "nfkc_section_content",
            "target_heading": heading_path[-1],
        }

    if len(compatible) > 1:
        return {
            "status": "alignment_ambiguous",
            "candidate_count": len(compatible),
            "candidate_positions": [position for position, _ in compatible[:20]],
            "target_heading": heading_path[-1],
        }
    if len(compatible) == 1:
        position, section = compatible[0]
        if position < cursor:
            return {
                "status": "alignment_non_monotonic",
                "candidate_count": 1,
                "position": position,
                "cursor_before": cursor,
                "target_heading": heading_path[-1],
            }
        return {
            "status": "ok",
            "alignment_method": "section_constrained_unique_exact",
            "position": position,
            "cursor_after": max(cursor, position + 1),
            "section": section,
            "candidate_count": 1,
            "heading_match_count": len(compatible_sections),
        }

    # A global lookup is diagnostic and may only succeed when its sole match is
    # demonstrably inside a matching Markdown section. In practice this is the
    # same safety condition as above; it exists to make fallback use auditable.
    global_positions = find_all_exact(normalized_source, normalized_chunk)
    global_compatible: list[tuple[int, MarkdownSection]] = []
    for position in global_positions:
        for section in compatible_sections:
            if (
                section.body_normalized_start <= position
                and position + len(normalized_chunk) <= section.section_normalized_end
            ):
                global_compatible.append((position, section))
    global_compatible = sorted(set(global_compatible), key=lambda item: (item[0], item[1].heading_line))
    if len(global_positions) == 1 and len(global_compatible) == 1:
        position, section = global_compatible[0]
        if position >= cursor:
            return {
                "status": "ok",
                "alignment_method": "global_unique_section_compatible_exact",
                "position": position,
                "cursor_after": max(cursor, position + 1),
                "section": section,
                "candidate_count": 1,
                "heading_match_count": len(compatible_sections),
            }
        return {
            "status": "alignment_non_monotonic",
            "candidate_count": 1,
            "position": position,
            "cursor_before": cursor,
            "target_heading": heading_path[-1],
        }
    return {
        "status": "source_alignment_failed",
        "candidate_count": len(global_compatible),
        "global_content_match_count": len(global_positions),
        "heading_match_count": len(compatible_sections),
        "target_heading": heading_path[-1],
    }


def table_audit_classification(content: str, heading_path: list[str]) -> dict[str, Any]:
    without_tables = TABLE_BLOCK_RE.sub(" ", content)
    outside_text = HTML_TAG_RE.sub(" ", without_tables)
    narrative = re.sub(r"\s+", " ", outside_text).strip()
    material = re.sub(r"[\s\W_]+", "", narrative)
    boilerplate = re.sub(r"单位|币种|人民币|适用|不适用|是|否", "", material)
    mixed_narrative = len(boilerplate) >= 20
    haystack = "\n".join([*heading_path, content])
    subject = "governance_or_other_non_financial"
    for category, needles in TABLE_SUBJECT_RULES:
        if any(needle in haystack for needle in needles):
            subject = category
            break
    return {
        "classification": "mixed_narrative" if mixed_narrative else subject,
        "subject_category": subject,
        "mixed_narrative": mixed_narrative,
        "outside_table_text": narrative,
    }


def semantic_tags(section_path: list[str], content: str) -> list[str]:
    haystack = "\n".join([*section_path, content])
    return [tag for tag, needles in TAG_RULES.items() if any(needle in haystack for needle in needles)]


def ensure_symlink(link_path: Path, target_path: Path) -> None:
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.is_symlink():
        if link_path.resolve(strict=False) == target_path.resolve(strict=False):
            return
        link_path.unlink()
    elif link_path.exists():
        raise RuntimeError(f"Refusing to replace non-symlink index artifact: {link_path}")
    os.symlink(target_path.as_posix(), link_path.as_posix())


def runtime_git_commit(runtime_dir: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", runtime_dir.as_posix(), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() or None


def embedding_dimension(parquet_path: Path) -> int:
    parquet = pq.ParquetFile(parquet_path)
    first_batch = next(parquet.iter_batches(batch_size=1, columns=["embedding"]))
    first = first_batch.column(0)[0].as_py()
    return len(first)


def evidence_schema() -> dict[str, Any]:
    required = [
        "schema_version",
        "evidence_chunk_id",
        "a2rag_chunk_id",
        "document_id",
        "company_name",
        "stock_code",
        "report_year",
        "section_path",
        "semantic_tags",
        "line_range",
        "source_markdown",
        "content",
    ]
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": SCHEMA_EVIDENCE_CHUNK,
        "title": "FinGLMQA Phase 7 evidence chunk",
        "type": "object",
        "required": required,
        "properties": {
            "schema_version": {"const": SCHEMA_EVIDENCE_CHUNK},
            "evidence_chunk_id": {"type": "string", "minLength": 1},
            "a2rag_chunk_id": {"type": "string", "minLength": 1},
            "document_id": {"type": "string", "minLength": 1},
            "company_name": {"type": "string", "minLength": 1},
            "company_full": {"type": ["string", "null"]},
            "stock_code": {"type": "string", "pattern": "^[0-9]{6}$"},
            "stock_symbol": {"type": ["string", "null"]},
            "report_year": {"type": "integer"},
            "section_path": {"type": "array", "items": {"type": "string"}},
            "semantic_tags": {"type": "array", "items": {"type": "string"}, "uniqueItems": True},
            "line_range": {
                "type": "array",
                "prefixItems": [{"type": "integer", "minimum": 1}, {"type": "integer", "minimum": 1}],
                "minItems": 2,
                "maxItems": 2,
            },
            "source_markdown": {"type": "string", "minLength": 1},
            "source_content_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "content": {"type": "string", "minLength": 1},
            "content_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "a2rag_metadata": {"type": "object"},
            "provenance": {"type": "object"},
        },
        "additionalProperties": True,
    }


def main() -> int:
    root = workspace_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-manifest", type=Path, default=root / "data/corpus_package/corpus_manifest.json")
    parser.add_argument("--company-year-index", type=Path, default=root / "data/corpus_package/company_year_index.jsonl")
    parser.add_argument(
        "--source-a2rag-dir",
        type=Path,
        default=root / f"refs/source_profile/a2rag_bge_m3_outputs/{A2RAG_LABEL}",
    )
    parser.add_argument("--runtime-dir", type=Path, default=root / "refs/a2rag_runtime")
    parser.add_argument("--evidence-output", type=Path, default=root / "data/corpus_package/evidence_chunks.jsonl")
    parser.add_argument("--schema-output", type=Path, default=root / "data/schemas/evidence_chunks.schema.json")
    parser.add_argument("--index-dir", type=Path, default=root / "data/indexes/a2rag_index")
    parser.add_argument("--run-dir", type=Path, default=root / "runs/phase_07")
    args = parser.parse_args()

    dense_parquet = args.source_a2rag_dir / "chunk_embeddings/vdb_chunk.parquet"
    source_documents_path = args.source_a2rag_dir / "documents.json"
    for required in [args.corpus_manifest, args.company_year_index, dense_parquet, source_documents_path]:
        if not required.exists():
            raise FileNotFoundError(required)

    corpus_manifest = load_json(args.corpus_manifest)
    corpus_docs = list(corpus_manifest.get("documents") or [])
    resolver_rows = load_jsonl(args.company_year_index)
    expected_document_count = len(corpus_docs)
    if not expected_document_count or len(resolver_rows) != expected_document_count:
        raise RuntimeError(
            "Phase 7 corpus/resolver count mismatch: "
            f"manifest={expected_document_count}, resolver={len(resolver_rows)}"
        )
    if any(row.get("status") != "unique" or row.get("candidate_count") != 1 for row in resolver_rows):
        raise RuntimeError("Company-year resolver contains non-unique rows; stop before building Phase 7")

    resolver_by_source_id: dict[str, dict[str, Any]] = {}
    resolver_by_document_id: dict[str, dict[str, Any]] = {}
    for row in resolver_rows:
        source_id = Path(str(row["markdown_path"])).name
        if source_id in resolver_by_source_id:
            raise RuntimeError(f"Duplicate resolver source filename: {source_id}")
        resolver_by_source_id[source_id] = row
        resolver_by_document_id[str(row["document_id"])] = row

    corpus_by_source_id: dict[str, dict[str, Any]] = {}
    for doc in corpus_docs:
        source_id = Path(str(doc["source_a2rag_doc"])).name
        corpus_by_source_id[source_id] = doc

    source_documents = load_json(source_documents_path).get("documents") or []
    source_manifest_by_id = {str(row["source_id"]): row for row in source_documents}
    source_hash_matches = 0
    source_hash_mismatches: list[dict[str, Any]] = []
    for source_id, doc in corpus_by_source_id.items():
        indexed = source_manifest_by_id.get(source_id)
        matches = indexed is not None and indexed.get("content_sha256") == doc.get("content_sha256")
        if matches:
            source_hash_matches += 1
        else:
            source_hash_mismatches.append({
                "source_id": source_id,
                "corpus_sha256": doc.get("content_sha256"),
                "index_sha256": indexed.get("content_sha256") if indexed else None,
            })
    if source_hash_matches != len(corpus_docs):
        raise RuntimeError(
            f"Reused A2RAG index source hashes do not match the corpus: "
            f"{source_hash_matches}/{len(corpus_docs)}; samples={source_hash_mismatches[:3]}"
        )

    dense_columns = ["hash_id", "content", "metadata"]
    dense_frame = pd.read_parquet(dense_parquet, columns=dense_columns)
    dense_row_count = len(dense_frame)
    if dense_frame["hash_id"].nunique() != dense_row_count:
        raise RuntimeError("Reused A2RAG dense index has duplicate chunk IDs")
    dense_frame["parsed_metadata"] = [parse_metadata(value) for value in dense_frame["metadata"]]
    dense_frame["source_id"] = [str(value.get("source_id") or "") for value in dense_frame["parsed_metadata"]]
    dense_frame["chunk_index"] = [int(value.get("chunk_index") or 0) for value in dense_frame["parsed_metadata"]]

    missing_sources = sorted(set(dense_frame["source_id"]) - set(resolver_by_source_id))
    if missing_sources:
        raise RuntimeError(f"Dense chunks have source IDs outside the Phase 2 resolver: {missing_sources[:5]}")

    evidence_rows: list[dict[str, Any]] = []
    excluded_samples: list[dict[str, Any]] = []
    table_classification_rows: list[dict[str, Any]] = []
    table_sample_buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    excluded_counts: Counter[str] = Counter()
    alignment_counts: Counter[str] = Counter()
    table_classification_counts: Counter[str] = Counter()
    table_subject_counts: Counter[str] = Counter()
    table_mixed_subject_counts: Counter[str] = Counter()
    semantic_tag_counts: Counter[str] = Counter()
    document_chunk_ids: dict[str, list[str]] = defaultdict(list)
    source_line_counts: dict[str, int] = {}

    for source_id, group in dense_frame.groupby("source_id", sort=True):
        resolver = resolver_by_source_id[source_id]
        corpus_doc = corpus_by_source_id[source_id]
        source_path = Path(str(resolver["markdown_path"]))
        source_text = source_path.read_text(encoding="utf-8", errors="replace")
        actual_source_hash = sha256_text(source_text)
        if actual_source_hash != corpus_doc.get("content_sha256"):
            raise RuntimeError(f"Source Markdown changed after Phase 2: {source_path}")
        normalized_source, source_offsets = normalized_source_with_offsets(source_text)
        starts = line_start_offsets(source_text)
        source_lines = source_text.splitlines(keepends=True)
        _, sections_by_heading = parse_markdown_sections(source_text, source_offsets)
        source_line_counts[source_id] = len(source_text.splitlines())
        cursor = 0

        for _, dense_row in group.sort_values("chunk_index", kind="stable").iterrows():
            chunk_id = str(dense_row["hash_id"])
            content = str(dense_row["content"])
            metadata = dict(dense_row["parsed_metadata"])
            section_path = [str(value) for value in metadata.get("heading_path") or [] if str(value).strip()]
            if TABLE_HTML_RE.search(content):
                excluded_counts["contains_table_html"] += 1
                classification = table_audit_classification(content, section_path)
                table_classification_counts[classification["classification"]] += 1
                table_subject_counts[classification["subject_category"]] += 1
                if classification["mixed_narrative"]:
                    table_mixed_subject_counts[classification["subject_category"]] += 1
                table_row = {
                    "schema_version": "finglmqa.phase7.table_exclusion_classification.v1",
                    "a2rag_chunk_id": chunk_id,
                    "document_id": resolver["document_id"],
                    "source_id": source_id,
                    "chunk_index": metadata.get("chunk_index"),
                    "heading_path": section_path,
                    "classification": classification["classification"],
                    "subject_category": classification["subject_category"],
                    "mixed_narrative": classification["mixed_narrative"],
                    "outside_table_text_sha256": sha256_text(classification["outside_table_text"]),
                    "outside_table_text_preview": classification["outside_table_text"][:300],
                    "content_sha256": sha256_text(content),
                    "policy": "excluded_no_safe_narrative_fragment_provenance",
                }
                table_classification_rows.append(table_row)
                sample_keys = [classification["classification"]]
                if classification["mixed_narrative"]:
                    sample_keys.append(f"mixed_narrative:{classification['subject_category']}")
                for key in sample_keys:
                    if len(table_sample_buckets[key]) < 20:
                        table_sample_buckets[key].append({"sample_stratum": key, **table_row})
                continue

            normalized_chunk = normalize_without_whitespace(content)
            if not normalized_chunk:
                excluded_counts["empty_after_normalization"] += 1
                continue

            alignment = align_chunk_to_section(
                normalized_source=normalized_source,
                normalized_chunk=normalized_chunk,
                heading_path=section_path,
                sections_by_heading=sections_by_heading,
                cursor=cursor,
            )
            if alignment["status"] != "ok":
                reason = str(alignment["status"])
                excluded_counts[reason] += 1
                if len(excluded_samples) < 200:
                    excluded_samples.append({
                        "a2rag_chunk_id": chunk_id,
                        "source_id": source_id,
                        "chunk_index": metadata.get("chunk_index"),
                        "reason": reason,
                        "alignment_diagnostics": {
                            key: value
                            for key, value in alignment.items()
                            if key not in {"section"}
                        },
                        "heading_path": section_path,
                        "content_sha256": sha256_text(content),
                        "content_preview": content[:300],
                    })
                continue

            position = int(alignment["position"])
            cursor_before = cursor
            cursor = int(alignment["cursor_after"])
            alignment_method = str(alignment["alignment_method"])
            section = alignment["section"]
            start_offset = source_offsets[position]
            end_offset = source_offsets[position + len(normalized_chunk) - 1]
            line_range = [line_for_offset(starts, start_offset), line_for_offset(starts, end_offset)]
            source_span = "".join(source_lines[line_range[0] - 1 : line_range[1]])
            if normalized_chunk not in normalize_without_whitespace(source_span):
                raise RuntimeError(f"Internal line provenance failure for {chunk_id}")

            tags = semantic_tags(section_path, content)
            semantic_tag_counts.update(tags)
            alignment_counts[alignment_method] += 1

            evidence_row = {
                "schema_version": SCHEMA_EVIDENCE_CHUNK,
                "builder_version": BUILDER_VERSION,
                "evidence_chunk_id": chunk_id,
                "a2rag_chunk_id": chunk_id,
                "document_id": resolver["document_id"],
                "company_name": resolver["stock_name"],
                "company_full": resolver.get("company_full"),
                "stock_code": resolver["stock_code"],
                "stock_symbol": resolver.get("stock_symbol"),
                "report_year": int(resolver["report_year"]),
                "section_path": section_path,
                "semantic_tags": tags,
                "line_range": line_range,
                "source_markdown": relative_path(source_path, root),
                "source_content_sha256": actual_source_hash,
                "content": content,
                "content_sha256": sha256_text(content),
                "token_count": int(metadata.get("token_count") or 0),
                "a2rag_metadata": {
                    "runtime_version": "2.0.0-alpha.5",
                    "embedding_model": "BAAI/bge-m3",
                    "chunking_mode": metadata.get("chunking_mode"),
                    "chunk_index": metadata.get("chunk_index"),
                    "source_id": source_id,
                    "heading_path": metadata.get("heading_path") or [],
                },
                "provenance": {
                    "alignment_method": alignment_method,
                    "normalized_character_start": position,
                    "normalized_character_end": position + len(normalized_chunk),
                    "source_character_start": start_offset,
                    "source_character_end": end_offset + 1,
                    "cursor_before": cursor_before,
                    "cursor_after": cursor,
                    "cursor_monotonic": cursor >= cursor_before,
                    "target_heading": section.heading,
                    "section_heading_line": section.heading_line,
                    "section_line_range": [section.heading_line, section.section_line_end],
                    "heading_match_count": alignment["heading_match_count"],
                    "compatible_content_match_count": alignment["candidate_count"],
                    "contains_table_html": False,
                    "table_chunk_policy": "excluded_from_type3_allow_list",
                },
            }
            evidence_rows.append(evidence_row)
            document_chunk_ids[str(resolver["document_id"])].append(chunk_id)

    evidence_rows.sort(key=lambda row: (row["document_id"], row["a2rag_metadata"]["chunk_index"], row["a2rag_chunk_id"]))
    evidence_ids = [row["a2rag_chunk_id"] for row in evidence_rows]
    if len(evidence_ids) != len(set(evidence_ids)):
        raise RuntimeError("Evidence chunk IDs are not unique")
    if set(document_chunk_ids) != set(resolver_by_document_id):
        missing_docs = sorted(set(resolver_by_document_id) - set(document_chunk_ids))
        raise RuntimeError(f"Evidence index does not cover all resolver documents: {missing_docs}")

    document_map_rows: list[dict[str, Any]] = []
    runtime_documents: list[dict[str, Any]] = []
    for document_id in sorted(document_chunk_ids):
        resolver = resolver_by_document_id[document_id]
        chunk_ids = document_chunk_ids[document_id]
        source_id = Path(str(resolver["markdown_path"])).name
        doc_row = {
            "schema_version": SCHEMA_DOCUMENT_CHUNK_MAP,
            "document_id": document_id,
            "company_name": resolver["stock_name"],
            "company_full": resolver.get("company_full"),
            "stock_code": resolver["stock_code"],
            "stock_symbol": resolver.get("stock_symbol"),
            "report_year": int(resolver["report_year"]),
            "source_id": source_id,
            "source_markdown": relative_path(Path(str(resolver["markdown_path"])), root),
            "content_sha256": corpus_by_source_id[source_id]["content_sha256"],
            "chunk_count": len(chunk_ids),
            "chunk_ids": chunk_ids,
        }
        document_map_rows.append(doc_row)
        runtime_documents.append({
            "source_id": source_id,
            "source_path": doc_row["source_markdown"],
            "content_sha256": doc_row["content_sha256"],
            "num_chunks": len(chunk_ids),
            "chunk_ids": chunk_ids,
            "indexed_at": None,
        })

    atomic_write_jsonl(args.evidence_output, evidence_rows)
    atomic_write_json(args.schema_output, evidence_schema())
    document_map_path = args.index_dir / "document_chunk_map.jsonl"
    atomic_write_jsonl(document_map_path, document_map_rows)

    runtime_label_dir = args.index_dir / A2RAG_LABEL
    runtime_documents_path = runtime_label_dir / "documents.json"
    atomic_write_json(runtime_documents_path, {"documents": runtime_documents})
    ensure_symlink(runtime_label_dir / "chunk_embeddings/vdb_chunk.parquet", dense_parquet)
    ensure_symlink(args.index_dir / "evidence_chunks.jsonl", args.evidence_output)

    evidence_sha = sha256_file(args.evidence_output)
    document_map_sha = sha256_file(document_map_path)
    schema_sha = sha256_file(args.schema_output)
    source_dense_sha = sha256_file(dense_parquet)
    parquet_file = pq.ParquetFile(dense_parquet)
    vector_dimension = embedding_dimension(dense_parquet)

    reports_dir = args.run_dir / "reports"
    table_classification_path = reports_dir / "table_exclusion_classification.jsonl"
    table_samples_path = reports_dir / "table_exclusion_samples.jsonl"
    table_audit_path = reports_dir / "table_exclusion_audit.json"
    atomic_write_jsonl(table_classification_path, table_classification_rows)
    atomic_write_jsonl(
        table_samples_path,
        [row for key in sorted(table_sample_buckets) for row in table_sample_buckets[key]],
    )
    table_audit = {
        "schema_version": "finglmqa.phase7.table_exclusion_audit.v1",
        "policy": "All table-bearing chunks remain excluded; mixed narrative is not recovered without fragment-level vector/provenance support.",
        "classified_chunks": len(table_classification_rows),
        "classification_counts": dict(sorted(table_classification_counts.items())),
        "subject_counts": dict(sorted(table_subject_counts.items())),
        "mixed_narrative_by_subject": dict(sorted(table_mixed_subject_counts.items())),
        "sample_count_by_stratum": {key: len(rows) for key, rows in sorted(table_sample_buckets.items())},
        "classification_artifact": relative_path(table_classification_path, root),
        "sample_artifact": relative_path(table_samples_path, root),
    }
    atomic_write_json(table_audit_path, table_audit)

    deterministic_hashes = {
        "evidence_chunks_sha256": evidence_sha,
        "document_chunk_map_sha256": document_map_sha,
        "evidence_schema_sha256": schema_sha,
        "source_dense_parquet_sha256": source_dense_sha,
        "table_exclusion_classification_sha256": sha256_file(table_classification_path),
        "table_exclusion_samples_sha256": sha256_file(table_samples_path),
        "table_exclusion_audit_sha256": sha256_file(table_audit_path),
    }
    previous_repeatability = None
    repeatability_path = args.run_dir / "repeatability_report.json"
    if repeatability_path.exists():
        previous_repeatability = load_json(repeatability_path).get("current_hashes")
    repeated_hashes_stable = previous_repeatability == deterministic_hashes if previous_repeatability else None

    index_manifest = {
        "schema_version": SCHEMA_INDEX_MANIFEST,
        "builder_version": BUILDER_VERSION,
        "built_from_corpus_generated_at_utc": corpus_manifest.get("generated_at_utc"),
        "a2rag": {
            "runtime_version": "2.0.0-alpha.5",
            "runtime_git_commit": runtime_git_commit(args.runtime_dir),
            "retrieval_mode": "dpr",
            "embedding_model": "BAAI/bge-m3",
            "embedding_dimension": vector_dimension,
            "chunking_mode": "markdown",
            "chunk_min_tokens": 800,
            "chunk_target_tokens": 1000,
            "chunk_max_tokens": 1200,
            "chunk_overlap_tokens": 100,
        },
        "index_strategy": {
            "dense_vectors": "reused_read_only_symlink",
            "company_year_prefilter": "required_before_dense_scoring",
            "type3_allow_list": "non_table_and_source_aligned_chunks_only",
            "numeric_source": "phase6_selected_financial_facts_only",
            "direct_unfiltered_a2rag_retrieve_allowed": False,
        },
        "inputs": {
            "corpus_manifest": relative_path(args.corpus_manifest, root),
            "company_year_index": relative_path(args.company_year_index, root),
            "source_dense_parquet": relative_path(dense_parquet, root),
            "source_documents_manifest": relative_path(source_documents_path, root),
        },
        "external_symlink_audit": {
            "source_dense_parquet_resolved": dense_parquet.resolve().as_posix(),
            "source_documents_manifest_resolved": source_documents_path.resolve().as_posix(),
            "source_markdown_root_resolved": (root / "refs/source_markdown").resolve().as_posix(),
        },
        "artifacts": {
            "evidence_chunks": relative_path(args.evidence_output, root),
            "document_chunk_map": relative_path(document_map_path, root),
            "evidence_schema": relative_path(args.schema_output, root),
            "runtime_dense_parquet_symlink": relative_path(runtime_label_dir / "chunk_embeddings/vdb_chunk.parquet", root),
            "runtime_documents": relative_path(runtime_documents_path, root),
            "table_exclusion_classification": relative_path(table_classification_path, root),
            "table_exclusion_samples": relative_path(table_samples_path, root),
            "table_exclusion_audit": relative_path(table_audit_path, root),
        },
        "counts": {
            "source_documents": len(corpus_docs),
            "source_hash_matches": source_hash_matches,
            "source_dense_chunks": dense_row_count,
            "source_dense_parquet_rows": parquet_file.metadata.num_rows,
            "evidence_chunks": len(evidence_rows),
            "evidence_documents": len(document_chunk_ids),
            "excluded_chunks": sum(excluded_counts.values()),
            "excluded_by_reason": dict(sorted(excluded_counts.items())),
            "alignment_methods": dict(sorted(alignment_counts.items())),
            "semantic_tag_counts": dict(sorted(semantic_tag_counts.items())),
            "table_exclusion_classification": dict(sorted(table_classification_counts.items())),
            "table_subject_counts": dict(sorted(table_subject_counts.items())),
        },
        "hashes": deterministic_hashes,
    }
    atomic_write_json(args.index_dir / "index_manifest.json", index_manifest)

    atomic_write_json(repeatability_path, {
        "schema_version": "finglmqa.phase7.repeatability_report.v1",
        "builder_version": BUILDER_VERSION,
        "previous_hashes": previous_repeatability,
        "current_hashes": deterministic_hashes,
        "repeated_hashes_stable": repeated_hashes_stable,
        "note": "Run the builder a second time; stable becomes true when all deterministic hashes match.",
    })

    atomic_write_jsonl(reports_dir / "excluded_chunk_samples.jsonl", excluded_samples)
    build_report = {
        "schema_version": SCHEMA_BUILD_REPORT,
        "builder_version": BUILDER_VERSION,
        "command": " ".join(sys.argv),
        "inputs": index_manifest["inputs"],
        "artifacts": index_manifest["artifacts"],
        "counts": index_manifest["counts"],
        "hashes": deterministic_hashes,
        "validations": {
            "resolver_rows_unique": True,
            "source_hashes_match": source_hash_matches == len(corpus_docs),
            "dense_chunk_ids_unique": True,
            "evidence_chunk_ids_unique": True,
            "evidence_covers_all_documents": len(document_chunk_ids) == len(corpus_docs),
            "evidence_ids_are_dense_ids": set(evidence_ids).issubset(set(dense_frame["hash_id"])),
            "all_evidence_chunks_line_aligned_during_build": sum(alignment_counts.values()) == len(evidence_rows),
            "all_evidence_alignments_section_compatible": all(
                row["provenance"].get("compatible_content_match_count") == 1
                and row["provenance"].get("section_heading_line")
                for row in evidence_rows
            ),
            "all_alignment_cursors_monotonic": all(
                row["provenance"].get("cursor_monotonic") is True for row in evidence_rows
            ),
            "all_table_html_chunks_excluded": all(not TABLE_HTML_RE.search(row["content"]) for row in evidence_rows),
            "all_table_html_chunks_classified": len(table_classification_rows) == excluded_counts["contains_table_html"],
            "runtime_vector_is_symlink": (runtime_label_dir / "chunk_embeddings/vdb_chunk.parquet").is_symlink(),
            "runtime_vector_target_matches": (runtime_label_dir / "chunk_embeddings/vdb_chunk.parquet").resolve() == dense_parquet.resolve(),
            "source_index_was_not_copied": (runtime_label_dir / "chunk_embeddings/vdb_chunk.parquet").is_symlink(),
        },
        "repeatability_after_this_run": repeated_hashes_stable,
    }
    atomic_write_json(args.run_dir / "build_report.json", build_report)

    print(json.dumps({
        "builder_version": BUILDER_VERSION,
        "counts": index_manifest["counts"],
        "hashes": deterministic_hashes,
        "repeatability_after_this_run": repeated_hashes_stable,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
