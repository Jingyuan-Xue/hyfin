#!/usr/bin/env python3
"""Prepare local-only, sanitized inputs for a replaceable Type 3 corpus profile."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import unicodedata
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS_ID = "annual_reports_nano5_2021_2023_v1"
DEFAULT_SOURCE = Path(
    "/home/coder/demo/financial_report_testset_5companies_2021_2023/"
    "final_5companies_30files/md"
)
DEFAULT_FR20 = Path("/home/coder/demo/NANO-Finbenchmark/data/fr_20_ragas_seed.jsonl")
DEFAULT_EXISTING = (
    ROOT
    / "data/corpus_package/type3/annual_reports_170_v1/corpus_manifest.json"
)
UPSTREAM_FIELDS = (
    "mapped_report_uids",
    "question",
    "question_file",
    "source_id",
    "source_split",
    "type",
    "uid",
)
_FILE_RE = re.compile(
    r"^A(?P<stock_code>\d{6})_(?P<company>.+?)_(?P<year>\d{4})年年度报告\.md$"
)


class NewCorpusPreparationError(ValueError):
    """Raised when a proposed new-corpus input is ambiguous or unsafe."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    atomic_write(path, canonical_bytes(value))


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    atomic_write(path, b"".join(canonical_bytes(dict(row)) for row in rows))


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise NewCorpusPreparationError(f"blank JSONL row: {path}:{line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise NewCorpusPreparationError(
                    f"expected object: {path}:{line_number}"
                )
            yield value


def clean_company(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    return re.sub(r"\s+", "", normalized).strip("_")


def parse_report_filename(name: str) -> dict[str, Any]:
    match = _FILE_RE.fullmatch(unicodedata.normalize("NFKC", name))
    if match is None:
        raise NewCorpusPreparationError(f"unsupported report filename: {name}")
    company = clean_company(match.group("company"))
    if not company:
        raise NewCorpusPreparationError(f"empty company in filename: {name}")
    return {
        "stock_code": match.group("stock_code"),
        "company": company,
        "report_year": int(match.group("year")),
    }


def parse_report_path(path: Path) -> dict[str, Any]:
    parsed = parse_report_filename(path.name)
    return {
        "document_id": path.stem,
        **parsed,
        "source_path": path.resolve(),
        "source_name": path.name,
        "source_sha256": sha256_file(path),
    }


def discover_reports(source_root: Path) -> list[dict[str, Any]]:
    paths = sorted(source_root.rglob("*.md"))
    reports = [parse_report_path(path) for path in paths]
    document_ids = [str(row["document_id"]) for row in reports]
    stock_year = [
        (str(row["stock_code"]), int(row["report_year"])) for row in reports
    ]
    source_names = [str(row["source_name"]) for row in reports]
    if len(set(document_ids)) != len(reports):
        raise NewCorpusPreparationError("document_id collision")
    if len(set(stock_year)) != len(reports):
        raise NewCorpusPreparationError("stock/year collision")
    if len(set(source_names)) != len(reports):
        raise NewCorpusPreparationError("flat source filename collision")
    return reports


def existing_stock_codes(corpus_manifest: Path) -> set[str]:
    value = json.loads(corpus_manifest.read_text(encoding="utf-8"))
    return {
        str(row["stock_code"])
        for row in value.get("documents") or ()
        if isinstance(row, Mapping) and row.get("stock_code")
    }


def ensure_flat_symlinks(
    reports: list[dict[str, Any]],
    source_ref: Path,
) -> None:
    source_ref.mkdir(parents=True, exist_ok=True)
    expected_names = {str(row["source_name"]) for row in reports}
    unexpected = [
        path.name for path in source_ref.iterdir() if path.name not in expected_names
    ]
    if unexpected:
        raise NewCorpusPreparationError(
            f"source-ref contains unexpected entries: {sorted(unexpected)!r}"
        )
    for report in reports:
        link = source_ref / str(report["source_name"])
        target = Path(report["source_path"])
        if link.exists() or link.is_symlink():
            if not link.is_symlink() or link.resolve() != target.resolve():
                raise NewCorpusPreparationError(f"source-ref collision: {link}")
            continue
        link.symlink_to(target)


def upstream_manifest(
    reports: list[dict[str, Any]],
    *,
    corpus_id: str,
) -> dict[str, Any]:
    selected = [
        {
            "doc_id": row["document_id"],
            "a2rag_doc": Path(row["source_path"]).as_posix(),
            "source_markdown": Path(row["source_path"]).as_posix(),
            "stock_code": row["stock_code"],
            "stock_name": row["company"],
            "company_full": row["company"],
            "report_year": row["report_year"],
            "aliases": [
                row["stock_code"],
                f"A{row['stock_code']}",
                row["company"],
            ],
        }
        for row in reports
    ]
    return {
        "schema_version": "finglmqa.type3.new_corpus_upstream.v1",
        "corpus_id": corpus_id,
        "report_count": len(selected),
        "selected_reports": selected,
        "local_only": True,
        "license_status": "unverified_do_not_redistribute",
    }


def source_uid(source_doc: str) -> str:
    parsed = parse_report_filename(Path(source_doc).name)
    return f"A{parsed['stock_code']}_{parsed['report_year']}"


def sanitize_list_questions(
    fr20_path: Path,
    *,
    type_label: str = "nano-type3-list",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for value in read_jsonl(fr20_path):
        if value.get("subtype") != "list_retrieval":
            continue
        uid = str(value.get("id") or "")
        question = str(value.get("question") or "").strip()
        source_doc = str(value.get("source_doc") or "")
        if not uid or not question or not source_doc or uid in seen:
            raise NewCorpusPreparationError("invalid list-retrieval question")
        seen.add(uid)
        row = {
            "mapped_report_uids": [source_uid(source_doc)],
            "question": question,
            "question_file": fr20_path.as_posix(),
            "source_id": uid,
            "source_split": "nano-fr20-local",
            "type": type_label,
            "uid": uid,
        }
        if set(row) != set(UPSTREAM_FIELDS):
            raise AssertionError("sanitized question schema drift")
        rows.append(row)
    return rows


def structural_questions(
    reports: list[dict[str, Any]],
    *,
    type_label: str = "nano-type3-structural",
) -> list[dict[str, Any]]:
    templates = (
        "{company}{year}年年度报告披露的主要业务和产品有哪些？",
        "{company}{year}年经营表现发生了哪些主要变化，原因是什么？",
        "{company}{year}年面临哪些主要风险，采取了哪些应对措施？",
        "{company}{year}年在研发创新、现金流或关键财务指标方面有哪些重要变化？",
    )
    rows: list[dict[str, Any]] = []
    for report in reports:
        mapped = f"A{report['stock_code']}_{report['report_year']}"
        for ordinal, template in enumerate(templates, 1):
            uid = (
                f"struct-{report['stock_code']}-{report['report_year']}-"
                f"{ordinal:02d}"
            )
            rows.append(
                {
                    "mapped_report_uids": [mapped],
                    "question": template.format(
                        company=report["company"], year=report["report_year"]
                    ),
                    "question_file": "generated:generic_type3_structural_v1",
                    "source_id": uid,
                    "source_split": "local-structural-smoke",
                    "type": type_label,
                    "uid": uid,
                }
            )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--fr20", type=Path, default=DEFAULT_FR20)
    parser.add_argument("--existing-corpus", type=Path, default=DEFAULT_EXISTING)
    parser.add_argument("--corpus-id", default=DEFAULT_CORPUS_ID)
    parser.add_argument("--expected-documents", type=int, default=15)
    parser.add_argument("--expected-list-questions", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    source_root = args.source_root.resolve()
    fr20 = args.fr20.resolve()
    existing = args.existing_corpus.resolve()
    output_dir = args.output_dir.resolve()
    source_ref = repo_root / "refs/type3_corpora" / args.corpus_id
    reports = discover_reports(source_root)
    if len(reports) != args.expected_documents:
        raise NewCorpusPreparationError(
            f"expected {args.expected_documents} documents, got {len(reports)}"
        )
    new_codes = {str(row["stock_code"]) for row in reports}
    overlap = new_codes.intersection(existing_stock_codes(existing))
    if overlap:
        raise NewCorpusPreparationError(
            f"new corpus overlaps existing stock codes: {sorted(overlap)!r}"
        )
    ensure_flat_symlinks(reports, source_ref)
    list_questions = sanitize_list_questions(fr20)
    if len(list_questions) != args.expected_list_questions:
        raise NewCorpusPreparationError(
            f"expected {args.expected_list_questions} list questions, "
            f"got {len(list_questions)}"
        )
    structural = structural_questions(reports)

    upstream_path = output_dir / "selected_reports.json"
    list_path = output_dir / "nano_type3_list_questions.jsonl"
    structural_path = output_dir / "structural_smoke_questions.jsonl"
    write_json(upstream_path, upstream_manifest(reports, corpus_id=args.corpus_id))
    write_jsonl(list_path, list_questions)
    write_jsonl(structural_path, structural)
    before = {
        str(row["source_path"]): str(row["source_sha256"]) for row in reports
    }
    after = {
        str(row["source_path"]): sha256_file(Path(row["source_path"]))
        for row in reports
    }
    if before != after:
        raise NewCorpusPreparationError("source Markdown changed during preparation")
    manifest = {
        "schema_version": "finglmqa.type3.new_corpus_preparation.v1",
        "corpus_id": args.corpus_id,
        "local_only": True,
        "license_status": "unverified_do_not_redistribute",
        "source_root": source_root.as_posix(),
        "source_ref": source_ref.relative_to(repo_root).as_posix(),
        "documents": len(reports),
        "stock_codes": sorted(new_codes),
        "report_years": sorted({int(row["report_year"]) for row in reports}),
        "existing_stock_code_overlap": [],
        "list_questions": len(list_questions),
        "structural_questions": len(structural),
        "source_unchanged": True,
        "inputs": {
            "existing_corpus_sha256": sha256_file(existing),
            "fr20_sha256": sha256_file(fr20),
            "source_hashes_sha256": hashlib.sha256(
                canonical_bytes(before)
            ).hexdigest(),
        },
        "artifacts": {
            "selected_reports.json": sha256_file(upstream_path),
            "nano_type3_list_questions.jsonl": sha256_file(list_path),
            "structural_smoke_questions.jsonl": sha256_file(structural_path),
        },
    }
    write_json(output_dir / "preparation_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
