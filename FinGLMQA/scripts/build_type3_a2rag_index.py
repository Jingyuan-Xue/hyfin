#!/usr/bin/env python3
"""Build the isolated Type 3 A2RAG text package and document shards.

Only the corpus profile, strict UTF-8 Markdown, and complete Phase 3 table
ranges are accepted as evidence inputs.  Question files and benchmark
annotations are intentionally absent from this command-line contract.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from finglmqa.type3_a2rag_retriever import (  # noqa: E402
    ATOM_SCHEMA,
    BUILDER_VERSION,
    EMBEDDING_DIMENSION,
    INDEX_SCHEMA,
    UNIT_SCHEMA,
    Type3A2RAGError,
    build_bm25_shard,
    build_retrieval_units,
    build_text_atoms,
    canonical_json_bytes,
    read_json,
    read_jsonl,
    retrieval_text,
    semantic_sha256,
    sha256_bytes,
    sha256_file,
    shard_id,
    validate_table_ranges,
)
from finglmqa.type3_corpus_profile import (  # noqa: E402
    source_snapshot,
    validate_corpus_profile,
)


MAX_UNITS = 150_000
MAX_PREDICTED_BYTES = 3 * 1024**3
PROBE_UNIT_COUNT = 1_000
PROBE_BATCH_SIZE = 8
DEFAULT_MODEL = Path(
    "/home/coder/demo/models/models--BAAI--bge-m3/snapshots/"
    "5617a9f61b028005a4858fdac845db406aefb181"
)


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return canonical_json_bytes(dict(value))


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(_json_bytes(value))
    temporary.replace(path)


def atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        for row in rows:
            handle.write(_json_bytes(row))
    temporary.replace(path)


def _strict_source(path: Path, expected_sha256: str) -> tuple[bytes, str]:
    raw = path.read_bytes()
    actual = sha256_bytes(raw)
    if actual != expected_sha256:
        raise Type3A2RAGError(f"source hash drift: {path.name}")
    try:
        source = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise Type3A2RAGError(f"source is not strict UTF-8: {path.name}") from exc
    if source.encode("utf-8") != raw:
        raise Type3A2RAGError(f"UTF-8 round trip differs: {path.name}")
    return raw, source


def _table_path(table_documents_dir: Path, document_id: str) -> Path:
    path = table_documents_dir / document_id / "table_blocks.jsonl"
    if not path.is_file():
        raise Type3A2RAGError(f"table block shard missing: {document_id}")
    return path


def parse_document(
    *,
    profile: Mapping[str, Any],
    document: Mapping[str, Any],
    source_root: Path,
    table_documents_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    source_path = source_root / str(document["source_markdown"])
    raw, source = _strict_source(source_path, str(document["source_sha256"]))
    table_path = _table_path(table_documents_dir, str(document["document_id"]))
    table_rows = list(read_jsonl(table_path))
    tables = validate_table_ranges(
        source, table_rows, document_id=str(document["document_id"])
    )
    source_markdown = (
        Path(str(profile["source_ref"])) / str(document["source_markdown"])
    ).as_posix()
    atoms = build_text_atoms(
        corpus_id=str(profile["corpus_id"]),
        document_id=str(document["document_id"]),
        source_markdown=source_markdown,
        source_sha256=str(document["source_sha256"]),
        source=source,
        tables=tables,
    )
    units = build_retrieval_units(
        corpus_id=str(profile["corpus_id"]),
        document_id=str(document["document_id"]),
        source_markdown=source_markdown,
        source_sha256=str(document["source_sha256"]),
        source=source,
        atoms=atoms,
    )
    table_intervals = [(value.start, value.end) for value in tables]
    for atom in atoms:
        start, end = atom["char_range"]
        if any(start < table_end and end > table_start for table_start, table_end in table_intervals):
            raise Type3A2RAGError(f"atom overlaps table: {atom['atom_id']}")
    if not units:
        raise Type3A2RAGError(f"document has no retrieval units: {document['document_id']}")
    summary = {
        "schema_version": "finglmqa.type3.a2rag.document_summary.v1",
        "corpus_id": profile["corpus_id"],
        "document_id": document["document_id"],
        "source_markdown": source_markdown,
        "source_sha256": document["source_sha256"],
        "source_bytes": len(raw),
        "source_chars": len(source),
        "table_count": len(tables),
        "atom_count": len(atoms),
        "unit_count": len(units),
        "heading_atom_count": sum(row["atom_kind"] == "heading" for row in atoms),
        "paragraph_atom_count": sum(row["atom_kind"] == "paragraph" for row in atoms),
        "list_atom_count": sum(row["atom_kind"] == "list" for row in atoms),
        "adjacent_atom_count": sum(bool(row["adjacent_table_ids"]) for row in atoms),
        "adjacent_unit_count": sum(bool(row["adjacent_table_ids"]) for row in units),
        "max_unit_chars": max(row["char_range"][1] - row["char_range"][0] for row in units),
        "oversize_single_atom_unit_count": sum(bool(row["oversize_single_atom"]) for row in units),
        "atoms_semantic_sha256": semantic_sha256(atoms),
        "units_semantic_sha256": semantic_sha256(units),
        "table_blocks_sha256": sha256_file(table_path),
    }
    return atoms, units, summary


def scan_corpus(args: argparse.Namespace, profile: Mapping[str, Any]) -> dict[str, Any]:
    source_root = (ROOT / str(profile["source_ref"])).resolve()
    documents = sorted(profile["documents"], key=lambda row: str(row["document_id"]))
    summaries: list[dict[str, Any]] = []
    exact_atom_bytes = 0
    exact_unit_bytes = 0
    exact_index_unit_bytes = 0
    exact_bm25_bytes = 0
    started = time.perf_counter()
    for ordinal, document in enumerate(documents, 1):
        atoms, units, summary = parse_document(
            profile=profile,
            document=document,
            source_root=source_root,
            table_documents_dir=args.table_documents_dir,
        )
        bm25 = build_bm25_shard(units)
        exact_atom_bytes += sum(len(_json_bytes(row)) for row in atoms)
        unit_bytes = sum(len(_json_bytes(row)) for row in units)
        exact_unit_bytes += unit_bytes
        exact_index_unit_bytes += unit_bytes
        exact_bm25_bytes += len(_json_bytes(bm25))
        summaries.append(summary)
        if ordinal % 20 == 0 or ordinal == len(documents):
            print(
                f"scan {ordinal}/{len(documents)} units={sum(row['unit_count'] for row in summaries)}",
                file=sys.stderr,
                flush=True,
            )
    unit_count = sum(int(row["unit_count"]) for row in summaries)
    atom_count = sum(int(row["atom_count"]) for row in summaries)
    dense_bytes = unit_count * EMBEDDING_DIMENSION * np.dtype(np.float16).itemsize * 2
    overhead_bytes = 64 * 1024**2
    predicted = (
        exact_atom_bytes + exact_unit_bytes + exact_index_unit_bytes
        + exact_bm25_bytes + dense_bytes + overhead_bytes
    )
    if unit_count > MAX_UNITS:
        raise Type3A2RAGError(f"unit count gate exceeded: {unit_count}>{MAX_UNITS}")
    if predicted > MAX_PREDICTED_BYTES:
        raise Type3A2RAGError(
            f"predicted disk gate exceeded: {predicted}>{MAX_PREDICTED_BYTES}"
        )
    if len(summaries) != int(profile["document_count"]):
        raise Type3A2RAGError("document count differs after scan")
    report = {
        "schema_version": "finglmqa.type3.a2rag.dry_run.v1",
        "builder_version": BUILDER_VERSION,
        "corpus_id": profile["corpus_id"],
        "corpus_profile_sha256": profile["profile_sha256"],
        "source_snapshot_sha256": semantic_sha256(source_snapshot(profile, workspace_root=ROOT)),
        "document_count": len(summaries),
        "table_count": sum(int(row["table_count"]) for row in summaries),
        "atom_count": atom_count,
        "unit_count": unit_count,
        "heading_atom_count": sum(int(row["heading_atom_count"]) for row in summaries),
        "paragraph_atom_count": sum(int(row["paragraph_atom_count"]) for row in summaries),
        "list_atom_count": sum(int(row["list_atom_count"]) for row in summaries),
        "adjacent_atom_count": sum(int(row["adjacent_atom_count"]) for row in summaries),
        "adjacent_unit_count": sum(int(row["adjacent_unit_count"]) for row in summaries),
        "max_unit_chars": max(int(row["max_unit_chars"]) for row in summaries),
        "oversize_single_atom_unit_count": sum(
            int(row["oversize_single_atom_unit_count"]) for row in summaries
        ),
        "predicted_artifact_bytes": predicted,
        "predicted_artifact_gib": round(predicted / 1024**3, 6),
        "gates": {
            "strict_utf8": True,
            "source_hashes_match": True,
            "table_ranges_complete_and_exact": True,
            "atoms_do_not_overlap_tables": True,
            "units_at_most_150000": unit_count <= MAX_UNITS,
            "predicted_disk_at_most_3gib": predicted <= MAX_PREDICTED_BYTES,
        },
        "document_summaries": summaries,
        "elapsed_seconds": round(time.perf_counter() - started, 6),
    }
    return report


def _model_revision(model_path: Path) -> dict[str, Any]:
    if not model_path.is_dir():
        raise Type3A2RAGError(f"local model directory missing: {model_path}")
    config = model_path / "config.json"
    modules = model_path / "modules.json"
    if not config.is_file() or not modules.is_file():
        raise Type3A2RAGError("local BGE-M3 snapshot is incomplete")
    return {
        "name": "BAAI/bge-m3",
        "local_path": model_path.resolve().as_posix(),
        "snapshot_revision": model_path.resolve().name,
        "config_sha256": sha256_file(config),
        "modules_sha256": sha256_file(modules),
        "dimension": EMBEDDING_DIMENSION,
        "normalized": True,
        "storage_dtype": "float16",
    }


def _load_model(model_path: Path) -> Any:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        model_path.resolve().as_posix(), device="cuda", local_files_only=True
    )
    model.max_seq_length = 8192
    dimension = int(model.get_sentence_embedding_dimension() or 0)
    if dimension != EMBEDDING_DIMENSION:
        raise Type3A2RAGError(f"BGE-M3 dimension mismatch: {dimension}")
    return model


def _encode(model: Any, texts: Sequence[str], *, batch_size: int = PROBE_BATCH_SIZE) -> np.ndarray:
    vectors = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    result = np.asarray(vectors, dtype=np.float32)
    if result.shape != (len(texts), EMBEDDING_DIMENSION):
        raise Type3A2RAGError(f"embedding output shape differs: {result.shape}")
    if not np.isfinite(result).all():
        raise Type3A2RAGError("embedding output contains non-finite values")
    return result


def probe(args: argparse.Namespace, profile: Mapping[str, Any]) -> dict[str, Any]:
    source_root = (ROOT / str(profile["source_ref"])).resolve()
    sample: list[dict[str, Any]] = []
    for document in sorted(profile["documents"], key=lambda row: str(row["document_id"])):
        _, units, _ = parse_document(
            profile=profile,
            document=document,
            source_root=source_root,
            table_documents_dir=args.table_documents_dir,
        )
        sample.extend(units[: PROBE_UNIT_COUNT - len(sample)])
        if len(sample) == PROBE_UNIT_COUNT:
            break
    if len(sample) != PROBE_UNIT_COUNT:
        raise Type3A2RAGError(f"GPU probe needs exactly {PROBE_UNIT_COUNT} units")
    started = time.perf_counter()
    try:
        model = _load_model(args.model_path)
        vectors = _encode(
            model,
            [retrieval_text(row, include_heading=True) for row in sample],
            batch_size=PROBE_BATCH_SIZE,
        )
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            raise Type3A2RAGError("GPU probe OOM; full build forbidden") from exc
        raise
    report = {
        "schema_version": "finglmqa.type3.a2rag.gpu_probe.v1",
        "builder_version": BUILDER_VERSION,
        "corpus_id": profile["corpus_id"],
        "unit_count": len(sample),
        "batch_size": PROBE_BATCH_SIZE,
        "device": "cuda",
        "embedding_shape": list(vectors.shape),
        "embedding_dtype": str(vectors.dtype),
        "embedding_sha256": sha256_bytes(vectors.astype(np.float32).tobytes(order="C")),
        "model": _model_revision(args.model_path),
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        "passed": True,
    }
    return report


def _npy_atomic(path: Path, values: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
    temporary.replace(path)


def _write_schemas(schema_dir: Path) -> dict[str, str]:
    schemas = {
        "a2rag_atom_v1.schema.json": {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": ATOM_SCHEMA,
            "type": "object",
            "required": [
                "schema_version", "atom_id", "corpus_id", "document_id", "atom_kind",
                "content", "content_sha256", "line_range", "char_range", "byte_range",
                "source_markdown", "source_sha256", "heading_path", "adjacent_table_ids",
            ],
            "additionalProperties": True,
        },
        "a2rag_unit_v1.schema.json": {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": UNIT_SCHEMA,
            "type": "object",
            "required": [
                "schema_version", "unit_id", "corpus_id", "document_id", "atom_ids",
                "content", "content_sha256", "line_range", "char_range", "byte_range",
                "source_markdown", "source_sha256", "heading_path", "adjacent_table_ids",
            ],
            "additionalProperties": True,
        },
        "a2rag_index_manifest_v1.schema.json": {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": INDEX_SCHEMA,
            "type": "object",
            "required": [
                "schema_version", "builder_version", "corpus_id", "embedding_model",
                "document_count", "unit_count", "documents", "artifacts",
            ],
            "additionalProperties": True,
        },
    }
    hashes: dict[str, str] = {}
    for name, value in schemas.items():
        path = schema_dir / name
        atomic_json(path, value)
        hashes[name] = sha256_file(path)
    return hashes


def build(args: argparse.Namespace, profile: Mapping[str, Any], dry_run: Mapping[str, Any]) -> dict[str, Any]:
    if args.package_dir.exists() or args.index_dir.exists():
        raise Type3A2RAGError("Phase 2 package/index target already exists; refusing overwrite")
    probe_report_path = args.run_dir / "gpu_probe_report.json"
    if not probe_report_path.is_file() or not read_json(probe_report_path).get("passed"):
        raise Type3A2RAGError("passing 1000-unit batch8 GPU probe is required before build")
    before = source_snapshot(profile, workspace_root=ROOT)
    source_root = (ROOT / str(profile["source_ref"])).resolve()
    args.package_dir.mkdir(parents=True, exist_ok=False)
    args.index_dir.mkdir(parents=True, exist_ok=False)
    shards_root = args.index_dir / "shards"
    shards_root.mkdir()
    model = _load_model(args.model_path)
    started = time.perf_counter()
    atom_path = args.package_dir / "text_atoms.jsonl"
    unit_path = args.package_dir / "retrieval_units.jsonl"
    summary_path = args.package_dir / "document_summaries.jsonl"
    atom_tmp = atom_path.with_suffix(".jsonl.tmp")
    unit_tmp = unit_path.with_suffix(".jsonl.tmp")
    summary_tmp = summary_path.with_suffix(".jsonl.tmp")
    manifest_documents: list[dict[str, Any]] = []
    all_summaries: list[dict[str, Any]] = []

    with ExitStack() as stack:
        atom_handle = stack.enter_context(atom_tmp.open("wb"))
        unit_handle = stack.enter_context(unit_tmp.open("wb"))
        summary_handle = stack.enter_context(summary_tmp.open("wb"))
        documents = sorted(profile["documents"], key=lambda row: str(row["document_id"]))
        for ordinal, document in enumerate(documents, 1):
            atoms, units, summary = parse_document(
                profile=profile,
                document=document,
                source_root=source_root,
                table_documents_dir=args.table_documents_dir,
            )
            for row in atoms:
                atom_handle.write(_json_bytes(row))
            for row in units:
                unit_handle.write(_json_bytes(row))
            summary_handle.write(_json_bytes(summary))
            all_summaries.append(summary)

            shard_name = shard_id(str(document["document_id"]))
            shard_dir = shards_root / shard_name
            shard_dir.mkdir()
            shard_units_path = shard_dir / "units.jsonl"
            bm25_path = shard_dir / "bm25.json"
            atomic_jsonl(shard_units_path, units)
            atomic_json(bm25_path, build_bm25_shard(units))
            contexts = [retrieval_text(row, include_heading=True) for row in units]
            contents = [retrieval_text(row, include_heading=False) for row in units]
            context_vectors = _encode(model, contexts).astype(np.float16)
            content_vectors = _encode(model, contents).astype(np.float16)
            context_path = shard_dir / "dense_context.npy"
            content_path = shard_dir / "dense_content.npy"
            _npy_atomic(context_path, context_vectors)
            _npy_atomic(content_path, content_vectors)
            manifest_documents.append(
                {
                    "document_id": document["document_id"],
                    "source_markdown": summary["source_markdown"],
                    "source_sha256": document["source_sha256"],
                    "shard_path": f"shards/{shard_name}",
                    "unit_count": len(units),
                    "unit_ids_sha256": semantic_sha256([row["unit_id"] for row in units]),
                    "artifacts": {
                        "units.jsonl": sha256_file(shard_units_path),
                        "bm25.json": sha256_file(bm25_path),
                        "dense_context.npy": sha256_file(context_path),
                        "dense_content.npy": sha256_file(content_path),
                    },
                }
            )
            print(
                f"build {ordinal}/{len(documents)} doc_units={len(units)}",
                file=sys.stderr,
                flush=True,
            )
    atom_tmp.replace(atom_path)
    unit_tmp.replace(unit_path)
    summary_tmp.replace(summary_path)
    after = source_snapshot(profile, workspace_root=ROOT)
    if before != after:
        raise Type3A2RAGError("source hash drift during build")

    schema_hashes = _write_schemas(args.schema_dir)
    package_manifest = {
        "schema_version": "finglmqa.type3.a2rag.text_package_manifest.v1",
        "builder_version": BUILDER_VERSION,
        "corpus_id": profile["corpus_id"],
        "corpus_profile_sha256": profile["profile_sha256"],
        "document_count": len(all_summaries),
        "table_count": sum(row["table_count"] for row in all_summaries),
        "atom_count": sum(row["atom_count"] for row in all_summaries),
        "unit_count": sum(row["unit_count"] for row in all_summaries),
        "source_snapshot_sha256": semantic_sha256(after),
        "inputs": {
            "corpus_manifest": args.corpus_manifest.resolve().as_posix(),
            "corpus_manifest_sha256": sha256_file(args.corpus_manifest),
            "table_documents_dir": args.table_documents_dir.resolve().as_posix(),
        },
        "artifacts": {
            "text_atoms.jsonl": sha256_file(atom_path),
            "retrieval_units.jsonl": sha256_file(unit_path),
            "document_summaries.jsonl": sha256_file(summary_path),
        },
    }
    atomic_json(args.package_dir / "manifest.json", package_manifest)
    index_manifest = {
        "schema_version": INDEX_SCHEMA,
        "builder_version": BUILDER_VERSION,
        "corpus_id": profile["corpus_id"],
        "corpus_profile_sha256": profile["profile_sha256"],
        "source_snapshot_sha256": semantic_sha256(after),
        "document_count": len(manifest_documents),
        "unit_count": sum(int(row["unit_count"]) for row in manifest_documents),
        "document_prefilter_required": True,
        "cross_document_scoring_allowed": False,
        "embedding_model": _model_revision(args.model_path),
        "dense_variants": {
            "context": "heading_path + exact source content",
            "content": "exact source content (no-heading ablation)",
        },
        "sparse": {
            "kind": "document-sharded BM25",
            "fields": ["body", "heading", "adjacency"],
            "tokenizer": "latin-word+han-unigram+han-bigram-v1",
        },
        "documents": manifest_documents,
        "artifacts": {
            "text_package_manifest": (args.package_dir / "manifest.json").resolve().as_posix(),
            "text_package_manifest_sha256": sha256_file(args.package_dir / "manifest.json"),
            "schemas": schema_hashes,
        },
    }
    atomic_json(args.index_dir / "index_manifest.json", index_manifest)
    report = {
        "schema_version": "finglmqa.type3.a2rag.build_report.v1",
        "builder_version": BUILDER_VERSION,
        "corpus_id": profile["corpus_id"],
        "document_count": index_manifest["document_count"],
        "table_count": package_manifest["table_count"],
        "atom_count": package_manifest["atom_count"],
        "unit_count": index_manifest["unit_count"],
        "dry_run_report_sha256": sha256_file(args.run_dir / "dry_run_report.json"),
        "gpu_probe_report_sha256": sha256_file(probe_report_path),
        "package_manifest_sha256": sha256_file(args.package_dir / "manifest.json"),
        "index_manifest_sha256": sha256_file(args.index_dir / "index_manifest.json"),
        "source_hashes_unchanged": before == after,
        "artifact_bytes": sum(
            path.stat().st_size
            for root in (args.package_dir, args.index_dir)
            for path in root.rglob("*") if path.is_file()
        ),
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        "passed": True,
    }
    return report


def _hash_expected_rows(
    args: argparse.Namespace, profile: Mapping[str, Any]
) -> tuple[str, str, str, list[dict[str, Any]], dict[str, dict[str, str]]]:
    atom_hash = hashlib.sha256()
    unit_hash = hashlib.sha256()
    summary_hash = hashlib.sha256()
    summaries: list[dict[str, Any]] = []
    shard_hashes: dict[str, dict[str, str]] = {}
    source_root = (ROOT / str(profile["source_ref"])).resolve()
    for document in sorted(profile["documents"], key=lambda row: str(row["document_id"])):
        atoms, units, summary = parse_document(
            profile=profile,
            document=document,
            source_root=source_root,
            table_documents_dir=args.table_documents_dir,
        )
        for row in atoms:
            atom_hash.update(_json_bytes(row))
        for row in units:
            unit_hash.update(_json_bytes(row))
        summary_hash.update(_json_bytes(summary))
        summaries.append(summary)
        shard_hashes[str(document["document_id"])] = {
            "units.jsonl": hashlib.sha256(b"".join(_json_bytes(row) for row in units)).hexdigest(),
            "bm25.json": hashlib.sha256(_json_bytes(build_bm25_shard(units))).hexdigest(),
        }
    return atom_hash.hexdigest(), unit_hash.hexdigest(), summary_hash.hexdigest(), summaries, shard_hashes


def verify_existing(
    args: argparse.Namespace, profile: Mapping[str, Any]
) -> dict[str, Any]:
    manifest = read_json(args.index_dir / "index_manifest.json")
    atom_hash, unit_hash, summary_hash, summaries, expected_shards = _hash_expected_rows(args, profile)
    package_manifest = read_json(args.package_dir / "manifest.json")
    checks = {
        "source_snapshot": manifest["source_snapshot_sha256"]
        == semantic_sha256(source_snapshot(profile, workspace_root=ROOT)),
        "text_atoms_rebuild": atom_hash == sha256_file(args.package_dir / "text_atoms.jsonl"),
        "retrieval_units_rebuild": unit_hash == sha256_file(args.package_dir / "retrieval_units.jsonl"),
        "document_summaries_rebuild": summary_hash
        == sha256_file(args.package_dir / "document_summaries.jsonl"),
        "package_atom_hash": package_manifest["artifacts"]["text_atoms.jsonl"] == atom_hash,
        "package_unit_hash": package_manifest["artifacts"]["retrieval_units.jsonl"] == unit_hash,
        "counts_rebuild": (
            sum(row["atom_count"] for row in summaries) == package_manifest["atom_count"]
            and sum(row["unit_count"] for row in summaries) == manifest["unit_count"]
        ),
    }
    for document in manifest["documents"]:
        document_id = str(document["document_id"])
        shard_dir = args.index_dir / str(document["shard_path"])
        for name in ("units.jsonl", "bm25.json"):
            checks[f"{document_id}:{name}:rebuild"] = (
                sha256_file(shard_dir / name) == expected_shards[document_id][name]
            )
        for name in ("dense_context.npy", "dense_content.npy"):
            checks[f"{document_id}:{name}:manifest_hash"] = (
                sha256_file(shard_dir / name) == document["artifacts"][name]
            )
            matrix = np.load(shard_dir / name, mmap_mode="r")
            checks[f"{document_id}:{name}:shape"] = matrix.shape == (
                int(document["unit_count"]), EMBEDDING_DIMENSION
            )
    failed = sorted(key for key, value in checks.items() if not value)
    return {
        "schema_version": "finglmqa.type3.a2rag.repeatability_report.v1",
        "builder_version": BUILDER_VERSION,
        "corpus_id": profile["corpus_id"],
        "reparsed_document_count": len(summaries),
        "dense_reencoded": False,
        "checks": checks,
        "failed_checks": failed,
        "passed": not failed,
    }


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus-manifest", type=Path,
        default=ROOT / "data/corpus_package/type3/annual_reports_170_v1/corpus_manifest.json",
    )
    parser.add_argument(
        "--table-documents-dir", type=Path,
        default=ROOT / "data/corpus_package/documents",
    )
    parser.add_argument(
        "--package-dir", type=Path,
        default=ROOT / "data/corpus_package/type3/annual_reports_170_v1/a2rag_text_v1",
    )
    parser.add_argument(
        "--index-dir", type=Path,
        default=ROOT / "data/indexes/type3/annual_reports_170_v1/a2rag",
    )
    parser.add_argument(
        "--schema-dir", type=Path, default=ROOT / "data/schemas/type3",
    )
    parser.add_argument(
        "--run-dir", type=Path,
        default=ROOT / "runs/type3_a2rag_tabgr_experiment_v1/annual_reports_170_v1/phase_2",
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--probe", action="store_true")
    mode.add_argument("--build", action="store_true")
    mode.add_argument("--verify-existing", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = arguments()
    profile = validate_corpus_profile(read_json(args.corpus_manifest))
    args.run_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        report = scan_corpus(args, profile)
        atomic_json(args.run_dir / "dry_run_report.json", report)
    elif args.probe:
        if not (args.run_dir / "dry_run_report.json").is_file():
            raise Type3A2RAGError("dry run report required before GPU probe")
        report = probe(args, profile)
        atomic_json(args.run_dir / "gpu_probe_report.json", report)
    elif args.build:
        dry_run = read_json(args.run_dir / "dry_run_report.json")
        if not all(dry_run.get("gates", {}).values()):
            raise Type3A2RAGError("dry run gates did not all pass")
        report = build(args, profile, dry_run)
        atomic_json(args.run_dir / "build_report.json", report)
    else:
        report = verify_existing(args, profile)
        atomic_json(args.run_dir / "repeatability_report.json", report)
        if not report["passed"]:
            raise Type3A2RAGError(f"repeatability failed: {report['failed_checks'][:5]}")
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
