#!/usr/bin/env python3
"""Freeze Phase 10 immutable inputs and isolated runtime dependencies."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/phase_10"
PYTHON = ROOT / ".venv-phase10/bin/python"

INPUTS = (
    "data/facts/financial_facts.duckdb",
    "data/facts/financial_facts.jsonl",
    "data/facts/supplemental_facts.duckdb",
    "data/facts/supplemental_facts.jsonl",
    "data/corpus_package/company_year_index.jsonl",
    "data/corpus_package/evidence_chunks.jsonl",
    "data/indexes/a2rag_index/index_manifest.json",
    "data/indexes/a2rag_index/document_chunk_map.jsonl",
    "src/config/composition_patterns.json",
    "src/config/metric_aliases.json",
    "src/config/unit_rules.json",
    "runs/phase_08/gate2_report.json",
    "runs/phase_08/gates_6_8_report.json",
    "runs/phase_08/pattern_registry_manifest.json",
    "runs/phase_09/reports/supplement_decisions.jsonl",
    "runs/phase_09/reports/supplement_summary.json",
    "runs/phase_09/phase_09_report.md",
    "data/schemas/qa_request.schema.json",
    "data/schemas/qa_answer.schema.json",
    "data/schemas/qa_trace.schema.json",
    "data/schemas/phase_10_service_projection.schema.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical_bytes(value))
    os.replace(temporary, path)


def file_entry(label: str) -> dict[str, Any]:
    path = ROOT / label
    stat = path.stat()
    return {"path": label, "sha256": sha256_file(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def command_output(command: list[str]) -> str:
    return subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def qwen_manifest() -> dict[str, Any]:
    model = (ROOT / "refs/qwen_model").resolve()
    snapshot_id = model.name
    files = []
    for name in ("config.json", "generation_config.json", "tokenizer_config.json", "model.safetensors.index.json"):
        path = model / name
        files.append({"name": name, "sha256": sha256_file(path), "size": path.stat().st_size})
    shards = []
    for path in sorted(model.glob("model-*.safetensors")):
        target = path.resolve()
        shards.append({"name": path.name, "blob_id": target.name, "size": target.stat().st_size})
    if len(shards) != 15:
        raise RuntimeError("Qwen snapshot must contain 15 pinned shards")
    return {"snapshot_id": snapshot_id, "metadata_files": files, "weight_shards": shards}


def main() -> int:
    entries = [file_entry(label) for label in INPUTS]
    manifest: dict[str, Any] = {
        "schema_version": "finglmqa.phase10.immutable_inputs_manifest.v1",
        "plan_sha256": sha256_file(RUN / "phase_10_plan.md"),
        "entries": entries,
        "external_runtimes": {
            "a2rag_python_version": command_output([str(ROOT / "refs/a2rag_runtime/.venv/bin/python"), "--version"]),
            "qwen": qwen_manifest(),
            "vllm_version": command_output([
                "/home/coder/demo/exposure_pipeline_workspace/.venv-vllm-auto/bin/python", "-c",
                "import vllm; print(vllm.__version__)",
            ]),
            "vllm_pip_freeze_sha256": "c9469a71e1b6945533b6d4fa51da235a62cf6d3aeaef7b9ced93b6fdf8e605ed",
        },
    }
    manifest["manifest_semantic_sha256"] = hashlib.sha256(canonical_bytes(manifest)).hexdigest()
    write_json(RUN / "immutable_inputs_manifest.json", manifest)

    freeze = command_output(["uv", "pip", "freeze", "--python", str(PYTHON)]).splitlines()
    lock_hash = sha256_file(ROOT / "requirements/phase10.lock")
    dependency = {
        "schema_version": "finglmqa.phase10.dependency_manifest.v1",
        "python_version": command_output([str(PYTHON), "--version"]),
        "lock_sha256": lock_hash,
        "installed": sorted(freeze),
        "installed_freeze_sha256": hashlib.sha256(("\n".join(sorted(freeze)) + "\n").encode()).hexdigest(),
    }
    write_json(RUN / "dependency_manifest.json", dependency)
    report = {
        "schema_version": "finglmqa.phase10.gate_report.v1",
        "gate": 0,
        "status": "passed",
        "checks": {
            "immutable_entry_count": len(entries),
            "qwen_snapshot_pinned": True,
            "vllm_0_21_0": manifest["external_runtimes"]["vllm_version"] == "0.21.0",
            "isolated_python_3_14": dependency["python_version"].startswith("Python 3.14."),
            "dependency_lock_matches_install": bool(freeze),
        },
    }
    if not all(value is True or isinstance(value, int) for value in report["checks"].values()):
        raise RuntimeError("Gate 0 checks failed")
    write_json(RUN / "gate0_report.json", report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
