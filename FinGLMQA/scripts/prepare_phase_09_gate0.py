#!/usr/bin/env python3
"""Record and validate Phase 9 immutable inputs and the one dependency."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/phase_09"
TABGR = ROOT / "refs/tabgr_runtime/build_graphs/graph_to_text_triple_full.py"

PINS = {
    "data/facts/financial_facts.duckdb": "b3e8fed65ddc1ccd5954083a4df64f3eab2150294cae08a11424f3bc5744f278",
    "data/facts/financial_facts.jsonl": "abeb4b3b221aac74705b84c80469c03b23fd8638d67004c75dd7a512c6841405",
    "data/indexes/canonical_metric_candidates.jsonl": "8371c8a2e9f62d5dfd09d0fd6d14bfe14a1ca85d0273d9e0e2a02430952e5099",
    "data/corpus_package/tabgr_table_corpus.jsonl": "a6190b8c8e2f8bafe0f1ae7e0d5a7dcb7ca6de6c3acc4c3c41b7775d4336e369",
    "data/corpus_package/table_cells.jsonl": "41c14754ac8875498550b7986ff7a2ba5d61f3ea3839b11f4a7e249d6d1bf6a4",
    "data/indexes/tabgr_table_index.jsonl": "ee4a58887a535e1ce8427f6fb3a9a5f15e20bbe2cb927f3256f674aae4367291",
    "data/corpus_package/company_year_index.jsonl": "fb605221d096159435f24cdc8651e4679b039667b2dd7826290bd657ab6b7b00",
    "runs/phase_06/reports/candidate_decisions.jsonl": "588b321a612297ebe5fc5dfe4548d8c502a99653ec958839d1b46e9539282077",
    "runs/phase_06/reports/conflict_groups.jsonl": "8e1c6b56921bb98c952519351411c1541cef7df8d9e6c5401096d70520b814ad",
    "src/config/metric_aliases.json": "12f786608da8508741df11975070bf631eebcb919083a11a1ee0de843ba15ddc",
    "src/config/unit_rules.json": "ef948c839cc4ae040ddeb5a6b3c1f5ffdd8a167eb2f251dc4f1d80511ff52cfb",
    "external:tabgr_runtime/build_graphs/graph_to_text_triple_full.py": "7d193807d5f74b3281c8bd52c0d6da76f1f149cd5e92c4c82b47de4b8708d316",
}

RELEASE_INPUTS = (
    "data/corpus_package/evidence_chunks.jsonl",
    "data/indexes/a2rag_index/document_chunk_map.jsonl",
    "runs/phase_07/phase_07_report.md",
    "runs/phase_08/gate2_report.json",
    "runs/phase_08/gates_6_8_report.json",
    "runs/phase_08/pattern_registry_manifest.json",
    "src/config/composition_patterns.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def entry(label: str, path: Path, expected: str | None = None) -> dict[str, Any]:
    stat = path.stat()
    actual = sha256_file(path)
    return {
        "path": label,
        "sha256": actual,
        "expected_sha256": expected,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "matches_pin": expected is None or actual == expected,
    }


def canonical_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def dependency_manifest() -> dict[str, Any]:
    distribution = importlib.metadata.distribution("networkx")
    if distribution.version != "3.5":
        raise RuntimeError(f"networkx pin mismatch: {distribution.version}")
    rows: list[dict[str, Any]] = []
    for relative in sorted(distribution.files or (), key=str):
        path = Path(distribution.locate_file(relative))
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        rows.append({"path": str(relative), "sha256": sha256_file(path), "size": path.stat().st_size})
    tree_payload = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "schema_version": "finglmqa.phase9.dependency_manifest.v1",
        "install_command": "uv pip install --python .venv/bin/python networkx==3.5",
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "dependencies": [{
            "name": "networkx",
            "version": distribution.version,
            "file_count": len(rows),
            "installed_tree_sha256": hashlib.sha256(tree_payload).hexdigest(),
        }],
    }


def import_tabgr() -> None:
    spec = importlib.util.spec_from_file_location("finglmqa_phase9_tabgr_probe", TABGR)
    if spec is None or spec.loader is None:
        raise RuntimeError("TabGR module spec unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in (
        "_parse_triples", "_build_index", "_build_neighbors", "_build_personalization",
        "_run_ppr", "grouped_string_with_cell_merges_w",
    ):
        if not callable(getattr(module, name, None)):
            raise RuntimeError(f"TabGR callable missing: {name}")


def main() -> int:
    entries: list[dict[str, Any]] = []
    for label, expected in PINS.items():
        path = TABGR if label.startswith("external:") else ROOT / label
        entries.append(entry(label, path, expected))
    if not all(row["matches_pin"] for row in entries):
        raise RuntimeError("one or more immutable input hash pins do not match")
    release = [entry(label, ROOT / label) for label in RELEASE_INPUTS]
    manifest = {
        "schema_version": "finglmqa.phase9.immutable_inputs_manifest.v1",
        "plan_sha256": sha256_file(RUN / "phase_09_plan.md"),
        "entries": entries,
        "phase7_phase8_release_inputs": release,
    }
    canonical_write(RUN / "immutable_inputs_manifest.json", manifest)
    canonical_write(RUN / "dependency_manifest.json", dependency_manifest())
    import_tabgr()
    report = {
        "schema_version": "finglmqa.phase9.gate_report.v1",
        "gate": 0,
        "status": "passed",
        "checks": {
            "immutable_pin_count": len(entries),
            "immutable_pins_match": True,
            "networkx_exact_version": True,
            "tabgr_import_succeeds": True,
            "phase7_phase8_release_hash_count": len(release),
        },
    }
    canonical_write(RUN / "gate0_report.json", report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
