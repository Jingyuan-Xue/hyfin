#!/usr/bin/env python3
"""Record deterministic path/hash/mtime manifests for Phase 9 wave barriers."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
WAVES = ROOT / "runs/phase_09/waves"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def records(paths: Iterable[str]) -> list[dict[str, Any]]:
    rows = []
    for label in sorted(set(paths)):
        path = ROOT / label
        if not path.is_file():
            continue
        stat = path.stat()
        rows.append({"path": label, "sha256": sha(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    return rows


def write(name: str, wave: str, boundary: str, paths: Iterable[str]) -> list[dict[str, Any]]:
    rows = records(paths)
    value = {
        "schema_version": "finglmqa.phase9.wave_manifest.v1",
        "wave": wave, "boundary": boundary, "entries": rows,
    }
    path = WAVES / name
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return rows


def main() -> int:
    w0_before = ["runs/phase_09/phase_09_plan.md"]
    w0 = [
        *w0_before, "runs/phase_09/immutable_inputs_manifest.json", "runs/phase_09/dependency_manifest.json",
        "runs/phase_09/gate0_report.json", "runs/phase_09/gate1_report.json",
        "runs/phase_09/reports/supplement_requests.jsonl", "runs/phase_09/reports/request_universe_summary.json",
        "src/finglmqa/supplement_contracts.py", "scripts/build_supplement_requests.py", "scripts/prepare_phase_09_gate0.py",
        "data/schemas/supplemental_facts.schema.json", "data/schemas/supplement_decision.schema.json",
        "data/schemas/supplement_lookup_result.schema.json",
    ]
    w1 = [
        *w0, "src/finglmqa/tabgr_adapter.py", "src/finglmqa/supplement_validation.py",
        "tests/test_phase09_contracts.py", "tests/test_phase09_tabgr_adapter.py", "tests/test_phase09_validation.py",
        "runs/phase_09/gate2_report.json", "runs/phase_09/gate3_report.json",
    ]
    w2 = [
        *w1, "src/finglmqa/supplement_store.py", "scripts/build_supplemental_facts.py",
        "data/facts/supplemental_facts.jsonl", "data/facts/supplemental_facts.duckdb",
        "runs/phase_09/reports/supplement_decisions.jsonl", "runs/phase_09/reports/supplement_trace.jsonl",
        "runs/phase_09/reports/supplement_summary.json", "runs/phase_09/gate4_report.json", "runs/phase_09/gate5_report.json",
        "runs/phase_09/repeatability/run2/supplemental_facts.jsonl",
        "runs/phase_09/repeatability/run2/supplement_decisions.jsonl",
        "runs/phase_09/repeatability/run2/supplement_trace.jsonl",
    ]
    w3 = [
        *w2, "src/finglmqa/composer.py", "runs/phase_09/gate6_report.json",
        "runs/phase_09/repeatability/phase8_real_1.json",
        "runs/phase_09/repeatability/phase8_real_2.json",
    ]
    w4 = [
        *w3, "scripts/validate_phase_09_gates.py",
        *[f"scripts/validate_phase_09_gate{index}.py" for index in range(8)],
        "scripts/record_phase_09_waves.py", "runs/phase_09/gate7_report.json",
        "runs/phase_09/reports/qingdao_port_case_report.json", "runs/phase_09/phase_09_report.md",
        "README.md", "docs/DECISIONS.md", "docs/ISSUES.md", "docs/PROGRESS.md", "docs/ARTIFACTS_MANIFEST.md",
    ]
    write("w0_before.json", "w0", "before", w0_before)
    write("w0_after.json", "w0", "after", w0)
    write("w1_before.json", "w1", "before", w0)
    write("w1_after.json", "w1", "after", w1)
    write("w2_before.json", "w2", "before", w1)
    write("w2_after.json", "w2", "after", w2)
    write("w3_before.json", "w3", "before", w2)
    write("w3_after.json", "w3", "after", w3)
    write("w4_before.json", "w4", "before", w3)
    rows = write("w4_after.json", "w4", "after", w4)
    print(json.dumps({"wave": "w4", "files": len(rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
