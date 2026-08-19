#!/usr/bin/env python3
"""Run the frozen Type 3 benchmark through the deterministic Phase 8 pipeline.

This experiment deliberately instantiates ``EvidenceExecutor`` without a
GeneratorPort.  A2RAG/BGE-M3 may retrieve evidence, but no generative LLM is
started or called.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from finglmqa.contracts import canonical_json_bytes  # noqa: E402
from finglmqa.pipeline import build_default_pipeline  # noqa: E402


DEFAULT_ORACLE = ROOT / "runs/phase_08/benchmark_decomposition_oracle.jsonl"
DEFAULT_OUT_DIR = ROOT / "runs/type3_no_llm_experiment"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    atomic_write(path, canonical_json_bytes(dict(value)))


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    atomic_write(path, b"".join(canonical_json_bytes(dict(row)) for row in rows))


def request(case_id: str, question: str) -> dict[str, Any]:
    return {
        "schema_version": "finglmqa.phase8.qa_request.v1",
        "request_id": case_id.replace(":", "_"),
        "question": question,
        "locale": "zh-CN",
        "trace_delivery": "reference",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", type=Path, default=DEFAULT_ORACLE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model-cache", type=Path, default=Path("/home/coder/demo/models"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.oracle = args.oracle.resolve()
    args.out_dir = args.out_dir.resolve()
    args.model_cache = args.model_cache.resolve()
    oracle = [
        row for row in read_jsonl(args.oracle)
        if row["source"]["benchmark_type"] == "3-1"
    ]
    oracle.sort(key=lambda row: row["source"]["selected_ordinal"])
    if len(oracle) != 260:
        raise RuntimeError(f"Expected 260 Type 3 rows, got {len(oracle)}")

    pipeline, transport = build_default_pipeline(
        evidence_enabled=True,
        device=args.device,
        model_cache=args.model_cache,
    )
    if transport is None:
        raise RuntimeError("Evidence transport was not created")

    rows: list[dict[str, Any]] = []
    statuses: Counter[str] = Counter()
    generator_modes: Counter[str] = Counter()
    trace_dir = args.out_dir / "traces"
    try:
        transport.ping()
        for ordinal, gold in enumerate(oracle, start=1):
            case_id = gold["case_id"]
            payload = request(case_id, gold["source"]["question"])
            run = pipeline.run(payload)
            answer = run.answer
            modes = sorted({
                row["trace"].get("generator_mode")
                for row in answer["subplans"]
                if row["backend"] == "evidence" and row.get("trace", {}).get("generator_mode")
            })
            for mode in modes:
                generator_modes[mode] += 1
            if any(not mode.startswith("deterministic_") for mode in modes):
                raise RuntimeError(f"Generative evidence mode detected: {case_id} {modes}")
            trace_hash = run.trace["trace_hash"]
            write_json(trace_dir / f"{trace_hash}.json", run.trace)
            response = {
                "answer": answer["answer_text"],
                "citations": answer["citations"],
                "status": answer["status"],
                "errors": answer["errors"],
                "warnings": answer["warnings"],
                "trace_hash": trace_hash,
                "generator_modes": modes,
            }
            statuses[answer["status"]] += 1
            rows.append({
                "case_id": case_id,
                "kind": "benchmark",
                "oracle_match": True,
                "request": payload,
                "response": response,
            })
            if ordinal % 25 == 0 or ordinal == len(oracle):
                print(f"completed={ordinal}/{len(oracle)}", flush=True)
    finally:
        transport.close()

    output_path = args.out_dir / "http_evaluation.jsonl"
    write_jsonl(output_path, rows)
    report = {
        "schema_version": "finglmqa.type3_no_llm_experiment.v1",
        "rows": len(rows),
        "unique_questions": len({row["request"]["question"] for row in rows}),
        "nonempty_answers": sum(bool(row["response"]["answer"].strip()) for row in rows),
        "status_counts": dict(sorted(statuses.items())),
        "generator_mode_counts": dict(sorted(generator_modes.items())),
        "generative_llm_used": False,
        "device": args.device,
        "inputs": {
            "oracle_sha256": sha256_file(args.oracle),
            "evidence_executor_sha256": sha256_file(ROOT / "src/finglmqa/evidence_executor.py"),
            "retriever_sha256": sha256_file(ROOT / "scripts/query_type3_evidence.py"),
            "pipeline_sha256": sha256_file(ROOT / "src/finglmqa/pipeline.py"),
        },
        "artifacts": {
            "answers": output_path.relative_to(ROOT).as_posix(),
            "answers_sha256": sha256_file(output_path),
            "traces": trace_dir.relative_to(ROOT).as_posix(),
        },
    }
    write_json(args.out_dir / "run_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
