#!/usr/bin/env python3
"""Run evidence-bounded local Qwen organization over frozen Type 3 v8 output.

The answer chain projects only case identity, question, authorized v8 answer,
citations, and numeric authorizations.  Benchmark prompts, prompt answers,
keywords, and reference answers are neither opened nor accepted as arguments.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from finglmqa.contracts import canonical_json_bytes, semantic_sha256  # noqa: E402
from finglmqa.qwen_shadow import VLLMShadowServer  # noqa: E402
from finglmqa.type3_qwen36_organizer_v8 import (  # noqa: E402
    MAX_SELECTED_SEGMENTS,
    OpenAICompatibleClient,
    PROMPT_CONTRACT_HASH,
    PROMPT_VERSION,
    RESULT_SCHEMA,
    STRUCTURED_OUTPUT_VERSION,
    Type3Qwen36OrganizerV8,
)


DEFAULT_V8_DIR = ROOT / "runs/type3_no_llm_experiment_v8"
DEFAULT_OUTPUT_ROOT = ROOT / "runs/type3_qwen36_organization_v8"
DEFAULT_MODEL = ROOT / "refs/qwen_model"
DEFAULT_VLLM_BIN = Path(
    "/home/coder/demo/exposure_pipeline_workspace/.venv-vllm-auto/bin/vllm"
)
FORBIDDEN_BENCHMARK_FIELDS = frozenset({
    "prompt", "prompt_answer", "prom_answer", "key_word", "keyword",
    "reference", "references", "reference_answer", "gold", "answer_key",
})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise RuntimeError(f"expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    atomic_write(path, canonical_json_bytes(value))


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    atomic_write(path, b"".join(canonical_json_bytes(dict(row)) for row in rows))


def _forbidden_keys(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower() in FORBIDDEN_BENCHMARK_FIELDS:
                found.add(str(key))
            found.update(_forbidden_keys(child))
    elif isinstance(value, list):
        for child in value:
            found.update(_forbidden_keys(child))
    return found


def load_authorized_inputs(v8_dir: Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    report_path = v8_dir / "run_report.json"
    answers_path = v8_dir / "http_evaluation.jsonl"
    traces_path = v8_dir / "deterministic_traces.jsonl"
    report = read_json(report_path)
    expected_answers = report.get("stages", {}).get("full", {}).get("answers_sha256")
    expected_traces = report.get("stages", {}).get("full", {}).get("traces_sha256")
    actual_answers = sha256_file(answers_path)
    actual_traces = sha256_file(traces_path)
    if actual_answers != expected_answers or actual_traces != expected_traces:
        raise RuntimeError("frozen v8 source hashes differ from run_report")

    answer_rows = read_jsonl(answers_path)
    trace_rows = read_jsonl(traces_path)
    if len(answer_rows) != 260 or len(trace_rows) != 260:
        raise RuntimeError("frozen v8 sources must each contain 260 rows")
    if any(_forbidden_keys(row) for row in answer_rows + trace_rows):
        raise RuntimeError("a forbidden benchmark field appeared in v8 sources")

    traces_by_case: dict[str, list[dict[str, Any]]] = {}
    for row in trace_rows:
        case_id = row.get("case_id")
        authorizations = row.get("numeric_authorizations")
        if not isinstance(case_id, str) or not isinstance(authorizations, list):
            raise RuntimeError("v8 numeric authorization projection is invalid")
        if case_id in traces_by_case:
            raise RuntimeError("duplicate v8 trace case_id")
        # No other trace field crosses the answer boundary.
        traces_by_case[case_id] = authorizations

    projected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in answer_rows:
        case_id = row.get("case_id")
        request = row.get("request")
        response = row.get("response")
        if (
            not isinstance(case_id, str)
            or case_id in seen
            or case_id not in traces_by_case
            or not isinstance(request, Mapping)
            or not isinstance(response, Mapping)
            or not isinstance(request.get("question"), str)
            or not isinstance(response.get("answer"), str)
            or not isinstance(response.get("citations"), list)
        ):
            raise RuntimeError("v8 authorized answer projection is invalid")
        seen.add(case_id)
        projected.append({
            "case_id": case_id,
            "question": request["question"],
            "authorized_answer": response["answer"],
            "citations": response["citations"],
            "numeric_authorizations": traces_by_case[case_id],
        })
    if seen != set(traces_by_case):
        raise RuntimeError("v8 answer/authorization case sets differ")
    hashes = {
        "v8_http_evaluation_sha256": actual_answers,
        "v8_deterministic_traces_sha256": actual_traces,
        "v8_run_report_sha256": sha256_file(report_path),
    }
    return projected, hashes


def _vllm_version(binary: Path) -> str:
    try:
        completed = subprocess.run(
            [str(binary), "--version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return completed.stdout.strip() or completed.stderr.strip()
    except Exception:
        return "unavailable"


def model_identity(model_path: Path, vllm_binary: Path) -> dict[str, Any]:
    resolved = model_path.resolve()
    config_path = resolved / "config.json"
    tokenizer_path = resolved / "tokenizer_config.json"
    readme_path = resolved / "README.md"
    config = read_json(config_path)
    title = None
    for line in readme_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# "):
            title = line[2:].strip()
            break
    return {
        "local_requested_path": model_path.relative_to(ROOT).as_posix(),
        "local_snapshot_path": resolved.as_posix(),
        "snapshot_revision": resolved.name,
        "readme_model_title": title,
        "config_model_type": config.get("model_type"),
        "config_architectures": config.get("architectures"),
        "config_sha256": sha256_file(config_path),
        "tokenizer_config_sha256": sha256_file(tokenizer_path),
        "backend": "vLLM OpenAI-compatible local server",
        "vllm_version": _vllm_version(vllm_binary),
        "language_model_only": True,
        "dtype": "bfloat16",
    }


def freeze_manifest(
    *,
    source_hashes: Mapping[str, str],
    model: str,
    model_path: Path,
    vllm_binary: Path,
) -> dict[str, Any]:
    generation_config = {
        "served_model_name": model,
        "temperature": 0,
        "top_p": 1,
        "seed": 0,
        "max_tokens": 256,
        "enable_thinking": False,
        "max_selected_segments": MAX_SELECTED_SEGMENTS,
        "structured_output_version": STRUCTURED_OUTPUT_VERSION,
    }
    manifest: dict[str, Any] = {
        "schema_version": "finglmqa.experimental.type3_qwen36_v8.freeze.v1",
        "frozen_before_model_invocation": True,
        "frozen_before_scoring": True,
        "prompt_version": PROMPT_VERSION,
        "prompt_contract_sha256": PROMPT_CONTRACT_HASH,
        "generation_config": generation_config,
        "generation_config_sha256": semantic_sha256(generation_config),
        "source_hashes": dict(sorted(source_hashes.items())),
        "code_hashes": {
            "organizer_sha256": sha256_file(ROOT / "src/finglmqa/type3_qwen36_organizer_v8.py"),
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "tests_sha256": sha256_file(ROOT / "tests/test_type3_qwen36_organizer_v8.py"),
        },
        "model_identity": model_identity(model_path, vllm_binary),
        "answer_chain_consumed_fields": [
            "case_id (join identity only)",
            "question",
            "v8 authorized answer",
            "v8 citations",
            "v8 numeric_authorizations",
        ],
        "forbidden_benchmark_fields_consumed": [],
        "case_company_year_rules": False,
        "manifest_fingerprint": "",
    }
    manifest["manifest_fingerprint"] = semantic_sha256({
        key: value for key, value in manifest.items() if key != "manifest_fingerprint"
    })
    return manifest


def http_projection(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "case_id": result["case_id"],
        "kind": "benchmark",
        "oracle_match": True,
        "request": {"question": result["question"]},
        "response": {
            "answer": result["answer"],
            "citations": result["citations"],
            "status": result["status"],
            "errors": (
                [{"failure_code": result["generator_outcome"]}]
                if result["status"] == "error" else []
            ),
            "warnings": [],
            "generator_modes": [PROMPT_VERSION, "authorized_segment_selection"],
        },
        "experimental_profile": PROMPT_VERSION,
    }


def run_cases(
    cases: Sequence[Mapping[str, Any]],
    *,
    organizer: Type3Qwen36OrganizerV8,
    repeat_count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    results: list[dict[str, Any]] = []
    for ordinal, case in enumerate(cases, 1):
        result = organizer.organize(**case)
        results.append(result)
        print(
            f"[{ordinal}/{len(cases)}] {case['case_id']} {result['generator_outcome']}",
            file=sys.stderr,
            flush=True,
        )
    repeats: list[dict[str, Any]] = []
    for ordinal, case in enumerate(cases[:repeat_count], 1):
        result = organizer.organize(**case)
        repeats.append(result)
        print(
            f"[repeat {ordinal}/{repeat_count}] {case['case_id']} {result['generator_outcome']}",
            file=sys.stderr,
            flush=True,
        )
    return results, repeats


def safety_validation(
    cases: Sequence[Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
    repeats: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_case = {row["case_id"]: row for row in cases}
    citation_exact = all(
        row["citations"] in ([], by_case[row["case_id"]]["citations"])
        for row in results
    )
    repeat_exact = all(
        repeated == results[index]
        for index, repeated in enumerate(repeats)
    )
    gates = [row["gate_report"] for row in results]
    return {
        "schema_version": "finglmqa.experimental.type3_qwen36_v8.safety.v1",
        "rows": len(results),
        "all_rows_terminal": len(results) == len(cases),
        "citation_projection_is_empty_or_exact_v8_projection": citation_exact,
        "all_output_citation_scopes_passed": all(row["citation_scope_passed"] for row in gates),
        "all_output_text_supported_by_authorized_v8_answer": all(
            row["authorized_answer_support_passed"] for row in gates
        ),
        "all_output_numbers_authorized": all(row["numeric_authorization_passed"] for row in gates),
        "any_model_text_accepted": any(row["model_text_accepted"] for row in gates),
        "repeat_count": len(repeats),
        "repeat_results_exact": repeat_exact,
        "rejected_selection_count": sum(len(row["rejected_selections"]) for row in results),
        "passed": all((
            len(results) == len(cases),
            citation_exact,
            all(row["citation_scope_passed"] for row in gates),
            all(row["authorized_answer_support_passed"] for row in gates),
            all(row["numeric_authorization_passed"] for row in gates),
            not any(row["model_text_accepted"] for row in gates),
            repeat_exact,
        )),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v8-dir", type=Path, default=DEFAULT_V8_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--model", default="finglmqa-qwen3.6-27b")
    parser.add_argument("--vllm-bin", type=Path, default=DEFAULT_VLLM_BIN)
    parser.add_argument("--base-url", help="reuse an already-running local OpenAI-compatible server")
    parser.add_argument("--limit", type=int, help="deterministic prefix smoke size; omit for all 260")
    parser.add_argument("--repeat-count", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if not output_dir.is_relative_to(DEFAULT_OUTPUT_ROOT.resolve()):
        raise RuntimeError("output-dir must remain under runs/type3_qwen36_organization_v8")
    if args.limit is not None and not 1 <= args.limit <= 260:
        raise RuntimeError("limit must be between 1 and 260")
    if not 0 <= args.repeat_count <= 260:
        raise RuntimeError("repeat-count must be between 0 and 260")

    cases, source_hashes = load_authorized_inputs(args.v8_dir.resolve())
    if args.limit is not None:
        cases = cases[: args.limit]
    repeat_count = min(args.repeat_count, len(cases))

    # This is the first write and occurs before any inference server starts.
    frozen = freeze_manifest(
        source_hashes=source_hashes,
        model=args.model,
        model_path=args.model_path,
        vllm_binary=args.vllm_bin,
    )
    write_json(output_dir / "freeze_manifest.json", frozen)

    server: VLLMShadowServer | None = None
    try:
        if args.base_url:
            base_url = args.base_url
            served_model = args.model
        else:
            os.environ["FINGLMQA_QWEN_MODEL"] = args.model_path.resolve().as_posix()
            os.environ["FINGLMQA_QWEN_SERVED_NAME"] = args.model
            os.environ["FINGLMQA_VLLM_BIN"] = args.vllm_bin.resolve().as_posix()
            server = VLLMShadowServer()
            server.start()
            base_url = server.base_url
            served_model = server.served_name
        organizer = Type3Qwen36OrganizerV8(
            OpenAICompatibleClient(base_url),
            model=served_model,
        )
        results, repeats = run_cases(
            cases,
            organizer=organizer,
            repeat_count=repeat_count,
        )
    finally:
        if server is not None:
            server.stop()

    validation = safety_validation(cases, results, repeats)
    write_jsonl(output_dir / "results.jsonl", results)
    write_jsonl(output_dir / "repeat_results.jsonl", repeats)
    write_jsonl(output_dir / "http_evaluation.jsonl", map(http_projection, results))
    write_json(output_dir / "safety_validation.json", validation)

    statuses = Counter(row["status"] for row in results)
    outcomes = Counter(row["generator_outcome"] for row in results)
    report = {
        "schema_version": "finglmqa.experimental.type3_qwen36_v8.run_report.v1",
        "profile_version": PROMPT_VERSION,
        "result_schema_version": RESULT_SCHEMA,
        "model": frozen["model_identity"],
        "backend_status": "completed",
        "input_rows": len(cases),
        "terminal_rows": len(results),
        "nonempty_answers": sum(bool(row["answer"]) for row in results),
        "coverage_rate": round(sum(bool(row["answer"]) for row in results) / len(results), 8),
        "status_counts": dict(sorted(statuses.items())),
        "generator_outcome_counts": dict(sorted(outcomes.items())),
        "safety_validation_passed": validation["passed"],
        "repeat_results_exact": validation["repeat_results_exact"],
        "repeat_count": len(repeats),
        "freeze_manifest_fingerprint": frozen["manifest_fingerprint"],
        "artifacts": {
            "results_sha256": sha256_file(output_dir / "results.jsonl"),
            "repeat_results_sha256": sha256_file(output_dir / "repeat_results.jsonl"),
            "http_evaluation_sha256": sha256_file(output_dir / "http_evaluation.jsonl"),
            "safety_validation_sha256": sha256_file(output_dir / "safety_validation.json"),
        },
        "benchmark_fields_loaded_by_answer_chain": ["case_id", "question"],
        "forbidden_benchmark_fields_loaded_by_answer_chain": [],
        "benchmark_scoring_used_for_prompt_or_rule_selection": False,
    }
    write_json(output_dir / "run_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if validation["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
