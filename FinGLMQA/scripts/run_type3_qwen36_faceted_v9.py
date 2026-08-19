#!/usr/bin/env python3
"""Run the general document-scoped Qwen faceted evidence v9 experiment.

Use the pinned A2RAG environment so the existing BGE-M3 query encoder and
read-only dense index are available:

    refs/a2rag_runtime/.venv/bin/python scripts/run_type3_qwen36_faceted_v9.py

The answer chain reads only questions, resolved document identities, frozen v8
answers/citations, immutable document evidence and model-selected IDs.  Gold
answers, prompt answers, scoring keywords and benchmark references are never
loaded by this process.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence

import httpx


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from finglmqa.contracts import canonical_json_bytes, semantic_sha256  # noqa: E402
from finglmqa.type3_qwen36_faceted_v9 import (  # noqa: E402
    MAX_CANDIDATES,
    MAX_FACETS,
    MAX_SELECTED_FRAGMENTS,
    PROFILE_VERSION,
    PROMPT_CONTRACT_HASH,
    PROMPT_VERSION,
    RESULT_SCHEMA,
    SELECTOR_SEEDS,
    Type3Qwen36FacetedV9,
)
from query_type3_evidence import Type3EvidenceRetriever  # noqa: E402


DEFAULT_V8_DIR = ROOT / "runs/type3_no_llm_experiment_v8"
DEFAULT_OUTPUT_DIR = ROOT / "runs/type3_qwen36_faceted_v9/full"
DEFAULT_OUTPUT_ROOT = ROOT / "runs/type3_qwen36_faceted_v9"
DEFAULT_MODEL_PATH = ROOT / "refs/qwen_model"
DEFAULT_VLLM_BIN = Path(
    "/home/coder/demo/exposure_pipeline_workspace/.venv-vllm-auto/bin/vllm"
)
DEFAULT_MODEL_NAME = "finglmqa-qwen3.6-27b-v9"
FORBIDDEN_FIELDS = frozenset({
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
                raise RuntimeError(f"expected JSON object: {path}:{line_number}")
            rows.append(value)
    return rows


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
    atomic_write(path, canonical_json_bytes(value))


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    atomic_write(path, b"".join(canonical_json_bytes(dict(row)) for row in rows))


def forbidden_keys(value: Any) -> set[str]:
    result: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower() in FORBIDDEN_FIELDS:
                result.add(str(key))
            result.update(forbidden_keys(child))
    elif isinstance(value, list):
        for child in value:
            result.update(forbidden_keys(child))
    return result


def load_cases(v8_dir: Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    report_path = v8_dir / "run_report.json"
    answers_path = v8_dir / "http_evaluation.jsonl"
    traces_path = v8_dir / "deterministic_traces.jsonl"
    report = read_json(report_path)
    actual_answers = sha256_file(answers_path)
    actual_traces = sha256_file(traces_path)
    expected = report.get("stages", {}).get("full", {})
    if (
        actual_answers != expected.get("answers_sha256")
        or actual_traces != expected.get("traces_sha256")
    ):
        raise RuntimeError("frozen v8 source hashes differ from run_report")
    answers = read_jsonl(answers_path)
    traces = read_jsonl(traces_path)
    if len(answers) != 260 or len(traces) != 260:
        raise RuntimeError("frozen v8 sources must contain exactly 260 rows")
    if any(forbidden_keys(row) for row in (*answers, *traces)):
        raise RuntimeError("forbidden benchmark annotations appeared in answer inputs")
    trace_by_case = {str(row["case_id"]): row for row in traces}
    if len(trace_by_case) != len(traces):
        raise RuntimeError("duplicate case_id in v8 traces")

    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in answers:
        case_id = str(row.get("case_id") or "")
        request = row.get("request")
        response = row.get("response")
        trace = trace_by_case.get(case_id)
        if (
            not case_id
            or case_id in seen
            or not isinstance(request, Mapping)
            or not isinstance(response, Mapping)
            or not isinstance(trace, Mapping)
            or not isinstance(request.get("question"), str)
            or not isinstance(response.get("answer"), str)
            or not isinstance(response.get("citations"), list)
            or not isinstance(trace.get("document_id"), str)
        ):
            raise RuntimeError(f"invalid v8 projection: {case_id!r}")
        seen.add(case_id)
        cases.append({
            "case_id": case_id,
            "question": request["question"],
            "document_id": trace["document_id"],
            "baseline_answer": response["answer"],
            "baseline_citations": response["citations"],
        })
    if seen != set(trace_by_case):
        raise RuntimeError("v8 answer/trace case sets differ")
    return cases, {
        "v8_http_evaluation_sha256": actual_answers,
        "v8_deterministic_traces_sha256": actual_traces,
        "v8_run_report_sha256": sha256_file(report_path),
    }


class OpenAICompatibleClient:
    def __init__(self, base_url: str, *, timeout_seconds: float = 300.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def complete(self, body: Mapping[str, Any]) -> Mapping[str, Any]:
        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(
                f"{self.base_url}/chat/completions", json=dict(body)
            )
            response.raise_for_status()
            value = response.json()
        if not isinstance(value, Mapping):
            raise RuntimeError("chat completion envelope is not an object")
        return value


class VLLMV9Server:
    """Own the isolated local vLLM process and always stop its process group."""

    def __init__(
        self,
        *,
        binary: Path,
        model_path: Path,
        served_name: str,
        host: str = "127.0.0.1",
        port: int = 8012,
    ) -> None:
        self.binary = binary.resolve()
        self.model_path = model_path.resolve()
        self.served_name = served_name
        self.host = host
        self.port = port
        self.process: subprocess.Popen[bytes] | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    def healthy(self) -> bool:
        try:
            return httpx.get(f"{self.base_url}/models", timeout=2).status_code == 200
        except Exception:
            return False

    def start(self, timeout_seconds: float = 600.0) -> None:
        if self.process is not None or self.healthy():
            raise RuntimeError("v9 Qwen port is already occupied")
        if not self.binary.is_file() or not self.model_path.exists():
            raise RuntimeError("pinned vLLM binary or Qwen model is missing")
        command = [
            str(self.binary), "serve", str(self.model_path),
            "--host", self.host,
            "--port", str(self.port),
            "--served-model-name", self.served_name,
            "--dtype", "bfloat16",
            "--max-model-len", "24576",
            "--gpu-memory-utilization", "0.82",
            "--max-num-seqs", "3",
            "--seed", "0",
            "--generation-config", "vllm",
            "--language-model-only",
            "--no-trust-remote-code",
            "--no-enable-log-requests",
            "--disable-uvicorn-access-log",
            "--disable-log-stats",
            "--uvicorn-log-level", "warning",
        ]
        env = {
            **os.environ,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "VLLM_NO_USAGE_STATS": "1",
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
        }
        self.process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                self.stop()
                raise RuntimeError("v9 Qwen vLLM exited during startup")
            if self.healthy():
                return
            time.sleep(1)
        self.stop()
        raise TimeoutError("v9 Qwen vLLM startup timed out")

    def stop(self) -> None:
        process = self.process
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=10)
        self.process = None


def vllm_version(binary: Path) -> str:
    try:
        result = subprocess.run(
            [str(binary), "--version"], capture_output=True, text=True,
            check=True, timeout=30,
        )
        return (result.stdout or result.stderr).strip()
    except Exception:
        return "unavailable"


def freeze_manifest(
    *,
    source_hashes: Mapping[str, str],
    model_path: Path,
    model_name: str,
    vllm_binary: Path,
) -> dict[str, Any]:
    config_path = model_path.resolve() / "config.json"
    generation = {
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "planner_seed": 0,
        "selector_seeds": list(SELECTOR_SEEDS),
        "max_tokens": 1024,
        "enable_thinking": True,
        "structured_outputs": True,
        "max_facets": MAX_FACETS,
        "max_candidates": MAX_CANDIDATES,
        "max_selected_fragments": MAX_SELECTED_FRAGMENTS,
    }
    manifest = {
        "schema_version": "finglmqa.experimental.type3_qwen36_faceted_v9.freeze.v1",
        "frozen_before_model_invocation": True,
        "frozen_before_scoring": True,
        "profile_version": PROFILE_VERSION,
        "prompt_version": PROMPT_VERSION,
        "prompt_contract_sha256": PROMPT_CONTRACT_HASH,
        "generation_config": generation,
        "generation_config_sha256": semantic_sha256(generation),
        "source_hashes": dict(sorted(source_hashes.items())),
        "index_hashes": {
            "evidence_chunks_sha256": sha256_file(
                ROOT / "data/corpus_package/evidence_chunks.jsonl"
            ),
            "document_map_sha256": sha256_file(
                ROOT / "data/indexes/a2rag_index/document_chunk_map.jsonl"
            ),
            "dense_index_manifest_sha256": sha256_file(
                ROOT / "data/indexes/a2rag_index/index_manifest.json"
            ),
        },
        "code_hashes": {
            "faceted_v9_sha256": sha256_file(
                ROOT / "src/finglmqa/type3_qwen36_faceted_v9.py"
            ),
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "retriever_sha256": sha256_file(
                ROOT / "scripts/query_type3_evidence.py"
            ),
        },
        "model_identity": {
            "served_name": model_name,
            "requested_path": model_path.relative_to(ROOT).as_posix(),
            "snapshot_path": model_path.resolve().as_posix(),
            "snapshot_revision": model_path.resolve().name,
            "config_sha256": sha256_file(config_path),
            "vllm_version": vllm_version(vllm_binary),
            "dtype": "bfloat16",
            "max_model_len": 24576,
        },
        "answer_chain_consumed_fields": [
            "case_id (join identity only)", "question", "resolved document_id",
            "v8 answer", "v8 citations", "document-scoped evidence",
        ],
        "forbidden_benchmark_fields_consumed": [],
        "case_company_year_rules": False,
        "manifest_fingerprint": "",
    }
    manifest["manifest_fingerprint"] = semantic_sha256({
        key: value for key, value in manifest.items() if key != "manifest_fingerprint"
    })
    return manifest


def projection(result: Mapping[str, Any], *, answer_field: str, profile: str) -> dict[str, Any]:
    answer = str(result[answer_field])
    return {
        "case_id": result["case_id"],
        "kind": "benchmark",
        "oracle_match": True,
        "request": {"question": result["question"]},
        "response": {
            "answer": answer,
            "citations": result["citations"],
            "status": "ok" if answer.strip() else "not_found",
            "errors": [],
            "warnings": [],
            "generator_modes": [profile, "document_scoped_extractive_ids_only"],
        },
        "experimental_profile": profile,
    }


def task_cache_key(case: Mapping[str, Any]) -> str:
    return semantic_sha256({
        "question": case["question"],
        "document_id": case["document_id"],
        "baseline_answer": case["baseline_answer"],
        "baseline_citations": case["baseline_citations"],
    })


def clone_for_case(result: Mapping[str, Any], case_id: str) -> dict[str, Any]:
    value = copy.deepcopy(dict(result))
    value["case_id"] = case_id
    value["result_fingerprint"] = semantic_sha256({
        "schema_version": value["schema_version"],
        "profile_version": value["profile_version"],
        "case_id": value["case_id"],
        "question": value["question"],
        "document_id": value["document_id"],
        "answer": value["answer"],
        "citations": value["citations"],
        "status": value["status"],
        "facets": value["facets"],
        "selected_fragment_ids": value["selected_fragment_ids"],
        "gate_report": value["gate_report"],
    })
    return value


def run_cases(
    cases: Sequence[Mapping[str, Any]],
    *,
    engine: Type3Qwen36FacetedV9,
) -> tuple[list[dict[str, Any]], int]:
    cache: dict[str, dict[str, Any]] = {}
    results: list[dict[str, Any]] = []
    for ordinal, case in enumerate(cases, 1):
        key = task_cache_key(case)
        if key in cache:
            result = clone_for_case(cache[key], str(case["case_id"]))
            cache_status = "cache"
        else:
            result = engine.answer(**case)
            cache[key] = copy.deepcopy(result)
            cache_status = "inference"
        results.append(result)
        print(
            f"[{ordinal}/{len(cases)}] {case['case_id']} {cache_status} "
            f"{result['planner_outcome']} {result['selector_outcome']} "
            f"selected={len(result['selected_fragment_ids'])}",
            file=sys.stderr,
            flush=True,
        )
    return results, len(cache)


def run_repeats(
    cases: Sequence[Mapping[str, Any]],
    *,
    engine: Type3Qwen36FacetedV9,
    repeat_count: int,
) -> list[dict[str, Any]]:
    repeats: list[dict[str, Any]] = []
    for ordinal, case in enumerate(cases[:repeat_count], 1):
        result = engine.answer(**case)
        repeats.append(result)
        print(
            f"[repeat {ordinal}/{repeat_count}] {case['case_id']} "
            f"{result['selector_outcome']}",
            file=sys.stderr,
            flush=True,
        )
    return repeats


def safety_validation(
    cases: Sequence[Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
    repeats: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_case = {str(row["case_id"]): row for row in cases}
    result_by_case = {str(row["case_id"]): row for row in results}
    repeat_exact = all(
        repeated.get("result_fingerprint")
        == (result_by_case.get(str(repeated["case_id"])) or {}).get("result_fingerprint")
        for repeated in repeats
    )
    cross_document = sum(
        1
        for result in results
        for citation in result.get("citations") or []
        if isinstance(citation, Mapping)
        and citation.get("document_id") not in (None, result["document_id"])
    )
    baseline_suffix_failures = sum(
        1
        for result in results
        if by_case[str(result["case_id"])]["baseline_answer"].strip()
        and not str(result["answer"]).endswith(
            by_case[str(result["case_id"])]["baseline_answer"].strip()
        )
    )
    failed_gates = sum(not bool(result["gate_report"]["passed"]) for result in results)
    model_text_accepted = sum(
        bool(result["gate_report"]["model_text_accepted"]) for result in results
    )
    unsupported_numbers = sum(
        not bool(result["gate_report"]["selected_numbers_source_supported"])
        for result in results
    )
    unsupported_text = sum(
        not bool(result["gate_report"]["selected_text_verbatim_supported"])
        for result in results
    )
    passed = all((
        len(results) == len(cases),
        all(str(result["answer"]).strip() for result in results),
        cross_document == 0,
        baseline_suffix_failures == 0,
        failed_gates == 0,
        model_text_accepted == 0,
        unsupported_numbers == 0,
        unsupported_text == 0,
        repeat_exact,
    ))
    return {
        "schema_version": "finglmqa.experimental.type3_qwen36_faceted_v9.safety.v1",
        "rows": len(results),
        "all_rows_terminal": len(results) == len(cases),
        "nonempty_answers": sum(bool(str(result["answer"]).strip()) for result in results),
        "cross_document_citation_count": cross_document,
        "baseline_suffix_failure_count": baseline_suffix_failures,
        "failed_gate_count": failed_gates,
        "unsupported_selected_text_count": unsupported_text,
        "unsupported_selected_number_count": unsupported_numbers,
        "model_free_text_accepted_count": model_text_accepted,
        "repeat_count": len(repeats),
        "repeat_final_projection_exact": repeat_exact,
        "passed": passed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v8-dir", type=Path, default=DEFAULT_V8_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--vllm-bin", type=Path, default=DEFAULT_VLLM_BIN)
    parser.add_argument("--base-url", help="reuse a running local OpenAI-compatible server")
    parser.add_argument("--port", type=int, default=8012)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--repeat-count", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--embedding-cache", type=Path,
        default=Path(os.environ.get("FINGLMQA_EMBEDDING_CACHE", "/home/coder/demo/models")),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if not output_dir.is_relative_to(DEFAULT_OUTPUT_ROOT.resolve()):
        raise RuntimeError("output-dir must remain under runs/type3_qwen36_faceted_v9")
    if args.limit is not None and not 1 <= args.limit <= 260:
        raise RuntimeError("limit must be between 1 and 260")
    if not 0 <= args.repeat_count <= 260:
        raise RuntimeError("repeat-count must be between 0 and 260")

    cases, source_hashes = load_cases(args.v8_dir.resolve())
    if args.limit is not None:
        cases = cases[: args.limit]
    repeat_count = min(args.repeat_count, len(cases))
    frozen = freeze_manifest(
        source_hashes=source_hashes,
        model_path=args.model_path,
        model_name=args.model,
        vllm_binary=args.vllm_bin,
    )
    # The manifest is the first write and precedes both BGE and Qwen inference.
    write_json(output_dir / "freeze_manifest.json", frozen)

    os.environ["FINGLMQA_EMBEDDING_CACHE"] = args.embedding_cache.resolve().as_posix()
    retriever = Type3EvidenceRetriever(
        root=ROOT,
        device=args.device,
        model_cache=args.embedding_cache,
        load_dense=True,
    )
    server: VLLMV9Server | None = None
    try:
        if args.base_url:
            base_url = args.base_url
            served_name = args.model
        else:
            server = VLLMV9Server(
                binary=args.vllm_bin,
                model_path=args.model_path,
                served_name=args.model,
                port=args.port,
            )
            server.start()
            base_url = server.base_url
            served_name = server.served_name
        engine = Type3Qwen36FacetedV9(
            OpenAICompatibleClient(base_url),
            retriever,
            root=ROOT,
            model=served_name,
        )
        results, unique_tasks = run_cases(cases, engine=engine)
        repeats = run_repeats(cases, engine=engine, repeat_count=repeat_count)
    finally:
        if server is not None:
            server.stop()

    validation = safety_validation(cases, results, repeats)
    write_jsonl(output_dir / "results.jsonl", results)
    write_jsonl(output_dir / "repeat_results.jsonl", repeats)
    write_jsonl(
        output_dir / "http_evaluation.jsonl",
        (
            projection(result, answer_field="answer", profile=PROFILE_VERSION)
            for result in results
        ),
    )
    qwen_plan_profile = PROFILE_VERSION + "-qwen-plan-deterministic-selector"
    write_jsonl(
        output_dir / "ablations/qwen_plan_deterministic/http_evaluation.jsonl",
        (
            projection(
                result,
                answer_field="qwen_plan_deterministic_selector_answer",
                profile=qwen_plan_profile,
            )
            for result in results
        ),
    )
    no_qwen_profile = PROFILE_VERSION + "-no-qwen-same-index"
    write_jsonl(
        output_dir / "ablations/no_qwen_same_index/http_evaluation.jsonl",
        (
            projection(
                result,
                answer_field="no_qwen_same_index_answer",
                profile=no_qwen_profile,
            )
            for result in results
        ),
    )
    write_json(output_dir / "safety_validation.json", validation)

    report = {
        "schema_version": "finglmqa.experimental.type3_qwen36_faceted_v9.run_report.v1",
        "profile_version": PROFILE_VERSION,
        "result_schema_version": RESULT_SCHEMA,
        "input_rows": len(cases),
        "unique_inference_tasks": unique_tasks,
        "evaluation_duplicate_cache_hits": len(cases) - unique_tasks,
        "terminal_rows": len(results),
        "nonempty_answers": sum(bool(result["answer"].strip()) for result in results),
        "planner_outcome_counts": dict(sorted(Counter(
            result["planner_outcome"] for result in results
        ).items())),
        "selector_outcome_counts": dict(sorted(Counter(
            result["selector_outcome"] for result in results
        ).items())),
        "selected_fragment_count_distribution": dict(sorted(Counter(
            len(result["selected_fragment_ids"]) for result in results
        ).items())),
        "safety_validation_passed": validation["passed"],
        "repeat_final_projection_exact": validation["repeat_final_projection_exact"],
        "repeat_count": len(repeats),
        "freeze_manifest_fingerprint": frozen["manifest_fingerprint"],
        "artifacts": {
            "results_sha256": sha256_file(output_dir / "results.jsonl"),
            "repeat_results_sha256": sha256_file(output_dir / "repeat_results.jsonl"),
            "http_evaluation_sha256": sha256_file(output_dir / "http_evaluation.jsonl"),
            "qwen_plan_deterministic_http_sha256": sha256_file(
                output_dir / "ablations/qwen_plan_deterministic/http_evaluation.jsonl"
            ),
            "no_qwen_same_index_http_sha256": sha256_file(
                output_dir / "ablations/no_qwen_same_index/http_evaluation.jsonl"
            ),
            "safety_validation_sha256": sha256_file(
                output_dir / "safety_validation.json"
            ),
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
