#!/usr/bin/env python3
"""Score the frozen Phase 10 benchmark answers with the public answer-only rubric.

The answer/key-word gates intentionally reuse BIG-Finbenchmark's V1 evaluator.
The primary similarity is cosine similarity from the project's frozen BGE-M3
encoder.  The evaluator's original SequenceMatcher score is emitted as an
audit-only secondary result.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import tempfile
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK_ROOT = Path("/home/coder/demo/BIG-Finbenchmark/data/by_type")
DEFAULT_EVALUATOR = Path("/home/coder/demo/BIG-Finbenchmark/scripts/evaluate_finglm.py")
DEFAULT_MODEL = Path(
    "/home/coder/demo/models/models--BAAI--bge-m3/snapshots/"
    "5617a9f61b028005a4858fdac845db406aefb181"
)
DEFAULT_ORACLE = ROOT / "runs/phase_08/benchmark_decomposition_oracle.jsonl"
DEFAULT_HTTP_RESULTS = ROOT / "runs/phase_10/http_evaluation.jsonl"
DEFAULT_OUT_DIR = ROOT / "runs/no_llm_benchmark_scoring"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
    return rows


def canonical_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_json(path: Path, payload: Any) -> None:
    atomic_write(path, canonical_bytes(payload))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    atomic_write(path, b"".join(canonical_bytes(row) for row in rows))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_public_evaluator(path: Path):
    spec = importlib.util.spec_from_file_location("big_finbenchmark_evaluator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalize_embedding_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).strip()
    return " ".join(text.split())


def load_selected_gold(
    oracle_rows: list[dict[str, Any]], benchmark_root: Path
) -> list[dict[str, Any]]:
    gold: dict[tuple[str, str], dict[str, Any]] = {}
    for type_dir in sorted(benchmark_root.glob("type_*")):
        benchmark_type = type_dir.name.removeprefix("type_")
        questions = {row["uid"]: row for row in load_jsonl(type_dir / "questions.jsonl")}
        answers = {row["uid"]: row for row in load_jsonl(type_dir / "answers.jsonl")}
        if set(questions) != set(answers):
            raise RuntimeError(f"Question/answer UID mismatch: {type_dir}")
        for uid, question in questions.items():
            answer = answers[uid]
            gold[(benchmark_type, uid)] = {
                "uid": uid,
                "source_split": question["source_split"],
                "source_id": question["source_id"],
                "type": question["type"],
                "question": question["question"],
                "prompt": answer.get("prompt") or {},
                "references": answer.get("references") or [],
            }

    selected: list[dict[str, Any]] = []
    for oracle in sorted(oracle_rows, key=lambda row: row["source"]["selected_ordinal"]):
        source = oracle["source"]
        key = (source["benchmark_type"], source["uid"])
        if key not in gold:
            raise RuntimeError(f"Selected gold row is missing: {key}")
        row = gold[key]
        if row["question"] != source["question"]:
            raise RuntimeError(f"Oracle/gold question mismatch: {key}")
        selected.append(row)
    return selected


def load_predictions(http_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    predictions: dict[str, dict[str, Any]] = {}
    for row in http_rows:
        if row.get("kind") != "benchmark":
            continue
        case_id = str(row["case_id"])
        if case_id in predictions:
            raise RuntimeError(f"Duplicate HTTP benchmark case: {case_id}")
        response = row["response"]
        predictions[case_id] = {
            "answer": str(response.get("answer") or ""),
            "status": response["status"],
            "errors": response.get("errors") or [],
            "oracle_match": row.get("oracle_match"),
            "request_question": row["request"]["question"],
        }
    return predictions


def embed_texts(
    texts: list[str], model_path: Path, device: str, batch_size: int
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    random.seed(0)
    np.random.seed(0)

    import torch
    from sentence_transformers import SentenceTransformer

    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    unique: list[str] = []
    seen: set[str] = set()
    for value in texts:
        text = normalize_embedding_text(value)
        if not text or text in seen:
            continue
        seen.add(text)
        unique.append(text)

    model = SentenceTransformer(model_path.as_posix(), device=device)
    vectors = model.encode(
        unique,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    vector_map = {text: vectors[index] for index, text in enumerate(unique)}
    metadata = {
        "model": "BAAI/bge-m3",
        "snapshot": model_path.name,
        "model_path_basename": model_path.name,
        "device": device,
        "batch_size": batch_size,
        "embedding_dimension": int(vectors.shape[1]) if len(vectors) else 0,
        "unique_text_count": len(unique),
        "normalized_embeddings": True,
        "cosine_clamped_to_unit_interval": True,
    }
    return vector_map, metadata


def bge_max_similarity(
    prediction: str, references: list[Any], vectors: dict[str, np.ndarray]
) -> tuple[float, str | None]:
    prediction_text = normalize_embedding_text(prediction)
    if not prediction_text or prediction_text not in vectors:
        return 0.0, None
    best_score = 0.0
    best_reference: str | None = None
    prediction_vector = vectors[prediction_text]
    for reference in references:
        reference_value = str(reference)
        reference_text = normalize_embedding_text(reference_value)
        if not reference_text or reference_text not in vectors:
            continue
        cosine = float(np.dot(prediction_vector, vectors[reference_text]))
        score = max(0.0, min(1.0, cosine))
        if score > best_score:
            best_score = score
            best_reference = reference_value
    return best_score, best_reference


def score_from_similarity(
    row: dict[str, Any],
    prediction: str,
    similarity: float,
    best_reference: str | None,
    evaluator: Any,
) -> dict[str, Any]:
    prompt = row.get("prompt") or {}
    prom_answer = prompt.get("prom_answer")
    key_word = prompt.get("key_word")
    prom_answer_ok = evaluator.answer_match(prediction, prom_answer) if prom_answer else None
    keyword_ok, matched_keywords, missing_keywords = (
        evaluator.keyword_match(prediction, key_word) if key_word else (None, [], [])
    )

    if not prompt or (not prom_answer and not key_word):
        score = similarity
        reason = "no_scorable_prompt_similarity_only" if prompt else "no_prompt_similarity_only"
    elif prom_answer:
        if not prom_answer_ok:
            score = 0.0
            reason = "prom_answer_not_matched"
        else:
            keyword_bonus = 0.25 if keyword_ok else 0.0
            score = 0.25 + keyword_bonus + 0.5 * similarity
            reason = "prom_answer_matched"
            if keyword_ok is True:
                reason += "_keyword_matched"
            elif keyword_ok is False:
                reason += "_keyword_missing"
            else:
                reason += "_no_keyword"
    else:
        keyword_bonus = 0.25 if keyword_ok else 0.0
        score = keyword_bonus + 0.75 * similarity
        reason = "no_prom_answer"
        reason += "_keyword_matched" if keyword_ok else "_keyword_missing"

    return {
        "score": round(float(score), 6),
        "max_similarity": round(float(similarity), 6),
        "best_reference": best_reference,
        "prom_answer": prom_answer,
        "prom_answer_match": prom_answer_ok,
        "key_word": key_word,
        "keyword_match": keyword_ok,
        "matched_keywords": matched_keywords,
        "missing_keywords": missing_keywords,
        "score_reason": reason,
    }


def sequence_max_similarity(
    prediction: str, references: list[Any], evaluator: Any
) -> tuple[float, str | None]:
    if not prediction.strip():
        return 0.0, None
    return evaluator.max_similarity(prediction, references, "sequence")


def aggregate(details: list[dict[str, Any]], score_key: str) -> dict[str, Any]:
    def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
        scores = [float(row[score_key]["score"]) for row in rows]
        total = round(sum(scores), 6)
        count = len(scores)
        return {
            "count": count,
            "total_score": total,
            "average_score": round(total / count, 6) if count else 0.0,
            "average_percent": round(100.0 * total / count, 4) if count else 0.0,
            "zero_count": sum(score == 0 for score in scores),
            "positive_count": sum(score > 0 for score in scores),
            "at_least_0_5": sum(score >= 0.5 for score in scores),
            "at_least_0_75": sum(score >= 0.75 for score in scores),
            "at_least_0_9": sum(score >= 0.9 for score in scores),
        }

    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_status: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in details:
        by_type[row["type"]].append(row)
        by_family[row["type"].split("-", 1)[0]].append(row)
        by_status[row["service_status"]].append(row)
    return {
        "overall": summary(details),
        "answered_only": summary([row for row in details if row["answer_nonempty"]]),
        "by_family": {key: summary(rows) for key, rows in sorted(by_family.items())},
        "by_type": {key: summary(rows) for key, rows in sorted(by_type.items())},
        "by_service_status": {key: summary(rows) for key, rows in sorted(by_status.items())},
    }


def render_report(report: dict[str, Any]) -> str:
    primary = report["scores"]["bge_m3"]["overall"]
    sequence = report["scores"]["sequence_audit"]["overall"]
    lines = [
        "# 无生成式 LLM Benchmark 评分报告",
        "",
        f"本报告对 {primary['count']} 道 benchmark 答案进行评分。答案生成链路未使用生成式 LLM；主评分中的语义相似度由本地 BGE-M3 编码器计算。",
        "",
        "## 结果",
        "",
        f"- 主分（BGE-M3）：总分 {primary['total_score']} / {primary['count']}，平均 {primary['average_score']}（{primary['average_percent']}%）。",
        f"- 有非空答案：{report['coverage']['nonempty_answers']} / {report['coverage']['total']}；空答案按规则计 0 分。",
        f"- 仅看非空答案：平均 {report['scores']['bge_m3']['answered_only']['average_score']}（{report['scores']['bge_m3']['answered_only']['average_percent']}%）。",
        f"- 原仓库 V1 字符序列对照：总分 {sequence['total_score']}，平均 {sequence['average_score']}（{sequence['average_percent']}%）。",
        "",
        "## 按主类（BGE-M3 主分）",
        "",
        "| 主类 | 题数 | 总分 | 均分 | 非零题 |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, value in report["scores"]["bge_m3"]["by_family"].items():
        lines.append(
            f"| Type {key} | {value['count']} | {value['total_score']} | "
            f"{value['average_score']} | {value['positive_count']} |"
        )
    lines.extend([
        "",
        "## 按题型（BGE-M3 主分）",
        "",
        "| 题型 | 题数 | 总分 | 均分 | 非零题 |",
        "|---|---:|---:|---:|---:|",
    ])
    for key, value in report["scores"]["bge_m3"]["by_type"].items():
        lines.append(
            f"| {key} | {value['count']} | {value['total_score']} | "
            f"{value['average_score']} | {value['positive_count']} |"
        )
    lines.extend(
        [
            "",
            "## 评分口径",
            "",
            "- `prom_answer` 未命中：0 分。",
            "- `prom_answer` 命中且全部 `key_word` 命中：`0.25 + 0.25 + 0.5 × max_similarity`。",
            "- `prom_answer` 命中但关键词未全部命中：`0.25 + 0.5 × max_similarity`。",
            "- 无可评分 prompt：仅取三条参考答案中的最高相似度。",
            "- 仅有关键词、没有 `prom_answer` 的扩展情形沿用仓库 V1：`keyword_bonus + 0.75 × max_similarity`。",
            "- 数字和关键词命中逻辑直接复用 BIG-Finbenchmark 的公开 V1 evaluator；语义余弦值限制在 `[0, 1]`。",
            "",
            "## 说明",
            "",
            "BGE-M3 只参与评测相似度，不参与正式答案生成。",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", type=Path, default=DEFAULT_ORACLE)
    parser.add_argument("--http-results", type=Path, default=DEFAULT_HTTP_RESULTS)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--evaluator", type=Path, default=DEFAULT_EVALUATOR)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--benchmark-types",
        default="1,1-2,2-1,2-2,3-1",
        help="Comma-separated frozen benchmark types to score",
    )
    parser.add_argument("--expected-count", type=int, default=1003)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.oracle = args.oracle.resolve()
    args.http_results = args.http_results.resolve()
    args.benchmark_root = args.benchmark_root.resolve()
    args.evaluator = args.evaluator.resolve()
    args.model = args.model.resolve()
    args.out_dir = args.out_dir.resolve()
    evaluator = load_public_evaluator(args.evaluator)
    selected_types = {value.strip() for value in args.benchmark_types.split(",") if value.strip()}
    oracle_rows = [
        row for row in load_jsonl(args.oracle)
        if row["source"]["benchmark_type"] in selected_types
    ]
    selected = load_selected_gold(oracle_rows, args.benchmark_root)
    predictions = load_predictions(load_jsonl(args.http_results))

    if len(selected) != args.expected_count:
        raise RuntimeError(
            f"Expected {args.expected_count} selected benchmark rows, got {len(selected)}"
        )
    expected_case_ids = {
        f"benchmark:{row['type']}:{row['uid']}" for row in selected
    }
    predictions = {
        case_id: row for case_id, row in predictions.items() if case_id in expected_case_ids
    }
    if len(predictions) != args.expected_count:
        raise RuntimeError(
            f"Expected {args.expected_count} matching benchmark predictions, got {len(predictions)}"
        )

    inputs_to_embed: list[str] = []
    joined: list[tuple[dict[str, Any], dict[str, Any], str]] = []
    for row in selected:
        case_id = f"benchmark:{row['type']}:{row['uid']}"
        if case_id not in predictions:
            raise RuntimeError(f"HTTP prediction is missing: {case_id}")
        prediction = predictions[case_id]
        if prediction["request_question"] != row["question"]:
            raise RuntimeError(f"HTTP/gold question mismatch: {case_id}")
        if prediction["oracle_match"] is not True:
            raise RuntimeError(f"HTTP oracle mismatch: {case_id}")
        answer = prediction["answer"]
        inputs_to_embed.append(answer)
        inputs_to_embed.extend(str(value) for value in row["references"])
        joined.append((row, prediction, case_id))

    vectors, embedding_metadata = embed_texts(
        inputs_to_embed, args.model, args.device, args.batch_size
    )

    details: list[dict[str, Any]] = []
    extracted_predictions: list[dict[str, Any]] = []
    for ordinal, (row, prediction, case_id) in enumerate(joined):
        answer = prediction["answer"]
        bge_similarity, bge_reference = bge_max_similarity(answer, row["references"], vectors)
        sequence_similarity, sequence_reference = sequence_max_similarity(
            answer, row["references"], evaluator
        )
        bge_score = score_from_similarity(
            row, answer, bge_similarity, bge_reference, evaluator
        )
        sequence_score = score_from_similarity(
            row, answer, sequence_similarity, sequence_reference, evaluator
        )
        failure_codes = [error.get("failure_code") for error in prediction["errors"]]
        detail = {
            "schema_version": "finglmqa.no_llm_benchmark_score_detail.v1",
            "selected_ordinal": ordinal,
            "case_id": case_id,
            "uid": row["uid"],
            "source_split": row["source_split"],
            "source_id": row["source_id"],
            "type": row["type"],
            "question": row["question"],
            "prompt": row["prompt"],
            "references": row["references"],
            "prediction": answer,
            "answer_nonempty": bool(answer.strip()),
            "service_status": prediction["status"],
            "service_failure_codes": failure_codes,
            "bge_m3": bge_score,
            "sequence_audit": sequence_score,
        }
        details.append(detail)
        extracted_predictions.append(
            {
                "uid": row["uid"],
                "source_split": row["source_split"],
                "source_id": row["source_id"],
                "type": row["type"],
                "answer": answer,
                "service_status": prediction["status"],
            }
        )

    score_reason_counts = Counter(row["bge_m3"]["score_reason"] for row in details)
    status_counts = Counter(row["service_status"] for row in details)
    nonempty = sum(row["answer_nonempty"] for row in details)
    prom_rows = [row for row in details if row["bge_m3"]["prom_answer_match"] is not None]
    keyword_rows = [row for row in details if row["bge_m3"]["keyword_match"] is not None]
    report = {
        "schema_version": "finglmqa.no_llm_benchmark_score_report.v1",
        "scope": {
            "description": "Phase 10 deterministic official answers for the frozen benchmark subset",
            "requested_label": "1000 questions",
            "actual_complete_count": len(details),
            "selected_types": sorted(selected_types),
            "types": dict(sorted(Counter(row["type"] for row in details).items())),
            "answer_generation_uses_generative_llm": False,
            "scoring_uses_embedding_encoder": True,
        },
        "coverage": {
            "total": len(details),
            "nonempty_answers": nonempty,
            "empty_answers": len(details) - nonempty,
            "service_status_counts": dict(sorted(status_counts.items())),
        },
        "gates": {
            "prom_answer_applicable": len(prom_rows),
            "prom_answer_matched": sum(row["bge_m3"]["prom_answer_match"] is True for row in prom_rows),
            "prom_answer_not_matched": sum(row["bge_m3"]["prom_answer_match"] is False for row in prom_rows),
            "keyword_applicable": len(keyword_rows),
            "keyword_matched": sum(row["bge_m3"]["keyword_match"] is True for row in keyword_rows),
            "keyword_not_matched": sum(row["bge_m3"]["keyword_match"] is False for row in keyword_rows),
            "score_reason_counts": dict(sorted(score_reason_counts.items())),
        },
        "scores": {
            "bge_m3": aggregate(details, "bge_m3"),
            "sequence_audit": aggregate(details, "sequence_audit"),
        },
        "similarity": {
            "primary": embedding_metadata,
            "audit": {
                "method": "difflib.SequenceMatcher on public evaluator normalized text",
                "evaluator_sha256": sha256_file(args.evaluator),
            },
        },
        "inputs": {
            "oracle": {"path": args.oracle.relative_to(ROOT).as_posix(), "sha256": sha256_file(args.oracle)},
            "http_results": {
                "path": args.http_results.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(args.http_results),
            },
            "benchmark_root": args.benchmark_root.as_posix(),
            "benchmark_files": [
                {
                    "path": path.relative_to(args.benchmark_root).as_posix(),
                    "sha256": sha256_file(path),
                }
                for path in sorted(args.benchmark_root.glob("type_*/[aq]*.jsonl"))
                if path.name in {"answers.jsonl", "questions.jsonl"}
            ],
            "public_evaluator": args.evaluator.as_posix(),
            "scorer_script_sha256": sha256_file(Path(__file__)),
            "model_snapshot": args.model.name,
        },
    }

    out_dir = args.out_dir
    details_path = out_dir / "score_details.jsonl"
    predictions_path = out_dir / "predictions.jsonl"
    report_path = out_dir / "score_report.json"
    markdown_path = out_dir / "score_report.md"
    write_jsonl(details_path, details)
    write_jsonl(predictions_path, extracted_predictions)
    write_json(report_path, report)
    atomic_write(markdown_path, render_report(report).encode("utf-8"))

    lowest_nonempty = sorted(
        (row for row in details if row["answer_nonempty"]),
        key=lambda row: (row["bge_m3"]["score"], row["selected_ordinal"]),
    )[:50]
    write_jsonl(out_dir / "lowest_50_nonempty.jsonl", lowest_nonempty)
    manifest = {
        "schema_version": "finglmqa.no_llm_benchmark_score_manifest.v1",
        "artifacts": {
            path.name: {"size": path.stat().st_size, "sha256": sha256_file(path)}
            for path in (details_path, predictions_path, report_path, markdown_path, out_dir / "lowest_50_nonempty.jsonl")
        },
    }
    write_json(out_dir / "manifest.json", manifest)

    primary = report["scores"]["bge_m3"]["overall"]
    print(f"Scored {primary['count']} benchmark answers")
    print(f"BGE-M3 total={primary['total_score']} average={primary['average_score']}")
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
