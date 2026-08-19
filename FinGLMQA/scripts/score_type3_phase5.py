#!/usr/bin/env python3
"""Score frozen Type 3 Phase 5 artifacts in a scorer-only process."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import re
import shutil
import statistics
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import score_no_llm_benchmark as base  # noqa: E402


DEFAULT_QUESTIONS = (
    ROOT
    / "data/corpus_package/type3/annual_reports_170_v1/questions/"
    "type3_260_dev_v1/questions.jsonl"
)
DEFAULT_PHASE4 = (
    ROOT
    / "runs/type3_a2rag_tabgr_experiment_v1/annual_reports_170_v1/phase_4/"
    "evaluation/independent_validator"
)
DEFAULT_R2 = (
    ROOT
    / "runs/type3_a2rag_tabgr_experiment_v1/annual_reports_170_v1/phase_5/"
    "r2_compact_baseline_v2_frozen/fresh_process_1"
)
DEFAULT_V10 = ROOT / "runs/type3_qwen36_hybrid_coverage_v10/full/http_evaluation.jsonl"
DEFAULT_OUT = (
    ROOT
    / "runs/type3_a2rag_tabgr_experiment_v1/annual_reports_170_v1/phase_5/"
    "evaluation/scoring_v2_frozen"
)

CORPUS_ID = "annual_reports_170_v1"
QUESTION_PROFILE_ID = "type3_260_dev_v1"
R2_ARM = "r2_compact_baseline_v2"
R2_PROFILE = "type3-a2rag-tabgr-compact-baseline-v2"
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_ACTIVE_STAGING: Path | None = None

PHASE4_ARMS = {
    "union": Path("full/answers.jsonl"),
    "text_only": Path("ablations/text_only/answers.jsonl"),
    "table_only": Path("ablations/table_only/answers.jsonl"),
    "legacy_table_only": Path("ablations/legacy_table_only/answers.jsonl"),
    "v2_table_only": Path("ablations/v2_table_only/answers.jsonl"),
    "no_route_quota": Path("ablations/no_route_quota/answers.jsonl"),
    "no_adjacency": Path("ablations/no_adjacency/answers.jsonl"),
    "no_fact_join": Path("ablations/no_fact_join/answers.jsonl"),
    "no_table_semantics": Path("ablations/no_table_semantics/answers.jsonl"),
}


class Phase5ScorerError(ValueError):
    """Raised when a frozen scoring input is incomplete or has drifted."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return base.load_jsonl(path)


def write_json(path: Path, value: Any) -> None:
    base.write_json(path, value)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    base.write_jsonl(path, rows)


def semantic_sha256(value: Any) -> str:
    return hashlib.sha256(base.canonical_bytes(value)).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase5ScorerError(f"expected JSON object: {path}")
    return value


def _paths_overlap(first: Path, second: Path) -> bool:
    left = first.resolve()
    right = second.resolve()
    return left == right or left.is_relative_to(right) or right.is_relative_to(left)


def _safe_output_dir(
    output_dir: Path,
    *,
    protected_paths: list[Path],
) -> tuple[Path, Path]:
    output = output_dir.resolve()
    for protected in protected_paths:
        if _paths_overlap(output, protected):
            raise Phase5ScorerError(
                f"scorer output overlaps frozen input: {output} vs {protected.resolve()}"
            )
    if output.exists():
        raise Phase5ScorerError(f"scorer output namespace must not exist: {output}")
    staging = output.parent / f".{output.name}.staging"
    if staging.exists():
        raise Phase5ScorerError(f"stale scorer staging directory exists: {staging}")
    return output, staging


def _validate_r2_manifest(
    r2_dir: Path,
    questions: list[dict[str, str]],
) -> dict[str, Any]:
    manifest_path = r2_dir / "run_manifest.json"
    answers_path = r2_dir / "answers.jsonl"
    traces_path = r2_dir / "semantic_traces.jsonl"
    manifest = _read_json(manifest_path)
    expected_ids = [row["question_id"] for row in questions]
    if (
        manifest.get("schema_version")
        != "finglmqa.type3.phase5.compact_run_manifest.v2"
        or manifest.get("profile_version") != R2_PROFILE
        or manifest.get("composer_version") != "type3-phase5-compact-composer-v2"
        or manifest.get("corpus_id") != CORPUS_ID
        or manifest.get("question_profile_id") != QUESTION_PROFILE_ID
        or manifest.get("question_count") != len(questions)
        or manifest.get("question_ids_sha256") != semantic_sha256(expected_ids)
    ):
        raise Phase5ScorerError("R2 run manifest identity differs")
    unsigned = {key: value for key, value in manifest.items() if key != "run_fingerprint"}
    if manifest.get("run_fingerprint") != semantic_sha256(unsigned):
        raise Phase5ScorerError("R2 run manifest fingerprint differs")
    artifacts = manifest.get("artifacts") or {}
    if (
        artifacts.get("answers.jsonl") != sha256_file(answers_path)
        or artifacts.get("semantic_traces.jsonl") != sha256_file(traces_path)
    ):
        raise Phase5ScorerError("R2 artifact hash differs from run manifest")
    code = manifest.get("code") or {}
    expected_code = {
        "composer_sha256": sha256_file(
            ROOT / "src/finglmqa/type3_phase5_compact_composer.py"
        ),
        "runner_sha256": sha256_file(ROOT / "scripts/run_type3_phase5_compact.py"),
        "answer_schema_sha256": sha256_file(
            ROOT / "data/schemas/type3/phase5_compact_answer_v2.schema.json"
        ),
    }
    if code != expected_code:
        raise Phase5ScorerError("R2 generator code/schema hash differs")
    safety = manifest.get("safety") or {}
    if (
        safety.get("nonempty_answers") != len(questions)
        or safety.get("masked_placeholders") != 0
        or safety.get("cross_document_citations") != 0
    ):
        raise Phase5ScorerError("R2 safety gate differs")
    boundary = manifest.get("generator_boundary") or {}
    if (
        boundary.get("benchmark_annotations_available") is not False
        or boundary.get("scorer_outputs_available") is not False
        or boundary.get("input_output_disjointness_enforced") is not True
        or boundary.get("inputs_rehashed_after_generation") is not True
    ):
        raise Phase5ScorerError("R2 generator boundary differs")
    trace_rows = read_jsonl(traces_path)
    if (
        len(trace_rows) != len(questions)
        or [str(row.get("question_id") or "") for row in trace_rows] != expected_ids
    ):
        raise Phase5ScorerError("R2 semantic trace case sequence differs")
    return manifest


def _validate_phase4_reports(
    phase4_dir: Path,
    questions: list[dict[str, str]],
) -> dict[str, Path]:
    validation_path = phase4_dir / "validation_report.json"
    ablation_path = phase4_dir / "ablation_report.json"
    validation = _read_json(validation_path)
    ablation = _read_json(ablation_path)
    if (
        validation.get("schema_version")
        != "finglmqa.type3.a2rag_tabgr.validation_report.v1"
        or validation.get("status") != "passed"
        or validation.get("failed_gates") != []
        or validation.get("corpus_id") != CORPUS_ID
        or validation.get("question_profile_id") != QUESTION_PROFILE_ID
        or validation.get("question_count") != len(questions)
        or validation.get("union", {}).get("cross_document_evidence") != 0
        or validation.get("union", {}).get("unsupported_numeric_literals") != 0
    ):
        raise Phase5ScorerError("Phase 4 validation report differs")
    if (
        ablation.get("schema_version")
        != "finglmqa.type3.a2rag_tabgr.ablation_report.v1"
        or ablation.get("corpus_id") != CORPUS_ID
        or ablation.get("question_profile_id") != QUESTION_PROFILE_ID
        or set(ablation.get("arms") or {}) != set(PHASE4_ARMS)
    ):
        raise Phase5ScorerError("Phase 4 ablation report differs")
    if validation.get("ablation_report_sha256") != sha256_file(ablation_path):
        raise Phase5ScorerError("Phase 4 validation/ablation report hash differs")
    for arm, relative in PHASE4_ARMS.items():
        expected = ablation["arms"][arm]
        path = phase4_dir / relative
        if (
            expected.get("arm") != arm
            or expected.get("question_count") != len(questions)
            or expected.get("nonempty_answers") != len(questions)
            or expected.get("cross_document_evidence") != 0
            or expected.get("unsupported_numeric_literals") != 0
            or expected.get("answer_packets_sha256") != sha256_file(path)
        ):
            raise Phase5ScorerError(f"Phase 4 arm binding differs: {arm}")
    return {
        "phase4_validation_report": validation_path,
        "phase4_ablation_report": ablation_path,
    }


def _benchmark_input_paths(benchmark_root: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for type_dir in sorted(benchmark_root.glob("type_*")):
        for name in ("questions.jsonl", "answers.jsonl"):
            path = type_dir / name
            if not path.is_file():
                raise Phase5ScorerError(f"benchmark input is missing: {path}")
            paths[f"benchmark_{type_dir.name}_{name}"] = path
    if not paths:
        raise Phase5ScorerError("benchmark root has no type inputs")
    return paths


def _model_identity(model: Path) -> dict[str, Any]:
    required = {
        "config": model / "config.json",
        "modules": model / "modules.json",
        "sentence_bert_config": model / "sentence_bert_config.json",
        "weights": model / "pytorch_model.bin",
    }
    for path in required.values():
        if not path.exists():
            raise Phase5ScorerError(f"embedding model artifact is missing: {path}")
    resolved_weights = required["weights"].resolve()
    if not _HEX64_RE.fullmatch(resolved_weights.name):
        raise Phase5ScorerError("embedding weight blob is not content-addressed")
    return {
        "snapshot_path": model.as_posix(),
        "snapshot_revision": model.name,
        "config_sha256": sha256_file(required["config"]),
        "modules_sha256": sha256_file(required["modules"]),
        "sentence_bert_config_sha256": sha256_file(required["sentence_bert_config"]),
        "weights_content_address": resolved_weights.name,
        "weights_size": resolved_weights.stat().st_size,
    }


def _questions(path: Path) -> list[dict[str, str]]:
    rows = read_jsonl(path)
    expected = {"question_id", "question", "document_id"}
    checked: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        if set(row) != expected:
            raise Phase5ScorerError("sanitized question schema differs")
        value = {key: str(row[key]) for key in expected}
        if value["question_id"] in seen:
            raise Phase5ScorerError("duplicate sanitized question id")
        seen.add(value["question_id"])
        checked.append(value)
    return checked


def _packet_predictions(
    path: Path,
    questions: list[dict[str, str]],
    *,
    expected_arm: str | None,
    expected_profile: str | None = None,
) -> dict[str, dict[str, Any]]:
    expected = {row["question_id"]: row for row in questions}
    predictions: dict[str, dict[str, Any]] = {}
    for packet in read_jsonl(path):
        question_id = str(packet.get("question_id") or "")
        if question_id not in expected or question_id in predictions:
            raise Phase5ScorerError(f"invalid prediction id in {path}: {question_id}")
        question = expected[question_id]
        if (
            packet.get("question") != question["question"]
            or packet.get("document_id") != question["document_id"]
        ):
            raise Phase5ScorerError(f"prediction/question mismatch: {question_id}")
        if expected_profile is not None:
            if (
                packet.get("schema_version")
                != "finglmqa.type3.phase5.compact_answer.v2"
                or packet.get("profile_version") != expected_profile
            ):
                raise Phase5ScorerError(f"R2 prediction profile mismatch: {question_id}")
        if expected_arm is not None:
            if packet.get("arm") != expected_arm:
                raise Phase5ScorerError(f"prediction arm mismatch: {question_id}")
            trace = packet.get("semantic_trace") or {}
            if trace.get("arm") != expected_arm:
                raise Phase5ScorerError(f"prediction trace arm mismatch: {question_id}")
        answer = str(packet.get("answer_safe_text") or "")
        if expected_profile is not None:
            citations = packet.get("citations")
            if not isinstance(citations, list) or any(
                not isinstance(citation, Mapping)
                or citation.get("corpus_id") != CORPUS_ID
                or citation.get("document_id") != question["document_id"]
                for citation in citations
            ):
                raise Phase5ScorerError(f"R2 citation identity mismatch: {question_id}")
        predictions[question_id] = {
            "answer": answer,
            "status": "ok" if answer.strip() else "not_found",
            "document_id": question["document_id"],
            "question": question["question"],
            "placeholder_count": answer.count("[未经授权数值]"),
            "selected_candidate_count": len(packet.get("selected_candidate_ids") or ()),
            "evidence": packet.get("evidence") or (),
            "semantic_trace": packet.get("semantic_trace") or {},
        }
    if set(predictions) != set(expected):
        raise Phase5ScorerError(f"prediction case set differs: {path}")
    return predictions


def _v10_predictions(
    path: Path,
    questions: list[dict[str, str]],
) -> dict[str, dict[str, Any]]:
    expected = {row["question_id"]: row for row in questions}
    predictions: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        if row.get("kind") != "benchmark":
            continue
        question_id = str(row.get("case_id") or "")
        if question_id not in expected or question_id in predictions:
            raise Phase5ScorerError(f"invalid v10 prediction id: {question_id}")
        response = row.get("response") or {}
        request = row.get("request") or {}
        question = expected[question_id]
        if request.get("question") != question["question"]:
            raise Phase5ScorerError(f"v10 question mismatch: {question_id}")
        answer = str(response.get("answer") or "")
        predictions[question_id] = {
            "answer": answer,
            "status": str(response.get("status") or "ok"),
            "document_id": question["document_id"],
            "question": question["question"],
            "placeholder_count": answer.count("[未经授权数值]"),
            "selected_candidate_count": 0,
            "evidence": (),
            "semantic_trace": {},
        }
    if set(predictions) != set(expected):
        raise Phase5ScorerError("v10 prediction case set differs")
    return predictions


def _intent_buckets(question: str) -> list[str]:
    buckets: list[str] = []
    rules = {
        "causal_change": (
            "原因",
            "为何",
            "为什么",
            "变动",
            "变化",
            "增长",
            "下降",
            "增加",
            "减少",
        ),
        "financial_table": (
            "收入",
            "利润",
            "现金流",
            "资产",
            "负债",
            "费用",
            "成本",
            "毛利",
            "客户",
            "供应商",
            "股东",
            "金额",
            "财务",
            "会计",
        ),
        "risk_strategy": (
            "风险",
            "战略",
            "竞争力",
            "优势",
            "研发",
            "创新",
            "行业",
            "业务",
            "产品",
            "市场",
        ),
        "governance_entity": (
            "董事",
            "监事",
            "高管",
            "治理",
            "子公司",
            "关联",
            "诉讼",
            "处罚",
            "整改",
            "员工",
            "股权",
        ),
        "negative_existence": ("是否", "不存在", "未发生", "没有", "无"),
    }
    for name, terms in rules.items():
        if any(term in question for term in terms):
            buckets.append(name)
    return buckets or ["other"]


def _length_bucket(length: int) -> str:
    if length <= 512:
        return "chars_le_512"
    if length <= 2048:
        return "chars_513_2048"
    if length <= 4096:
        return "chars_2049_4096"
    if length <= 8192:
        return "chars_4097_8192"
    return "chars_gt_8192"


def _placeholder_bucket(count: int) -> str:
    if count == 0:
        return "masks_0"
    if count <= 20:
        return "masks_1_20"
    if count <= 100:
        return "masks_21_100"
    if count <= 300:
        return "masks_101_300"
    return "masks_gt_300"


def _summary(scores: list[float]) -> dict[str, Any]:
    total = round(sum(scores), 6)
    return {
        "count": len(scores),
        "total_score": total,
        "average_score": round(total / len(scores), 6) if scores else 0.0,
        "zero_count": sum(value == 0 for value in scores),
        "at_least_0_5": sum(value >= 0.5 for value in scores),
        "at_least_0_75": sum(value >= 0.75 for value in scores),
        "at_least_0_9": sum(value >= 0.9 for value in scores),
    }


def _bucket_report(details: list[dict[str, Any]]) -> dict[str, Any]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for row in details:
        score = float(row["bge_m3"]["score"])
        for intent in row["intent_buckets"]:
            buckets[f"intent:{intent}"].append(score)
        buckets[f"length:{row['answer_length_bucket']}"].append(score)
        buckets[f"placeholder:{row['placeholder_bucket']}"].append(score)
    return {key: _summary(values) for key, values in sorted(buckets.items())}


def _cluster_scores(
    details: list[dict[str, Any]],
) -> dict[tuple[str, str], float]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in details:
        grouped[(row["document_id"], row["question"])].append(
            float(row["bge_m3"]["score"])
        )
    return {
        key: sum(values) / len(values)
        for key, values in sorted(grouped.items())
    }


def _bootstrap_ci(deltas: list[float], *, iterations: int = 10000) -> list[float]:
    if not deltas:
        return [0.0, 0.0]
    random_generator = random.Random(0)
    means: list[float] = []
    count = len(deltas)
    for _ in range(iterations):
        means.append(
            sum(deltas[random_generator.randrange(count)] for _ in range(count))
            / count
        )
    means.sort()
    lower = means[math.floor(0.025 * (iterations - 1))]
    upper = means[math.floor(0.975 * (iterations - 1))]
    return [round(lower, 6), round(upper, 6)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--phase4-dir", type=Path, default=DEFAULT_PHASE4)
    parser.add_argument("--r2-dir", type=Path, default=DEFAULT_R2)
    parser.add_argument("--v10-results", type=Path, default=DEFAULT_V10)
    parser.add_argument("--oracle", type=Path, default=base.DEFAULT_ORACLE)
    parser.add_argument("--benchmark-root", type=Path, default=base.DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--evaluator", type=Path, default=base.DEFAULT_EVALUATOR)
    parser.add_argument("--model", type=Path, default=base.DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def _main() -> int:
    global _ACTIVE_STAGING
    args = parse_args()
    args.questions = args.questions.resolve()
    args.phase4_dir = args.phase4_dir.resolve()
    args.r2_dir = args.r2_dir.resolve()
    args.v10_results = args.v10_results.resolve()
    args.oracle = args.oracle.resolve()
    args.benchmark_root = args.benchmark_root.resolve()
    args.evaluator = args.evaluator.resolve()
    args.model = args.model.resolve()
    args.out_dir = args.out_dir.resolve()

    # Generator artifacts are completely validated, loaded, and hash-frozen before gold.
    questions = _questions(args.questions)
    if len(questions) != 260:
        raise Phase5ScorerError("Phase 5 scorer requires the frozen 260-question profile")
    r2_manifest = _validate_r2_manifest(args.r2_dir, questions)
    phase4_reports = _validate_phase4_reports(args.phase4_dir, questions)
    predictions: dict[str, dict[str, dict[str, Any]]] = {
        "v10": _v10_predictions(args.v10_results, questions),
        R2_ARM: _packet_predictions(
            args.r2_dir / "answers.jsonl",
            questions,
            expected_arm=None,
            expected_profile=R2_PROFILE,
        ),
    }
    input_paths = {
        "questions": args.questions,
        "v10": args.v10_results,
        "r2_answers": args.r2_dir / "answers.jsonl",
        "r2_traces": args.r2_dir / "semantic_traces.jsonl",
        "r2_manifest": args.r2_dir / "run_manifest.json",
        **phase4_reports,
    }
    for arm, relative in PHASE4_ARMS.items():
        path = args.phase4_dir / relative
        predictions[arm] = _packet_predictions(path, questions, expected_arm=arm)
        input_paths[f"phase4_{arm}"] = path
    frozen_hashes = {key: sha256_file(path) for key, path in input_paths.items()}
    benchmark_paths = _benchmark_input_paths(args.benchmark_root)
    model_identity = _model_identity(args.model)
    final_out_dir, staging_dir = _safe_output_dir(
        args.out_dir,
        protected_paths=[
            *input_paths.values(),
            args.oracle,
            args.benchmark_root,
            args.evaluator,
            args.model,
        ],
    )
    args.out_dir = staging_dir
    _ACTIVE_STAGING = staging_dir

    # Scorer-only boundary begins here.
    evaluator = base.load_public_evaluator(args.evaluator)
    oracle_rows = [
        row
        for row in read_jsonl(args.oracle)
        if row["source"]["benchmark_type"] == "3-1"
    ]
    gold = base.load_selected_gold(oracle_rows, args.benchmark_root)
    if len(gold) != len(questions):
        raise Phase5ScorerError("gold/question counts differ")
    gold_by_id = {
        f"benchmark:{row['type']}:{row['uid']}": row
        for row in gold
    }
    ordered_ids = [row["question_id"] for row in questions]
    if set(ordered_ids) != set(gold_by_id):
        raise Phase5ScorerError("gold/question ids differ")
    for row in questions:
        if gold_by_id[row["question_id"]]["question"] != row["question"]:
            raise Phase5ScorerError("gold/sanitized question mismatch")

    embedding_inputs: list[str] = []
    for row in gold:
        embedding_inputs.extend(str(value) for value in row["references"])
    for arm_predictions in predictions.values():
        embedding_inputs.extend(
            arm_predictions[question_id]["answer"] for question_id in ordered_ids
        )
    vectors, embedding_metadata = base.embed_texts(
        embedding_inputs, args.model, args.device, args.batch_size
    )

    all_details: dict[str, list[dict[str, Any]]] = {}
    reports: dict[str, dict[str, Any]] = {}
    for arm, arm_predictions in predictions.items():
        details: list[dict[str, Any]] = []
        for ordinal, question_id in enumerate(ordered_ids):
            gold_row = gold_by_id[question_id]
            prediction = arm_predictions[question_id]
            answer = prediction["answer"]
            similarity, reference = base.bge_max_similarity(
                answer, gold_row["references"], vectors
            )
            sequence_similarity, sequence_reference = base.sequence_max_similarity(
                answer, gold_row["references"], evaluator
            )
            bge_score = base.score_from_similarity(
                gold_row, answer, similarity, reference, evaluator
            )
            sequence_score = base.score_from_similarity(
                gold_row,
                answer,
                sequence_similarity,
                sequence_reference,
                evaluator,
            )
            detail = {
                "schema_version": "finglmqa.type3.phase5.score_detail.v1",
                "arm": arm,
                "selected_ordinal": ordinal,
                "case_id": question_id,
                "uid": gold_row["uid"],
                "type": gold_row["type"],
                "question": gold_row["question"],
                "document_id": prediction["document_id"],
                "prediction": answer,
                "answer_nonempty": bool(answer.strip()),
                "answer_characters": len(answer),
                "answer_length_bucket": _length_bucket(len(answer)),
                "placeholder_count": prediction["placeholder_count"],
                "placeholder_bucket": _placeholder_bucket(
                    prediction["placeholder_count"]
                ),
                "intent_buckets": _intent_buckets(gold_row["question"]),
                "selected_candidate_count": prediction["selected_candidate_count"],
                "bge_m3": bge_score,
                "sequence_audit": sequence_score,
            }
            details.append(detail)
        all_details[arm] = details
        cluster_scores = _cluster_scores(details)
        reports[arm] = {
            "schema_version": "finglmqa.type3.phase5.arm_score_report.v1",
            "arm": arm,
            "benchmark_weighted": _summary(
                [float(row["bge_m3"]["score"]) for row in details]
            ),
            "semantic_task_macro": {
                **_summary(list(cluster_scores.values())),
                "unique_tasks": len(cluster_scores),
            },
            "coverage": {
                "nonempty_answers": sum(row["answer_nonempty"] for row in details),
                "placeholder_questions": sum(
                    row["placeholder_count"] > 0 for row in details
                ),
                "placeholder_total": sum(
                    row["placeholder_count"] for row in details
                ),
                "answer_characters_mean": round(
                    statistics.mean(row["answer_characters"] for row in details), 3
                ),
                "answer_characters_median": statistics.median(
                    row["answer_characters"] for row in details
                ),
                "answer_characters_max": max(
                    row["answer_characters"] for row in details
                ),
            },
            "buckets": _bucket_report(details),
        }
        arm_dir = args.out_dir / "arms" / arm
        write_jsonl(arm_dir / "score_details.jsonl", details)
        write_json(
            arm_dir / "score_report.json",
            reports[arm],
        )

    baseline_details = {
        row["case_id"]: row for row in all_details["v10"]
    }
    baseline_clusters = _cluster_scores(all_details["v10"])
    comparisons: dict[str, Any] = {}
    paired_rows: list[dict[str, Any]] = []
    for arm, details in all_details.items():
        if arm == "v10":
            continue
        deltas = []
        for row in details:
            baseline = baseline_details[row["case_id"]]
            delta = round(
                float(row["bge_m3"]["score"])
                - float(baseline["bge_m3"]["score"]),
                6,
            )
            deltas.append(delta)
            paired_rows.append(
                {
                    "arm": arm,
                    "case_id": row["case_id"],
                    "document_id": row["document_id"],
                    "question": row["question"],
                    "score": row["bge_m3"]["score"],
                    "v10_score": baseline["bge_m3"]["score"],
                    "delta": delta,
                }
            )
        clusters = _cluster_scores(details)
        cluster_deltas = [
            clusters[key] - baseline_clusters[key] for key in baseline_clusters
        ]
        comparisons[arm] = {
            "benchmark_weighted_delta": round(sum(deltas) / len(deltas), 6),
            "wins": sum(value > 0 for value in deltas),
            "ties": sum(value == 0 for value in deltas),
            "losses": sum(value < 0 for value in deltas),
            "semantic_task_macro_delta": round(
                sum(cluster_deltas) / len(cluster_deltas), 6
            ),
            "semantic_task_cluster_bootstrap_95_ci": _bootstrap_ci(cluster_deltas),
        }

    r2_report = reports[R2_ARM]
    v10_report = reports["v10"]
    r2_comparison = comparisons[R2_ARM]
    bucket_failures: list[dict[str, Any]] = []
    for bucket, current in r2_report["buckets"].items():
        baseline = v10_report["buckets"].get(bucket)
        if (
            baseline
            and current["count"] >= 10
            and baseline["count"] == current["count"]
        ):
            delta = round(
                current["average_score"] - baseline["average_score"], 6
            )
            if delta < -0.02:
                bucket_failures.append(
                    {"bucket": bucket, "count": current["count"], "delta": delta}
                )
    quality_gate = {
        "benchmark_weighted_at_least_0_771640": (
            r2_report["benchmark_weighted"]["average_score"] >= 0.771640
        ),
        "semantic_task_macro_at_least_v10_plus_0_003": (
            r2_report["semantic_task_macro"]["average_score"]
            >= v10_report["semantic_task_macro"]["average_score"] + 0.003
        ),
        "semantic_task_bootstrap_ci_lower_above_zero": (
            r2_comparison["semantic_task_cluster_bootstrap_95_ci"][0] > 0
        ),
        "all_answers_nonempty": (
            r2_report["coverage"]["nonempty_answers"] == len(questions)
        ),
        "final_placeholders_zero": (
            r2_report["coverage"]["placeholder_total"] == 0
        ),
        "supported_bucket_regressions_within_0_02": not bucket_failures,
        "bucket_failures": bucket_failures,
    }
    quality_gate["passed"] = all(
        value for key, value in quality_gate.items()
        if key not in {"bucket_failures", "passed"}
    )

    comparison_dir = args.out_dir / "comparison"
    write_jsonl(comparison_dir / "paired_deltas.jsonl", paired_rows)
    summary = {
        "schema_version": "finglmqa.type3.phase5.comparison.v2",
        "corpus_id": CORPUS_ID,
        "question_profile_id": QUESTION_PROFILE_ID,
        "question_count": len(questions),
        "semantic_task_cluster_count": len(_cluster_scores(all_details[R2_ARM])),
        "arms": reports,
        "comparisons_vs_v10": comparisons,
        "r2_quality_gate": quality_gate,
        "scoring_boundary": {
            "generator_artifacts_loaded_before_gold": True,
            "generator_input_hashes_frozen_before_gold": frozen_hashes,
            "gold_available_to_generator": False,
            "scorer_imported_generator_pipeline": False,
        },
        "embedding": embedding_metadata,
    }
    write_json(comparison_dir / "ablation_summary.json", summary)

    after_hashes = {key: sha256_file(path) for key, path in input_paths.items()}
    if after_hashes != frozen_hashes:
        raise Phase5ScorerError("generator artifact hash changed during scoring")
    artifact_paths = sorted(
        path for path in args.out_dir.rglob("*") if path.is_file()
    )
    benchmark_hashes = {
        key: {"path": path.as_posix(), "sha256": sha256_file(path)}
        for key, path in sorted(benchmark_paths.items())
    }
    manifest_unsigned = {
        "schema_version": "finglmqa.type3.phase5.scoring_manifest.v2",
        "corpus_id": CORPUS_ID,
        "question_profile_id": QUESTION_PROFILE_ID,
        "question_count": len(questions),
        "r2_profile_version": R2_PROFILE,
        "r2_run_fingerprint": r2_manifest["run_fingerprint"],
        "arm_set": sorted(predictions),
        "scoring_configuration": {
            "primary_similarity": "BAAI/bge-m3 cosine",
            "device": args.device,
            "batch_size": args.batch_size,
            "bootstrap_seed": 0,
            "bootstrap_iterations": 10000,
            "cluster_key": ["document_id", "question"],
        },
        "inputs": {
            **{
                key: {"path": path.as_posix(), "sha256": frozen_hashes[key]}
                for key, path in sorted(input_paths.items())
            },
            "oracle": {"path": args.oracle.as_posix(), "sha256": sha256_file(args.oracle)},
            "evaluator": {
                "path": args.evaluator.as_posix(),
                "sha256": sha256_file(args.evaluator),
            },
            "scorer": {
                "path": Path(__file__).resolve().as_posix(),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "base_scorer": {
                "path": Path(base.__file__).resolve().as_posix(),
                "sha256": sha256_file(Path(base.__file__).resolve()),
            },
        },
        "benchmark_inputs": benchmark_hashes,
        "embedding_model_identity": model_identity,
        "generator_artifacts_unchanged_after_scoring": True,
        "scoring_boundary": {
            "generator_artifacts_loaded_and_validated_before_gold": True,
            "generator_input_hashes_frozen_before_gold": frozen_hashes,
            "gold_available_to_generator": False,
            "scorer_imported_generator_pipeline": False,
            "staging_then_atomic_publish": True,
        },
        "artifacts": {
            path.relative_to(args.out_dir).as_posix(): {
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in artifact_paths
        },
    }
    manifest = {
        **manifest_unsigned,
        "scoring_fingerprint": semantic_sha256(manifest_unsigned),
    }
    write_json(args.out_dir / "manifest.json", manifest)
    args.out_dir.rename(final_out_dir)
    _ACTIVE_STAGING = None
    print(
        json.dumps(
            {
                "status": "passed",
                "v10": v10_report["benchmark_weighted"]["average_score"],
                "r2": r2_report["benchmark_weighted"]["average_score"],
                "r2_semantic_task_macro": r2_report["semantic_task_macro"][
                    "average_score"
                ],
                "r2_quality_gate": quality_gate["passed"],
                "output_dir": final_out_dir.as_posix(),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def main() -> int:
    global _ACTIVE_STAGING
    try:
        return _main()
    finally:
        if _ACTIVE_STAGING is not None and _ACTIVE_STAGING.exists():
            shutil.rmtree(_ACTIVE_STAGING)
        _ACTIVE_STAGING = None


if __name__ == "__main__":
    raise SystemExit(main())
