#!/usr/bin/env python3
"""Score the experimental Phase B.2 fast-iteration arms.

The generator output is loaded and hashed before benchmark references are read.
This scorer intentionally compares only three arms:

* r2_baseline
* structural_anchor
* zeroed_control
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import score_no_llm_benchmark as base  # noqa: E402


DEFAULT_QUESTIONS = (
    ROOT
    / "data/corpus_package/type3/annual_reports_170_v1/questions/"
    / "type3_260_dev_v1/questions.jsonl"
)
DEFAULT_ORACLE = ROOT / "runs/phase_08/benchmark_decomposition_oracle.jsonl"
ARMS = ("r2_baseline", "structural_anchor", "zeroed_control")


class FastScorerError(RuntimeError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return base.load_jsonl(path)


def summary(values: list[float]) -> dict[str, Any]:
    total = sum(values)
    return {
        "count": len(values),
        "average_score": round(total / len(values), 6) if values else 0.0,
        "at_least_0_5": sum(value >= 0.5 for value in values),
        "at_least_0_75": sum(value >= 0.75 for value in values),
        "at_least_0_9": sum(value >= 0.9 for value in values),
    }


def load_questions(path: Path) -> list[dict[str, str]]:
    rows = read_jsonl(path)
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        if set(row) != {"question_id", "question", "document_id"}:
            raise FastScorerError("sanitized question schema differs")
        question_id = str(row["question_id"])
        if question_id in seen:
            raise FastScorerError(f"duplicate question id: {question_id}")
        seen.add(question_id)
        result.append(
            {
                "question_id": question_id,
                "question": str(row["question"]),
                "document_id": str(row["document_id"]),
            }
        )
    if len(result) != 260:
        raise FastScorerError(f"expected 260 questions, got {len(result)}")
    return result


def load_predictions(
    path: Path,
    questions: list[dict[str, str]],
) -> dict[str, dict[str, str]]:
    expected = {row["question_id"]: row for row in questions}
    predictions: dict[str, dict[str, str]] = {}
    for row in read_jsonl(path):
        question_id = str(row.get("question_id") or "")
        if question_id not in expected or question_id in predictions:
            raise FastScorerError(f"invalid prediction id: {question_id}")
        question = expected[question_id]
        if (
            row.get("document_id") != question["document_id"]
            or row.get("question") != question["question"]
        ):
            raise FastScorerError(f"prediction/question mismatch: {question_id}")
        answers = row.get("answers")
        if not isinstance(answers, dict) or set(answers) != set(ARMS):
            raise FastScorerError(f"prediction arm set differs: {question_id}")
        values = {arm: str(answers[arm] or "") for arm in ARMS}
        if any(not value.strip() for value in values.values()):
            raise FastScorerError(f"empty arm answer: {question_id}")
        predictions[question_id] = values
    if set(predictions) != set(expected):
        raise FastScorerError("prediction question set differs")
    return predictions


def selected_type3_gold(
    oracle_path: Path,
    benchmark_root: Path,
    questions: list[dict[str, str]],
) -> list[dict[str, Any]]:
    oracle_rows = [
        row
        for row in read_jsonl(oracle_path)
        if row.get("source", {}).get("benchmark_type") == "3-1"
    ]
    gold = base.load_selected_gold(oracle_rows, benchmark_root)
    if len(gold) != len(questions):
        raise FastScorerError("gold/question count differs")
    gold_by_id = {
        f"benchmark:{row['type']}:{row['uid']}": row
        for row in gold
    }
    ordered: list[dict[str, Any]] = []
    for question in questions:
        gold_row = gold_by_id.get(question["question_id"])
        if gold_row is None or gold_row["question"] != question["question"]:
            raise FastScorerError(
                f"gold question mismatch: {question['question_id']}"
            )
        ordered.append(gold_row)
    return ordered


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--oracle", type=Path, default=DEFAULT_ORACLE)
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=base.DEFAULT_BENCHMARK_ROOT,
    )
    parser.add_argument("--evaluator", type=Path, default=base.DEFAULT_EVALUATOR)
    parser.add_argument("--model", type=Path, default=base.DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    predictions_path = args.predictions.resolve()
    output = args.out_dir.resolve()
    staging = output.parent / f".{output.name}.staging"
    if output.exists() or staging.exists():
        raise FastScorerError("scoring output or staging already exists")

    # Freeze generator artifacts before benchmark references become available.
    questions = load_questions(args.questions.resolve())
    predictions = load_predictions(predictions_path, questions)
    frozen_prediction_sha256 = sha256_file(predictions_path)

    gold = selected_type3_gold(
        args.oracle.resolve(),
        args.benchmark_root.resolve(),
        questions,
    )
    evaluator = base.load_public_evaluator(args.evaluator.resolve())

    embedding_inputs: list[str] = []
    for row in gold:
        embedding_inputs.extend(str(value) for value in row["references"])
    for question in questions:
        values = predictions[question["question_id"]]
        embedding_inputs.extend(values[arm] for arm in ARMS)
    vectors, embedding = base.embed_texts(
        embedding_inputs,
        args.model.resolve(),
        args.device,
        args.batch_size,
    )

    details_by_arm: dict[str, list[dict[str, Any]]] = {
        arm: [] for arm in ARMS
    }
    scores_by_arm: dict[str, list[float]] = {arm: [] for arm in ARMS}
    cluster_values: dict[str, dict[tuple[str, str], list[float]]] = {
        arm: defaultdict(list) for arm in ARMS
    }
    gold_by_id = {
        f"benchmark:{row['type']}:{row['uid']}": row for row in gold
    }

    for question in questions:
        question_id = question["question_id"]
        gold_row = gold_by_id[question_id]
        for arm in ARMS:
            answer = predictions[question_id][arm]
            similarity, reference = base.bge_max_similarity(
                answer,
                gold_row["references"],
                vectors,
            )
            score = base.score_from_similarity(
                gold_row,
                answer,
                similarity,
                reference,
                evaluator,
            )
            numeric_score = float(score["score"])
            scores_by_arm[arm].append(numeric_score)
            cluster_values[arm][
                (question["document_id"], question["question"])
            ].append(numeric_score)
            details_by_arm[arm].append(
                {
                    "question_id": question_id,
                    "document_id": question["document_id"],
                    "question": question["question"],
                    "answer": answer,
                    "score": score,
                }
            )

    reports: dict[str, dict[str, Any]] = {}
    cluster_means: dict[str, dict[tuple[str, str], float]] = {}
    for arm in ARMS:
        cluster_means[arm] = {
            key: sum(values) / len(values)
            for key, values in cluster_values[arm].items()
        }
        reports[arm] = {
            "benchmark_weighted": summary(scores_by_arm[arm]),
            "semantic_task_macro": summary(
                list(cluster_means[arm].values())
            ),
        }

    baseline_details = {
        row["question_id"]: row for row in details_by_arm["r2_baseline"]
    }
    comparisons: dict[str, Any] = {}
    paired_rows: list[dict[str, Any]] = []
    for arm in ("structural_anchor", "zeroed_control"):
        wins = ties = losses = 0
        deltas: list[float] = []
        for row in details_by_arm[arm]:
            baseline = baseline_details[row["question_id"]]
            delta = float(row["score"]["score"]) - float(
                baseline["score"]["score"]
            )
            deltas.append(delta)
            if delta > 1e-12:
                wins += 1
            elif delta < -1e-12:
                losses += 1
            else:
                ties += 1
            paired_rows.append(
                {
                    "question_id": row["question_id"],
                    "arm": arm,
                    "baseline_score": baseline["score"]["score"],
                    "arm_score": row["score"]["score"],
                    "delta": round(delta, 12),
                }
            )
        macro_deltas = [
            cluster_means[arm][key] - cluster_means["r2_baseline"][key]
            for key in cluster_means["r2_baseline"]
        ]
        comparisons[arm] = {
            "benchmark_weighted_delta": round(
                sum(deltas) / len(deltas), 6
            ),
            "semantic_task_macro_delta": round(
                sum(macro_deltas) / len(macro_deltas), 6
            ),
            "wins": wins,
            "ties": ties,
            "losses": losses,
        }

    if sha256_file(predictions_path) != frozen_prediction_sha256:
        raise FastScorerError("generator predictions changed during scoring")

    staging.mkdir(parents=True)
    try:
        for arm in ARMS:
            base.write_jsonl(
                staging / f"{arm}_score_details.jsonl",
                details_by_arm[arm],
            )
        base.write_jsonl(staging / "paired_deltas.jsonl", paired_rows)
        result = {
            "schema_version": "finglmqa.type3.phase51.b2_fast_score.v1",
            "question_count": len(questions),
            "arms": reports,
            "comparisons_vs_r2": comparisons,
            "embedding": embedding,
            "boundary": {
                "generator_predictions_loaded_and_hashed_before_gold": True,
                "generator_predictions_sha256": frozen_prediction_sha256,
                "generator_predictions_unchanged_after_scoring": True,
            },
        }
        base.write_json(staging / "score_summary.json", result)
        staging.rename(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(
        json.dumps(
            {
                "output": str(output),
                "structural_delta": comparisons["structural_anchor"][
                    "benchmark_weighted_delta"
                ],
                "zeroed_delta": comparisons["zeroed_control"][
                    "benchmark_weighted_delta"
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
