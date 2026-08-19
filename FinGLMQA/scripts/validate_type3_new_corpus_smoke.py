#!/usr/bin/env python3
"""Validate the local-only Phase 6 new-company portability experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from finglmqa.type3_corpus_profile import (  # noqa: E402
    load_corpus_profile,
    source_snapshot,
)
from finglmqa.type3_evidence_fusion import (  # noqa: E402
    canonical_json_bytes,
    semantic_sha256,
)


CORPUS_ID = "annual_reports_nano5_2021_2023_v1"
PHASE6 = (
    ROOT
    / "runs/type3_a2rag_tabgr_experiment_v1"
    / CORPUS_ID
    / "phase_6"
)
PACKAGE = ROOT / "data/corpus_package/type3" / CORPUS_ID
NANO_GOLD = Path("/home/coder/demo/NANO-Finbenchmark/data/fr_20_ragas_seed.jsonl")
DEFAULT_OUT = PHASE6 / "evaluation"
LIST_IDS = ("FR-16", "FR-17", "FR-18", "FR-19", "FR-20")
MASK = "[未经授权数值]"


class Phase6ValidationError(ValueError):
    """Raised when a portability/safety invariant differs."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase6ValidationError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise Phase6ValidationError(f"blank JSONL row: {path}:{line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise Phase6ValidationError(f"expected JSON object: {path}:{line_number}")
            rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def write_markdown(path: Path, report: Mapping[str, Any]) -> None:
    diagnostic = report["local_list_diagnostic"]
    lines = [
        "# Phase 6 新公司迁移验证",
        "",
        f"- 状态：`{report['status']}`",
        f"- 语料：`{report['corpus_id']}`，15 份年报、5 家新公司、2021–2023。",
        "- 原始 Markdown：未修改；本轮只读取冻结引用并验证 SHA-256。",
        "- 许可：未核验，仅限本地实验，不得重新分发。",
        "",
        "## 结构与安全 Gate",
        "",
    ]
    for gate, passed in report["gates"].items():
        lines.append(f"- `{gate}`: {'PASS' if passed else 'FAIL'}")
    lines.extend(
        [
            "",
            "## 5 题本地列表诊断",
            "",
            f"- Union evidence/item recall：{diagnostic['union_item_recall']:.6f} "
            f"({diagnostic['union_items_found']}/{diagnostic['expected_items']})。",
            f"- Compact final/item recall：{diagnostic['compact_item_recall']:.6f} "
            f"({diagnostic['compact_items_found']}/{diagnostic['expected_items']})。",
            f"- Compact 完整覆盖：{diagnostic['compact_full_question_count']}/5。",
            "",
            "该 5 题来自本地 NANO 数据，仅用于 scorer-only 诊断，样本过小且许可未核验，"
            "不能作为对外质量结论。结果说明索引迁移和检索已有基础，但 compact selector "
            "会丢失部分已召回列表事实，表格列表检索也仍是下一阶段短板。",
            "",
            "## 结论",
            "",
            "新公司语料的扫描、解析、A2RAG、TabGR、双路融合和 compact 安全输出均可复现；"
            "基础设施迁移通过。由于本地列表诊断未达到推广标准，本轮不据此宣称新公司 QA "
            "质量通过，也不修改默认服务配置。",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _compact(value: str) -> str:
    return re.sub(r"\s+", "", value).lower()


def expected_items(row: Mapping[str, Any]) -> list[str]:
    for component in row.get("eval_spec", {}).get("components", []):
        if component.get("type") == "list_f1":
            items = component.get("expected_items")
            if isinstance(items, list) and all(isinstance(item, str) for item in items):
                return [str(item) for item in items]
    raise Phase6ValidationError(f"local list gold lacks expected_items: {row.get('id')}")


def item_hits(answer: str, items: list[str]) -> list[str]:
    projected = _compact(answer)
    return [item for item in items if _compact(item) in projected]


def _verify_pair(path: Path) -> dict[str, Any]:
    first = path / "fresh_process_1"
    second = path / "fresh_process_2"
    files = ("answers.jsonl", "semantic_traces.jsonl", "run_manifest.json")
    hashes: dict[str, str] = {}
    for name in files:
        first_path = first / name
        second_path = second / name
        first_hash = sha256_file(first_path)
        second_hash = sha256_file(second_path)
        if first_hash != second_hash:
            raise Phase6ValidationError(f"fresh process drift: {path.name}/{name}")
        hashes[name] = first_hash
    return {"byte_identical": True, "hashes": hashes}


def _verify_union(
    name: str,
    *,
    question_count: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = PHASE6 / name
    repeat = _verify_pair(root)
    manifest = read_json(root / "fresh_process_1/run_manifest.json")
    answers_path = root / "fresh_process_1/answers.jsonl"
    rows = read_jsonl(answers_path)
    if (
        manifest.get("schema_version") != "finglmqa.type3.a2rag_tabgr.run_manifest.v1"
        or manifest.get("corpus_id") != CORPUS_ID
        or manifest.get("arm") != "union"
        or manifest.get("question_count") != question_count
        or manifest.get("artifacts", {}).get("answers.jsonl") != sha256_file(answers_path)
        or manifest.get("safety", {}).get("cross_document_evidence") != 0
        or manifest.get("safety", {}).get("unsupported_numeric_literals") != 0
        or len(rows) != question_count
    ):
        raise Phase6ValidationError(f"union run contract differs: {name}")
    return rows, {"repeatability": repeat, "manifest": manifest}


def _verify_compact(
    name: str,
    *,
    union_rows: list[dict[str, Any]],
    corpus: Mapping[str, Any],
    question_count: int,
    validator: Draft202012Validator,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = PHASE6 / name
    repeat = _verify_pair(root)
    manifest = read_json(root / "fresh_process_1/run_manifest.json")
    answers_path = root / "fresh_process_1/answers.jsonl"
    rows = read_jsonl(answers_path)
    if (
        manifest.get("schema_version")
        != "finglmqa.type3.phase5.compact_run_manifest.v2"
        or manifest.get("corpus_id") != CORPUS_ID
        or manifest.get("question_count") != question_count
        or manifest.get("configuration", {}).get("legacy_baseline_enabled") is not False
        or manifest.get("artifacts", {}).get("answers.jsonl") != sha256_file(answers_path)
        or manifest.get("safety", {}).get("nonempty_answers") != question_count
        or manifest.get("safety", {}).get("masked_placeholders") != 0
        or manifest.get("safety", {}).get("cross_document_citations") != 0
        or len(rows) != question_count
    ):
        raise Phase6ValidationError(f"compact run contract differs: {name}")
    corpus_documents = {
        row["document_id"]: row for row in corpus["documents"]
    }
    union_by_id = {row["question_id"]: row for row in union_rows}
    for row in rows:
        errors = list(validator.iter_errors(row))
        if errors:
            raise Phase6ValidationError(
                f"compact output schema differs: {name}/{row.get('question_id')}"
            )
        if not row["answer_safe_text"].strip() or MASK in row["answer_safe_text"]:
            raise Phase6ValidationError("compact output is empty or masked")
        document = corpus_documents[row["document_id"]]
        selected = row["selected_candidate_ids"]
        if len(selected) != 1:
            raise Phase6ValidationError("new-corpus compact output must select one candidate")
        packet = union_by_id[row["question_id"]]
        evidence = {
            value["candidate_id"]: value for value in packet["evidence"]
        }[selected[0]]
        if evidence["document_id"] != row["document_id"]:
            raise Phase6ValidationError("selected evidence crosses document boundary")
        if row["answer_safe_text"] not in evidence["answer_safe_text"].replace(MASK, ""):
            raise Phase6ValidationError("compact answer does not backlink to selected evidence")
        selected_citation = [
            value
            for value in row["citations"]
            if value["citation_kind"] == "phase5_a2rag_text"
        ]
        if (
            len(selected_citation) != 1
            or selected_citation[0]["source_sha256"] != document["source_sha256"]
        ):
            raise Phase6ValidationError("compact citation source hash differs")
    return rows, {"repeatability": repeat, "manifest": manifest}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nano-gold", type=Path, default=NANO_GOLD)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    corpus = load_corpus_profile(PACKAGE / "corpus_manifest.json")
    before = source_snapshot(corpus, workspace_root=ROOT)

    preparation = read_json(PHASE6 / "profile_inputs/preparation_manifest.json")
    scan = read_json(PHASE6 / "source_scan_v1/corpus_scan_report.json")
    parse = read_json(PHASE6 / "source_parse_v1/phase_03_run_report.json")
    a2rag = read_json(PHASE6 / "a2rag/build_report.json")
    a2repeat = read_json(PHASE6 / "a2rag/repeatability_report.json")
    tabgr = read_json(PHASE6 / "tabgr/build_report.json")
    schema = read_json(
        ROOT / "data/schemas/type3/phase5_compact_answer_v2.schema.json"
    )
    validator = Draft202012Validator(schema)

    list_union, list_union_audit = _verify_union(
        "list5_union",
        question_count=5,
    )
    list_compact, list_compact_audit = _verify_compact(
        "list5_compact_v2_frozen",
        union_rows=list_union,
        corpus=corpus,
        question_count=5,
        validator=validator,
    )
    structural_union, structural_union_audit = _verify_union(
        "structural_union",
        question_count=60,
    )
    structural_compact, structural_compact_audit = _verify_compact(
        "structural_compact_v2_frozen",
        union_rows=structural_union,
        corpus=corpus,
        question_count=60,
        validator=validator,
    )

    gold = {
        row["id"]: row
        for row in read_jsonl(args.nano_gold.resolve())
        if row.get("id") in LIST_IDS
    }
    if set(gold) != set(LIST_IDS):
        raise Phase6ValidationError("local NANO diagnostic case set differs")
    union_by_id = {
        row["question_id"].rsplit(":", 1)[-1]: row for row in list_union
    }
    compact_by_id = {
        row["question_id"].rsplit(":", 1)[-1]: row for row in list_compact
    }
    diagnostics: list[dict[str, Any]] = []
    expected_total = union_found = compact_found = compact_full = 0
    for case_id in LIST_IDS:
        items = expected_items(gold[case_id])
        union_hits = item_hits(union_by_id[case_id]["answer_safe_text"], items)
        compact_hits = item_hits(compact_by_id[case_id]["answer_safe_text"], items)
        expected_total += len(items)
        union_found += len(union_hits)
        compact_found += len(compact_hits)
        compact_full += len(compact_hits) == len(items)
        diagnostics.append(
            {
                "case_id": case_id,
                "expected_item_count": len(items),
                "union_item_hits": len(union_hits),
                "compact_item_hits": len(compact_hits),
                "compact_full_coverage": len(compact_hits) == len(items),
            }
        )

    after = source_snapshot(corpus, workspace_root=ROOT)
    gates = {
        "local_only_license_guard": (
            preparation.get("local_only") is True
            and preparation.get("license_status") == "unverified_do_not_redistribute"
        ),
        "new_company_stock_codes_disjoint": (
            preparation.get("existing_stock_code_overlap") == []
        ),
        "source_markdown_unchanged": before == after,
        "scan_15_of_15_valid": (
            scan.get("counts", {}).get("valid_documents") == 15
            and scan.get("counts", {}).get("invalid_documents") == 0
        ),
        "parse_15_of_15_and_zero_malformed": (
            parse.get("summary", {}).get("documents_parsed") == 15
            and parse.get("summary", {}).get("documents_failed") == 0
            and parse.get("summary", {}).get("malformed_or_unsupported_tables") == 0
        ),
        "a2rag_build_and_repeatability_passed": (
            a2rag.get("passed") is True
            and a2rag.get("source_hashes_unchanged") is True
            and a2repeat.get("passed") is True
            and a2repeat.get("failed_checks") == []
        ),
        "tabgr_build_passed": (
            tabgr.get("status") == "passed"
            and tabgr.get("source_unchanged") is True
            and tabgr.get("stop_conditions") == []
            and tabgr.get("counts", {}).get("ready_tables") == 4185
        ),
        "list5_union_and_compact_repeatable": (
            list_union_audit["repeatability"]["byte_identical"]
            and list_compact_audit["repeatability"]["byte_identical"]
        ),
        "structural_union_and_compact_repeatable": (
            structural_union_audit["repeatability"]["byte_identical"]
            and structural_compact_audit["repeatability"]["byte_identical"]
        ),
        "all_65_compact_answers_nonempty_safe": (
            len(list_compact) + len(structural_compact) == 65
            and all(
                row["answer_safe_text"].strip() and MASK not in row["answer_safe_text"]
                for row in [*list_compact, *structural_compact]
            )
        ),
    }
    if not all(gates.values()):
        raise Phase6ValidationError(
            f"Phase 6 infrastructure gate failed: "
            f"{[key for key, value in gates.items() if not value]!r}"
        )

    report = {
        "schema_version": "finglmqa.type3.phase6.new_corpus_validation.v1",
        "status": "conditional_pass_infrastructure_quality_not_promoted",
        "corpus_id": CORPUS_ID,
        "corpus_profile_sha256": corpus["profile_sha256"],
        "document_count": corpus["document_count"],
        "stock_codes": preparation["stock_codes"],
        "report_years": preparation["report_years"],
        "license_scope": {
            "local_only": True,
            "license_status": "unverified_do_not_redistribute",
            "redistribution_allowed": False,
        },
        "gates": gates,
        "source_freeze": {
            "source_hashes_sha256_before": semantic_sha256(before),
            "source_hashes_sha256_after": semantic_sha256(after),
            "source_unchanged": before == after,
        },
        "pipeline_counts": {
            "text_blocks": parse["summary"]["text_blocks"],
            "table_blocks": parse["summary"]["table_blocks"],
            "a2rag_atoms": a2rag["atom_count"],
            "a2rag_units": a2rag["unit_count"],
            "tabgr_ready_tables": tabgr["counts"]["ready_tables"],
            "tabgr_row_evidence": tabgr["counts"]["row_evidence"],
        },
        "run_artifacts": {
            "list5_union": list_union_audit,
            "list5_compact_v2_frozen": list_compact_audit,
            "structural_union": structural_union_audit,
            "structural_compact_v2_frozen": structural_compact_audit,
        },
        "local_list_diagnostic": {
            "scorer_only_gold_path": args.nano_gold.resolve().as_posix(),
            "scorer_only_gold_sha256": sha256_file(args.nano_gold.resolve()),
            "question_count": 5,
            "expected_items": expected_total,
            "union_items_found": union_found,
            "union_item_recall": round(union_found / expected_total, 6),
            "compact_items_found": compact_found,
            "compact_item_recall": round(compact_found / expected_total, 6),
            "compact_full_question_count": compact_full,
            "per_case": diagnostics,
            "promotion_gate_passed": False,
            "scope_note": (
                "Tiny local-only diagnostic; license unverified; not a public or "
                "statistically sufficient quality evaluation."
            ),
        },
        "decision": {
            "infrastructure_portability_passed": True,
            "new_company_quality_promoted": False,
            "default_service_modified": False,
            "next_focus": [
                "TabGR list/category retrieval for table-backed Type 3 questions",
                "compact selector coverage preservation over already-retrieved list facts",
                "held-out licensed multi-company evaluation before promotion",
            ],
        },
    }
    out_dir = args.out_dir.resolve()
    write_json(out_dir / "new_corpus_validation_report.json", report)
    write_markdown(out_dir / "new_corpus_validation_report.md", report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "infrastructure_passed": True,
                "compact_item_recall": report["local_list_diagnostic"][
                    "compact_item_recall"
                ],
                "output_dir": out_dir.as_posix(),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
