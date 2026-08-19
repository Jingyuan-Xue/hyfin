#!/usr/bin/env python3
"""Create deterministic Phase 8 Gate 0 manifests and planning gold fixtures."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

EXTERNAL_ROOTS = (
    (Path("/home/coder/demo/output/finglm_mineru_markdown_1000_profile"), "source_profile_output"),
    (Path("/home/coder/demo/BIG-Finbenchmark"), "benchmark_source"),
    (Path("/home/coder/demo/models"), "models"),
    (Path("/home/coder/demo/A2RAG"), "a2rag_source"),
)

from finglmqa.contracts import canonical_json_bytes, semantic_sha256, validate_pattern_registry  # noqa: E402

SCHEMA_GATE0 = "finglmqa.phase8.gate0_report.v1"
SCHEMA_IMMUTABLE = "finglmqa.phase8.immutable_inputs_manifest.v1"
SCHEMA_ORACLE = "finglmqa.phase8.benchmark_decomposition_oracle.v1"
SCHEMA_GENERAL = "finglmqa.phase8.general_decomposition_gold.v1"

TYPE_COUNTS = {"1": 312, "1-2": 138, "2-1": 285, "2-2": 8, "3-1": 260}
SUPPORTED_TYPE1_UIDS = ["B:372", "C:734", "A:97", "pre:97", "C:617", "B:1575", "C:568", "B:1964", "C:864"]
SUPPORTED_FORMULA_UIDS = ["B:503", "B:775", "B:891", "B:1090", "pre:4160", "pre:4481"]
SELECTION_MISMATCH_UIDS = {
    "A:636", "pre:681", "pre:1647", "pre:1343", "pre:4025", "pre:3050",
}
METADATA_KEYWORDS = frozenset({
    "证券代码", "股票代码", "证券编号", "证券简称", "股票简称", "企业名称", "公司名称", "中文名称",
    "外文名称", "英文名称", "外文名称缩写", "英文简称", "办公地址", "注册地址", "注册地",
    "公司网址", "官方网址", "网站地址", "法定代表人", "法人代表", "电子信箱", "电子邮箱", "公司邮箱",
})


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


def write_json(path: Path, payload: Any) -> None:
    atomic_write(path, canonical_json_bytes(payload))


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    data = b"".join(canonical_json_bytes(row) for row in rows)
    atomic_write(path, data)


def portable(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        for external_root, label in EXTERNAL_ROOTS:
            try:
                relative = resolved.relative_to(external_root.resolve())
            except ValueError:
                continue
            return f"external:{label}/{relative.as_posix()}"
        return f"external:unclassified/{resolved.as_posix().lstrip('/')}"


def file_record(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": portable(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256_file(path),
    }


def symlink_record(path: Path) -> dict[str, Any]:
    stat = path.lstat()
    target = os.readlink(path)
    resolved = path.resolve()
    result = {
        "path": path.relative_to(ROOT).as_posix(),
        "kind": "symlink",
        "lstat_mtime_ns": stat.st_mtime_ns,
        "readlink": target,
        "resolved_name": resolved.name,
        "target_exists": resolved.exists(),
    }
    if resolved.is_file():
        result["target"] = file_record(resolved)
    return result


def collect_files(paths: Iterable[Path]) -> list[Path]:
    result: set[Path] = set()
    for path in paths:
        if path.is_symlink() or not path.exists():
            continue
        if path.is_file():
            result.add(path)
        elif path.is_dir():
            result.update(child for child in path.rglob("*") if child.is_file() and "__pycache__" not in child.parts)
    return sorted(result, key=lambda item: portable(item))


def command_output(command: list[str], cwd: Path | None = None) -> str:
    return subprocess.run(command, cwd=cwd, check=True, text=True, capture_output=True).stdout.strip()


def build_immutable_manifest(selected_rows: list[dict[str, Any]]) -> dict[str, Any]:
    answer_paths = {Path(row["question_file"]).with_name("answers.jsonl") for row in selected_rows}
    question_paths = {Path(row["question_file"]) for row in selected_rows}
    bge_snapshot = Path("/home/coder/demo/models/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181")
    bge_configs = [
        bge_snapshot / name for name in (
            "config.json", "modules.json", "sentence_bert_config.json", "config_sentence_transformers.json",
            "tokenizer_config.json", "special_tokens_map.json",
        )
    ]
    a2rag_sources = [
        Path("/home/coder/demo/A2RAG/src/a2rag/config/settings.py"),
        Path("/home/coder/demo/A2RAG/src/a2rag/providers/local_embedding.py"),
        Path("/home/coder/demo/A2RAG/src/a2rag/models/embeddings.py"),
    ]
    inputs = [
        ROOT / "data/facts/financial_facts.jsonl",
        ROOT / "data/facts/financial_facts.duckdb",
        ROOT / "data/schemas/financial_facts.schema.json",
        ROOT / "data/indexes/canonical_metric_candidates.jsonl",
        ROOT / "runs/phase_06/reports/candidate_decisions.jsonl",
        ROOT / "runs/phase_06/reports/conflict_groups.jsonl",
        ROOT / "runs/phase_06/phase_06_run_report.json",
        ROOT / "runs/phase_06/repeatability_report.json",
        ROOT / "data/corpus_package/evidence_chunks.jsonl",
        ROOT / "data/corpus_package/company_year_index.jsonl",
        ROOT / "data/corpus_package/corpus_manifest.json",
        ROOT / "data/schemas/evidence_chunks.schema.json",
        ROOT / "data/indexes/a2rag_index",
        ROOT / "runs/phase_07/build_report.json",
        ROOT / "runs/phase_07/repeatability_report.json",
        ROOT / "runs/phase_07/reports/table_exclusion_audit.json",
        ROOT / "runs/phase_07/reports/table_exclusion_classification.jsonl",
        ROOT / "scripts/build_a2rag_text_index.py",
        ROOT / "scripts/query_type3_evidence.py",
        ROOT / "scripts/validate_phase_07.py",
        ROOT / "tests/test_phase07_alignment.py",
        ROOT / "tests/test_phase07_query_policy.py",
        ROOT / "src/config/metric_aliases.json",
        ROOT / "src/config/unit_rules.json",
        ROOT / "refs/source_profile/01_selection/selected_questions.jsonl",
        ROOT / "refs/source_profile/01_selection/question_report_links.jsonl",
        (ROOT / "refs/source_markdown").resolve(),
        ROOT / "env/finglmqa.local.env",
        *question_paths,
        *answer_paths,
        *bge_configs,
        *a2rag_sources,
    ]
    files = collect_files(inputs)
    entries = [file_record(path) for path in files]
    symlinks = [
        symlink_record(ROOT / name)
        for name in ("refs/a2rag_runtime", "refs/qwen_model", "refs/tabgr_runtime", "refs/source_markdown")
    ]
    workspace_freeze = command_output(["uv", "pip", "freeze", "--python", str(ROOT / ".venv/bin/python")]).splitlines()
    a2rag_freeze = command_output(["uv", "pip", "freeze", "--python", str(ROOT / "refs/a2rag_runtime/.venv/bin/python")]).splitlines()
    a2rag_commit = command_output(["git", "rev-parse", "HEAD"], Path("/home/coder/demo/A2RAG"))
    a2rag_status = command_output(["git", "status", "--short"], Path("/home/coder/demo/A2RAG"))
    manifest = {
        "schema_version": SCHEMA_IMMUTABLE,
        "entries": entries,
        "symlinks": symlinks,
        "runtime": {
            "workspace_python": command_output([str(ROOT / ".venv/bin/python"), "--version"]),
            "a2rag_python": command_output([str(ROOT / "refs/a2rag_runtime/.venv/bin/python"), "--version"]),
            "workspace_pip_freeze_sha256": semantic_sha256(sorted(workspace_freeze)),
            "a2rag_pip_freeze_sha256": semantic_sha256(sorted(a2rag_freeze)),
            "a2rag_git_commit": a2rag_commit,
            "a2rag_dirty_status_sha256": semantic_sha256(a2rag_status.splitlines()),
            "a2rag_dirty": bool(a2rag_status),
            "bge_snapshot": "5617a9f61b028005a4858fdac845db406aefb181",
        },
    }
    manifest["manifest_semantic_sha256"] = semantic_sha256(manifest)
    return manifest


def load_answers(selected_rows: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    answers: dict[str, dict[str, Any]] = {}
    hashes: dict[str, str] = {}
    for path in sorted({Path(row["question_file"]).with_name("answers.jsonl") for row in selected_rows}):
        hashes[path.as_posix()] = sha256_file(path)
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            answers[f"{row['type']}:{row['uid']}"] = row
    return answers, hashes


def split_concerns(key_word: str, expected: int) -> list[str]:
    values = [value.strip() for value in re.split(r"[、,，]|和|及", key_word or "") if value.strip()]
    if expected == 2 and len(values) < 2:
        values.extend(f"concern_{index + 1}" for index in range(len(values), 2))
    return values[:expected] if expected else values


def build_benchmark_oracle(
    selected_rows: list[dict[str, Any]],
    answers: dict[str, dict[str, Any]],
    registry_hash: str,
) -> list[dict[str, Any]]:
    document_rows = [json.loads(line) for line in (ROOT / "data/corpus_package/company_year_index.jsonl").read_text(encoding="utf-8").splitlines()]
    aliases = sorted({alias for row in document_rows for alias in row["aliases"]}, key=lambda value: (-len(value), value))
    dedup: dict[str, list[str]] = defaultdict(list)
    for row in selected_rows:
        dedup[semantic_sha256(row["question"])].append(row["uid"])
    output: list[dict[str, Any]] = []
    for ordinal, row in enumerate(selected_rows):
        answer = answers[f"{row['type']}:{row['uid']}"]
        prompt = answer.get("prompt", {})
        benchmark_type = row["type"]
        report_uid = row["mapped_report_uids"][0]
        stock_code = report_uid.split("_", 1)[0].removeprefix("A")
        explicit_years = [int(value) for value in re.findall(r"(?<!\d)(20\d{2})(?!\d)", row["question"])]
        explicit_years = list(dict.fromkeys(explicit_years))
        if benchmark_type == "1":
            concerns = split_concerns(prompt.get("key_word", ""), 1)
            operation = (
                "metadata_lookup"
                if (concerns and concerns[0] in METADATA_KEYWORDS)
                or any(keyword in row["question"] for keyword in METADATA_KEYWORDS)
                else "fact_lookup"
            )
            pattern_id, signatures = "single_node", [("fact", operation)]
        elif benchmark_type == "1-2":
            concerns = split_concerns(prompt.get("key_word", ""), 2)
            pattern_id, signatures = "parallel_concerns", [("fact", "fact_lookup")] * 2
        elif benchmark_type == "2-1":
            concerns = [prompt.get("公式") or prompt.get("key_word") or "formula"]
            pattern_id, signatures = "single_node", [("formula", "formula_compute")]
        elif benchmark_type == "2-2":
            if len(explicit_years) == 2 and "至" in row["question"]:
                periods = list(range(min(explicit_years), max(explicit_years) + 1))
            elif len(explicit_years) == 1 and re.search(r"上(?:一)?年|上一年度", row["question"]):
                periods = [explicit_years[0] - 1, explicit_years[0]]
            else:
                periods = explicit_years or [int(prompt.get("year", 0))]
            concerns = [prompt.get("key_word") or "法定代表人"]
            pattern_id, signatures = "period_compare", [("fact", "metadata_lookup")] * len(periods)
        else:
            concerns = ["narrative"]
            pattern_id, signatures = "single_node", [("evidence", "document_retrieval")]
        alias_hit = next((alias for alias in aliases if alias and alias in row["question"]), None)
        mapping_consistency = "selection_mismatch" if row["uid"] in SELECTION_MISMATCH_UIDS else ("consistent" if alias_hit else "alias_gap")
        executable = row["uid"] in SUPPORTED_TYPE1_UIDS or row["uid"] in SUPPORTED_FORMULA_UIDS or benchmark_type == "3-1"
        output.append({
            "schema_version": SCHEMA_ORACLE,
            "oracle_version": "1.0.0",
            "case_id": f"benchmark:{benchmark_type}:{row['uid']}",
            "source": {
                "uid": row["uid"], "benchmark_type": benchmark_type, "selected_ordinal": ordinal,
                "question": row["question"], "question_sha256": hashlib.sha256(row["question"].encode()).hexdigest(),
                "mapped_report_uids": row["mapped_report_uids"],
            },
            "dedup": {
                "normalized_question_sha256": semantic_sha256(row["question"]),
                "group_uids": dedup[semantic_sha256(row["question"])],
            },
            "offline_gold_hints": {
                "company_full": prompt.get("ent_name"), "company_short": prompt.get("ent_short_name"),
                "prompt_year": prompt.get("year"), "ordered_concerns": concerns,
                "formula": prompt.get("公式"), "expected_rendered_answer": prompt.get("prom_answer"),
                "test_only": True,
            },
            "expected_analysis": {
                "explicit_years": explicit_years, "ordered_concerns": concerns,
                "dynamic_target_dependency": False,
            },
            "expected_resolution": {
                "mapping_consistency": mapping_consistency,
                "expected_stock_code": None if mapping_consistency == "selection_mismatch" else stock_code,
                "expected_report_uid": None if mapping_consistency == "selection_mismatch" else report_uid,
            },
            "expected_composition": {
                "pattern_id": pattern_id, "pattern_version": "1.0.0",
                "ordered_subplan_signatures": [
                    {"ordinal": index, "backend": backend, "operation": operation, "required": True}
                    for index, (backend, operation) in enumerate(signatures)
                ],
                "minimum_usable_results": 1, "optional_subplans": [],
                "registry_semantic_sha256": registry_hash,
            },
            "capability": {
                "executable_in_phase8": executable and mapping_consistency != "selection_mismatch",
                "fixture_pin": "type1_exact" if row["uid"] in SUPPORTED_TYPE1_UIDS else ("formula_exact" if row["uid"] in SUPPORTED_FORMULA_UIDS else None),
            },
        })
    return output


def general_case(case_id: str, category: str, question: str, pattern: str | None, signatures: list[str], *, layer: str = "production_snapshot", failure: str | None = None) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_GENERAL,
        "case_id": case_id,
        "category": category,
        "fixture_layer": layer,
        "question": question,
        "question_sha256": hashlib.sha256(question.encode()).hexdigest(),
        "expected": {
            "pattern_id": pattern,
            "ordered_subplan_signatures": signatures,
            "failure_code": failure,
            "backend_call_count": 0 if pattern is None else None,
        },
        "review_status": "manually_frozen",
    }


def build_general_gold() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    add = rows.append
    multi = "multi_intent"
    add(general_case("G01", multi, "飞亚达2019年营业收入是多少，并解释变动原因？", "single_document_bundle", ["fact:fact_lookup", "evidence:document_retrieval"]))
    add(general_case("G02", multi, "东阿阿胶2019年营业收入增长率是多少，并解释下降原因？", "single_document_bundle", ["formula:formula_compute", "evidence:document_retrieval"]))
    add(general_case("G03", multi, "飞亚达2019年营业收入、归母净利润是多少，并概述经营情况？", "single_document_bundle", ["fact:fact_lookup", "fact:fact_lookup", "evidence:document_retrieval"]))
    add(general_case("G04", multi, "比较飞亚达和东阿阿胶2019年营业收入，并分别解释原因。", "entity_section_bundle", ["fact", "evidence", "fact", "evidence"]))
    add(general_case("G05", multi, "比较示例公司2019和2020年营业收入，并分别解释各年原因。", "period_section_bundle", ["fact", "evidence", "fact", "evidence"], layer="synthetic_resolver_overlay"))
    partial = "multi_company_partial_resolution"
    add(general_case("G06", partial, "比较飞亚达、东阿阿胶和未收录公司的营业收入。", "entity_compare", ["fact:ready", "fact:ready", "fact:blocked"]))
    add(general_case("G07", partial, "比较飞亚达和回天新材2019年营业收入。", "entity_compare", ["fact:ready", "fact:blocked"], layer="synthetic_resolver_overlay"))
    add(general_case("G08", partial, "比较飞亚达和示例歧义公司的营业收入。", "entity_compare", ["fact:ready", "fact:blocked"], layer="synthetic_resolver_overlay"))
    add(general_case("G09", partial, "列出飞亚达和未收录公司的营业收入。", "entity_list", ["fact:ready", "fact:blocked"]))
    add(general_case("G10", partial, "比较飞亚达和未收录公司收入并分别解释原因。", "entity_section_bundle", ["fact:ready", "evidence:ready", "fact:blocked", "evidence:blocked"]))
    concerns = "multi_concern_formula"
    add(general_case("G11", concerns, "飞亚达2019年营业收入和归母净利润是多少？", "parallel_concerns", ["fact", "fact"]))
    add(general_case("G12", concerns, "飞亚达2019年营业收入及其增长率是多少？", "parallel_concerns", ["fact", "formula"]))
    add(general_case("G13", concerns, "东阿阿胶2019年营业收入、净资产、归母净利润增长率是多少？", "parallel_concerns", ["formula", "formula", "formula"]))
    add(general_case("G14", concerns, "一品红2020年营业收入和归母净利润增长率是多少？", "parallel_concerns", ["formula", "formula"]))
    add(general_case("G15", concerns, "飞亚达证券代码、2019年营业收入及增长率是多少？", "parallel_concerns", ["fact:metadata_lookup", "fact:fact_lookup", "formula"]))
    periods = "multi_period_formula"
    add(general_case("G16", periods, "根据飞亚达2019年报计算2018、2019年营业收入增长率。", "period_list", ["formula", "formula"]))
    add(general_case("G17", periods, "比较东阿阿胶2018、2019年归母净利润增长率。", "period_compare", ["formula", "formula"]))
    add(general_case("G18", periods, "根据一品红2020年报计算2019、2020年营业收入增长率。", "period_list", ["formula", "formula"]))
    add(general_case("G19", periods, "根据回天新材2020年报计算2019、2020年营业收入和归母净利润增长率。", "period_list", ["formula"] * 4))
    add(general_case("G20", periods, "根据朗迪集团2021年报计算2020、2021年营业收入、净资产、归母净利润增长率。", "period_list", ["formula"] * 6))
    dynamic = "dynamic_target_dependency"
    dynamic_questions = [
        "若飞亚达增长率为正则解释增长原因，否则分析风险。",
        "比较飞亚达和东阿阿胶收入，只解释胜者原因。",
        "找出飞亚达2017至2019年收入最高年份，再解释该年原因。",
        "找出2019年收入高于全体平均值的公司，再列出其证券代码。",
        "若飞亚达收入高于净资产则返回收入增长率，否则返回ROE。",
    ]
    for index, question in enumerate(dynamic_questions, 21):
        add(general_case(f"G{index:02d}", dynamic, question, None, [], failure="COMPOSITION_UNSUPPORTED"))
    ambiguity = "narrative_year_ambiguity"
    for index, topic in enumerate(("核心竞争力", "主要风险", "经营情况", "收入变动原因", "社会责任"), 26):
        add(general_case(f"G{index:02d}", ambiguity, f"示例多报告公司的{topic}是什么？", "single_node", ["evidence:blocked"], layer="synthetic_resolver_overlay", failure="RESOLVER_AMBIGUOUS"))
    mixed = "metadata_financial_mix"
    add(general_case("G31", mixed, "飞亚达证券代码和2019年营业收入是多少？", "parallel_concerns", ["fact:metadata_lookup", "fact:fact_lookup"]))
    add(general_case("G32", mixed, "东阿阿胶证券简称和2019年归母净利润是多少？", "parallel_concerns", ["fact:metadata_lookup", "fact:fact_lookup"]))
    add(general_case("G33", mixed, "东阿阿胶证券代码和2019年营业收入增长率是多少？", "parallel_concerns", ["fact:metadata_lookup", "formula"]))
    add(general_case("G34", mixed, "飞亚达、东阿阿胶的证券代码和2019年营业收入是多少？", "entity_list", ["fact"] * 4))
    add(general_case("G35", mixed, "飞亚达证券简称、2019年营业收入和总资产是多少？", "parallel_concerns", ["fact"] * 3))
    rank_reason = "corpus_rank_then_reason"
    rank_questions = [
        "2019年收入最高公司是谁，为什么？",
        "2019年总资产Top 3，并分别解释其风险。",
        "2019年收入增长率最低公司及下降原因是什么？",
        "按归母净利润排名，并解释第一名增长原因。",
        "找出ROE高于平均值的公司并总结其原因。",
    ]
    for index, question in enumerate(rank_questions, 36):
        add(general_case(f"G{index:02d}", rank_reason, question, None, [], failure="COMPOSITION_UNSUPPORTED"))
    assert len(rows) == 40
    return rows


def deterministic_projection(value: Any) -> Any:
    ignored = {"generated_at_utc", "elapsed_seconds", "machine_readable_report", "report"}
    if isinstance(value, dict):
        return {key: deterministic_projection(item) for key, item in sorted(value.items()) if key not in ignored}
    if isinstance(value, list):
        return [deterministic_projection(item) for item in value]
    return value


def main() -> int:
    run_dir = ROOT / "runs/phase_08"
    selected_path = ROOT / "refs/source_profile/01_selection/selected_questions.jsonl"
    selected_rows = [json.loads(line) for line in selected_path.read_text(encoding="utf-8").splitlines()]
    answers, answer_hashes = load_answers(selected_rows)
    registry_path = ROOT / "src/config/composition_patterns.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    validate_pattern_registry(registry)
    registry_hash = semantic_sha256(registry)
    registry_manifest = {
        "schema_version": "finglmqa.phase8.pattern_registry_manifest.v1",
        "registry_version": registry["registry_version"],
        "pattern_count": len(registry["patterns"]),
        "pattern_ids": [row["pattern_id"] for row in registry["patterns"]],
        "registry_semantic_sha256": registry_hash,
        "registry_file_sha256": sha256_file(registry_path),
    }
    oracle = build_benchmark_oracle(selected_rows, answers, registry_hash)
    general_gold = build_general_gold()
    # Freeze the complete planning projection after checking that it agrees
    # with the independently authored coarse benchmark/general expectation.
    # Gate 2 compares this full snapshot, so resolver-blocked nodes, payload,
    # dependencies, quorum, and registry drift cannot pass as a backend-only
    # signature match.
    from validate_phase_08_gate2 import (  # local script import; never used by runtime QA
        execute_planning,
        frozen_projection,
        request as planning_request,
        synthetic_records,
    )
    from finglmqa.analyzer import QuestionAnalyzer
    from finglmqa.composition import TopologyCompositionPlanner
    from finglmqa.resolver import ScopeResolver

    analyzer = QuestionAnalyzer()
    production_resolver = ScopeResolver()
    synthetic_resolver = ScopeResolver(records=synthetic_records())
    planner = TopologyCompositionPlanner()
    for gold in oracle:
        actual = execute_planning(
            analyzer, production_resolver, planner,
            planning_request(gold["case_id"], gold["source"]["question"]),
        )
        if actual["plan"] is None:
            raise RuntimeError(f"benchmark gold unexpectedly became terminal: {gold['case_id']}")
        if actual["plan"]["pattern_id"] != gold["expected_composition"]["pattern_id"]:
            raise RuntimeError(f"benchmark pattern disagrees with offline gold: {gold['case_id']}")
        actual_pairs = [(row["backend"], row["operation"]) for row in actual["plan"]["subplans"]]
        expected_pairs = [
            (row["backend"], row["operation"])
            for row in gold["expected_composition"]["ordered_subplan_signatures"]
        ]
        if actual_pairs != expected_pairs:
            raise RuntimeError(f"benchmark decomposition disagrees with offline gold: {gold['case_id']}")
        gold["oracle_version"] = "1.1.0"
        gold["expected_planning_projection"] = frozen_projection(actual)
        plan = actual["plan"]
        gold["expected_composition"].update({
            "pattern_version": plan["pattern_version"],
            "registry_semantic_sha256": plan["registry_semantic_sha256"],
            "minimum_usable_results": plan["composition_policy"]["minimum_usable_results"],
            "ordered_subplan_signatures": [
                {
                    "ordinal": row["ordinal"], "backend": row["backend"],
                    "operation": row["operation"], "planning_state": row["planning_state"],
                    "entity_key": row["entity_key"], "period_key": row["period_key"],
                    "concern_key": row["concern_key"], "required": row["required"],
                }
                for row in plan["subplans"]
            ],
        })

    synthetic_case_ids = {"G05", "G07", "G08", "G26", "G27", "G28", "G29", "G30"}
    for gold in general_gold:
        resolver = synthetic_resolver if gold["case_id"] in synthetic_case_ids else production_resolver
        actual = execute_planning(
            analyzer, resolver, planner,
            planning_request(gold["case_id"], gold["question"]),
        )
        expected = gold["expected"]
        actual_pattern = actual["plan"]["pattern_id"] if actual["plan"] else None
        actual_failure = actual["terminal"]["failure_code"] if actual["terminal"] else None
        if actual_pattern != expected["pattern_id"] or (
            expected["pattern_id"] is None and actual_failure != expected["failure_code"]
        ):
            raise RuntimeError(f"General planning disagrees with manual gold: {gold['case_id']}")
        gold["expected_planning_projection"] = frozen_projection(actual)
    by_uid = {row["uid"]: row for row in selected_rows}
    supported_manifest = {
        "schema_version": "finglmqa.phase8.supported_fixture_manifest.v1",
        "type1_exact": [
            {"uid": uid, "question": by_uid[uid]["question"], "mapped_report_uids": by_uid[uid]["mapped_report_uids"], "answer_prompt": answers[f"1:{uid}"]["prompt"]}
            for uid in SUPPORTED_TYPE1_UIDS
        ],
        "formula_exact": [
            {"uid": uid, "question": by_uid[uid]["question"], "mapped_report_uids": by_uid[uid]["mapped_report_uids"], "answer_prompt": answers[f"2-1:{uid}"]["prompt"]}
            for uid in SUPPORTED_FORMULA_UIDS
        ],
        "counts": {"type1_exact": 9, "formula_exact": 6},
    }
    immutable_manifest = build_immutable_manifest(selected_rows)
    revalidation_paths = [
        ROOT / "runs/phase_08/reports/phase7_revalidation_1.json",
        ROOT / "runs/phase_08/reports/phase7_revalidation_run2/report.json",
    ]
    revalidations = [json.loads(path.read_text(encoding="utf-8")) for path in revalidation_paths]
    projections = [deterministic_projection(report) for report in revalidations]
    comparison = {
        "schema_version": "finglmqa.phase8.phase7_revalidation_comparison.v1",
        "runs": [file_record(path) for path in revalidation_paths],
        "all_checks_passed": all(report["all_checks_passed"] for report in revalidations),
        "deterministic_projection_equal": projections[0] == projections[1],
        "projection_sha256": [semantic_sha256(projection) for projection in projections],
        "retriever_sha256": sha256_file(ROOT / "scripts/query_type3_evidence.py"),
        "evidence_sha256": sha256_file(ROOT / "data/corpus_package/evidence_chunks.jsonl"),
    }
    type_counts = Counter(row["type"] for row in selected_rows)
    unique_questions = len({row["question"] for row in selected_rows})
    gate0 = {
        "schema_version": SCHEMA_GATE0,
        "checks": {
            "selected_row_count_1003": len(selected_rows) == 1003,
            "selected_unique_questions_702": unique_questions == 702,
            "benchmark_type_counts": dict(type_counts) == TYPE_COUNTS,
            "oracle_row_count_1003": len(oracle) == 1003,
            "general_gold_count_40": len(general_gold) == 40,
            "pattern_count_9": registry_manifest["pattern_count"] == 9,
            "supported_type1_count_9": len(supported_manifest["type1_exact"]) == 9,
            "supported_formula_count_6": len(supported_manifest["formula_exact"]) == 6,
            "phase7_revalidation_both_green": comparison["all_checks_passed"],
            "phase7_revalidation_deterministic_projection_equal": comparison["deterministic_projection_equal"],
        },
        "artifacts": {
            "immutable_manifest": "runs/phase_08/immutable_inputs_manifest.json",
            "pattern_registry_manifest": "runs/phase_08/pattern_registry_manifest.json",
            "benchmark_oracle": "runs/phase_08/benchmark_decomposition_oracle.jsonl",
            "general_gold": "runs/phase_08/general_decomposition_gold.jsonl",
            "supported_fixture_manifest": "runs/phase_08/supported_fixture_manifest.json",
            "phase7_revalidation_comparison": "runs/phase_08/reports/phase7_revalidation_comparison.json",
        },
        "benchmark": {"rows": len(selected_rows), "unique_questions": unique_questions, "type_counts": dict(type_counts), "answer_file_hashes": answer_hashes},
    }
    gate0["all_checks_passed"] = all(gate0["checks"].values())
    write_json(run_dir / "immutable_inputs_manifest.json", immutable_manifest)
    write_json(run_dir / "pattern_registry_manifest.json", registry_manifest)
    write_jsonl(run_dir / "benchmark_decomposition_oracle.jsonl", oracle)
    write_jsonl(run_dir / "general_decomposition_gold.jsonl", general_gold)
    write_json(run_dir / "supported_fixture_manifest.json", supported_manifest)
    write_json(run_dir / "reports/phase7_revalidation_comparison.json", comparison)
    write_json(run_dir / "gate0_report.json", gate0)
    print(json.dumps({"all_checks_passed": gate0["all_checks_passed"], "checks": gate0["checks"]}, ensure_ascii=False, indent=2))
    return 0 if gate0["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
