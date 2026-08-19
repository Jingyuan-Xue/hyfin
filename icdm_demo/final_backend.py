"""Backend wired to one coherent, paper-facing new_method run.

The previous demo mixed the intermediate SW3 classifier, the dense-label run,
and an older evaluation directory.  This module intentionally uses the final
``rerun_fixed_20260722`` run for labels and experiments, while grounding each case
in its matching Stage-1 A2RAG/TabGR artifact.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import urllib.error
import urllib.parse
import urllib.request
import re
import time
import uuid
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from icdm_demo.backend import (
    APP_DIR,
    ONE_DOC,
    RUNS,
    featured_payload,
    llm_configured,
    read_csv,
    read_json,
    service_up,
    stage2_path,
)
from icdm_demo.multi_backend import EVIDENCE_CACHE, INPUT_DIR, refactor_paths
from icdm_demo.translation_service import TranslationService, translation_metadata


REPO_ROOT = APP_DIR.parent
FINGLMQA_API_URL = os.environ.get("FINGLMQA_API_URL", "http://127.0.0.1:8010").rstrip("/")
RISK_EXPOSURE_API_URL = os.environ.get("RISK_EXPOSURE_API_URL", "http://127.0.0.1:8012").rstrip("/")
FINAL_RUN = REPO_ROOT / "output/refactor_pipeline_430101/rerun_fixed_20260722/final_best_1000"
FINAL_LABELS = FINAL_RUN / "labels.jsonl"
FINAL_EVAL = FINAL_RUN / "comovement_with_shenwan"
FINAL_RANDOM = FINAL_EVAL / "summary.json"
TARGETS = REPO_ROOT / "output/refactor_pipeline_430101/run/baseline_shenwan_430101_selected25.csv"
GREE_TABLES = ONE_DOC / "02_tables_jsonl/A000651_格力电器_2021年年度报告_tables.jsonl"
FINGLMQA_SOURCE_ROOT = REPO_ROOT / "FinGLMQA/refs/source_markdown"
MAX_CONSOLIDATED_REPORTS = 3

app = FastAPI(title="HyFin ICDM Demo API", version="2.0.0")


TRANSLATOR = TranslationService(REPO_ROOT)


def requested_language(request: Request) -> str:
    value = request.query_params.get("lang") or request.headers.get("X-Display-Language") or "zh"
    return "en" if value.lower().startswith("en") else "zh"


def upstream_query(request: Request) -> str:
    return urllib.parse.urlencode(
        [(key, value) for key, value in request.query_params.multi_items() if key != "lang"]
    )


def _copy_payload(payload: Any) -> Any:
    return json.loads(json.dumps(payload, ensure_ascii=False))


def _translate_refs(refs: list[tuple[dict[str, Any], str]], source: str = "zh", target: str = "en") -> None:
    if not refs:
        return
    values = [str(container.get(key) or "") for container, key in refs]
    translated = TRANSLATOR.translate_many(values, source, target)
    for (container, key), original, display in zip(refs, values, translated):
        container[f"{key}_original"] = original
        container[key] = display


def project_case_payload(payload: dict[str, Any], language: str) -> dict[str, Any]:
    projected = _copy_payload(payload)
    if language != "en":
        projected["_translation"] = translation_metadata(TRANSLATOR, language)
        return projected
    refs: list[tuple[dict[str, Any], str]] = []
    company = projected.get("company") or {}
    if company.get("name"):
        refs.append((company, "name"))
    text = projected.get("text_evidence") or {}
    for key in ("question", "answer", "excerpt"):
        if text.get(key):
            refs.append((text, key))
    for item in text.get("items") or []:
        for key in ("question", "answer", "excerpt", "title", "heading"):
            if item.get(key):
                refs.append((item, key))
        supports = item.get("supports") or []
        if supports:
            item["supports_original"] = list(supports)
            item["supports"] = TRANSLATOR.translate_many(supports)
    table = projected.get("table_evidence") or {}
    for item in table.get("items") or []:
        for key in ("title", "heading", "keep_reason", "graph_text", "reasoning_trace"):
            if item.get(key):
                refs.append((item, key))
        if item.get("table_text"):
            item["table_text_original"] = item["table_text"]
            item["table_text"] = TRANSLATOR.translate_markdown_table(item["table_text"])
        supports = item.get("supports") or []
        if supports:
            item["supports_original"] = list(supports)
            item["supports"] = TRANSLATOR.translate_many(supports)
    for label in projected.get("labels") or []:
        for key in ("Tag", "Definition", "Reason"):
            if label.get(key):
                refs.append((label, key))
        evidence = label.get("Evidence") or []
        if evidence:
            label["Evidence_original"] = list(evidence)
            label["Evidence"] = TRANSLATOR.translate_many(evidence)
    quality = projected.get("quality") or {}
    if quality.get("Notes"):
        refs.append((quality, "Notes"))
    _translate_refs(refs)
    projected["_translation"] = translation_metadata(TRANSLATOR, language)
    return projected


def project_case_catalog(payload: list[dict[str, Any]], language: str) -> list[dict[str, Any]]:
    projected = _copy_payload(payload)
    if language != "en":
        return projected
    refs = [(item, "name") for item in projected if item.get("name")]
    _translate_refs(refs)
    return projected


def project_documents(payload: dict[str, Any], language: str) -> dict[str, Any]:
    projected = _copy_payload(payload)
    documents = projected.get("documents") or []
    if language == "en":
        names = [str(item.get("stock_name") or "") for item in documents]
        full_names = [str(item.get("company_full") or "") for item in documents]
        display_names = TRANSLATOR.translate_many(names)
        display_full_names = TRANSLATOR.translate_many(full_names)
        for item, display_name, display_full in zip(documents, display_names, display_full_names):
            item["display_name"] = display_name
            item["display_company_full"] = display_full
            item["display_title"] = f"{display_name} · {item.get('report_year', '')} Annual Report"
    projected["_translation"] = translation_metadata(TRANSLATOR, language)
    return projected


def project_risk_catalog(payload: dict[str, Any], language: str) -> dict[str, Any]:
    projected = _copy_payload(payload)
    companies = projected.get("companies") or []
    if language == "en":
        values = [str(item.get("company_name") or "") for item in companies]
        translated = TRANSLATOR.translate_many(values)
        for item, original, display in zip(companies, values, translated):
            item["company_name_original"] = original
            item["display_name"] = display
    projected["_translation"] = translation_metadata(TRANSLATOR, language)
    return projected


def project_risk_detail(payload: dict[str, Any], language: str) -> dict[str, Any]:
    projected = _copy_payload(payload)
    if language != "en":
        projected["_translation"] = translation_metadata(TRANSLATOR, language)
        return projected
    refs: list[tuple[dict[str, Any], str]] = []
    company = projected.get("company") or {}
    if company.get("company_name"):
        company["company_name_original"] = company["company_name"]
        company["display_name"] = TRANSLATOR.translate(company["company_name"])
    for exposure in projected.get("risk_exposures") or []:
        exposure["CategoryDisplay"] = TRANSLATOR.translate(exposure.get("Category", ""))
        for key in ("RiskName", "Subcategory", "Reason"):
            if exposure.get(key):
                refs.append((exposure, key))
        mitigants = exposure.get("Mitigants") or []
        if mitigants:
            exposure["Mitigants_original"] = list(mitigants)
            exposure["Mitigants"] = TRANSLATOR.translate_many(mitigants)
        for evidence in exposure.get("Evidence") or []:
            for key in ("EvidenceQuote", "Interpretation"):
                if evidence.get(key):
                    refs.append((evidence, key))
    _translate_refs(refs)
    projected["_translation"] = translation_metadata(TRANSLATOR, language)
    return projected


_QA_TRANSLATED_KEYS = {
    "answer", "answer_text", "text", "summary", "content",
    "claim", "claim_text", "evidence_text", "source_excerpt", "excerpt",
    "supporting_text", "quote", "snippet", "title", "heading", "section", "reason", "interpretation",
}


def _translate_qa_node(value: Any, key: str = "") -> Any:
    if isinstance(value, dict):
        output = {}
        for child_key, child_value in value.items():
            if child_key == "section_path" and isinstance(child_value, list):
                output[child_key + "_original"] = list(child_value)
                output[child_key] = TRANSLATOR.translate_many(child_value)
            elif child_key in _QA_TRANSLATED_KEYS and isinstance(child_value, str):
                output[child_key + "_original"] = child_value
                output[child_key] = TRANSLATOR.translate(child_value)
            else:
                output[child_key] = _translate_qa_node(child_value, child_key)
        return output
    if isinstance(value, list):
        return [_translate_qa_node(item, key) for item in value]
    return value


def project_qa_payload(payload: dict[str, Any], language: str) -> dict[str, Any]:
    projected = _copy_payload(payload)
    if language == "en":
        projected = _translate_qa_node(projected)
    projected["_translation"] = translation_metadata(TRANSLATOR, language)
    return projected


@lru_cache(maxsize=16)
def _qa_source_lines(source_name: str) -> tuple[str, ...]:
    """Read one corpus source using a basename-only, read-only lookup."""
    safe_name = Path(source_name).name
    if not safe_name:
        return ()
    source_path = FINGLMQA_SOURCE_ROOT / safe_name
    if not source_path.is_file():
        return ()
    try:
        return tuple(source_path.read_text(encoding="utf-8", errors="replace").splitlines())
    except OSError:
        return ()


def enrich_qa_evidence(payload: dict[str, Any]) -> dict[str, Any]:
    """Attach the exact cited source-line span for the presentation frontend."""
    enriched = _copy_payload(payload)
    for citation in enriched.get("citations") or []:
        if not isinstance(citation, dict) or citation.get("excerpt"):
            continue
        provenance = citation.get("provenance") or {}
        source_name = str(provenance.get("source_markdown") or "")
        line_range = provenance.get("line_range")
        if not source_name or not isinstance(line_range, list) or len(line_range) != 2:
            continue
        start, end = line_range
        if not isinstance(start, int) or not isinstance(end, int) or start < 1 or end < start:
            continue
        lines = _qa_source_lines(source_name)
        if not lines or start > len(lines):
            continue
        excerpt = "\n".join(lines[start - 1 : min(end, len(lines))]).strip()
        if excerpt:
            citation["excerpt"] = excerpt
    return enriched

def restore_originals(value: Any) -> Any:
    if isinstance(value, dict):
        output: dict[str, Any] = {}
        for key, child in value.items():
            if key.endswith("_original") or key == "_translation":
                continue
            original_key = f"{key}_original"
            output[key] = restore_originals(value.get(original_key, child))
        return output
    if isinstance(value, list):
        return [restore_originals(item) for item in value]
    return value



def response_json(response: Response) -> dict[str, Any]:
    try:
        return json.loads(bytes(response.body))
    except (AttributeError, TypeError, json.JSONDecodeError):
        return {}


def translated_json_response(response: Response, payload: dict[str, Any]) -> Response:
    return JSONResponse(content=payload, status_code=response.status_code)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def clean_company_name(value: str) -> str:
    return re.sub(r"\s+", "", str(value or ""))


def target_names() -> dict[str, str]:
    return {
        row["company_id"]: clean_company_name(row["company_name"])
        for row in read_csv(TARGETS)
    }


def final_labels_by_company() -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(FINAL_LABELS):
        grouped[str(row["company_id"])].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda item: (-float(item.get("confidence", 0)), str(item.get("label", ""))))
    return dict(grouped)


def cjk_bigrams(value: str) -> set[str]:
    grams: set[str] = set()
    for chunk in re.findall(r"[\u3400-\u9fff]{2,}", value):
        grams.update(chunk[index : index + 2] for index in range(len(chunk) - 1))
    return grams


def relevance_score(text: str, labels: list[dict[str, Any]]) -> float:
    haystack = str(text or "")
    if not haystack:
        return -1.0
    score = 0.0
    query_parts: list[str] = []
    for item in labels[:8]:
        label = str(item.get("label") or item.get("Tag") or "")
        reason = str(item.get("audit_brief_reason") or item.get("Reason") or "")
        evidence = " ".join(map(str, item.get("Evidence") or []))
        query_parts.extend([label, reason, evidence])
        if label and label in haystack:
            score += 25.0 * float(item.get("confidence", item.get("Confidence", 0.5)))
    query = " ".join(query_parts)
    score += sum(0.7 for gram in cjk_bigrams(query) if gram in haystack)
    for number in set(re.findall(r"\d[\d,.]*%?", query)):
        if len(number) >= 3 and number in haystack:
            score += 5.0
    return score


def aligned_supports(text: str, labels: list[dict[str, Any]], limit: int = 4) -> list[str]:
    aliases = {
        "物业管理": ("物业管理", "物业服务"),
        "商业地产": ("商业地产", "购物中心", "商业运营"),
        "购物中心运营": ("购物中心", "商业运营"),
        "物流仓储": ("物流仓储", "仓储物流", "仓储服务"),
        "长租公寓": ("长租公寓", "泊寓"),
        "运营管理服务": ("运营管理服务", "运营管理费"),
    }
    supported: list[tuple[float, str]] = []
    for item in labels:
        label = str(item.get("label") or item.get("Tag") or "")
        variants = list(aliases.get(label, (label,)))
        for suffix in ("制造与销售", "开发与运营", "生产销售", "制造", "业务"):
            if label.endswith(suffix) and len(label) > len(suffix) + 1:
                variants.append(label[: -len(suffix)])
        hits = [variant for variant in variants if len(variant) >= 2 and variant in text]
        if hits:
            supported.append((max(len(value) for value in hits), label))
    return [label for _, label in sorted(supported, reverse=True)[:limit]]


def normalize_dense_labels(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for index, row in enumerate(rows):
        output.append(
            {
                "Tag": row.get("label", ""),
                "Confidence": float(row.get("confidence", 0)),
                "Definition": "",
                "Evidence": [],
                "Reason": row.get("audit_brief_reason") or "Evidence retained by the final dense-hybrid run.",
                "Role": "primary" if index == 0 else "secondary",
                "source": row.get("source", "new_method_dense_hybrid"),
            }
        )
    return output


def channel_labels(labels: list[dict[str, Any]], channel: str) -> list[dict[str, Any]]:
    """Restrict alignment terms to claims that name the current modality."""
    selected: list[dict[str, Any]] = []
    for item in labels:
        reason = str(item.get("audit_brief_reason") or item.get("Reason") or "")
        evidence = list(map(str, item.get("Evidence") or []))
        if channel == "text":
            channel_evidence = [value for value in evidence if "A2RAG" in value or "文字" in value]
            matches = "文字" in reason or "A2RAG" in reason or bool(channel_evidence)
        else:
            channel_evidence = [value for value in evidence if "表格" in value or "TabGR" in value]
            matches = "表格" in reason or "TabGR" in reason or bool(channel_evidence)
        if matches:
            normalized = dict(item)
            if evidence:
                normalized["Evidence"] = channel_evidence
            selected.append(normalized)
    return selected or labels


def select_text_evidence(text: dict[str, Any], labels: list[dict[str, Any]]) -> dict[str, Any]:
    labels = channel_labels(labels, "text")
    docs = list(text.get("supporting_docs") or [])
    scores = list(text.get("doc_scores") or [])
    if not docs:
        return {
            "question": text.get("question"),
            "answer": text.get("answer"),
            "excerpt": text.get("answer", ""),
            "score": None,
            "supporting_documents": 0,
            "retrieval_rank": None,
            "supports": [],
            "items": [],
        }
    ranked = sorted(
        enumerate(docs),
        key=lambda pair: (relevance_score(pair[1], labels), scores[pair[0]] if pair[0] < len(scores) else 0),
        reverse=True,
    )
    items = [
        {
            "excerpt": evidence,
            "score": scores[original_index] if original_index < len(scores) else None,
            "retrieval_rank": original_index + 1,
            "supports": aligned_supports(evidence, labels),
        }
        for original_index, evidence in ranked
    ]
    index, excerpt = ranked[0]
    return {
        "question": text.get("question"),
        "answer": text.get("answer"),
        "excerpt": excerpt,
        "score": scores[index] if index < len(scores) else None,
        "supporting_documents": len(docs),
        "retrieval_rank": index + 1,
        "selection": "label-grounded rerank over A2RAG supporting_docs",
        "supports": aligned_supports(excerpt, labels),
        "items": items,
    }


def select_table_evidence(table: dict[str, Any], labels: list[dict[str, Any]]) -> dict[str, Any]:
    labels = channel_labels(labels, "table")
    raw_items = table.get("tables")
    if not isinstance(raw_items, list):
        raw_items = table.get("items")
    items = list(raw_items or [])
    ordered = []
    for original_index, item in enumerate(items):
        normalized = dict(item)
        evidence_text = " ".join(
            str(normalized.get(key) or "")
            for key in ("title", "heading", "table_text", "graph_text", "tabgr_answer", "tabgr_text")
        )
        normalized["retrieval_rank"] = original_index + 1
        normalized["supports"] = aligned_supports(evidence_text, labels)
        ordered.append(normalized)
    return {
        "question": table.get("question"),
        "source": table.get("source", "tabgr"),
        "tables": len(items),
        "items": ordered,
        "selection": "TabGR score order from repaired Stage-1 cache",
    }


def matrix_to_markdown(matrix: list[list[Any]], wanted: set[str] | None = None) -> str:
    if not matrix:
        return ""
    rows = [[str(cell or "").replace("|", "／").strip() for cell in row] for row in matrix]
    if wanted:
        selected = [row[:6] for row in rows[1:] if row and row[0] in wanted]
        rows = [["产品", "2021 年收入", "占比", "2020 年收入", "占比", "同比"]] + selected
    else:
        width = min(max(len(row) for row in rows), 5)
        rows = [(row + [""] * width)[:width] for row in rows[:7]]
    if len(rows) < 2:
        return ""
    header = rows[0]
    return "\n".join(
        ["|" + "|".join(header) + "|", "|" + "|".join("---" for _ in header) + "|"]
        + ["|" + "|".join(row) + "|" for row in rows[1:]]
    )


def grounded_gree_payload() -> dict[str, Any]:
    payload = featured_payload()
    labels = payload.get("labels") or []
    evidence_dir = ONE_DOC / "05_evidence/A000651_格力电器_2021年年度报告"
    text = read_json(evidence_dir / "text_evidence.json")
    qa = text.get("qa") or text
    payload["text_evidence"] = select_text_evidence(
        {
            "question": text.get("question"),
            "answer": text.get("answer"),
            "supporting_docs": qa.get("supporting_docs") or (text.get("retrieve") or {}).get("docs") or [],
            "doc_scores": qa.get("doc_scores") or (text.get("retrieve") or {}).get("doc_scores") or [],
        },
        labels,
    )

    table_artifact = read_json(evidence_dir / "table_evidence.json")
    table_rows = read_jsonl(GREE_TABLES)
    table_by_id = {row.get("table_id"): row for row in table_rows}
    selected = select_table_evidence(table_artifact, labels)
    for item in selected["items"]:
        raw = table_by_id.get(item.get("table_id"))
        if raw:
            item["title"] = raw.get("heading") or item.get("heading")
            item["table_text"] = matrix_to_markdown(
                raw.get("matrix") or [], {"空调", "生活电器", "智能装备", "绿色能源"}
            )
            item["supports"] = aligned_supports(item["table_text"], labels)
    payload["table_evidence"] = selected
    payload["provenance"] = {
        "run": "hybrid_pipeline_one_doc",
        "text_artifact": str((evidence_dir / "text_evidence.json").relative_to(REPO_ROOT)),
        "table_artifact": str((evidence_dir / "table_evidence.json").relative_to(REPO_ROOT)),
        "label_artifact": str(stage2_path().relative_to(REPO_ROOT)),
    }
    return payload


def case_catalog() -> list[dict[str, Any]]:
    catalog = [
        {
            "id": "gree-2021",
            "company_id": "A000651",
            "name": "格力电器",
            "ticker": "000651.SZ",
            "year": 2021,
            "source": "hybrid_pipeline_one_doc",
            "labels": 3,
        }
    ]
    names = target_names()
    for company_id, rows in final_labels_by_company().items():
        try:
            refactor_paths(company_id)
        except FileNotFoundError:
            continue
        catalog.append(
            {
                "id": f"{company_id}-2023",
                "company_id": company_id,
                "name": names.get(company_id, clean_company_name(rows[0].get("company_name", company_id))),
                "ticker": f"{company_id[1:]}.SZ",
                "year": 2023,
                "source": "rerun_fixed_20260722",
                "labels": len(rows),
            }
        )
    return catalog


def refactor_payload(case_id: str) -> dict[str, Any]:
    company_id = case_id.split("-", 1)[0]
    source_path, evidence_path, _ = refactor_paths(company_id)
    evidence = read_json(evidence_path)
    label_rows = final_labels_by_company().get(company_id) or []
    if not label_rows:
        raise FileNotFoundError(f"No labels in final run for {company_id}")
    labels = normalize_dense_labels(label_rows)
    text = evidence.get("text") or {}
    table = evidence.get("table") or {}
    names = target_names()
    return {
        "id": case_id,
        "company": {
            "id": company_id,
            "ticker": f"{company_id[1:]}.SZ",
            "name": names.get(company_id, company_id),
            "year": 2023,
        },
        "execution": {
            "mode": "artifact-backed",
            "text_source": text.get("source", "a2rag"),
            "table_source": table.get("source", "tabgr"),
            "stage2_status": "final-filtered",
            "stage2_model": "dense_business_v2_source_isolated",
            "processed_at": None,
        },
        "stats": {
            "tables_seen": len(table.get("tables") or []),
            "tables_kept": len(table.get("tables") or []),
            "graph_nodes": len((table.get("graph") or {}).get("nodes", [])),
            "graph_edges": len((table.get("graph") or {}).get("edges", [])),
            "stage2_input_chars": int(text.get("chars", 0)) + int(table.get("chars", 0)),
        },
        "text_evidence": select_text_evidence(text, label_rows),
        "table_evidence": select_table_evidence(table, label_rows),
        "labels": labels,
        "quality": {
            "NeedHumanReview": False,
            "Sufficiency": "final-filtered",
            "Notes": "Evidence is selected from the matching Stage-1 cache using final-label alignment.",
        },
        "provenance": {
            "run": "rerun_fixed_20260722",
            "source_document": str(source_path.relative_to(REPO_ROOT)),
            "evidence_artifact": str(evidence_path.relative_to(REPO_ROOT)),
            "label_artifact": str(FINAL_LABELS.relative_to(REPO_ROOT)),
        },
    }


def case_payload(case_id: str) -> dict[str, Any]:
    if case_id == "gree-2021":
        return grounded_gree_payload()
    available = {item["id"] for item in case_catalog()}
    if case_id not in available:
        raise FileNotFoundError(f"Unknown final-run case: {case_id}")
    return refactor_payload(case_id)


def experiment_payload() -> dict[str, Any]:
    summary = read_json(FINAL_EVAL / "summary.json")
    random_summary = read_json(FINAL_RANDOM)
    label_rows = read_jsonl(FINAL_LABELS)
    names = target_names()
    by_company: dict[str, list[dict[str, Any]]] = defaultdict(list)
    counts: Counter[str] = Counter()
    for row in label_rows:
        by_company[row["company_id"]].append(row)
        counts[row["label"]] += 1

    companies = []
    for company_id, rows in sorted(by_company.items()):
        rows.sort(key=lambda item: -float(item.get("confidence", 0)))
        labels = [row["label"] for row in rows]
        companies.append([company_id, names.get(company_id, company_id), labels[0], labels])

    group_rows = read_csv(FINAL_EVAL / "cluster_group_comovement.csv")
    clusters = []
    for row in group_rows:
        members = [token for token in row.get("companies", "").split(";") if token]
        clusters.append(
            [
                row["label"],
                int(row["company_count"]),
                float(row["market_neutral_comovement"]) if row.get("market_neutral_comovement") else None,
                row["label"],
                "、".join(names.get(company_id, company_id) for company_id in members),
            ]
        )

    company_count = len(by_company)
    labels = [
        [label, count, math.log((1 + company_count) / (1 + count)) + 1]
        for label, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]
    new_partition = summary["l4_cluster_partition"]
    separation = summary["l4_cluster_separation"]
    baseline = summary["sw3_whole_partition"]
    random_partition = summary["random_baseline"]
    return {
        "generated_at": summary.get("generated_at"),
        "run_config": "dense hybrid · confidence ≥ 0.45 · max DF ratio 0.55 · min 3 labels · merge 0.30",
        "summary": {
            "officialCompanies": summary["companies_in_official_sw3"],
            "labeledCompanies": summary["companies_with_new_method_labels"],
            "priceCompanies": summary["usable_companies"],
            "groups": new_partition["groups"],
            "randomTrials": random_partition["trials"],
            "officialComovement": baseline["market_neutral_comovement"],
            "withinComovement": separation["within_market_neutral_comovement"],
            "crossComovement": separation["cross_market_neutral_comovement"],
            "separation": separation["separation"],
            "pValue": random_partition["p_value_within_ge_observed"],
            "liftVsRandom": separation["within_market_neutral_comovement"] - random_partition["mean_within_market_neutral_comovement"],
            "usedPairs": new_partition["used_pairs"],
        },
        "companies": companies,
        "clusters": clusters,
        "labels": labels,
        "provenance": {
            "labels": str(FINAL_LABELS.relative_to(REPO_ROOT)),
            "evaluation": str((FINAL_EVAL / "summary.json").relative_to(REPO_ROOT)),
            "random_baseline": str(FINAL_RANDOM.relative_to(REPO_ROOT)),
        },
    }


def run_files(case_id: str) -> list[tuple[str, Path]]:
    if case_id == "gree-2021":
        evidence_dir = ONE_DOC / "05_evidence/A000651_格力电器_2021年年度报告"
        return [
            ("parse", ONE_DOC / "01_clean_text/A000651_格力电器_2021年年度报告.md"),
            ("retrieve", evidence_dir / "text_evidence.json"),
            ("align", evidence_dir / "table_evidence.json"),
            ("generate", stage2_path()),
        ]
    company_id = case_id.split("-", 1)[0]
    source, evidence, _ = refactor_paths(company_id)
    return [("parse", source), ("retrieve", evidence), ("align", evidence), ("generate", FINAL_LABELS)]


async def execute_case_run(run_id: str, case_id: str) -> None:
    run = RUNS[run_id]
    try:
        for index, (_, path) in enumerate(run_files(case_id)):
            started = time.perf_counter()
            run["stages"][index]["status"] = "running"
            content = path.read_bytes()
            await asyncio.sleep(0.35)
            run["stages"][index].update(
                status="complete",
                bytes=len(content),
                elapsed_ms=round((time.perf_counter() - started) * 1000),
            )
        run["status"] = "complete"
        run["result"] = case_payload(case_id)
        run["completed_at"] = time.time()
    except Exception as exc:
        run["status"] = "failed"
        run["error"] = str(exc)


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "backend": "fastapi",
        "data_run": "rerun_fixed_20260722",
        "artifacts_ready": all(path.is_file() for path in (FINAL_LABELS, FINAL_EVAL / "summary.json", FINAL_RANDOM)),
        "case_count": len(case_catalog()),
        "llm_configured": llm_configured(),
        "services": {
            "a2rag": service_up(8011),
            "tabgr": service_up(8002),
            "finglmqa": service_up(8010),
            "risk_exposure": service_up(8012),
        },
        "translation": TRANSLATOR.health(),
    }


@app.get("/api/experiments")
def experiments() -> dict[str, Any]:
    return experiment_payload()


@app.get("/api/translation/health")
def translation_health() -> dict[str, Any]:
    return TRANSLATOR.health()


@app.post("/api/translation/qa")
async def translate_qa_projection(request: Request) -> Response:
    raw = await request.body()
    if len(raw) > 2_000_000:
        raise HTTPException(status_code=413, detail="Translation payload is too large")
    incoming = json.loads(raw or b"{}")
    language = "en" if str(incoming.get("lang", "zh")).lower().startswith("en") else "zh"
    payload = incoming.get("payload")
    if not isinstance(payload, dict):
        raise HTTPException(status_code=422, detail="payload must be an object")
    if language == "en":
        projected = await asyncio.to_thread(project_qa_payload, restore_originals(payload), "en")
    else:
        projected = restore_originals(payload)
        projected["_translation"] = translation_metadata(TRANSLATOR, "zh")
    return JSONResponse(content=projected)


@app.get("/api/demo/featured")
def featured(request: Request) -> dict[str, Any]:
    return project_case_payload(grounded_gree_payload(), requested_language(request))


@app.get("/api/demo/cases")
def cases(request: Request) -> list[dict[str, Any]]:
    return project_case_catalog(case_catalog(), requested_language(request))


@app.get("/api/demo/cases/{case_id}")
def get_case(case_id: str, request: Request) -> dict[str, Any]:
    try:
        return project_case_payload(case_payload(case_id), requested_language(request))
    except (FileNotFoundError, KeyError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/demo/cases/{case_id}/runs", status_code=202)
async def create_case_run(case_id: str) -> dict[str, str]:
    try:
        case_payload(case_id)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    run_id = uuid.uuid4().hex[:12]
    RUNS[run_id] = {
        "run_id": run_id,
        "case_id": case_id,
        "status": "running",
        "mode": "artifact-backed-final-run",
        "created_at": time.time(),
        "stages": [
            {"name": "parse", "status": "pending"},
            {"name": "retrieve", "status": "pending"},
            {"name": "align", "status": "pending"},
            {"name": "generate", "status": "pending"},
        ],
    }
    asyncio.create_task(execute_case_run(run_id, case_id))
    return {"run_id": run_id, "case_id": case_id, "status": "running", "mode": "artifact-backed-final-run"}


@app.get("/api/demo/runs/{run_id}")
def get_run(run_id: str, request: Request) -> dict[str, Any]:
    run = RUNS.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    projected = _copy_payload(run)
    if projected.get("result"):
        projected["result"] = project_case_payload(projected["result"], requested_language(request))
    return projected



def _json_service_response(
    base_url: str,
    unavailable_detail: str,
    path: str,
    *,
    method: str = "GET",
    body: bytes | None = None,
    query: str = "",
    timeout: float = 15.0,
) -> Response:
    """Forward a fixed JSON service route without exposing another browser origin."""
    url = f"{base_url}{path}"
    if query:
        url = f"{url}?{query}"
    headers = {"Accept": "application/json"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    upstream = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(upstream, timeout=timeout) as result:
            payload = result.read()
            status_code = result.status
            content_type = result.headers.get("Content-Type", "application/json")
    except urllib.error.HTTPError as exc:
        payload = exc.read()
        status_code = exc.code
        content_type = exc.headers.get("Content-Type", "application/json")
    except (urllib.error.URLError, TimeoutError, OSError):
        payload = json.dumps(
            {"ready": False, "status": "error", "detail": unavailable_detail},
            ensure_ascii=False,
        ).encode("utf-8")
        status_code = 503
        content_type = "application/json; charset=utf-8"
    media_type = content_type.split(";", 1)[0].strip() or "application/json"
    return Response(content=payload, status_code=status_code, media_type=media_type)


async def _proxy_json_service(
    base_url: str,
    unavailable_detail: str,
    path: str,
    *,
    method: str = "GET",
    body: bytes | None = None,
    query: str = "",
    timeout: float = 15.0,
) -> Response:
    return await asyncio.to_thread(
        _json_service_response,
        base_url,
        unavailable_detail,
        path,
        method=method,
        body=body,
        query=query,
        timeout=timeout,
    )


async def _proxy_finglmqa(
    path: str,
    *,
    method: str = "GET",
    body: bytes | None = None,
    query: str = "",
    timeout: float = 15.0,
) -> Response:
    return await _proxy_json_service(
        FINGLMQA_API_URL,
        "FinGLMQA service is unavailable. Start the service on port 8010.",
        path,
        method=method,
        body=body,
        query=query,
        timeout=timeout,
    )


async def _proxy_risk_exposure(path: str, *, query: str = "", timeout: float = 10.0) -> Response:
    return await _proxy_json_service(
        RISK_EXPOSURE_API_URL,
        "Risk exposure service is unavailable. Start the service on port 8012.",
        path,
        query=query,
        timeout=timeout,
    )


def _chat_configuration() -> tuple[str, str, str]:
    """Read the same OpenAI-compatible settings the QA worker already uses."""
    base_url = os.environ.get("A2RAG_CHAT_BASE_URL") or os.environ.get("FINGLMQA_CHAT_BASE_URL") or ""
    model = os.environ.get("A2RAG_CHAT_MODEL") or os.environ.get("FINGLMQA_CHAT_MODEL") or ""
    api_key = os.environ.get("A2RAG_API_KEY") or os.environ.get("FINGLMQA_CHAT_API_KEY") or ""
    if not base_url or not model or not api_key:
        raise RuntimeError("online chat configuration is incomplete")
    return base_url.rstrip("/"), model, api_key


def _chat_completion_text(messages: list[dict[str, str]], *, timeout: float = 60.0) -> tuple[str, str]:
    """One blocking chat completion; returns the content and its finish reason.

    A three-report comparison needs materially more room than a single answer,
    and a summary cut mid-sentence is worse than a short one, so the caller is
    told when the model stopped because it ran out of budget.
    """
    base_url, model, api_key = _chat_configuration()
    body = json.dumps(
        {
            "model": model,
            "messages": messages,
            "temperature": 0,
            "top_p": 1,
            "max_tokens": 2400,
        },
        ensure_ascii=False,
    ).encode("utf-8")
    upstream = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=body,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(upstream, timeout=timeout) as result:
        envelope = json.loads(result.read())
    choice = envelope["choices"][0]
    content = choice["message"]["content"]
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("chat completion returned no content")
    return content.strip(), str(choice.get("finish_reason") or "")


# English phrases like "the selected companies" translate into company-shaped
# Chinese nouns.  The frozen resolver treats any such noun as a company mention
# and then ignores the request's company hint (resolver._ordered_mentions only
# falls back to the hint when the question mentions no company at all), so the
# whole question resolves to nothing.  Multi-report questions therefore drop the
# placeholder and name the report explicitly.
_GENERIC_COMPANY_RE = re.compile(
    r"(?:所选|选定|选中|所选的|上述|这些|这几家|各家|各|该等|该|本|这家)(?:的)?"
    r"(?:公司|企业|年报|报告)(?:中|里|内)?(?:的)?"
)


def scope_question_to_report(question: str, company: str, report_year: Any) -> str:
    """Name one report explicitly instead of relying on a generic placeholder."""
    stripped = _GENERIC_COMPANY_RE.sub("", question).strip()
    if not stripped:
        stripped = question.strip()
    # A question that already states its own years ("2017年、2018年…") must not
    # gain a second, contradictory-looking one from the report scope.
    if re.search(r"(?:19|20)\d{2}\s*年", stripped):
        return f"{company}{stripped}"
    return f"{company}{report_year}年{stripped}"


def _report_label(result: dict[str, Any]) -> str:
    return f"{result.get('company') or '?'} · {result.get('report_year') or '?'}"


def _consolidation_messages(question: str, results: list[dict[str, Any]]) -> list[dict[str, str]]:
    blocks = []
    for result in results:
        answer = str(result.get("answer") or "").strip()
        blocks.append(
            f"【{_report_label(result)}】\n"
            + (answer if answer else "（该年报未产生回答）")
        )
    return [
        {
            "role": "system",
            "content": (
                "你是年报问答的汇总编辑。下面给出同一个问题在若干份年报上分别检索得到的回答。"
                "只能使用这些已给出的回答内容，不得引入外部知识，不得杜撰或重新计算任何数字，"
                "公司名称必须逐字保留。请输出一段整合后的回答：先概括共同点，再指出各公司之间的差异。"
                "回答严格控制在180个汉字以内、三句话以内，使共同点与差异均能直接读出。"
                "只有在确实存在标注为“（该年报未产生回答）”的年报时，才说明哪些年报没有给出回答；"
                "若所有年报都有回答，则不要提及缺失。不要逐份罗列原文，也不要输出表格。"
            ),
        },
        {
            "role": "user",
            "content": f"问题：{question}\n\n各年报的检索回答：\n\n" + "\n\n".join(blocks),
        },
    ]


def _fallback_consolidation(results: list[dict[str, Any]]) -> str:
    return "\n\n".join(
        f"【{_report_label(result)}】\n"
        + (str(result.get("answer") or "").strip() or "（该年报未产生回答）")
        for result in results
    )


@app.get("/api/finglmqa/health")
async def finglmqa_health() -> Response:
    return await _proxy_finglmqa("/health/ready", timeout=6.0)


@app.get("/api/finglmqa/meta")
async def finglmqa_meta() -> Response:
    return await _proxy_finglmqa("/api/v1/meta", timeout=6.0)


@app.get("/api/finglmqa/documents")
async def finglmqa_documents(request: Request) -> Response:
    language = requested_language(request)
    response = await _proxy_finglmqa("/api/v1/documents", query=upstream_query(request), timeout=15.0)
    payload = response_json(response)
    if response.status_code // 100 != 2 or not payload:
        return response
    projected = await asyncio.to_thread(project_documents, payload, language)
    return translated_json_response(response, projected)


@app.post("/api/finglmqa/qa")
async def finglmqa_qa(request: Request) -> Response:
    incoming = await request.json()
    response_language = "en" if str(incoming.pop("response_language", "zh")).lower().startswith("en") else "zh"
    question_language = str(incoming.pop("question_language", "zh")).lower()
    canonical_question = str(incoming.pop("canonical_question_zh", "") or "").strip()
    display_question = str(incoming.pop("display_question", "") or incoming.get("question") or "").strip()
    scope_prefix = bool(incoming.pop("scope_company_prefix", False))
    if canonical_question:
        incoming["question"] = canonical_question
    elif question_language.startswith("en"):
        translated_question = await asyncio.to_thread(TRANSLATOR.translate, display_question, "en", "zh")
        if translated_question == display_question and re.search(r"[A-Za-z]{2,}", display_question):
            return JSONResponse(
                status_code=503,
                content={"status": "translation_unavailable", "answer": "", "errors": [{"message": "English-to-Chinese query translation is unavailable."}]},
            )
        incoming["question"] = translated_question
    else:
        incoming["question"] = display_question
    unscoped_question = incoming["question"]
    if scope_prefix and incoming.get("company") and incoming.get("report_year") is not None:
        incoming["question"] = scope_question_to_report(
            unscoped_question, str(incoming["company"]), incoming["report_year"]
        )
    response = await _proxy_finglmqa(
        "/api/v1/qa",
        method="POST",
        body=json.dumps(incoming, ensure_ascii=False).encode("utf-8"),
        timeout=180.0,
    )
    payload = response_json(response)
    if response.status_code // 100 != 2 or not payload:
        return response
    enriched = await asyncio.to_thread(enrich_qa_evidence, payload)
    projected = await asyncio.to_thread(project_qa_payload, enriched, response_language)
    projected["display_question"] = display_question
    # The question shown to the user stays company-neutral; the per-report scoped
    # form is reported separately so one report's name never labels a combined
    # multi-report answer.
    projected["canonical_question_zh"] = unscoped_question
    if incoming["question"] != unscoped_question:
        projected["scoped_question_zh"] = incoming["question"]
    return translated_json_response(response, projected)


@app.post("/api/finglmqa/consolidate")
async def finglmqa_consolidate(request: Request) -> Response:
    """Merge several single-report answers into one consolidated answer.

    Retrieval already happened once per report; this route only organizes,
    compares and summarizes those answers through the online model.  Every
    per-report answer is restored to its original Chinese first, so the
    consolidation reasons over the retrieved text rather than a translation.
    """
    raw = await request.body()
    if len(raw) > 2_000_000:
        raise HTTPException(status_code=413, detail="Consolidation payload is too large")
    incoming = json.loads(raw or b"{}")
    response_language = "en" if str(incoming.get("response_language", "zh")).lower().startswith("en") else "zh"
    display_question = str(incoming.get("display_question") or incoming.get("question") or "").strip()
    question = str(incoming.get("question") or display_question).strip()
    raw_results = incoming.get("results")
    if not isinstance(raw_results, list) or not 2 <= len(raw_results) <= MAX_CONSOLIDATED_REPORTS:
        raise HTTPException(
            status_code=422,
            detail=f"results must hold between 2 and {MAX_CONSOLIDATED_REPORTS} reports",
        )
    if not question:
        raise HTTPException(status_code=422, detail="question must not be empty")

    results = [restore_originals(row) if isinstance(row, dict) else {} for row in raw_results]

    citations: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    for result in results:
        scope = {
            "company": result.get("display_company") or result.get("company"),
            "report_year": result.get("report_year"),
        }
        for citation in result.get("citations") or []:
            if isinstance(citation, dict):
                citations.append({**citation, "report_scope": scope})
        if result.get("status") != "ok":
            warnings.append({
                "message": f"{_report_label(result)}: {result.get('status') or 'no answer'}"
            })

    try:
        answer, finish_reason = await asyncio.to_thread(
            _chat_completion_text, _consolidation_messages(question, results)
        )
        status = "ok" if any(result.get("status") == "ok" for result in results) else "partial"
        if finish_reason == "length":
            status = "partial"
            warnings.append({"message": "The consolidated answer was cut short by the model's output limit."})
    except Exception:
        answer = _fallback_consolidation(results)
        status = "partial"
        warnings.append({"message": "Cross-report consolidation was unavailable; per-report answers are shown instead."})

    payload: dict[str, Any] = {
        "answer": answer,
        "status": status,
        "citations": citations,
        "errors": [],
        "warnings": warnings,
        "reports": [
            {
                "company": result.get("company"),
                "report_year": result.get("report_year"),
                "status": result.get("status"),
            }
            for result in results
        ],
    }
    projected = await asyncio.to_thread(project_qa_payload, payload, response_language)
    projected["display_question"] = display_question
    projected["canonical_question_zh"] = question
    return JSONResponse(content=projected)


@app.get("/api/risk/health")
async def risk_exposure_health() -> Response:
    return await _proxy_risk_exposure("/health/ready", timeout=6.0)


@app.get("/api/risk/meta")
async def risk_exposure_meta() -> Response:
    return await _proxy_risk_exposure("/api/v1/meta", timeout=6.0)


@app.get("/api/risk/companies")
async def risk_exposure_companies(request: Request) -> Response:
    language = requested_language(request)
    response = await _proxy_risk_exposure("/api/v1/companies", query=upstream_query(request))
    payload = response_json(response)
    if response.status_code // 100 != 2 or not payload:
        return response
    projected = await asyncio.to_thread(project_risk_catalog, payload, language)
    return translated_json_response(response, projected)


@app.get("/api/risk/companies/{company_id}")
async def risk_exposure_company(company_id: str, request: Request) -> Response:
    language = requested_language(request)
    safe_company_id = urllib.parse.quote(company_id, safe="")
    response = await _proxy_risk_exposure(
        f"/api/v1/companies/{safe_company_id}", query=upstream_query(request), timeout=30.0
    )
    payload = response_json(response)
    if response.status_code // 100 != 2 or not payload:
        return response
    projected = await asyncio.to_thread(project_risk_detail, payload, language)
    return translated_json_response(response, projected)


@app.get("/api/risk/risk-factors")
async def risk_exposure_factors(request: Request) -> Response:
    query = urllib.parse.urlencode(list(request.query_params.multi_items()))
    return await _proxy_risk_exposure("/api/v1/risk-factors", query=query)


@app.get("/api/risk/evaluation")
async def risk_exposure_evaluation() -> Response:
    return await _proxy_risk_exposure("/api/v1/evaluation")


@app.get("/api/risk/compare")
async def risk_exposure_compare(request: Request) -> Response:
    query = urllib.parse.urlencode(list(request.query_params.multi_items()))
    return await _proxy_risk_exposure("/api/v1/compare", query=query)


app.mount("/", StaticFiles(directory=APP_DIR, html=True), name="frontend")
