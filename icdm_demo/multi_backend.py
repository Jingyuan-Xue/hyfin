from __future__ import annotations

import asyncio
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from icdm_demo.backend import (
    APP_DIR,
    EVAL_DIR,
    ONE_DOC,
    RUNS,
    experiment_payload,
    featured_payload,
    llm_configured,
    read_csv,
    read_json,
    service_up,
    stage2_path,
)

REFACTOR_ROOT = APP_DIR.parent / "output/refactor_pipeline_430101"
INPUT_DIR = REFACTOR_ROOT / "input_docs"
EVIDENCE_CACHE = REFACTOR_ROOT / "rerun_fixed_20260722/stage1_evidence_cache"
LABEL_CACHE = REFACTOR_ROOT / "rerun_fixed_20260722/hybrid_dense/dense_label_cache"

app = FastAPI(title="HyFin ICDM Multi-case Demo API", version="1.1.0")


def company_names() -> dict[str, str]:
    return {
        row["company_id"]: row["company_name"].strip()
        for row in read_csv(EVAL_DIR / "target_companies.csv")
    }


def refactor_paths(company_id: str) -> tuple[Path, Path, Path]:
    inputs = sorted(INPUT_DIR.glob(f"{company_id}_*.md"))
    evidence = sorted(EVIDENCE_CACHE.glob(f"{company_id}_*.json"))
    labels = sorted(LABEL_CACHE.glob(f"{company_id}_*.json"))
    if not inputs or not evidence or not labels:
        raise FileNotFoundError(f"Incomplete artifacts for {company_id}")
    return inputs[0], evidence[0], labels[0]


def case_catalog() -> list[dict[str, Any]]:
    cases = [
        {
            "id": "gree-2021",
            "company_id": "A000651",
            "name": "格力电器",
            "ticker": "000651.SZ",
            "year": 2021,
            "source": "hybrid_pipeline_one_doc",
        }
    ]
    for company_id, name in company_names().items():
        try:
            refactor_paths(company_id)
        except FileNotFoundError:
            continue
        cases.append(
            {
                "id": f"{company_id}-2023",
                "company_id": company_id,
                "name": name,
                "ticker": f"{company_id[1:]}.SZ",
                "year": 2023,
                "source": "refactor_pipeline_430101",
            }
        )
    return cases


def refactor_payload(case_id: str) -> dict[str, Any]:
    company_id = case_id.split("-", 1)[0]
    source_path, evidence_path, label_path = refactor_paths(company_id)
    evidence = read_json(evidence_path)
    labels = read_json(label_path)
    text = evidence.get("text") or {}
    table = evidence.get("table") or {}
    company_name = company_names().get(company_id, company_id)

    normalized_labels = []
    for index, item in enumerate(labels.get("primary_industries", [])):
        normalized_labels.append(
            {
                "Tag": item.get("industry_name", ""),
                "Confidence": item.get("confidence", 0),
                "Definition": "",
                "Evidence": item.get("evidence_ids", []),
                "Reason": item.get("notes", "") or "Primary industry inferred from hybrid evidence.",
                "Role": "primary",
            }
        )
    for item in labels.get("secondary_industries", []):
        normalized_labels.append(
            {
                "Tag": item.get("industry_name", ""),
                "Confidence": item.get("confidence", 0),
                "Definition": "",
                "Evidence": item.get("evidence_ids", []),
                "Reason": item.get("notes", "") or "Secondary industry inferred from hybrid evidence.",
                "Role": "secondary",
            }
        )

    supporting_docs = text.get("supporting_docs") or []
    scores = text.get("doc_scores") or []
    tables = table.get("tables") or []
    return {
        "id": case_id,
        "company": {
            "id": company_id,
            "ticker": f"{company_id[1:]}.SZ",
            "name": company_name,
            "year": 2023,
        },
        "execution": {
            "mode": "artifact-backed",
            "text_source": text.get("source", "a2rag"),
            "table_source": table.get("source", "tabgr"),
            "stage2_status": "success",
            "stage2_model": "cached Stage2",
            "processed_at": None,
        },
        "stats": {
            "tables_seen": len(tables),
            "tables_kept": len(tables),
            "graph_nodes": len((table.get("graph") or {}).get("nodes", [])),
            "graph_edges": len((table.get("graph") or {}).get("edges", [])),
            "stage2_input_chars": int(text.get("chars", 0)) + int(table.get("chars", 0)),
        },
        "text_evidence": {
            "question": text.get("question"),
            "answer": text.get("answer"),
            "excerpt": supporting_docs[0] if supporting_docs else text.get("answer", ""),
            "score": scores[0] if scores else None,
            "supporting_documents": len(supporting_docs),
        },
        "table_evidence": {
            "question": table.get("question"),
            "source": table.get("source"),
            "tables": len(tables),
            "items": tables,
        },
        "labels": normalized_labels,
        "quality": {
            "NeedHumanReview": False,
            "Sufficiency": "sufficient",
            "Notes": (evidence.get("cross_validation_notes") or [""])[0],
        },
        "artifact_paths": {
            "source": str(source_path.relative_to(APP_DIR.parent)),
            "evidence": str(evidence_path.relative_to(APP_DIR.parent)),
            "labels": str(label_path.relative_to(APP_DIR.parent)),
        },
    }


def case_payload(case_id: str) -> dict[str, Any]:
    if case_id == "gree-2021":
        return featured_payload()
    available = {item["id"] for item in case_catalog()}
    if case_id not in available:
        raise FileNotFoundError(f"Unknown case: {case_id}")
    return refactor_payload(case_id)


def run_files(case_id: str) -> list[tuple[str, Path]]:
    if case_id == "gree-2021":
        evidence_dir = ONE_DOC / "05_evidence/A000651_格力电器_2021年年度报告"
        return [
            ("parse", ONE_DOC / "01_clean_text/A000651_格力电器_2021年年度报告.md"),
            ("retrieve", evidence_dir / "text_evidence.json"),
            ("align", ONE_DOC / "06_stage2_inputs/A000651_格力电器_2021年年度报告_hybrid_industry_structured.md"),
            ("generate", stage2_path()),
        ]
    source, evidence, labels = refactor_paths(case_id.split("-", 1)[0])
    return [("parse", source), ("retrieve", evidence), ("align", evidence), ("generate", labels)]


async def execute_case_run(run_id: str, case_id: str) -> None:
    run = RUNS[run_id]
    try:
        for index, (name, path) in enumerate(run_files(case_id)):
            started = time.perf_counter()
            run["stages"][index]["status"] = "running"
            content = path.read_bytes()
            await asyncio.sleep(0.4)
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
        "artifacts_ready": True,
        "case_count": len(case_catalog()),
        "llm_configured": llm_configured(),
        "services": {"a2rag": service_up(8000), "tabgr": service_up(8002)},
    }


@app.get("/api/experiments")
def experiments() -> dict[str, Any]:
    return experiment_payload()


@app.get("/api/demo/featured")
def featured() -> dict[str, Any]:
    return featured_payload()


@app.get("/api/demo/cases")
def cases() -> list[dict[str, Any]]:
    return case_catalog()


@app.get("/api/demo/cases/{case_id}")
def get_case(case_id: str) -> dict[str, Any]:
    try:
        return case_payload(case_id)
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
        "mode": "artifact-backed",
        "created_at": time.time(),
        "stages": [
            {"name": "parse", "status": "pending"},
            {"name": "retrieve", "status": "pending"},
            {"name": "align", "status": "pending"},
            {"name": "generate", "status": "pending"},
        ],
    }
    asyncio.create_task(execute_case_run(run_id, case_id))
    return {"run_id": run_id, "case_id": case_id, "status": "running", "mode": "artifact-backed"}


@app.get("/api/demo/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    run = RUNS.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


app.mount("/", StaticFiles(directory=APP_DIR, html=True), name="frontend")
