from __future__ import annotations

import asyncio
import csv
import json
import socket
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
ONE_DOC = REPO_ROOT / "output/hybrid_pipeline_one_doc"
EVIDENCE_DIR = ONE_DOC / "05_evidence/A000651_格力电器_2021年年度报告"
LABEL_DIR = ONE_DOC / "07_stage2_labels/A000651_格力电器_2021年年度报告_hybrid_industry_structured"
EVAL_DIR = REPO_ROOT / "output/sw3_l4_eval_430101_hybrid"

app = FastAPI(title="HyFin ICDM Demo API", version="1.0.0")
RUNS: dict[str, dict[str, Any]] = {}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def service_up(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.15):
            return True
    except OSError:
        return False


def llm_configured() -> bool:
    env_file = REPO_ROOT / "A2RAG/.env"
    if not env_file.is_file():
        return False
    keys = {"A2RAG_API_KEY", "OPENAI_API_KEY", "API_KEY"}
    for line in env_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "=" not in line or line.lstrip().startswith("#"):
            continue
        key, value = line.split("=", 1)
        if key.strip() in keys and value.strip().strip("'\""):
            return True
    return False


def stage2_path() -> Path:
    matches = sorted(LABEL_DIR.glob("*_stage2.json"))
    if not matches:
        raise FileNotFoundError("Stage2 label artifact is missing")
    return matches[0]


def run_meta_path() -> Path:
    matches = sorted(LABEL_DIR.glob("*_run_meta.json"))
    if not matches:
        raise FileNotFoundError("Stage2 run metadata is missing")
    return matches[0]


def featured_payload() -> dict[str, Any]:
    manifest = read_json(EVIDENCE_DIR / "manifest.json")
    text = read_json(EVIDENCE_DIR / "text_evidence.json")
    table = read_json(EVIDENCE_DIR / "table_evidence.json")
    stage2 = read_json(stage2_path())
    run_meta = read_json(run_meta_path())
    qa = text.get("qa") or {}
    docs = qa.get("supporting_docs") or (text.get("retrieve") or {}).get("docs") or []
    scores = qa.get("doc_scores") or (text.get("retrieve") or {}).get("doc_scores") or []
    return {
        "id": "gree-2021",
        "company": {"id": "A000651", "ticker": "000651.SZ", "name": "格力电器", "year": 2021},
        "execution": {
            "mode": "artifact-backed",
            "text_source": manifest["text_evidence"]["source"],
            "table_source": manifest["table_evidence"]["source"],
            "stage2_status": run_meta.get("status"),
            "stage2_model": run_meta.get("model"),
            "processed_at": run_meta.get("processed_at"),
        },
        "stats": {
            "tables_seen": manifest["table_stats"]["tables_seen"],
            "tables_kept": manifest["table_stats"]["tables_kept"],
            "graph_nodes": manifest["graph_summary"]["nodes"],
            "graph_edges": manifest["graph_summary"]["edges"],
            "stage2_input_chars": manifest["stage2_input_chars"],
        },
        "text_evidence": {
            "question": text.get("question"),
            "answer": text.get("answer"),
            "excerpt": docs[0] if docs else "",
            "score": scores[0] if scores else None,
            "supporting_documents": len(docs),
        },
        "table_evidence": {
            "question": table.get("question"),
            "source": table.get("source"),
            "tables": table.get("tables"),
            "items": table.get("items", []),
        },
        "labels": stage2.get("LeafIndustryTags", []),
        "quality": stage2.get("QualityFlag", {}),
    }


def experiment_payload() -> dict[str, Any]:
    summary = read_json(EVAL_DIR / "summary.json")
    targets = read_csv(EVAL_DIR / "target_companies.csv")
    cluster_rows = read_csv(EVAL_DIR / "clusters.csv")
    movement_rows = read_csv(EVAL_DIR / "cluster_group_comovement.csv")
    label_rows = read_csv(EVAL_DIR / "label_df.csv")

    company_cluster: dict[str, str] = {}
    for cluster in cluster_rows:
        cluster_id = cluster["cluster_id"].replace("cluster_", "")
        for token in cluster["companies"].split(";"):
            company_id = token.strip().split(":", 1)[0]
            if company_id:
                company_cluster[company_id] = cluster_id

    movement = {
        row["label"]: float(row["market_neutral_comovement"])
        for row in movement_rows
        if row.get("market_neutral_comovement")
    }
    companies = [
        [
            row["company_id"],
            row["company_name"].strip(),
            company_cluster.get(row["company_id"], ""),
            [label.strip() for label in row["clean_labels"].split(";") if label.strip()],
        ]
        for row in targets
    ]
    clusters = [
        [
            row["cluster_id"].replace("cluster_", ""),
            int(row["company_count"]),
            movement.get(row["cluster_id"]),
            " · ".join(item.split(":", 1)[0] for item in row["top_clean_labels"].split(";")[:3]),
            "、".join(item.strip().split(":", 1)[-1] for item in row["companies"].split(";")),
        ]
        for row in cluster_rows
    ]
    labels = [
        [row["canonical_label"], int(row["company_df"]), float(row["idf"])]
        for row in label_rows
        if row.get("kept") == "True" and row.get("idf")
    ]
    return {
        "generated_at": summary.get("generated_at"),
        "summary": {
            "officialCompanies": summary["companies_in_official_sw3"],
            "labeledCompanies": summary["companies_with_new_method_labels"],
            "priceCompanies": summary["usable_companies"],
            "groups": summary["l4_cluster_partition"]["groups"],
            "randomTrials": summary["random_baseline"]["trials"],
            "officialComovement": summary["sw3_whole_partition"]["market_neutral_comovement"],
            "withinComovement": summary["l4_cluster_separation"]["within_market_neutral_comovement"],
            "crossComovement": summary["l4_cluster_separation"]["cross_market_neutral_comovement"],
            "separation": summary["l4_cluster_separation"]["separation"],
            "pValue": summary["random_baseline"]["p_value_within_ge_observed"],
        },
        "companies": companies,
        "clusters": clusters,
        "labels": labels,
    }


async def execute_artifact_run(run_id: str) -> None:
    run = RUNS[run_id]
    try:
        stage_files = [
            ("parse", ONE_DOC / "01_clean_text/A000651_格力电器_2021年年度报告.md"),
            ("retrieve", EVIDENCE_DIR / "text_evidence.json"),
            ("align", ONE_DOC / "06_stage2_inputs/A000651_格力电器_2021年年度报告_hybrid_industry_structured.md"),
            ("generate", stage2_path()),
        ]
        for index, (name, path) in enumerate(stage_files):
            started = time.perf_counter()
            run["stages"][index]["status"] = "running"
            if not path.is_file():
                raise FileNotFoundError(str(path))
            content = path.read_text(encoding="utf-8", errors="ignore")
            await asyncio.sleep(0.45)
            run["stages"][index].update(
                status="complete",
                bytes=len(content.encode("utf-8")),
                elapsed_ms=round((time.perf_counter() - started) * 1000),
            )
        run["status"] = "complete"
        run["result"] = featured_payload()
        run["completed_at"] = time.time()
    except Exception as exc:
        run["status"] = "failed"
        run["error"] = str(exc)


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "backend": "fastapi",
        "artifacts_ready": EVIDENCE_DIR.is_dir() and EVAL_DIR.is_dir(),
        "llm_configured": llm_configured(),
        "services": {"a2rag": service_up(8000), "tabgr": service_up(8002)},
    }


@app.get("/api/demo/featured")
def featured() -> dict[str, Any]:
    try:
        return featured_payload()
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail=f"Featured artifacts unavailable: {exc}") from exc


@app.get("/api/experiments")
def experiments() -> dict[str, Any]:
    try:
        return experiment_payload()
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail=f"Evaluation artifacts unavailable: {exc}") from exc


@app.post("/api/demo/runs", status_code=202)
async def create_run() -> dict[str, str]:
    run_id = uuid.uuid4().hex[:12]
    RUNS[run_id] = {
        "run_id": run_id,
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
    asyncio.create_task(execute_artifact_run(run_id))
    return {"run_id": run_id, "status": "running", "mode": "artifact-backed"}


@app.get("/api/demo/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    run = RUNS.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


app.mount("/", StaticFiles(directory=APP_DIR, html=True), name="frontend")
