#!/usr/bin/env python3
"""Run the 25-document × 3-recommendation browser QA regression.

The script calls the same public endpoint as the browser, appends one durable
JSONL record after every completed query, and continuously refreshes failure and
summary artifacts. Interrupted runs can be resumed safely.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


PROMPTS = (
    (
        "revenue",
        lambda company, year: f"{company}{year}年营业收入是多少？",
    ),
    (
        "risk",
        lambda company, year: f"{company}{year}年面临哪些经营风险？",
    ),
    (
        "business",
        lambda company, year: f"请简要分析{company}{year}年的主要业务和经营模式",
    ),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def json_request(
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout: float,
) -> tuple[int, dict[str, Any]]:
    body = None if payload is None else json.dumps(
        payload, ensure_ascii=False
    ).encode("utf-8")
    request = Request(
        url,
        data=body,
        method="GET" if body is None else "POST",
        headers={
            "Accept": "application/json",
            **({"Content-Type": "application/json"} if body is not None else {}),
        },
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read()
            status = int(response.status)
    except HTTPError as exc:
        raw = exc.read()
        status = int(exc.code)
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("response root must be a JSON object")
    return status, value


def load_documents(base_url: str, timeout: float, limit: int) -> list[dict[str, Any]]:
    status, value = json_request(
        f"{base_url.rstrip('/')}/api/finglmqa/documents",
        timeout=timeout,
    )
    if status != 200:
        raise RuntimeError(f"document catalog returned HTTP {status}")
    documents = value.get("documents")
    if not isinstance(documents, list) or len(documents) < limit:
        raise RuntimeError(
            f"document catalog has {len(documents) if isinstance(documents, list) else 0} "
            f"rows; at least {limit} are required"
        )
    selected = documents[:limit]
    required = {"document_id", "stock_code", "stock_name", "report_year"}
    for index, row in enumerate(selected, 1):
        if not isinstance(row, dict) or not required.issubset(row):
            raise RuntimeError(f"document row {index} is incomplete")
    return selected


def build_cases(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for document_index, document in enumerate(documents, 1):
        company = str(document["stock_name"])
        year = int(document["report_year"])
        for prompt_index, (prompt_key, builder) in enumerate(PROMPTS, 1):
            cases.append({
                "case_id": f"d{document_index:02d}-q{prompt_index}-{prompt_key}",
                "document_index": document_index,
                "prompt_index": prompt_index,
                "prompt_key": prompt_key,
                "document_id": str(document["document_id"]),
                "stock_code": str(document["stock_code"]),
                "company": company,
                "report_year": year,
                "question": builder(company, year),
            })
    return cases


def error_codes(response: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for item in response.get("errors") or []:
        if isinstance(item, dict):
            code = item.get("failure_code")
            if isinstance(code, str) and code:
                values.append(code)
    return values


def evaluate(http_status: int | None, response: dict[str, Any] | None, exception: str | None) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if exception:
        reasons.append("request_exception")
        return False, reasons
    if http_status != 200:
        reasons.append("http_error")
    if not isinstance(response, dict):
        reasons.append("invalid_response")
        return False, reasons
    status = response.get("status")
    answer = response.get("answer")
    citations = response.get("citations")
    if status not in {"ok", "partial"}:
        reasons.append(f"status_{status or 'missing'}")
    if not isinstance(answer, str) or not answer.strip():
        reasons.append("empty_answer")
    if not isinstance(citations, list) or not citations:
        reasons.append("missing_citations")
    return not reasons, reasons


def read_latest(path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return latest
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and isinstance(row.get("case_id"), str):
                latest[row["case_id"]] = row
    return latest


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def refresh_reports(
    output_dir: Path,
    cases: list[dict[str, Any]],
    latest: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    ordered = [latest[case["case_id"]] for case in cases if case["case_id"] in latest]
    failures = [row for row in ordered if not row.get("passed")]
    passed = [row for row in ordered if row.get("passed")]
    failure_reasons = Counter(
        reason for row in failures for reason in row.get("failure_reasons") or []
    )
    failure_codes = Counter(
        code for row in failures for code in row.get("error_codes") or []
    )
    prompt_totals = Counter(row["prompt_key"] for row in ordered)
    prompt_failures = Counter(row["prompt_key"] for row in failures)
    latencies = [float(row["elapsed_seconds"]) for row in ordered]
    summary = {
        "schema_version": "icdm_demo.qa_recommendation_regression.v1",
        "updated_at": utc_now(),
        "expected_cases": len(cases),
        "completed_cases": len(ordered),
        "passed_cases": len(passed),
        "failed_cases": len(failures),
        "pass_rate": round(len(passed) / len(ordered), 4) if ordered else 0.0,
        "elapsed_seconds": {
            "average": round(sum(latencies) / len(latencies), 3) if latencies else None,
            "maximum": round(max(latencies), 3) if latencies else None,
        },
        "by_prompt": {
            key: {
                "completed": prompt_totals[key],
                "failed": prompt_failures[key],
            }
            for key, _ in PROMPTS
        },
        "failure_reasons": dict(sorted(failure_reasons.items())),
        "failure_codes": dict(sorted(failure_codes.items())),
        "failed_case_ids": [row["case_id"] for row in failures],
    }
    write_json(output_dir / "summary.json", summary)
    with (output_dir / "failures.jsonl").open("w", encoding="utf-8") as handle:
        for row in failures:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    lines = [
        "# QA 推荐问题回归汇总",
        "",
        f"- 更新时间：{summary['updated_at']}",
        f"- 进度：{summary['completed_cases']} / {summary['expected_cases']}",
        f"- 通过：{summary['passed_cases']}",
        f"- 失败：{summary['failed_cases']}",
        f"- 通过率：{summary['pass_rate']:.2%}",
        "",
        "## 按问题类型",
        "",
    ]
    for key, _ in PROMPTS:
        item = summary["by_prompt"][key]
        lines.append(f"- {key}: {item['completed'] - item['failed']} 通过 / {item['failed']} 失败")
    lines.extend(["", "## 失败原因", ""])
    if failure_reasons:
        lines.extend(f"- {key}: {count}" for key, count in sorted(failure_reasons.items()))
    else:
        lines.append("- 暂无")
    lines.extend(["", "## 后端错误码", ""])
    if failure_codes:
        lines.extend(f"- {key}: {count}" for key, count in sorted(failure_codes.items()))
    else:
        lines.append("- 暂无")
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def run_case(base_url: str, case: dict[str, Any], timeout: float, attempt: int) -> dict[str, Any]:
    started = time.monotonic()
    http_status: int | None = None
    response: dict[str, Any] | None = None
    exception: str | None = None
    try:
        http_status, response = json_request(
            f"{base_url.rstrip('/')}/api/finglmqa/qa",
            payload={
                "question": case["question"],
                "company": case["company"],
                "report_year": case["report_year"],
            },
            timeout=timeout,
        )
    except (TimeoutError, URLError, ValueError, OSError) as exc:
        exception = f"{type(exc).__name__}: {exc}"
    elapsed = round(time.monotonic() - started, 3)
    passed, reasons = evaluate(http_status, response, exception)
    answer = response.get("answer") if isinstance(response, dict) else None
    citations = response.get("citations") if isinstance(response, dict) else None
    return {
        **case,
        "attempt": attempt,
        "started_at": utc_now(),
        "elapsed_seconds": elapsed,
        "http_status": http_status,
        "response_status": response.get("status") if isinstance(response, dict) else None,
        "answer": answer if isinstance(answer, str) else "",
        "citation_count": len(citations) if isinstance(citations, list) else 0,
        "error_codes": error_codes(response or {}),
        "errors": response.get("errors") if isinstance(response, dict) else [],
        "warnings": response.get("warnings") if isinstance(response, dict) else [],
        "demo_trace": response.get("demo_trace") if isinstance(response, dict) else None,
        "exception": exception,
        "passed": passed,
        "failure_reasons": reasons,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:4173")
    parser.add_argument("--document-limit", type=int, default=25)
    parser.add_argument("--timeout", type=float, default=150.0)
    parser.add_argument(
        "--output-dir",
        default="output/qa_recommended_75_20260723",
    )
    parser.add_argument(
        "--rerun-failures",
        action="store_true",
        help="Run only cases whose latest recorded attempt failed.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Start a new results file instead of resuming.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.document_limit <= 0:
        raise SystemExit("--document-limit must be positive")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"
    if args.fresh:
        results_path.write_text("", encoding="utf-8")
    documents = load_documents(args.base_url, args.timeout, args.document_limit)
    cases = build_cases(documents)
    if len(cases) != args.document_limit * len(PROMPTS):
        raise RuntimeError("case generation count mismatch")
    write_json(output_dir / "cases.json", cases)
    latest = read_latest(results_path)
    if args.rerun_failures:
        pending = [
            case for case in cases
            if case["case_id"] in latest and not latest[case["case_id"]].get("passed")
        ]
    else:
        pending = [
            case for case in cases
            if case["case_id"] not in latest or not latest[case["case_id"]].get("passed")
        ]
    log_path = output_dir / "run.log"

    def log(message: str) -> None:
        line = f"[{utc_now()}] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    log(
        f"target_documents={len(documents)} target_queries={len(cases)} "
        f"pending={len(pending)} base_url={args.base_url}"
    )
    refresh_reports(output_dir, cases, latest)
    for ordinal, case in enumerate(pending, 1):
        previous = latest.get(case["case_id"])
        attempt = int(previous.get("attempt", 0)) + 1 if previous else 1
        log(
            f"[{ordinal}/{len(pending)}] START {case['case_id']} "
            f"{case['company']} | {case['question']}"
        )
        row = run_case(args.base_url, case, args.timeout, attempt)
        with results_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        latest[case["case_id"]] = row
        summary = refresh_reports(output_dir, cases, latest)
        outcome = "PASS" if row["passed"] else "FAIL"
        details = ",".join(row["failure_reasons"]) or "-"
        log(
            f"[{ordinal}/{len(pending)}] {outcome} {case['case_id']} "
            f"status={row['response_status']} citations={row['citation_count']} "
            f"elapsed={row['elapsed_seconds']:.3f}s reasons={details} "
            f"total_pass={summary['passed_cases']} total_fail={summary['failed_cases']}"
        )
    summary = refresh_reports(output_dir, cases, latest)
    log(
        f"DONE completed={summary['completed_cases']}/{summary['expected_cases']} "
        f"passed={summary['passed_cases']} failed={summary['failed_cases']}"
    )
    return 0 if summary["failed_cases"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

