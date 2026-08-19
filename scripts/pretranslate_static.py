#!/usr/bin/env python3
"""Populate the reviewed static dictionary through the configured translator."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from icdm_demo.translation_service import TranslationService

STATIC_PATH = ROOT / "i18n/static_zh_en.json"
QA_INDEX = ROOT / "FinGLMQA/data/corpus_package/company_year_index.jsonl"
TARGETS = ROOT / "output/refactor_pipeline_430101/run/baseline_shenwan_430101_selected25.csv"
LABELS = ROOT / "output/refactor_pipeline_430101/rerun_fixed_20260722/final_best_1000/labels.jsonl"
CJK = re.compile(r"[\u3400-\u9fff]")


def jsonl(path: Path):
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


def source_terms() -> list[str]:
    values: set[str] = set()
    for row in jsonl(QA_INDEX) or []:
        values.update(filter(None, (row.get("stock_name"), row.get("company_full"))))
    if TARGETS.is_file():
        with TARGETS.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                values.add(str(row.get("company_name") or "").strip())
    for row in jsonl(LABELS) or []:
        values.add(str(row.get("label") or "").strip())
    return sorted(value for value in values if value and CJK.search(value))


def write_atomic(payload: dict) -> None:
    STATIC_PATH.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=STATIC_PATH.parent, delete=False) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, STATIC_PATH)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="report missing entries without translating")
    args = parser.parse_args()
    payload = json.loads(STATIC_PATH.read_text(encoding="utf-8")) if STATIC_PATH.is_file() else {}
    mapping = dict(payload.get("zh_en") or {})
    missing = [value for value in source_terms() if not mapping.get(value)]
    print(f"Static entries: {len(mapping)}")
    print(f"Missing source terms: {len(missing)}")
    if args.check or not missing:
        return 1 if missing else 0

    translator = TranslationService(ROOT)
    if not translator.configured:
        print("Tencent translation is not configured; set the server-side environment variables first.")
        return 2
    translated = translator.translate_many(missing, "zh", "en")
    failed = []
    for source, target in zip(missing, translated):
        if not target or CJK.search(target):
            failed.append(source)
            continue
        mapping[source] = target
    payload.update(
        schema_version="fingraphrag.i18n.static.v1",
        generated_by=translator.provider,
        zh_en=mapping,
        en_zh=dict(payload.get("en_zh") or {}),
    )
    write_atomic(payload)
    print(f"Added entries: {len(missing) - len(failed)}")
    print(f"Untranslated entries: {len(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
