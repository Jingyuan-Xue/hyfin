#!/usr/bin/env python3
"""Build and validate the non-destructive Phase 11 closeout artifacts."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
PHASE11 = ROOT / "runs" / "phase_11"
REPORTS = PHASE11 / "reports"
DATE = "2026-07-14"

STANDARD_SECTIONS = (
    "Date",
    "Environment",
    "Inputs",
    "Changed Components",
    "Generated Artifacts",
    "Verification Commands",
    "Verification Results",
    "Issues Encountered",
    "Discarded / Archived Artifacts",
    "Subagents Used",
    "User Confirmations Needed",
    "Decision",
)

PHASE_DATES = {
    1: "2026-07-02",
    2: "2026-07-03",
    3: "2026-07-03",
    4: "2026-07-03",
    5: "2026-07-03",
    6: "2026-07-13",
    7: "2026-07-13",
    8: "2026-07-13",
    9: "2026-07-14",
    10: "2026-07-14",
}

CRITICAL_PATHS = tuple(
    [f"runs/phase_{phase:02d}/phase_{phase:02d}_report.md" for phase in range(2, 11)]
    + [
        "runs/phase_03/phase_03_run_report.json",
        "runs/phase_06/phase_06_run_report.json",
        "runs/phase_06/repeatability_report.json",
        "runs/phase_07/build_report.json",
        "runs/phase_07/repeatability_report.json",
        "runs/phase_08/immutable_inputs_manifest.json",
        "runs/phase_08/pattern_registry_manifest.json",
        "runs/phase_08/supported_fixture_manifest.json",
        "runs/phase_09/immutable_inputs_manifest.json",
        "runs/phase_09/dependency_manifest.json",
        "runs/phase_10/immutable_inputs_manifest.json",
        "runs/phase_10/dependency_manifest.json",
        "runs/phase_10/release_manifest.json",
    ]
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value))


def snapshot() -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for relative in CRITICAL_PATHS:
        path = ROOT / relative
        if not path.is_file():
            raise RuntimeError(f"critical Phase 1-10 file is missing: {relative}")
        stat = path.stat()
        entries.append(
            {
                "mtime_ns": stat.st_mtime_ns,
                "path": relative,
                "sha256": sha256(path),
                "size": stat.st_size,
            }
        )
    semantic = hashlib.sha256(canonical_bytes(entries)).hexdigest()
    return {
        "entries": entries,
        "entry_count": len(entries),
        "schema_version": "finglmqa.phase11.critical_snapshot.v1",
        "semantic_sha256": semantic,
    }


def capture_before() -> None:
    target = REPORTS / "critical_snapshot_before.json"
    if target.exists():
        raise RuntimeError(f"refusing to replace existing before snapshot: {target}")
    write_json(target, snapshot())


def markdown_headings(path: Path) -> dict[str, str]:
    headings: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^##\s+(.+?)\s*$", line)
        if match:
            name = match.group(1).strip()
            headings[name.casefold()] = name
    return headings


def anchor(name: str) -> str:
    value = name.strip().lower()
    value = re.sub(r"[^\w\s/-]", "", value)
    return re.sub(r"[\s/]+", "-", value).strip("-")


def supplement_text(phase: int, missing: Iterable[str]) -> str:
    primary = f"runs/phase_{phase:02d}/phase_{phase:02d}_report.md"
    lines = [
        f"# Phase {phase} Standard-Section Supplement",
        "",
        "This Phase 11 supplement supplies standard report headings omitted by the",
        f"historical `{primary}`. It does not amend the historical report, rerun the",
        "phase, or introduce new claims. The primary report and the corresponding",
        "entry in `docs/PROGRESS.md` remain authoritative for phase-specific detail.",
        "",
    ]
    content = {
        "Date": f"- Recorded phase date: {PHASE_DATES[phase]}.",
        "Environment": (
            "- Use the environment recorded by the primary report and `docs/PROGRESS.md`; "
            "Phase 11 did not reactivate or revalidate that historical runtime."
        ),
        "Inputs": (
            "- Inputs are those identified by the primary report, predecessor phase, and "
            "artifact manifest; Phase 11 adds none."
        ),
        "Changed Components": (
            "- The primary report's implementation/outcome sections and the Phase completion "
            "entry in `docs/PROGRESS.md` are the authoritative change record."
        ),
        "Generated Artifacts": (
            "- Generated outputs are enumerated by the primary report and "
            "`docs/ARTIFACTS_MANIFEST.md`; this supplement generates no phase output."
        ),
        "Verification Commands": (
            "- Use only commands retained in the primary report or phase Gate reports. "
            "Phase 11 does not claim an unrecorded historical command."
        ),
        "Verification Results": (
            "- The primary report and Gate artifacts retain the historical result. Phase 11 "
            "checks their presence and integrity, not the original runtime."
        ),
        "Issues Encountered": (
            "- Resolved and open findings are recorded in `docs/ISSUES.md`; no historical "
            "issue is reclassified by this supplement."
        ),
        "Discarded / Archived Artifacts": (
            "- No historical artifact is moved or deleted. Phase 11 disposition is recorded "
            "separately in `runs/phase_11/reports/artifact_disposition.jsonl`."
        ),
        "Subagents Used": (
            "- Refer to the primary report and `docs/PROGRESS.md`. Absence of a retained entry "
            "is reported as not recorded, not as proof that none was used."
        ),
        "User Confirmations Needed": (
            "- No outstanding confirmation is identified for this completed phase in current "
            "governance; open future work remains governed by `docs/ISSUES.md`."
        ),
        "Decision": f"- Phase {phase} remains complete; this supplement does not change its decision.",
    }
    for section in missing:
        lines.extend([f"## {section}", "", content[section], ""])
    return "\n".join(lines).rstrip() + "\n"


def build_report_coverage() -> None:
    phases: list[dict[str, Any]] = []
    for phase in range(1, 11):
        primary_rel = f"runs/phase_{phase:02d}/phase_{phase:02d}_report.md"
        primary = ROOT / primary_rel
        if not primary.is_file():
            raise RuntimeError(f"missing primary phase report: {primary_rel}")
        headings = markdown_headings(primary)
        missing = [name for name in STANDARD_SECTIONS if name.casefold() not in headings]
        supplement_rel: str | None = None
        if phase >= 2 and missing:
            supplement_rel = f"runs/phase_11/reports/phase_{phase:02d}_report_supplement.md"
            (ROOT / supplement_rel).write_text(
                supplement_text(phase, missing), encoding="utf-8", newline="\n"
            )
        sections: list[dict[str, Any]] = []
        for name in STANDARD_SECTIONS:
            original = headings.get(name.casefold())
            if original:
                sections.append(
                    {
                        "section": name,
                        "source_anchor": anchor(original),
                        "source_heading": original,
                        "source_kind": "primary_report",
                        "source_path": primary_rel,
                    }
                )
            elif supplement_rel:
                sections.append(
                    {
                        "section": name,
                        "source_anchor": anchor(name),
                        "source_heading": name,
                        "source_kind": "phase11_supplement",
                        "source_path": supplement_rel,
                    }
                )
            else:
                raise RuntimeError(f"Phase {phase} has uncovered standard section: {name}")
        phases.append(
            {
                "all_12_sections_covered": len(sections) == len(STANDARD_SECTIONS),
                "historical_report_unchanged": phase >= 2,
                "phase": phase,
                "primary_report": primary_rel,
                "sections": sections,
                "supplement": supplement_rel,
            }
        )
    result = {
        "all_phases_complete": all(item["all_12_sections_covered"] for item in phases),
        "phase_count": len(phases),
        "phases": phases,
        "schema_version": "finglmqa.phase11.report_coverage.v1",
        "standard_sections": list(STANDARD_SECTIONS),
    }
    write_json(REPORTS / "report_coverage.json", result)


def iter_path_values(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "path" and isinstance(child, str):
                yield child
            yield from iter_path_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_path_values(child)


def manifest_reverse_references() -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    candidates = sorted((ROOT / "runs").glob("phase_*/**/*manifest*.json"))
    candidates += sorted((ROOT / "runs").glob("phase_*/**/release_manifest.json"))
    for manifest in dict.fromkeys(candidates):
        try:
            value = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        manifest_rel = manifest.relative_to(ROOT).as_posix()
        for target in iter_path_values(value):
            if target.startswith("external:"):
                continue
            result.setdefault(target, []).append(manifest_rel)
    return {key: sorted(set(value)) for key, value in result.items()}


def artifact_type(path: Path) -> str:
    names = {
        ".md": "markdown",
        ".json": "json",
        ".jsonl": "jsonl",
        ".duckdb": "duckdb",
        ".log": "log",
        ".lock": "lock",
    }
    return names.get(path.suffix.lower(), path.suffix.lower().lstrip(".") or "file")


def disposition(relative: str) -> tuple[str, bool, bool, str]:
    if relative.startswith("runs/phase_10/runtime/"):
        return (
            "runtime_ephemeral",
            True,
            True,
            "Active service runtime state; inventoried in place and excluded from release cleanup.",
        )
    if "/repeatability/run2/" in relative or "/phase7_revalidation_run2/" in relative:
        return (
            "archive",
            True,
            False,
            "Secondary repeatability copy; retained in place pending separate cleanup authorization.",
        )
    rebuildable = any(
        token in relative
        for token in ("/gate", "/reports/", "/waves/", "/repeatability/", "_report.json")
    )
    return (
        "keep",
        rebuildable,
        False,
        "Authoritative report, release, audit, provenance, or reproducibility artifact.",
    )


def build_artifact_disposition() -> None:
    reverse = manifest_reverse_references()
    records: list[dict[str, Any]] = []
    for phase in range(1, 11):
        base = ROOT / "runs" / f"phase_{phase:02d}"
        if not base.exists():
            continue
        for path in sorted(item for item in base.rglob("*") if item.is_file()):
            relative = path.relative_to(ROOT).as_posix()
            category, rebuildable, runtime_active, reason = disposition(relative)
            stat = path.stat()
            records.append(
                {
                    "disposition": category,
                    "manifest_references": reverse.get(relative, []),
                    "path": relative,
                    "phase": phase,
                    "reason": reason,
                    "rebuildable": rebuildable,
                    "runtime_active": runtime_active,
                    "sha256": sha256(path),
                    "size": stat.st_size,
                    "type": artifact_type(path),
                }
            )
    target = REPORTS / "artifact_disposition.jsonl"
    target.write_bytes(b"".join(canonical_bytes(record) for record in records))
    counts: dict[str, int] = {}
    for record in records:
        counts[record["disposition"]] = counts.get(record["disposition"], 0) + 1
    write_json(
        REPORTS / "artifact_disposition_summary.json",
        {
            "counts_by_disposition": counts,
            "deleted_or_moved_count": 0,
            "record_count": len(records),
            "schema_version": "finglmqa.phase11.artifact_disposition_summary.v1",
        },
    )


def artifact_manifest_paths() -> tuple[list[str], list[str]]:
    text = (ROOT / "docs" / "ARTIFACTS_MANIFEST.md").read_text(encoding="utf-8")
    candidates: set[str] = set()
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        for token in re.findall(r"`([^`]+)`", line):
            if token.startswith(
                (
                    "README.md",
                    ".venv",
                    "requirements/",
                    "env/",
                    "refs/",
                    "src/",
                    "scripts/",
                    "data/",
                    "runs/",
                    "logs/",
                    "tests/",
                )
            ):
                candidates.add(token.rstrip(".,"))
    missing: list[str] = []
    checked: list[str] = []
    planned_self_outputs = {
        "runs/phase_11/release_manifest.json",
        "runs/phase_11/reports/governance_validation.json",
    }
    for candidate in sorted(candidates):
        checked.append(candidate)
        resolved = str(ROOT / candidate)
        if candidate in planned_self_outputs:
            # These two paths are produced by the same atomic closeout command.
            # Their bytes are verified immediately after finalization.
            exists = True
        elif any(char in candidate for char in "*?["):
            exists = bool(glob.glob(resolved))
        else:
            exists = os.path.lexists(resolved)
        if not exists:
            missing.append(candidate)
    return checked, missing


def load_json(relative: str) -> dict[str, Any]:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def verify_release_manifest(relative: str) -> list[str]:
    manifest = load_json(relative)
    failures: list[str] = []
    artifacts = manifest.get("artifacts", [])
    if not isinstance(artifacts, list):
        return ["artifacts:not_array"]
    paths = [entry.get("path") for entry in artifacts if isinstance(entry, dict)]
    if len(paths) != len(artifacts):
        failures.append("artifacts:invalid_entry")
    if len(paths) != len(set(paths)):
        failures.append("artifacts:duplicate_path")
    if (
        manifest.get("schema_version") == "finglmqa.phase11.release_manifest.v1"
        and paths != sorted(paths)
    ):
        failures.append("artifacts:noncanonical_order")
    for entry in artifacts:
        if not isinstance(entry, dict) or set(entry) != {"path", "sha256", "size"}:
            failures.append("artifacts:invalid_shape")
            continue
        path = ROOT / entry["path"]
        if not path.is_file():
            failures.append(f"missing:{entry['path']}")
            continue
        if path.stat().st_size != entry["size"]:
            failures.append(f"size:{entry['path']}")
        if sha256(path) != entry["sha256"]:
            failures.append(f"sha256:{entry['path']}")
    recorded_semantic = manifest.get("manifest_semantic_sha256")
    if manifest.get("schema_version") == "finglmqa.phase11.release_manifest.v1":
        expected_semantic = hashlib.sha256(canonical_bytes(artifacts)).hexdigest()
    else:
        semantic_payload = dict(manifest)
        semantic_payload.pop("manifest_semantic_sha256", None)
        expected_semantic = hashlib.sha256(canonical_bytes(semantic_payload)).hexdigest()
    if recorded_semantic != expected_semantic:
        failures.append("manifest_semantic_sha256")
    return failures


def governance_validation() -> dict[str, Any]:
    gate6 = load_json("runs/phase_10/gate6_report.json")
    gate7 = load_json("runs/phase_10/gate7_http_report.json")
    gate8 = load_json("runs/phase_10/gate8_report.json")
    promotion = load_json("runs/phase_10/promotion_readiness.json")
    coverage = load_json("runs/phase_11/reports/report_coverage.json")
    issues = (ROOT / "docs" / "ISSUES.md").read_text(encoding="utf-8")
    decisions = (ROOT / "docs" / "DECISIONS.md").read_text(encoding="utf-8")
    progress = (ROOT / "docs" / "PROGRESS.md").read_text(encoding="utf-8")
    checked_paths, missing_paths = artifact_manifest_paths()
    phase10_gate_files = [
        *(f"runs/phase_10/gate{gate}_report.json" for gate in range(0, 7)),
        "runs/phase_10/gate6_http_report.json",
        "runs/phase_10/gate7_http_report.json",
        "runs/phase_10/gate8_report.json",
    ]
    phase10_gate_statuses = {
        path: load_json(path).get("status")
        for path in phase10_gate_files
        if (ROOT / path).is_file()
    }
    normalized_issues = " ".join(issues.split())
    normalized_decisions = " ".join(decisions.split())
    normalized_progress = " ".join(progress.split())
    checks = {
        "all_phase_reports_exist": all(
            (ROOT / f"runs/phase_{phase:02d}/phase_{phase:02d}_report.md").is_file()
            for phase in range(1, 11)
        ),
        "artifact_manifest_paths_exist": not missing_paths,
        "phase10_1003_benchmark_executed": gate7["checks"]["benchmark_1003_executed"],
        "phase10_40_general_executed": gate7["checks"]["general_40_executed"],
        "phase10_all_gate_reports_present": all((ROOT / path).is_file() for path in phase10_gate_files),
        "phase10_all_gate_reports_passed": len(phase10_gate_statuses) == len(phase10_gate_files)
        and all(status == "passed" for status in phase10_gate_statuses.values()),
        "phase10_gate8_passed": gate8["status"] == "passed",
        "phase10_promotion_false": promotion["promotion_readiness"] is False,
        "phase10_release_manifest_valid": not verify_release_manifest(
            "runs/phase_10/release_manifest.json"
        ),
        "phase8_171_tests": gate6["checks"]["phase8_171_tests_pass"],
        "phase9_14_tests": gate6["checks"]["phase9_14_tests_pass"],
        "report_coverage_10_by_12": coverage["all_phases_complete"]
        and coverage["phase_count"] == 10
        and all(len(item["sections"]) == 12 for item in coverage["phases"]),
        "phase12_block_consistent": all(
            marker in text
            for text, marker in (
                (normalized_issues, "open_blocking_phase12"),
                (normalized_issues, "Phase 12 must not start"),
                (normalized_decisions, "Phase 12"),
                (normalized_decisions, "promotion_readiness=false"),
                (normalized_progress, "Phase 12"),
                (normalized_progress, "promotion_readiness=false"),
            )
        ),
    }
    result = {
        "all_checks_passed": all(checks.values()),
        "artifact_paths_checked": len(checked_paths),
        "checks": checks,
        "details": {
            "artifact_manifest_missing_paths": missing_paths,
            "phase10_gate_files": phase10_gate_files,
            "phase10_gate_statuses": phase10_gate_statuses,
            "phase10_promotion_readiness": promotion["promotion_readiness"],
        },
        "schema_version": "finglmqa.phase11.governance_validation.v1",
    }
    write_json(REPORTS / "governance_validation.json", result)
    return result


def compare_snapshots() -> dict[str, Any]:
    before = load_json("runs/phase_11/reports/critical_snapshot_before.json")
    after = snapshot()
    write_json(REPORTS / "critical_snapshot_after.json", after)
    before_by_path = {entry["path"]: entry for entry in before["entries"]}
    after_by_path = {entry["path"]: entry for entry in after["entries"]}
    changed: list[dict[str, Any]] = []
    for path in sorted(set(before_by_path) | set(after_by_path)):
        left = before_by_path.get(path)
        right = after_by_path.get(path)
        if left != right:
            changed.append({"after": right, "before": left, "path": path})
    result = {
        "all_critical_phase1_10_files_unchanged": not changed,
        "changed": changed,
        "compared_count": len(before_by_path),
        "new_phase1_report_excluded_by_authorization": True,
        "schema_version": "finglmqa.phase11.upstream_drift_validation.v1",
    }
    write_json(REPORTS / "upstream_drift_validation.json", result)
    return result


def build() -> None:
    build_report_coverage()
    build_artifact_disposition()


def finalize() -> None:
    drift = compare_snapshots()
    governance = governance_validation()
    if not drift["all_critical_phase1_10_files_unchanged"]:
        raise RuntimeError("critical Phase 1-10 files drifted")
    if not governance["all_checks_passed"]:
        raise RuntimeError(f"governance validation failed: {governance['checks']}")

    release_paths = [
        "runs/phase_01/phase_01_report.md",
        "runs/phase_11/phase_11_plan.md",
        "runs/phase_11/phase_11_report.md",
        "scripts/finalize_phase_11.py",
        "docs/PROGRESS.md",
        "docs/ISSUES.md",
        "docs/DECISIONS.md",
        "docs/ARTIFACTS_MANIFEST.md",
    ]
    release_paths.extend(
        path.relative_to(ROOT).as_posix()
        for path in sorted(REPORTS.iterdir())
        if path.is_file()
    )
    artifacts: list[dict[str, Any]] = []
    for relative in sorted(set(release_paths)):
        path = ROOT / relative
        if not path.is_file():
            raise RuntimeError(f"Phase 11 release artifact missing: {relative}")
        artifacts.append(
            {"path": relative, "sha256": sha256(path), "size": path.stat().st_size}
        )
    semantic = hashlib.sha256(canonical_bytes(artifacts)).hexdigest()
    write_json(
        PHASE11 / "release_manifest.json",
        {
            "artifacts": artifacts,
            "manifest_semantic_sha256": semantic,
            "phase12_authorized": False,
            "schema_version": "finglmqa.phase11.release_manifest.v1",
            "upstream_drift_count": len(drift["changed"]),
        },
    )


def validate_release() -> None:
    failures = verify_release_manifest("runs/phase_11/release_manifest.json")
    if failures:
        raise RuntimeError(f"Phase 11 release verification failed: {failures}")
    manifest = load_json("runs/phase_11/release_manifest.json")
    if manifest.get("phase12_authorized") is not False:
        raise RuntimeError("Phase 11 release unexpectedly authorizes Phase 12")
    governance = load_json("runs/phase_11/reports/governance_validation.json")
    drift = load_json("runs/phase_11/reports/upstream_drift_validation.json")
    if not governance["all_checks_passed"] or not drift[
        "all_critical_phase1_10_files_unchanged"
    ]:
        raise RuntimeError("Phase 11 closeout reports are not passing")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=("capture-before", "build", "finalize", "validate-release")
    )
    args = parser.parse_args()
    if args.command == "capture-before":
        capture_before()
    elif args.command == "build":
        build()
    elif args.command == "finalize":
        finalize()
    else:
        validate_release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
