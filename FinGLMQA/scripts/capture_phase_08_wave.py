#!/usr/bin/env python3
"""Capture deterministic file state at a Phase 8 ownership wave boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def state(path_text: str) -> dict:
    path = ROOT / path_text
    if not path.exists():
        return {"path": path_text, "exists": False, "size": None, "mtime_ns": None, "sha256": None}
    stat = path.stat()
    return {
        "path": path_text,
        "exists": True,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wave", required=True)
    parser.add_argument("--boundary", required=True, choices=("before", "after"))
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()
    payload = {
        "schema_version": "finglmqa.phase8.wave_file_state.v1",
        "wave": args.wave,
        "boundary": args.boundary,
        "files": [state(path) for path in sorted(set(args.paths))],
    }
    destination = ROOT / f"runs/phase_08/waves/{args.wave}_{args.boundary}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=f".{destination.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    print(destination.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
