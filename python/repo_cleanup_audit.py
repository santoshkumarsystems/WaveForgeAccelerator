#!/usr/bin/env python3
"""
Non-destructive final-release cleanup audit for WaveAccel.

Reports cleanup candidates only. Deletes nothing.

Excluded from recursive scanning because they are dependency/build trees:
  .git, .venv, venv, build, _deps, CMakeFiles, node_modules

Author: Santosh Kumar
Copyright 2026 Santosh Kumar
SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


PRUNE_DIRS = {
    ".git",
    ".venv",
    "venv",
    "build",
    "_deps",
    "CMakeFiles",
    "node_modules",
}

CANDIDATE_SUFFIXES = (
    ".zip",
    ".bak",
    ".pyc",
    ".engine",
    ".trt.engine",
)


def should_report_file(path: Path) -> bool:
    name = path.name
    if name.endswith(CANDIDATE_SUFFIXES):
        return True
    if ".pre_" in name and name.endswith(".bak"):
        return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=".")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()

    candidates: set[Path] = set()

    for current_root, dirnames, filenames in os.walk(repo):
        dirnames[:] = [
            d for d in dirnames
            if d not in PRUNE_DIRS
        ]

        current = Path(current_root)

        # __pycache__ itself is a cleanup candidate; do not descend into it.
        if "__pycache__" in dirnames:
            candidates.add(current / "__pycache__")
            dirnames.remove("__pycache__")

        for filename in filenames:
            path = current / filename
            if should_report_file(path):
                candidates.add(path)

    print("WaveAccel final-release cleanup audit")
    print("=" * 72)
    print(
        "Excluded trees: "
        + ", ".join(sorted(PRUNE_DIRS))
    )

    if not candidates:
        print("\nNo obvious temporary/generated cleanup candidates found.")
    else:
        print("\nCandidates (nothing deleted):")
        for path in sorted(candidates):
            print(f"  {path.relative_to(repo)}")

    required = [
        "artifacts/wave_model.onnx",
        "artifacts/wave_model.onnx.data",
        "python/tensorrt_reference.py",
        "docs/tensorrt_reference.md",
        "python/waveaccel_target_contract.py",
        "python/tvm_waveaccel_partition.py",
        "python/tvm_byoc_execution_plan_bridge.py",
        "python/tvm_emit_waveaccel_compiled_plan.py",
    ]

    print("\nRequired final artifacts:")
    missing = []

    for rel in required:
        item = repo / rel
        state = "OK" if item.exists() else "MISSING"
        print(f"  {state:7s} {rel}")
        if not item.exists():
            missing.append(rel)

    if missing:
        raise SystemExit(
            "\nRelease audit incomplete: required files are missing."
        )

    print("\nRelease cleanup audit: READY FOR REVIEW")


if __name__ == "__main__":
    main()
