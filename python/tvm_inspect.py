#!/usr/bin/env python3
"""
Inspect the exported WaveAccel ONNX model through Apache TVM Relax and verify
that TVM's canonical compiler graph matches the expected network structure.

Expected Relax canonicalization for the current WaveAccel MLP:

    3 x permute_dims
    3 x matmul
    4 x add
    2 x relu
    1 x multiply

The three ONNX Gemm nodes are expected to become:

    permute_dims(weight) -> matmul -> add(bias)

Author: Santosh Kumar
Copyright 2026 Santosh Kumar
SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import argparse
import re
from collections import OrderedDict
from pathlib import Path

import onnx
import tvm
from tvm.relax.frontend.onnx import from_onnx


EXPECTED_OP_COUNTS = OrderedDict(
    [
        ("permute_dims", 3),
        ("matmul", 3),
        ("add", 4),
        ("relu", 2),
        ("multiply", 1),
    ]
)


def count_relax_ops(script_text: str) -> dict[str, int]:
    """Count selected Relax operator calls in TVM Script text."""
    counts: dict[str, int] = {}

    patterns = {
        "permute_dims": r"\bR\.permute_dims\(",
        "matmul": r"\bR\.matmul\(",
        "add": r"\bR\.add\(",
        "relu": r"\bR\.nn\.relu\(",
        "multiply": r"\bR\.multiply\(",
    }

    for name, pattern in patterns.items():
        counts[name] = len(re.findall(pattern, script_text))

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect WaveAccel ONNX through TVM Relax."
    )
    parser.add_argument(
        "--onnx",
        default="artifacts/wave_model.onnx",
        help="Path to WaveAccel ONNX model.",
    )
    parser.add_argument(
        "--show-meta",
        action="store_true",
        help="Include TVM constant metadata in the printed IR.",
    )
    args = parser.parse_args()

    onnx_path = Path(args.onnx)

    if not onnx_path.is_file():
        raise FileNotFoundError(
            f"ONNX model not found: {onnx_path}"
        )

    print("WaveAccel TVM inspection")
    print("=" * 72)
    print(f"TVM version: {tvm.__version__}")
    print(f"ONNX model:  {onnx_path}")

    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)

    mod = from_onnx(
        model,
        shape_dict={"wave_samples": [1, 64]},
        dtype_dict={"wave_samples": "float32"},
    )

    script_text = mod.script(
        show_meta=args.show_meta
    )

    print("\n================ TVM RELAX IR ================\n")
    print(script_text)

    counts = count_relax_ops(script_text)

    print("\n================ OP VERIFICATION =============\n")

    passed = True

    for op_name, expected in EXPECTED_OP_COUNTS.items():
        actual = counts[op_name]
        status = "PASS" if actual == expected else "FAIL"

        print(
            f"{op_name:14s} "
            f"expected={expected} "
            f"actual={actual} "
            f"{status}"
        )

        if actual != expected:
            passed = False

    if not passed:
        raise RuntimeError(
            "TVM Relax operator structure does not match "
            "the expected WaveAccel model."
        )

    print("\nCanonicalization:")
    print("  ONNX Gemm x3")
    print("      -> TVM permute_dims x3")
    print("      -> TVM matmul       x3")
    print("      -> TVM add(bias)    x3")
    print("  ReLU x2 -> TVM relu x2")
    print("  target_std  -> TVM multiply x1")
    print("  target_mean -> TVM add x1")

    print("\nTVM ONNX graph verification: PASS")


if __name__ == "__main__":
    main()
