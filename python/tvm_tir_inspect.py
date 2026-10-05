#!/usr/bin/env python3
"""
Inspect the low-level TVM PrimFuncs generated from the WaveAccel ONNX model.

This version targets Apache TVM 0.27's newer TIRx/S-TIR stack.  It deliberately
avoids legacy `tvm.tir` type checks and legacy PrimFunc `buffer_map` access.

For this WaveAccel model, LegalizeOps generates clearly named PrimFuncs:
    transpose, matmul, add, relu,
    transpose1, matmul1, add1, relu1,
    transpose2, matmul2, add2, multiply

We inspect the three matmul PrimFuncs directly and print their low-level loop IR.

Author: Santosh Kumar
Copyright 2026 Santosh Kumar
SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
import tvm
from tvm.relax.frontend.onnx import from_onnx


EXPECTED_MATMUL_NAMES = ("matmul", "matmul1", "matmul2")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect WaveAccel TVM 0.27 PrimFuncs."
    )
    parser.add_argument(
        "--onnx",
        default="artifacts/wave_model.onnx",
        help="Path to the WaveAccel ONNX model.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Print every generated PrimFunc, not only matmul kernels.",
    )
    args = parser.parse_args()

    onnx_path = Path(args.onnx)

    if not onnx_path.is_file():
        raise FileNotFoundError(
            f"ONNX model not found: {onnx_path}"
        )

    print("WaveAccel TVM 0.27 PrimFunc inspection")
    print("=" * 72)
    print(f"TVM version: {tvm.__version__}")
    print(f"ONNX model:  {onnx_path}")

    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)

    relax_mod = from_onnx(
        model,
        shape_dict={"wave_samples": [1, 64]},
        dtype_dict={"wave_samples": "float32"},
    )

    legalized = tvm.relax.transform.LegalizeOps()(
        relax_mod
    )

    primfuncs = {}

    print("\n================ FUNCTION TYPES ========================\n")

    for global_var, function in legalized.functions.items():
        name = global_var.name_hint
        type_name = type(function).__name__

        print(f"{name:30s} type={type_name}")

        if type_name == "PrimFunc":
            primfuncs[name] = function

    print(
        f"\nGenerated PrimFuncs detected: {len(primfuncs)}"
    )

    if not primfuncs:
        raise RuntimeError(
            "No PrimFuncs found after Relax LegalizeOps."
        )

    missing = [
        name
        for name in EXPECTED_MATMUL_NAMES
        if name not in primfuncs
    ]

    if missing:
        raise RuntimeError(
            "Expected matmul PrimFuncs were not generated: "
            + ", ".join(missing)
        )

    print("\n================ MATMUL PRIMFUNCS ======================\n")

    selected_names = (
        list(primfuncs.keys())
        if args.full
        else list(EXPECTED_MATMUL_NAMES)
    )

    for name in selected_names:
        function = primfuncs[name]

        print(f"\n----- {name} -----\n")
        print(function.script())

    print("\n================ EXPECTED LOGICAL SHAPES ===============\n")
    print("matmul  : 1x64  x 64x32 -> 1x32")
    print("matmul1 : 1x32  x 32x16 -> 1x16")
    print("matmul2 : 1x16  x 16x2  -> 1x2")

    print("\n================ FIRST MATMUL MENTAL MODEL =============\n")
    print("Conceptually:")
    print("")
    print("  for output_col in 0..31:")
    print("      acc = 0")
    print("      for k in 0..63:")
    print("          acc += input[k] * weight[k, output_col]")
    print("")
    print("WaveAccel's validated 4 KiB runtime tiles those 32 output")
    print("channels as:")
    print("")
    print("  tile 0 -> outputs  0..12  (13 rows)")
    print("  tile 1 -> outputs 13..25  (13 rows)")
    print("  tile 2 -> outputs 26..31  ( 6 rows)")
    print("")
    print("The next compiler step is to identify the corresponding")
    print("TVM output/reduction loops and schedule the output loop.")

    print("\nTVM PrimFunc inspection: PASS")


if __name__ == "__main__":
    main()
