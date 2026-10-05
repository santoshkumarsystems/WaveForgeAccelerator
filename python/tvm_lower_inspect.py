#!/usr/bin/env python3
"""
Lower the WaveAccel TVM Relax graph into low-level TIR PrimFuncs and inspect
the compiler boundary without changing the existing C++ WaveAccel backend.

Development principle:
    prove each compiler layer independently before integrating it with the
    known-correct runtime/backend.

Pipeline inspected by this script:

    ONNX
      -> TVM Relax
      -> relax.transform.LegalizeOps
      -> Relax call_tir sites + generated low-level PrimFuncs

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect WaveAccel Relax -> TIR legalization."
    )
    parser.add_argument(
        "--onnx",
        default="artifacts/wave_model.onnx",
        help="Path to the WaveAccel ONNX model.",
    )
    parser.add_argument(
        "--show-relax",
        action="store_true",
        help="Also print the pre-legalization Relax IR.",
    )
    parser.add_argument(
        "--show-meta",
        action="store_true",
        help="Include TVM metadata when printing IR.",
    )
    args = parser.parse_args()

    onnx_path = Path(args.onnx)

    if not onnx_path.is_file():
        raise FileNotFoundError(
            f"ONNX model not found: {onnx_path}"
        )

    print("WaveAccel TVM lowering inspection")
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

    if args.show_relax:
        print(
            "\n================ PRE-LEGALIZATION RELAX IR ================\n"
        )
        print(
            relax_mod.script(
                show_meta=args.show_meta
            )
        )

    # Official TVM Relax lowering step:
    # high-level Relax operators -> R.call_tir + generated low-level PrimFuncs.
    legalized_mod = tvm.relax.transform.LegalizeOps()(
        relax_mod
    )

    legalized_text = legalized_mod.script(
        show_meta=args.show_meta
    )

    print(
        "\n================ LEGALIZED RELAX + TIR ================\n"
    )
    print(legalized_text)

    print(
        "\n================ MODULE FUNCTIONS ======================\n"
    )

    function_count = 0

    for global_var, function in legalized_mod.functions.items():
        function_count += 1
        print(
            f"{global_var.name_hint:30s} "
            f"type={type(function).__name__}"
        )

    call_tir_count = legalized_text.count("R.call_tir(")

    # TVM's textual syntax can use different namespace aliases as the project
    # evolves, so this is informational rather than a brittle hard assertion.
    prim_func_markers = (
        legalized_text.count(".prim_func(")
        + legalized_text.count("@T.prim_func")
        + legalized_text.count("@Ts.prim_func")
    )

    print(
        "\n================ LOWERING SUMMARY ======================\n"
    )
    print(f"module functions:      {function_count}")
    print(f"Relax call_tir sites:  {call_tir_count}")
    print(
        "printed prim_func markers: "
        f"{prim_func_markers} "
        "(informational)"
    )

    if call_tir_count == 0:
        raise RuntimeError(
            "LegalizeOps produced no visible R.call_tir sites; "
            "inspect the emitted IR/API behavior before proceeding."
        )

    print("\nWhat changed:")
    print("  BEFORE: Relax matmul/add/relu/multiply/permute_dims")
    print("  AFTER:  Relax call_tir sites invoke low-level generated kernels")
    print("  C++ WaveAccel backend remains unchanged")

    print("\nTVM Relax -> TIR legalization: PASS")


if __name__ == "__main__":
    main()
