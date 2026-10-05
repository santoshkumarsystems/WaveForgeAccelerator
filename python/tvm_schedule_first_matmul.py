#!/usr/bin/env python3
"""
Apply the first explicit WaveAccel-aware TVM schedule transformation.

This script takes the first TVM-generated matmul:

    (1 x 64) @ (64 x 32) -> (1 x 32)

and tiles the 32-output spatial loop with an inner extent of 13.

Why 13?
    WaveAccel 4 KiB SRAM:
        4096 B total SRAM
        - 392 B transient reservation
        - 128 B bias
        = 3576 B available for the first GEMM weight tile

    One logical output channel needs:
        64 FP32 weights * 4 B = 256 B

    floor(3576 / 256) = 13 output channels per tile.

This reproduces the compiler-loop counterpart of the already validated
WaveAccel runtime plan:

    outputs  0..12 -> 13 channels -> 3328 B
    outputs 13..25 -> 13 channels -> 3328 B
    outputs 26..31 ->  6 channels -> 1536 B

Important:
    This step changes the TVM loop schedule only. It does NOT yet introduce
    explicit SRAM cache buffers, DMA operations, or WaveAccel code generation.
    Those come later.

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


OUTPUT_CHANNELS = 32
REDUCTION_K = 64
FP32_BYTES = 4
SRAM_BYTES = 4096
TRANSIENT_BYTES = 392
BIAS_BYTES = 128


def get_global_function(mod, name: str):
    """Return a named global function from a TVM IRModule."""
    for global_var, function in mod.functions.items():
        if global_var.name_hint == name:
            return function

    raise KeyError(f"TVM function not found: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tile WaveAccel's first TVM matmul output loop."
    )
    parser.add_argument(
        "--onnx",
        default="artifacts/wave_model.onnx",
        help="Path to the WaveAccel ONNX model.",
    )
    args = parser.parse_args()

    onnx_path = Path(args.onnx)

    if not onnx_path.is_file():
        raise FileNotFoundError(
            f"ONNX model not found: {onnx_path}"
        )

    weight_capacity = (
        SRAM_BYTES
        - TRANSIENT_BYTES
        - BIAS_BYTES
    )

    bytes_per_output = REDUCTION_K * FP32_BYTES
    outputs_per_tile = weight_capacity // bytes_per_output

    if outputs_per_tile <= 0:
        raise RuntimeError(
            "SRAM cannot hold one output-channel weight slice"
        )

    tile_sizes = []

    remaining = OUTPUT_CHANNELS

    while remaining > 0:
        current = min(outputs_per_tile, remaining)
        tile_sizes.append(current)
        remaining -= current

    print("WaveAccel TVM schedule: first matmul")
    print("=" * 72)
    print(f"TVM version:                {tvm.__version__}")
    print(f"ONNX model:                 {onnx_path}")
    print(f"SRAM capacity:              {SRAM_BYTES} B")
    print(f"Transient reservation:      {TRANSIENT_BYTES} B")
    print(f"Bias reservation:           {BIAS_BYTES} B")
    print(f"Weight tile capacity:       {weight_capacity} B")
    print(f"Bytes/output channel:       {bytes_per_output} B")
    print(f"Compiler tile inner extent: {outputs_per_tile}")
    print(f"Logical valid tile sizes:   {tile_sizes}")
    print(
        "Logical weight bytes:       "
        f"{[size * bytes_per_output for size in tile_sizes]}"
    )

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

    first_matmul = get_global_function(
        legalized,
        "matmul",
    )

    # Isolate the generated matmul PrimFunc so the scheduling API operates
    # only on schedulable TensorIR.
    matmul_mod = tvm.IRModule(
        {"main": first_matmul}
    )

    print(
        "\n================ BEFORE SCHEDULE =======================\n"
    )
    print(matmul_mod.script())

    # TVM 0.27 scheduling lives in tvm.s_tir.
    sch = tvm.s_tir.Schedule(matmul_mod)

    block = sch.get_sblock(
        name="matmul",
        func_name="main",
    )

    loops = sch.get_loops(block)

    if len(loops) != 3:
        raise RuntimeError(
            "Expected first matmul to have exactly 3 loops "
            f"(batch, output, reduction); found {len(loops)}"
        )

    batch_loop, output_loop, reduction_loop = loops

    # Split the spatial output loop:
    #
    #     32 outputs
    #          ↓
    #     outer tile loop = ceil(32/13) = 3
    #     inner tile loop = 13
    #
    # disable_predication=False ensures TVM can guard the final partial tile.
    tile_outer, tile_inner = sch.split(
        loop=output_loop,
        factors=[None, outputs_per_tile],
        preserve_unit_iters=True,
        disable_predication=False,
    )

    print(
        "\n================ AFTER 13-WIDE OUTPUT TILING ===========\n"
    )
    print(sch.mod.script())

    print(
        "\n================ SCHEDULE TRACE ========================\n"
    )
    try:
        sch.trace.show()
    except Exception as error:
        # Trace printing is informative only; the transformed module above is
        # the authoritative result.
        print(f"trace display unavailable: {error}")

    print(
        "\n================ INTERPRETATION ========================\n"
    )
    print("Original TVM loop:")
    print("  output extent = 32")
    print("")
    print("Scheduled TVM loop:")
    print("  outer tile extent = ceil(32 / 13) = 3")
    print("  inner output extent = 13")
    print("  final outer tile contains only 6 valid outputs")
    print("")
    print("Equivalent WaveAccel logical tiles:")
    print("  tile 0: outputs  0..12 -> 13 * 256 = 3328 B")
    print("  tile 1: outputs 13..25 -> 13 * 256 = 3328 B")
    print("  tile 2: outputs 26..31 ->  6 * 256 = 1536 B")
    print("")
    print("What this proves:")
    print("  our manually validated SRAM decision can now be expressed")
    print("  as a TVM compiler schedule transformation.")
    print("")
    print("What this does NOT do yet:")
    print("  - no explicit SRAM cache allocation")
    print("  - no DMA generation")
    print("  - no WaveAccel target/codegen")
    print("  - no replacement of the known-correct C++ backend")

    print("\nTVM 13-wide output-loop tiling: PASS")


if __name__ == "__main__":
    main()
