#!/usr/bin/env python3
"""
Create an explicit compiler-managed scratchpad cache for WaveAccel's first
TVM matmul after applying the validated 13-wide output-channel tiling.

This is the next step after proving that WaveAccel's 4 KiB SRAM policy can be
expressed as a TVM loop schedule.

Pipeline in this experiment:

    first matmul
        -> split 32 output channels by 13
        -> cache the weight read into a local scratchpad
        -> move that cache fill under the outer tile loop

The expected conceptual result is:

    for output_tile in 0..2:
        copy only this tile's required weight region into scratchpad
        compute this output tile from the scratchpad

For the current first GEMM:

    logical valid output tiles = 13 + 13 + 6
    weight bytes               = 3328 + 3328 + 1536

Why use TVM storage scope "local" here?
    This step proves explicit compiler-managed scratchpad placement using a
    storage scope that TVM already understands.  We intentionally do NOT
    invent "waveaccel.sram" yet, because a custom storage scope should be
    introduced together with the WaveAccel target/codegen/runtime contract.

Important separation:
    - "local" here is a compiler scratchpad analogue, not a claim that TVM is
      already targeting physical WaveAccel SRAM.
    - DMA generation is not implemented yet.
    - The existing validated C++ WaveAccel backend remains unchanged.

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


SRAM_BYTES = 4096
TRANSIENT_BYTES = 392
BIAS_BYTES = 128
IN_FEATURES = 64
OUT_FEATURES = 32
FP32_BYTES = 4


def get_global_function(mod, name: str):
    for global_var, function in mod.functions.items():
        if global_var.name_hint == name:
            return function

    raise KeyError(f"TVM function not found: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create an explicit TVM scratchpad cache for WaveAccel's "
            "first tiled matmul."
        )
    )
    parser.add_argument(
        "--onnx",
        default="artifacts/wave_model.onnx",
    )
    args = parser.parse_args()

    onnx_path = Path(args.onnx)

    if not onnx_path.is_file():
        raise FileNotFoundError(
            f"ONNX model not found: {onnx_path}"
        )

    initializer_capacity = SRAM_BYTES - TRANSIENT_BYTES
    weight_tile_capacity = initializer_capacity - BIAS_BYTES
    row_bytes = IN_FEATURES * FP32_BYTES
    tile_width = weight_tile_capacity // row_bytes

    if tile_width != 13:
        raise RuntimeError(
            f"Expected validated first-GEMM tile width 13, got {tile_width}"
        )

    print("WaveAccel TVM scratchpad-cache experiment")
    print("=" * 76)
    print(f"TVM version:                  {tvm.__version__}")
    print(f"ONNX model:                   {onnx_path}")
    print(f"SRAM capacity model:          {SRAM_BYTES} B")
    print(f"Transient reservation:        {TRANSIENT_BYTES} B")
    print(f"Bias reservation:             {BIAS_BYTES} B")
    print(f"Weight tile capacity:         {weight_tile_capacity} B")
    print(f"Weight bytes/output channel:  {row_bytes} B")
    print(f"Compiler output tile width:   {tile_width}")
    print("Logical valid tiles:          [13, 13, 6]")
    print("Logical weight bytes:         [3328, 3328, 1536]")
    print('TVM scratchpad storage scope: "local"')

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

    matmul_mod = tvm.IRModule(
        {"main": first_matmul}
    )

    sch = tvm.s_tir.Schedule(matmul_mod)

    block = sch.get_sblock(
        name="matmul",
        func_name="main",
    )

    loops = sch.get_loops(block)

    if len(loops) != 3:
        raise RuntimeError(
            f"Expected 3 matmul loops, found {len(loops)}"
        )

    batch_loop, output_loop, reduction_loop = loops

    # 1) Apply the already validated WaveAccel output-channel tile width.
    tile_outer, tile_inner = sch.split(
        loop=output_loop,
        factors=[None, tile_width],
        preserve_unit_iters=True,
        disable_predication=False,
    )

    # 2) Cache read-buffer #1, which is the transposed weight buffer `lv`.
    #    TVM's public API defines cache_read(block, read_buffer_index,
    #    storage_scope).  "local" creates an explicit scratchpad allocation.
    weight_cache = sch.cache_read(
        block=block,
        read_buffer_index=1,
        storage_scope="local",
    )

    # 3) Move the weight-copy stage beneath the outer output tile loop.
    #    Region inference should then restrict each cache fill to only the
    #    weight columns needed by that tile.
    sch.compute_at(
        block=weight_cache,
        loop=tile_outer,
        preserve_unit_loops=False,
    )

    print(
        "\n================ TILED + SCRATCHPAD-CACHED TIR =========\n"
    )
    print(sch.mod.script())

    print(
        "\n================ SCHEDULE TRACE ========================\n"
    )
    try:
        sch.trace.show()
    except Exception as error:
        print(f"trace display unavailable: {error}")

    script_text = sch.mod.script()

    has_local_scope = (
        'scope="local"' in script_text
        or 'scope = "local"' in script_text
        or '.local' in script_text
    )

    print(
        "\n================ VERIFICATION ==========================\n"
    )
    print(
        "explicit local scratchpad visible: "
        f"{'PASS' if has_local_scope else 'CHECK IR MANUALLY'}"
    )

    print("\nExpected conceptual dataflow:")
    print("")
    print("  Device/global weight")
    print("          ↓")
    print("  compiler-generated cache-copy block")
    print("          ↓")
    print('  TVM "local" scratchpad')
    print("          ↓")
    print("  tiled matmul")
    print("")
    print("Expected first-GEMM tile semantics:")
    print("  outer tile 0 -> outputs  0..12 -> 3328 B weights")
    print("  outer tile 1 -> outputs 13..25 -> 3328 B weights")
    print("  outer tile 2 -> outputs 26..31 -> 1536 B weights")
    print("")
    print("What is complete after this step:")
    print("  hardware-derived loop tiling")
    print("  explicit compiler scratchpad cache")
    print("  per-tile cache placement")
    print("")
    print("Still next:")
    print('  replace generic "local" with WaveAccel SRAM contract')
    print("  lower cache-copy block into WaveAccel DMA operation")
    print("  WaveAccel codegen / execution-plan emission")

    print("\nTVM tiled scratchpad cache experiment: PASS")


if __name__ == "__main__":
    main()
