#!/usr/bin/env python3
"""
Apply a reusable WaveAccel SRAM-aware TVM scheduling policy to every GEMM
lowered from the current ONNX model.

Engineering goal:
    move from a one-off "split first matmul by 13" experiment to a generic
    target policy derived from hardware constraints + real ONNX tensor shapes.

Current WaveAccel target model:
    SRAM capacity               = configurable (default 4 KiB)
    transient reservation       = 392 B for batch-size-1 current model
    dtype                       = FP32
    GEMM tiling policy          = output-channel / row tiling
    supported scheduled pattern = ONNX Gemm -> TVM matmul

For each ONNX Gemm:
    1. Read real weight/bias shapes and byte sizes.
    2. Compute initializer staging requirement.
    3. If it fits, leave the TVM matmul loop unsplit.
    4. If it does not fit, reserve the bias and derive the maximum number
       of whole output channels that fit in SRAM.
    5. Split TVM's output loop by that compiler-derived tile width.

For the current model at 4 KiB SRAM:
    Gemm 0: 8192 B weight + 128 B bias -> tiled 13 + 13 + 6
    Gemm 1: 2048 B weight +  64 B bias -> staged whole
    Gemm 2:  128 B weight +   8 B bias -> staged whole

This script intentionally owns only the TVM scheduling decision. Downstream
WaveAccel stages serialize the compiler plan, map logical model memory, lower
DeviceCommand streams, and execute the validated mock backend. This script does
not emit physical DMA descriptors or a vendor instruction stream.

Author: Santosh Kumar
Copyright 2026 Santosh Kumar
SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
import tvm
from onnx import numpy_helper
from tvm.relax.frontend.onnx import from_onnx

from waveaccel_target_contract import (
    FP32_BYTES,
    WaveAccelTargetContract,
)


@dataclass(frozen=True)
class GemmInfo:
    index: int
    tvm_name: str
    weight_name: str
    bias_name: str
    out_features: int
    in_features: int
    weight_bytes: int
    bias_bytes: int

    @property
    def initializer_bytes(self) -> int:
        return self.weight_bytes + self.bias_bytes

    @property
    def row_bytes(self) -> int:
        return self.in_features * FP32_BYTES


def get_global_function(mod, name: str):
    for global_var, function in mod.functions.items():
        if global_var.name_hint == name:
            return function
    raise KeyError(f"TVM function not found: {name}")


def extract_gemms(model: onnx.ModelProto) -> list[GemmInfo]:
    initializers = {
        tensor.name: numpy_helper.to_array(tensor)
        for tensor in model.graph.initializer
    }

    gemms: list[GemmInfo] = []

    for node in model.graph.node:
        if node.op_type != "Gemm":
            continue

        if len(node.input) < 3:
            raise RuntimeError(
                "WaveAccel V1 expects Gemm with input, weight, and bias"
            )

        weight_name = node.input[1]
        bias_name = node.input[2]

        weight = initializers[weight_name]
        bias = initializers[bias_name]

        if weight.dtype != np.float32 or bias.dtype != np.float32:
            raise RuntimeError(
                "WaveAccel V1 scheduling policy currently expects FP32"
            )

        if weight.ndim != 2 or bias.ndim != 1:
            raise RuntimeError(
                "Unexpected Gemm weight/bias rank"
            )

        out_features, in_features = map(int, weight.shape)

        if int(bias.shape[0]) != out_features:
            raise RuntimeError(
                "Gemm bias size does not match output features"
            )

        index = len(gemms)
        tvm_name = "matmul" if index == 0 else f"matmul{index}"

        gemms.append(
            GemmInfo(
                index=index,
                tvm_name=tvm_name,
                weight_name=weight_name,
                bias_name=bias_name,
                out_features=out_features,
                in_features=in_features,
                weight_bytes=int(weight.nbytes),
                bias_bytes=int(bias.nbytes),
            )
        )

    if not gemms:
        raise RuntimeError("No ONNX Gemm nodes found")

    return gemms


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply WaveAccel SRAM-aware policy to TVM matmuls."
    )
    parser.add_argument(
        "--onnx",
        default="artifacts/wave_model.onnx",
    )
    parser.add_argument(
        "--sram-kib",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--transient-bytes",
        type=int,
        default=392,
        help="Current batch-size-1 transient reservation.",
    )
    parser.add_argument(
        "--show-ir",
        action="store_true",
        help="Print scheduled TIR for every GEMM.",
    )
    args = parser.parse_args()

    if args.sram_kib <= 0:
        raise ValueError("SRAM KiB must be positive")

    target = WaveAccelTargetContract(
        sram_bytes=args.sram_kib * 1024,
        transient_bytes=args.transient_bytes,
    )

    onnx_path = Path(args.onnx)

    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)

    target.validate_model_ops(
        node.op_type
        for node in model.graph.node
    )

    gemms = extract_gemms(model)

    relax_mod = from_onnx(
        model,
        shape_dict={"wave_samples": [1, 64]},
        dtype_dict={"wave_samples": "float32"},
    )

    legalized = tvm.relax.transform.LegalizeOps()(
        relax_mod
    )

    print("WaveAccel TVM target scheduling policy")
    print("=" * 76)
    print(f"TVM version:                  {tvm.__version__}")
    print(f"ONNX model:                   {onnx_path}")
    print(f"SRAM capacity:                {target.sram_bytes} B")
    print(f"Transient reservation:        {target.transient_bytes} B")
    print(
        "Initializer staging capacity: "
        f"{target.initializer_capacity_bytes} B"
    )

    tiled_count = 0
    whole_count = 0

    for gemm in gemms:
        print(
            f"\n---------------- GEMM {gemm.index} / {gemm.tvm_name} "
            f"----------------"
        )
        print(
            f"shape:                         "
            f"{gemm.in_features} -> {gemm.out_features}"
        )
        print(f"weight bytes:                  {gemm.weight_bytes}")
        print(f"bias bytes:                    {gemm.bias_bytes}")
        print(f"initializer bytes:             {gemm.initializer_bytes}")

        primfunc = get_global_function(
            legalized,
            gemm.tvm_name,
        )

        matmul_mod = tvm.IRModule(
            {"main": primfunc}
        )

        decision = target.plan_gemm(
            out_features=gemm.out_features,
            in_features=gemm.in_features,
            weight_bytes=gemm.weight_bytes,
            bias_bytes=gemm.bias_bytes,
        )

        if decision.strategy == "STAGED_WHOLE":
            whole_count += 1
            print("policy:                        STAGED_WHOLE")
            print("TVM schedule action:           none")
            if args.show_ir:
                print("\nTIR:\n")
                print(matmul_mod.script())
            continue

        weight_tile_capacity = (
            decision.weight_tile_capacity_bytes
        )
        tile_width = decision.tile_width
        tiles = list(decision.tile_counts)

        tiled_count += 1

        print("policy:                        TILED")
        print(f"weight row bytes:              {gemm.row_bytes}")
        print(
            f"weight tile capacity:          "
            f"{weight_tile_capacity}"
        )
        print(f"compiler tile width:           {tile_width}")
        print(f"logical output tiles:          {tiles}")
        print(
            f"logical weight bytes/tile:     "
            f"{[size * gemm.row_bytes for size in tiles]}"
        )

        sch = tvm.s_tir.Schedule(matmul_mod)

        block = sch.get_sblock(
            name="matmul",
            func_name="main",
        )

        loops = sch.get_loops(block)

        if len(loops) != 3:
            raise RuntimeError(
                f"Gemm {gemm.index}: expected 3 loops, "
                f"found {len(loops)}"
            )

        _, output_loop, _ = loops

        sch.split(
            loop=output_loop,
            factors=[None, tile_width],
            preserve_unit_iters=True,
            disable_predication=False,
        )

        print(
            "TVM schedule action:           "
            f"split output loop by {tile_width}"
        )

        if args.show_ir:
            print("\nScheduled TIR:\n")
            print(sch.mod.script())

    print(
        "\n================ POLICY SUMMARY ========================"
    )
    print(f"GEMMs discovered:              {len(gemms)}")
    print(f"GEMMs staged whole:            {whole_count}")
    print(f"GEMMs compiler-tiled:          {tiled_count}")

    if args.sram_kib == 4:
        if len(gemms) != 3 or tiled_count != 1 or whole_count != 2:
            raise RuntimeError(
                "4 KiB policy no longer matches the validated "
                "WaveAccel runtime behavior"
            )

        first = gemms[0]
        first_decision = target.plan_gemm(
            out_features=first.out_features,
            in_features=first.in_features,
            weight_bytes=first.weight_bytes,
            bias_bytes=first.bias_bytes,
        )

        if first_decision.tile_width != 13:
            raise RuntimeError(
                "Expected first GEMM tile width 13, got "
                f"{first_decision.tile_width}"
            )

        print(
            "4 KiB runtime/compiler policy parity: PASS"
        )

    print("\nMeaning:")
    print("  Hardware constraints -> compiler scheduling decision")
    print("  instead of a hard-coded '13' in the TVM transformation.")
    print("")
    print("Target contract:")
    print("  - one source of truth for SRAM/dtype/op capabilities")
    print("  - one source of truth for Gemm staging/tiling decisions")
    print("Downstream, intentionally separate:")
    print("  - execution-plan transport")
    print("  - logical device-memory mapping")
    print("  - DeviceCommand lowering")
    print("  - validated C++ MockDevice / NumericalExecutor backend")

    print("\nWaveAccel target-aware TVM scheduling policy: PASS")


if __name__ == "__main__":
    main()
