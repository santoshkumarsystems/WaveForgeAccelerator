#!/usr/bin/env python3
"""
Emit a compiler -> runtime execution-plan contract for WaveAccel.

This is the bridge between the TVM/compiler side and the existing validated
WaveAccel C++ runtime.

Pipeline:

    ONNX
      -> TVM Relax
      -> LegalizeOps / PrimFuncs
      -> WaveAccel target contract + scheduling policy
      -> waveaccel_plan.json
      -> dependency-free C++ execution-plan transport
      -> DeviceMemoryMap / DeviceCommandStream
      -> validated C++ MockDevice backend

The emitted plan records:
    - target SRAM constraints
    - ordered ONNX graph nodes
    - initializer byte requirements
    - GEMM staging policy
    - compiler-derived output-channel tiles
    - explicit runtime actions for tiled GEMMs

Important:
    This is a compiler/runtime contract, not hardware machine code.
    Logical model/SRAM addresses are lowered by later WaveAccel backend stages.
    Physical DMA descriptors, MMIO addresses, and vendor instructions are not
    emitted by this compiler-plan stage.

Author: Santosh Kumar
Copyright 2026 Santosh Kumar
SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import tvm
from onnx import numpy_helper
from tvm.relax.frontend.onnx import from_onnx

from waveaccel_target_contract import WaveAccelTargetContract


PLAN_SCHEMA_VERSION = 1


def tensor_nbytes(array: np.ndarray) -> int:
    return int(array.nbytes)


def initializer_map(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    return {
        tensor.name: numpy_helper.to_array(tensor)
        for tensor in model.graph.initializer
    }


def tvm_primfunc_names(model: onnx.ModelProto) -> set[str]:
    """Verify that TVM can legalize the model and return generated PrimFuncs."""
    relax_mod = from_onnx(
        model,
        shape_dict={"wave_samples": [1, 64]},
        dtype_dict={"wave_samples": "float32"},
    )

    legalized = tvm.relax.transform.LegalizeOps()(relax_mod)

    return {
        global_var.name_hint
        for global_var, function in legalized.functions.items()
        if type(function).__name__ == "PrimFunc"
    }


def gemm_tvm_name(gemm_index: int) -> str:
    return "matmul" if gemm_index == 0 else f"matmul{gemm_index}"


def make_tiles(
    *,
    out_features: int,
    row_bytes: int,
    weight_tile_capacity: int,
) -> list[dict[str, Any]]:
    tile_width = weight_tile_capacity // row_bytes

    if tile_width <= 0:
        raise RuntimeError(
            "SRAM cannot hold one complete GEMM output-channel weight row"
        )

    tile_width = min(tile_width, out_features)

    tiles: list[dict[str, Any]] = []
    start = 0
    tile_index = 0

    while start < out_features:
        count = min(tile_width, out_features - start)
        end = start + count - 1

        tiles.append(
            {
                "tile_index": tile_index,
                "output_start": start,
                "output_end": end,
                "output_count": count,
                "weight_bytes": count * row_bytes,
                "actions": [
                    "STAGE_WEIGHT_TILE_TO_SRAM",
                    "EXECUTE_GEMM_TILE",
                ],
            }
        )

        start += count
        tile_index += 1

    return tiles


def build_plan(
    model: onnx.ModelProto,
    *,
    model_path: Path,
    sram_bytes: int,
    transient_bytes: int,
) -> dict[str, Any]:
    target = WaveAccelTargetContract(
        sram_bytes=sram_bytes,
        transient_bytes=transient_bytes,
    )

    target.validate_model_ops(
        node.op_type
        for node in model.graph.node
    )

    initializers = initializer_map(model)
    primfunc_names = tvm_primfunc_names(model)

    nodes: list[dict[str, Any]] = []
    gemm_index = 0

    for node_index, node in enumerate(model.graph.node):
        init_inputs = [
            name
            for name in node.input
            if name in initializers
        ]

        init_bytes = {
            name: tensor_nbytes(initializers[name])
            for name in init_inputs
        }

        entry: dict[str, Any] = {
            "node_index": node_index,
            "op_type": node.op_type,
            "inputs": list(node.input),
            "outputs": list(node.output),
            "initializer_bytes": init_bytes,
            "initializer_total_bytes": sum(init_bytes.values()),
        }

        if node.op_type == "Gemm":
            if len(node.input) < 3:
                raise RuntimeError(
                    f"Gemm node {node_index} lacks weight/bias inputs"
                )

            weight_name = node.input[1]
            bias_name = node.input[2]

            weight = initializers[weight_name]
            bias = initializers[bias_name]

            if (
                str(weight.dtype) != target.dtype or
                str(bias.dtype) != target.dtype
            ):
                raise RuntimeError(
                    "WaveAccel target contract currently supports "
                    f"{target.dtype} Gemm only"
                )

            if weight.ndim != 2 or bias.ndim != 1:
                raise RuntimeError(
                    f"Unexpected Gemm tensor rank at node {node_index}"
                )

            # ONNX export from PyTorch stores Linear weights as
            # [out_features, in_features]. TVM canonicalizes this to
            # [in_features, out_features] before matmul.
            out_features, in_features = map(int, weight.shape)

            if int(bias.shape[0]) != out_features:
                raise RuntimeError(
                    f"Gemm bias/output mismatch at node {node_index}"
                )

            tvm_name = gemm_tvm_name(gemm_index)

            if tvm_name not in primfunc_names:
                raise RuntimeError(
                    f"Expected TVM PrimFunc '{tvm_name}' was not generated"
                )

            weight_bytes = tensor_nbytes(weight)
            bias_bytes = tensor_nbytes(bias)

            decision = target.plan_gemm(
                out_features=out_features,
                in_features=in_features,
                weight_bytes=weight_bytes,
                bias_bytes=bias_bytes,
            )

            row_bytes = decision.row_bytes
            initializer_total = decision.initializer_bytes

            entry["tvm_primfunc"] = tvm_name
            entry["shape"] = {
                "in_features": in_features,
                "out_features": out_features,
            }
            entry["weight"] = {
                "name": weight_name,
                "bytes": weight_bytes,
                "row_bytes": row_bytes,
            }
            entry["bias"] = {
                "name": bias_name,
                "bytes": bias_bytes,
            }

            if decision.strategy == "STAGED_WHOLE":
                entry["strategy"] = "STAGED_WHOLE"
                entry["actions"] = [
                    "STAGE_INITIALIZERS_TO_SRAM",
                    "EXECUTE_GEMM",
                ]
            else:
                tiles = make_tiles(
                    out_features=out_features,
                    row_bytes=row_bytes,
                    weight_tile_capacity=
                        decision.weight_tile_capacity_bytes,
                )

                if (
                    [tile["output_count"] for tile in tiles] !=
                    list(decision.tile_counts)
                ):
                    raise RuntimeError(
                        "Execution-plan tile serialization diverged from "
                        "WaveAccel target-contract decision"
                    )

                entry["strategy"] = decision.strategy
                entry["weight_tile_capacity_bytes"] = (
                    decision.weight_tile_capacity_bytes
                )
                entry["compiler_tile_width"] = decision.tile_width
                entry["actions"] = [
                    "STAGE_FIXED_INITIALIZERS_TO_SRAM",
                    "FOR_EACH_TILE",
                ]
                entry["tiles"] = tiles

            gemm_index += 1

        else:
            entry["strategy"] = target.plan_initializer_strategy(
                op_type=node.op_type,
                initializer_bytes=entry["initializer_total_bytes"],
            )

            if entry["strategy"] == "STAGED_WHOLE":
                entry["actions"] = [
                    "STAGE_INITIALIZERS_TO_SRAM",
                    f"EXECUTE_{node.op_type.upper()}",
                ]
            else:
                entry["actions"] = [
                    f"EXECUTE_{node.op_type.upper()}",
                ]

        nodes.append(entry)

    plan: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "format": "waveaccel.execution_plan",
        "producer": {
            "tool": "tvm_emit_waveaccel_plan.py",
            "tvm_version": tvm.__version__,
        },
        "model": {
            "onnx_path": str(model_path),
            "input_name": "wave_samples",
            "input_shape": [1, 64],
            "dtype": "float32",
        },
        "target": target.to_plan_dict(),
        "nodes": nodes,
    }

    return plan


def validate_current_model_plan(plan: dict[str, Any]) -> None:
    """Strong regression checks for the current WaveAccel reference model."""
    gemms = [
        node
        for node in plan["nodes"]
        if node["op_type"] == "Gemm"
    ]

    if len(gemms) != 3:
        raise RuntimeError(
            f"Expected 3 Gemm nodes, found {len(gemms)}"
        )

    first, second, third = gemms

    if first["strategy"] != "TILED":
        raise RuntimeError("Expected first Gemm to be tiled")

    tile_counts = [
        tile["output_count"]
        for tile in first["tiles"]
    ]

    if tile_counts != [13, 13, 6]:
        raise RuntimeError(
            f"Expected first Gemm tiles [13, 13, 6], got {tile_counts}"
        )

    tile_bytes = [
        tile["weight_bytes"]
        for tile in first["tiles"]
    ]

    if tile_bytes != [3328, 3328, 1536]:
        raise RuntimeError(
            "First Gemm tile bytes no longer match validated runtime"
        )

    if second["strategy"] != "STAGED_WHOLE":
        raise RuntimeError("Expected second Gemm staged whole")

    if third["strategy"] != "STAGED_WHOLE":
        raise RuntimeError("Expected third Gemm staged whole")


def print_summary(plan: dict[str, Any], output_path: Path) -> None:
    print("WaveAccel compiler -> runtime plan emission")
    print("=" * 76)
    print(
        f"TVM version:                    "
        f"{plan['producer']['tvm_version']}"
    )
    print(
        f"SRAM capacity:                  "
        f"{plan['target']['sram_bytes']} B"
    )
    print(
        f"Transient reservation:          "
        f"{plan['target']['transient_reservation_bytes']} B"
    )
    print(
        f"Initializer staging capacity:   "
        f"{plan['target']['initializer_staging_capacity_bytes']} B"
    )
    print(
        f"Target contract schema:         "
        f"{plan['target']['contract_schema_version']}"
    )
    print(
        f"Target dtype:                   "
        f"{plan['target']['dtype']}"
    )
    print(
        "Supported ops:                  "
        f"{plan['target']['supported_ops']}"
    )

    for node in plan["nodes"]:
        print(
            f"\nnode {node['node_index']:2d} "
            f"{node['op_type']:10s} "
            f"strategy={node['strategy']}"
        )

        if node["op_type"] == "Gemm":
            shape = node["shape"]
            print(
                f"  shape:                         "
                f"{shape['in_features']} -> {shape['out_features']}"
            )
            print(
                f"  TVM PrimFunc:                  "
                f"{node['tvm_primfunc']}"
            )

            if node["strategy"] == "TILED":
                print(
                    f"  compiler tile width:           "
                    f"{node['compiler_tile_width']}"
                )
                print(
                    "  logical tile counts:           "
                    f"{[t['output_count'] for t in node['tiles']]}"
                )
                print(
                    "  logical tile weight bytes:     "
                    f"{[t['weight_bytes'] for t in node['tiles']]}"
                )
            else:
                print(
                    f"  initializer bytes:             "
                    f"{node['initializer_total_bytes']}"
                )

    print(
        "\n4 KiB runtime/compiler contract validation: PASS"
    )
    print(f"Execution plan written:          {output_path}")
    print(
        "\nBackend boundary:"
        "\n  C++ transport consumes stable V1 execution-plan fields."
        "\n  Later backend stages map logical device memory and lower"
        "\n  the plan into DeviceCommandStream operations."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Emit WaveAccel compiler/runtime execution plan."
    )
    parser.add_argument(
        "--onnx",
        default="artifacts/wave_model.onnx",
    )
    parser.add_argument(
        "--output",
        default="artifacts/waveaccel_plan.json",
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
    )
    args = parser.parse_args()

    model_path = Path(args.onnx)
    output_path = Path(args.output)

    if not model_path.is_file():
        raise FileNotFoundError(
            f"ONNX model not found: {model_path}"
        )

    model = onnx.load(model_path)
    onnx.checker.check_model(model)

    plan = build_plan(
        model,
        model_path=model_path,
        sram_bytes=args.sram_kib * 1024,
        transient_bytes=args.transient_bytes,
    )

    if args.sram_kib == 4 and args.transient_bytes == 392:
        validate_current_model_plan(plan)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(plan, indent=2) + "\n",
        encoding="utf-8",
    )

    print_summary(
        plan,
        output_path,
    )


if __name__ == "__main__":
    main()
