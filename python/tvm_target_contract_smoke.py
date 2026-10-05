#!/usr/bin/env python3
"""
Validate that WaveAccel TVM scheduling and execution-plan emission share the
same compiler-side target/backend contract.

This is a compiler contract smoke test. It does not claim that WaveAccel is a
registered TVM TargetKind or a physical accelerator backend.

Author: Santosh Kumar
Copyright 2026 Santosh Kumar
SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import numpy_helper

from tvm_emit_waveaccel_plan import build_plan
from waveaccel_target_contract import (
    TARGET_CONTRACT_SCHEMA_VERSION,
    WaveAccelTargetContract,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate WaveAccel compiler target contract."
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
    )
    args = parser.parse_args()

    model_path = Path(args.onnx)

    if not model_path.is_file():
        raise FileNotFoundError(
            f"ONNX model not found: {model_path}"
        )

    model = onnx.load(model_path)
    onnx.checker.check_model(model)

    target = WaveAccelTargetContract(
        sram_bytes=args.sram_kib * 1024,
        transient_bytes=args.transient_bytes,
    )

    target.validate_model_ops(
        node.op_type
        for node in model.graph.node
    )

    initializers = {
        tensor.name: numpy_helper.to_array(tensor)
        for tensor in model.graph.initializer
    }

    gemm_decisions = []

    for node in model.graph.node:
        if node.op_type != "Gemm":
            continue

        weight = initializers[node.input[1]]
        bias = initializers[node.input[2]]

        out_features, in_features = map(int, weight.shape)

        decision = target.plan_gemm(
            out_features=out_features,
            in_features=in_features,
            weight_bytes=int(weight.nbytes),
            bias_bytes=int(bias.nbytes),
        )

        gemm_decisions.append(decision)

    plan = build_plan(
        model,
        model_path=model_path,
        sram_bytes=target.sram_bytes,
        transient_bytes=target.transient_bytes,
    )

    if plan["target"] != target.to_plan_dict():
        raise RuntimeError(
            "Execution-plan target metadata diverged from target contract"
        )

    gemm_nodes = [
        node
        for node in plan["nodes"]
        if node["op_type"] == "Gemm"
    ]

    if len(gemm_nodes) != len(gemm_decisions):
        raise RuntimeError(
            "Gemm count differs between contract and execution plan"
        )

    for node, decision in zip(gemm_nodes, gemm_decisions):
        if node["strategy"] != decision.strategy:
            raise RuntimeError(
                f"Node {node['node_index']} strategy diverged from "
                "target contract"
            )

        if decision.strategy == "TILED":
            plan_counts = [
                tile["output_count"]
                for tile in node["tiles"]
            ]

            if plan_counts != list(decision.tile_counts):
                raise RuntimeError(
                    f"Node {node['node_index']} tile counts diverged "
                    "from target contract"
                )

            if node["compiler_tile_width"] != decision.tile_width:
                raise RuntimeError(
                    f"Node {node['node_index']} tile width diverged "
                    "from target contract"
                )

    print("WaveAccel compiler-side target/backend contract")
    print("=" * 72)
    print(
        f"contract schema:               "
        f"{TARGET_CONTRACT_SCHEMA_VERSION}"
    )
    print(f"target name:                   {target.name}")
    print(f"dtype:                         {target.dtype}")
    print(f"element bytes:                 {target.element_bytes}")
    print(f"SRAM bytes:                    {target.sram_bytes}")
    print(
        f"transient reservation:         "
        f"{target.transient_bytes}"
    )
    print(
        f"initializer staging capacity:  "
        f"{target.initializer_capacity_bytes}"
    )
    print(
        f"supported ops:                 "
        f"{list(target.supported_ops)}"
    )
    print(
        f"memory spaces:                 "
        f"{target.device_memory_space} -> {target.sram_memory_space}"
    )
    print(
        f"Gemm tiling axis:              "
        f"{target.gemm_tiling_axis}"
    )
    print(
        f"Gemm tile unit:                "
        f"{target.gemm_tile_unit}"
    )

    for index, decision in enumerate(gemm_decisions):
        print(
            f"Gemm {index}: "
            f"strategy={decision.strategy} "
            f"row_bytes={decision.row_bytes} "
            f"tile_width={decision.tile_width} "
            f"tiles={list(decision.tile_counts)}"
        )

    if args.sram_kib == 4 and args.transient_bytes == 392:
        expected = [
            ("TILED", [13, 13, 6]),
            ("STAGED_WHOLE", [16]),
            ("STAGED_WHOLE", [2]),
        ]

        actual = [
            (
                decision.strategy,
                list(decision.tile_counts),
            )
            for decision in gemm_decisions
        ]

        if actual != expected:
            raise RuntimeError(
                f"4 KiB reference contract changed: {actual}"
            )

    print(
        "\nTVM policy / execution-plan target contract parity: PASS"
    )


if __name__ == "__main__":
    main()
