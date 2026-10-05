#!/usr/bin/env python3
"""
Emit one integrated WaveAccel compiler artifact that includes both:
  1. the existing SRAM-aware ExecutionPlan, and
  2. the validated TVM BYOC -> ExecutionPlan partition mapping.

This is the point where BYOC selection stops being a side-report and becomes
part of the compiler-produced plan artifact consumed by downstream WaveAccel
tooling.

The existing C++ manifest exporter intentionally remains compatible because
the stable target/nodes fields are unchanged and the additional "byoc" section
is compiler metadata.

Author: Santosh Kumar
Copyright 2026 Santosh Kumar
SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnx

from tvm_byoc_execution_plan_bridge import build_bridge
from tvm_emit_waveaccel_plan import (
    build_plan,
    validate_current_model_plan,
)
from waveaccel_target_contract import WaveAccelTargetContract


INTEGRATED_COMPILER_SCHEMA_VERSION = 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Emit the integrated WaveAccel TVM/BYOC/compiler execution plan."
        )
    )
    parser.add_argument(
        "--onnx",
        default="artifacts/wave_model.onnx",
    )
    parser.add_argument(
        "--output",
        default="artifacts/waveaccel_compiled_plan.json",
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

    target = WaveAccelTargetContract(
        sram_bytes=args.sram_kib * 1024,
        transient_bytes=args.transient_bytes,
    )

    # Existing validated compiler plan generation.
    plan = build_plan(
        model,
        model_path=model_path,
        sram_bytes=target.sram_bytes,
        transient_bytes=target.transient_bytes,
    )

    if args.sram_kib == 4 and args.transient_bytes == 392:
        validate_current_model_plan(plan)

    # TVM BYOC selection must now agree with the same plan before this
    # integrated compiler artifact can be emitted.
    bridge = build_bridge(
        model=model,
        plan=plan,
        target=target,
    )

    if not bridge["full_graph_coverage"]:
        raise RuntimeError(
            "Integrated compiler plan requires full current-model "
            "WaveAccel BYOC coverage"
        )

    if (
        bridge["execution_plan_node_count"] !=
        len(plan["nodes"])
    ):
        raise RuntimeError(
            "BYOC/ExecutionPlan node accounting mismatch"
        )

    # Keep the existing execution-plan schema/fields at top level so the
    # dependency-free C++ manifest exporter remains compatible.
    plan["compiler_integration"] = {
        "schema_version": INTEGRATED_COMPILER_SCHEMA_VERSION,
        "kind": "tvm_relax_byoc",
        "backend": bridge["backend"],
        "partitioning": {
            "bind_constants": bridge["bind_constants"],
            "annotate_codegen": bridge["annotate_codegen"],
            "full_graph_coverage": bridge["full_graph_coverage"],
            "partition_count": bridge["partition_count"],
        },
    }

    plan["byoc"] = {
        "backend": bridge["backend"],
        "partition_count": bridge["partition_count"],
        "full_graph_coverage": bridge["full_graph_coverage"],
        "partitions": bridge["partitions"],
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(plan, indent=2) + "\n",
        encoding="utf-8",
    )

    print("WaveAccel integrated TVM/BYOC compiler plan")
    print("=" * 76)
    print(f"ONNX model:                     {model_path}")
    print(f"target:                         {target.name}")
    print(f"dtype:                          {target.dtype}")
    print(f"SRAM:                           {target.sram_bytes} B")
    print(
        f"initializer staging capacity:   "
        f"{target.initializer_capacity_bytes} B"
    )
    print(
        f"ExecutionPlan nodes:            "
        f"{len(plan['nodes'])}"
    )
    print(
        f"WaveAccel BYOC partitions:      "
        f"{bridge['partition_count']}"
    )
    print("BYOC full graph coverage:       YES")

    print("\nIntegrated partition -> plan mapping")
    for partition in bridge["partitions"]:
        print(
            f"  P{partition['partition_index']} "
            f"{partition['composite']}"
        )
        print(
            f"     plan nodes "
            f"{partition['execution_plan_node_indices']} "
            f"{partition['onnx_ops']}"
        )

    first_gemm = next(
        node
        for node in plan["nodes"]
        if node["op_type"] == "Gemm"
    )

    print(
        "\nfirst Gemm compiler tiles:       "
        f"{[tile['output_count'] for tile in first_gemm['tiles']]}"
    )

    print(
        f"integrated plan written:         "
        f"{output_path}"
    )

    print(
        "\nTVM BYOC selection + SRAM-aware ExecutionPlan emission: PASS"
    )


if __name__ == "__main__":
    main()
