#!/usr/bin/env python3
"""
Bridge TVM WaveAccel BYOC partitions to the existing WaveAccel ExecutionPlan.

This milestone does not change C++ runtime behavior. It proves that the graph
regions selected by TVM's BYOC partitioner correspond exactly, in topological
order, to the ONNX nodes already consumed by the WaveAccel execution-plan
backend.

Validated chain:
    ONNX
      -> Relax
      -> ConvertToDataflow
      -> FuseOpsByPattern(Codegen="waveaccel")
      -> ordered BYOC partitions
      -> ONNX node ranges
      -> existing waveaccel_plan.json nodes

The emitted bridge is compiler metadata, not machine code and not a physical
accelerator ABI.

Author: Santosh Kumar
Copyright 2026 Santosh Kumar
SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import onnx
from tvm.relax.frontend.onnx import from_onnx
from tvm.relax.transform import ConvertToDataflow, FuseOpsByPattern

from tvm_waveaccel_partition import waveaccel_patterns
from waveaccel_target_contract import WaveAccelTargetContract


BACKEND = "waveaccel"


@dataclass(frozen=True)
class PartitionSpec:
    composite: str
    onnx_ops: tuple[str, ...]


PARTITION_SPECS = {
    f"{BACKEND}.matmul_bias_relu": PartitionSpec(
        composite=f"{BACKEND}.matmul_bias_relu",
        onnx_ops=("Gemm", "Relu"),
    ),
    f"{BACKEND}.matmul_bias": PartitionSpec(
        composite=f"{BACKEND}.matmul_bias",
        onnx_ops=("Gemm",),
    ),
    f"{BACKEND}.multiply": PartitionSpec(
        composite=f"{BACKEND}.multiply",
        onnx_ops=("Mul",),
    ),
    f"{BACKEND}.add": PartitionSpec(
        composite=f"{BACKEND}.add",
        onnx_ops=("Add",),
    ),
}


def partition_model(model, target: WaveAccelTargetContract):
    relax_mod = from_onnx(
        model,
        shape_dict={"wave_samples": [1, 64]},
        dtype_dict={"wave_samples": target.dtype},
    )

    dataflow_mod = ConvertToDataflow()(relax_mod)

    return FuseOpsByPattern(
        waveaccel_patterns(),
        bind_constants=False,
        annotate_codegen=True,
    )(dataflow_mod)


def _function_blocks(ir_text: str) -> dict[str, str]:
    """
    Split top-level TVMScript @R.function definitions into textual blocks.

    We use TVMScript only to inspect BYOC metadata and main-call ordering.
    Compiler semantics still come from TVM's transformed IRModule.
    """
    matches = list(
        re.finditer(
            r"(?m)^    @R\.function\n    def ([A-Za-z0-9_]+)\(",
            ir_text,
        )
    )

    blocks: dict[str, str] = {}

    for index, match in enumerate(matches):
        start = match.start()
        end = (
            matches[index + 1].start()
            if index + 1 < len(matches)
            else len(ir_text)
        )
        blocks[match.group(1)] = ir_text[start:end]

    return blocks


def _composite_by_wrapper(ir_text: str) -> dict[str, str]:
    wrappers: dict[str, str] = {}

    for name, block in _function_blocks(ir_text).items():
        if name == "main":
            continue

        if not re.search(
            r'["\']Codegen["\']\s*:\s*["\']waveaccel["\']',
            block,
        ):
            continue

        composites = re.findall(
            r'["\']Composite["\']\s*:\s*["\'](waveaccel\.[^"\']+)["\']',
            block,
        )

        if len(composites) != 1:
            raise RuntimeError(
                f"Expected one WaveAccel Composite in wrapper {name}, "
                f"found {composites}"
            )

        wrappers[name] = composites[0]

    return wrappers


def _main_wrapper_call_order(ir_text: str) -> list[str]:
    blocks = _function_blocks(ir_text)

    if "main" not in blocks:
        raise RuntimeError("Partitioned Relax IR has no main function")

    main_block = blocks["main"]

    return re.findall(
        r"cls\.([A-Za-z0-9_]+_waveaccel)\(",
        main_block,
    )


def _validate_plan_header(
    plan: dict[str, Any],
    target: WaveAccelTargetContract,
) -> None:
    if plan.get("format") != "waveaccel.execution_plan":
        raise RuntimeError(
            "Unexpected execution-plan format"
        )

    target_json = plan.get("target", {})

    expected_target = target.to_plan_dict()

    if target_json != expected_target:
        raise RuntimeError(
            "Execution-plan target metadata does not match "
            "WaveAccelTargetContract"
        )


def build_bridge(
    *,
    model,
    plan: dict[str, Any],
    target: WaveAccelTargetContract,
) -> dict[str, Any]:
    target.validate_model_ops(
        node.op_type
        for node in model.graph.node
    )

    _validate_plan_header(plan, target)

    partitioned = partition_model(model, target)
    ir_text = partitioned.script()

    composite_by_wrapper = _composite_by_wrapper(ir_text)
    call_order = _main_wrapper_call_order(ir_text)

    if not call_order:
        raise RuntimeError(
            "No WaveAccel BYOC wrapper calls found in Relax main"
        )

    if len(call_order) != len(composite_by_wrapper):
        raise RuntimeError(
            "WaveAccel wrapper definition/call counts differ: "
            f"definitions={len(composite_by_wrapper)} "
            f"main_calls={len(call_order)}"
        )

    if len(set(call_order)) != len(call_order):
        raise RuntimeError(
            "Current WaveAccel bridge expects one call per BYOC wrapper"
        )

    onnx_ops = [
        node.op_type
        for node in model.graph.node
    ]

    plan_nodes = plan.get("nodes", [])

    if len(plan_nodes) != len(onnx_ops):
        raise RuntimeError(
            "ExecutionPlan node count does not match ONNX graph"
        )

    for index, (onnx_op, plan_node) in enumerate(
        zip(onnx_ops, plan_nodes)
    ):
        if (
            int(plan_node["node_index"]) != index or
            plan_node["op_type"] != onnx_op
        ):
            raise RuntimeError(
                f"ExecutionPlan node {index} does not match ONNX graph"
            )

    cursor = 0
    partitions: list[dict[str, Any]] = []

    for partition_index, wrapper in enumerate(call_order):
        if wrapper not in composite_by_wrapper:
            raise RuntimeError(
                f"Relax main calls unknown WaveAccel wrapper: {wrapper}"
            )

        composite = composite_by_wrapper[wrapper]

        if composite not in PARTITION_SPECS:
            raise RuntimeError(
                f"Unsupported WaveAccel composite in bridge: {composite}"
            )

        spec = PARTITION_SPECS[composite]
        count = len(spec.onnx_ops)

        actual_ops = tuple(
            onnx_ops[cursor:cursor + count]
        )

        if actual_ops != spec.onnx_ops:
            raise RuntimeError(
                f"Partition {partition_index} ({composite}) expected ONNX "
                f"ops {list(spec.onnx_ops)} at node {cursor}, "
                f"found {list(actual_ops)}"
            )

        node_indices = list(
            range(cursor, cursor + count)
        )

        initializer_bytes = sum(
            int(plan_nodes[index]["initializer_total_bytes"])
            for index in node_indices
        )

        tvm_primfuncs = [
            plan_nodes[index].get("tvm_primfunc", "")
            for index in node_indices
            if plan_nodes[index].get("tvm_primfunc", "")
        ]

        partitions.append(
            {
                "partition_index": partition_index,
                "wrapper_function": wrapper,
                "composite": composite,
                "onnx_node_indices": node_indices,
                "onnx_ops": list(spec.onnx_ops),
                "execution_plan_node_indices": node_indices,
                "initializer_total_bytes": initializer_bytes,
                "tvm_primfuncs": tvm_primfuncs,
            }
        )

        cursor += count

    if cursor != len(onnx_ops):
        raise RuntimeError(
            "WaveAccel BYOC partitions do not cover the full current ONNX graph: "
            f"covered {cursor} of {len(onnx_ops)} nodes"
        )

    flattened = [
        index
        for partition in partitions
        for index in partition["execution_plan_node_indices"]
    ]

    expected_indices = list(range(len(plan_nodes)))

    if flattened != expected_indices:
        raise RuntimeError(
            "BYOC -> ExecutionPlan mapping is not exact/topological: "
            f"{flattened}"
        )

    return {
        "schema_version": 1,
        "format": "waveaccel.byoc_execution_plan_bridge",
        "backend": BACKEND,
        "target_name": target.name,
        "dtype": target.dtype,
        "bind_constants": False,
        "annotate_codegen": True,
        "partition_count": len(partitions),
        "onnx_node_count": len(onnx_ops),
        "execution_plan_node_count": len(plan_nodes),
        "full_graph_coverage": True,
        "partitions": partitions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate TVM WaveAccel BYOC partitions against "
            "waveaccel_plan.json."
        )
    )
    parser.add_argument(
        "--onnx",
        default="artifacts/wave_model.onnx",
    )
    parser.add_argument(
        "--plan",
        default="artifacts/waveaccel_plan.json",
    )
    parser.add_argument(
        "--output",
        default="artifacts/waveaccel_byoc_bridge.json",
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
    plan_path = Path(args.plan)
    output_path = Path(args.output)

    if not model_path.is_file():
        raise FileNotFoundError(
            f"ONNX model not found: {model_path}"
        )

    if not plan_path.is_file():
        raise FileNotFoundError(
            f"Execution plan not found: {plan_path}"
        )

    target = WaveAccelTargetContract(
        sram_bytes=args.sram_kib * 1024,
        transient_bytes=args.transient_bytes,
    )

    model = onnx.load(model_path)
    onnx.checker.check_model(model)

    plan = json.loads(
        plan_path.read_text(encoding="utf-8")
    )

    bridge = build_bridge(
        model=model,
        plan=plan,
        target=target,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path.write_text(
        json.dumps(bridge, indent=2) + "\n",
        encoding="utf-8",
    )

    print("WaveAccel TVM BYOC -> ExecutionPlan bridge")
    print("=" * 76)
    print(f"ONNX model:                     {model_path}")
    print(f"Execution plan:                 {plan_path}")
    print(f"backend:                        {bridge['backend']}")
    print(f"dtype:                          {bridge['dtype']}")
    print(f"BYOC partitions:                {bridge['partition_count']}")
    print(f"ONNX nodes:                     {bridge['onnx_node_count']}")
    print(
        f"ExecutionPlan nodes:            "
        f"{bridge['execution_plan_node_count']}"
    )
    print("full graph coverage:            YES")

    print("\nPartition mapping")

    for partition in bridge["partitions"]:
        print(
            f"  P{partition['partition_index']}: "
            f"{partition['composite']}"
        )
        print(
            f"      ONNX/plan nodes: "
            f"{partition['onnx_node_indices']} "
            f"{partition['onnx_ops']}"
        )
        print(
            f"      initializer bytes: "
            f"{partition['initializer_total_bytes']}"
        )

        if partition["tvm_primfuncs"]:
            print(
                f"      TVM PrimFuncs: "
                f"{partition['tvm_primfuncs']}"
            )

    print(f"\nbridge written:                 {output_path}")
    print(
        "\nTVM BYOC partition -> existing WaveAccel ExecutionPlan: PASS"
    )


if __name__ == "__main__":
    main()
