#!/usr/bin/env python3
"""
Convert WaveAccel's JSON compiler plan into a strict key/value manifest for
the dependency-free C++ runtime parser.

The JSON file remains the canonical human-readable compiler artifact.
The manifest is a deterministic transport representation for C++ V1 so the
runtime does not need to add a JSON library dependency.

Author: Santosh Kumar
Copyright 2026 Santosh Kumar
SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MAGIC = "WAVEACCEL_EXECUTION_PLAN_V1"


def emit(lines: list[str], key: str, value: Any) -> None:
    text = str(value)
    if "\n" in text or "\r" in text or "=" in key:
        raise ValueError(f"Unsupported manifest value for {key!r}")
    lines.append(f"{key}={text}")


def emit_actions(
    lines: list[str],
    prefix: str,
    actions: list[str],
) -> None:
    emit(lines, prefix + "action_count", len(actions))
    for index, action in enumerate(actions):
        emit(lines, prefix + f"action.{index}", action)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export WaveAccel C++ execution-plan manifest."
    )
    parser.add_argument(
        "--input",
        default="artifacts/waveaccel_plan.json",
    )
    parser.add_argument(
        "--output",
        default="artifacts/waveaccel_plan.manifest",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Compiler plan JSON not found: {input_path}"
        )

    plan = json.loads(input_path.read_text(encoding="utf-8"))

    if plan.get("schema_version") != 1:
        raise ValueError("Expected schema_version=1")

    if plan.get("format") != "waveaccel.execution_plan":
        raise ValueError("Unexpected execution-plan format")

    lines = [MAGIC]
    emit(lines, "schema_version", plan["schema_version"])
    emit(lines, "format", plan["format"])
    emit(lines, "producer.tvm_version", plan["producer"]["tvm_version"])
    target = plan["target"]

    emit(lines, "target.name", target["name"])
    emit(
        lines,
        "target.contract_schema_version",
        target["contract_schema_version"],
    )
    emit(lines, "target.dtype", target["dtype"])
    emit(lines, "target.element_bytes", target["element_bytes"])

    supported_ops = target["supported_ops"]
    emit(lines, "target.supported_op_count", len(supported_ops))
    for index, op_type in enumerate(supported_ops):
        emit(lines, f"target.supported_op.{index}", op_type)

    emit(
        lines,
        "target.memory_space.model_initializers",
        target["memory_spaces"]["model_initializers"],
    )
    emit(
        lines,
        "target.memory_space.initializer_staging",
        target["memory_spaces"]["initializer_staging"],
    )
    emit(
        lines,
        "target.gemm_policy.tiling_axis",
        target["gemm_policy"]["tiling_axis"],
    )
    emit(
        lines,
        "target.gemm_policy.tile_unit",
        target["gemm_policy"]["tile_unit"],
    )
    emit(
        lines,
        "target.gemm_policy.tiled_fixed_initializer",
        target["gemm_policy"]["tiled_fixed_initializer"],
    )

    emit(lines, "target.sram_bytes", target["sram_bytes"])
    emit(
        lines,
        "target.transient_reservation_bytes",
        target["transient_reservation_bytes"],
    )
    emit(
        lines,
        "target.initializer_staging_capacity_bytes",
        target["initializer_staging_capacity_bytes"],
    )

    nodes = plan["nodes"]
    emit(lines, "node_count", len(nodes))

    for node_position, node in enumerate(nodes):
        prefix = f"node.{node_position}."

        emit(lines, prefix + "node_index", node["node_index"])
        emit(lines, prefix + "op_type", node["op_type"])
        emit(lines, prefix + "strategy", node["strategy"])
        emit(
            lines,
            prefix + "initializer_total_bytes",
            node["initializer_total_bytes"],
        )
        emit_actions(lines, prefix, node["actions"])

        if node["op_type"] != "Gemm":
            continue

        emit(lines, prefix + "tvm_primfunc", node["tvm_primfunc"])
        emit(
            lines,
            prefix + "in_features",
            node["shape"]["in_features"],
        )
        emit(
            lines,
            prefix + "out_features",
            node["shape"]["out_features"],
        )
        emit(
            lines,
            prefix + "weight_bytes",
            node["weight"]["bytes"],
        )
        emit(
            lines,
            prefix + "weight_row_bytes",
            node["weight"]["row_bytes"],
        )
        emit(
            lines,
            prefix + "bias_bytes",
            node["bias"]["bytes"],
        )

        if node["strategy"] != "TILED":
            continue

        emit(
            lines,
            prefix + "compiler_tile_width",
            node["compiler_tile_width"],
        )

        tiles = node["tiles"]
        emit(lines, prefix + "tile_count", len(tiles))

        for tile_position, tile in enumerate(tiles):
            tile_prefix = prefix + f"tile.{tile_position}."
            emit(
                lines,
                tile_prefix + "tile_index",
                tile["tile_index"],
            )
            emit(
                lines,
                tile_prefix + "output_start",
                tile["output_start"],
            )
            emit(
                lines,
                tile_prefix + "output_end",
                tile["output_end"],
            )
            emit(
                lines,
                tile_prefix + "output_count",
                tile["output_count"],
            )
            emit(
                lines,
                tile_prefix + "weight_bytes",
                tile["weight_bytes"],
            )
            emit_actions(
                lines,
                tile_prefix,
                tile["actions"],
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("WaveAccel C++ plan manifest export")
    print("=" * 68)
    print(f"input:   {input_path}")
    print(f"output:  {output_path}")
    print(f"nodes:   {len(nodes)}")

    first = nodes[0]
    if (
        first["op_type"] != "Gemm"
        or first["strategy"] != "TILED"
        or [t["output_count"] for t in first["tiles"]]
        != [13, 13, 6]
    ):
        raise RuntimeError(
            "Current 4 KiB reference plan does not match validated tiling"
        )

    print(
        "first Gemm compiler tiles: "
        f"{[t['output_count'] for t in first['tiles']]}"
    )
    print("C++ transport manifest validation: PASS")


if __name__ == "__main__":
    main()
