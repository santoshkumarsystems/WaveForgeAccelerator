#!/usr/bin/env python3
"""
WaveAccel compiler-side target/backend contract.

This module is the single source of truth for the generic WaveAccel V1 target
constraints consumed by TVM scheduling policy and execution-plan emission.

It does NOT register a proprietary TVM TargetKind and it does NOT describe
physical accelerator addresses or a vendor ISA. It defines the logical
compiler/runtime contract used by the WaveAccel simulator backend.

Author: Santosh Kumar
Copyright 2026 Santosh Kumar
SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

from dataclasses import dataclass


TARGET_CONTRACT_SCHEMA_VERSION = 1
FP32_BYTES = 4


@dataclass(frozen=True)
class GemmPolicyDecision:
    """Target-derived staging/tiling decision for one FP32 Gemm."""

    strategy: str
    row_bytes: int
    initializer_bytes: int
    weight_tile_capacity_bytes: int
    tile_width: int
    tile_counts: tuple[int, ...]
    tile_weight_bytes: tuple[int, ...]


@dataclass(frozen=True)
class WaveAccelTargetContract:
    """
    Generic WaveAccel V1 compiler target contract.

    Memory-space names are logical compiler/runtime concepts:
      - device_memory: model initializer image after host->device residency.
      - sram: accelerator-local initializer staging/scratchpad region.

    They are not physical addresses, IOVAs, BAR mappings, or vendor-specific
    memory-space encodings.
    """

    sram_bytes: int
    transient_bytes: int

    name: str = "waveaccel-sim"
    dtype: str = "float32"
    element_bytes: int = FP32_BYTES

    supported_ops: tuple[str, ...] = (
        "Gemm",
        "Relu",
        "Mul",
        "Add",
    )

    device_memory_space: str = "device_memory"
    sram_memory_space: str = "sram"
    gemm_tiling_axis: str = "output_channel"
    gemm_tile_unit: str = "complete_output_weight_row"

    def __post_init__(self) -> None:
        if self.sram_bytes <= 0:
            raise ValueError("SRAM bytes must be positive")

        if self.transient_bytes < 0:
            raise ValueError("Transient bytes cannot be negative")

        if self.transient_bytes >= self.sram_bytes:
            raise ValueError(
                "Transient reservation must leave positive initializer capacity"
            )

        if self.dtype != "float32" or self.element_bytes != FP32_BYTES:
            raise ValueError(
                "WaveAccel V1 target contract currently supports FP32 only"
            )

    @property
    def initializer_capacity_bytes(self) -> int:
        return self.sram_bytes - self.transient_bytes

    def validate_model_ops(self, op_types) -> None:
        unsupported = sorted(
            {
                str(op_type)
                for op_type in op_types
                if str(op_type) not in self.supported_ops
            }
        )

        if unsupported:
            raise RuntimeError(
                "WaveAccel target contract does not support ONNX op(s): "
                + ", ".join(unsupported)
            )

    def plan_initializer_strategy(
        self,
        *,
        op_type: str,
        initializer_bytes: int,
    ) -> str:
        if op_type not in self.supported_ops:
            raise RuntimeError(
                f"WaveAccel target contract does not support op: {op_type}"
            )

        if initializer_bytes < 0:
            raise ValueError("Initializer bytes cannot be negative")

        if initializer_bytes == 0:
            return "EXECUTE"

        if initializer_bytes > self.initializer_capacity_bytes:
            raise RuntimeError(
                f"{op_type} initializer state ({initializer_bytes} B) "
                "does not fit WaveAccel initializer staging capacity "
                f"({self.initializer_capacity_bytes} B)"
            )

        return "STAGED_WHOLE"

    def plan_gemm(
        self,
        *,
        out_features: int,
        in_features: int,
        weight_bytes: int,
        bias_bytes: int,
    ) -> GemmPolicyDecision:
        if out_features <= 0 or in_features <= 0:
            raise ValueError("Gemm dimensions must be positive")

        if weight_bytes <= 0 or bias_bytes <= 0:
            raise ValueError("WaveAccel V1 Gemm requires weight and bias")

        row_bytes = in_features * self.element_bytes

        expected_weight_bytes = (
            out_features * in_features * self.element_bytes
        )

        if weight_bytes != expected_weight_bytes:
            raise RuntimeError(
                "Gemm weight bytes do not match FP32 "
                "[out_features, in_features] geometry"
            )

        expected_bias_bytes = out_features * self.element_bytes

        if bias_bytes != expected_bias_bytes:
            raise RuntimeError(
                "Gemm bias bytes do not match FP32 output geometry"
            )

        initializer_bytes = weight_bytes + bias_bytes

        if initializer_bytes <= self.initializer_capacity_bytes:
            return GemmPolicyDecision(
                strategy="STAGED_WHOLE",
                row_bytes=row_bytes,
                initializer_bytes=initializer_bytes,
                weight_tile_capacity_bytes=0,
                tile_width=out_features,
                tile_counts=(out_features,),
                tile_weight_bytes=(weight_bytes,),
            )

        weight_tile_capacity = (
            self.initializer_capacity_bytes - bias_bytes
        )

        if weight_tile_capacity <= 0:
            raise RuntimeError(
                "Gemm bias leaves no SRAM capacity for one weight tile"
            )

        tile_width = weight_tile_capacity // row_bytes

        if tile_width <= 0:
            raise RuntimeError(
                "WaveAccel SRAM cannot hold one complete Gemm output row"
            )

        tile_width = min(tile_width, out_features)

        counts: list[int] = []
        remaining = out_features

        while remaining > 0:
            count = min(tile_width, remaining)
            counts.append(count)
            remaining -= count

        return GemmPolicyDecision(
            strategy="TILED",
            row_bytes=row_bytes,
            initializer_bytes=initializer_bytes,
            weight_tile_capacity_bytes=weight_tile_capacity,
            tile_width=tile_width,
            tile_counts=tuple(counts),
            tile_weight_bytes=tuple(
                count * row_bytes
                for count in counts
            ),
        )

    def to_plan_dict(self) -> dict:
        """
        Serialize compiler-visible target metadata into waveaccel_plan.json.

        Existing V1 keys are preserved so the current dependency-free manifest
        exporter and C++ runtime transport remain compatible. Additional fields
        document the target contract for compiler-side validation and tooling.
        """
        return {
            "name": self.name,
            "contract_schema_version": TARGET_CONTRACT_SCHEMA_VERSION,
            "sram_bytes": self.sram_bytes,
            "transient_reservation_bytes": self.transient_bytes,
            "initializer_staging_capacity_bytes":
                self.initializer_capacity_bytes,
            "dtype": self.dtype,
            "element_bytes": self.element_bytes,
            "supported_ops": list(self.supported_ops),
            "memory_spaces": {
                "model_initializers": self.device_memory_space,
                "initializer_staging": self.sram_memory_space,
            },
            "gemm_policy": {
                "tiling_axis": self.gemm_tiling_axis,
                "tile_unit": self.gemm_tile_unit,
                "tiled_fixed_initializer": "bias",
            },
        }
