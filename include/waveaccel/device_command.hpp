#pragma once

// Author: Santosh Kumar
// Copyright 2026 Santosh Kumar
// SPDX-License-Identifier: Apache-2.0

#include "waveaccel/compiled_plan_adapter.hpp"
#include "waveaccel/device_memory.hpp"
#include "waveaccel/execution_plan.hpp"
#include "waveaccel/model_memory.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace waveaccel {

/*
 * Logical accelerator command stream.
 *
 * IMPORTANT:
 * These commands model a generic accelerator backend contract. They are not
 * physical DMA descriptors, MMIO commands, IOVAs, or a proprietary ISA.
 */
enum class DeviceCommandKind {
    DmaDeviceToSram,
    GemmTile,
    Gemm,
    Relu,
    Mul,
    Add,
};

struct DeviceCommand {
    DeviceCommandKind kind{};

    std::size_t node_index{};
    std::size_t tile_index{};

    // Diagnostic identity of the initializer region being staged.
    std::string source_region;

    // Offset inside source_region, retained for human-readable diagnostics.
    std::size_t source_offset_bytes{};

    // Absolute byte offset in WaveAccel's logical model-initializer device
    // address space. This is the authoritative backend source address.
    std::size_t device_offset_bytes{};

    std::size_t sram_offset_bytes{};
    std::size_t bytes{};

    // Compute metadata for tiled GEMM.
    std::size_t output_start{};
    std::size_t output_count{};
    std::size_t total_output_count{};
};

struct DeviceCommandStream {
    std::vector<DeviceCommand> commands;
    std::size_t device_to_sram_bytes{};
    std::size_t max_sram_initializer_end_bytes{};
};

// Address-aware lowering used by the compiler-driven Runtime path.
DeviceCommandStream lower_to_device_commands(
    const ExecutionPlan& plan,
    const CompiledRuntimeSchedule& schedule,
    const ModelMemoryInfo& model
);

// Legacy plan-only lowering retained for independent control-plane tests. Its
// device offsets are relative to its symbolic source regions, not the global
// model initializer address space. New runtime/backend code should use the
// ModelMemoryInfo overload above.
DeviceCommandStream lower_to_device_commands(
    const ExecutionPlan& plan,
    const CompiledRuntimeSchedule& schedule
);

std::string to_string(DeviceCommandKind kind);

}  // namespace waveaccel
