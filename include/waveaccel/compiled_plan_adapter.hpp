#pragma once

// Author: Santosh Kumar
// Copyright 2026 Santosh Kumar
// SPDX-License-Identifier: Apache-2.0

#include "waveaccel/execution_plan.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace waveaccel {

enum class CompiledCommandKind {
    StageFixedInitializers,
    StageWeightTile,
    StageInitializersWhole,
    ExecuteGemmTile,
    ExecuteGemm,
    ExecuteRelu,
    ExecuteMul,
    ExecuteAdd,
};

struct CompiledRuntimeCommand {
    CompiledCommandKind kind{};
    std::size_t node_index{};
    std::size_t tile_index{};
    std::size_t bytes{};
    std::size_t output_start{};
    std::size_t output_count{};
};

struct CompiledRuntimeSchedule {
    std::vector<CompiledRuntimeCommand> commands;
    std::size_t device_to_sram_bytes{};
};

CompiledRuntimeSchedule compile_runtime_schedule(
    const ExecutionPlan& plan);

std::string to_string(CompiledCommandKind kind);

}  // namespace waveaccel
