/*
 * WaveAccel - AI Accelerator Runtime Prototype
 *
 * Graph-aware accelerator SRAM memory planner.
 *
 * Author: Santosh Kumar
 * Copyright 2026 Santosh Kumar
 * SPDX-License-Identifier: Apache-2.0
 *
 * For a tiled rank-2 GEMM weight tensor, V1 now creates row-aligned tiles.
 * This means the memory schedule can be executed numerically tile-by-tile
 * without splitting a matrix row across SRAM transfers.
 *
 * NOTE:
 * This is a first-order row-output tiling strategy for WaveAccel's current
 * MLP model. It is not yet TVM-generated M/N/K tensorization or a proprietary
 * accelerator schedule.
 */

#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "waveaccel/model_memory.hpp"

namespace waveaccel {

enum class MemoryStrategy {
    SramResident,
    Tiled,
};

enum class TensorMemoryStrategy {
    Resident,
    StagedWhole,
    Tiled,
};

enum class NodeMemoryStrategy {
    Resident,
    NoInitializerData,
    StagedWhole,
    SingleTensorTiled,
};

struct TensorMemoryPlan {
    std::string name;
    std::string shape;

    std::size_t bytes{0};
    std::size_t tile_count{0};

    TensorMemoryStrategy strategy{
        TensorMemoryStrategy::StagedWhole
    };
};

struct GemmRowTilePlan {
    std::size_t tile_index{0};
    std::size_t row_begin{0};
    std::size_t row_count{0};
    std::size_t bytes{0};
};

struct NodeMemoryPlan {
    std::size_t node_index{0};
    std::string op;

    std::size_t initializer_bytes{0};

    NodeMemoryStrategy strategy{
        NodeMemoryStrategy::NoInitializerData
    };

    std::string tiled_tensor_name;
    std::size_t tiled_tensor_bytes{0};
    std::size_t fixed_initializer_bytes{0};
    std::size_t tile_capacity_bytes{0};
    std::size_t tile_count{0};

    // For WaveAccel V1 Gemm tiling: each tile owns complete output rows.
    std::size_t weight_total_rows{0};
    std::size_t weight_row_bytes{0};
    std::vector<GemmRowTilePlan> gemm_row_tiles;
};

struct MemoryPlan {
    std::size_t sram_capacity_bytes{0};

    std::size_t persistent_bytes{0};
    std::size_t transient_bytes{0};
    std::size_t estimated_working_set_bytes{0};

    std::size_t initializer_staging_capacity_bytes{0};

    MemoryStrategy strategy{MemoryStrategy::Tiled};

    std::vector<TensorMemoryPlan> tensors;
    std::vector<NodeMemoryPlan> nodes;
};

MemoryPlan build_memory_plan(
    const ModelMemoryInfo& model,
    std::size_t sram_capacity_bytes
);

const char* memory_strategy_name(
    MemoryStrategy strategy
) noexcept;

const char* tensor_memory_strategy_name(
    TensorMemoryStrategy strategy
) noexcept;

const char* node_memory_strategy_name(
    NodeMemoryStrategy strategy
) noexcept;

}  // namespace waveaccel
