#pragma once

// Author: Santosh Kumar
// Copyright 2026 Santosh Kumar
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace waveaccel {

struct ExecutionPlanTile {
    std::size_t tile_index{};
    std::size_t output_start{};
    std::size_t output_end{};
    std::size_t output_count{};
    std::size_t weight_bytes{};
    std::vector<std::string> actions;
};

struct ExecutionPlanNode {
    std::size_t node_index{};
    std::string op_type;
    std::string strategy;
    std::string tvm_primfunc;

    std::size_t initializer_total_bytes{};

    std::size_t in_features{};
    std::size_t out_features{};
    std::size_t weight_bytes{};
    std::size_t weight_row_bytes{};
    std::size_t bias_bytes{};

    std::size_t compiler_tile_width{};
    std::vector<std::string> actions;
    std::vector<ExecutionPlanTile> tiles;
};

struct ExecutionPlanTarget {
    std::string name;

    // Compiler/backend target contract transported from the TVM/Python side.
    // These are logical WaveAccel V1 semantics, not vendor hardware metadata.
    std::uint32_t contract_schema_version{};
    std::string dtype;
    std::size_t element_bytes{};
    std::vector<std::string> supported_ops;

    std::string model_initializer_memory_space;
    std::string initializer_staging_memory_space;

    std::string gemm_tiling_axis;
    std::string gemm_tile_unit;
    std::string gemm_tiled_fixed_initializer;

    std::size_t sram_bytes{};
    std::size_t transient_reservation_bytes{};
    std::size_t initializer_staging_capacity_bytes{};
};

struct ExecutionPlan {
    std::uint32_t schema_version{};
    std::string format;
    std::string tvm_version;
    ExecutionPlanTarget target;
    std::vector<ExecutionPlanNode> nodes;
};

ExecutionPlan load_execution_plan_manifest(const std::string& path);

}  // namespace waveaccel
