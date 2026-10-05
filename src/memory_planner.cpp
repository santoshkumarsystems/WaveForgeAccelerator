/*
 * WaveAccel - AI Accelerator Runtime Prototype
 *
 * Graph-aware SRAM memory planner with row-aligned GEMM tiling.
 *
 * Author: Santosh Kumar
 * Copyright 2026 Santosh Kumar
 * SPDX-License-Identifier: Apache-2.0
 */

#include "waveaccel/memory_planner.hpp"

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>
#include <vector>

namespace waveaccel {

namespace {

std::size_t ceil_div(
    std::size_t numerator,
    std::size_t denominator
) {
    return (numerator + denominator - 1) / denominator;
}

const ModelTensorInfo& find_tensor(
    const ModelMemoryInfo& model,
    const std::string& name
) {
    const auto found = std::find_if(
        model.initializers.begin(),
        model.initializers.end(),
        [&name](const ModelTensorInfo& tensor) {
            return tensor.name == name;
        }
    );

    if (found == model.initializers.end()) {
        throw std::runtime_error(
            "memory planner cannot find initializer: " + name
        );
    }

    return *found;
}

std::vector<std::size_t> parse_shape(
    const std::string& shape
) {
    std::vector<std::size_t> dims;

    std::size_t begin = 0;

    while (begin < shape.size()) {
        const auto end = shape.find('x', begin);
        const auto token =
            shape.substr(
                begin,
                end == std::string::npos
                    ? std::string::npos
                    : end - begin
            );

        if (token.empty()) {
            throw std::runtime_error(
                "invalid tensor shape: " + shape
            );
        }

        for (const char ch : token) {
            if (!std::isdigit(
                    static_cast<unsigned char>(ch)
                )) {
                throw std::runtime_error(
                    "non-numeric tensor shape: " + shape
                );
            }
        }

        const auto value =
            static_cast<std::size_t>(
                std::stoull(token)
            );

        if (value == 0) {
            throw std::runtime_error(
                "zero tensor dimension: " + shape
            );
        }

        dims.push_back(value);

        if (end == std::string::npos) {
            break;
        }

        begin = end + 1;
    }

    return dims;
}

void build_row_aligned_gemm_tiles(
    const ModelTensorInfo& weight,
    NodeMemoryPlan& node_plan
) {
    const auto dims = parse_shape(weight.shape);

    if (dims.size() != 2) {
        throw std::runtime_error(
            "WaveAccel V1 row tiling requires a rank-2 Gemm weight: " +
            weight.name
        );
    }

    const std::size_t total_rows = dims[0];

    if (weight.bytes % total_rows != 0) {
        throw std::runtime_error(
            "Gemm weight bytes are not divisible by output rows"
        );
    }

    const std::size_t row_bytes =
        weight.bytes / total_rows;

    if (row_bytes == 0) {
        throw std::runtime_error(
            "Gemm row size cannot be zero"
        );
    }

    const std::size_t max_rows_per_tile =
        node_plan.tile_capacity_bytes / row_bytes;

    if (max_rows_per_tile == 0) {
        throw std::runtime_error(
            "SRAM cannot hold one complete Gemm weight row"
        );
    }

    node_plan.weight_total_rows = total_rows;
    node_plan.weight_row_bytes = row_bytes;

    std::size_t row_begin = 0;
    std::size_t tile_index = 0;

    while (row_begin < total_rows) {
        const std::size_t rows_remaining =
            total_rows - row_begin;

        const std::size_t row_count =
            std::min(
                max_rows_per_tile,
                rows_remaining
            );

        node_plan.gemm_row_tiles.push_back(
            GemmRowTilePlan{
                tile_index,
                row_begin,
                row_count,
                row_count * row_bytes,
            }
        );

        row_begin += row_count;
        ++tile_index;
    }

    node_plan.tile_count =
        node_plan.gemm_row_tiles.size();
}

}  // namespace

MemoryPlan build_memory_plan(
    const ModelMemoryInfo& model,
    std::size_t sram_capacity_bytes
) {
    if (sram_capacity_bytes == 0) {
        throw std::invalid_argument(
            "SRAM capacity must be greater than zero"
        );
    }

    MemoryPlan plan;

    plan.sram_capacity_bytes = sram_capacity_bytes;
    plan.persistent_bytes = model.initializer_bytes;

    plan.transient_bytes =
        model.input_bytes +
        model.largest_activation_bytes +
        model.output_bytes;

    plan.estimated_working_set_bytes =
        plan.persistent_bytes +
        plan.transient_bytes;

    if (plan.transient_bytes >= sram_capacity_bytes) {
        throw std::runtime_error(
            "SRAM is too small for transient inference data"
        );
    }

    plan.initializer_staging_capacity_bytes =
        sram_capacity_bytes -
        plan.transient_bytes;

    const bool whole_model_fits =
        plan.estimated_working_set_bytes <=
        sram_capacity_bytes;

    plan.strategy =
        whole_model_fits
            ? MemoryStrategy::SramResident
            : MemoryStrategy::Tiled;

    for (const auto& tensor : model.initializers) {
        TensorMemoryPlan tensor_plan;

        tensor_plan.name = tensor.name;
        tensor_plan.shape = tensor.shape;
        tensor_plan.bytes = tensor.bytes;

        if (whole_model_fits) {
            tensor_plan.strategy =
                TensorMemoryStrategy::Resident;
            tensor_plan.tile_count = 1;
        } else if (
            tensor.bytes <=
            plan.initializer_staging_capacity_bytes
        ) {
            tensor_plan.strategy =
                TensorMemoryStrategy::StagedWhole;
            tensor_plan.tile_count = 1;
        } else {
            tensor_plan.strategy =
                TensorMemoryStrategy::Tiled;

            tensor_plan.tile_count =
                ceil_div(
                    tensor.bytes,
                    plan.initializer_staging_capacity_bytes
                );
        }

        plan.tensors.push_back(
            std::move(tensor_plan)
        );
    }

    for (
        std::size_t node_index = 0;
        node_index < model.nodes.size();
        ++node_index
    ) {
        const auto& node = model.nodes[node_index];

        NodeMemoryPlan node_plan;
        node_plan.node_index = node_index;
        node_plan.op = node.op;

        if (node.initializers.empty()) {
            node_plan.strategy =
                NodeMemoryStrategy::NoInitializerData;

            plan.nodes.push_back(
                std::move(node_plan)
            );

            continue;
        }

        std::size_t total_initializer_bytes = 0;

        for (const auto& initializer_name : node.initializers) {
            total_initializer_bytes +=
                find_tensor(model, initializer_name).bytes;
        }

        node_plan.initializer_bytes =
            total_initializer_bytes;

        if (whole_model_fits) {
            node_plan.strategy =
                NodeMemoryStrategy::Resident;

            plan.nodes.push_back(
                std::move(node_plan)
            );

            continue;
        }

        if (
            total_initializer_bytes <=
            plan.initializer_staging_capacity_bytes
        ) {
            node_plan.strategy =
                NodeMemoryStrategy::StagedWhole;

            plan.nodes.push_back(
                std::move(node_plan)
            );

            continue;
        }

        const ModelTensorInfo* largest = nullptr;
        std::size_t fixed_bytes = 0;

        for (const auto& initializer_name : node.initializers) {
            const auto& tensor =
                find_tensor(model, initializer_name);

            if (
                largest == nullptr ||
                tensor.bytes > largest->bytes
            ) {
                largest = &tensor;
            }
        }

        if (largest == nullptr) {
            throw std::runtime_error(
                "node tiling selected without an initializer"
            );
        }

        for (const auto& initializer_name : node.initializers) {
            const auto& tensor =
                find_tensor(model, initializer_name);

            if (tensor.name != largest->name) {
                fixed_bytes += tensor.bytes;
            }
        }

        if (
            fixed_bytes >=
            plan.initializer_staging_capacity_bytes
        ) {
            throw std::runtime_error(
                "node requires unsupported multi-tensor tiling: " +
                node.op
            );
        }

        node_plan.strategy =
            NodeMemoryStrategy::SingleTensorTiled;

        node_plan.tiled_tensor_name =
            largest->name;

        node_plan.tiled_tensor_bytes =
            largest->bytes;

        node_plan.fixed_initializer_bytes =
            fixed_bytes;

        node_plan.tile_capacity_bytes =
            plan.initializer_staging_capacity_bytes -
            fixed_bytes;

        if (node.op != "Gemm") {
            throw std::runtime_error(
                "WaveAccel V1 only supports row-aligned tiling for Gemm"
            );
        }

        build_row_aligned_gemm_tiles(
            *largest,
            node_plan
        );

        plan.nodes.push_back(
            std::move(node_plan)
        );
    }

    return plan;
}

const char* memory_strategy_name(
    MemoryStrategy strategy
) noexcept {
    switch (strategy) {
        case MemoryStrategy::SramResident:
            return "SRAM_RESIDENT";

        case MemoryStrategy::Tiled:
            return "TILED";
    }

    return "UNKNOWN";
}

const char* tensor_memory_strategy_name(
    TensorMemoryStrategy strategy
) noexcept {
    switch (strategy) {
        case TensorMemoryStrategy::Resident:
            return "RESIDENT";

        case TensorMemoryStrategy::StagedWhole:
            return "STAGED_WHOLE";

        case TensorMemoryStrategy::Tiled:
            return "TILED";
    }

    return "UNKNOWN";
}

const char* node_memory_strategy_name(
    NodeMemoryStrategy strategy
) noexcept {
    switch (strategy) {
        case NodeMemoryStrategy::Resident:
            return "RESIDENT";

        case NodeMemoryStrategy::NoInitializerData:
            return "NO_INITIALIZER_DATA";

        case NodeMemoryStrategy::StagedWhole:
            return "STAGED_WHOLE";

        case NodeMemoryStrategy::SingleTensorTiled:
            return "SINGLE_TENSOR_TILED";
    }

    return "UNKNOWN";
}

}  // namespace waveaccel
