// Author: Santosh Kumar
// Copyright 2026 Santosh Kumar
// SPDX-License-Identifier: Apache-2.0

#include "waveaccel/compiled_plan_adapter.hpp"

#include <stdexcept>
#include <string>

namespace waveaccel {
namespace {

void append(
    CompiledRuntimeSchedule& schedule,
    CompiledCommandKind kind,
    std::size_t node_index,
    std::size_t bytes = 0,
    std::size_t tile_index = 0,
    std::size_t output_start = 0,
    std::size_t output_count = 0) {

    schedule.commands.push_back(
        CompiledRuntimeCommand{
            kind,
            node_index,
            tile_index,
            bytes,
            output_start,
            output_count});

    switch (kind) {
        case CompiledCommandKind::StageFixedInitializers:
        case CompiledCommandKind::StageWeightTile:
        case CompiledCommandKind::StageInitializersWhole:
            schedule.device_to_sram_bytes += bytes;
            break;

        default:
            break;
    }
}

}  // namespace

CompiledRuntimeSchedule compile_runtime_schedule(
    const ExecutionPlan& plan) {

    CompiledRuntimeSchedule schedule;

    for (const auto& node : plan.nodes) {
        if (node.op_type == "Gemm") {
            if (node.strategy == "TILED") {
                if (node.bias_bytes == 0) {
                    throw std::runtime_error(
                        "Tiled Gemm requires fixed bias initializers");
                }

                append(
                    schedule,
                    CompiledCommandKind::StageFixedInitializers,
                    node.node_index,
                    node.bias_bytes);

                if (node.tiles.empty()) {
                    throw std::runtime_error(
                        "Tiled Gemm has no compiler-provided tiles");
                }

                for (const auto& tile : node.tiles) {
                    append(
                        schedule,
                        CompiledCommandKind::StageWeightTile,
                        node.node_index,
                        tile.weight_bytes,
                        tile.tile_index,
                        tile.output_start,
                        tile.output_count);

                    append(
                        schedule,
                        CompiledCommandKind::ExecuteGemmTile,
                        node.node_index,
                        0,
                        tile.tile_index,
                        tile.output_start,
                        tile.output_count);
                }

                continue;
            }

            if (node.strategy == "STAGED_WHOLE") {
                append(
                    schedule,
                    CompiledCommandKind::StageInitializersWhole,
                    node.node_index,
                    node.initializer_total_bytes);

                append(
                    schedule,
                    CompiledCommandKind::ExecuteGemm,
                    node.node_index);
                continue;
            }

            throw std::runtime_error(
                "Unsupported compiler Gemm strategy at node " +
                std::to_string(node.node_index) + ": " +
                node.strategy);
        }

        if (node.op_type == "Relu") {
            if (node.strategy != "EXECUTE") {
                throw std::runtime_error(
                    "Relu compiler strategy must be EXECUTE");
            }

            append(
                schedule,
                CompiledCommandKind::ExecuteRelu,
                node.node_index);
            continue;
        }

        if (node.op_type == "Mul") {
            if (node.strategy != "STAGED_WHOLE") {
                throw std::runtime_error(
                    "Mul compiler strategy must be STAGED_WHOLE");
            }

            append(
                schedule,
                CompiledCommandKind::StageInitializersWhole,
                node.node_index,
                node.initializer_total_bytes);

            append(
                schedule,
                CompiledCommandKind::ExecuteMul,
                node.node_index);
            continue;
        }

        if (node.op_type == "Add") {
            if (node.strategy != "STAGED_WHOLE") {
                throw std::runtime_error(
                    "Add compiler strategy must be STAGED_WHOLE");
            }

            append(
                schedule,
                CompiledCommandKind::StageInitializersWhole,
                node.node_index,
                node.initializer_total_bytes);

            append(
                schedule,
                CompiledCommandKind::ExecuteAdd,
                node.node_index);
            continue;
        }

        throw std::runtime_error(
            "Unsupported compiler op_type at node " +
            std::to_string(node.node_index) + ": " +
            node.op_type);
    }

    return schedule;
}

std::string to_string(CompiledCommandKind kind) {
    switch (kind) {
        case CompiledCommandKind::StageFixedInitializers:
            return "STAGE_FIXED_INITIALIZERS";
        case CompiledCommandKind::StageWeightTile:
            return "STAGE_WEIGHT_TILE";
        case CompiledCommandKind::StageInitializersWhole:
            return "STAGE_INITIALIZERS_WHOLE";
        case CompiledCommandKind::ExecuteGemmTile:
            return "EXECUTE_GEMM_TILE";
        case CompiledCommandKind::ExecuteGemm:
            return "EXECUTE_GEMM";
        case CompiledCommandKind::ExecuteRelu:
            return "EXECUTE_RELU";
        case CompiledCommandKind::ExecuteMul:
            return "EXECUTE_MUL";
        case CompiledCommandKind::ExecuteAdd:
            return "EXECUTE_ADD";
    }

    throw std::runtime_error("Unknown compiled command kind");
}

}  // namespace waveaccel
