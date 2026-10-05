// Author: Santosh Kumar
// Copyright 2026 Santosh Kumar
// SPDX-License-Identifier: Apache-2.0

#include "waveaccel/device_command.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace waveaccel {
namespace {

const ExecutionPlanNode& node_at(
    const ExecutionPlan& plan,
    std::size_t node_index
) {
    if (node_index >= plan.nodes.size()) {
        throw std::runtime_error(
            "device-command lowering received invalid node index"
        );
    }

    return plan.nodes[node_index];
}

const ModelGraphNodeInfo& model_node_at(
    const ModelMemoryInfo& model,
    std::size_t node_index
) {
    if (node_index >= model.nodes.size()) {
        throw std::runtime_error(
            "device-command lowering received invalid model node index"
        );
    }

    return model.nodes[node_index];
}

void validate_device_range(
    const DeviceMemoryMap* map,
    std::size_t device_offset_bytes,
    std::size_t bytes
) {
    if (map == nullptr) {
        return;
    }

    if (
        device_offset_bytes > map->total_bytes ||
        bytes > map->total_bytes - device_offset_bytes
    ) {
        throw std::runtime_error(
            "device-command DMA exceeds logical device-memory model region"
        );
    }
}

void append_dma(
    DeviceCommandStream& stream,
    const ExecutionPlan& plan,
    const DeviceMemoryMap* map,
    std::size_t node_index,
    std::size_t tile_index,
    std::string source_region,
    std::size_t source_offset_bytes,
    std::size_t device_offset_bytes,
    std::size_t sram_offset_bytes,
    std::size_t bytes
) {
    const auto initializer_capacity =
        plan.target.initializer_staging_capacity_bytes;

    if (
        sram_offset_bytes > initializer_capacity ||
        bytes > initializer_capacity - sram_offset_bytes
    ) {
        throw std::runtime_error(
            "device-command DMA exceeds compiler-declared SRAM "
            "initializer staging capacity"
        );
    }

    validate_device_range(
        map,
        device_offset_bytes,
        bytes
    );

    DeviceCommand command;
    command.kind = DeviceCommandKind::DmaDeviceToSram;
    command.node_index = node_index;
    command.tile_index = tile_index;
    command.source_region = std::move(source_region);
    command.source_offset_bytes = source_offset_bytes;
    command.device_offset_bytes = device_offset_bytes;
    command.sram_offset_bytes = sram_offset_bytes;
    command.bytes = bytes;

    stream.commands.push_back(std::move(command));

    stream.device_to_sram_bytes += bytes;
    stream.max_sram_initializer_end_bytes =
        std::max(
            stream.max_sram_initializer_end_bytes,
            sram_offset_bytes + bytes
        );
}

void append_compute(
    DeviceCommandStream& stream,
    DeviceCommandKind kind,
    std::size_t node_index,
    std::size_t tile_index = 0,
    std::size_t output_start = 0,
    std::size_t output_count = 0,
    std::size_t total_output_count = 0
) {
    DeviceCommand command;
    command.kind = kind;
    command.node_index = node_index;
    command.tile_index = tile_index;
    command.output_start = output_start;
    command.output_count = output_count;
    command.total_output_count = total_output_count;

    stream.commands.push_back(std::move(command));
}

std::pair<std::size_t, std::size_t> contiguous_initializer_span(
    const ModelMemoryInfo& model,
    const DeviceMemoryMap& map,
    std::size_t node_index,
    std::size_t expected_bytes
) {
    const auto& graph_node =
        model_node_at(model, node_index);

    if (graph_node.initializers.empty()) {
        if (expected_bytes != 0) {
            throw std::runtime_error(
                "compiler expects initializer bytes for initializer-free node"
            );
        }
        return {0, 0};
    }

    const auto& first =
        find_device_memory_region(
            map,
            graph_node.initializers.front()
        );

    std::size_t next_offset = first.offset_bytes;
    std::size_t total = 0;

    for (const auto& name : graph_node.initializers) {
        const auto& region =
            find_device_memory_region(map, name);

        if (region.offset_bytes != next_offset) {
            throw std::runtime_error(
                "node initializers are not contiguous in logical device memory"
            );
        }

        if (
            region.bytes >
            std::numeric_limits<std::size_t>::max() - total
        ) {
            throw std::runtime_error(
                "node initializer span overflow"
            );
        }

        total += region.bytes;
        next_offset += region.bytes;
    }

    if (total != expected_bytes) {
        throw std::runtime_error(
            "node initializer device-memory span does not match compiler bytes"
        );
    }

    return {first.offset_bytes, total};
}

DeviceCommandStream lower_impl(
    const ExecutionPlan& plan,
    const CompiledRuntimeSchedule& schedule,
    const ModelMemoryInfo* model,
    const DeviceMemoryMap* map
) {
    DeviceCommandStream stream;

    if ((model == nullptr) != (map == nullptr)) {
        throw std::logic_error(
            "device-command lowering requires model and map together"
        );
    }

    if (model != nullptr) {
        if (model->nodes.size() != plan.nodes.size()) {
            throw std::runtime_error(
                "device-memory lowering model/plan node counts differ"
            );
        }

        if (map->total_bytes != model->initializer_bytes) {
            throw std::runtime_error(
                "device-memory map/model initializer byte totals differ"
            );
        }
    }

    for (const auto& compiled_command : schedule.commands) {
        const auto& node =
            node_at(plan, compiled_command.node_index);

        switch (compiled_command.kind) {
            case CompiledCommandKind::StageFixedInitializers: {
                if (model == nullptr) {
                    append_dma(
                        stream,
                        plan,
                        nullptr,
                        compiled_command.node_index,
                        0,
                        "node." +
                            std::to_string(compiled_command.node_index) +
                            ".fixed_initializers",
                        0,
                        0,
                        0,
                        compiled_command.bytes
                    );
                    break;
                }

                const auto& graph_node =
                    model_node_at(
                        *model,
                        compiled_command.node_index
                    );

                if (
                    node.op_type != "Gemm" ||
                    graph_node.initializers.size() != 2
                ) {
                    throw std::runtime_error(
                        "fixed-initializer DMA requires Gemm weight+bias metadata"
                    );
                }

                const auto& bias =
                    find_device_memory_region(
                        *map,
                        graph_node.initializers[1]
                    );

                if (
                    bias.bytes != node.bias_bytes ||
                    bias.bytes != compiled_command.bytes
                ) {
                    throw std::runtime_error(
                        "Gemm bias device-memory bytes do not match compiler plan"
                    );
                }

                append_dma(
                    stream,
                    plan,
                    map,
                    compiled_command.node_index,
                    0,
                    bias.tensor_name,
                    0,
                    bias.offset_bytes,
                    0,
                    bias.bytes
                );
                break;
            }

            case CompiledCommandKind::StageWeightTile: {
                if (
                    node.op_type != "Gemm" ||
                    node.strategy != "TILED"
                ) {
                    throw std::runtime_error(
                        "weight-tile DMA requires tiled Gemm node"
                    );
                }

                const std::size_t source_offset =
                    compiled_command.output_start *
                    node.weight_row_bytes;

                const std::size_t sram_offset =
                    node.bias_bytes;

                if (model == nullptr) {
                    append_dma(
                        stream,
                        plan,
                        nullptr,
                        compiled_command.node_index,
                        compiled_command.tile_index,
                        "node." +
                            std::to_string(compiled_command.node_index) +
                            ".weight",
                        source_offset,
                        source_offset,
                        sram_offset,
                        compiled_command.bytes
                    );
                    break;
                }

                const auto& graph_node =
                    model_node_at(
                        *model,
                        compiled_command.node_index
                    );

                if (graph_node.initializers.size() != 2) {
                    throw std::runtime_error(
                        "tiled Gemm requires weight+bias initializer metadata"
                    );
                }

                const auto& weight =
                    find_device_memory_region(
                        *map,
                        graph_node.initializers[0]
                    );

                if (weight.bytes != node.weight_bytes) {
                    throw std::runtime_error(
                        "Gemm weight device-memory bytes do not match compiler plan"
                    );
                }

                if (
                    source_offset > weight.bytes ||
                    compiled_command.bytes >
                        weight.bytes - source_offset
                ) {
                    throw std::runtime_error(
                        "Gemm weight tile exceeds logical weight region"
                    );
                }

                append_dma(
                    stream,
                    plan,
                    map,
                    compiled_command.node_index,
                    compiled_command.tile_index,
                    weight.tensor_name,
                    source_offset,
                    weight.offset_bytes + source_offset,
                    sram_offset,
                    compiled_command.bytes
                );
                break;
            }

            case CompiledCommandKind::StageInitializersWhole: {
                if (model == nullptr) {
                    append_dma(
                        stream,
                        plan,
                        nullptr,
                        compiled_command.node_index,
                        0,
                        "node." +
                            std::to_string(compiled_command.node_index) +
                            ".initializers",
                        0,
                        0,
                        0,
                        compiled_command.bytes
                    );
                    break;
                }

                const auto& graph_node =
                    model_node_at(
                        *model,
                        compiled_command.node_index
                    );

                const auto span =
                    contiguous_initializer_span(
                        *model,
                        *map,
                        compiled_command.node_index,
                        compiled_command.bytes
                    );

                const std::string label =
                    graph_node.initializers.size() == 1
                        ? graph_node.initializers.front()
                        : "node." +
                            std::to_string(compiled_command.node_index) +
                            ".contiguous_initializers";

                append_dma(
                    stream,
                    plan,
                    map,
                    compiled_command.node_index,
                    0,
                    label,
                    0,
                    span.first,
                    0,
                    span.second
                );
                break;
            }

            case CompiledCommandKind::ExecuteGemmTile:
                append_compute(
                    stream,
                    DeviceCommandKind::GemmTile,
                    compiled_command.node_index,
                    compiled_command.tile_index,
                    compiled_command.output_start,
                    compiled_command.output_count,
                    node.out_features
                );
                break;

            case CompiledCommandKind::ExecuteGemm:
                append_compute(
                    stream,
                    DeviceCommandKind::Gemm,
                    compiled_command.node_index
                );
                break;

            case CompiledCommandKind::ExecuteRelu:
                append_compute(
                    stream,
                    DeviceCommandKind::Relu,
                    compiled_command.node_index
                );
                break;

            case CompiledCommandKind::ExecuteMul:
                append_compute(
                    stream,
                    DeviceCommandKind::Mul,
                    compiled_command.node_index
                );
                break;

            case CompiledCommandKind::ExecuteAdd:
                append_compute(
                    stream,
                    DeviceCommandKind::Add,
                    compiled_command.node_index
                );
                break;
        }
    }

    if (
        stream.device_to_sram_bytes !=
        schedule.device_to_sram_bytes
    ) {
        throw std::runtime_error(
            "device-command lowering changed Device->SRAM byte accounting"
        );
    }

    return stream;
}

}  // namespace

DeviceCommandStream lower_to_device_commands(
    const ExecutionPlan& plan,
    const CompiledRuntimeSchedule& schedule,
    const ModelMemoryInfo& model
) {
    const auto map = build_device_memory_map(model);

    return lower_impl(
        plan,
        schedule,
        &model,
        &map
    );
}

DeviceCommandStream lower_to_device_commands(
    const ExecutionPlan& plan,
    const CompiledRuntimeSchedule& schedule
) {
    return lower_impl(
        plan,
        schedule,
        nullptr,
        nullptr
    );
}

std::string to_string(DeviceCommandKind kind) {
    switch (kind) {
        case DeviceCommandKind::DmaDeviceToSram:
            return "DMA_DEVICE_TO_SRAM";
        case DeviceCommandKind::GemmTile:
            return "GEMM_TILE";
        case DeviceCommandKind::Gemm:
            return "GEMM";
        case DeviceCommandKind::Relu:
            return "RELU";
        case DeviceCommandKind::Mul:
            return "MUL";
        case DeviceCommandKind::Add:
            return "ADD";
    }

    throw std::runtime_error(
        "unknown WaveAccel device command kind"
    );
}

}  // namespace waveaccel
