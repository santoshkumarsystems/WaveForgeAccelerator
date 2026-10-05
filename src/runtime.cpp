/*
 * WaveAccel - AI Accelerator Runtime Prototype
 *
 * Graph-driven runtime with real row-aligned tiled GEMM execution.
 *
 * Author: Santosh Kumar
 * Copyright 2026 Santosh Kumar
 * SPDX-License-Identifier: Apache-2.0
 */

#include "waveaccel/runtime.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "waveaccel/compiled_plan_adapter.hpp"
#include "waveaccel/device_command.hpp"
#include "waveaccel/numerical_executor.hpp"

namespace waveaccel {

namespace {

std::string node_prefix(
    std::size_t node_index
) {
    return "NODE_" + std::to_string(node_index) + "_";
}

}  // namespace

Runtime::Runtime(
    ModelMemoryInfo model,
    MockDevice device
)
    : model_(std::move(model)),
      device_(std::move(device)),
      memory_plan_(
          build_memory_plan(
              model_,
              device_.sram_capacity_bytes()
          )
      ) {
    if (model_.nodes.size() != memory_plan_.nodes.size()) {
        throw std::runtime_error(
            "runtime graph and memory-plan node counts differ"
        );
    }
}


void Runtime::set_execution_plan(
    ExecutionPlan plan
) {
    if (plan.target.contract_schema_version != 1) {
        throw std::runtime_error(
            "compiler execution-plan target contract schema is unsupported"
        );
    }

    if (
        plan.target.dtype != "float32" ||
        plan.target.element_bytes != sizeof(float)
    ) {
        throw std::runtime_error(
            "compiler execution-plan dtype is not supported by Runtime"
        );
    }

    if (
        plan.target.model_initializer_memory_space !=
            "device_memory" ||
        plan.target.initializer_staging_memory_space !=
            "sram"
    ) {
        throw std::runtime_error(
            "compiler execution-plan memory-space contract is unsupported"
        );
    }

    if (
        plan.target.gemm_tiling_axis != "output_channel" ||
        plan.target.gemm_tile_unit !=
            "complete_output_weight_row" ||
        plan.target.gemm_tiled_fixed_initializer != "bias"
    ) {
        throw std::runtime_error(
            "compiler execution-plan Gemm policy contract is unsupported"
        );
    }

    if (
        plan.target.sram_bytes !=
        device_.sram_capacity_bytes()
    ) {
        throw std::runtime_error(
            "compiler execution-plan SRAM capacity does not match device"
        );
    }

    if (plan.nodes.size() != model_.nodes.size()) {
        throw std::runtime_error(
            "compiler execution-plan node count does not match ONNX graph"
        );
    }

    std::size_t compiler_initializer_bytes = 0;

    for (
        std::size_t node_index = 0;
        node_index < plan.nodes.size();
        ++node_index
    ) {
        const auto& compiler_node =
            plan.nodes[node_index];

        const auto& model_node =
            model_.nodes[node_index];

        if (
            std::find(
                plan.target.supported_ops.begin(),
                plan.target.supported_ops.end(),
                model_node.op
            ) == plan.target.supported_ops.end()
        ) {
            throw std::runtime_error(
                "compiler execution-plan target does not support model op: " +
                model_node.op
            );
        }

        if (
            compiler_node.node_index != node_index ||
            compiler_node.op_type != model_node.op
        ) {
            throw std::runtime_error(
                "compiler execution-plan graph does not match ONNX graph"
            );
        }

        compiler_initializer_bytes +=
            compiler_node.initializer_total_bytes;
    }

    if (
        compiler_initializer_bytes !=
        model_.initializer_bytes
    ) {
        throw std::runtime_error(
            "compiler execution-plan initializer bytes do not match model"
        );
    }

    device_.configure_address_contract(
        model_.initializer_bytes,
        plan.target.initializer_staging_capacity_bytes
    );

    execution_plan_ = std::move(plan);
}

Task Runtime::build_compute_task(
    std::size_t node_index
) const {
    const auto& node = model_.nodes.at(node_index);

    const std::string prefix =
        node_prefix(node_index);

    if (node.op == "Gemm") {
        return Task{
            TaskKind::Gemm,
            prefix + "GEMM",
            0,
            node_index,
            kNoTileIndex,
            1,
            0,
            0,
            0,
        };
    }

    if (node.op == "Relu") {
        return Task{
            TaskKind::Relu,
            prefix + "RELU",
            0,
            node_index,
            kNoTileIndex,
            1,
            0,
            0,
            0,
        };
    }

    if (node.op == "Mul") {
        return Task{
            TaskKind::Mul,
            prefix + "MUL",
            0,
            node_index,
            kNoTileIndex,
            1,
            0,
            0,
            0,
        };
    }

    if (node.op == "Add") {
        return Task{
            TaskKind::Add,
            prefix + "ADD",
            0,
            node_index,
            kNoTileIndex,
            1,
            0,
            0,
            0,
        };
    }

    throw std::runtime_error(
        "unsupported ONNX operator in runtime: " +
        node.op
    );
}

std::vector<Task> Runtime::build_plan() const {
    std::vector<Task> plan;

    plan.push_back(
        Task{
            TaskKind::H2D,
            "DMA_H2D_INPUT",
            model_.input_bytes,
        }
    );

    if (!device_.model_in_device_memory()) {
        plan.push_back(
            Task{
                TaskKind::LoadModelToDevice,
                "DMA_H2D_MODEL_INITIALIZERS",
                model_.initializer_bytes,
            }
        );
    }

    if (execution_plan_.has_value()) {
        const auto compiled =
            compile_runtime_schedule(
                *execution_plan_
            );

        for (const auto& command : compiled.commands) {
            if (
                command.node_index >=
                execution_plan_->nodes.size()
            ) {
                throw std::runtime_error(
                    "compiler runtime command has invalid node index"
                );
            }

            const auto& compiler_node =
                execution_plan_->nodes[
                    command.node_index
                ];

            const std::string prefix =
                node_prefix(command.node_index);

            switch (command.kind) {
                case CompiledCommandKind::StageFixedInitializers:
                    plan.push_back(
                        Task{
                            TaskKind::StageToSram,
                            prefix +
                                "STAGE_FIXED_INITIALIZERS",
                            command.bytes,
                        }
                    );
                    break;

                case CompiledCommandKind::StageWeightTile:
                    plan.push_back(
                        Task{
                            TaskKind::StageToSram,
                            prefix +
                                "STAGE_WEIGHT_TILE_" +
                                std::to_string(
                                    command.tile_index
                                ),
                            command.bytes,
                        }
                    );
                    break;

                case CompiledCommandKind::StageInitializersWhole:
                    plan.push_back(
                        Task{
                            TaskKind::StageToSram,
                            prefix +
                                "STAGE_INITIALIZERS_WHOLE",
                            command.bytes,
                        }
                    );
                    break;

                case CompiledCommandKind::ExecuteGemmTile:
                    if (
                        compiler_node.strategy != "TILED" ||
                        compiler_node.tiles.empty()
                    ) {
                        throw std::runtime_error(
                            "compiler GemmTile command lacks tiled node metadata"
                        );
                    }

                    plan.push_back(
                        Task{
                            TaskKind::GemmTile,
                            prefix +
                                "GEMM_TILE_" +
                                std::to_string(
                                    command.tile_index
                                ),
                            0,
                            command.node_index,
                            command.tile_index,
                            compiler_node.tiles.size(),
                            command.output_start,
                            command.output_count,
                            compiler_node.out_features,
                        }
                    );
                    break;

                case CompiledCommandKind::ExecuteGemm:
                case CompiledCommandKind::ExecuteRelu:
                case CompiledCommandKind::ExecuteMul:
                case CompiledCommandKind::ExecuteAdd:
                    plan.push_back(
                        build_compute_task(
                            command.node_index
                        )
                    );
                    break;
            }
        }

        plan.push_back(
            Task{
                TaskKind::D2H,
                "DMA_D2H_OUTPUT",
                model_.output_bytes,
            }
        );

        return plan;
    }

    if (
        memory_plan_.strategy ==
        MemoryStrategy::SramResident
    ) {
        if (!device_.model_in_sram()) {
            plan.push_back(
                Task{
                    TaskKind::StageModelToSram,
                    "STAGE_MODEL_TO_SRAM",
                    model_.initializer_bytes,
                }
            );
        }

        for (
            std::size_t node_index = 0;
            node_index < model_.nodes.size();
            ++node_index
        ) {
            plan.push_back(
                build_compute_task(node_index)
            );
        }
    } else {
        for (
            std::size_t node_index = 0;
            node_index < model_.nodes.size();
            ++node_index
        ) {
            const auto& node_plan =
                memory_plan_.nodes.at(node_index);

            const std::string prefix =
                node_prefix(node_index);

            switch (node_plan.strategy) {
                case NodeMemoryStrategy::NoInitializerData:
                case NodeMemoryStrategy::Resident:
                    plan.push_back(
                        build_compute_task(node_index)
                    );
                    break;

                case NodeMemoryStrategy::StagedWhole:
                    plan.push_back(
                        Task{
                            TaskKind::StageToSram,
                            prefix +
                                "STAGE_INITIALIZERS_WHOLE",
                            node_plan.initializer_bytes,
                        }
                    );

                    plan.push_back(
                        build_compute_task(node_index)
                    );
                    break;

                case NodeMemoryStrategy::SingleTensorTiled: {
                    if (
                        node_plan.fixed_initializer_bytes > 0
                    ) {
                        plan.push_back(
                            Task{
                                TaskKind::StageToSram,
                                prefix +
                                    "STAGE_FIXED_INITIALIZERS",
                                node_plan.
                                    fixed_initializer_bytes,
                            }
                        );
                    }

                    if (node_plan.gemm_row_tiles.empty()) {
                        throw std::runtime_error(
                            "tiled Gemm has no row tiles"
                        );
                    }

                    for (
                        const auto& tile :
                            node_plan.gemm_row_tiles
                    ) {
                        plan.push_back(
                            Task{
                                TaskKind::StageToSram,
                                prefix +
                                    "STAGE_WEIGHT_TILE_" +
                                    std::to_string(
                                        tile.tile_index
                                    ),
                                tile.bytes,
                            }
                        );

                        plan.push_back(
                            Task{
                                TaskKind::GemmTile,
                                prefix +
                                    "GEMM_TILE_" +
                                    std::to_string(
                                        tile.tile_index
                                    ),
                                0,
                                node_index,
                                tile.tile_index,
                                node_plan.tile_count,
                                tile.row_begin,
                                tile.row_count,
                                node_plan.weight_total_rows,
                            }
                        );
                    }

                    break;
                }
            }
        }
    }

    plan.push_back(
        Task{
            TaskKind::D2H,
            "DMA_D2H_OUTPUT",
            model_.output_bytes,
        }
    );

    return plan;
}

Profiler Runtime::run_once() {
    Profiler profiler;

    const auto plan = build_plan();

    for (const auto& task : plan) {
        device_.execute(task, profiler);
    }

    return profiler;
}

RuntimeExecutionResult Runtime::run_once(
    const std::vector<float>& input,
    const TensorStore& tensors
) {
    // compiler-driven numerical path executes DeviceCommandStream directly.
    // Host<->Device transfers remain the outer Runtime envelope.
    if (execution_plan_.has_value()) {
        RuntimeExecutionResult result;

        NumericalExecutor executor(
            model_,
            tensors
        );

        std::vector<float> activation = input;

        device_.execute(
            Task{
                TaskKind::H2D,
                "DMA_H2D_INPUT",
                model_.input_bytes,
            },
            result.profiler
        );

        if (!device_.model_in_device_memory()) {
            device_.execute(
                Task{
                    TaskKind::LoadModelToDevice,
                    "DMA_H2D_MODEL_INITIALIZERS",
                    model_.initializer_bytes,
                },
                result.profiler
            );
        }

        const auto compiled =
            compile_runtime_schedule(
                *execution_plan_
            );

        const auto device_commands =
            lower_to_device_commands(
                *execution_plan_,
                compiled,
                model_
            );

        std::vector<float> tiled_output;
        std::size_t active_tiled_node =
            kNoNodeIndex;
        std::size_t expected_tile_index = 0;

        for (
            const auto& command :
            device_commands.commands
        ) {
            if (
                command.node_index >=
                model_.nodes.size()
            ) {
                throw std::runtime_error(
                    "DeviceCommand has invalid ONNX node index"
                );
            }

            device_.execute(
                command,
                result.profiler
            );

            if (
                command.kind ==
                DeviceCommandKind::DmaDeviceToSram
            ) {
                continue;
            }

            const auto& node =
                model_.nodes[
                    command.node_index
                ];

            if (
                command.kind ==
                DeviceCommandKind::GemmTile
            ) {
                if (
                    active_tiled_node ==
                    kNoNodeIndex
                ) {
                    active_tiled_node =
                        command.node_index;

                    expected_tile_index = 0;

                    tiled_output.clear();
                    tiled_output.reserve(
                        command.total_output_count
                    );
                }

                if (
                    active_tiled_node !=
                        command.node_index ||
                    command.tile_index !=
                        expected_tile_index
                ) {
                    throw std::runtime_error(
                        "out-of-order DeviceCommand tiled Gemm"
                    );
                }

                if (
                    command.output_start !=
                    tiled_output.size()
                ) {
                    throw std::runtime_error(
                        "non-contiguous DeviceCommand tiled Gemm output"
                    );
                }

                const auto partial =
                    executor.execute_gemm_rows(
                        node,
                        activation,
                        command.output_start,
                        command.output_count
                    );

                tiled_output.insert(
                    tiled_output.end(),
                    partial.begin(),
                    partial.end()
                );

                ++expected_tile_index;

                if (
                    tiled_output.size() ==
                    command.total_output_count
                ) {
                    activation =
                        std::move(tiled_output);

                    tiled_output.clear();
                    active_tiled_node =
                        kNoNodeIndex;
                    expected_tile_index = 0;
                } else if (
                    tiled_output.size() >
                    command.total_output_count
                ) {
                    throw std::runtime_error(
                        "DeviceCommand tiled Gemm output overflow"
                    );
                }

                continue;
            }

            if (
                active_tiled_node !=
                kNoNodeIndex
            ) {
                throw std::runtime_error(
                    "non-tiled DeviceCommand compute encountered before "
                    "tiled Gemm completed"
                );
            }

            activation =
                executor.execute_node(
                    node,
                    activation
                );
        }

        if (
            active_tiled_node !=
            kNoNodeIndex
        ) {
            throw std::runtime_error(
                "compiler-driven runtime ended before tiled Gemm completed"
            );
        }

        device_.execute(
            Task{
                TaskKind::D2H,
                "DMA_D2H_OUTPUT",
                model_.output_bytes,
            },
            result.profiler
        );

        result.output =
            std::move(activation);

        return result;
    }

    RuntimeExecutionResult result;

    NumericalExecutor executor(
        model_,
        tensors
    );

    std::vector<float> activation = input;

    // A tiled Gemm keeps the input activation unchanged while each output-row
    // tile is computed. Tile outputs are appended in row order. Only after the
    // final tile do they become the activation for the next ONNX node.
    std::vector<float> tiled_output;
    std::size_t active_tiled_node = kNoNodeIndex;
    std::size_t expected_tile_index = 0;

    const auto plan = build_plan();

    for (const auto& task : plan) {
        device_.execute(
            task,
            result.profiler
        );

        if (!is_compute_task(task.kind)) {
            continue;
        }

        if (
            task.node_index == kNoNodeIndex ||
            task.node_index >= model_.nodes.size()
        ) {
            throw std::runtime_error(
                "compute task has invalid ONNX node index"
            );
        }

        const auto& node =
            model_.nodes[task.node_index];

        if (task.kind == TaskKind::GemmTile) {
            if (active_tiled_node == kNoNodeIndex) {
                active_tiled_node = task.node_index;
                expected_tile_index = 0;
                tiled_output.clear();
                tiled_output.reserve(task.total_rows);
            }

            if (
                active_tiled_node != task.node_index ||
                task.tile_index != expected_tile_index
            ) {
                throw std::runtime_error(
                    "out-of-order tiled Gemm runtime schedule"
                );
            }

            const auto partial =
                executor.execute_gemm_rows(
                    node,
                    activation,
                    task.row_begin,
                    task.row_count
                );

            tiled_output.insert(
                tiled_output.end(),
                partial.begin(),
                partial.end()
            );

            ++expected_tile_index;

            if (expected_tile_index == task.tile_count) {
                if (
                    tiled_output.size() !=
                    task.total_rows
                ) {
                    throw std::runtime_error(
                        "tiled Gemm output row count mismatch"
                    );
                }

                activation =
                    std::move(tiled_output);

                tiled_output.clear();
                active_tiled_node = kNoNodeIndex;
                expected_tile_index = 0;
            }

            continue;
        }

        if (active_tiled_node != kNoNodeIndex) {
            throw std::runtime_error(
                "non-tiled compute encountered before "
                "tiled Gemm completed"
            );
        }

        activation =
            executor.execute_node(
                node,
                activation
            );
    }

    if (active_tiled_node != kNoNodeIndex) {
        throw std::runtime_error(
            "runtime ended before tiled Gemm completed"
        );
    }

    result.output = std::move(activation);
    return result;
}

}  // namespace waveaccel
