/*
 * WaveAccel - compiler plan vs manual planner parity smoke test.
 *
 * This test keeps the manually validated MemoryPlanner as an independent
 * reference and checks that the compiler-driven Runtime emits the same Task
 * schedule for the current 4 KiB target.
 *
 * It also runs real float32 inference through both plans and verifies that the
 * numerical outputs are identical.
 *
 * Author: Santosh Kumar
 * Copyright 2026 Santosh Kumar
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "waveaccel/device.hpp"
#include "waveaccel/execution_plan.hpp"
#include "waveaccel/model_memory.hpp"
#include "waveaccel/numerical_executor.hpp"
#include "waveaccel/runtime.hpp"
#include "waveaccel/task.hpp"
#include "waveaccel/tensor_store.hpp"

namespace {

const char* task_kind_name(
    waveaccel::TaskKind kind
) {
    using waveaccel::TaskKind;

    switch (kind) {
        case TaskKind::H2D:
            return "H2D";
        case TaskKind::LoadModelToDevice:
            return "LoadModelToDevice";
        case TaskKind::StageToSram:
            return "StageToSram";
        case TaskKind::StageModelToSram:
            return "StageModelToSram";
        case TaskKind::Gemm:
            return "Gemm";
        case TaskKind::GemmTile:
            return "GemmTile";
        case TaskKind::Relu:
            return "Relu";
        case TaskKind::Mul:
            return "Mul";
        case TaskKind::Add:
            return "Add";
        case TaskKind::D2H:
            return "D2H";
    }

    return "Unknown";
}

bool same_task(
    const waveaccel::Task& lhs,
    const waveaccel::Task& rhs
) {
    return
        lhs.kind == rhs.kind &&
        lhs.name == rhs.name &&
        lhs.bytes == rhs.bytes &&
        lhs.node_index == rhs.node_index &&
        lhs.tile_index == rhs.tile_index &&
        lhs.tile_count == rhs.tile_count &&
        lhs.row_begin == rhs.row_begin &&
        lhs.row_count == rhs.row_count &&
        lhs.total_rows == rhs.total_rows;
}

void print_task(
    const char* label,
    std::size_t index,
    const waveaccel::Task& task
) {
    std::cout
        << label
        << "[" << index << "] "
        << task_kind_name(task.kind)
        << " name=" << task.name
        << " bytes=" << task.bytes
        << " node=" << task.node_index
        << " tile=" << task.tile_index
        << " tile_count=" << task.tile_count
        << " row_begin=" << task.row_begin
        << " row_count=" << task.row_count
        << " total_rows=" << task.total_rows
        << '\n';
}

float max_abs_error(
    const std::vector<float>& lhs,
    const std::vector<float>& rhs
) {
    if (lhs.size() != rhs.size()) {
        throw std::runtime_error(
            "cannot compare outputs with different sizes"
        );
    }

    float error = 0.0F;

    for (
        std::size_t index = 0;
        index < lhs.size();
        ++index
    ) {
        const float current =
            std::fabs(lhs[index] - rhs[index]);

        if (current > error) {
            error = current;
        }
    }

    return error;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 5) {
            std::cerr
                << "usage: compiler_manual_plan_parity "
                << "<memory-manifest> "
                << "<weights-file> "
                << "<sample-file> "
                << "<execution-plan-manifest>\n";
            return 2;
        }

        const auto model =
            waveaccel::load_model_memory_manifest(
                argv[1]
            );

        const auto tensors =
            waveaccel::TensorStore::load(
                argv[2]
            );

        const auto sample =
            waveaccel::load_numerical_sample(
                argv[3]
            );

        auto execution_plan =
            waveaccel::load_execution_plan_manifest(
                argv[4]
            );

        waveaccel::MockDeviceConfig config;
        config.sram_capacity_bytes =
            execution_plan.target.sram_bytes;

        // Independent reference: existing C++ MemoryPlanner path.
        waveaccel::Runtime manual_runtime(
            model,
            waveaccel::MockDevice{config}
        );

        // Compiler authority path: TVM/WaveAccel execution plan.
        waveaccel::Runtime compiler_runtime(
            model,
            waveaccel::MockDevice{config}
        );

        compiler_runtime.set_execution_plan(
            std::move(execution_plan)
        );

        const auto manual_plan =
            manual_runtime.build_plan();

        const auto compiler_plan =
            compiler_runtime.build_plan();

        std::cout
            << "WaveAccel manual/compiler plan parity\n"
            << "=====================================\n"
            << "manual task count:   "
            << manual_plan.size()
            << '\n'
            << "compiler task count: "
            << compiler_plan.size()
            << '\n';

        if (manual_plan.size() != compiler_plan.size()) {
            std::cout
                << "\nTask-count mismatch. "
                << "Printing both plans for diagnosis.\n";

            const std::size_t max_count =
                manual_plan.size() > compiler_plan.size()
                    ? manual_plan.size()
                    : compiler_plan.size();

            for (
                std::size_t index = 0;
                index < max_count;
                ++index
            ) {
                if (index < manual_plan.size()) {
                    print_task(
                        "manual  ",
                        index,
                        manual_plan[index]
                    );
                }

                if (index < compiler_plan.size()) {
                    print_task(
                        "compiler",
                        index,
                        compiler_plan[index]
                    );
                }
            }

            throw std::runtime_error(
                "manual/compiler task counts differ"
            );
        }

        for (
            std::size_t index = 0;
            index < manual_plan.size();
            ++index
        ) {
            if (!same_task(
                    manual_plan[index],
                    compiler_plan[index])) {
                std::cout
                    << "\nFirst task mismatch at index "
                    << index
                    << '\n';

                print_task(
                    "manual  ",
                    index,
                    manual_plan[index]
                );

                print_task(
                    "compiler",
                    index,
                    compiler_plan[index]
                );

                throw std::runtime_error(
                    "manual/compiler runtime plans differ"
                );
            }
        }

        std::cout
            << "Task schedule parity: PASS\n";

        const auto manual_result =
            manual_runtime.run_once(
                sample.input,
                tensors
            );

        const auto compiler_result =
            compiler_runtime.run_once(
                sample.input,
                tensors
            );

        const float output_error =
            max_abs_error(
                manual_result.output,
                compiler_result.output
            );

        std::cout
            << std::scientific
            << std::setprecision(9)
            << "manual/compiler output error: "
            << output_error
            << '\n';

        if (output_error != 0.0F) {
            throw std::runtime_error(
                "manual/compiler numerical outputs differ"
            );
        }

        if (
            manual_result.profiler.device_sram_transfer_bytes() !=
            compiler_result.profiler.device_sram_transfer_bytes()
        ) {
            throw std::runtime_error(
                "manual/compiler Device->SRAM accounting differs"
            );
        }

        std::cout
            << std::fixed
            << std::setprecision(0)
            << "Device/SRAM bytes: "
            << compiler_result.profiler.device_sram_transfer_bytes()
            << '\n';

        std::cout
            << "\nmanual MemoryPlanner == compiler plan: PASS\n";

        return 0;
    } catch (const std::exception& error) {
        std::cerr
            << "\nmanual MemoryPlanner == compiler plan: FAIL\n"
            << error.what()
            << '\n';
        return 1;
    }
}
