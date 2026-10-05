/*
 * WaveAccel - compiler-driven runtime integration smoke.
 *
 * Proves that:
 *   TVM/WaveAccel compiler plan
 *      -> C++ ExecutionPlan
 *      -> CompiledRuntimeSchedule
 *      -> Runtime::build_plan()
 *      -> existing MockDevice + NumericalExecutor
 *
 * Numerical inference remains real float32 computation.
 * Timing/data movement remain simulated accelerator behavior.
 *
 * Author: Santosh Kumar
 * Copyright 2026 Santosh Kumar
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
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

float max_abs_error(
    const std::vector<float>& lhs,
    const std::vector<float>& rhs
) {
    if (lhs.size() != rhs.size()) {
        throw std::runtime_error(
            "cannot compare vectors with different sizes"
        );
    }

    float error = 0.0F;

    for (
        std::size_t index = 0;
        index < lhs.size();
        ++index
    ) {
        error = std::max(
            error,
            std::fabs(lhs[index] - rhs[index])
        );
    }

    return error;
}

void print_vector(
    const char* label,
    const std::vector<float>& values
) {
    std::cout << label << " [";

    for (
        std::size_t index = 0;
        index < values.size();
        ++index
    ) {
        if (index != 0) {
            std::cout << ", ";
        }

        std::cout
            << std::fixed
            << std::setprecision(9)
            << values[index];
    }

    std::cout << "]\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 5) {
            std::cerr
                << "usage: compiler_driven_runtime_smoke "
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

        auto compiler_plan =
            waveaccel::load_execution_plan_manifest(
                argv[4]
            );

        waveaccel::MockDeviceConfig config;
        config.sram_capacity_bytes =
            compiler_plan.target.sram_bytes;

        waveaccel::Runtime runtime(
            model,
            waveaccel::MockDevice{config}
        );

        runtime.set_execution_plan(
            std::move(compiler_plan)
        );

        if (!runtime.compiler_plan_enabled()) {
            throw std::runtime_error(
                "compiler execution plan was not enabled"
            );
        }

        const auto plan = runtime.build_plan();

        std::vector<std::size_t> tile_bytes;
        std::vector<std::size_t> tile_rows;

        for (const auto& task : plan) {
            if (
                task.kind == waveaccel::TaskKind::StageToSram &&
                task.name.find("NODE_0_STAGE_WEIGHT_TILE_") == 0
            ) {
                tile_bytes.push_back(task.bytes);
            }

            if (
                task.kind ==
                waveaccel::TaskKind::GemmTile
            ) {
                tile_rows.push_back(task.row_count);
            }
        }

        const std::vector<std::size_t>
            expected_tile_bytes{
                3328, 3328, 1536
            };

        const std::vector<std::size_t>
            expected_tile_rows{
                13, 13, 6
            };

        if (tile_bytes != expected_tile_bytes) {
            throw std::runtime_error(
                "Runtime Task plan does not preserve compiler tile bytes"
            );
        }

        if (tile_rows != expected_tile_rows) {
            throw std::runtime_error(
                "Runtime Task plan does not preserve compiler tile rows"
            );
        }

        const auto cold =
            runtime.run_once(
                sample.input,
                tensors
            );

        const auto warm =
            runtime.run_once(
                sample.input,
                tensors
            );

        const float cold_error =
            max_abs_error(
                cold.output,
                sample.reference_output
            );

        const float warm_error =
            max_abs_error(
                warm.output,
                sample.reference_output
            );

        const float cold_warm_error =
            max_abs_error(
                cold.output,
                warm.output
            );

        print_vector(
            "ONNX Runtime ref:   ",
            sample.reference_output
        );

        print_vector(
            "WaveAccel compiled: ",
            cold.output
        );

        std::cout
            << std::scientific
            << std::setprecision(9)
            << "cold parity error:  "
            << cold_error
            << '\n'
            << "warm parity error:  "
            << warm_error
            << '\n'
            << "cold/warm error:    "
            << cold_warm_error
            << '\n';

        std::cout
            << std::fixed
            << std::setprecision(0)
            << "cold device/SRAM bytes: "
            << cold.profiler.device_sram_transfer_bytes()
            << '\n'
            << "warm device/SRAM bytes: "
            << warm.profiler.device_sram_transfer_bytes()
            << '\n';

        if (cold_error > 1.0e-5F ||
            warm_error > 1.0e-5F) {
            throw std::runtime_error(
                "compiler-driven runtime numerical parity failed"
            );
        }

        if (cold_warm_error != 0.0F) {
            throw std::runtime_error(
                "compiler-driven cold/warm outputs differ"
            );
        }

        if (
            cold.profiler.device_sram_transfer_bytes() !=
                10584 ||
            warm.profiler.device_sram_transfer_bytes() !=
                10584
        ) {
            throw std::runtime_error(
                "compiler-driven Device/SRAM byte accounting mismatch"
            );
        }

        std::cout
            << "\ncompiler -> C++ runtime -> numerical execution: PASS\n";

        return 0;
    } catch (const std::exception& error) {
        std::cerr
            << "compiler-driven runtime integration: FAIL\n"
            << error.what()
            << '\n';
        return 1;
    }
}
