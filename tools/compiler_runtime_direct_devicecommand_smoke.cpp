/*
 * WaveAccel - compiler Runtime direct DeviceCommand execution smoke test.
 *
 * Verifies the integrated numerical Runtime path no longer routes compiler
 * staging/compute operations through generic Task objects.
 *
 * Host<->Device transfers remain Runtime-envelope Task operations.
 * Accelerator-internal operations must appear as DEVICE_CMD_* profiler events.
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

bool starts_with(
    const std::string& value,
    const std::string& prefix
) {
    return
        value.size() >= prefix.size() &&
        value.compare(
            0,
            prefix.size(),
            prefix
        ) == 0;
}

std::size_t direct_command_event_count(
    const waveaccel::Profiler& profiler
) {
    std::size_t count = 0;

    for (const auto& event : profiler.events()) {
        if (
            starts_with(
                event.name,
                "DEVICE_CMD_NODE_"
            )
        ) {
            ++count;
        }
    }

    return count;
}

bool has_legacy_internal_task_event(
    const waveaccel::Profiler& profiler
) {
    for (const auto& event : profiler.events()) {
        if (
            starts_with(event.name, "NODE_") ||
            starts_with(event.name, "STAGE_")
        ) {
            return true;
        }
    }

    return false;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 5) {
            std::cerr
                << "usage: compiler_runtime_direct_devicecommand_smoke "
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

        waveaccel::Runtime runtime(
            model,
            waveaccel::MockDevice{config}
        );

        runtime.set_execution_plan(
            std::move(execution_plan)
        );

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

        const auto cold_direct_events =
            direct_command_event_count(
                cold.profiler
            );

        const auto warm_direct_events =
            direct_command_event_count(
                warm.profiler
            );

        if (
            cold_direct_events != 17 ||
            warm_direct_events != 17
        ) {
            throw std::runtime_error(
                "integrated Runtime did not execute all 17 "
                "accelerator-internal DeviceCommands directly"
            );
        }

        if (
            has_legacy_internal_task_event(
                cold.profiler
            ) ||
            has_legacy_internal_task_event(
                warm.profiler
            )
        ) {
            throw std::runtime_error(
                "compiler-driven Runtime still emitted legacy internal "
                "Task events"
            );
        }

        if (
            cold.profiler.device_sram_transfer_bytes() !=
                10584 ||
            warm.profiler.device_sram_transfer_bytes() !=
                10584
        ) {
            throw std::runtime_error(
                "compiler-driven direct Runtime Device/SRAM bytes mismatch"
            );
        }

        if (
            cold.profiler.host_device_transfer_bytes() !=
                10848 ||
            warm.profiler.host_device_transfer_bytes() !=
                264
        ) {
            throw std::runtime_error(
                "compiler-driven direct Runtime Host/Device bytes mismatch"
            );
        }

        if (
            cold_error > 1.0e-5F ||
            warm_error > 1.0e-5F
        ) {
            throw std::runtime_error(
                "compiler-driven direct Runtime numerical parity failed"
            );
        }

        if (cold_warm_error != 0.0F) {
            throw std::runtime_error(
                "compiler-driven direct Runtime cold/warm outputs differ"
            );
        }

        std::cout
            << "WaveAccel compiler Runtime direct backend\n"
            << "========================================\n"
            << "direct DeviceCommand events cold: "
            << cold_direct_events
            << '\n'
            << "direct DeviceCommand events warm: "
            << warm_direct_events
            << '\n'
            << "cold Device/SRAM bytes: "
            << cold.profiler.device_sram_transfer_bytes()
            << '\n'
            << "warm Device/SRAM bytes: "
            << warm.profiler.device_sram_transfer_bytes()
            << '\n'
            << "cold Host/Device bytes: "
            << cold.profiler.host_device_transfer_bytes()
            << '\n'
            << "warm Host/Device bytes: "
            << warm.profiler.host_device_transfer_bytes()
            << '\n'
            << std::scientific
            << std::setprecision(9)
            << "cold ONNX parity error: "
            << cold_error
            << '\n'
            << "warm ONNX parity error: "
            << warm_error
            << '\n'
            << "cold/warm error: "
            << cold_warm_error
            << '\n'
            << "\ncompiler Runtime -> DeviceCommandStream -> "
               "MockDevice: PASS\n";

        return 0;
    } catch (const std::exception& error) {
        std::cerr
            << "compiler Runtime direct DeviceCommand execution: FAIL\n"
            << error.what()
            << '\n';
        return 1;
    }
}
