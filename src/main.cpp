/*
 * WaveAccel - AI Accelerator Runtime Prototype
 *
 * Integrated graph-driven accelerator-runtime demonstration.
 *
 * Author: Santosh Kumar
 * Copyright 2026 Santosh Kumar
 * SPDX-License-Identifier: Apache-2.0
 *
 * Usage:
 *   waveaccel_demo <memory-manifest> <weights-file> <sample-file> [sram-kib]
 *
 * Examples:
 *   ./build/waveaccel_demo \
 *       artifacts/wave_model.memory \
 *       artifacts/wave_model.weights \
 *       artifacts/wave_sample.bin \
 *       16
 *
 *   ./build/waveaccel_demo \
 *       artifacts/wave_model.memory \
 *       artifacts/wave_model.weights \
 *       artifacts/wave_sample.bin \
 *       4
 *
 * Numerical predictions are real float32 inference results. Timing and
 * accelerator bandwidth values are explicit simulations.
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "waveaccel/device.hpp"
#include "waveaccel/memory_planner.hpp"
#include "waveaccel/model_memory.hpp"
#include "waveaccel/numerical_executor.hpp"
#include "waveaccel/profiler.hpp"
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

void print_profile(
    const char* label,
    const waveaccel::Profiler& profiler
) {
    std::cout << "\n" << label << "\n";
    std::cout
        << "------------------------------------------------------------\n";

    std::cout
        << std::left
        << std::setw(32) << "task"
        << std::setw(14) << "domain"
        << std::right
        << std::setw(12) << "sim_us"
        << std::setw(12) << "bytes"
        << '\n';

    for (const auto& event : profiler.events()) {
        std::cout
            << std::left
            << std::setw(32) << event.name
            << std::setw(14)
            << waveaccel::transfer_domain_name(
                   event.transfer_domain
               )
            << std::right
            << std::setw(12)
            << std::fixed
            << std::setprecision(4)
            << event.simulated_us
            << std::setw(12)
            << event.bytes
            << '\n';
    }

    std::cout
        << "total simulated us:       "
        << profiler.total_simulated_us()
        << '\n';

    std::cout
        << "host/device bytes:        "
        << profiler.host_device_transfer_bytes()
        << '\n';

    std::cout
        << "device/SRAM bytes:        "
        << profiler.device_sram_transfer_bytes()
        << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 4 || argc > 5) {
            std::cerr
                << "usage: waveaccel_demo "
                << "<memory-manifest> "
                << "<weights-file> "
                << "<sample-file> "
                << "[sram-kib]\n";

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

        std::size_t sram_kib = 16;

        if (argc == 5) {
            sram_kib = static_cast<std::size_t>(
                std::stoull(argv[4])
            );
        }

        if (sram_kib == 0) {
            throw std::invalid_argument(
                "SRAM capacity must be positive"
            );
        }

        waveaccel::MockDeviceConfig config;
        config.sram_capacity_bytes =
            sram_kib * 1024U;

        waveaccel::Runtime runtime(
            model,
            waveaccel::MockDevice{config}
        );

        std::cout
            << "WaveAccel integrated accelerator runtime\n"
            << "Numerical prediction: REAL float32 computation\n"
            << "Timing/data movement:  SIMULATED accelerator model\n\n";

        std::cout
            << "ONNX graph nodes:          "
            << model.node_count
            << '\n';

        std::cout
            << "ONNX initializer bytes:    "
            << model.initializer_bytes
            << '\n';

        std::cout
            << "SRAM capacity:             "
            << config.sram_capacity_bytes
            << " B\n";

        std::cout
            << "memory strategy:           "
            << waveaccel::memory_strategy_name(
                   runtime.memory_plan().strategy
               )
            << '\n';

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

        std::cout << "\nNumerical inference\n";
        std::cout
            << "------------------------------------------------------------\n";

        print_vector(
            "ground truth:        ",
            sample.ground_truth
        );

        print_vector(
            "ONNX Runtime ref:    ",
            sample.reference_output
        );

        print_vector(
            "WaveAccel cold:      ",
            cold.output
        );

        print_vector(
            "WaveAccel warm:      ",
            warm.output
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

        std::cout
            << "cold parity error:   "
            << std::scientific
            << cold_error
            << '\n';

        std::cout
            << "warm parity error:   "
            << std::scientific
            << warm_error
            << '\n';

        print_profile(
            "Inference #1 (cold model state)",
            cold.profiler
        );

        print_profile(
            "Inference #2 (warm model state)",
            warm.profiler
        );

        return 0;
    } catch (const std::exception& error) {
        std::cerr
            << "waveaccel_demo error: "
            << error.what()
            << '\n';

        return 1;
    }
}
