/*
 * WaveAccel - direct MockDevice DeviceCommand execution smoke test.
 *
 * Verifies that compiler-lowered logical accelerator commands can be consumed
 * directly by MockDevice without first converting them back into generic Task
 * objects.
 *
 * Author: Santosh Kumar
 * Copyright 2026 Santosh Kumar
 * SPDX-License-Identifier: Apache-2.0
 */

#include "waveaccel/compiled_plan_adapter.hpp"
#include "waveaccel/device.hpp"
#include "waveaccel/device_command.hpp"
#include "waveaccel/execution_plan.hpp"
#include "waveaccel/profiler.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
    try {
        const std::string path =
            argc > 1
                ? argv[1]
                : "artifacts/waveaccel_plan.manifest";

        const auto plan =
            waveaccel::load_execution_plan_manifest(path);

        const auto schedule =
            waveaccel::compile_runtime_schedule(plan);

        const auto stream =
            waveaccel::lower_to_device_commands(
                plan,
                schedule
            );

        waveaccel::MockDeviceConfig config;
        config.sram_capacity_bytes =
            plan.target.sram_bytes;

        waveaccel::MockDevice device{config};
        waveaccel::Profiler profiler;

        for (const auto& command : stream.commands) {
            device.execute(command, profiler);
        }

        if (
            profiler.events().size() !=
            stream.commands.size()
        ) {
            throw std::runtime_error(
                "not every DeviceCommand was executed/profiled"
            );
        }

        if (
            profiler.host_device_transfer_bytes() != 0
        ) {
            throw std::runtime_error(
                "internal DeviceCommand stream recorded Host/Device bytes"
            );
        }

        if (
            profiler.device_sram_transfer_bytes() != 10584
        ) {
            throw std::runtime_error(
                "DeviceCommand Device/SRAM byte accounting mismatch"
            );
        }

        const double expected_compute_us =
            config.gemm_us +
            2.0 * config.tiled_gemm_dispatch_overhead_us +
            2.0 * config.gemm_us +
            4.0 * config.elementwise_us;

        const double expected_transfer_us =
            10584.0 /
            config.device_sram_bandwidth_bytes_per_us;

        const double expected_total_us =
            expected_compute_us + expected_transfer_us;

        const double timing_error =
            std::fabs(
                profiler.total_simulated_us() -
                expected_total_us
            );

        if (timing_error > 1.0e-12) {
            throw std::runtime_error(
                "DeviceCommand simulated timing mismatch"
            );
        }

        std::cout
            << "WaveAccel direct DeviceCommand execution\n"
            << "========================================\n"
            << "commands executed:       "
            << stream.commands.size()
            << '\n'
            << "Device/SRAM bytes:       "
            << profiler.device_sram_transfer_bytes()
            << '\n'
            << "Host/Device bytes:       "
            << profiler.host_device_transfer_bytes()
            << '\n'
            << std::fixed
            << std::setprecision(6)
            << "total simulated us:      "
            << profiler.total_simulated_us()
            << '\n'
            << "\ndirect MockDevice DeviceCommand execution: PASS\n";

        return 0;
    } catch (const std::exception& error) {
        std::cerr
            << "direct MockDevice DeviceCommand execution: FAIL\n"
            << error.what()
            << '\n';
        return 1;
    }
}
