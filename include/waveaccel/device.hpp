/*
 * WaveAccel - AI Accelerator Runtime Prototype
 *
 * Generic mock AI accelerator device.
 *
 * Author: Santosh Kumar
 * Copyright 2026 Santosh Kumar
 * SPDX-License-Identifier: Apache-2.0
 *
 * The mock device models two data-movement domains:
 *
 *   Host DRAM <-> Device memory
 *   Device memory <-> On-chip SRAM
 *
 * It also models abstract operator-dispatch costs.
 *
 * NOTE:
 * Bandwidth and timing values are explicit simulation parameters. They are
 * not measurements or specifications of any commercial accelerator.
 */

#pragma once

#include <cstddef>

#include "waveaccel/device_command.hpp"
#include "waveaccel/profiler.hpp"
#include "waveaccel/task.hpp"

namespace waveaccel {

struct MockDeviceConfig {
    // Explicit simulation parameters only.
    double host_device_bandwidth_bytes_per_us{12000.0};
    double device_sram_bandwidth_bytes_per_us{64000.0};

    double gemm_us{4.0};
    double tiled_gemm_dispatch_overhead_us{0.15};
    double elementwise_us{0.5};

    std::size_t sram_capacity_bytes{16U * 1024U};
};

class MockDevice {
public:
    explicit MockDevice(MockDeviceConfig config = {});

    // Legacy/reference path.
    void execute(const Task& task, Profiler& profiler);

    // Compiler/backend path: consume logical accelerator commands directly.
    void execute(
        const DeviceCommand& command,
        Profiler& profiler
    );

    // Install the logical memory bounds established by the compiler/runtime.
    // Device addresses are offsets in WaveAccel's generic model-initializer
    // address space, not physical addresses, IOVAs, or MMIO addresses.
    void configure_address_contract(
        std::size_t device_memory_bytes,
        std::size_t initializer_staging_capacity_bytes
    );

    std::size_t sram_capacity_bytes() const noexcept {
        return config_.sram_capacity_bytes;
    }

    bool model_in_device_memory() const noexcept {
        return model_in_device_memory_;
    }

    bool model_in_sram() const noexcept {
        return model_in_sram_;
    }

private:
    double host_device_transfer_time_us(
        std::size_t bytes
    ) const;

    double device_sram_transfer_time_us(
        std::size_t bytes
    ) const;

    MockDeviceConfig config_;

    // Device memory is a logical off-chip/on-package accelerator memory tier
    // in this generic model. SRAM is the constrained on-chip working store.
    bool model_in_device_memory_{false};
    bool model_in_sram_{false};

    // Zero means no compiler/runtime logical address contract is installed.
    // This preserves standalone/reference tests that exercise DeviceCommand
    // timing without an address-aware model image.
    std::size_t logical_device_memory_bytes_{0};
    std::size_t initializer_staging_capacity_bytes_{0};
};

}  // namespace waveaccel
