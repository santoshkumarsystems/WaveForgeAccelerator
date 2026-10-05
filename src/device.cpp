/*
 * WaveAccel - AI Accelerator Runtime Prototype
 *
 * Generic mock AI accelerator execution model.
 *
 * Author: Santosh Kumar
 * Copyright 2026 Santosh Kumar
 * SPDX-License-Identifier: Apache-2.0
 */

#include "waveaccel/device.hpp"

#include <stdexcept>
#include <string>
#include <utility>

namespace waveaccel {

MockDevice::MockDevice(MockDeviceConfig config)
    : config_(config) {
    if (
        config_.host_device_bandwidth_bytes_per_us <= 0.0
    ) {
        throw std::invalid_argument(
            "host/device bandwidth must be positive"
        );
    }

    if (
        config_.device_sram_bandwidth_bytes_per_us <= 0.0
    ) {
        throw std::invalid_argument(
            "device/SRAM bandwidth must be positive"
        );
    }

    if (config_.gemm_us < 0.0 ||
        config_.tiled_gemm_dispatch_overhead_us < 0.0 ||
        config_.elementwise_us < 0.0) {
        throw std::invalid_argument(
            "simulation compute timings must be non-negative"
        );
    }

    if (config_.sram_capacity_bytes == 0) {
        throw std::invalid_argument(
            "SRAM capacity must be positive"
        );
    }
}

double MockDevice::host_device_transfer_time_us(
    std::size_t bytes
) const {
    return static_cast<double>(bytes) /
           config_.host_device_bandwidth_bytes_per_us;
}

double MockDevice::device_sram_transfer_time_us(
    std::size_t bytes
) const {
    return static_cast<double>(bytes) /
           config_.device_sram_bandwidth_bytes_per_us;
}

void MockDevice::configure_address_contract(
    std::size_t device_memory_bytes,
    std::size_t initializer_staging_capacity_bytes
) {
    if (device_memory_bytes == 0) {
        throw std::invalid_argument(
            "logical device-memory capacity must be positive"
        );
    }

    if (
        initializer_staging_capacity_bytes == 0 ||
        initializer_staging_capacity_bytes >
            config_.sram_capacity_bytes
    ) {
        throw std::invalid_argument(
            "initializer staging capacity must be within MockDevice SRAM"
        );
    }

    logical_device_memory_bytes_ =
        device_memory_bytes;

    initializer_staging_capacity_bytes_ =
        initializer_staging_capacity_bytes;
}

void MockDevice::execute(
    const Task& task,
    Profiler& profiler
) {
    double simulated_us = 0.0;
    std::size_t transfer_bytes = 0;
    TransferDomain transfer_domain =
        TransferDomain::None;

    switch (task.kind) {
        case TaskKind::H2D:
        case TaskKind::D2H:
        case TaskKind::LoadModelToDevice:
            simulated_us =
                host_device_transfer_time_us(task.bytes);

            transfer_bytes = task.bytes;
            transfer_domain =
                TransferDomain::HostDevice;
            break;

        case TaskKind::StageToSram:
        case TaskKind::StageModelToSram:
            simulated_us =
                device_sram_transfer_time_us(task.bytes);

            transfer_bytes = task.bytes;
            transfer_domain =
                TransferDomain::DeviceSram;
            break;

        case TaskKind::Gemm:
            simulated_us = config_.gemm_us;
            break;

        case TaskKind::GemmTile:
            if (
                task.total_rows == 0 ||
                task.row_count == 0 ||
                task.tile_count == 0
            ) {
                throw std::runtime_error(
                    "invalid GemmTile scheduling metadata"
                );
            }

            simulated_us =
                config_.gemm_us *
                (
                    static_cast<double>(task.row_count) /
                    static_cast<double>(task.total_rows)
                );

            if (task.tile_index > 0) {
                simulated_us +=
                    config_.tiled_gemm_dispatch_overhead_us;
            }

            break;

        case TaskKind::Relu:
        case TaskKind::Mul:
        case TaskKind::Add:
            simulated_us = config_.elementwise_us;
            break;
    }

    profiler.record(
        ProfileEvent{
            task.name,
            simulated_us,
            transfer_bytes,
            transfer_domain,
        }
    );

    if (task.kind == TaskKind::LoadModelToDevice) {
        model_in_device_memory_ = true;
    }

    if (task.kind == TaskKind::StageModelToSram) {
        model_in_sram_ = true;
    }
}

void MockDevice::execute(
    const DeviceCommand& command,
    Profiler& profiler
) {
    double simulated_us = 0.0;
    std::size_t transfer_bytes = 0;
    TransferDomain transfer_domain =
        TransferDomain::None;

    switch (command.kind) {
        case DeviceCommandKind::DmaDeviceToSram: {
            if (
                command.sram_offset_bytes >
                    config_.sram_capacity_bytes ||
                command.bytes >
                    config_.sram_capacity_bytes -
                        command.sram_offset_bytes
            ) {
                throw std::runtime_error(
                    "DeviceCommand DMA exceeds MockDevice SRAM capacity"
                );
            }

            if (logical_device_memory_bytes_ != 0) {
                if (
                    command.device_offset_bytes >
                        logical_device_memory_bytes_ ||
                    command.bytes >
                        logical_device_memory_bytes_ -
                            command.device_offset_bytes
                ) {
                    throw std::runtime_error(
                        "DeviceCommand DMA exceeds installed logical "
                        "device-memory contract"
                    );
                }
            }

            if (
                initializer_staging_capacity_bytes_ != 0 &&
                (
                    command.sram_offset_bytes >
                        initializer_staging_capacity_bytes_ ||
                    command.bytes >
                        initializer_staging_capacity_bytes_ -
                            command.sram_offset_bytes
                )
            ) {
                throw std::runtime_error(
                    "DeviceCommand DMA exceeds installed initializer "
                    "staging contract"
                );
            }

            simulated_us =
                device_sram_transfer_time_us(command.bytes);

            transfer_bytes = command.bytes;
            transfer_domain =
                TransferDomain::DeviceSram;
            break;
        }

        case DeviceCommandKind::GemmTile:
            if (
                command.output_count == 0 ||
                command.total_output_count == 0 ||
                command.output_start + command.output_count >
                    command.total_output_count
            ) {
                throw std::runtime_error(
                    "invalid DeviceCommand GemmTile metadata"
                );
            }

            simulated_us =
                config_.gemm_us *
                (
                    static_cast<double>(command.output_count) /
                    static_cast<double>(
                        command.total_output_count
                    )
                );

            if (command.tile_index > 0) {
                simulated_us +=
                    config_.tiled_gemm_dispatch_overhead_us;
            }

            break;

        case DeviceCommandKind::Gemm:
            simulated_us = config_.gemm_us;
            break;

        case DeviceCommandKind::Relu:
        case DeviceCommandKind::Mul:
        case DeviceCommandKind::Add:
            simulated_us = config_.elementwise_us;
            break;
    }

    std::string event_name =
        "DEVICE_CMD_NODE_" +
        std::to_string(command.node_index) +
        "_" +
        to_string(command.kind);

    if (
        command.kind == DeviceCommandKind::GemmTile
    ) {
        event_name +=
            "_" + std::to_string(command.tile_index);
    }

    profiler.record(
        ProfileEvent{
            std::move(event_name),
            simulated_us,
            transfer_bytes,
            transfer_domain,
        }
    );
}

}  // namespace waveaccel
