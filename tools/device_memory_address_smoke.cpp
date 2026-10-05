/*
 * WaveAccel - AI Accelerator Runtime Prototype
 *
 * Validate logical device-memory addresses against real ONNX initializer
 * metadata and the real trained tensor payload file.
 *
 * Author: Santosh Kumar
 * Copyright 2026 Santosh Kumar
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstddef>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include "waveaccel/compiled_plan_adapter.hpp"
#include "waveaccel/device_command.hpp"
#include "waveaccel/device_memory.hpp"
#include "waveaccel/execution_plan.hpp"
#include "waveaccel/model_memory.hpp"
#include "waveaccel/tensor_store.hpp"

namespace {

bool is_dma(
    const waveaccel::DeviceCommand& command
) {
    return command.kind ==
        waveaccel::DeviceCommandKind::DmaDeviceToSram;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 4) {
            std::cerr
                << "usage: device_memory_address_smoke "
                << "<memory-manifest> <weights-file> "
                << "<execution-plan-manifest>\n";
            return 2;
        }

        const auto model =
            waveaccel::load_model_memory_manifest(argv[1]);

        const auto tensors =
            waveaccel::TensorStore::load(argv[2]);

        const auto plan =
            waveaccel::load_execution_plan_manifest(argv[3]);

        const auto map =
            waveaccel::build_device_memory_map(model);

        if (map.total_bytes != model.initializer_bytes) {
            throw std::runtime_error(
                "logical device-memory byte total mismatch"
            );
        }

        std::size_t expected_offset = 0;

        std::cout
            << "WaveAccel logical device-memory map\n"
            << "===================================\n";

        for (const auto& region : map.regions) {
            if (region.offset_bytes != expected_offset) {
                throw std::runtime_error(
                    "logical device-memory regions are not contiguous"
                );
            }

            const auto& tensor =
                tensors.at(region.tensor_name);

            const std::size_t tensor_bytes =
                tensor.values.size() * sizeof(float);

            if (tensor_bytes != region.bytes) {
                throw std::runtime_error(
                    "logical device-memory bytes do not match tensor payload: " +
                    region.tensor_name
                );
            }

            std::cout
                << "offset "
                << std::setw(5)
                << region.offset_bytes
                << "  bytes "
                << std::setw(5)
                << region.bytes
                << "  "
                << region.tensor_name
                << '\n';

            expected_offset += region.bytes;
        }

        if (expected_offset != map.total_bytes) {
            throw std::runtime_error(
                "logical device-memory final offset mismatch"
            );
        }

        const auto schedule =
            waveaccel::compile_runtime_schedule(plan);

        const auto stream =
            waveaccel::lower_to_device_commands(
                plan,
                schedule,
                model
            );

        std::size_t dma_count = 0;
        bool verified_tiled_weight_address = false;
        bool verified_tiled_bias_address = false;

        std::cout
            << "\nAddress-aware DeviceCommand stream\n"
            << "==================================\n";

        for (const auto& command : stream.commands) {
            if (!is_dma(command)) {
                continue;
            }

            ++dma_count;

            if (
                command.device_offset_bytes > map.total_bytes ||
                command.bytes >
                    map.total_bytes - command.device_offset_bytes
            ) {
                throw std::runtime_error(
                    "DeviceCommand DMA leaves logical model memory"
                );
            }

            const auto& graph_node =
                model.nodes.at(command.node_index);

            const auto& plan_node =
                plan.nodes.at(command.node_index);

            if (
                plan_node.op_type == "Gemm" &&
                plan_node.strategy == "TILED"
            ) {
                if (graph_node.initializers.size() != 2) {
                    throw std::runtime_error(
                        "tiled Gemm does not expose weight+bias initializers"
                    );
                }

                const auto& weight =
                    waveaccel::find_device_memory_region(
                        map,
                        graph_node.initializers[0]
                    );

                const auto& bias =
                    waveaccel::find_device_memory_region(
                        map,
                        graph_node.initializers[1]
                    );

                if (command.source_region == bias.tensor_name) {
                    if (
                        command.device_offset_bytes != bias.offset_bytes ||
                        command.source_offset_bytes != 0 ||
                        command.sram_offset_bytes != 0 ||
                        command.bytes != bias.bytes
                    ) {
                        throw std::runtime_error(
                            "tiled Gemm bias DMA address mismatch"
                        );
                    }
                    verified_tiled_bias_address = true;
                }

                if (command.source_region == weight.tensor_name) {
                    if (
                        command.device_offset_bytes !=
                            weight.offset_bytes +
                            command.source_offset_bytes ||
                        command.sram_offset_bytes !=
                            plan_node.bias_bytes
                    ) {
                        throw std::runtime_error(
                            "tiled Gemm weight DMA address mismatch"
                        );
                    }
                    verified_tiled_weight_address = true;
                }
            }

            std::cout
                << "DMA node "
                << command.node_index
                << "  dev_off="
                << command.device_offset_bytes
                << "  sram_off="
                << command.sram_offset_bytes
                << "  bytes="
                << command.bytes
                << "  source="
                << command.source_region
                << '\n';
        }

        if (!verified_tiled_bias_address) {
            throw std::runtime_error(
                "did not verify tiled Gemm bias address"
            );
        }

        if (!verified_tiled_weight_address) {
            throw std::runtime_error(
                "did not verify tiled Gemm weight address"
            );
        }

        if (
            stream.device_to_sram_bytes !=
            schedule.device_to_sram_bytes
        ) {
            throw std::runtime_error(
                "Device/SRAM byte accounting changed"
            );
        }

        std::cout
            << "\nmodel device bytes: "
            << map.total_bytes
            << '\n'
            << "DMA commands:       "
            << dma_count
            << '\n'
            << "Device/SRAM bytes:  "
            << stream.device_to_sram_bytes
            << '\n'
            << "logical device-memory addressing: PASS\n";

        return 0;
    } catch (const std::exception& error) {
        std::cerr
            << "device_memory_address_smoke: FAIL: "
            << error.what()
            << '\n';
        return 1;
    }
}
