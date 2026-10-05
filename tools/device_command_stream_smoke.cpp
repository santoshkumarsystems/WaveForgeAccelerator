// Author: Santosh Kumar
// Copyright 2026 Santosh Kumar
// SPDX-License-Identifier: Apache-2.0

#include "waveaccel/compiled_plan_adapter.hpp"
#include "waveaccel/device_command.hpp"
#include "waveaccel/execution_plan.hpp"

#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    try {
        const std::string path =
            argc > 1
                ? argv[1]
                : "artifacts/waveaccel_plan.manifest";

        const auto plan =
            waveaccel::load_execution_plan_manifest(path);

        const auto runtime_schedule =
            waveaccel::compile_runtime_schedule(plan);

        const auto stream =
            waveaccel::lower_to_device_commands(
                plan,
                runtime_schedule
            );

        std::vector<std::size_t> node0_weight_bytes;
        std::vector<std::size_t> node0_weight_source_offsets;
        std::vector<std::size_t> node0_weight_sram_offsets;
        std::vector<std::size_t> tile_output_counts;

        std::cout
            << "WaveAccel logical device command stream\n"
            << "=======================================\n";

        for (const auto& command : stream.commands) {
            std::cout
                << std::left
                << std::setw(22)
                << waveaccel::to_string(command.kind)
                << " node=" << command.node_index;

            if (
                command.kind ==
                waveaccel::DeviceCommandKind::DmaDeviceToSram
            ) {
                std::cout
                    << " src=" << command.source_region
                    << " src_off=" << command.source_offset_bytes
                    << " sram_off=" << command.sram_offset_bytes
                    << " bytes=" << command.bytes;

                if (
                    command.node_index == 0 &&
                    command.source_region == "node.0.weight"
                ) {
                    node0_weight_bytes.push_back(
                        command.bytes
                    );

                    node0_weight_source_offsets.push_back(
                        command.source_offset_bytes
                    );

                    node0_weight_sram_offsets.push_back(
                        command.sram_offset_bytes
                    );
                }
            }

            if (
                command.kind ==
                waveaccel::DeviceCommandKind::GemmTile
            ) {
                std::cout
                    << " tile=" << command.tile_index
                    << " outputs="
                    << command.output_start
                    << ".."
                    << (
                        command.output_start +
                        command.output_count -
                        1
                    );

                tile_output_counts.push_back(
                    command.output_count
                );
            }

            std::cout << '\n';
        }

        const std::vector<std::size_t>
            expected_weight_bytes{
                3328, 3328, 1536
            };

        const std::vector<std::size_t>
            expected_source_offsets{
                0, 3328, 6656
            };

        const std::vector<std::size_t>
            expected_sram_offsets{
                128, 128, 128
            };

        const std::vector<std::size_t>
            expected_output_counts{
                13, 13, 6
            };

        if (node0_weight_bytes != expected_weight_bytes) {
            throw std::runtime_error(
                "weight DMA byte sequence mismatch"
            );
        }

        if (
            node0_weight_source_offsets !=
            expected_source_offsets
        ) {
            throw std::runtime_error(
                "weight DMA source-offset sequence mismatch"
            );
        }

        if (
            node0_weight_sram_offsets !=
            expected_sram_offsets
        ) {
            throw std::runtime_error(
                "weight DMA SRAM-offset sequence mismatch"
            );
        }

        if (tile_output_counts != expected_output_counts) {
            throw std::runtime_error(
                "GemmTile output-count sequence mismatch"
            );
        }

        if (stream.device_to_sram_bytes != 10584) {
            throw std::runtime_error(
                "expected 10584 total Device->SRAM bytes"
            );
        }

        if (
            stream.max_sram_initializer_end_bytes >
            plan.target.initializer_staging_capacity_bytes
        ) {
            throw std::runtime_error(
                "logical device command stream exceeds SRAM staging capacity"
            );
        }

        std::cout
            << "\nDevice->SRAM bytes:       "
            << stream.device_to_sram_bytes
            << '\n'
            << "Max initializer SRAM end: "
            << stream.max_sram_initializer_end_bytes
            << " / "
            << plan.target.initializer_staging_capacity_bytes
            << " B\n"
            << "\nlogical DMA/SRAM command lowering: PASS\n";

        return 0;
    } catch (const std::exception& error) {
        std::cerr
            << "logical DMA/SRAM command lowering: FAIL\n"
            << error.what()
            << '\n';
        return 1;
    }
}
