// Author: Santosh Kumar
// Copyright 2026 Santosh Kumar
// SPDX-License-Identifier: Apache-2.0

#include "waveaccel/compiled_plan_adapter.hpp"
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

        const auto schedule =
            waveaccel::compile_runtime_schedule(plan);

        const std::vector<std::size_t> expected_tile_bytes{
            3328, 3328, 1536};
        const std::vector<std::size_t> expected_tile_outputs{
            13, 13, 6};

        std::vector<std::size_t> actual_tile_bytes;
        std::vector<std::size_t> actual_tile_outputs;
        std::size_t gemm_tile_exec_count = 0;

        std::cout
            << "WaveAccel compiler-plan runtime schedule\n"
            << "========================================\n";

        for (const auto& command : schedule.commands) {
            std::cout
                << "node " << std::setw(2) << command.node_index
                << "  " << std::setw(28)
                << waveaccel::to_string(command.kind)
                << " bytes=" << std::setw(4) << command.bytes;

            if (command.kind ==
                    waveaccel::CompiledCommandKind::StageWeightTile ||
                command.kind ==
                    waveaccel::CompiledCommandKind::ExecuteGemmTile) {
                std::cout
                    << " tile=" << command.tile_index
                    << " outputs="
                    << command.output_start
                    << ".."
                    << (command.output_start +
                        command.output_count - 1);
            }

            std::cout << '\n';

            if (command.kind ==
                waveaccel::CompiledCommandKind::StageWeightTile) {
                actual_tile_bytes.push_back(command.bytes);
                actual_tile_outputs.push_back(
                    command.output_count);
            }

            if (command.kind ==
                waveaccel::CompiledCommandKind::ExecuteGemmTile) {
                ++gemm_tile_exec_count;
            }
        }

        if (actual_tile_bytes != expected_tile_bytes) {
            throw std::runtime_error(
                "Compiler-plan tile byte sequence mismatch");
        }

        if (actual_tile_outputs != expected_tile_outputs) {
            throw std::runtime_error(
                "Compiler-plan tile output sequence mismatch");
        }

        if (gemm_tile_exec_count != 3) {
            throw std::runtime_error(
                "Expected three GEMM tile executions");
        }

        if (schedule.device_to_sram_bytes != 10584) {
            throw std::runtime_error(
                "Expected exactly 10584 Device->SRAM bytes; got " +
                std::to_string(schedule.device_to_sram_bytes));
        }

        std::cout
            << "\ndevice/SRAM bytes from compiler plan: "
            << schedule.device_to_sram_bytes
            << "\ncompiler-plan runtime schedule: PASS\n";

        return 0;
    } catch (const std::exception& error) {
        std::cerr
            << "compiler-plan runtime schedule: FAIL\n"
            << error.what() << '\n';
        return 1;
    }
}
