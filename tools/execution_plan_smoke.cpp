// Author: Santosh Kumar
// Copyright 2026 Santosh Kumar
// SPDX-License-Identifier: Apache-2.0

#include "waveaccel/execution_plan.hpp"

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

        if (plan.target.sram_bytes != 4096) {
            throw std::runtime_error(
                "Expected 4096-byte SRAM target");
        }

        if (plan.nodes.size() != 7) {
            throw std::runtime_error(
                "Expected 7 graph nodes");
        }

        const auto& first = plan.nodes.at(0);

        if (first.op_type != "Gemm" ||
            first.strategy != "TILED" ||
            first.compiler_tile_width != 13) {
            throw std::runtime_error(
                "First Gemm compiler policy mismatch");
        }

        const std::vector<std::size_t> expected_counts{
            13, 13, 6};
        const std::vector<std::size_t> expected_bytes{
            3328, 3328, 1536};

        if (first.tiles.size() != expected_counts.size()) {
            throw std::runtime_error(
                "First Gemm tile count mismatch");
        }

        for (std::size_t i = 0; i < first.tiles.size(); ++i) {
            if (first.tiles[i].output_count != expected_counts[i] ||
                first.tiles[i].weight_bytes != expected_bytes[i]) {
                throw std::runtime_error(
                    "First Gemm tile geometry mismatch");
            }
        }

        if (plan.nodes.at(2).strategy != "STAGED_WHOLE" ||
            plan.nodes.at(4).strategy != "STAGED_WHOLE") {
            throw std::runtime_error(
                "Expected Gemm1/Gemm2 STAGED_WHOLE");
        }

        std::cout
            << "WaveAccel compiler-plan C++ parser\n"
            << "==================================\n"
            << "schema version:          "
            << plan.schema_version << '\n'
            << "TVM version:             "
            << plan.tvm_version << '\n'
            << "SRAM bytes:              "
            << plan.target.sram_bytes << '\n'
            << "graph nodes:             "
            << plan.nodes.size() << '\n'
            << "first Gemm strategy:     "
            << first.strategy << '\n'
            << "compiler tile width:     "
            << first.compiler_tile_width << '\n'
            << "logical tile counts:     ["
            << first.tiles[0].output_count << ", "
            << first.tiles[1].output_count << ", "
            << first.tiles[2].output_count << "]\n"
            << "logical weight bytes:    ["
            << first.tiles[0].weight_bytes << ", "
            << first.tiles[1].weight_bytes << ", "
            << first.tiles[2].weight_bytes << "]\n"
            << "\nC++ execution-plan parsing: PASS\n";

        return 0;
    } catch (const std::exception& error) {
        std::cerr
            << "C++ execution-plan parsing: FAIL\n"
            << error.what() << '\n';
        return 1;
    }
}
