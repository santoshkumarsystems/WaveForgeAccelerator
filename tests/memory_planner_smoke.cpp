/*
 * WaveAccel - AI Accelerator Runtime Prototype
 *
 * Graph-aware SRAM memory-planning smoke test with row-aligned Gemm tiles.
 *
 * Author: Santosh Kumar
 * Copyright 2026 Santosh Kumar
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cassert>
#include <iostream>

#include "waveaccel/memory_planner.hpp"
#include "waveaccel/model_memory.hpp"

namespace {

void print_plan(
    const char* label,
    const waveaccel::MemoryPlan& plan
) {
    std::cout << "\n" << label << '\n';

    std::cout
        << "  overall strategy: "
        << waveaccel::memory_strategy_name(plan.strategy)
        << '\n';

    std::cout
        << "  SRAM capacity: "
        << plan.sram_capacity_bytes
        << " B\n";

    std::cout
        << "  transient reservation: "
        << plan.transient_bytes
        << " B\n";

    std::cout
        << "  initializer staging capacity: "
        << plan.initializer_staging_capacity_bytes
        << " B\n";

    std::cout << "  node plans:\n";

    for (const auto& node : plan.nodes) {
        std::cout
            << "    [" << node.node_index << "] "
            << node.op
            << " initializer_bytes="
            << node.initializer_bytes
            << " strategy="
            << waveaccel::node_memory_strategy_name(
                   node.strategy
               );

        if (
            node.strategy ==
            waveaccel::NodeMemoryStrategy::SingleTensorTiled
        ) {
            std::cout
                << " tiled_tensor="
                << node.tiled_tensor_name
                << " fixed_bytes="
                << node.fixed_initializer_bytes
                << " capacity="
                << node.tile_capacity_bytes
                << " row_bytes="
                << node.weight_row_bytes
                << " tiles="
                << node.tile_count;

            for (const auto& tile : node.gemm_row_tiles) {
                std::cout
                    << " [tile"
                    << tile.tile_index
                    << ": rows="
                    << tile.row_begin
                    << ".."
                    << (tile.row_begin + tile.row_count - 1)
                    << " bytes="
                    << tile.bytes
                    << "]";
            }
        }

        std::cout << '\n';
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr
            << "usage: memory_planner_smoke "
            << "<memory-manifest>\n";
        return 2;
    }

    const auto model =
        waveaccel::load_model_memory_manifest(argv[1]);

    const auto resident =
        waveaccel::build_memory_plan(
            model,
            16U * 1024U
        );

    assert(
        resident.strategy ==
        waveaccel::MemoryStrategy::SramResident
    );

    assert(resident.estimated_working_set_bytes == 10976);
    assert(resident.nodes.size() == 7);

    const auto tiled =
        waveaccel::build_memory_plan(
            model,
            4U * 1024U
        );

    assert(
        tiled.strategy ==
        waveaccel::MemoryStrategy::Tiled
    );

    assert(tiled.transient_bytes == 392);
    assert(tiled.initializer_staging_capacity_bytes == 3704);
    assert(tiled.nodes.size() == 7);

    const auto& gemm0 = tiled.nodes[0];

    assert(gemm0.op == "Gemm");

    assert(
        gemm0.strategy ==
        waveaccel::NodeMemoryStrategy::SingleTensorTiled
    );

    assert(
        gemm0.tiled_tensor_name ==
        "model.net.0.weight"
    );

    assert(gemm0.tiled_tensor_bytes == 8192);
    assert(gemm0.fixed_initializer_bytes == 128);
    assert(gemm0.tile_capacity_bytes == 3576);

    // 32x64 FP32 weight:
    //   one output row = 64 * 4 = 256 B
    //   floor(3576 / 256) = 13 complete rows per SRAM tile
    //   rows = 13 + 13 + 6
    //   bytes = 3328 + 3328 + 1536 = 8192
    assert(gemm0.weight_total_rows == 32);
    assert(gemm0.weight_row_bytes == 256);
    assert(gemm0.tile_count == 3);
    assert(gemm0.gemm_row_tiles.size() == 3);

    assert(gemm0.gemm_row_tiles[0].row_begin == 0);
    assert(gemm0.gemm_row_tiles[0].row_count == 13);
    assert(gemm0.gemm_row_tiles[0].bytes == 3328);

    assert(gemm0.gemm_row_tiles[1].row_begin == 13);
    assert(gemm0.gemm_row_tiles[1].row_count == 13);
    assert(gemm0.gemm_row_tiles[1].bytes == 3328);

    assert(gemm0.gemm_row_tiles[2].row_begin == 26);
    assert(gemm0.gemm_row_tiles[2].row_count == 6);
    assert(gemm0.gemm_row_tiles[2].bytes == 1536);

    assert(
        3328U + 3328U + 1536U ==
        8192U
    );

    assert(
        tiled.nodes[2].strategy ==
        waveaccel::NodeMemoryStrategy::StagedWhole
    );

    assert(
        tiled.nodes[4].strategy ==
        waveaccel::NodeMemoryStrategy::StagedWhole
    );

    print_plan("16 KiB SRAM", resident);
    print_plan("4 KiB SRAM", tiled);

    std::cout
        << "\nRow-aligned graph memory planner: PASS\n";

    return 0;
}
