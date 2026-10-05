/*
 * WaveAccel - AI Accelerator Runtime Prototype
 *
 * Runtime task definitions.
 *
 * Author: Santosh Kumar
 * Copyright 2026 Santosh Kumar
 * SPDX-License-Identifier: Apache-2.0
 *
 * Tasks represent generic accelerator-runtime operations generated from the
 * ONNX graph and WaveAccel's SRAM memory plan.
 *
 * NOTE:
 * These tasks model generic accelerator behavior. They do not represent the
 * proprietary command stream of any commercial accelerator.
 */

#pragma once

#include <cstddef>
#include <limits>
#include <string>

namespace waveaccel {

inline constexpr std::size_t kNoNodeIndex =
    std::numeric_limits<std::size_t>::max();

inline constexpr std::size_t kNoTileIndex =
    std::numeric_limits<std::size_t>::max();

enum class TaskKind {
    H2D,
    LoadModelToDevice,
    StageToSram,
    StageModelToSram,
    Gemm,
    GemmTile,
    Relu,
    Mul,
    Add,
    D2H,
};

struct Task {
    TaskKind kind;
    std::string name;

    // Number of bytes moved for transfer tasks.
    std::size_t bytes{0};

    // ONNX graph node associated with a compute task.
    std::size_t node_index{kNoNodeIndex};

    // Real row-aligned GEMM tile metadata.
    std::size_t tile_index{kNoTileIndex};
    std::size_t tile_count{1};
    std::size_t row_begin{0};
    std::size_t row_count{0};
    std::size_t total_rows{0};
};

inline bool is_compute_task(TaskKind kind) noexcept {
    switch (kind) {
        case TaskKind::Gemm:
        case TaskKind::GemmTile:
        case TaskKind::Relu:
        case TaskKind::Mul:
        case TaskKind::Add:
            return true;

        case TaskKind::H2D:
        case TaskKind::LoadModelToDevice:
        case TaskKind::StageToSram:
        case TaskKind::StageModelToSram:
        case TaskKind::D2H:
            return false;
    }

    return false;
}

}  // namespace waveaccel
