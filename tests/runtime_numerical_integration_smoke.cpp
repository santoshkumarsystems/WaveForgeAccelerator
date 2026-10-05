/*
 * WaveAccel - AI Accelerator Runtime Prototype
 *
 * Integrated runtime + real row-tiled numerical inference smoke test.
 *
 * Author: Santosh Kumar
 * Copyright 2026 Santosh Kumar
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <cassert>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "waveaccel/device.hpp"
#include "waveaccel/model_memory.hpp"
#include "waveaccel/numerical_executor.hpp"
#include "waveaccel/runtime.hpp"
#include "waveaccel/tensor_store.hpp"

namespace {

float max_abs_error(
    const std::vector<float>& lhs,
    const std::vector<float>& rhs
) {
    if (lhs.size() != rhs.size()) {
        throw std::runtime_error(
            "integration parity vectors differ in size"
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

const waveaccel::ProfileEvent* find_event(
    const waveaccel::Profiler& profiler,
    const std::string& name
) {
    for (const auto& event : profiler.events()) {
        if (event.name == name) {
            return &event;
        }
    }

    return nullptr;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
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

    waveaccel::MockDeviceConfig resident_config;
    resident_config.sram_capacity_bytes =
        16U * 1024U;

    waveaccel::Runtime resident_runtime(
        model,
        waveaccel::MockDevice{resident_config}
    );

    const auto resident_cold =
        resident_runtime.run_once(
            sample.input,
            tensors
        );

    const auto resident_warm =
        resident_runtime.run_once(
            sample.input,
            tensors
        );

    assert(
        max_abs_error(
            resident_cold.output,
            sample.reference_output
        ) <= 1.0e-5F
    );

    assert(
        max_abs_error(
            resident_warm.output,
            sample.reference_output
        ) <= 1.0e-5F
    );

    assert(
        resident_warm.profiler.device_sram_transfer_bytes() ==
        0U
    );

    waveaccel::MockDeviceConfig tiled_config;
    tiled_config.sram_capacity_bytes =
        4U * 1024U;

    waveaccel::Runtime tiled_runtime(
        model,
        waveaccel::MockDevice{tiled_config}
    );

    const auto tiled_cold =
        tiled_runtime.run_once(
            sample.input,
            tensors
        );

    const auto tiled_warm =
        tiled_runtime.run_once(
            sample.input,
            tensors
        );

    // Real row-tiled Gemm execution must remain numerically equivalent to
    // ONNX Runtime and to the resident full-Gemm path.
    assert(
        max_abs_error(
            tiled_cold.output,
            sample.reference_output
        ) <= 1.0e-5F
    );

    assert(
        max_abs_error(
            tiled_warm.output,
            sample.reference_output
        ) <= 1.0e-5F
    );

    assert(
        max_abs_error(
            tiled_warm.output,
            resident_warm.output
        ) <= 1.0e-6F
    );

    assert(
        tiled_warm.profiler.host_device_transfer_bytes() ==
        256U + 8U
    );

    assert(
        tiled_warm.profiler.device_sram_transfer_bytes() ==
        10584U
    );

    const auto* stage0 =
        find_event(
            tiled_warm.profiler,
            "NODE_0_STAGE_WEIGHT_TILE_0"
        );

    const auto* stage1 =
        find_event(
            tiled_warm.profiler,
            "NODE_0_STAGE_WEIGHT_TILE_1"
        );

    const auto* stage2 =
        find_event(
            tiled_warm.profiler,
            "NODE_0_STAGE_WEIGHT_TILE_2"
        );

    assert(stage0 != nullptr && stage0->bytes == 3328);
    assert(stage1 != nullptr && stage1->bytes == 3328);
    assert(stage2 != nullptr && stage2->bytes == 1536);

    assert(
        find_event(
            tiled_warm.profiler,
            "NODE_0_GEMM_TILE_0"
        ) != nullptr
    );

    assert(
        find_event(
            tiled_warm.profiler,
            "NODE_0_GEMM_TILE_1"
        ) != nullptr
    );

    assert(
        find_event(
            tiled_warm.profiler,
            "NODE_0_GEMM_TILE_2"
        ) != nullptr
    );

    return 0;
}
