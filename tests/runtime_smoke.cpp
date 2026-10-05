/*
 * WaveAccel - AI Accelerator Runtime Prototype
 *
 * Graph-driven runtime smoke test.
 *
 * Author: Santosh Kumar
 * Copyright 2026 Santosh Kumar
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cassert>
#include <cstddef>
#include <string>

#include "waveaccel/device.hpp"
#include "waveaccel/model_memory.hpp"
#include "waveaccel/runtime.hpp"

namespace {

waveaccel::ModelMemoryInfo make_model_fixture() {
    waveaccel::ModelMemoryInfo model;

    model.batch_size = 1;
    model.input_bytes = 256;
    model.output_bytes = 8;

    model.initializer_count = 8;
    model.initializer_elements = 2646;
    model.initializer_bytes = 10584;

    model.largest_initializer_name =
        "model.net.0.weight";

    model.largest_initializer_bytes = 8192;

    model.largest_activation_name = "linear";
    model.largest_activation_bytes = 128;

    model.operator_sequence =
        "Gemm,Relu,Gemm,Relu,Gemm,Mul,Add";

    model.node_count = 7;

    model.initializers = {
        {"model.net.0.weight", "32x64", 2048, 8192},
        {"model.net.0.bias", "32", 32, 128},
        {"model.net.2.weight", "16x32", 512, 2048},
        {"model.net.2.bias", "16", 16, 64},
        {"model.net.4.weight", "2x16", 32, 128},
        {"model.net.4.bias", "2", 2, 8},
        {"target_mean", "2", 2, 8},
        {"target_std", "2", 2, 8},
    };

    model.nodes = {
        {
            "Gemm",
            {
                "wave_samples",
                "model.net.0.weight",
                "model.net.0.bias",
            },
            {"linear"},
            2,
            {
                "model.net.0.weight",
                "model.net.0.bias",
            },
        },
        {
            "Relu",
            {"linear"},
            {"relu"},
            0,
            {},
        },
        {
            "Gemm",
            {
                "relu",
                "model.net.2.weight",
                "model.net.2.bias",
            },
            {"linear_1"},
            2,
            {
                "model.net.2.weight",
                "model.net.2.bias",
            },
        },
        {
            "Relu",
            {"linear_1"},
            {"relu_1"},
            0,
            {},
        },
        {
            "Gemm",
            {
                "relu_1",
                "model.net.4.weight",
                "model.net.4.bias",
            },
            {"linear_2"},
            2,
            {
                "model.net.4.weight",
                "model.net.4.bias",
            },
        },
        {
            "Mul",
            {"linear_2", "target_std"},
            {"mul_10"},
            1,
            {"target_std"},
        },
        {
            "Add",
            {"mul_10", "target_mean"},
            {"wave_parameters"},
            1,
            {"target_mean"},
        },
    };

    return model;
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

int main() {
    waveaccel::MockDeviceConfig resident_config;
    resident_config.sram_capacity_bytes =
        16U * 1024U;

    waveaccel::Runtime resident_runtime(
        make_model_fixture(),
        waveaccel::MockDevice{resident_config}
    );

    const auto resident_cold =
        resident_runtime.run_once();

    assert(resident_runtime.model_in_device_memory());
    assert(resident_runtime.model_in_sram());

    assert(
        resident_cold.host_device_transfer_bytes() ==
        256U + 10584U + 8U
    );

    assert(
        resident_cold.device_sram_transfer_bytes() ==
        10584U
    );

    const auto resident_warm =
        resident_runtime.run_once();

    assert(
        resident_warm.host_device_transfer_bytes() ==
        256U + 8U
    );

    assert(
        resident_warm.device_sram_transfer_bytes() ==
        0U
    );

    waveaccel::MockDeviceConfig tiled_config;
    tiled_config.sram_capacity_bytes =
        4U * 1024U;

    waveaccel::Runtime tiled_runtime(
        make_model_fixture(),
        waveaccel::MockDevice{tiled_config}
    );

    const auto tiled_cold =
        tiled_runtime.run_once();

    assert(tiled_runtime.model_in_device_memory());
    assert(!tiled_runtime.model_in_sram());

    assert(
        tiled_cold.host_device_transfer_bytes() ==
        256U + 10584U + 8U
    );

    assert(
        tiled_cold.device_sram_transfer_bytes() ==
        10584U
    );

    const auto* stage0 =
        find_event(
            tiled_cold,
            "NODE_0_STAGE_WEIGHT_TILE_0"
        );

    const auto* stage1 =
        find_event(
            tiled_cold,
            "NODE_0_STAGE_WEIGHT_TILE_1"
        );

    const auto* stage2 =
        find_event(
            tiled_cold,
            "NODE_0_STAGE_WEIGHT_TILE_2"
        );

    assert(stage0 != nullptr && stage0->bytes == 3328);
    assert(stage1 != nullptr && stage1->bytes == 3328);
    assert(stage2 != nullptr && stage2->bytes == 1536);

    assert(
        find_event(
            tiled_cold,
            "NODE_0_GEMM_TILE_0"
        ) != nullptr
    );

    assert(
        find_event(
            tiled_cold,
            "NODE_0_GEMM_TILE_1"
        ) != nullptr
    );

    assert(
        find_event(
            tiled_cold,
            "NODE_0_GEMM_TILE_2"
        ) != nullptr
    );

    const auto tiled_warm =
        tiled_runtime.run_once();

    assert(
        tiled_warm.host_device_transfer_bytes() ==
        256U + 8U
    );

    assert(
        tiled_warm.device_sram_transfer_bytes() ==
        10584U
    );

    return 0;
}
