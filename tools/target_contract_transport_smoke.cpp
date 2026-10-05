/*
 * WaveAccel - AI Accelerator Runtime Prototype
 *
 * Compiler target-contract transport smoke test.
 *
 * Author: Santosh Kumar
 * Copyright 2026 Santosh Kumar
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "waveaccel/device.hpp"
#include "waveaccel/execution_plan.hpp"
#include "waveaccel/model_memory.hpp"
#include "waveaccel/runtime.hpp"

namespace {

bool has_op(
    const std::vector<std::string>& ops,
    const std::string& op
) {
    return std::find(
        ops.begin(),
        ops.end(),
        op
    ) != ops.end();
}

template <typename Fn>
void require_runtime_error(
    const char* label,
    Fn&& fn
) {
    bool rejected = false;

    try {
        fn();
    } catch (const std::runtime_error&) {
        rejected = true;
    }

    if (!rejected) {
        throw std::runtime_error(
            std::string("expected Runtime rejection: ") + label
        );
    }

    std::cout << "rejected: " << label << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 3) {
            std::cerr
                << "usage: target_contract_transport_smoke "
                << "<memory-manifest> <execution-plan-manifest>\n";
            return 2;
        }

        const auto model =
            waveaccel::load_model_memory_manifest(argv[1]);

        const auto plan =
            waveaccel::load_execution_plan_manifest(argv[2]);

        const auto& target = plan.target;

        if (target.contract_schema_version != 1) {
            throw std::runtime_error(
                "unexpected target contract schema"
            );
        }

        if (
            target.name != "waveaccel-sim" ||
            target.dtype != "float32" ||
            target.element_bytes != sizeof(float)
        ) {
            throw std::runtime_error(
                "target identity/dtype contract mismatch"
            );
        }

        const std::vector<std::string> expected_ops{
            "Gemm",
            "Relu",
            "Mul",
            "Add",
        };

        if (target.supported_ops != expected_ops) {
            throw std::runtime_error(
                "supported-op transport mismatch"
            );
        }

        if (
            target.model_initializer_memory_space !=
                "device_memory" ||
            target.initializer_staging_memory_space !=
                "sram"
        ) {
            throw std::runtime_error(
                "memory-space transport mismatch"
            );
        }

        if (
            target.gemm_tiling_axis != "output_channel" ||
            target.gemm_tile_unit !=
                "complete_output_weight_row" ||
            target.gemm_tiled_fixed_initializer != "bias"
        ) {
            throw std::runtime_error(
                "Gemm-policy transport mismatch"
            );
        }

        for (const auto& node : model.nodes) {
            if (!has_op(target.supported_ops, node.op)) {
                throw std::runtime_error(
                    "model contains op not declared by target: " +
                    node.op
                );
            }
        }

        waveaccel::MockDeviceConfig config;
        config.sram_capacity_bytes =
            target.sram_bytes;

        {
            waveaccel::Runtime runtime(
                model,
                waveaccel::MockDevice{config}
            );

            runtime.set_execution_plan(plan);
        }

        require_runtime_error(
            "unsupported dtype",
            [&] {
                auto bad = plan;
                bad.target.dtype = "float16";

                waveaccel::Runtime runtime(
                    model,
                    waveaccel::MockDevice{config}
                );
                runtime.set_execution_plan(std::move(bad));
            }
        );

        require_runtime_error(
            "unsupported logical memory-space contract",
            [&] {
                auto bad = plan;
                bad.target.initializer_staging_memory_space =
                    "unknown_sram";

                waveaccel::Runtime runtime(
                    model,
                    waveaccel::MockDevice{config}
                );
                runtime.set_execution_plan(std::move(bad));
            }
        );

        require_runtime_error(
            "model op omitted from target supported-op contract",
            [&] {
                auto bad = plan;
                bad.target.supported_ops.erase(
                    std::remove(
                        bad.target.supported_ops.begin(),
                        bad.target.supported_ops.end(),
                        "Gemm"
                    ),
                    bad.target.supported_ops.end()
                );

                waveaccel::Runtime runtime(
                    model,
                    waveaccel::MockDevice{config}
                );
                runtime.set_execution_plan(std::move(bad));
            }
        );

        std::cout
            << "\nWaveAccel C++ target contract\n"
            << "========================================\n"
            << "contract schema:              "
            << target.contract_schema_version
            << '\n'
            << "target:                       "
            << target.name
            << '\n'
            << "dtype / element bytes:        "
            << target.dtype
            << " / "
            << target.element_bytes
            << '\n'
            << "memory spaces:                "
            << target.model_initializer_memory_space
            << " -> "
            << target.initializer_staging_memory_space
            << '\n'
            << "Gemm tiling axis:             "
            << target.gemm_tiling_axis
            << '\n'
            << "Gemm tile unit:               "
            << target.gemm_tile_unit
            << '\n'
            << "tiled fixed initializer:      "
            << target.gemm_tiled_fixed_initializer
            << '\n'
            << "supported ops:                ";

        for (
            std::size_t i = 0;
            i < target.supported_ops.size();
            ++i
        ) {
            if (i != 0) {
                std::cout << ", ";
            }
            std::cout << target.supported_ops[i];
        }

        std::cout
            << "\n\nPython compiler contract -> C++ Runtime contract: PASS\n";

        return 0;
    } catch (const std::exception& error) {
        std::cerr
            << "target_contract_transport_smoke: FAIL: "
            << error.what()
            << '\n';
        return 1;
    }
}
