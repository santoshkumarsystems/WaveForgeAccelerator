/*
 * WaveAccel - AI Accelerator Runtime Prototype
 *
 * MockDevice logical address-contract validation smoke test.
 *
 * Author: Santosh Kumar
 * Copyright 2026 Santosh Kumar
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

#include "waveaccel/device.hpp"
#include "waveaccel/device_command.hpp"
#include "waveaccel/execution_plan.hpp"
#include "waveaccel/model_memory.hpp"
#include "waveaccel/runtime.hpp"

namespace {

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
            std::string("expected rejection: ") + label
        );
    }

    std::cout << "rejected: " << label << '\n';
}

template <typename Fn>
void require_invalid_argument(
    const char* label,
    Fn&& fn
) {
    bool rejected = false;

    try {
        fn();
    } catch (const std::invalid_argument&) {
        rejected = true;
    }

    if (!rejected) {
        throw std::runtime_error(
            std::string("expected invalid_argument: ") + label
        );
    }

    std::cout << "rejected: " << label << '\n';
}

waveaccel::DeviceCommand make_dma(
    std::size_t device_offset,
    std::size_t sram_offset,
    std::size_t bytes
) {
    waveaccel::DeviceCommand command;
    command.kind =
        waveaccel::DeviceCommandKind::DmaDeviceToSram;
    command.node_index = 0;
    command.source_region = "contract_test";
    command.device_offset_bytes = device_offset;
    command.sram_offset_bytes = sram_offset;
    command.bytes = bytes;
    return command;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 3) {
            std::cerr
                << "usage: device_address_contract_smoke "
                << "<memory-manifest> <execution-plan-manifest>\n";
            return 2;
        }

        const auto model =
            waveaccel::load_model_memory_manifest(argv[1]);

        const auto plan =
            waveaccel::load_execution_plan_manifest(argv[2]);

        waveaccel::MockDeviceConfig config;
        config.sram_capacity_bytes =
            plan.target.sram_bytes;

        waveaccel::MockDevice device{config};

        device.configure_address_contract(
            model.initializer_bytes,
            plan.target.initializer_staging_capacity_bytes
        );

        waveaccel::Profiler profiler;

        // Valid range: last byte of logical model image.
        const auto valid =
            make_dma(
                model.initializer_bytes - 1U,
                0,
                1
            );

        device.execute(valid, profiler);

        require_runtime_error(
            "device source range overflow",
            [&] {
                auto bad = make_dma(
                    model.initializer_bytes - 1U,
                    0,
                    2
                );
                device.execute(bad, profiler);
            }
        );

        require_runtime_error(
            "initializer staging range overflow",
            [&] {
                auto bad = make_dma(
                    0,
                    plan.target.initializer_staging_capacity_bytes - 1U,
                    2
                );
                device.execute(bad, profiler);
            }
        );

        require_runtime_error(
            "full SRAM range overflow",
            [&] {
                // Use a second unconfigured device so this specifically checks
                // the physical simulated SRAM bound rather than staging bound.
                waveaccel::MockDevice raw_device{config};
                waveaccel::Profiler raw_profiler;

                auto bad = make_dma(
                    0,
                    config.sram_capacity_bytes,
                    1
                );
                raw_device.execute(bad, raw_profiler);
            }
        );

        require_invalid_argument(
            "staging capacity larger than SRAM",
            [&] {
                waveaccel::MockDevice invalid_device{config};
                invalid_device.configure_address_contract(
                    model.initializer_bytes,
                    config.sram_capacity_bytes + 1U
                );
            }
        );

        // Verify Runtime::set_execution_plan installs/validates the same
        // contract. A compiler plan cannot declare an initializer staging
        // region larger than the device SRAM.
        require_invalid_argument(
            "Runtime rejects invalid compiler staging contract",
            [&] {
                auto invalid_plan = plan;
                invalid_plan.target.initializer_staging_capacity_bytes =
                    invalid_plan.target.sram_bytes + 1U;

                waveaccel::Runtime runtime(
                    model,
                    waveaccel::MockDevice{config}
                );

                runtime.set_execution_plan(
                    std::move(invalid_plan)
                );
            }
        );

        std::cout
            << "\nlogical device bytes:       "
            << model.initializer_bytes
            << '\n'
            << "SRAM bytes:                 "
            << plan.target.sram_bytes
            << '\n'
            << "initializer staging bytes:  "
            << plan.target.initializer_staging_capacity_bytes
            << '\n'
            << "MockDevice address contract: PASS\n";

        return 0;
    } catch (const std::exception& error) {
        std::cerr
            << "device_address_contract_smoke: FAIL: "
            << error.what()
            << '\n';
        return 1;
    }
}
