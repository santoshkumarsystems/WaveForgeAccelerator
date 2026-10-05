/*
 * WaveAccel - AI Accelerator Runtime Prototype
 *
 * Mock runtime profiler.
 *
 * Author: Santosh Kumar
 * Copyright 2026 Santosh Kumar
 * SPDX-License-Identifier: Apache-2.0
 *
 * All timing values recorded here are simulation outputs unless explicitly
 * stated otherwise. They are not real-hardware benchmark measurements.
 */

#pragma once

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace waveaccel {

enum class TransferDomain {
    None,
    HostDevice,
    DeviceSram,
};

inline const char* transfer_domain_name(
    TransferDomain domain
) noexcept {
    switch (domain) {
        case TransferDomain::None:
            return "COMPUTE";

        case TransferDomain::HostDevice:
            return "HOST_DEVICE";

        case TransferDomain::DeviceSram:
            return "DEVICE_SRAM";
    }

    return "UNKNOWN";
}

struct ProfileEvent {
    std::string name;
    double simulated_us{0.0};
    std::size_t bytes{0};
    TransferDomain transfer_domain{TransferDomain::None};
};

class Profiler {
public:
    void record(ProfileEvent event) {
        events_.push_back(std::move(event));
    }

    const std::vector<ProfileEvent>& events() const noexcept {
        return events_;
    }

    double total_simulated_us() const noexcept {
        double total = 0.0;

        for (const auto& event : events_) {
            total += event.simulated_us;
        }

        return total;
    }

    std::size_t total_transfer_bytes() const noexcept {
        std::size_t total = 0;

        for (const auto& event : events_) {
            total += event.bytes;
        }

        return total;
    }

    std::size_t host_device_transfer_bytes() const noexcept {
        std::size_t total = 0;

        for (const auto& event : events_) {
            if (
                event.transfer_domain ==
                TransferDomain::HostDevice
            ) {
                total += event.bytes;
            }
        }

        return total;
    }

    std::size_t device_sram_transfer_bytes() const noexcept {
        std::size_t total = 0;

        for (const auto& event : events_) {
            if (
                event.transfer_domain ==
                TransferDomain::DeviceSram
            ) {
                total += event.bytes;
            }
        }

        return total;
    }

private:
    std::vector<ProfileEvent> events_;
};

}  // namespace waveaccel
