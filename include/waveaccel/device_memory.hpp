/*
 * WaveAccel - AI Accelerator Runtime Prototype
 *
 * Logical device-memory layout for model initializers.
 *
 * Author: Santosh Kumar
 * Copyright 2026 Santosh Kumar
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "waveaccel/model_memory.hpp"

namespace waveaccel {

/*
 * This is a logical accelerator address space used by WaveAccel's generic
 * backend contract. Offsets are not physical PCIe addresses, MMIO addresses,
 * IOVAs, or proprietary accelerator addresses.
 */
struct DeviceMemoryRegion {
    std::string tensor_name;
    std::size_t offset_bytes{};
    std::size_t bytes{};
};

struct DeviceMemoryMap {
    std::vector<DeviceMemoryRegion> regions;
    std::size_t total_bytes{};
};

DeviceMemoryMap build_device_memory_map(
    const ModelMemoryInfo& model
);

const DeviceMemoryRegion& find_device_memory_region(
    const DeviceMemoryMap& map,
    const std::string& tensor_name
);

}  // namespace waveaccel
