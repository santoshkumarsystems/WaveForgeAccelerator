/*
 * WaveAccel - AI Accelerator Runtime Prototype
 *
 * Logical device-memory layout implementation.
 *
 * Author: Santosh Kumar
 * Copyright 2026 Santosh Kumar
 * SPDX-License-Identifier: Apache-2.0
 */

#include "waveaccel/device_memory.hpp"

#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace waveaccel {

DeviceMemoryMap build_device_memory_map(
    const ModelMemoryInfo& model
) {
    DeviceMemoryMap map;
    map.regions.reserve(model.initializers.size());

    std::unordered_map<std::string, bool> seen;
    std::size_t next_offset = 0;

    for (const auto& tensor : model.initializers) {
        if (tensor.name.empty() || tensor.bytes == 0) {
            throw std::runtime_error(
                "cannot map incomplete model initializer"
            );
        }

        if (!seen.emplace(tensor.name, true).second) {
            throw std::runtime_error(
                "duplicate initializer in device-memory map: " +
                tensor.name
            );
        }

        if (
            tensor.bytes >
            std::numeric_limits<std::size_t>::max() - next_offset
        ) {
            throw std::runtime_error(
                "device-memory initializer layout overflow"
            );
        }

        map.regions.push_back(
            DeviceMemoryRegion{
                tensor.name,
                next_offset,
                tensor.bytes,
            }
        );

        next_offset += tensor.bytes;
    }

    map.total_bytes = next_offset;

    if (map.regions.size() != model.initializer_count) {
        throw std::runtime_error(
            "device-memory region count does not match model initializer count"
        );
    }

    if (map.total_bytes != model.initializer_bytes) {
        throw std::runtime_error(
            "device-memory byte total does not match model initializer bytes"
        );
    }

    return map;
}

const DeviceMemoryRegion& find_device_memory_region(
    const DeviceMemoryMap& map,
    const std::string& tensor_name
) {
    for (const auto& region : map.regions) {
        if (region.tensor_name == tensor_name) {
            return region;
        }
    }

    throw std::runtime_error(
        "device-memory map does not contain initializer: " +
        tensor_name
    );
}

}  // namespace waveaccel
