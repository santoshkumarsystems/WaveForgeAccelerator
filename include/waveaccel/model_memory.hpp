/*
 * WaveAccel - AI Accelerator Runtime Prototype
 *
 * ONNX-derived model memory and graph metadata.
 *
 * Author: Santosh Kumar
 * Copyright 2026 Santosh Kumar
 * SPDX-License-Identifier: Apache-2.0
 *
 * Model metadata is generated from the exported ONNX model. WaveAccel uses
 * this information for generic accelerator memory-planning and graph-driven
 * scheduling experiments instead of hard-coding model tensor sizes or graph
 * relationships.
 *
 * NOTE:
 * Accelerator timing, SRAM behavior, DMA behavior, and tiling modeled by
 * WaveAccel are simulations and are not measurements or representations of
 * proprietary commercial accelerator hardware.
 */

#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace waveaccel {

struct ModelTensorInfo {
    std::string name;
    std::string shape;
    std::size_t elements{0};
    std::size_t bytes{0};
};

struct ModelGraphNodeInfo {
    std::string op;

    std::vector<std::string> inputs;
    std::vector<std::string> outputs;

    std::size_t initializer_count{0};
    std::vector<std::string> initializers;
};

struct ModelMemoryInfo {
    std::size_t batch_size{1};

    std::size_t input_bytes{0};
    std::size_t output_bytes{0};

    std::size_t initializer_count{0};
    std::size_t initializer_elements{0};
    std::size_t initializer_bytes{0};

    std::string largest_initializer_name;
    std::size_t largest_initializer_bytes{0};

    std::string largest_activation_name;
    std::size_t largest_activation_bytes{0};

    std::string operator_sequence;

    std::size_t node_count{0};

    // Persistent tensors extracted from the real ONNX model.
    std::vector<ModelTensorInfo> initializers;

    // ONNX execution graph extracted from the real ONNX model.
    std::vector<ModelGraphNodeInfo> nodes;
};

ModelMemoryInfo load_model_memory_manifest(const std::string& path);

}  // namespace waveaccel
