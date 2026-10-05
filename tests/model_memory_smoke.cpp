/*
 * WaveAccel - AI Accelerator Runtime Prototype
 *
 * Smoke test for ONNX-derived model memory and graph metadata.
 *
 * Author: Santosh Kumar
 * Copyright 2026 Santosh Kumar
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cassert>
#include <iostream>

#include "waveaccel/model_memory.hpp"

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: model_memory_smoke <memory-manifest>\n";
        return 2;
    }

    const auto info =
        waveaccel::load_model_memory_manifest(argv[1]);

    assert(info.batch_size == 1);
    assert(info.input_bytes == 256);
    assert(info.output_bytes == 8);

    assert(info.initializer_count == 8);
    assert(info.initializers.size() == 8);
    assert(info.initializer_elements == 2646);
    assert(info.initializer_bytes == 10584);

    assert(info.initializers[0].name == "model.net.0.weight");
    assert(info.initializers[0].shape == "32x64");
    assert(info.initializers[0].elements == 2048);
    assert(info.initializers[0].bytes == 8192);

    assert(info.initializers[2].name == "model.net.2.weight");
    assert(info.initializers[2].shape == "16x32");
    assert(info.initializers[2].bytes == 2048);

    assert(info.initializers[4].name == "model.net.4.weight");
    assert(info.initializers[4].shape == "2x16");
    assert(info.initializers[4].bytes == 128);

    assert(info.initializers[6].name == "target_mean");
    assert(info.initializers[7].name == "target_std");

    assert(info.node_count == 7);
    assert(info.nodes.size() == 7);

    assert(info.nodes[0].op == "Gemm");
    assert(info.nodes[0].inputs.size() == 3);
    assert(info.nodes[0].outputs.size() == 1);
    assert(info.nodes[0].initializer_count == 2);
    assert(info.nodes[0].initializers.size() == 2);
    assert(
        info.nodes[0].initializers[0] ==
        "model.net.0.weight"
    );
    assert(
        info.nodes[0].initializers[1] ==
        "model.net.0.bias"
    );

    assert(info.nodes[1].op == "Relu");
    assert(info.nodes[1].initializer_count == 0);
    assert(info.nodes[1].initializers.empty());

    assert(info.nodes[2].op == "Gemm");
    assert(
        info.nodes[2].initializers[0] ==
        "model.net.2.weight"
    );

    assert(info.nodes[4].op == "Gemm");
    assert(
        info.nodes[4].initializers[0] ==
        "model.net.4.weight"
    );

    assert(info.nodes[5].op == "Mul");
    assert(info.nodes[5].initializer_count == 1);
    assert(info.nodes[5].initializers[0] == "target_std");

    assert(info.nodes[6].op == "Add");
    assert(info.nodes[6].initializer_count == 1);
    assert(info.nodes[6].initializers[0] == "target_mean");

    std::cout << "Model memory + graph manifest: PASS\n";

    std::cout << "\nONNX initializer tensors:\n";
    for (const auto& tensor : info.initializers) {
        std::cout
            << "  " << tensor.name
            << " shape=" << tensor.shape
            << " bytes=" << tensor.bytes
            << '\n';
    }

    std::cout << "\nONNX graph nodes:\n";
    for (std::size_t index = 0; index < info.nodes.size(); ++index) {
        const auto& node = info.nodes[index];

        std::cout
            << "  [" << index << "] "
            << node.op
            << " initializer_count="
            << node.initializer_count
            << '\n';
    }

    return 0;
}
