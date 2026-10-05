/*
 * WaveAccel - AI Accelerator Runtime Prototype
 *
 * Float32 numerical executor for the WaveAccel V1 ONNX graph.
 *
 * Author: Santosh Kumar
 * Copyright 2026 Santosh Kumar
 * SPDX-License-Identifier: Apache-2.0
 *
 * Supported V1 operators:
 *   Gemm  (PyTorch Linear convention: transB=1, alpha=1, beta=1)
 *   Relu
 *   Mul
 *   Add
 *
 * The executor supports both full GEMM and real output-row GEMM tiles.
 * Predictions use the actual trained ONNX initializer values.
 */

#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "waveaccel/model_memory.hpp"
#include "waveaccel/tensor_store.hpp"

namespace waveaccel {

class NumericalExecutor {
public:
    NumericalExecutor(
        const ModelMemoryInfo& model,
        const TensorStore& tensors
    );

    std::vector<float> execute_node(
        const ModelGraphNodeInfo& node,
        const std::vector<float>& input
    ) const;

    // Execute a contiguous range of output rows from one Gemm node.
    std::vector<float> execute_gemm_rows(
        const ModelGraphNodeInfo& node,
        const std::vector<float>& input,
        std::size_t row_begin,
        std::size_t row_count
    ) const;

    std::vector<float> run(
        const std::vector<float>& input
    ) const;

private:
    std::vector<float> execute_gemm(
        const ModelGraphNodeInfo& node,
        const std::vector<float>& input
    ) const;

    std::vector<float> execute_relu(
        const std::vector<float>& input
    ) const;

    std::vector<float> execute_elementwise(
        const ModelGraphNodeInfo& node,
        const std::vector<float>& input,
        bool multiply
    ) const;

    const ModelMemoryInfo& model_;
    const TensorStore& tensors_;
};

struct NumericalSample {
    std::vector<float> input;
    std::vector<float> ground_truth;
    std::vector<float> reference_output;
};

NumericalSample load_numerical_sample(
    const std::string& path
);

}  // namespace waveaccel
