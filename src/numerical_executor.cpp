/*
 * WaveAccel - AI Accelerator Runtime Prototype
 *
 * Float32 numerical executor with real row-aligned tiled GEMM support.
 *
 * Author: Santosh Kumar
 * Copyright 2026 Santosh Kumar
 * SPDX-License-Identifier: Apache-2.0
 */

#include "waveaccel/numerical_executor.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace waveaccel {

namespace {

constexpr std::array<char, 8> kSampleMagic{
    'W', 'A', 'S', 'M', 'P', '0', '0', '1'
};

constexpr std::uint32_t kFormatVersion = 1;

template <typename T>
T read_scalar(std::ifstream& input) {
    T value{};

    input.read(
        reinterpret_cast<char*>(&value),
        sizeof(T)
    );

    if (!input) {
        throw std::runtime_error(
            "unexpected end of numerical sample file"
        );
    }

    return value;
}

std::vector<float> read_float_vector(
    std::ifstream& input,
    std::uint32_t count
) {
    std::vector<float> values(count);

    input.read(
        reinterpret_cast<char*>(values.data()),
        static_cast<std::streamsize>(
            values.size() * sizeof(float)
        )
    );

    if (!input) {
        throw std::runtime_error(
            "unexpected end of numerical sample values"
        );
    }

    return values;
}

}  // namespace

NumericalExecutor::NumericalExecutor(
    const ModelMemoryInfo& model,
    const TensorStore& tensors
)
    : model_(model),
      tensors_(tensors) {
    if (model_.nodes.empty()) {
        throw std::invalid_argument(
            "numerical executor requires graph nodes"
        );
    }

    if (tensors_.size() != model_.initializer_count) {
        throw std::invalid_argument(
            "tensor store count does not match ONNX manifest"
        );
    }
}

std::vector<float> NumericalExecutor::execute_gemm_rows(
    const ModelGraphNodeInfo& node,
    const std::vector<float>& input,
    std::size_t row_begin,
    std::size_t row_count
) const {
    if (node.initializers.size() != 2) {
        throw std::runtime_error(
            "WaveAccel V1 Gemm requires weight and bias initializers"
        );
    }

    const auto& weight =
        tensors_.at(node.initializers[0]);

    const auto& bias =
        tensors_.at(node.initializers[1]);

    if (weight.shape.size() != 2) {
        throw std::runtime_error(
            "Gemm weight tensor must have rank 2"
        );
    }

    if (bias.shape.size() != 1) {
        throw std::runtime_error(
            "Gemm bias tensor must have rank 1"
        );
    }

    const auto output_size =
        static_cast<std::size_t>(weight.shape[0]);

    const auto input_size =
        static_cast<std::size_t>(weight.shape[1]);

    if (input.size() != input_size) {
        throw std::runtime_error(
            "Gemm input size does not match weight shape"
        );
    }

    if (bias.values.size() != output_size) {
        throw std::runtime_error(
            "Gemm bias size does not match output size"
        );
    }

    if (
        weight.values.size() !=
        output_size * input_size
    ) {
        throw std::runtime_error(
            "Gemm weight storage size mismatch"
        );
    }

    if (
        row_begin > output_size ||
        row_count > output_size - row_begin
    ) {
        throw std::runtime_error(
            "Gemm row tile lies outside output range"
        );
    }

    std::vector<float> output(row_count);

    for (
        std::size_t local_row = 0;
        local_row < row_count;
        ++local_row
    ) {
        const std::size_t row =
            row_begin + local_row;

        float sum = bias.values[row];

        const std::size_t row_offset =
            row * input_size;

        for (
            std::size_t column = 0;
            column < input_size;
            ++column
        ) {
            sum +=
                input[column] *
                weight.values[
                    row_offset + column
                ];
        }

        output[local_row] = sum;
    }

    return output;
}

std::vector<float> NumericalExecutor::execute_gemm(
    const ModelGraphNodeInfo& node,
    const std::vector<float>& input
) const {
    const auto& weight =
        tensors_.at(node.initializers.at(0));

    if (weight.shape.size() != 2) {
        throw std::runtime_error(
            "Gemm weight tensor must have rank 2"
        );
    }

    return execute_gemm_rows(
        node,
        input,
        0,
        static_cast<std::size_t>(weight.shape[0])
    );
}

std::vector<float> NumericalExecutor::execute_relu(
    const std::vector<float>& input
) const {
    std::vector<float> output = input;

    for (float& value : output) {
        value = std::max(0.0F, value);
    }

    return output;
}

std::vector<float> NumericalExecutor::execute_elementwise(
    const ModelGraphNodeInfo& node,
    const std::vector<float>& input,
    bool multiply
) const {
    if (node.initializers.size() != 1) {
        throw std::runtime_error(
            "elementwise node requires one initializer"
        );
    }

    const auto& constant =
        tensors_.at(node.initializers[0]);

    if (constant.values.size() != input.size()) {
        throw std::runtime_error(
            "elementwise constant size does not match activation size"
        );
    }

    std::vector<float> output(input.size());

    for (
        std::size_t index = 0;
        index < input.size();
        ++index
    ) {
        output[index] =
            multiply
                ? input[index] * constant.values[index]
                : input[index] + constant.values[index];
    }

    return output;
}

std::vector<float> NumericalExecutor::execute_node(
    const ModelGraphNodeInfo& node,
    const std::vector<float>& input
) const {
    if (node.op == "Gemm") {
        return execute_gemm(node, input);
    }

    if (node.op == "Relu") {
        return execute_relu(input);
    }

    if (node.op == "Mul") {
        return execute_elementwise(
            node,
            input,
            true
        );
    }

    if (node.op == "Add") {
        return execute_elementwise(
            node,
            input,
            false
        );
    }

    throw std::runtime_error(
        "unsupported numerical ONNX operator: " +
        node.op
    );
}

std::vector<float> NumericalExecutor::run(
    const std::vector<float>& input
) const {
    std::vector<float> activation = input;

    for (const auto& node : model_.nodes) {
        activation =
            execute_node(
                node,
                activation
            );
    }

    return activation;
}

NumericalSample load_numerical_sample(
    const std::string& path
) {
    std::ifstream input(
        path,
        std::ios::binary
    );

    if (!input) {
        throw std::runtime_error(
            "unable to open numerical sample file: " +
            path
        );
    }

    std::array<char, 8> magic{};

    input.read(
        magic.data(),
        static_cast<std::streamsize>(magic.size())
    );

    if (!input || magic != kSampleMagic) {
        throw std::runtime_error(
            "invalid WaveAccel sample-file magic"
        );
    }

    const auto version =
        read_scalar<std::uint32_t>(input);

    if (version != kFormatVersion) {
        throw std::runtime_error(
            "unsupported WaveAccel sample-file version"
        );
    }

    const auto input_count =
        read_scalar<std::uint32_t>(input);

    const auto target_count =
        read_scalar<std::uint32_t>(input);

    const auto reference_count =
        read_scalar<std::uint32_t>(input);

    NumericalSample sample;

    sample.input =
        read_float_vector(input, input_count);

    sample.ground_truth =
        read_float_vector(input, target_count);

    sample.reference_output =
        read_float_vector(input, reference_count);

    return sample;
}

}  // namespace waveaccel
