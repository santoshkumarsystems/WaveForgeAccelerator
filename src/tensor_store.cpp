/*
 * WaveAccel - AI Accelerator Runtime Prototype
 *
 * Real ONNX initializer tensor loader.
 *
 * Author: Santosh Kumar
 * Copyright 2026 Santosh Kumar
 * SPDX-License-Identifier: Apache-2.0
 */

#include "waveaccel/tensor_store.hpp"

#include <array>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace waveaccel {

namespace {

constexpr std::array<char, 8> kWeightsMagic{
    'W', 'A', 'W', 'G', 'T', '0', '0', '1'
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
            "unexpected end of binary tensor file"
        );
    }

    return value;
}

std::string read_string(
    std::ifstream& input,
    std::uint32_t length
) {
    if (length == 0 || length > 1U << 20U) {
        throw std::runtime_error(
            "invalid tensor-name length"
        );
    }

    std::string value(length, '\0');

    input.read(value.data(), length);

    if (!input) {
        throw std::runtime_error(
            "unexpected end of tensor name"
        );
    }

    return value;
}

}  // namespace

TensorStore TensorStore::load(
    const std::string& path
) {
    std::ifstream input(
        path,
        std::ios::binary
    );

    if (!input) {
        throw std::runtime_error(
            "unable to open tensor file: " + path
        );
    }

    std::array<char, 8> magic{};

    input.read(
        magic.data(),
        static_cast<std::streamsize>(magic.size())
    );

    if (!input || magic != kWeightsMagic) {
        throw std::runtime_error(
            "invalid WaveAccel tensor-file magic"
        );
    }

    const auto version =
        read_scalar<std::uint32_t>(input);

    if (version != kFormatVersion) {
        throw std::runtime_error(
            "unsupported WaveAccel tensor-file version"
        );
    }

    const auto tensor_count =
        read_scalar<std::uint32_t>(input);

    TensorStore store;

    for (
        std::uint32_t tensor_index = 0;
        tensor_index < tensor_count;
        ++tensor_index
    ) {
        const auto name_length =
            read_scalar<std::uint32_t>(input);

        TensorData tensor;
        tensor.name =
            read_string(input, name_length);

        const auto rank =
            read_scalar<std::uint32_t>(input);

        if (rank == 0 || rank > 16) {
            throw std::runtime_error(
                "unsupported tensor rank for " +
                tensor.name
            );
        }

        tensor.shape.reserve(rank);

        std::uint64_t expected_values = 1;

        for (
            std::uint32_t dim_index = 0;
            dim_index < rank;
            ++dim_index
        ) {
            const auto dim =
                read_scalar<std::uint64_t>(input);

            if (dim == 0) {
                throw std::runtime_error(
                    "zero tensor dimension for " +
                    tensor.name
                );
            }

            if (
                expected_values >
                std::numeric_limits<std::uint64_t>::max() /
                    dim
            ) {
                throw std::runtime_error(
                    "tensor element-count overflow"
                );
            }

            expected_values *= dim;
            tensor.shape.push_back(dim);
        }

        const auto value_count =
            read_scalar<std::uint64_t>(input);

        if (value_count != expected_values) {
            throw std::runtime_error(
                "tensor shape/value-count mismatch for " +
                tensor.name
            );
        }

        if (
            value_count >
            static_cast<std::uint64_t>(
                std::numeric_limits<std::size_t>::max()
            )
        ) {
            throw std::runtime_error(
                "tensor is too large for this process"
            );
        }

        tensor.values.resize(
            static_cast<std::size_t>(value_count)
        );

        input.read(
            reinterpret_cast<char*>(
                tensor.values.data()
            ),
            static_cast<std::streamsize>(
                tensor.values.size() * sizeof(float)
            )
        );

        if (!input) {
            throw std::runtime_error(
                "unexpected end of tensor values for " +
                tensor.name
            );
        }

        const auto inserted =
            store.tensors_.emplace(
                tensor.name,
                std::move(tensor)
            );

        if (!inserted.second) {
            throw std::runtime_error(
                "duplicate tensor name in tensor file"
            );
        }
    }

    return store;
}

const TensorData& TensorStore::at(
    const std::string& name
) const {
    const auto found = tensors_.find(name);

    if (found == tensors_.end()) {
        throw std::runtime_error(
            "tensor store does not contain: " + name
        );
    }

    return found->second;
}

}  // namespace waveaccel
