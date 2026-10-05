/*
 * WaveAccel - AI Accelerator Runtime Prototype
 *
 * Real float32 numerical inference demonstration.
 *
 * Author: Santosh Kumar
 * Copyright 2026 Santosh Kumar
 * SPDX-License-Identifier: Apache-2.0
 *
 * This executable performs real C++ numerical computation with the trained
 * ONNX initializer values. It does not report simulated accelerator timing.
 */

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include "waveaccel/model_memory.hpp"
#include "waveaccel/numerical_executor.hpp"
#include "waveaccel/tensor_store.hpp"

namespace {

float max_abs_error(
    const std::vector<float>& lhs,
    const std::vector<float>& rhs
) {
    if (lhs.size() != rhs.size()) {
        throw std::runtime_error(
            "cannot compare vectors with different sizes"
        );
    }

    float error = 0.0F;

    for (
        std::size_t index = 0;
        index < lhs.size();
        ++index
    ) {
        error = std::max(
            error,
            std::fabs(lhs[index] - rhs[index])
        );
    }

    return error;
}

void print_pair(
    const char* label,
    const std::vector<float>& values
) {
    std::cout
        << label
        << " [";

    for (
        std::size_t index = 0;
        index < values.size();
        ++index
    ) {
        if (index != 0) {
            std::cout << ", ";
        }

        std::cout
            << std::fixed
            << std::setprecision(9)
            << values[index];
    }

    std::cout << "]\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 4) {
            std::cerr
                << "usage: waveaccel_numerical_demo "
                << "<memory-manifest> "
                << "<weights-file> "
                << "<sample-file>\n";

            return 2;
        }

        const auto model =
            waveaccel::load_model_memory_manifest(
                argv[1]
            );

        const auto tensors =
            waveaccel::TensorStore::load(
                argv[2]
            );

        const auto sample =
            waveaccel::load_numerical_sample(
                argv[3]
            );

        const waveaccel::NumericalExecutor executor(
            model,
            tensors
        );

        const auto prediction =
            executor.run(sample.input);

        print_pair(
            "ground truth:        ",
            sample.ground_truth
        );

        print_pair(
            "ONNX Runtime ref:    ",
            sample.reference_output
        );

        print_pair(
            "WaveAccel C++:       ",
            prediction
        );

        const float error =
            max_abs_error(
                prediction,
                sample.reference_output
            );

        std::cout
            << "max abs parity error: "
            << std::scientific
            << error
            << '\n';

        return 0;
    } catch (const std::exception& error) {
        std::cerr
            << "waveaccel_numerical_demo error: "
            << error.what()
            << '\n';

        return 1;
    }
}
