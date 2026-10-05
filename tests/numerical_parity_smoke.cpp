/*
 * WaveAccel - AI Accelerator Runtime Prototype
 *
 * Numerical parity test against an ONNX Runtime reference prediction.
 *
 * Author: Santosh Kumar
 * Copyright 2026 Santosh Kumar
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

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
            "parity vectors have different sizes"
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

}  // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr
            << "usage: numerical_parity_smoke "
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

    assert(prediction.size() == 2);
    assert(sample.reference_output.size() == 2);

    const float error =
        max_abs_error(
            prediction,
            sample.reference_output
        );

    // Float32 CPU execution order can differ slightly from ONNX Runtime.
    // A 1e-5 bound is tight enough to prove numerical parity for this model.
    assert(error <= 1.0e-5F);

    std::cout
        << "WaveAccel C++ numerical parity: PASS\n"
        << "max_abs_error="
        << error
        << '\n';

    return 0;
}
