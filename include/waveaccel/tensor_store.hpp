/*
 * WaveAccel - AI Accelerator Runtime Prototype
 *
 * Real ONNX initializer tensor storage for numerical execution.
 *
 * Author: Santosh Kumar
 * Copyright 2026 Santosh Kumar
 * SPDX-License-Identifier: Apache-2.0
 *
 * The binary tensor file is generated from the exported ONNX model by
 * python/export_numerical_fixture.py. Tensor values are real trained float32
 * initializer values, not simulated data.
 */

#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace waveaccel {

struct TensorData {
    std::string name;
    std::vector<std::uint64_t> shape;
    std::vector<float> values;
};

class TensorStore {
public:
    static TensorStore load(const std::string& path);

    const TensorData& at(const std::string& name) const;

    std::size_t size() const noexcept {
        return tensors_.size();
    }

private:
    std::unordered_map<std::string, TensorData> tensors_;
};

}  // namespace waveaccel
