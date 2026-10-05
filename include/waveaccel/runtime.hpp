/*
 * WaveAccel - AI Accelerator Runtime Prototype
 *
 * Graph-driven mock accelerator runtime with real numerical inference.
 *
 * Author: Santosh Kumar
 * Copyright 2026 Santosh Kumar
 * SPDX-License-Identifier: Apache-2.0
 *
 * One runtime invocation now combines:
 *
 *   ONNX-derived graph metadata
 *   + tensor metadata
 *   + SRAM memory planning
 *   + simulated Host/Device/SRAM data movement
 *   + simulated operator dispatch timing
 *   + real float32 inference using trained ONNX initializer values
 *
 * NOTE:
 * Numerical predictions are real computations. Accelerator timing, bandwidth,
 * SRAM behavior, and DMA/data movement remain generic simulations and do not
 * reproduce proprietary commercial accelerator implementations.
 */

#pragma once

#include <optional>
#include <vector>

#include "waveaccel/device.hpp"
#include "waveaccel/execution_plan.hpp"
#include "waveaccel/memory_planner.hpp"
#include "waveaccel/model_memory.hpp"
#include "waveaccel/profiler.hpp"
#include "waveaccel/task.hpp"
#include "waveaccel/tensor_store.hpp"

namespace waveaccel {

struct RuntimeExecutionResult {
    Profiler profiler;
    std::vector<float> output;
};

class Runtime {
public:
    Runtime(
        ModelMemoryInfo model,
        MockDevice device = MockDevice{}
    );

    std::vector<Task> build_plan() const;

    // Install a compiler-produced execution plan. Once installed, build_plan()
    // uses compiler decisions instead of independently deriving tiled/staged
    // node scheduling from MemoryPlanner.
    void set_execution_plan(ExecutionPlan plan);

    bool compiler_plan_enabled() const noexcept {
        return execution_plan_.has_value();
    }

    // Control-plane-only path retained for scheduling/memory smoke tests.
    Profiler run_once();

    // Integrated path: execute the same scheduled compute nodes numerically
    // with the real trained initializer values.
    RuntimeExecutionResult run_once(
        const std::vector<float>& input,
        const TensorStore& tensors
    );

    const ModelMemoryInfo& model() const noexcept {
        return model_;
    }

    const MemoryPlan& memory_plan() const noexcept {
        return memory_plan_;
    }

    bool model_in_device_memory() const noexcept {
        return device_.model_in_device_memory();
    }

    bool model_in_sram() const noexcept {
        return device_.model_in_sram();
    }

private:
    Task build_compute_task(
        std::size_t node_index
    ) const;

    ModelMemoryInfo model_;
    MockDevice device_;
    MemoryPlan memory_plan_;

    // Optional compiler authority. The manual MemoryPlan remains available as
    // the independently validated reference/fallback path.
    std::optional<ExecutionPlan> execution_plan_;
};

}  // namespace waveaccel
