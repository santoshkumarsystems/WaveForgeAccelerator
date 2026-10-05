# WaveForge Accelerator Engineering Decisions

WaveForge Accelerator is an experimental **end-to-end AI accelerator compiler/runtime project**.

These decisions explain why the project was built in the order it was, how correctness is protected across the AI model/compiler/runtime stack, and which capabilities are intentionally real versus simulated.

> **Engineering principle:** Correctness before optimization. Visibility before abstraction. Hardware constraints before compiler automation.

---

## 1. Train the AI model from scratch

**Problem:** Starting from a downloaded pretrained model would hide the model-building stage and weaken the end-to-end systems story.

**Why it matters:** WaveForge Accelerator is intended to demonstrate the full path from physical data generation through AI model training, portable model export, compiler integration, and accelerator-style runtime execution.

**Decision:** Train a compact neural network from scratch in PyTorch using deterministic data produced by cuWaves.

**Implementation:**

```text
cuWaves physical simulation
  ↓
deterministic dataset generation
  ↓
10,000 examples
  ↓
64 spatial samples per example
  ↓
PyTorch WaveRegressor
  ↓
training from scratch
  ↓
trained amplitude + wavelength predictor
```

**Validation:**

```text
2,642 learned FP32 parameters
validation MAE:
  amplitude:  0.005691
  wavelength: 0.005026
```

**Current limitation:** The current neural network is intentionally compact so the full compiler/runtime path remains visible and explainable.

**Next step:** Larger graphs or additional model families can be added after the compiler/runtime foundation is stable.

---

## 2. Keep cuWaves independent

**Problem:** Coupling the physics project to the accelerator project would make both harder to evolve.

**Why it matters:** cuWaves should remain a standalone numerical/physics project while WaveForge Accelerator consumes it as a deterministic physical-data source.

**Decision:** Keep the dependency direction one-way:

```text
cuWaves → WaveForge Accelerator
```

**Implementation:** WaveForge Accelerator can use a local cuWaves checkout during development or fetch the pinned public release through CMake.

**Validation:** The public build path successfully fetched cuWaves `v0.0.1` and passed the full C++ regression suite.

**Current limitation:** The current dataset is a single-wave spatial regression problem.

**Next step:** Extend the data contract only when a new model/runtime experiment requires it.

---

## 3. Preserve real learned weights and real numerical execution

**Problem:** A scheduling-only simulator can look convincing while never executing the trained model.

**Why it matters:** Runtime and compiler behavior should be connected to an actual trained neural network rather than synthetic placeholder math.

**Decision:** Preserve real trained weights through ONNX and execute real FP32 operations in C++.

**Implementation:**

```text
PyTorch trained weights
  ↓
ONNX initializers
  ↓
WaveForge model-memory artifacts
  ↓
C++ GEMM / ReLU / Mul / Add
```

**Validation:** C++ output matches ONNX Runtime within float32 tolerance.

```text
max absolute error:
2.384185791e-07
```

**Current limitation:** The numerical backend supports the current operator set only.

**Next step:** Add operators when required by a larger model rather than broadening support speculatively.

---

## 4. Use real inference runtimes as external references

**Problem:** A custom accelerator runtime needs independent numerical reference points.

**Why it matters:** Comparing against established runtimes proves that the exported model remains portable and numerically valid.

**Decision:** Validate the same ONNX model with:

```text
ONNX Runtime
OpenVINO
TensorRT
```

**Implementation:**

- ONNX Runtime provides the main numerical baseline.
- OpenVINO provides a real local CPU inference path.
- TensorRT provides a real NVIDIA Tesla T4 reference path.

**Validation:**

```text
PyTorch ↔ ONNX Runtime max abs error:
2.38418579e-07

TensorRT ↔ ONNX Runtime max abs error:
0
```

**Current limitation:** TensorRT is a reference runtime, not the WaveForge backend.

**Next step:** None required for the current milestone; keep TensorRT intentionally small and reproducible.

---

## 5. Correctness before optimization

**Problem:** Accelerator projects become difficult to debug when scheduling, memory movement, compiler transformations, tiling, and optimized math are introduced simultaneously.

**Why it matters:** A compiler-generated plan is only meaningful if it can be compared with a known-correct implementation.

**Decision:** Validate real numerical inference, manual memory planning, residency, tiling, and scheduling before automating those decisions through TVM.

**Implementation:**

```text
manual graph/runtime path
  ↓
validated correctness
  ↓
compiler-generated plan
  ↓
manual/compiler parity
```

**Validation:**

```text
compiler_manual_plan_parity
numerical_parity_smoke
runtime_numerical_integration_smoke
```

**Current limitation:** The parity contract is validated for the current model and target configuration.

**Next step:** Re-run the same parity methodology when expanding model/operator coverage.

---

## 6. Visibility before abstraction

**Problem:** Hiding accelerator behavior behind frameworks too early makes memory and scheduling decisions difficult to explain.

**Why it matters:** The project is intended to expose the responsibilities normally hidden inside an AI accelerator runtime.

**Decision:** Make graph nodes, byte counts, SRAM decisions, tiles, logical addresses, commands, residency, and profiler events explicit.

**Implementation:**

```text
ExecutionPlan
CompiledRuntimeSchedule
DeviceMemoryMap
DeviceCommandStream
MockDevice
Profiler
```

**Validation:** Dedicated smoke tests validate each layer independently.

**Current limitation:** The explicit design is intentionally more verbose than a production runtime abstraction.

**Next step:** Introduce additional abstraction only after preserving equivalent observability.

---

## 7. Hardware constraints before compiler automation

**Problem:** A compiler transformation is not meaningful if it does not encode a concrete target constraint.

**Why it matters:** WaveForge should demonstrate why a compiler chooses a tile size, not merely show that tiling happened.

**Decision:** Derive GEMM tiling from modeled SRAM capacity.

**Implementation:**

```text
SRAM                  4096 B
transient reserve      392 B
initializer capacity  3704 B
bias                    128 B
weight tile capacity  3576 B
row size                256 B
tile width               13
```

Compiler result:

```text
13 + 13 + 6 output-channel tiles
```

**Validation:** Compiler policy and runtime independently validate the same geometry.

**Current limitation:** This is output-channel row tiling, not a full tensor-core-style M/N/K tiling system.

**Next step:** Explore deeper TensorIR scheduling and tensorization only when the target model requires it.

---

## 8. Model an explicit memory hierarchy

**Problem:** A runtime that treats all memory as one flat array cannot demonstrate accelerator memory-management responsibilities.

**Why it matters:** Modern accelerator runtimes must reason about movement, capacity, residency, and staging.

**Decision:** Model:

```text
Host DRAM
  ↓
Logical Device Memory
  ↓
Logical SRAM
  ↓
Compute
```

**Implementation:** The model uses deterministic logical offsets in a contiguous initializer image.

Current initializer image:

```text
10,584 bytes
```

**Validation:**

```text
model_memory_smoke
memory_planner_smoke
device_memory_address_smoke
device_address_contract_smoke
```

**Current limitation:** These are logical model addresses, not real hardware addresses.

**Next step:** Real device memory / PCIe / DMA integration belongs to a later hardware-facing milestone.

---

## 9. Model cold/warm residency explicitly

**Problem:** Re-loading model state on every request hides one of the most important runtime memory-management decisions.

**Why it matters:** Weight residency changes both memory traffic and runtime behavior.

**Decision:** Model cold and warm execution paths.

**Implementation:**

```text
16 KiB SRAM:
  model fits
  warm path keeps model resident

4 KiB SRAM:
  model does not fit
  weights are staged/tiled per inference
```

**Validation:** Runtime accounting verifies the expected Device → SRAM behavior for both configurations.

**Current limitation:** Residency is modeled in software, not measured on physical accelerator SRAM.

**Next step:** Extend residency policy experiments when adding concurrent execution.

---

## 10. Logical DMA-style movement, not fake hardware claims

**Problem:** Explicit data movement is essential for accelerator-runtime modeling, but software-only transfers can be misrepresented as real PCIe/DMA behavior.

**Why it matters:** The project should demonstrate the control flow without making hardware claims it cannot support.

**Decision:** Represent Device Memory → SRAM movement as logical DMA-style commands.

**Implementation:**

```text
DmaDeviceToSram
device_offset_bytes
sram_offset_bytes
bytes
```

**Validation:** MockDevice rejects invalid device/SRAM ranges.

**Current limitation:** No real PCIe controller, DMA engine, IOVA, BAR, or physical accelerator is involved.

**Next step:** Real PCIe/DMA integration is a future hardware-facing milestone or can be demonstrated separately in low-level driver projects.

---

## 11. Target contract is authoritative

**Problem:** Duplicating target assumptions across Python compiler scripts and C++ runtime code can cause silent divergence.

**Why it matters:** Compiler and runtime must agree on dtype, supported operations, memory spaces, and capacity rules.

**Decision:** Define one compiler-side target contract and transport its semantics into C++.

**Current contract:**

```text
target:           waveaccel-sim
dtype:            float32
element bytes:    4
supported ops:    Gemm, Relu, Mul, Add
memory hierarchy: device_memory → sram
tiling axis:      output_channel
tile unit:        complete_output_weight_row
```

**Validation:** Runtime rejects incompatible dtype, memory-space semantics, operator contracts, or capacity assumptions.

**Current limitation:** The contract is intentionally scoped to the current accelerator model.

**Next step:** Extend the contract only when new capabilities are implemented.

---

## 12. BYOC selects the graph; WaveForge owns backend lowering

**Problem:** TVM partitioning and the C++ runtime could become two disconnected demonstrations.

**Why it matters:** Compiler-selected subgraphs need to map to the runtime plan that actually executes them.

**Decision:** Validate an exact mapping from TVM `Codegen="waveaccel"` partitions to WaveForge `ExecutionPlan` nodes.

**Implementation:**

```text
P0 waveaccel.matmul_bias_relu → [0, 1]
P1 waveaccel.matmul_bias_relu → [2, 3]
P2 waveaccel.matmul_bias      → [4]
P3 waveaccel.multiply         → [5]
P4 waveaccel.add              → [6]
```

**Validation:** Five BYOC partitions cover all seven current graph/plan nodes.

**Current limitation:** WaveForge currently uses a BYOC → ExecutionPlan bridge rather than TVM `RunCodegen`.

**Next step:** Implement a real TVM `RunCodegen` backend as the next compiler-depth milestone.

---

## 13. Keep RunCodegen as a future milestone

**Problem:** Claiming a native external-codegen backend before implementing it would overstate the project.

**Why it matters:** The current BYOC integration is already substantial and should be described precisely.

**Decision:** Clearly separate completed compiler work from future native TVM codegen work.

**Implemented now:**

```text
ONNX → Relax
PrimFunc / TIR inspection
target contract
SRAM-aware compiler policy
BYOC partitioning
BYOC → ExecutionPlan bridge
integrated compiler plan
```

**Not yet implemented:**

```text
TVM RunCodegen backend
native TVM TargetKind
custom TVM runtime Module
```

**Next step:** Add `RunCodegen` only after the current release is stable.

---

## 14. DeviceCommandStream is the runtime/device boundary

**Problem:** Calling numerical functions directly from graph nodes would hide the runtime-to-device interface.

**Why it matters:** Accelerator runtimes normally lower scheduled work into device-facing commands.

**Decision:** Introduce an explicit `DeviceCommandStream`.

**Current commands:**

```text
DmaDeviceToSram
GemmTile
Gemm
Relu
Mul
Add
```

**Validation:**

```text
device_command_stream_smoke
mockdevice_device_command_smoke
compiler_runtime_direct_devicecommand_smoke
```

**Current limitation:** Commands represent a generic accelerator interface, not a proprietary ISA.

**Next step:** Extend command semantics if future asynchronous/concurrent execution requires it.

---

## 15. Enforce address and SRAM contracts at the backend

**Problem:** A compiler/runtime system can silently produce invalid memory commands if the backend accepts arbitrary offsets.

**Why it matters:** The device boundary should independently enforce memory safety assumptions.

**Decision:** `MockDevice` validates source ranges, SRAM ranges, staging capacity, and target constraints.

**Validation:** `device_address_contract_smoke` proves invalid accesses are rejected.

**Current limitation:** Bounds are enforced in a software backend rather than hardware.

**Next step:** Preserve the same contract if a real device backend is added later.

---

## 16. Treat WaveForge accelerator timing as simulation

**Problem:** Generic latency/bandwidth constants can be mistaken for measured hardware performance.

**Why it matters:** OpenVINO and TensorRT provide real measured reference timings, while WaveForge currently has no physical accelerator.

**Decision:** Explicitly label:

```text
Host/Device bandwidth
Device/SRAM bandwidth
dispatch latency
operator timing
```

as simulated.

**Validation:** README and architecture documentation separate real runtime measurements from simulated accelerator timing.

**Current limitation:** No WaveForge physical hardware exists.

**Next step:** Only report real WaveForge performance if a real backend/hardware target is implemented.

---

## 17. TensorRT is a reference, not the project

**Problem:** Deep TensorRT optimization could distract from the compiler/runtime implementation.

**Why it matters:** TensorRT is useful to prove the ONNX model runs correctly on real NVIDIA hardware, but it should not become the central implementation.

**Decision:** Keep the TensorRT path intentionally small.

**Validation:** Tesla T4 output matches ONNX Runtime exactly for the recorded sample.

**Current limitation:** Serialized TensorRT engines are GPU/environment dependent and are not committed.

**Next step:** Rebuild the engine on the target NVIDIA environment when needed.

---

## 18. Test every architectural layer

**Problem:** A single end-to-end test can pass while hiding failures in individual layers.

**Why it matters:** WaveForge exposes many distinct contracts: model memory, SRAM planning, numerical execution, compiler/runtime integration, command lowering, and backend memory safety.

**Decision:** Maintain explicit tests for each layer.

Validated C++ suite:

```text
1  runtime_smoke
2  model_memory_smoke
3  memory_planner_smoke
4  numerical_parity_smoke
5  runtime_numerical_integration_smoke
6  compiler_driven_runtime_smoke
7  compiler_manual_plan_parity
8  device_command_stream_smoke
9  mockdevice_device_command_smoke
10 compiler_runtime_direct_devicecommand_smoke
11 device_memory_address_smoke
12 device_address_contract_smoke
13 target_contract_transport_smoke
```

**Validation:**

```text
13/13 tests passed
100% tests passed
```

**Current limitation:** Coverage is intentionally focused on the current model/runtime path.

**Next step:** Add tests alongside every new architectural capability.

---

## 19. Keep the public build reproducible

**Problem:** A project that only builds against the author's local dependency tree is difficult to evaluate.

**Why it matters:** A public GitHub user should not need a manually prepared sibling cuWaves checkout.

**Decision:** Support both:

```text
local development:
  -DWAVEACCEL_CUWAVES_SOURCE=../cuwaves

public build:
  CMake FetchContent → cuWaves v0.0.1
```

**Validation:** The public FetchContent path configured, built, and passed all 13 C++ tests without a local cuWaves override.

**Current limitation:** A literal post-push fresh-clone validation remains the final release check.

**Next step:** Clone the public repository into a clean directory and repeat the public build/test path after the initial push.

---

## 20. Preserve truthful scope

**Problem:** AI accelerator/compiler projects can become keyword-heavy and accidentally claim features that are only planned.

**Why it matters:** The portfolio should be technically strong and fully defensible.

**Decision:** Market completed capabilities clearly while keeping future work explicit.

**Current completed capabilities include:**

```text
AI model training from scratch
cuWaves-generated physical data
ONNX / ONNX Runtime
OpenVINO
TensorRT
TVM Relax / PrimFunc / TIR analysis
TVM BYOC
task scheduling
memory management
cold/warm residency
SRAM-aware tiling
logical DMA-style movement
ExecutionPlan
CompiledRuntimeSchedule
DeviceMemoryMap
DeviceCommandStream
MockDevice
real C++ numerical execution
profiling
13/13 layered tests
```

**Future capabilities include:**

```text
TVM RunCodegen
native TVM TargetKind
custom TVM runtime Module
concurrent scheduling
async command queues
pipelining
double buffering
quantization
richer fusion
LLVM / MLIR experiments
real PCIe / DMA
real accelerator driver/runtime
hardware-specific ISA
```

Future items are not presented as current implementation claims.
