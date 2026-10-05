# WaveForge Accelerator

**WaveForge Accelerator** is an experimental **end-to-end AI accelerator compiler/runtime project** covering **AI model training — a neural network trained from scratch in PyTorch** on deterministic cuWaves-generated physical data, followed by **ONNX export, ONNX Runtime, OpenVINO, TensorRT, Apache TVM BYOC, task scheduling, memory management, SRAM-aware tensor tiling, logical DMA-style data movement, kernel execution, profiling, and a C++ accelerator runtime**.

It demonstrates the complete path from **physical simulation → dataset generation → AI model training from scratch → trained neural-network weights → ONNX portability → real inference runtimes → TVM compiler partitioning and SRAM-aware planning → C++ ExecutionPlan → CompiledRuntimeSchedule → DeviceMemoryMap → DeviceCommandStream → MockDevice → real numerical execution**.

The project is designed to make core AI-system responsibilities visible and testable: **model training, graph execution, task scheduling, memory hierarchy and residency, tensor staging and tiling, kernel dispatch, data movement, compiler/runtime contracts, numerical correctness, and profiling**.

> **Engineering principle:** Correctness before optimization. Visibility before abstraction. Hardware constraints before compiler automation.

> **Naming note:** The public project name is **WaveForge Accelerator**. Some internal implementation identifiers intentionally remain `waveaccel`, `waveaccel-sim`, and `Codegen="waveaccel"` because they are established compiler/runtime interfaces in the current codebase.

---

## Why this project exists

Most inference demos stop at:

```text
model → ONNX → runtime → prediction
```

WaveForge Accelerator goes lower:

```text
model
  ↓
ONNX
  ↓
graph/compiler decisions
  ↓
memory planning
  ↓
transfers / residency
  ↓
command lowering
  ↓
accelerator-style runtime
  ↓
real numerical execution
```

The goal is not to emulate a proprietary accelerator. The goal is to make the responsibilities of an AI accelerator compiler/runtime visible and testable.

WaveForge Accelerator is a **generic accelerator/runtime model**. It does not emulate proprietary EnCharge AI hardware, NVIDIA hardware, or any other commercial accelerator.

---

## End-to-end architecture

```text
cuWaves C++ physics model
        ↓
dataset generation
        ↓
PyTorch training
        ↓
ONNX
        ↓
ONNX Runtime / OpenVINO / TensorRT references
        ↓
TVM Relax
        ↓
WaveForge Accelerator target contract
        ↓
TVM BYOC partitioning
        ↓
SRAM-aware compiler plan
        ↓
C++ ExecutionPlan transport
        ↓
DeviceMemoryMap
        ↓
DeviceCommandStream
        ↓
MockDevice
        ↓
real C++ numerical execution
```

See:

- `docs/ARCHITECTURE.md`
- `docs/design_decisions.md`
- `docs/validation_matrix.md`

---

## Data lineage and model task

WaveForge Accelerator does not invent an unrelated toy dataset.

The dataset is generated from the independent **cuWaves** C++ numerical project.

The underlying 1D wave form is:

```text
F(x,t) = A sin(kx - wt + phi)
```

with:

```text
k = 2π / λ
```

The current dataset uses a spatial snapshot at fixed time and contains **64 spatial field samples** per example.

The model predicts:

1. amplitude
2. wavelength

Why wavelength rather than frequency?

For a single spatial snapshot at fixed time, wavelength is directly encoded in the spatial oscillation. Temporal frequency is not uniquely identifiable without multiple time samples or an additional physical relationship.

---

## Reference model

```text
64 inputs
  ↓
Linear 64 → 32
  ↓
ReLU
  ↓
Linear 32 → 16
  ↓
ReLU
  ↓
Linear 16 → 2
  ↓
denormalization
```

The trained network has:

```text
2,642 learned FP32 parameters
```

The ONNX graph also contains four FP32 normalization constants.

The exported graph is:

```text
Gemm → Relu → Gemm → Relu → Gemm → Mul → Add
```

---

## Real vs simulated behavior

WaveForge Accelerator deliberately separates real numerical work from simulated accelerator behavior.

### Real

- cuWaves C++ physics/data generation
- PyTorch training
- ONNX model and learned weights
- ONNX Runtime inference
- OpenVINO CPU inference
- TensorRT inference on a real NVIDIA Tesla T4
- TVM Relax/TIR transformations
- compiler-derived SRAM tiling decisions
- C++ GEMM / ReLU / Mul / Add execution
- numerical parity checks

### Simulated

- Host ↔ Device bandwidth
- Device Memory ↔ SRAM bandwidth
- accelerator dispatch latency
- accelerator compute timing
- logical device-memory addresses
- logical SRAM behavior

Simulated timing is never presented as measured hardware performance.

---

## Quick start: public C++ build

WaveForge Accelerator can fetch the pinned cuWaves dependency automatically.

Validated public dependency path:

```text
WaveForge Accelerator
  ↓ FetchContent
cuWaves v0.0.1
  ↓
configure
  ↓
build
  ↓
13/13 C++ tests
```

From the repository root:

```bash
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
```

The public FetchContent path is pinned to:

```text
cuWaves v0.0.1
```

A local cuWaves checkout can be used during development:

```bash
cmake -S . -B build \
  -DWAVEACCEL_CUWAVES_SOURCE=../cuwaves
```

---

## Python environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On systems where the command is `python3`, use:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Dataset contract

The generated dataset is expected at:

```text
data/raw/cuwaves_wave_dataset.csv
```

The current deterministic dataset contains:

```text
10,000 examples
64 spatial samples per example
targets: amplitude, wavelength
```

The dataset is regenerable and is not treated as the source of truth for the project.

---

## AI model training — neural network trained from scratch in PyTorch

WaveForge Accelerator does not rely on a downloaded pretrained model. The compact `WaveRegressor` neural network is initialized and trained from scratch in PyTorch using the deterministic cuWaves-generated dataset.


```bash
python python/train.py \
  --dataset data/raw/cuwaves_wave_dataset.csv \
  --checkpoint artifacts/wave_model.pt
```

Validated training configuration:

```text
seed:       7
train/val:  8000 / 2000
batch:      128
optimizer:  Adam
lr:         1e-3
epochs:     200
```

Recorded validation MAE:

```text
amplitude:  0.005691
wavelength: 0.005026
```

---

## Export ONNX

```bash
python python/export_onnx.py \
  --checkpoint artifacts/wave_model.pt \
  --output artifacts/wave_model.onnx
```

The exported model uses ONNX external initializer data, so keep these files together:

```text
artifacts/wave_model.onnx
artifacts/wave_model.onnx.data
```

---

## Validate ONNX Runtime

```bash
python python/validate_onnx.py \
  --dataset data/raw/cuwaves_wave_dataset.csv \
  --checkpoint artifacts/wave_model.pt \
  --onnx artifacts/wave_model.onnx
```

Recorded PyTorch ↔ ONNX Runtime maximum absolute difference:

```text
2.38418579e-07
```

---

## OpenVINO reference

Probe available OpenVINO devices:

```bash
python python/openvino_probe.py
```

OpenVINO is used as a real local CPU reference runtime.

Recorded CPU inference timing for the current model:

```text
median: 0.017511 ms
mean:   0.020187 ms
```

These are real measurements on the local CPU environment used for validation.

---

## TensorRT reference

TensorRT is intentionally a small external reference, not the WaveForge Accelerator backend itself.

Validated Google Colab environment:

```text
GPU:             Tesla T4
VRAM:            15,360 MiB
TensorRT:        11.3.0.99
CUDA:            13.0
PyTorch:         2.11.0+cu130
ONNX Runtime:    1.30.0
```

Run:

```bash
python python/tensorrt_reference.py \
  --onnx artifacts/wave_model.onnx \
  --engine artifacts/wave_model.trt.engine \
  --warmup 50 \
  --runs 200
```

Recorded numerical parity:

```text
ONNX Runtime:
[[2.1766586303710938, 1.7368863821029663]]

TensorRT:
[[2.1766586303710938, 1.7368863821029663]]

max absolute error: 0
```

Recorded TensorRT GPU execution timing:

```text
median: 0.030816 ms
mean:   0.032567 ms
p95:    0.043565 ms
```

These are real TensorRT measurements on the reported T4. They are **not** compared against WaveForge Accelerator simulated timing as a hardware-performance claim.

Serialized TensorRT engine files are intentionally not committed:

```text
*.trt.engine
*.engine
```

See `docs/tensorrt_reference.md`.

---

## SRAM-aware compilation

For the validated 4 KiB target configuration:

```text
SRAM capacity:                4096 B
Transient reservation:        392 B
Initializer staging capacity: 3704 B
```

The first GEMM requires:

```text
weight: 8192 B
bias:    128 B
```

The compiler keeps the bias resident and tiles complete output-weight rows:

```text
row size:                  256 B
available for weight tile: 3576 B
floor(3576 / 256):          13 rows
```

Result:

```text
GEMM 0 output tiles: 13 + 13 + 6
weight bytes:         3328 + 3328 + 1536
```

The later GEMMs fit whole in the staging region.

This is real row-aligned output-channel decomposition. It is not a claim of tensor-core-style M/N/K tiling.

---

## TVM compiler integration

WaveForge Accelerator uses TVM 0.27.0.post1 for the compiler-side path.

Validated stages:

```text
ONNX
  ↓
Relax
  ↓
LegalizeOps / PrimFuncs
  ↓
WaveForge Accelerator target contract
  ↓
SRAM-aware scheduling
  ↓
ConvertToDataflow
  ↓
FuseOpsByPattern
  ↓
Codegen="waveaccel" partitions
```

The current model partitions into five WaveForge Accelerator BYOC regions:

```text
waveaccel.matmul_bias_relu × 2
waveaccel.matmul_bias      × 1
waveaccel.multiply         × 1
waveaccel.add              × 1
```

Those five partitions map exactly, in topological order, to all seven graph / ExecutionPlan nodes.

WaveForge Accelerator currently demonstrates **BYOC partitioning and compiler/runtime plan integration**.

It does not claim:

- a native TVM `TargetKind`
- proprietary code generation
- physical accelerator machine code
- `RunCodegen` integration
- a custom TVM runtime module

---

## Compiler/runtime target contract

The compiler-side target contract defines:

```text
target:              waveaccel-sim
schema:              1
dtype:               float32
element size:        4 bytes
supported ops:       Gemm, Relu, Mul, Add
memory hierarchy:    device_memory → sram
GEMM tiling axis:    output_channel
tile unit:           complete_output_weight_row
```

The same contract semantics are transported into C++.

The runtime rejects incompatible:

- dtype
- logical memory-space assumptions
- unsupported operations
- SRAM/staging capacity contracts

---

## Accelerator backend model

The C++ backend uses a logical memory hierarchy:

```text
Host DRAM
   ↓
Device Memory
   ↓
SRAM
   ↓
compute
```

Model initializers are assigned deterministic logical offsets in a contiguous device-memory image.

Current initializer image:

```text
10,584 bytes
```

The runtime lowers work into `DeviceCommandStream` operations such as:

```text
DMA Device → SRAM
GEMM tile
GEMM
ReLU
Mul
Add
```

The `MockDevice` validates:

- device source range
- initializer staging range
- SRAM capacity
- staging capacity
- command contract

The logical offsets are **not** physical PCIe, MMIO, IOVA, BAR, or proprietary accelerator addresses.

---

## Cold/warm residency model

At 16 KiB SRAM:

```text
model state fits in SRAM
cold inference: stage model
warm inference: keep model resident
```

At 4 KiB SRAM:

```text
model state does not fit in SRAM
weights are staged/tiled per inference
```

This makes residency and memory pressure explicit without pretending to model proprietary hardware.

---

## Numerical validation

WaveForge Accelerator executes real float32 math using the trained model weights.

Recorded reference:

```text
truth:
[2.166207790, 1.727644563]

ONNX Runtime:
[2.172150850, 1.733266115]

WaveForge Accelerator C++:
[2.172151089, 1.733266234]
```

Maximum C++ ↔ ONNX Runtime absolute error:

```text
2.384185791e-07
```

The validated 16 KiB and 4 KiB paths produce the same numerical result.

---

## Validation by architectural layer

WaveForge Accelerator does not treat testing as one flat checklist. Each layer has a test that proves a specific contract.

### Runtime foundation

```text
runtime_smoke
```

Validates basic runtime construction and execution behavior.

### Model / initializer memory

```text
model_memory_smoke
```

Validates model initializer parsing, tensor metadata, and byte accounting.

### SRAM planning

```text
memory_planner_smoke
```

Validates transient reservation, staging capacity, and SRAM-aware planning.

### Numerical backend

```text
numerical_parity_smoke
runtime_numerical_integration_smoke
```

Validates real FP32 execution and integration of numerical execution with the runtime.

### Compiler → runtime

```text
compiler_driven_runtime_smoke
compiler_manual_plan_parity
target_contract_transport_smoke
```

Validates:

- compiler-generated plan authority
- compiler/manual scheduling parity
- compiler target-contract transport into C++

### Command lowering

```text
device_command_stream_smoke
compiler_runtime_direct_devicecommand_smoke
```

Validates:

- lowering from planned work to `DeviceCommandStream`
- direct compiler-driven runtime execution through device commands

### Device backend

```text
mockdevice_device_command_smoke
device_memory_address_smoke
device_address_contract_smoke
```

Validates:

- MockDevice command execution
- deterministic logical device-memory addresses
- device/SRAM bounds enforcement

### Public C++ build validation

The public FetchContent build path was validated with no local cuWaves override:

```bash
cmake -S . -B build-public-test
cmake --build build-public-test -j
ctest --test-dir build-public-test --output-on-failure
```

Result:

```text
13/13 tests passed
100% tests passed
```

This validates the public dependency/build path. A literal fresh-clone validation is performed separately before release.

For the detailed matrix, see `docs/validation_matrix.md`.

---

## Repository structure

```text
artifacts/     portable model/compiler artifacts
cmake/         CMake integration
data/          regenerable dataset contract
docs/          architecture, design, validation, runtime references
include/       public C++ headers
python/        training/export/compiler/reference tooling
src/           C++ runtime/backend implementation
tests/         C++ regression tests
tools/         C++ smoke/integration utilities
```

---

## Important project status

WaveForge Accelerator is an experimental compiler/runtime prototype.

It demonstrates:

- real model training and portable ONNX export
- multiple inference-runtime references
- target-contract design
- TVM Relax/TIR analysis
- TVM BYOC partitioning
- compiler/runtime interface design
- SRAM-aware memory planning
- row-aligned GEMM tiling
- graph scheduling
- logical DMA command lowering
- deterministic logical device addressing
- backend contract validation
- real C++ numerical inference

It does **not** claim to model:

- proprietary accelerator ISA
- physical address mappings
- analog/CIM circuit behavior
- measured WaveForge Accelerator hardware bandwidth
- measured WaveForge Accelerator accelerator performance
- proprietary EnCharge AI implementation details

---

## Engineering methodology

```text
Correctness before optimization.
Visibility before abstraction.
Hardware constraints before compiler automation.
```

The manual path was intentionally validated before compiler automation.

That gives a known-correct baseline for:

```text
graph parsing
memory planning
SRAM residency
tiling
scheduling
numerical execution
```

TVM integration then replaces specific manual decisions with compiler-derived decisions rather than introducing an opaque compiler path first.

---

## License

Apache-2.0.
