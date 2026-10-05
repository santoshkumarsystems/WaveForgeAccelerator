# WaveForge Accelerator Validation Matrix

This document maps each validation to the architectural contract it proves.

## 1. Data and model layer

| Validation | What it proves |
|---|---|
| Deterministic cuWaves dataset generation | Training data comes from the real C++ physics dependency rather than an unrelated toy generator |
| PyTorch training/validation metrics | The compact model learns amplitude and wavelength from 64 spatial samples |
| ONNX export | The trained PyTorch model is portable |
| PyTorch ↔ ONNX Runtime parity | ONNX preserves numerical behavior |

Recorded PyTorch ↔ ONNX Runtime maximum absolute difference:

```text
2.38418579e-07
```

## 2. Runtime foundation

### `runtime_smoke`

Validates basic runtime construction and execution behavior.

## 3. Model / initializer memory

### `model_memory_smoke`

Validates:

- ONNX initializer metadata
- tensor shapes
- dtype assumptions
- byte accounting
- current model initializer footprint

Current logical initializer image:

```text
10,584 bytes
```

## 4. SRAM planner

### `memory_planner_smoke`

Validates:

- SRAM capacity handling
- transient reservation
- initializer staging capacity
- whole-vs-tiled decisions

Validated 4 KiB target:

```text
SRAM:                    4096 B
transient reservation:    392 B
initializer staging:     3704 B
GEMM 0 tiles:          13 + 13 + 6
```

## 5. Numerical backend

### `numerical_parity_smoke`

Validates the C++ FP32 implementation against the reference model.

### `runtime_numerical_integration_smoke`

Validates that the runtime executes the real numerical backend correctly rather than only simulating scheduling.

Recorded C++ ↔ ONNX Runtime maximum absolute difference:

```text
2.384185791e-07
```

## 6. Compiler → runtime plan authority

### `compiler_driven_runtime_smoke`

Validates that the runtime consumes the compiler-generated execution plan.

### `compiler_manual_plan_parity`

Validates that the compiler-derived task schedule matches the previously validated manual planner for the current model/target.

### `target_contract_transport_smoke`

Validates the compiler target contract as it crosses:

```text
Python compiler
  ↓
execution plan / manifest
  ↓
C++ parser
  ↓
Runtime validation
```

The runtime rejects incompatible dtype, memory-space, or supported-op contracts.

## 7. Device-command lowering

### `device_command_stream_smoke`

Validates lowering from planned operations into accelerator-style `DeviceCommand` objects.

Supported command categories include:

```text
DmaDeviceToSram
GemmTile
Gemm
Relu
Mul
Add
```

### `compiler_runtime_direct_devicecommand_smoke`

Validates the direct path:

```text
compiler plan
  ↓
Runtime
  ↓
DeviceCommandStream
  ↓
MockDevice
```

## 8. Device backend and memory contracts

### `mockdevice_device_command_smoke`

Validates direct MockDevice execution/acceptance of device commands.

### `device_memory_address_smoke`

Validates deterministic logical model-initializer addressing.

Current contiguous logical map totals:

```text
10,584 bytes
```

### `device_address_contract_smoke`

Validates rejection of:

- device source-range overflow
- initializer staging-range overflow
- full SRAM overflow
- invalid staging-capacity contracts

The addresses are logical model offsets, not physical PCIe/MMIO/IOVA/BAR addresses.

## 9. TVM compiler validation

Validated compiler stages include:

```text
ONNX → Relax
Relax → PrimFunc/TIR
target-policy evaluation
SRAM-aware schedule
BYOC partitioning
BYOC → ExecutionPlan bridge
integrated compiler-plan emission
```

Validated first-GEMM policy:

```text
row bytes:   256
tile width:   13
tiles:       [13, 13, 6]
```

Validated BYOC partition set:

```text
waveaccel.matmul_bias_relu × 2
waveaccel.matmul_bias      × 1
waveaccel.multiply         × 1
waveaccel.add              × 1
```

Full current graph coverage:

```text
YES
```

## 10. OpenVINO reference

OpenVINO is a real local CPU reference runtime.

Recorded timing:

```text
median: 0.017511 ms
mean:   0.020187 ms
```

These values are measurements of the local CPU/OpenVINO environment, not WaveForge Accelerator timing.

## 11. TensorRT reference

Validated environment:

```text
Google Colab
Tesla T4
TensorRT 11.3.0.99
CUDA 13.0
```

Numerical result:

```text
ONNX Runtime:
[[2.1766586303710938, 1.7368863821029663]]

TensorRT:
[[2.1766586303710938, 1.7368863821029663]]

max absolute error: 0
```

Measured TensorRT GPU execution:

```text
median: 0.030816 ms
mean:   0.032567 ms
p95:    0.043565 ms
```

TensorRT timing is a real T4 measurement. WaveForge Accelerator timing remains simulated.

## 12. Public dependency/build path

Validated without a local cuWaves override:

```bash
cmake -S . -B build-public-test
cmake --build build-public-test -j
ctest --test-dir build-public-test --output-on-failure
```

CMake fetched:

```text
cuWaves v0.0.1
```

Result:

```text
13/13 tests passed
100% tests passed
```

This validates the public FetchContent path. A literal fresh-clone build remains a separate final-release check.

## C++ regression inventory

| # | Test | Architectural layer |
|---:|---|---|
| 1 | `runtime_smoke` | Runtime foundation |
| 2 | `model_memory_smoke` | Model / initializer memory |
| 3 | `memory_planner_smoke` | SRAM planning |
| 4 | `numerical_parity_smoke` | Numerical backend |
| 5 | `runtime_numerical_integration_smoke` | Runtime + numerical integration |
| 6 | `compiler_driven_runtime_smoke` | Compiler → runtime |
| 7 | `compiler_manual_plan_parity` | Compiler/manual schedule parity |
| 8 | `device_command_stream_smoke` | Command lowering |
| 9 | `mockdevice_device_command_smoke` | Device backend |
| 10 | `compiler_runtime_direct_devicecommand_smoke` | Compiler → command → device path |
| 11 | `device_memory_address_smoke` | Logical device memory |
| 12 | `device_address_contract_smoke` | Device/SRAM bounds contract |
| 13 | `target_contract_transport_smoke` | Compiler/runtime target contract |

Validated public-build result:

```text
100% tests passed, 0 tests failed out of 13
```
