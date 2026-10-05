# TensorRT Reference

This is intentionally a **small real-runtime reference** for WaveForge Accelerator, similar in scope to the OpenVINO reference.

TensorRT is **not** the WaveForge compiler/runtime backend. It is used to validate that the same neural network that was **trained from scratch in PyTorch** on cuWaves-generated physical data can be exported to ONNX and executed correctly on real NVIDIA hardware.

## What it proves

- The WaveForge neural network trained from scratch in PyTorch can be exported to ONNX.
- TensorRT can parse the existing `wave_model.onnx`.
- TensorRT can build an NVIDIA GPU engine from that ONNX model.
- TensorRT output matches ONNX Runtime for one deterministic cuWaves-style sample.
- A simple single-request GPU execution latency can be measured on real NVIDIA hardware.

## What it does not claim

- TensorRT is not the WaveForge Accelerator backend.
- TensorRT latency is not a WaveForge accelerator-performance baseline.
- WaveForge accelerator timings remain simulated.
- The serialized TensorRT engine is not a portable project artifact.
- No FP16, INT8, custom TensorRT plugins, or custom CUDA kernels are used in this reference.

## Model lineage

The TensorRT reference executes the same model produced by the end-to-end WaveForge pipeline:

```text
cuWaves physical simulation
        ↓
deterministic dataset generation
        ↓
AI model training from scratch in PyTorch
        ↓
trained WaveRegressor weights
        ↓
ONNX export
        ↓
TensorRT engine build
        ↓
NVIDIA GPU inference
```

The current model predicts:

```text
amplitude
wavelength
```

from:

```text
64 spatial wave samples
```

## NVIDIA / Colab setup

First inspect the CUDA environment:

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

For CUDA 13.x, install:

```bash
python -m pip install --upgrade tensorrt onnxruntime
```

For a CUDA 12.x environment, use the matching TensorRT package if required by that environment:

```bash
python -m pip install --upgrade tensorrt-cu12 onnxruntime
```

For the validated CUDA 13 Colab environment, the working package set was:

```text
TensorRT:     11.3.0.99
ONNX Runtime: 1.30.0
PyTorch:      2.11.0+cu130
CUDA:         13.0
Protobuf:     5.29.6
```

The Python `onnx` package is not required by this reference script.

## Run

```bash
python python/tensorrt_reference.py \
  --onnx artifacts/wave_model.onnx \
  --engine artifacts/wave_model.trt.engine \
  --warmup 50 \
  --runs 200
```

The script stays FP32.

It does not enable:

```text
FP16
INT8
TensorRT plugins
custom CUDA kernels
```

## ONNX external-data requirement

The exported ONNX model uses an external initializer-data file.

Keep these files together:

```text
artifacts/wave_model.onnx
artifacts/wave_model.onnx.data
```

TensorRT will fail to parse the model if:

```text
artifacts/wave_model.onnx.data
```

is missing.

## Dynamic batch note

The exported ONNX model has a dynamic batch dimension.

TensorRT therefore requires an optimization profile at engine-build time.

For this reference:

```text
MIN = (1, 64)
OPT = (1, 64)
MAX = (1, 64)
```

The experiment intentionally measures only the single-request WaveForge input shape.

## Validated reference result

Environment:

```text
Google Colab
GPU:             Tesla T4
VRAM:            15,360 MiB
TensorRT:        11.3.0.99
CUDA:            13.0
PyTorch:         2.11.0+cu130
ONNX Runtime:    1.30.0
Protobuf:        5.29.6
```

TensorRT engine build:

```text
Input tensor:  wave_samples
Output tensor: wave_parameters
Engine size:   41,284 bytes
Warmup runs:   50
Measured runs: 200
```

Numerical parity:

```text
ONNX Runtime:
[[2.1766586303710938, 1.7368863821029663]]

TensorRT:
[[2.1766586303710938, 1.7368863821029663]]

Max absolute error: 0
Parity: PASS
```

Measured TensorRT GPU execution timing on the Tesla T4:

```text
Median: 0.030816 ms
Mean:   0.032567 ms
P95:    0.043565 ms
```

These are **real TensorRT measurements** on the reported NVIDIA GPU.

They must not be compared with WaveForge Accelerator's simulated accelerator timings as though both were measured hardware performance.

## Colab execution recipe

Mount Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Verify the NVIDIA environment:

```python
!nvidia-smi

import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.get_device_name(0))
```

Install the validated runtime dependencies:

```python
!pip install -q --upgrade tensorrt onnxruntime
!pip install -q --force-reinstall "protobuf>=5.29.1,<6"
```

Restart the Colab runtime after package installation if required.

Then ensure the project contains:

```text
python/tensorrt_reference.py
artifacts/wave_model.onnx
artifacts/wave_model.onnx.data
```

Run:

```bash
python python/tensorrt_reference.py \
  --onnx artifacts/wave_model.onnx \
  --engine artifacts/wave_model.trt.engine \
  --warmup 50 \
  --runs 200
```

The script automatically creates the batch-1 optimization profile when the ONNX input has a dynamic batch dimension.

## Engine portability

The generated engine:

```text
artifacts/wave_model.trt.engine
```

is environment/GPU dependent.

It should be rebuilt on the target NVIDIA environment rather than treated as a portable source artifact.

The portable source artifacts are:

```text
artifacts/wave_model.onnx
artifacts/wave_model.onnx.data
python/tensorrt_reference.py
```

## Git policy

Commit:

```text
python/tensorrt_reference.py
docs/tensorrt_reference.md
artifacts/wave_model.onnx
artifacts/wave_model.onnx.data
```

Do not commit:

```text
artifacts/wave_model.trt.engine
```

The repository `.gitignore` should include:

```text
*.trt.engine
*.engine
```

## Scope within WaveForge Accelerator

TensorRT is one validation path inside the larger project:

```text
cuWaves
  ↓
AI model training from scratch in PyTorch
  ↓
ONNX
  ├── ONNX Runtime
  ├── OpenVINO
  └── TensorRT
  ↓
Apache TVM BYOC
  ↓
SRAM-aware compiler plan
  ↓
C++ ExecutionPlan / DeviceCommandStream / MockDevice
```

The core WaveForge Accelerator work remains:

```text
task scheduling
memory management
SRAM-aware tensor tiling
logical DMA-style data movement
kernel execution
compiler/runtime contracts
C++ device-runtime execution
profiling
layered validation
```
