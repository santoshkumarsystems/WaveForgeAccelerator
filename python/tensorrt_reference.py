#!/usr/bin/env python3
"""
Minimal TensorRT reference for WaveAccel.

Purpose:
  - parse the existing FP32 ONNX model;
  - build one TensorRT engine on an NVIDIA GPU;
  - run one deterministic wave sample;
  - compare TensorRT output with ONNX Runtime;
  - report a simple single-request GPU execution latency.

This is intentionally a reference-runtime comparison, not part of the
WaveAccel backend implementation.

Requirements:
  - NVIDIA GPU
  - TensorRT 10.x or 11.x Python package
  - PyTorch with CUDA
  - onnxruntime

Author: Santosh Kumar
Copyright 2026 Santosh Kumar
SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import argparse
import math
import statistics
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import tensorrt as trt


def make_wave_sample() -> np.ndarray:
    """
    Deterministic sample consistent with the cuWaves 1-D wave form:
        A * sin(2*pi*x/lambda)
    for time=0 and phase=0.
    """
    amplitude = 2.17
    wavelength_m = 1.73
    dx_m = 0.05

    x = np.arange(64, dtype=np.float32) * np.float32(dx_m)
    values = (
        np.float32(amplitude)
        * np.sin(
            np.float32(2.0 * math.pi)
            * x
            / np.float32(wavelength_m)
        )
    )

    return np.ascontiguousarray(
        values.reshape(1, 64),
        dtype=np.float32,
    )


def build_engine(
    onnx_path: Path,
    engine_path: Path,
    workspace_mib: int,
) -> bytes:
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)

    # TensorRT 11 is strongly typed by default. Using flags=0 also remains the
    # straightforward ONNX-parser path for current TensorRT 10.x.
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)

    if not parser.parse_from_file(str(onnx_path)):
        errors = [
            str(parser.get_error(i))
            for i in range(parser.num_errors)
        ]
        raise RuntimeError(
            "TensorRT ONNX parse failed:\n" + "\n".join(errors)
        )

    if network.num_inputs != 1 or network.num_outputs != 1:
        raise RuntimeError(
            "WaveAccel TensorRT reference expects one input and one output"
        )

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE,
        workspace_mib * 1024 * 1024,
    )

    # The exported ONNX model has a dynamic batch dimension. TensorRT requires
    # at least one optimization profile for networks with dynamic inputs.
    # This reference intentionally benchmarks batch size 1 only, so MIN=OPT=MAX
    # is fixed to the actual WaveAccel sample shape [1, 64].
    input_tensor = network.get_input(0)
    declared_shape = tuple(int(dim) for dim in input_tensor.shape)

    if any(dim < 0 for dim in declared_shape):
        reference_shape = (1, 64)

        if len(declared_shape) != len(reference_shape):
            raise RuntimeError(
                "Unexpected TensorRT input rank for WaveAccel reference: "
                f"{declared_shape}"
            )

        for declared_dim, reference_dim in zip(
            declared_shape,
            reference_shape,
        ):
            if declared_dim >= 0 and declared_dim != reference_dim:
                raise RuntimeError(
                    "WaveAccel reference shape is incompatible with ONNX "
                    f"input shape {declared_shape}"
                )

        profile = builder.create_optimization_profile()
        profile.set_shape(
            input_tensor.name,
            reference_shape,
            reference_shape,
            reference_shape,
        )

        profile_index = config.add_optimization_profile(profile)

        if profile_index < 0:
            raise RuntimeError(
                "TensorRT rejected the WaveAccel optimization profile"
            )

    serialized = builder.build_serialized_network(
        network,
        config,
    )

    if serialized is None:
        raise RuntimeError("TensorRT engine build failed")

    engine_bytes = bytes(serialized)

    engine_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    engine_path.write_bytes(engine_bytes)

    return engine_bytes


def load_engine(engine_bytes: bytes):
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)

    engine = runtime.deserialize_cuda_engine(engine_bytes)

    if engine is None:
        raise RuntimeError(
            "TensorRT engine deserialization failed"
        )

    return runtime, engine


def io_names(engine) -> tuple[str, str]:
    inputs = []
    outputs = []

    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        mode = engine.get_tensor_mode(name)

        if mode == trt.TensorIOMode.INPUT:
            inputs.append(name)
        elif mode == trt.TensorIOMode.OUTPUT:
            outputs.append(name)

    if len(inputs) != 1 or len(outputs) != 1:
        raise RuntimeError(
            "Expected exactly one TensorRT input and output"
        )

    return inputs[0], outputs[0]


def require_fp32_io(engine, input_name: str, output_name: str) -> None:
    for name in (input_name, output_name):
        dtype = engine.get_tensor_dtype(name)

        if dtype != trt.float32:
            raise RuntimeError(
                f"Expected FP32 TensorRT I/O for {name}, got {dtype}"
            )


def run_ort_reference(
    onnx_path: Path,
    input_array: np.ndarray,
) -> np.ndarray:
    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name

    output = session.run(
        None,
        {input_name: input_array},
    )[0]

    return np.asarray(
        output,
        dtype=np.float32,
    )


def run_tensorrt(
    engine,
    input_array: np.ndarray,
    warmup: int,
    runs: int,
) -> tuple[np.ndarray, list[float]]:
    context = engine.create_execution_context()

    if context is None:
        raise RuntimeError(
            "TensorRT execution-context creation failed"
        )

    input_name, output_name = io_names(engine)
    require_fp32_io(
        engine,
        input_name,
        output_name,
    )

    input_shape = tuple(
        int(x)
        for x in input_array.shape
    )

    declared_input_shape = tuple(
        int(x)
        for x in engine.get_tensor_shape(input_name)
    )

    if any(dim < 0 for dim in declared_input_shape):
        if not context.set_input_shape(
            input_name,
            input_shape,
        ):
            raise RuntimeError(
                "TensorRT rejected input shape "
                f"{input_shape}"
            )
    elif declared_input_shape != input_shape:
        raise RuntimeError(
            f"TensorRT input shape is {declared_input_shape}, "
            f"sample is {input_shape}"
        )

    output_shape = tuple(
        int(x)
        for x in context.get_tensor_shape(output_name)
    )

    if any(dim <= 0 for dim in output_shape):
        raise RuntimeError(
            f"Unresolved TensorRT output shape: {output_shape}"
        )

    device_input = torch.from_numpy(
        input_array
    ).to(
        device="cuda",
        dtype=torch.float32,
    ).contiguous()

    device_output = torch.empty(
        output_shape,
        device="cuda",
        dtype=torch.float32,
    )

    if not context.set_tensor_address(
        input_name,
        int(device_input.data_ptr()),
    ):
        raise RuntimeError(
            "Failed to bind TensorRT input address"
        )

    if not context.set_tensor_address(
        output_name,
        int(device_output.data_ptr()),
    ):
        raise RuntimeError(
            "Failed to bind TensorRT output address"
        )

    stream = torch.cuda.Stream()

    for _ in range(warmup):
        ok = context.execute_async_v3(
            stream_handle=stream.cuda_stream
        )
        if not ok:
            raise RuntimeError(
                "TensorRT warmup inference failed"
            )

    stream.synchronize()

    timings_ms: list[float] = []

    for _ in range(runs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record(stream)

        ok = context.execute_async_v3(
            stream_handle=stream.cuda_stream
        )

        if not ok:
            raise RuntimeError(
                "TensorRT benchmark inference failed"
            )

        end.record(stream)
        end.synchronize()

        timings_ms.append(
            float(start.elapsed_time(end))
        )

    output = device_output.detach().cpu().numpy()

    return output, timings_ms


def percentile(values: list[float], q: float) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.percentile(array, q))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Minimal TensorRT reference for WaveAccel."
    )
    parser.add_argument(
        "--onnx",
        default="artifacts/wave_model.onnx",
    )
    parser.add_argument(
        "--engine",
        default="artifacts/wave_model.trt.engine",
    )
    parser.add_argument(
        "--workspace-mib",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--reuse-engine",
        action="store_true",
        help=(
            "Load an existing engine instead of rebuilding. "
            "Only use an engine built on the same compatible GPU/platform."
        ),
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "TensorRT reference requires an NVIDIA CUDA GPU"
        )

    if args.workspace_mib <= 0:
        raise ValueError(
            "workspace MiB must be positive"
        )

    if args.warmup < 0 or args.runs <= 0:
        raise ValueError(
            "warmup must be >= 0 and runs must be > 0"
        )

    onnx_path = Path(args.onnx)
    engine_path = Path(args.engine)

    if not onnx_path.is_file():
        raise FileNotFoundError(
            f"ONNX model not found: {onnx_path}"
        )

    if args.reuse_engine:
        if not engine_path.is_file():
            raise FileNotFoundError(
                f"TensorRT engine not found: {engine_path}"
            )
        engine_bytes = engine_path.read_bytes()
        build_mode = "reused existing engine"
    else:
        engine_bytes = build_engine(
            onnx_path,
            engine_path,
            args.workspace_mib,
        )
        build_mode = "built from ONNX"

    runtime, engine = load_engine(engine_bytes)

    sample = make_wave_sample()

    ort_output = run_ort_reference(
        onnx_path,
        sample,
    )

    trt_output, timings_ms = run_tensorrt(
        engine,
        sample,
        args.warmup,
        args.runs,
    )

    max_abs_error = float(
        np.max(
            np.abs(
                trt_output.astype(np.float64)
                - ort_output.astype(np.float64)
            )
        )
    )

    if not np.allclose(
        trt_output,
        ort_output,
        rtol=1e-4,
        atol=1e-4,
    ):
        raise RuntimeError(
            "TensorRT output does not match ONNX Runtime "
            f"within FP32 reference tolerance; "
            f"max_abs_error={max_abs_error:.9g}"
        )

    input_name, output_name = io_names(engine)

    print("WaveAccel TensorRT reference")
    print("=" * 72)
    print(f"TensorRT version:               {trt.__version__}")
    print(f"PyTorch CUDA:                   {torch.version.cuda}")
    print(f"GPU:                            {torch.cuda.get_device_name(0)}")
    print(f"ONNX model:                     {onnx_path}")
    print(f"engine:                         {engine_path}")
    print(f"engine mode:                    {build_mode}")
    print(f"engine bytes:                   {len(engine_bytes)}")
    print(f"input tensor:                   {input_name}")
    print(f"output tensor:                  {output_name}")
    print(f"warmup runs:                    {args.warmup}")
    print(f"measured runs:                  {args.runs}")

    print("\nReference output")
    print(f"ONNX Runtime:                   {ort_output.tolist()}")
    print(f"TensorRT:                       {trt_output.tolist()}")
    print(f"max absolute error:             {max_abs_error:.9g}")
    print("TensorRT / ORT parity:          PASS")

    print("\nTensorRT GPU execution timing")
    print(
        f"median:                         "
        f"{statistics.median(timings_ms):.6f} ms"
    )
    print(
        f"mean:                           "
        f"{statistics.mean(timings_ms):.6f} ms"
    )
    print(
        f"p95:                            "
        f"{percentile(timings_ms, 95.0):.6f} ms"
    )

    print(
        "\nNOTE: TensorRT timing above is a real measurement on the "
        "reported NVIDIA GPU."
    )
    print(
        "It is not comparable to WaveAccel's simulated accelerator "
        "timing as a hardware-performance claim."
    )
    print(
        "Serialized TensorRT engines are environment/GPU dependent; "
        "rebuild on the target GPU when needed."
    )

    # Keep runtime alive until after inference resources have gone out of scope.
    del engine
    del runtime


if __name__ == "__main__":
    main()
