"""
Export the real WaveAccel ONNX initializer values and a reference inference
sample for C++ numerical-parity testing.

The script produces two binary artifacts:

1. wave_model.weights
   - every ONNX initializer tensor
   - tensor names
   - tensor shapes
   - raw float32 values

2. wave_sample.bin
   - one 64-value wave input from the cuWaves-generated CSV
   - the physical ground-truth amplitude/wavelength from that CSV row
   - the ONNX Runtime reference prediction for the same input

The C++ WaveAccel numerical executor consumes these files and must reproduce
the ONNX Runtime prediction within a small floating-point tolerance.

Author: Santosh Kumar
Copyright 2026 Santosh Kumar
SPDX-License-Identifier: Apache-2.0

NOTE:
This exporter also validates the ONNX Gemm attributes expected by the V1 C++
executor. WaveAccel V1 supports the PyTorch Linear export convention:

    transA = 0
    transB = 1
    alpha  = 1.0
    beta   = 1.0

If the ONNX graph uses different Gemm semantics, export stops instead of
silently executing the wrong mathematics.
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
from onnx import helper, numpy_helper


WEIGHTS_MAGIC = b"WAWGT001"
SAMPLE_MAGIC = b"WASMP001"
FORMAT_VERSION = 1

FEATURE_NAMES = [f"x{i}" for i in range(64)]
TARGET_NAMES = ["amplitude", "wavelength"]


def write_u32(handle, value: int) -> None:
    handle.write(struct.pack("<I", value))


def write_u64(handle, value: int) -> None:
    handle.write(struct.pack("<Q", value))


def validate_gemm_semantics(model: onnx.ModelProto) -> None:
    """Require Gemm semantics implemented by the WaveAccel V1 executor."""
    gemm_count = 0

    for node in model.graph.node:
        if node.op_type != "Gemm":
            continue

        gemm_count += 1

        attrs = {
            attr.name: helper.get_attribute_value(attr)
            for attr in node.attribute
        }

        trans_a = int(attrs.get("transA", 0))
        trans_b = int(attrs.get("transB", 0))
        alpha = float(attrs.get("alpha", 1.0))
        beta = float(attrs.get("beta", 1.0))

        if not (
            trans_a == 0
            and trans_b == 1
            and alpha == 1.0
            and beta == 1.0
        ):
            raise RuntimeError(
                "Unsupported Gemm attributes for WaveAccel V1: "
                f"transA={trans_a}, transB={trans_b}, "
                f"alpha={alpha}, beta={beta}"
            )

    if gemm_count == 0:
        raise RuntimeError("ONNX model contains no Gemm nodes")

    print(
        "validated Gemm semantics: "
        "transA=0 transB=1 alpha=1 beta=1"
    )


def export_weights(
    model: onnx.ModelProto,
    output_path: Path,
) -> None:
    initializers = list(model.graph.initializer)

    if not initializers:
        raise RuntimeError("ONNX model contains no initializers")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("wb") as handle:
        handle.write(WEIGHTS_MAGIC)
        write_u32(handle, FORMAT_VERSION)
        write_u32(handle, len(initializers))

        total_values = 0
        total_bytes = 0

        for tensor in initializers:
            array = numpy_helper.to_array(tensor)

            if array.dtype != np.float32:
                raise RuntimeError(
                    f"WaveAccel V1 requires float32 initializer "
                    f"{tensor.name}; found {array.dtype}"
                )

            array = np.ascontiguousarray(array, dtype=np.float32)
            name_bytes = tensor.name.encode("utf-8")

            write_u32(handle, len(name_bytes))
            handle.write(name_bytes)

            write_u32(handle, array.ndim)

            for dim in array.shape:
                write_u64(handle, int(dim))

            value_count = int(array.size)
            write_u64(handle, value_count)

            raw = array.astype("<f4", copy=False).tobytes(order="C")
            handle.write(raw)

            total_values += value_count
            total_bytes += len(raw)

    print(f"weights file:                {output_path}")
    print(f"initializer tensors:         {len(initializers)}")
    print(f"initializer float values:    {total_values}")
    print(f"initializer raw bytes:       {total_bytes}")


def export_sample(
    onnx_path: Path,
    dataset_path: Path,
    row_index: int,
    output_path: Path,
) -> None:
    dataframe = pd.read_csv(dataset_path)

    if row_index < 0 or row_index >= len(dataframe):
        raise IndexError(
            f"row index {row_index} outside dataset size {len(dataframe)}"
        )

    missing = [
        name
        for name in FEATURE_NAMES + TARGET_NAMES
        if name not in dataframe.columns
    ]

    if missing:
        raise RuntimeError(
            f"dataset is missing required columns: {missing}"
        )

    row = dataframe.iloc[row_index]

    features = row[FEATURE_NAMES].to_numpy(
        dtype=np.float32,
        copy=True,
    )

    target = row[TARGET_NAMES].to_numpy(
        dtype=np.float32,
        copy=True,
    )

    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )

    input_info = session.get_inputs()

    if len(input_info) != 1:
        raise RuntimeError(
            f"expected one ONNX input, found {len(input_info)}"
        )

    outputs = session.run(
        None,
        {
            input_info[0].name:
                features.reshape(1, -1)
        },
    )

    if len(outputs) != 1:
        raise RuntimeError(
            f"expected one ONNX output, found {len(outputs)}"
        )

    reference = np.asarray(
        outputs[0],
        dtype=np.float32,
    ).reshape(-1)

    if reference.size != 2:
        raise RuntimeError(
            f"expected two model outputs, found {reference.size}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("wb") as handle:
        handle.write(SAMPLE_MAGIC)
        write_u32(handle, FORMAT_VERSION)

        write_u32(handle, int(features.size))
        write_u32(handle, int(target.size))
        write_u32(handle, int(reference.size))

        handle.write(
            features.astype("<f4", copy=False).tobytes()
        )
        handle.write(
            target.astype("<f4", copy=False).tobytes()
        )
        handle.write(
            reference.astype("<f4", copy=False).tobytes()
        )

    print(f"sample file:                 {output_path}")
    print(f"dataset row:                 {row_index}")
    print(f"input float values:          {features.size}")
    print(
        "ground truth [A, lambda]:   "
        f"[{target[0]:.9f}, {target[1]:.9f}]"
    )
    print(
        "ORT reference [A, lambda]:  "
        f"[{reference[0]:.9f}, {reference[1]:.9f}]"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--weights-output", required=True)
    parser.add_argument("--sample-output", required=True)
    parser.add_argument("--row", type=int, default=0)
    args = parser.parse_args()

    onnx_path = Path(args.onnx)
    dataset_path = Path(args.dataset)
    weights_output = Path(args.weights_output)
    sample_output = Path(args.sample_output)

    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)

    print("WaveAccel numerical fixture export")
    print("=" * 70)

    validate_gemm_semantics(model)

    export_weights(
        model,
        weights_output,
    )

    export_sample(
        onnx_path,
        dataset_path,
        args.row,
        sample_output,
    )


if __name__ == "__main__":
    main()
