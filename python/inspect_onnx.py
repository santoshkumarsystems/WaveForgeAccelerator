"""
Inspect the exported WaveAccel ONNX model.

Reports:
- Model input/output shapes
- Operator graph
- Initializer tensor names, shapes, parameter counts, and byte sizes
- Total constants stored in the ONNX model

Author: Santosh Kumar
Copyright 2026 Santosh Kumar
SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import argparse
from collections import Counter

import onnx
from onnx import TensorProto


DTYPE_BYTES = {
    TensorProto.FLOAT: 4,
    TensorProto.FLOAT16: 2,
    TensorProto.DOUBLE: 8,
    TensorProto.INT8: 1,
    TensorProto.UINT8: 1,
    TensorProto.INT16: 2,
    TensorProto.UINT16: 2,
    TensorProto.INT32: 4,
    TensorProto.UINT32: 4,
    TensorProto.INT64: 8,
    TensorProto.UINT64: 8,
    TensorProto.BOOL: 1,
}


def tensor_shape(tensor) -> list[int]:
    return list(tensor.dims)


def element_count(shape: list[int]) -> int:
    count = 1
    for dim in shape:
        count *= dim
    return count


def value_info_shape(value_info) -> str:
    dims = value_info.type.tensor_type.shape.dim
    values = []

    for dim in dims:
        if dim.HasField("dim_value"):
            values.append(str(dim.dim_value))
        elif dim.HasField("dim_param"):
            values.append(dim.dim_param)
        else:
            values.append("?")

    return "[" + ", ".join(values) + "]"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    args = parser.parse_args()

    model = onnx.load(args.onnx)
    onnx.checker.check_model(model)

    graph = model.graph

    print("ONNX MODEL INSPECTION")
    print("=" * 70)

    print("\nInputs:")
    for value in graph.input:
        print(f"  {value.name:30s} {value_info_shape(value)}")

    print("\nOutputs:")
    for value in graph.output:
        print(f"  {value.name:30s} {value_info_shape(value)}")

    print("\nOperators:")
    op_counts = Counter(node.op_type for node in graph.node)

    for op, count in sorted(op_counts.items()):
        print(f"  {op:20s} x {count}")

    print("\nGraph execution order:")
    for index, node in enumerate(graph.node):
        print(
            f"  {index:2d}: "
            f"{node.op_type:12s} "
            f"inputs={list(node.input)} "
            f"outputs={list(node.output)}"
        )

    print("\nStored tensors / initializers:")
    total_elements = 0
    total_bytes = 0

    for tensor in graph.initializer:
        shape = tensor_shape(tensor)
        elements = element_count(shape)
        bytes_per_element = DTYPE_BYTES.get(tensor.data_type, 0)
        size_bytes = elements * bytes_per_element

        total_elements += elements
        total_bytes += size_bytes

        print(
            f"  {tensor.name:40s} "
            f"shape={str(shape):16s} "
            f"elements={elements:6d} "
            f"bytes={size_bytes:6d}"
        )

    print("\nTotals:")
    print(f"  Stored tensor elements : {total_elements}")
    print(f"  Stored tensor bytes    : {total_bytes}")
    print(f"  Stored tensor KiB      : {total_bytes / 1024:.3f}")

    print("\nONNX checker: PASS")


if __name__ == "__main__":
    main()
