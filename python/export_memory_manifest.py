"""
Generate WaveAccel accelerator-memory metadata from an ONNX model.

The generated manifest contains:
- model input/output sizes,
- total ONNX initializer storage,
- every individual initializer tensor,
- tensor shapes and byte sizes,
- largest initializer,
- largest runtime activation,
- ONNX operator sequence,
- graph-node input/output relationships,
- initializer dependencies for every ONNX node.

WaveAccel consumes this information for accelerator memory-planning
experiments instead of hard-coding model sizes or graph relationships in C++.

Author: Santosh Kumar
Copyright 2026 Santosh Kumar
SPDX-License-Identifier: Apache-2.0

NOTE:
Tensor sizes and graph relationships are derived from the real exported ONNX
model.

Any SRAM hierarchy, DMA behavior, tiling strategy, or accelerator timing
modeled later by WaveAccel is a generic simulation and must not be interpreted
as measurements or proprietary behavior of commercial hardware.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, shape_inference


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


def element_count(shape: list[int]) -> int:
    """Return the number of elements represented by a concrete shape."""
    count = 1
    for dim in shape:
        count *= dim
    return count


def shape_string(shape: list[int]) -> str:
    """Convert a tensor shape into compact manifest form, e.g. 32x64."""
    return "x".join(str(dim) for dim in shape)


def concrete_shape(value_info, batch_size: int) -> list[int]:
    """Resolve an ONNX ValueInfo shape for a selected inference batch size."""
    result: list[int] = []

    for index, dim in enumerate(value_info.type.tensor_type.shape.dim):
        if dim.HasField("dim_value"):
            result.append(dim.dim_value)
        elif index == 0:
            # WaveAccel exports the batch dimension dynamically.
            result.append(batch_size)
        else:
            raise ValueError(
                f"Cannot resolve symbolic dimension for {value_info.name}"
            )

    return result


def value_info_bytes(value_info, batch_size: int) -> int:
    """Calculate storage required by an ONNX ValueInfo tensor."""
    tensor_type = value_info.type.tensor_type
    bytes_per_element = DTYPE_BYTES.get(tensor_type.elem_type)

    if bytes_per_element is None:
        raise ValueError(
            f"Unsupported ONNX datatype {tensor_type.elem_type} "
            f"for {value_info.name}"
        )

    shape = concrete_shape(value_info, batch_size)
    return element_count(shape) * bytes_per_element


def initializer_bytes(tensor) -> int:
    """Calculate storage required by an ONNX initializer tensor."""
    bytes_per_element = DTYPE_BYTES.get(tensor.data_type)

    if bytes_per_element is None:
        raise ValueError(
            f"Unsupported ONNX datatype {tensor.data_type} "
            f"for initializer {tensor.name}"
        )

    return element_count(list(tensor.dims)) * bytes_per_element


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate WaveAccel accelerator-memory metadata "
            "from an exported ONNX model."
        )
    )
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")

    model = onnx.load(args.onnx)
    onnx.checker.check_model(model)

    inferred = shape_inference.infer_shapes(model)
    graph = inferred.graph

    initializer_names = {
        tensor.name for tensor in graph.initializer
    }

    model_inputs = [
        value
        for value in graph.input
        if value.name not in initializer_names
    ]

    if len(model_inputs) != 1:
        raise RuntimeError(
            f"Expected exactly one model input, found {len(model_inputs)}"
        )

    if len(graph.output) != 1:
        raise RuntimeError(
            f"Expected exactly one model output, found {len(graph.output)}"
        )

    input_bytes = value_info_bytes(
        model_inputs[0],
        args.batch_size,
    )

    output_bytes = value_info_bytes(
        graph.output[0],
        args.batch_size,
    )

    initializers = []

    for tensor in graph.initializer:
        shape = list(tensor.dims)
        elements = element_count(shape)
        size_bytes = initializer_bytes(tensor)

        initializers.append(
            {
                "name": tensor.name,
                "shape": shape,
                "elements": elements,
                "bytes": size_bytes,
            }
        )

    if not initializers:
        raise RuntimeError("ONNX model contains no initializer tensors")

    total_initializer_elements = sum(
        item["elements"] for item in initializers
    )

    total_initializer_bytes = sum(
        item["bytes"] for item in initializers
    )

    largest_initializer = max(
        initializers,
        key=lambda item: item["bytes"],
    )

    intermediate_sizes: list[tuple[str, int]] = []

    for value in graph.value_info:
        # Initializers are persistent model state, not activations.
        if value.name in initializer_names:
            continue

        try:
            size_bytes = value_info_bytes(
                value,
                args.batch_size,
            )
        except ValueError:
            continue

        intermediate_sizes.append(
            (value.name, size_bytes)
        )

    if intermediate_sizes:
        largest_activation_name, largest_activation_bytes = max(
            intermediate_sizes,
            key=lambda item: item[1],
        )
    else:
        largest_activation_name = ""
        largest_activation_bytes = 0

    operator_sequence = ",".join(
        node.op_type for node in graph.node
    )

    graph_nodes = []

    for index, node in enumerate(graph.node):
        initializer_inputs = [
            name
            for name in node.input
            if name in initializer_names
        ]

        graph_nodes.append(
            {
                "index": index,
                "op": node.op_type,
                "inputs": list(node.input),
                "outputs": list(node.output),
                "initializers": initializer_inputs,
            }
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = [
        "# WaveAccel ONNX Memory Manifest",
        "#",
        "# Author: Santosh Kumar",
        "# Copyright 2026 Santosh Kumar",
        "# SPDX-License-Identifier: Apache-2.0",
        "#",
        "# Generated automatically from the exported ONNX model.",
        "# Do not manually edit model-size or graph-dependency values.",
        "",
        "# Model-level memory metadata",
        f"batch_size={args.batch_size}",
        f"input_bytes={input_bytes}",
        f"output_bytes={output_bytes}",
        f"initializer_count={len(initializers)}",
        f"initializer_elements={total_initializer_elements}",
        f"initializer_bytes={total_initializer_bytes}",
        f"largest_initializer_name={largest_initializer['name']}",
        f"largest_initializer_bytes={largest_initializer['bytes']}",
        f"largest_activation_name={largest_activation_name}",
        f"largest_activation_bytes={largest_activation_bytes}",
        f"operator_sequence={operator_sequence}",
        f"node_count={len(graph_nodes)}",
        "",
        "# Individual ONNX initializer tensors",
    ]

    for index, tensor in enumerate(initializers):
        prefix = f"initializer_{index}"

        manifest.extend(
            [
                f"{prefix}_name={tensor['name']}",
                f"{prefix}_shape={shape_string(tensor['shape'])}",
                f"{prefix}_elements={tensor['elements']}",
                f"{prefix}_bytes={tensor['bytes']}",
            ]
        )

    manifest.extend(
        [
            "",
            "# ONNX graph nodes and initializer dependencies",
        ]
    )

    for node in graph_nodes:
        prefix = f"node_{node['index']}"

        manifest.extend(
            [
                f"{prefix}_op={node['op']}",
                f"{prefix}_inputs={','.join(node['inputs'])}",
                f"{prefix}_outputs={','.join(node['outputs'])}",
                (
                    f"{prefix}_initializer_count="
                    f"{len(node['initializers'])}"
                ),
                (
                    f"{prefix}_initializers="
                    f"{','.join(node['initializers'])}"
                ),
            ]
        )

    output_path.write_text(
        "\n".join(manifest) + "\n",
        encoding="utf-8",
    )

    print("WaveAccel ONNX memory manifest")
    print("=" * 70)
    print(f"batch size:                  {args.batch_size}")
    print(f"input bytes:                 {input_bytes}")
    print(f"output bytes:                {output_bytes}")
    print(f"initializer count:           {len(initializers)}")
    print(
        f"initializer elements:        "
        f"{total_initializer_elements}"
    )
    print(
        f"initializer bytes:           "
        f"{total_initializer_bytes}"
    )
    print(
        f"largest initializer:         "
        f"{largest_initializer['name']} "
        f"({largest_initializer['bytes']} B)"
    )
    print(
        f"largest activation:          "
        f"{largest_activation_name} "
        f"({largest_activation_bytes} B)"
    )
    print(
        f"operator sequence:           "
        f"{operator_sequence}"
    )
    print(f"graph node count:            {len(graph_nodes)}")

    print("\nInitializer tensors:")
    for index, tensor in enumerate(initializers):
        print(
            f"  [{index}] "
            f"{tensor['name']:<25} "
            f"shape={shape_string(tensor['shape']):<8} "
            f"bytes={tensor['bytes']}"
        )

    print("\nGraph nodes:")
    for node in graph_nodes:
        initializer_text = (
            ",".join(node["initializers"])
            if node["initializers"]
            else "-"
        )

        print(
            f"  [{node['index']}] "
            f"{node['op']:<8} "
            f"initializers={initializer_text}"
        )

    print(f"\nmanifest written: {output_path}")


if __name__ == "__main__":
    main()
