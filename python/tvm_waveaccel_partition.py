#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import onnx
import tvm
from tvm.relax.backend.patterns import make_matmul_pattern
from tvm.relax.dpl import is_op, wildcard
from tvm.relax.frontend.onnx import from_onnx
from tvm.relax.transform import ConvertToDataflow, FusionPattern, FuseOpsByPattern

from waveaccel_target_contract import WaveAccelTargetContract

BACKEND = "waveaccel"


def _elementwise_pattern(op_name: str):
    lhs = wildcard()
    rhs = wildcard()
    root = is_op(op_name)(lhs, rhs).has_dtype("float32")
    return root, {"lhs": lhs, "rhs": rhs, "root": root}


def waveaccel_patterns() -> list[FusionPattern]:
    mm_bias_relu, mm_bias_relu_annotations = make_matmul_pattern(
        with_bias=True,
        activation="relax.nn.relu",
        transposed_rhs=True,
    )
    mm_bias_relu = mm_bias_relu.has_dtype("float32")

    mm_bias, mm_bias_annotations = make_matmul_pattern(
        with_bias=True,
        activation=None,
        transposed_rhs=True,
    )
    mm_bias = mm_bias.has_dtype("float32")

    multiply, multiply_annotations = _elementwise_pattern("relax.multiply")
    add, add_annotations = _elementwise_pattern("relax.add")

    return [
        FusionPattern(
            name=f"{BACKEND}.matmul_bias_relu",
            pattern=mm_bias_relu,
            annotation_patterns=mm_bias_relu_annotations,
        ),
        FusionPattern(
            name=f"{BACKEND}.matmul_bias",
            pattern=mm_bias,
            annotation_patterns=mm_bias_annotations,
        ),
        FusionPattern(
            name=f"{BACKEND}.multiply",
            pattern=multiply,
            annotation_patterns=multiply_annotations,
        ),
        FusionPattern(
            name=f"{BACKEND}.add",
            pattern=add,
            annotation_patterns=add_annotations,
        ),
    ]


def codegen_functions(mod: tvm.IRModule) -> list[str]:
    names = []
    for global_var, function in mod.functions.items():
        attrs = getattr(function, "attrs", None)
        if attrs is None:
            continue
        try:
            codegen = attrs["Codegen"]
        except (KeyError, TypeError):
            continue
        if str(codegen) == BACKEND:
            names.append(global_var.name_hint)
    return sorted(names)


def composite_counts_from_script(ir_text: str) -> Counter:
    pattern = r"[\"']Composite[\"']\s*:\s*[\"'](waveaccel\.[^\"']+)[\"']"
    return Counter(re.findall(pattern, ir_text))


def codegen_attr_count_from_script(ir_text: str) -> int:
    pattern = r"[\"']Codegen[\"']\s*:\s*[\"']waveaccel[\"']"
    return len(re.findall(pattern, ir_text))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Partition Relax graph for the WaveAccel BYOC backend."
    )
    parser.add_argument("--onnx", default="artifacts/wave_model.onnx")
    parser.add_argument("--sram-kib", type=int, default=4)
    parser.add_argument("--transient-bytes", type=int, default=392)
    parser.add_argument(
        "--summary",
        default="artifacts/waveaccel_partition_summary.json",
    )
    parser.add_argument("--show-ir", action="store_true")
    args = parser.parse_args()

    model_path = Path(args.onnx)
    if not model_path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {model_path}")

    target = WaveAccelTargetContract(
        sram_bytes=args.sram_kib * 1024,
        transient_bytes=args.transient_bytes,
    )

    model = onnx.load(model_path)
    onnx.checker.check_model(model)
    target.validate_model_ops(node.op_type for node in model.graph.node)

    relax_mod = from_onnx(
        model,
        shape_dict={"wave_samples": [1, 64]},
        dtype_dict={"wave_samples": target.dtype},
    )

    dataflow_mod = ConvertToDataflow()(relax_mod)

    partitioned = FuseOpsByPattern(
        waveaccel_patterns(),
        bind_constants=False,
        annotate_codegen=True,
    )(dataflow_mod)

    ir_text = partitioned.script()
    wrappers = codegen_functions(partitioned)
    composites = composite_counts_from_script(ir_text)
    codegen_attr_count = codegen_attr_count_from_script(ir_text)

    expected = Counter(
        {
            f"{BACKEND}.matmul_bias_relu": 2,
            f"{BACKEND}.matmul_bias": 1,
            f"{BACKEND}.multiply": 1,
            f"{BACKEND}.add": 1,
        }
    )

    print("WaveAccel TVM Relax BYOC partition")
    print("=" * 72)
    print(f"TVM version:                    {tvm.__version__}")
    print(f"ONNX model:                     {model_path}")
    print("ConvertToDataflow:              applied")
    print("dtype constraint:               DPL has_dtype(float32)")
    print("FusionPattern check callbacks:  none")
    print(f"backend prefix:                 {BACKEND}")
    print(f"target dtype:                   {target.dtype}")
    print(f"WaveAccel Codegen functions:    {len(wrappers)}")

    for name in wrappers:
        print(f"  {name}")

    print("\nComposite matches:")
    for name in expected:
        print(f"  {name:34s} {composites[name]}")

    if composites != expected:
        if args.show_ir:
            print("\nPartitioned Relax IR")
            print("=" * 72)
            print(ir_text)

        raise RuntimeError(
            "WaveAccel BYOC partition changed.\n"
            f"expected composites: {dict(expected)}\n"
            f"actual composites:   {dict(composites)}\n"
            "Run again with --show-ir to inspect the exact Relax structure."
        )

    expected_wrapper_count = sum(expected.values())

    if len(wrappers) != expected_wrapper_count:
        raise RuntimeError(
            "WaveAccel Codegen wrapper count changed: "
            f"expected {expected_wrapper_count}, got {len(wrappers)}"
        )

    if codegen_attr_count != expected_wrapper_count:
        raise RuntimeError(
            "TVMScript Codegen annotation count changed: "
            f"expected {expected_wrapper_count}, got {codegen_attr_count}"
        )

    summary = {
        "backend": BACKEND,
        "tvm_version": tvm.__version__,
        "model": str(model_path),
        "target_contract": target.to_plan_dict(),
        "pre_partition_passes": ["ConvertToDataflow"],
        "dtype_constraint": "DFPattern.has_dtype(float32)",
        "fusion_check_callbacks": False,
        "partition_stage": "FuseOpsByPattern",
        "bind_constants": False,
        "annotate_codegen": True,
        "run_codegen": False,
        "codegen_function_count": len(wrappers),
        "codegen_functions": wrappers,
        "composites": dict(composites),
    }

    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"\npartition summary:              {summary_path}")

    if args.show_ir:
        print("\nPartitioned Relax IR")
        print("=" * 72)
        print(ir_text)

    print("\nONNX -> Relax -> Dataflow -> WaveAccel BYOC partition: PASS")


if __name__ == "__main__":
    main()
