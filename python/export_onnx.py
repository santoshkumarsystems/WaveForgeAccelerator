from __future__ import annotations

import argparse
from pathlib import Path

import torch

from wave_model import PhysicalOutputModel, WaveRegressor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    state = torch.load(args.checkpoint, map_location="cpu")

    base = WaveRegressor()
    base.load_state_dict(state["state_dict"])
    model = PhysicalOutputModel(
        base,
        state["target_mean"],
        state["target_std"],
    ).eval()

    dummy = torch.zeros(1, 64, dtype=torch.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        args.output,
        input_names=["wave_samples"],
        output_names=["wave_parameters"],
        dynamic_axes={
            "wave_samples": {0: "batch"},
            "wave_parameters": {0: "batch"},
        },
        opset_version=18,
    )
    print(f"exported ONNX: {args.output}")


if __name__ == "__main__":
    main()
