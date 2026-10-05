from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import torch

from wave_model import PhysicalOutputModel, WaveRegressor


FEATURES = [f"x{i}" for i in range(64)]
TARGETS = ["amplitude", "wavelength"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--onnx", type=Path, required=True)
    p.add_argument("--rows", type=int, default=64)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    model_proto = onnx.load(args.onnx)
    onnx.checker.check_model(model_proto)

    state = torch.load(args.checkpoint, map_location="cpu")
    base = WaveRegressor()
    base.load_state_dict(state["state_dict"])
    torch_model = PhysicalOutputModel(
        base, state["target_mean"], state["target_std"]
    ).eval()

    df = pd.read_csv(args.dataset).head(args.rows)
    x = df[FEATURES].to_numpy(np.float32)
    truth = df[TARGETS].to_numpy(np.float32)

    with torch.no_grad():
        torch_out = torch_model(torch.from_numpy(x)).numpy()

    session = ort.InferenceSession(
        str(args.onnx),
        providers=["CPUExecutionProvider"],
    )
    ort_out = session.run(
        ["wave_parameters"], {"wave_samples": x}
    )[0]

    parity = np.max(np.abs(torch_out - ort_out))
    mae = np.mean(np.abs(ort_out - truth), axis=0)

    print(f"PyTorch <-> ONNX max_abs_error: {parity:.8e}")
    print(f"ONNX MAE amplitude:  {mae[0]:.6f}")
    print(f"ONNX MAE wavelength: {mae[1]:.6f}")

    if parity > 1e-4:
        raise SystemExit("ONNX parity check failed.")

    print("ONNX parity: PASS")


if __name__ == "__main__":
    main()
