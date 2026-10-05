"""
Train the WaveAccel compact neural network from cuWaves-generated data.

The V1 model learns the inverse mapping:

    64 spatial wave samples -> amplitude, wavelength

Training is performed from scratch using a deterministic train/validation split.

Author: Santosh Kumar
Copyright 2026 Santosh Kumar
SPDX-License-Identifier: Apache-2.0
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from wave_model import WaveRegressor


FEATURES = [f"x{i}" for i in range(64)]
TARGETS = ["amplitude", "wavelength"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    df = pd.read_csv(args.dataset)
    missing = [c for c in FEATURES + TARGETS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    x = torch.tensor(df[FEATURES].to_numpy(np.float32))
    y_physical = torch.tensor(df[TARGETS].to_numpy(np.float32))

    n = len(x)
    if n < 100:
        raise ValueError("Use at least 100 cuWaves samples for the V1 training run.")

    perm = torch.randperm(n)
    split = int(0.8 * n)
    train_idx, val_idx = perm[:split], perm[split:]

    target_mean = y_physical[train_idx].mean(dim=0)
    target_std = y_physical[train_idx].std(dim=0).clamp_min(1e-6)
    y = (y_physical - target_mean) / target_std

    train_ds = TensorDataset(x[train_idx], y[train_idx])
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True
    )

    model = WaveRegressor()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for xb, yb in train_loader:
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            total += loss.detach().item() * len(xb)
            count += len(xb)

        if epoch == 1 or epoch % 20 == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                pred_norm = model(x[val_idx])
                pred = pred_norm * target_std + target_mean
                mae = (pred - y_physical[val_idx]).abs().mean(dim=0)
            print(
                f"epoch={epoch:4d} "
                f"train_mse={total / count:.6f} "
                f"val_mae_amplitude={mae[0]:.6f} "
                f"val_mae_wavelength={mae[1]:.6f}"
            )

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "target_mean": target_mean,
            "target_std": target_std,
            "features": FEATURES,
            "targets": TARGETS,
            "seed": args.seed,
        },
        args.checkpoint,
    )
    print(f"saved checkpoint: {args.checkpoint}")


if __name__ == "__main__":
    main()
