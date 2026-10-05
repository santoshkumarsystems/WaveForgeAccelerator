from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import numpy as np
import openvino as ov
import pandas as pd


FEATURES = [f"x{i}" for i in range(64)]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--device", default="CPU")
    p.add_argument("--iterations", type=int, default=100)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    core = ov.Core()

    if args.device not in core.available_devices:
        raise SystemExit(
            f"{args.device!r} is unavailable. "
            f"Available: {core.available_devices}"
        )

    model = core.read_model(str(args.model))
    compiled = core.compile_model(model, args.device)

    df = pd.read_csv(args.dataset).head(1)
    x = df[FEATURES].to_numpy(np.float32)

    request = compiled.create_infer_request()
    request.infer({"wave_samples": x})  # warm-up

    samples_ms = []
    for _ in range(args.iterations):
        start = time.perf_counter_ns()
        result = request.infer({"wave_samples": x})
        stop = time.perf_counter_ns()
        samples_ms.append((stop - start) / 1e6)

    output = result[compiled.output(0)]

    print(f"device: {args.device}")
    print(f"prediction [amplitude, wavelength]: {output[0].tolist()}")
    print(f"median latency: {statistics.median(samples_ms):.6f} ms")
    print(f"mean latency:   {statistics.mean(samples_ms):.6f} ms")


if __name__ == "__main__":
    main()
