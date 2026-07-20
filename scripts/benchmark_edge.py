#!/usr/bin/env python3
"""Run a synthetic CPU edge-readiness benchmark without EEG data."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import torch

from thought2text_edge import load_pretrained_bundle


def percentile(values: list[float], fraction: float) -> float:
    return sorted(values)[round((len(values) - 1) * fraction)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, default=Path("weights"))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    model = load_pretrained_bundle(args.weights, precision="fp32", device="cpu")
    sample = torch.zeros(1, 1, 16, 440)

    with torch.inference_mode():
        for _ in range(args.warmup):
            model(sample)
        timings = []
        for _ in range(args.iterations):
            started = time.perf_counter()
            model(sample)
            timings.append((time.perf_counter() - started) * 1_000)

    adapter_parameters = model.adapter.parameter_count
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    report = {
        "benchmark_scope": "desktop CPU proxy; not an on-device mobile measurement",
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "threads": args.threads,
        "input_shape": list(sample.shape),
        "input_channel_reduction": "128 to 16 (8x)",
        "adapter_parameters": adapter_parameters,
        "total_encoder_parameters": total_parameters,
        "latency_ms_mean": round(statistics.mean(timings), 3),
        "latency_ms_median": round(statistics.median(timings), 3),
        "latency_ms_p95": round(percentile(timings, 0.95), 3),
    }
    print(json.dumps(report, indent=2))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
