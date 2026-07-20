#!/usr/bin/env python3
"""Create FP16 storage checkpoints from the released FP32 model bundle."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def to_fp16(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.half() if tensor.is_floating_point() else tensor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, default=Path("weights"))
    args = parser.parse_args()

    model_input = args.weights / "model.safetensors"
    model_output = args.weights / "model_fp16.safetensors"
    adapter_input = args.weights / "adapter.pth"
    adapter_output = args.weights / "adapter_fp16.pth"

    model_state = {key: to_fp16(value) for key, value in load_file(model_input).items()}
    save_file(model_state, model_output)

    try:
        adapter_state = torch.load(adapter_input, map_location="cpu", weights_only=True)
    except TypeError:
        adapter_state = torch.load(adapter_input, map_location="cpu")
    torch.save({key: to_fp16(value) for key, value in adapter_state.items()}, adapter_output)

    fp32_bytes = model_input.stat().st_size + adapter_input.stat().st_size
    fp16_bytes = model_output.stat().st_size + adapter_output.stat().st_size
    print(f"FP32 bundle: {fp32_bytes / 2**20:.2f} MiB")
    print(f"FP16 bundle: {fp16_bytes / 2**20:.2f} MiB")
    print(f"Storage reduction: {(1 - fp16_bytes / fp32_bytes) * 100:.1f}%")


if __name__ == "__main__":
    main()
