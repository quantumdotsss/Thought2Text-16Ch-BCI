"""Utilities for loading the data-free pretrained encoder bundle."""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file

from .adapter import ChannelAdapter, EEGEncoderWithAdapter
from .channelnet import ChannelNetConfig, ChannelNetModel


def load_pretrained_bundle(
    weights_dir: str | Path,
    *,
    precision: str = "fp32",
    device: str | torch.device = "cpu",
    freeze_backbone: bool = True,
) -> EEGEncoderWithAdapter:
    """Load the 16-channel adapter and ChannelNet weights without network access."""

    weights_dir = Path(weights_dir)
    if precision not in {"fp32", "fp16"}:
        raise ValueError("precision must be 'fp32' or 'fp16'")

    config = ChannelNetConfig.from_json_file(weights_dir / "config.json")
    backbone = ChannelNetModel(config)
    model_file = weights_dir / ("model.safetensors" if precision == "fp32" else "model_fp16.safetensors")
    backbone.load_state_dict(load_file(model_file), strict=True)

    adapter = ChannelAdapter()
    adapter_file = weights_dir / ("adapter.pth" if precision == "fp32" else "adapter_fp16.pth")
    adapter.load_checkpoint(adapter_file)

    model = EEGEncoderWithAdapter(backbone, adapter, freeze_backbone=freeze_backbone)
    if precision == "fp16":
        model = model.half()
    return model.eval().to(device)
