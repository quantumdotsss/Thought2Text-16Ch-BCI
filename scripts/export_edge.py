#!/usr/bin/env python3
"""Export the EEG encoder as a portable, data-free inference graph."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from thought2text_edge import load_pretrained_bundle


class FixedShapeEdgeModule(nn.Module):
    """A trace-friendly graph specialized for ``[B, 1, 16, 440]`` input."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.adapter = model.adapter.network
        self.backbone = model.backbone

    def forward(self, eeg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        adapted = self.adapter(eeg[:, 0]).unsqueeze(1)
        return self.backbone(adapted)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, default=Path("weights"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/eeg_encoder.pt"))
    parser.add_argument("--format", choices=("export", "torchscript"), default="torchscript")
    args = parser.parse_args()

    model = FixedShapeEdgeModule(
        load_pretrained_bundle(args.weights, precision="fp32", device="cpu")
    ).eval()
    example = torch.zeros(1, 1, 16, 440)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        if args.format == "export":
            exported = torch.export.export(model, (example,))
            torch.export.save(exported, args.output)
        else:
            traced = torch.jit.trace(model, example, strict=True)
            traced.save(str(args.output))

    print(f"Saved {args.format} graph to {args.output}")
    print("This is an export validation artifact, not a phone-specific runtime package.")


if __name__ == "__main__":
    main()
