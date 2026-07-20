"""Learnable channel adaptation for low-density EEG input."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class ChannelAdapter(nn.Module):
    """Map low-density EEG channels to the 128-channel ChannelNet layout.

    The two 1x1 convolutions mix channels independently at every time step, so
    the temporal resolution is unchanged. Both ``[B, C, T]`` and
    ``[B, 1, C, T]`` inputs are accepted.
    """

    def __init__(
        self,
        in_channels: int = 16,
        hidden_channels: int = 64,
        out_channels: int = 128,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if min(in_channels, hidden_channels, out_channels) <= 0:
            raise ValueError("Channel counts must be positive integers.")

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        if in_channels == out_channels:
            self.network: nn.Module = nn.Identity()
        else:
            self.network = nn.Sequential(
                nn.Conv1d(in_channels, hidden_channels, kernel_size=1, bias=False),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv1d(hidden_channels, out_channels, kernel_size=1, bias=False),
            )
            for layer in self.network.modules():
                if isinstance(layer, nn.Conv1d):
                    nn.init.kaiming_uniform_(layer.weight, a=5**0.5)

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        had_image_axis = eeg.ndim == 4
        if had_image_axis:
            if eeg.shape[1] != 1:
                raise ValueError(f"Expected [B, 1, C, T], received {tuple(eeg.shape)}")
            eeg = eeg[:, 0]
        elif eeg.ndim != 3:
            raise ValueError(f"Expected [B, C, T] or [B, 1, C, T], received {tuple(eeg.shape)}")

        if eeg.shape[1] != self.in_channels:
            raise ValueError(
                f"Adapter expects {self.in_channels} channels, received {eeg.shape[1]}"
            )

        adapted = self.network(eeg)
        return adapted.unsqueeze(1) if had_image_axis else adapted

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def load_checkpoint(self, checkpoint: str | Path) -> None:
        try:
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        except TypeError:  # PyTorch < 2.0 compatibility
            state = torch.load(checkpoint, map_location="cpu")

        # The research checkpoint used ``net`` as the Sequential attribute.
        state = {key.replace("net.", "network.", 1): value for key, value in state.items()}
        self.load_state_dict(state, strict=True)


class EEGEncoderWithAdapter(nn.Module):
    """Compose a channel adapter with the pretrained ChannelNet backbone."""

    def __init__(
        self,
        backbone: nn.Module,
        adapter: ChannelAdapter | None = None,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.adapter = adapter or ChannelAdapter()
        self.backbone = backbone
        if freeze_backbone:
            self.backbone.requires_grad_(False)

    def forward(self, eeg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if eeg.ndim == 3:
            eeg = eeg.unsqueeze(1)
        return self.backbone(self.adapter(eeg))
