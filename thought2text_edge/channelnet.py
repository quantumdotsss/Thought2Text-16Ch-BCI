"""ChannelNet backbone used by Thought2Text.

The implementation is derived from the MIT-licensed upstream Thought2Text
repository and kept architecture-compatible with the released checkpoint.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from transformers import PreTrainedModel, PretrainedConfig


class ChannelNetConfig(PretrainedConfig):
    model_type = "eeg_channelnet"

    def __init__(
        self,
        in_channels: int = 1,
        temp_channels: int = 10,
        out_channels: int = 50,
        num_classes: int = 40,
        embedding_size: int = 512,
        input_width: int = 440,
        input_height: int = 128,
        temporal_dilation_list: list | None = None,
        temporal_kernel: tuple[int, int] = (1, 33),
        temporal_stride: tuple[int, int] = (1, 2),
        num_temp_layers: int = 4,
        num_spatial_layers: int = 4,
        spatial_stride: tuple[int, int] = (2, 1),
        num_residual_blocks: int = 4,
        down_kernel: int = 3,
        down_stride: int = 2,
        **kwargs,
    ) -> None:
        if temporal_dilation_list is None:
            temporal_dilation_list = [(1, 1), (1, 2), (1, 4), (1, 8), (1, 16)]
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.temp_channels = temp_channels
        self.out_channels = out_channels
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.input_width = input_width
        self.input_height = input_height
        self.temporal_dilation_list = temporal_dilation_list
        self.temporal_kernel = temporal_kernel
        self.temporal_stride = temporal_stride
        self.num_temp_layers = num_temp_layers
        self.num_spatial_layers = num_spatial_layers
        self.spatial_stride = spatial_stride
        self.num_residual_blocks = num_residual_blocks
        self.down_kernel = down_kernel
        self.down_stride = down_stride


class ConvLayer2D(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel, stride, padding, dilation):
        super().__init__()
        self.add_module("norm", nn.BatchNorm2d(in_channels))
        self.add_module("relu", nn.ReLU(True))
        self.add_module(
            "conv",
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=True,
            ),
        )
        self.add_module("drop", nn.Dropout2d(0.2))


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_layers, kernel, stride, dilations):
        super().__init__()
        if len(dilations) < n_layers:
            dilations = dilations + [dilations[-1]] * (n_layers - len(dilations))
        padding = []
        for dilation in dilations:
            filter_size = kernel[1] * dilation[1] - 1
            temporal_padding = math.floor((filter_size - 1) / 2) - (dilation[1] // 2 - 1)
            padding.append((0, temporal_padding))
        self.layers = nn.ModuleList(
            [
                ConvLayer2D(in_channels, out_channels, kernel, stride, padding[i], dilations[i])
                for i in range(n_layers)
            ]
        )

    def forward(self, inputs):
        return torch.cat([layer(inputs) for layer in self.layers], dim=1)


class SpatialBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_layers, stride, input_height):
        super().__init__()
        kernels = [((input_height // (i + 1)), 1) for i in range(n_layers)]
        paddings = [(math.floor((kernel[0] - 1) / 2), 0) for kernel in kernels]
        self.layers = nn.ModuleList(
            [
                ConvLayer2D(in_channels, out_channels, kernels[i], stride, paddings[i], 1)
                for i in range(n_layers)
            ]
        )

    def forward(self, inputs):
        return torch.cat([layer(inputs) for layer in self.layers], dim=1)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, inputs):
        output = self.relu(self.bn1(self.conv1(inputs)))
        output = self.bn2(self.conv2(output))
        return self.relu(output + inputs)


class FeaturesExtractor(nn.Module):
    def __init__(self, config: ChannelNetConfig):
        super().__init__()
        temporal_channels = config.temp_channels * config.num_temp_layers
        spatial_channels = config.out_channels * config.num_spatial_layers
        self.temporal_block = TemporalBlock(
            config.in_channels,
            config.temp_channels,
            config.num_temp_layers,
            config.temporal_kernel,
            config.temporal_stride,
            config.temporal_dilation_list,
        )
        self.spatial_block = SpatialBlock(
            temporal_channels,
            config.out_channels,
            config.num_spatial_layers,
            config.spatial_stride,
            config.input_height,
        )
        self.res_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    ResidualBlock(spatial_channels),
                    ConvLayer2D(
                        spatial_channels,
                        spatial_channels,
                        config.down_kernel,
                        config.down_stride,
                        0,
                        1,
                    ),
                )
                for _ in range(config.num_residual_blocks)
            ]
        )
        self.final_conv = ConvLayer2D(
            spatial_channels,
            config.out_channels,
            config.down_kernel,
            1,
            0,
            1,
        )

    def forward(self, inputs):
        output = self.spatial_block(self.temporal_block(inputs))
        for block in self.res_blocks:
            output = block(output)
        return self.final_conv(output)


class ChannelNetModel(PreTrainedModel):
    config_class = ChannelNetConfig
    base_model_prefix = "channelnet"

    def __init__(self, config: ChannelNetConfig):
        super().__init__(config)
        self.encoder = FeaturesExtractor(config)
        with torch.no_grad():
            probe = torch.zeros(1, config.in_channels, config.input_height, config.input_width)
            encoding_size = self.encoder(probe).reshape(-1).numel()
        self.projector = nn.Linear(encoding_size, config.embedding_size)
        self.classifier = nn.Linear(config.embedding_size, config.num_classes)

    def forward(self, inputs):
        features = self.encoder(inputs).reshape(inputs.shape[0], -1)
        embedding = self.projector(features)
        logits = self.classifier(embedding)
        return embedding, logits
