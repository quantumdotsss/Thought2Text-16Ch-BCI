"""Low-density EEG adaptation components for Thought2Text."""

from .adapter import ChannelAdapter, EEGEncoderWithAdapter
from .bundle import load_pretrained_bundle
from .channelnet import ChannelNetConfig, ChannelNetModel

__all__ = [
    "ChannelAdapter",
    "ChannelNetConfig",
    "ChannelNetModel",
    "EEGEncoderWithAdapter",
    "load_pretrained_bundle",
]
