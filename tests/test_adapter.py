import pytest
import torch

from thought2text_edge import ChannelAdapter


def test_adapter_preserves_batch_and_time_axes():
    adapter = ChannelAdapter(in_channels=16, hidden_channels=64, out_channels=128)
    output = adapter(torch.randn(2, 1, 16, 440))
    assert output.shape == (2, 1, 128, 440)
    assert adapter.parameter_count == 9_216


def test_adapter_accepts_three_dimensional_input():
    adapter = ChannelAdapter(in_channels=16, out_channels=128)
    assert adapter(torch.randn(2, 16, 100)).shape == (2, 128, 100)


def test_identity_path_is_valid():
    adapter = ChannelAdapter(in_channels=128, out_channels=128)
    inputs = torch.randn(2, 1, 128, 20)
    assert torch.equal(adapter(inputs), inputs)


def test_adapter_rejects_wrong_channel_count():
    adapter = ChannelAdapter(in_channels=16)
    with pytest.raises(ValueError, match="expects 16 channels"):
        adapter(torch.randn(1, 1, 8, 440))
