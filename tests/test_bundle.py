from pathlib import Path

import torch

from thought2text_edge import load_pretrained_bundle


def test_released_bundle_smoke():
    weights = Path(__file__).parents[1] / "weights"
    model = load_pretrained_bundle(weights)
    with torch.inference_mode():
        embedding, logits = model(torch.zeros(1, 1, 16, 440))
    assert embedding.shape == (1, 512)
    assert logits.shape == (1, 40)
