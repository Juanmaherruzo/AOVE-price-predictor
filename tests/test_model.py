"""Tests for the bimodal LSTM architecture."""

import torch

from aove.model import AOVEPricePredictor


def test_forward_returns_one_value_per_sample() -> None:
    model = AOVEPricePredictor(hf_input_dim=5, macro_input_dim=5)
    x_hf = torch.randn(4, 10, 5)  # (batch, time_steps, hf_features)
    x_macro = torch.randn(4, 5)  # (batch, macro_features)
    out = model(x_hf, x_macro)
    assert out.shape == (4, 1)
    assert torch.isfinite(out).all()


def test_forward_is_deterministic_in_eval_mode() -> None:
    model = AOVEPricePredictor(hf_input_dim=5, macro_input_dim=5).eval()
    x_hf = torch.randn(2, 8, 5)
    x_macro = torch.randn(2, 5)
    with torch.no_grad():
        first = model(x_hf, x_macro)
        second = model(x_hf, x_macro)
    assert torch.allclose(first, second)
