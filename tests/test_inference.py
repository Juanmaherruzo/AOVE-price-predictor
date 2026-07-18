"""Tests for Monte Carlo Dropout inference."""

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from aove.inference import mc_dropout_predict
from aove.model import AOVEPricePredictor


def test_mc_dropout_returns_ordered_interval() -> None:
    torch.manual_seed(0)
    model = AOVEPricePredictor(hf_input_dim=5, macro_input_dim=5)
    x_hf = torch.randn(1, 10, 5)
    x_macro = torch.randn(1, 5)
    scaler = StandardScaler().fit(np.linspace(2.0, 6.0, 50).reshape(-1, 1))

    mean, low, high = mc_dropout_predict(model, x_hf, x_macro, scaler, n_samples=32)

    assert low <= high
    assert all(np.isfinite(v) for v in (mean, low, high))


def test_mc_dropout_restores_eval_mode() -> None:
    model = AOVEPricePredictor(hf_input_dim=5, macro_input_dim=5)
    scaler = StandardScaler().fit(np.linspace(2.0, 6.0, 50).reshape(-1, 1))
    mc_dropout_predict(
        model, torch.randn(1, 6, 5), torch.randn(1, 5), scaler, n_samples=8
    )
    assert not model.training  # eval mode restored after sampling
