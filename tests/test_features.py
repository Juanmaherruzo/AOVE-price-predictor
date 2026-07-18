"""Tests for the temporal sequence builder (leakage-free split)."""

import numpy as np
import pandas as pd
import pytest

from aove.config import HF_COLS, MACRO_COLS, TARGET_COL
from aove.features import TemporalSequenceBuilder


def _dataframe(n: int = 12) -> pd.DataFrame:
    index = pd.date_range("2020-01-06", periods=n, freq="W-MON")
    data = {col: np.linspace(0, 1, n) for col in HF_COLS + MACRO_COLS + [TARGET_COL]}
    return pd.DataFrame(data, index=index)


def test_build_sequences_shapes_and_split() -> None:
    builder = TemporalSequenceBuilder(time_steps=3)
    splits = builder.build_sequences_split(
        _dataframe(12), HF_COLS, MACRO_COLS, TARGET_COL, train_ratio=0.75
    )
    x_hf_train, x_macro_train, y_train = splits["train"]  # type: ignore[misc]
    assert x_hf_train.shape[1:] == (3, len(HF_COLS))
    assert x_macro_train.shape[1] == len(MACRO_COLS)
    assert len(x_hf_train) == len(y_train)

    x_hf_val, _, _ = splits["val"]  # type: ignore[misc]
    total_windows = 12 - 3  # n - time_steps
    assert len(x_hf_train) + len(x_hf_val) == total_windows


def test_short_series_raises() -> None:
    builder = TemporalSequenceBuilder(time_steps=20)
    with pytest.raises(ValueError, match="Training set too short"):
        builder.build_sequences_split(
            _dataframe(10), HF_COLS, MACRO_COLS, TARGET_COL, train_ratio=0.8
        )


def test_inverse_transform_roundtrip() -> None:
    builder = TemporalSequenceBuilder(time_steps=3)
    builder.build_sequences_split(
        _dataframe(12), HF_COLS, MACRO_COLS, TARGET_COL, train_ratio=0.75
    )
    scaled = builder.scaler_target.transform(np.array([[0.5]]))
    restored = builder.inverse_transform_target(scaled.ravel())
    assert np.isclose(restored[0], 0.5, atol=1e-5)
