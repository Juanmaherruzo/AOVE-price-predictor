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


def test_macro_lag_is_exactly_one_week() -> None:
    """The price the model sees must be week t-1, not t-2.

    Regression test. ``aove_lag_price`` is already shifted by one week in the
    ETL, so taking the macro row at t-1 shifted it twice and handed the model a
    price two weeks stale. The persistence baseline built from that feature then
    looked far weaker than a real one-week persistence, which flattered every
    comparison against it.
    """
    n, steps = 12, 3
    df = _dataframe(n)
    # Make the price series identifiable: row i carries the value i.
    df[TARGET_COL] = np.arange(n, dtype=float)
    # aove_lag_price is the target shifted one week, as align_macro_data builds it.
    df["aove_lag_price"] = df[TARGET_COL].shift(1).bfill()

    builder = TemporalSequenceBuilder(time_steps=steps)
    splits = builder.build_sequences_split(
        df, HF_COLS, MACRO_COLS, TARGET_COL, train_ratio=0.75
    )
    x_macro_train, y_train = splits["train"][1], splits["train"][2]  # type: ignore[misc]

    lag_idx = MACRO_COLS.index("aove_lag_price")
    scaler = builder.scaler_macro
    lag = x_macro_train[:, lag_idx] * scaler.scale_[lag_idx] + scaler.mean_[lag_idx]
    target = builder.inverse_transform_target(y_train)

    # Target at row t is t; the lag feature must therefore be t-1 exactly.
    assert np.allclose(lag, target - 1.0, atol=1e-4)


def test_climate_window_excludes_the_target_week() -> None:
    """The climate window must end at t-1: week t's weather is not observable."""
    n, steps = 12, 3
    df = _dataframe(n)
    df["rainfall_mm"] = np.arange(n, dtype=float)
    df[TARGET_COL] = np.arange(n, dtype=float)

    builder = TemporalSequenceBuilder(time_steps=steps)
    splits = builder.build_sequences_split(
        df, HF_COLS, MACRO_COLS, TARGET_COL, train_ratio=0.75
    )
    x_hf_train, y_train = splits["train"][0], splits["train"][2]  # type: ignore[misc]

    rain_idx = HF_COLS.index("rainfall_mm")
    scaler = builder.scaler_hf
    last_step = (
        x_hf_train[:, -1, rain_idx] * scaler.scale_[rain_idx] + scaler.mean_[rain_idx]
    )
    target = builder.inverse_transform_target(y_train)
    assert np.allclose(last_step, target - 1.0, atol=1e-4)


def test_inverse_transform_roundtrip() -> None:
    builder = TemporalSequenceBuilder(time_steps=3)
    builder.build_sequences_split(
        _dataframe(12), HF_COLS, MACRO_COLS, TARGET_COL, train_ratio=0.75
    )
    scaled = builder.scaler_target.transform(np.array([[0.5]]))
    restored = builder.inverse_transform_target(scaled.ravel())
    assert np.isclose(restored[0], 0.5, atol=1e-5)
