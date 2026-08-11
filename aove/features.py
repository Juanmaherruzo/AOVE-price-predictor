"""Temporal sequence construction with a leakage-free chronological split."""

import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

Split = dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]


class TemporalSequenceBuilder:
    """Transform a 1D timeline into overlapping historical windows.

    Scalers are fitted on the training rows only; the sequence split index is
    derived from the dataframe split index so both share the same temporal
    boundary (``split_seq_idx = df_split_idx - time_steps``), guaranteeing that
    no training window predicts a validation-period target.

    Alignment for a target at row ``t``:

    - ``x_hf`` is the climate window ``[t - time_steps, t - 1]``. Week ``t``'s
      own weather is excluded — it is not observable at forecast time.
    - ``x_macro`` is the macro row at ``t``. Those columns are lagged at source
      (publication delay in the ETL, plus ``aove_lag_weeks`` on the price), so
      reading row ``t`` yields the intended one-week price lag rather than
      compounding the shift.

    ``tests/test_features.py::test_macro_lag_is_exactly_one_week`` pins this.
    """

    def __init__(self, time_steps: int = 104) -> None:
        self.time_steps = time_steps
        self.scaler_hf = StandardScaler()
        self.scaler_macro = StandardScaler()
        self.scaler_target = StandardScaler()

    def build_sequences_split(
        self,
        df_final: pd.DataFrame,
        hf_cols: list[str],
        macro_cols: list[str],
        target_col: str,
        train_ratio: float = 0.8,
    ) -> dict[str, object]:
        """Build sliding-window sequences with a strictly chronological split."""
        logger.info("Building sequences with chronological train/val split...")

        n = len(df_final)
        df_split_idx = int(n * train_ratio)

        if df_split_idx <= self.time_steps:
            raise ValueError(
                f"Training set too short: {df_split_idx} rows < "
                f"time_steps={self.time_steps}. "
                "Reduce time_steps or increase train_ratio."
            )

        df_train = df_final.iloc[:df_split_idx]
        self.scaler_hf.fit(df_train[hf_cols].to_numpy())
        self.scaler_macro.fit(df_train[macro_cols].to_numpy())
        self.scaler_target.fit(df_train[[target_col]].to_numpy())

        hf_scaled = self.scaler_hf.transform(df_final[hf_cols].to_numpy()).astype(
            np.float32
        )
        macro_scaled = self.scaler_macro.transform(
            df_final[macro_cols].to_numpy()
        ).astype(np.float32)
        target_scaled = (
            self.scaler_target.transform(df_final[[target_col]].to_numpy())
            .astype(np.float32)
            .ravel()
        )

        # Zero-copy sliding window view; drop the last window (it has no target).
        # The climate window for target row t ends at t-1: week t's own weather is
        # not observable when the forecast is made.
        hf_windows = np.lib.stride_tricks.sliding_window_view(
            hf_scaled, window_shape=self.time_steps, axis=0
        )
        x_hf = hf_windows[:-1].transpose(0, 2, 1).copy()
        # The macro snapshot is taken from row t itself, not t-1. Every macro
        # column is already lagged at source: the bulletin figures carry a
        # publication delay applied in the ETL, and ``aove_lag_price`` is the
        # target shifted by ``aove_lag_weeks``. Reading row t-1 here applied the
        # shift a second time, so the model saw a price two weeks stale while
        # the configuration and the docs both claimed one.
        x_macro = macro_scaled[self.time_steps : n]
        y = target_scaled[self.time_steps :]

        split_seq_idx = df_split_idx - self.time_steps
        logger.info(
            "Total windows: %d | Train: %d | Val: %d",
            len(x_hf),
            split_seq_idx,
            len(x_hf) - split_seq_idx,
        )

        target_dates = df_final.index[self.time_steps :]
        val_dates = target_dates[split_seq_idx:]

        return {
            "train": (
                x_hf[:split_seq_idx],
                x_macro[:split_seq_idx],
                y[:split_seq_idx],
            ),
            "val": (
                x_hf[split_seq_idx:],
                x_macro[split_seq_idx:],
                y[split_seq_idx:],
            ),
            "val_dates": val_dates,
        }

    def inverse_transform_target(self, y_scaled: np.ndarray) -> np.ndarray:
        """Convert scaled predictions back to real-world EUR/kg."""
        inverted: np.ndarray = (
            self.scaler_target.inverse_transform(y_scaled.reshape(-1, 1))
            .ravel()
            .astype(np.float32)
        )
        return inverted
