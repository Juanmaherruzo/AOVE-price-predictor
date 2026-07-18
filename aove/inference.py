"""Inference utilities: model loading, tensor building and MC-Dropout sampling."""

import datetime as dt
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from aove.config import DEVICE, HF_COLS, MACRO_COLS, TARGET_COL, settings
from aove.etl import MarketETLPipeline
from aove.model import AOVEPricePredictor

logger = logging.getLogger(__name__)


def load_model(checkpoint_path: Path) -> AOVEPricePredictor:
    """Instantiate the network and load weights from ``checkpoint_path``."""
    model = AOVEPricePredictor(
        hf_input_dim=len(HF_COLS),
        macro_input_dim=len(MACRO_COLS),
    ).to(DEVICE)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()
    return model


def build_tensors(
    current_price: float,
    df_climate: pd.DataFrame,
    df_macro: pd.DataFrame,
    time_steps: int = 104,
    train_ratio: float = 0.8,
) -> tuple[torch.Tensor, torch.Tensor, StandardScaler, dt.date, dt.date]:
    """Run the ETL on in-memory frames and prepare the model input window.

    The current week's price is injected as the most recent observation and the
    lag feature of the last row is forced to it, so the model actually reacts to
    the user-provided input.
    """
    df_climate = df_climate.copy()
    df_macro = df_macro.copy()

    # Extend climate one week forward (clone last week — climate unknown yet).
    last_clim_date = df_climate["date"].max()
    df_clim_new = df_climate[df_climate["date"] == last_clim_date].copy()
    df_clim_new["date"] = last_clim_date + pd.Timedelta(weeks=1)
    df_climate = pd.concat([df_climate, df_clim_new], ignore_index=True)

    # Extend macro one week forward with the current price injected.
    df_macro = df_macro.sort_values("reference_date").reset_index(drop=True)
    df_macro_new = df_macro.iloc[-1:].copy()
    df_macro_new["reference_date"] = df_macro["reference_date"].iloc[-1] + pd.Timedelta(
        weeks=1
    )
    df_macro_new["aove_price_eur_kg"] = current_price
    df_macro = pd.concat([df_macro, df_macro_new], ignore_index=True)

    etl = MarketETLPipeline(publish_delay_days=settings.publish_delay_days)
    df_agg = etl.aggregate_climate(df_climate)
    df_final = etl.align_macro_data(
        df_agg, df_macro, aove_lag_weeks=settings.aove_lag_weeks
    )

    df_final.iloc[-1, df_final.columns.get_loc("aove_lag_price")] = np.float32(
        current_price
    )

    if len(df_final) < time_steps:
        raise ValueError(f"Need >= {time_steps} weeks of data, got {len(df_final)}.")

    split_idx = int(len(df_final) * train_ratio)
    df_train = df_final.iloc[:split_idx]

    sc_hf = StandardScaler().fit(df_train[HF_COLS].to_numpy())
    sc_mac = StandardScaler().fit(df_train[MACRO_COLS].to_numpy())
    sc_tgt = StandardScaler().fit(df_train[[TARGET_COL]].to_numpy())

    window = df_final.iloc[-time_steps:]
    hf_sc = sc_hf.transform(window[HF_COLS].to_numpy()).astype(np.float32)
    macro_sc = sc_mac.transform(window[MACRO_COLS].to_numpy()).astype(np.float32)

    x_hf = torch.tensor(hf_sc, dtype=torch.float32).unsqueeze(0)
    x_macro = torch.tensor(macro_sc[-1], dtype=torch.float32).unsqueeze(0)

    last_date = df_final.index[-1].date()
    pred_date = (df_final.index[-1] + dt.timedelta(weeks=1)).date()
    return x_hf, x_macro, sc_tgt, last_date, pred_date


def mc_dropout_predict(
    model: AOVEPricePredictor,
    x_hf: torch.Tensor,
    x_macro: torch.Tensor,
    scaler_target: StandardScaler,
    n_samples: int = 200,
) -> tuple[float, float, float]:
    """Run N stochastic forward passes with dropout active (Gal & Ghahramani, 2016).

    Returns ``(mean, p10, p90)`` in EUR/kg — an 80% empirical confidence interval.
    """
    model.train()  # keep dropout active during inference
    preds: list[float] = []
    with torch.no_grad():
        for _ in range(n_samples):
            out = model(x_hf.to(DEVICE), x_macro.to(DEVICE)).cpu().numpy()
            preds.append(
                float(scaler_target.inverse_transform(out.reshape(-1, 1))[0, 0])
            )
    model.eval()

    arr = np.array(preds)
    return (
        float(np.mean(arr)),
        float(np.percentile(arr, 10)),
        float(np.percentile(arr, 90)),
    )
