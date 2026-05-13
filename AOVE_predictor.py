"""
AOVE Oracle CLI — Production Inference Interface
=============================================================================
Interactive command-line tool for weekly AOVE price prediction.

Wraps the AOVEInferenceEngine from predict.py with:
  - Interactive user input for the current week's market price
  - Monte Carlo Dropout for confidence interval estimation
  - Clean terminal output with trend signal

Usage
-----
    python oracle_cli.py \\
        --model    best_aove_model.pth \\
        --climate  ./data/climate_dataset.csv \\
        --macro    ./data/macro_dataset.csv

    # Inside Jupyter: edit NOTEBOOK CONFIG below and run
"""

import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional
from datetime import timedelta
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

# ==============================================================================
# MODEL — must match aove_predictor.py exactly
# ==============================================================================
class AOVEPricePredictor(nn.Module):
    def __init__(
        self,
        hf_input_dim: int,
        macro_input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=hf_input_dim, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim + macro_input_dim, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x_hf: torch.Tensor, x_macro: torch.Tensor) -> torch.Tensor:
        h0 = torch.zeros(self.num_layers, x_hf.size(0), self.hidden_dim,
                         dtype=torch.float32, device=x_hf.device)
        c0 = torch.zeros(self.num_layers, x_hf.size(0), self.hidden_dim,
                         dtype=torch.float32, device=x_hf.device)
        out, _ = self.lstm(x_hf, (h0, c0))
        return self.fc(torch.cat([self.dropout(out[:, -1, :]), x_macro], dim=1))

# ==============================================================================
# ETL — must match aove_predictor.py exactly
# ==============================================================================
class MarketETLPipeline:
    def __init__(self, publish_delay_days: int = 15) -> None:
        self.publish_delay_days = publish_delay_days

    def aggregate_climate(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = ['rainfall_mm', 'temp_max_c', 'water_deficit_mm']
        def wavg(g):
            w = g['surface_ha'].values.astype(np.float64)
            return pd.Series(
                np.average(g[cols].values, weights=w, axis=0) if w.sum() > 0
                else g[cols].mean().values, index=cols)
        agg = (df.groupby('date')[cols + ['surface_ha']]
               .apply(wavg, include_groups=False).reset_index())
        wn = agg['date'].dt.isocalendar().week.astype(float)
        agg['time_sin'] = np.sin(wn * 2 * np.pi / 52).astype(np.float32)
        agg['time_cos'] = np.cos(wn * 2 * np.pi / 52).astype(np.float32)
        return agg.set_index('date')

    def align_macro_data(self, df_clim: pd.DataFrame, df_macro: pd.DataFrame,
                         aove_lag: int = 1) -> pd.DataFrame:
        m = df_macro.copy()
        m['publish_date'] = m['reference_date'] + pd.DateOffset(days=self.publish_delay_days)
        ms = m.sort_values('publish_date').set_index('publish_date')
        feat_cols = ['stock_delta_pct', 'surface_delta_pct', 'ipc_monthly', 'diesel_price_eur']
        df = pd.merge_asof(df_clim.sort_index(), ms[feat_cols + ['aove_price_eur_kg']],
                           left_index=True, right_index=True, direction='backward')
        if aove_lag > 0:
            df['aove_lag_price'] = df['aove_price_eur_kg'].shift(aove_lag).bfill()
        df.ffill(inplace=True)
        df.fillna(0.0, inplace=True)
        return df.astype(np.float32)

# ==============================================================================
# MONTE CARLO DROPOUT — confidence interval
# ==============================================================================
def mc_dropout_predict(
    model: AOVEPricePredictor,
    X_hf: torch.Tensor,
    X_macro: torch.Tensor,
    scaler_target: StandardScaler,
    n_samples: int = 200,
    device: torch.device = torch.device("cpu"),
) -> Tuple[float, float, float]:
    """
    Runs N stochastic forward passes with dropout ACTIVE (train mode) to
    estimate the predictive distribution. Returns (mean, lower_80, upper_80).

    Monte Carlo Dropout (Gal & Ghahramani, 2016) treats each forward pass
    with active dropout as a sample from an approximate Bayesian posterior.
    The spread of predictions across N samples gives an empirical confidence
    interval without requiring any additional training.
    """
    model.train()   # Keep dropout active during inference
    preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            out = model(X_hf.to(device), X_macro.to(device)).cpu().numpy()
            preds.append(float(scaler_target.inverse_transform(out.reshape(-1, 1))[0, 0]))

    model.eval()    # Restore eval mode after sampling
    preds_arr = np.array(preds)
    mean  = float(np.mean(preds_arr))
    lower = float(np.percentile(preds_arr, 10))   # 80% interval
    upper = float(np.percentile(preds_arr, 90))
    return mean, lower, upper

# ==============================================================================
# INFERENCE ENGINE
# ==============================================================================
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps"  if torch.backends.mps.is_available() else "cpu"
)

HF_COLS    = ['rainfall_mm', 'temp_max_c', 'water_deficit_mm', 'time_sin', 'time_cos']
MACRO_COLS = ['stock_delta_pct', 'surface_delta_pct', 'ipc_monthly',
              'diesel_price_eur', 'aove_lag_price']
TARGET_COL = 'aove_price_eur_kg'

def build_engine(checkpoint: str, climate_csv: str, macro_csv: str,
                 current_price: float, time_steps: int = 104,
                 train_ratio: float = 0.8) -> Tuple:
    """
    Loads data, fits scalers on training split, updates the last row of
    macro_dataset with the user-provided current price, and prepares tensors.
    """
    # Load data
    df_mun   = pd.read_csv(climate_csv, parse_dates=['date'])
    df_macro = pd.read_csv(macro_csv,   parse_dates=['reference_date'])

    # Inject current week's price as the most recent observation
    df_macro = df_macro.sort_values('reference_date').reset_index(drop=True)
    df_macro.loc[df_macro.index[-1], 'aove_price_eur_kg'] = current_price

    # ETL
    etl      = MarketETLPipeline()
    df_clim  = etl.aggregate_climate(df_mun)
    df_final = etl.align_macro_data(df_clim, df_macro, aove_lag=1)

    # Force the lag feature of the last row to reflect the user-provided price.
    # Without this, aove_lag_price on the last row = previous week from CSV,
    # making the model blind to the current_price input.
    df_final.iloc[-1, df_final.columns.get_loc('aove_lag_price')] = np.float32(current_price)

    # Fit scalers on training split only
    n         = len(df_final)
    split_idx = int(n * train_ratio)
    df_train  = df_final.iloc[:split_idx]

    sc_hf  = StandardScaler().fit(df_train[HF_COLS].values)
    sc_mac = StandardScaler().fit(df_train[MACRO_COLS].values)
    sc_tgt = StandardScaler().fit(df_train[[TARGET_COL]].values)

    # Build input window (last time_steps rows)
    if len(df_final) < time_steps:
        raise ValueError(f"Need >= {time_steps} weeks of data, got {len(df_final)}.")

    window    = df_final.iloc[-time_steps:]
    hf_sc     = sc_hf.transform(window[HF_COLS].values).astype(np.float32)
    macro_sc  = sc_mac.transform(window[MACRO_COLS].values).astype(np.float32)

    X_hf    = torch.tensor(hf_sc,        dtype=torch.float32).unsqueeze(0)
    X_macro = torch.tensor(macro_sc[-1], dtype=torch.float32).unsqueeze(0)

    # Prediction date = next Monday after last data row
    pred_date = (df_final.index[-1] + timedelta(weeks=1)).date()

    return X_hf, X_macro, sc_tgt, pred_date, df_final.index[-1].date()

# ==============================================================================
# CLI
# ==============================================================================
class AOVEOracleCLI:
    """
    Interactive CLI for weekly AOVE price prediction.
    Connects user input → ETL → MC Dropout inference → formatted output.
    """

    def __init__(self, model_path: str, climate_path: str, macro_path: str,
                 time_steps: int = 104, train_ratio: float = 0.8,
                 mc_samples: int = 200) -> None:
        self.model_path   = model_path
        self.climate_path = climate_path
        self.macro_path   = macro_path
        self.time_steps   = time_steps
        self.train_ratio  = train_ratio
        self.mc_samples   = mc_samples

    def _validate_paths(self) -> None:
        for p in [self.model_path, self.climate_path, self.macro_path]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"CRITICAL: Missing required file at {p}")

    def _load_model(self) -> AOVEPricePredictor:
        model = AOVEPricePredictor(
            hf_input_dim=len(HF_COLS),
            macro_input_dim=len(MACRO_COLS),
        ).to(DEVICE)
        model.load_state_dict(torch.load(self.model_path, map_location=DEVICE))
        model.eval()
        return model

    def get_user_input(self) -> float:
        """Interactive prompt for the current week's market price."""
        print("\n" + "="*52)
        print("  AOVE ORACLE — Weekly Price Predictor")
        print("="*52)
        while True:
            try:
                raw = input("\n  Enter current POOLred/MAPA price (EUR/kg) [e.g. 4.85]: ")
                price = float(raw.strip().replace(",", "."))
                if not (1.0 <= price <= 15.0):
                    print("  Warning: price outside 1.0-15.0 EUR/kg. Verify and confirm.")
                    confirm = input("  Continue anyway? (y/n): ").strip().lower()
                    if confirm != 'y':
                        continue
                return price
            except ValueError:
                print("  Invalid input — please enter a number (e.g. 4.85).")

    def run_inference(self, current_price: float) -> Tuple[float, float, float, object, object]:
        """Full pipeline: ETL → model load → MC Dropout → results."""
        print("\n  Loading data and running inference...")

        X_hf, X_macro, sc_tgt, pred_date, last_date = build_engine(
            checkpoint    = self.model_path,
            climate_csv   = self.climate_path,
            macro_csv     = self.macro_path,
            current_price = current_price,
            time_steps    = self.time_steps,
            train_ratio   = self.train_ratio,
        )

        model = self._load_model()

        mean_price, lower, upper = mc_dropout_predict(
            model, X_hf, X_macro, sc_tgt,
            n_samples=self.mc_samples, device=DEVICE,
        )

        return mean_price, lower, upper, pred_date, last_date

    def display_results(self, current_price: float, mean: float,
                        lower: float, upper: float,
                        pred_date: object, last_date: object) -> None:
        """Renders the prediction to the terminal."""
        trend     = "UP" if mean > current_price else "DOWN"
        trend_sym = "+" if mean > current_price else "-"
        delta     = abs(mean - current_price)
        width     = upper - lower

        # Rough probability: fraction of MC interval above current price
        # (simplified signal for non-technical users)
        direction_pct = int(100 * (mean - lower) / width) if width > 0 else 50
        direction_pct = max(5, min(95, direction_pct))

        print("\n" + "="*52)
        print("  AOVE ORACLE — NEXT WEEK FORECAST")
        print("="*52)
        print(f"  Data up to       : {last_date}")
        print(f"  Prediction week  : {pred_date}")
        print("-"*52)
        print(f"  Current price    :  {current_price:.2f} EUR/kg")
        print(f"  Predicted price  :  {mean:.2f} EUR/kg  ({trend_sym}{delta:.2f})")
        print(f"  80% range        :  {lower:.2f} — {upper:.2f} EUR/kg")
        print(f"  Trend signal     :  {trend}  ({direction_pct}% confidence)")
        print("="*52)
        print(f"  Note: range based on {self.mc_samples} MC Dropout samples.")
        print("="*52 + "\n")

    def run(self) -> None:
        """Main entry point — full interactive session."""
        self._validate_paths()
        current_price = self.get_user_input()
        mean, lower, upper, pred_date, last_date = self.run_inference(current_price)
        self.display_results(current_price, mean, lower, upper, pred_date, last_date)

# ==============================================================================
# ENTRY POINT
# ==============================================================================
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AOVE Oracle CLI — weekly EVOO price predictor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model",       type=str, default="best_aove_model.pth")
    parser.add_argument("--climate",     type=str, default="./data/climate_dataset.csv")
    parser.add_argument("--macro",       type=str, default="./data/macro_dataset.csv")
    parser.add_argument("--time_steps",  type=int, default=104)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--mc_samples",  type=int, default=200,
                        help="Monte Carlo Dropout samples for confidence interval (default: 200).")

    # Strip Jupyter/VS Code injected flags
    sys.argv = [sys.argv[0]] + [
        a for a in sys.argv[1:]
        if not a.startswith('--f=') and not a.startswith('-f=')
    ]
    args, _ = parser.parse_known_args()
    return args


if __name__ == "__main__":
    # ==========================================================================
    # NOTEBOOK CONFIG — edit when running inside Jupyter
    # ==========================================================================
    NOTEBOOK_MODEL       = "./AOVE_model/best_aove_model.pth"
    NOTEBOOK_CLIMATE     = "./data/climate_dataset.csv"
    NOTEBOOK_MACRO       = "./data/macro_dataset.csv"
    NOTEBOOK_TIME_STEPS  = 104
    NOTEBOOK_TRAIN_RATIO = 0.8
    NOTEBOOK_MC_SAMPLES  = 200
    # ==========================================================================

    args = _parse_args()
    in_jupyter = any("ipykernel" in a or "jupyter" in a for a in sys.argv)
    if in_jupyter:
        args.model       = NOTEBOOK_MODEL
        args.climate     = NOTEBOOK_CLIMATE
        args.macro       = NOTEBOOK_MACRO
        args.time_steps  = NOTEBOOK_TIME_STEPS
    
        args.train_ratio = NOTEBOOK_TRAIN_RATIO
        args.mc_samples  = NOTEBOOK_MC_SAMPLES

    oracle = AOVEOracleCLI(
        model_path   = args.model,
        climate_path = args.climate,
        macro_path   = args.macro,
        time_steps   = args.time_steps,
        train_ratio  = args.train_ratio,
        mc_samples   = args.mc_samples,
    )
    oracle.run()