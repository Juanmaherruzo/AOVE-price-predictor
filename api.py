"""
AOVE Oracle REST API — Production Server
=============================================================================
Self-contained FastAPI server for AOVE weekly price prediction.
No external imports from other project scripts required.

Run from terminal (NOT from Jupyter):
    pip install fastapi uvicorn
    uvicorn api:app --reload --port 8000

Then open:
    http://localhost:8000/docs       <- Swagger UI
    http://localhost:8000/redoc      <- ReDoc UI
    http://localhost:8000/health     <- Health check
    http://localhost:8000/predict    <- POST endpoint
"""

import os
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from typing import Optional, Tuple
from datetime import timedelta
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ==============================================================================
# CONFIGURATION — edit paths here
# ==============================================================================
MODEL_PATH   = "./AOVE_model/best_aove_model.pth"
CLIMATE_PATH = "./data/climate_dataset.csv"
MACRO_PATH   = "./data/macro_dataset.csv"
TIME_STEPS   = 104
TRAIN_RATIO  = 0.8

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps"  if torch.backends.mps.is_available() else "cpu"
)

HF_COLS    = ['rainfall_mm', 'temp_max_c', 'water_deficit_mm', 'time_sin', 'time_cos']
MACRO_COLS = ['stock_delta_pct', 'surface_delta_pct', 'ipc_monthly',
              'diesel_price_eur', 'aove_lag_price']
TARGET_COL = 'aove_price_eur_kg'

# ==============================================================================
# MODEL
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
# ETL
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
# INFERENCE
# ==============================================================================
def build_tensors(current_price: float, df_mun: pd.DataFrame, df_macro: pd.DataFrame) -> Tuple:
    """Runs ETL on in-memory dataframes, injects current price, returns model tensors."""
    df_mun   = df_mun.copy()
    df_macro = df_macro.copy()

    # Extend climate one week forward (clone last week — climate unknown yet)
    last_clim_date = df_mun['date'].max()
    df_clim_new    = df_mun[df_mun['date'] == last_clim_date].copy()
    df_clim_new['date'] = last_clim_date + pd.Timedelta(weeks=1)
    df_mun = pd.concat([df_mun, df_clim_new], ignore_index=True)

    # Extend macro one week forward with current_price injected
    df_macro = df_macro.sort_values('reference_date').reset_index(drop=True)
    df_macro_new = df_macro.iloc[-1:].copy()
    df_macro_new['reference_date']    = df_macro['reference_date'].iloc[-1] + pd.Timedelta(weeks=1)
    df_macro_new['aove_price_eur_kg'] = current_price
    df_macro = pd.concat([df_macro, df_macro_new], ignore_index=True)

    # ETL
    etl      = MarketETLPipeline()
    df_clim  = etl.aggregate_climate(df_mun)
    df_final = etl.align_macro_data(df_clim, df_macro, aove_lag=1)

    # Force lag feature on last row to current_price (critical fix)
    df_final.iloc[-1, df_final.columns.get_loc('aove_lag_price')] = np.float32(current_price)

    if len(df_final) < TIME_STEPS:
        raise ValueError(f"Need >= {TIME_STEPS} weeks of data, got {len(df_final)}.")

    # Fit scalers on training split only
    n         = len(df_final)
    split_idx = int(n * TRAIN_RATIO)
    df_train  = df_final.iloc[:split_idx]

    sc_hf  = StandardScaler().fit(df_train[HF_COLS].values)
    sc_mac = StandardScaler().fit(df_train[MACRO_COLS].values)
    sc_tgt = StandardScaler().fit(df_train[[TARGET_COL]].values)

    window   = df_final.iloc[-TIME_STEPS:]
    hf_sc    = sc_hf.transform(window[HF_COLS].values).astype(np.float32)
    macro_sc = sc_mac.transform(window[MACRO_COLS].values).astype(np.float32)

    X_hf    = torch.tensor(hf_sc,        dtype=torch.float32).unsqueeze(0)
    X_macro = torch.tensor(macro_sc[-1], dtype=torch.float32).unsqueeze(0)

    last_date = df_final.index[-1].date()
    pred_date = (df_final.index[-1] + timedelta(weeks=1)).date()

    return X_hf, X_macro, sc_tgt, last_date, pred_date

def mc_dropout_predict(
    model: AOVEPricePredictor,
    X_hf: torch.Tensor,
    X_macro: torch.Tensor,
    sc_tgt: StandardScaler,
    n_samples: int = 200,
) -> Tuple[float, float, float]:
    """Monte Carlo Dropout — returns (mean, p10, p90)."""
    model.train()
    preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            out = model(X_hf.to(DEVICE), X_macro.to(DEVICE)).cpu().numpy()
            preds.append(float(sc_tgt.inverse_transform(out.reshape(-1, 1))[0, 0]))
    model.eval()
    arr = np.array(preds)
    return float(np.mean(arr)), float(np.percentile(arr, 10)), float(np.percentile(arr, 90))

def load_model() -> AOVEPricePredictor:
    model = AOVEPricePredictor(
        hf_input_dim=len(HF_COLS),
        macro_input_dim=len(MACRO_COLS),
    ).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model


# ==============================================================================
# GLOBAL STATE — loaded once at startup, shared across all requests
# ==============================================================================
_model:    AOVEPricePredictor = None
_df_mun:   pd.DataFrame       = None
_df_macro: pd.DataFrame       = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _df_mun, _df_macro
    logger.info("Loading model and datasets into memory...")
    _model    = load_model()
    _df_mun   = pd.read_csv(CLIMATE_PATH, parse_dates=['date'])
    _df_macro = pd.read_csv(MACRO_PATH,   parse_dates=['reference_date'])
    logger.info(f"Ready — device: {DEVICE} | climate rows: {len(_df_mun)} | macro rows: {len(_df_macro)}")
    yield
    logger.info("Server shutting down.")


# ==============================================================================
# FASTAPI APP
# ==============================================================================
app = FastAPI(
    lifespan=lifespan,
    title="AOVE Oracle API",
    description=(
        "Weekly Extra Virgin Olive Oil (EVOO) origin price predictor. "
        "Powered by a bimodal LSTM trained on 10 years of climate and macro data. "
        "Confidence intervals via Monte Carlo Dropout (Gal & Ghahramani, 2016)."
    ),
    version="1.0.0",
)


# ── Pydantic schemas ──────────────────────────────────────────────────────────
class PredictionRequest(BaseModel):
    current_price: float = Field(
        ..., gt=1.0, lt=15.0,
        description="Current week's AOVE origin price (EUR/kg) from MAPA/POOLred bulletin.",
        example=4.85,
    )
    mc_samples: Optional[int] = Field(
        200, ge=10, le=500,
        description="Number of Monte Carlo Dropout forward passes (default: 200).",
    )


class ConfidenceInterval(BaseModel):
    low_80:  float = Field(description="10th percentile of MC distribution (EUR/kg)")
    high_80: float = Field(description="90th percentile of MC distribution (EUR/kg)")

class PredictionResponse(BaseModel):
    status:              str
    device:              str
    last_data_date:      str = Field(description="Last date included in the input window")
    prediction_week:     str = Field(description="Week being predicted (next Monday)")
    input_price:         float
    predicted_price:     float = Field(description="Mean MC Dropout prediction (EUR/kg)")
    confidence_interval: ConfidenceInterval
    trend_signal:        str  = Field(description="UP or DOWN vs current week")
    confidence_pct:      int  = Field(description="Directional confidence 5-95%")
    mae_reference:       float = Field(description="Model validation MAE (EUR/kg) for context")

app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/dashboard")
def dashboard():
    return FileResponse("dashboard.html")

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
def health():
    """Returns API status and verifies that model and data files are accessible."""
    files = {
        "model":   os.path.exists(MODEL_PATH),
        "climate": os.path.exists(CLIMATE_PATH),
        "macro":   os.path.exists(MACRO_PATH),
    }
    return {
        "status":  "online" if all(files.values()) else "degraded",
        "device":  str(DEVICE),
        "files":   files,
    }

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(request: PredictionRequest):
    """
    Predicts the AOVE origin price for the week following the current one.

    **Input:** current week's MAPA/POOLred bulletin price (EUR/kg).

    **Output:** predicted price + 80% confidence interval via Monte Carlo Dropout.

    The model uses the last 104 weeks (2 years) of spatially aggregated
    climate data (ERA5, 13 Andalusian locations) and macroeconomic indicators
    (CPI, diesel, stocks, olive surface variation).
    """
    try:
        logger.info(f"Prediction request: current_price={request.current_price} EUR/kg")

        X_hf, X_macro, sc_tgt, last_date, pred_date = build_tensors(
            request.current_price, _df_mun, _df_macro
        )

        mean, lower, upper = mc_dropout_predict(
            _model, X_hf, X_macro, sc_tgt, n_samples=request.mc_samples
        )

        trend = "UP" if mean > request.current_price else "DOWN"
        width = upper - lower
        direction_pct = int(100 * (mean - lower) / width) if width > 0 else 50
        direction_pct = max(5, min(95, direction_pct))

        logger.info(f"Prediction: {mean:.4f} EUR/kg [{lower:.4f}, {upper:.4f}] — {trend}")

        return PredictionResponse(
            status              = "success",
            device              = str(DEVICE),
            last_data_date      = str(last_date),
            prediction_week     = str(pred_date),
            input_price         = request.current_price,
            predicted_price     = round(mean, 4),
            confidence_interval = ConfidenceInterval(
                low_80=round(lower, 4), high_80=round(upper, 4)
            ),
            trend_signal        = trend,
            confidence_pct      = direction_pct,
            mae_reference       = 0.204,   # Validation MAE from training run
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

@app.get("/", tags=["System"])
def root():
    return {
        "name":    "AOVE Oracle API",
        "version": "1.0.0",
        "docs":    "/docs",
        "health":  "/health",
        "predict": "POST /predict",
    }