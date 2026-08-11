"""FastAPI server exposing the AOVE weekly price predictor."""

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from aove.config import DEVICE, settings
from aove.inference import build_tensors, load_model, mc_dropout_predict
from aove.model import AOVEPricePredictor

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Validation MAE of the model, surfaced to clients for context. The naive
# persistence baseline scores 0.057 EUR/kg on the same split and is reported
# alongside it so a client can see that the model does not beat it (see
# aove.benchmark and the README results table).
_MAE_REFERENCE = 0.073
_MAE_PERSISTENCE_BASELINE = 0.057


class _State:
    """Container for the objects loaded once at startup."""

    model: Optional[AOVEPricePredictor] = None
    df_climate: Optional[pd.DataFrame] = None
    df_macro: Optional[pd.DataFrame] = None


state = _State()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the model and datasets into memory once, shared across requests."""
    logger.info("Loading model and datasets into memory...")
    state.model = load_model(settings.checkpoint_path)
    state.df_climate = pd.read_csv(settings.climate_path, parse_dates=["date"])
    state.df_macro = pd.read_csv(settings.macro_path, parse_dates=["reference_date"])
    logger.info(
        "Ready - device: %s | climate rows: %d | macro rows: %d",
        DEVICE,
        len(state.df_climate),
        len(state.df_macro),
    )
    yield
    logger.info("Server shutting down.")


app = FastAPI(
    lifespan=lifespan,
    title="AOVE Oracle API",
    description=(
        "Weekly Extra Virgin Olive Oil (EVOO) origin price predictor. "
        "Bimodal LSTM trained on 10+ years of climate and macro data. "
        "Confidence intervals via Monte Carlo Dropout (Gal & Ghahramani, 2016)."
    ),
    version="1.0.0",
)

# Comma-separated list of allowed origins. Defaults to the bundled dashboard's
# own origin; set AOVE_CORS_ORIGINS=* only for a throwaway public demo.
_CORS_ORIGINS = [
    origin.strip()
    for origin in os.environ.get(
        "AOVE_CORS_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000"
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class PredictionRequest(BaseModel):
    current_price: float = Field(
        ...,
        gt=1.0,
        lt=15.0,
        description="Current week's AOVE origin price (EUR/kg).",
    )
    mc_samples: int = Field(
        200,
        ge=10,
        le=500,
        description="Number of Monte Carlo Dropout forward passes.",
    )


class ConfidenceInterval(BaseModel):
    low_80: float = Field(description="10th percentile of the MC distribution (EUR/kg)")
    high_80: float = Field(
        description="90th percentile of the MC distribution (EUR/kg)"
    )


class PredictionResponse(BaseModel):
    status: str
    device: str
    last_data_date: str
    prediction_week: str
    input_price: float
    predicted_price: float
    confidence_interval: ConfidenceInterval
    trend_signal: str
    confidence_pct: int
    mae_reference: float
    mae_persistence_baseline: float = Field(
        default=_MAE_PERSISTENCE_BASELINE,
        description=(
            "Validation MAE of repeating last week's price. The model does not "
            "beat it; both are returned so the prediction can be read in context."
        ),
    )


@app.get("/dashboard")
def dashboard() -> FileResponse:
    """Serve the static dashboard page."""
    if not settings.dashboard_path.exists():
        raise HTTPException(status_code=404, detail="Dashboard asset not found.")
    return FileResponse(settings.dashboard_path)


@app.get("/health", tags=["System"])
def health() -> dict[str, object]:
    """Report API status and verify that model and data files are accessible."""
    files = {
        "model": settings.checkpoint_path.exists(),
        "climate": settings.climate_path.exists(),
        "macro": settings.macro_path.exists(),
    }
    return {
        "status": "online" if all(files.values()) else "degraded",
        "device": str(DEVICE),
        "files": files,
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(request: PredictionRequest) -> PredictionResponse:
    """Predict the AOVE origin price for the week following the current one."""
    if state.model is None or state.df_climate is None or state.df_macro is None:
        raise HTTPException(status_code=503, detail="Model or data not loaded.")

    try:
        logger.info("Prediction request: current_price=%s", request.current_price)
        x_hf, x_macro, sc_tgt, last_date, pred_date = build_tensors(
            request.current_price,
            state.df_climate,
            state.df_macro,
            time_steps=settings.time_steps,
            train_ratio=settings.train_ratio,
        )
        mean, lower, upper = mc_dropout_predict(
            state.model, x_hf, x_macro, sc_tgt, n_samples=request.mc_samples
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    trend = "UP" if mean > request.current_price else "DOWN"
    width = upper - lower
    direction_pct = int(100 * (mean - lower) / width) if width > 0 else 50
    direction_pct = max(5, min(95, direction_pct))
    logger.info("Prediction: %.4f EUR/kg [%.4f, %.4f] - %s", mean, lower, upper, trend)

    return PredictionResponse(
        status="success",
        device=str(DEVICE),
        last_data_date=str(last_date),
        prediction_week=str(pred_date),
        input_price=request.current_price,
        predicted_price=round(mean, 4),
        confidence_interval=ConfidenceInterval(
            low_80=round(lower, 4), high_80=round(upper, 4)
        ),
        trend_signal=trend,
        confidence_pct=direction_pct,
        mae_reference=_MAE_REFERENCE,
    )


@app.get("/", tags=["System"])
def root() -> dict[str, str]:
    """Return basic API metadata."""
    return {
        "name": "AOVE Oracle API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "predict": "POST /predict",
    }
