"""Centralised configuration for the AOVE price predictor.

All paths, feature columns and hyperparameters live here as a single source of
truth, replacing the module-level constants that were previously duplicated
across the API, CLI and training scripts.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

import torch

# Repository root, resolved from this file rather than the working directory, so
# the installed console scripts (aove-cli, aove-train, aove-prepare,
# aove-benchmark) work from anywhere. Override with AOVE_ROOT when the data and
# checkpoint live outside the source tree — that is what the Docker image does.
PROJECT_ROOT = Path(os.environ.get("AOVE_ROOT", Path(__file__).resolve().parents[1]))

# Feature catalogue — shared by training, inference and the API.
HF_COLS: list[str] = [
    "rainfall_mm",
    "temp_max_c",
    "water_deficit_mm",
    "time_sin",
    "time_cos",
]
MACRO_COLS: list[str] = [
    "stock_delta_pct",
    "surface_delta_pct",
    "ipc_monthly",
    "diesel_price_eur",
    "aove_lag_price",
]
TARGET_COL: str = "aove_price_eur_kg"


def resolve_device() -> torch.device:
    """Select CUDA, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass(frozen=True)
class Settings:
    """Immutable runtime configuration for paths and hyperparameters."""

    checkpoint_path: Path = field(
        default_factory=lambda: PROJECT_ROOT / "AOVE_model" / "best_aove_model.pth"
    )
    climate_path: Path = field(
        default_factory=lambda: PROJECT_ROOT / "data" / "climate_dataset.csv"
    )
    macro_path: Path = field(
        default_factory=lambda: PROJECT_ROOT / "data" / "macro_dataset.csv"
    )
    dashboard_path: Path = field(
        default_factory=lambda: PROJECT_ROOT / "dashboard.html"
    )
    time_steps: int = 104
    train_ratio: float = 0.8
    mc_samples: int = 200
    publish_delay_days: int = 15
    aove_lag_weeks: int = 1


settings = Settings()
DEVICE = resolve_device()
