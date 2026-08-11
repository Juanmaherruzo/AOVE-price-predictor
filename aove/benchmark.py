"""Baseline comparison for the weekly EVOO price forecast.

A regression metric on a price series means nothing on its own. The target is
the origin price in week *t*, and ``aove_lag_price`` — the price in week *t-1* —
is one of the model's own macro inputs, so a model that learns nothing beyond
"repeat the last price I was given" already scores a high R². The only way to
read MAE, MAPE or R² as evidence of forecasting skill is against a naive
baseline built from the same information.

This module evaluates, on the identical chronological validation split used for
training:

``persistence``
    Predict the last price the model itself was given (``aove_lag_price``).
    Same information, zero parameters. This is the number the LSTM has to beat.

``drift``
    Persistence plus the average weekly change observed over the training
    split — a random walk with drift.

``lstm``
    The trained checkpoint, in deterministic ``eval`` mode.

The skill score reported for the LSTM is the fractional reduction in MAE over
persistence: ``1 - MAE_lstm / MAE_persistence``. Positive means the model adds
information; zero or negative means it does not.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import torch

from aove.config import DEVICE, HF_COLS, MACRO_COLS, TARGET_COL, settings
from aove.etl import MarketETLPipeline
from aove.features import TemporalSequenceBuilder
from aove.model import AOVEPricePredictor

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Metrics:
    """Point-forecast error metrics on one validation split.

    Attributes:
        name: Label of the predictor these metrics describe.
        mae: Mean absolute error, EUR/kg.
        rmse: Root mean squared error, EUR/kg.
        mape: Mean absolute percentage error, percent.
        r2: Coefficient of determination against the validation mean.
        n: Number of validation targets scored.
    """

    name: str
    mae: float
    rmse: float
    mape: float
    r2: float
    n: int


def compute_metrics(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    """Score one set of predictions against the observed prices.

    Args:
        name: Label for the predictor.
        y_true: Observed prices, EUR/kg.
        y_pred: Predicted prices, EUR/kg, aligned with ``y_true``.

    Returns:
        A :class:`Metrics` with MAE, RMSE, MAPE and R².

    Raises:
        ValueError: If the two arrays do not share a length.
    """
    true = np.asarray(y_true, dtype=np.float64).ravel()
    pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if true.shape != pred.shape:
        raise ValueError(f"shape mismatch: y_true {true.shape} vs y_pred {pred.shape}")

    error = pred - true
    ss_res = float(np.sum(error**2))
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    return Metrics(
        name=name,
        mae=float(np.mean(np.abs(error))),
        rmse=float(np.sqrt(np.mean(error**2))),
        mape=float(np.mean(np.abs(error / true)) * 100.0),
        r2=1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        n=int(true.size),
    )


def skill_score(model: Metrics, reference: Metrics) -> float:
    """Fractional MAE reduction of ``model`` over ``reference``.

    Returns:
        ``1 - MAE_model / MAE_reference``. Positive means the model beats the
        reference; ``0.0`` means it merely matches it.
    """
    if reference.mae == 0:
        return float("nan")
    return 1.0 - model.mae / reference.mae


def repeat_rate(prices: np.ndarray) -> float:
    """Fraction of consecutive observations that do not change.

    Reported alongside the metrics so the comparison can be read fairly: a
    series that frequently repeats its last value makes persistence trivially
    strong. On this validation split the rate is low (~6%), so persistence wins
    on genuine week-to-week price behaviour rather than on a flat series.
    """
    series = np.asarray(prices, dtype=np.float64).ravel()
    if series.size < 2:
        return float("nan")
    return float(np.mean(series[1:] == series[:-1]))


@dataclass(frozen=True)
class BenchmarkReport:
    """Baseline comparison over one validation split."""

    metrics: list[Metrics]
    lstm_skill_vs_persistence: float | None
    validation_repeat_rate: float
    validation_start: str
    validation_end: str

    def to_json(self, path: Path) -> None:
        """Write the comparison to ``path`` as JSON."""
        payload = {
            "metrics": [asdict(m) for m in self.metrics],
            "lstm_skill_vs_persistence": self.lstm_skill_vs_persistence,
            "validation_repeat_rate": self.validation_repeat_rate,
            "validation_start": self.validation_start,
            "validation_end": self.validation_end,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def summary(self) -> str:
        """Render the comparison as a fixed-width table."""
        header = f"{'predictor':<14}{'MAE':>9}{'RMSE':>9}{'MAPE':>9}{'R2':>9}"
        lines = [header, "-" * len(header)]
        for m in self.metrics:
            lines.append(
                f"{m.name:<14}{m.mae:>9.3f}{m.rmse:>9.3f}"
                f"{m.mape:>8.2f}%{m.r2:>9.3f}"
            )
        lines.append("-" * len(header))
        lines.append(
            f"Validation: {self.validation_start} to {self.validation_end} "
            f"(n={self.metrics[0].n})"
        )
        lines.append(
            f"Unchanged week-on-week prices in validation: "
            f"{self.validation_repeat_rate * 100:.1f}%"
        )
        if self.lstm_skill_vs_persistence is not None:
            verdict = (
                "the LSTM beats persistence"
                if self.lstm_skill_vs_persistence > 0
                else "the LSTM does NOT beat persistence"
            )
            lines.append(
                f"Skill vs persistence: {self.lstm_skill_vs_persistence * 100:+.1f}% "
                f"MAE reduction -> {verdict}."
            )
        return "\n".join(lines)


def run_benchmark(
    climate_csv: Path,
    macro_csv: Path,
    checkpoint: Path | None = None,
    time_steps: int = settings.time_steps,
    train_ratio: float = settings.train_ratio,
    aove_lag_weeks: int = settings.aove_lag_weeks,
) -> BenchmarkReport:
    """Score the naive baselines, and the checkpoint if one is supplied.

    Rebuilds the exact ETL, feature and split pipeline used for training, so the
    baselines and the model are scored on the same targets in the same order.

    Args:
        climate_csv: Weekly municipal climate observations.
        macro_csv: Weekly macroeconomic bulletin data (origin price, stocks,
            surface, CPI, diesel), aligned by publication date in the ETL.
        checkpoint: Trained ``state_dict``. When ``None``, only the baselines
            are scored.
        time_steps: Length of the climate window, in weeks.
        train_ratio: Fraction of the timeline used for training.
        aove_lag_weeks: Offset of the ``aove_lag_price`` feature.

    Returns:
        A :class:`BenchmarkReport`.

    Raises:
        ValueError: If ``aove_lag_weeks`` is not positive; the persistence
            baseline is defined by that feature.
    """
    if aove_lag_weeks <= 0:
        raise ValueError(
            "The persistence baseline needs the aove_lag_price feature; "
            f"got aove_lag_weeks={aove_lag_weeks}."
        )

    df_municipal = pd.read_csv(climate_csv, parse_dates=["date"])
    df_macro = pd.read_csv(macro_csv, parse_dates=["reference_date"])

    etl = MarketETLPipeline(publish_delay_days=settings.publish_delay_days)
    df_final = etl.align_macro_data(
        etl.aggregate_climate(df_municipal), df_macro, aove_lag_weeks=aove_lag_weeks
    )

    macro_cols = list(MACRO_COLS)
    builder = TemporalSequenceBuilder(time_steps=time_steps)
    splits = builder.build_sequences_split(
        df_final, HF_COLS, macro_cols, TARGET_COL, train_ratio=train_ratio
    )
    x_hf_val, x_macro_val, y_val_scaled = cast(
        "tuple[np.ndarray, np.ndarray, np.ndarray]", splits["val"]
    )
    _, x_macro_train, y_train_scaled = cast(
        "tuple[np.ndarray, np.ndarray, np.ndarray]", splits["train"]
    )
    val_dates = cast(pd.DatetimeIndex, splits["val_dates"])

    y_true = builder.inverse_transform_target(y_val_scaled)

    # The persistence prediction is the model's own lag feature, returned to
    # EUR/kg. It is the last column of the macro block (MACRO_COLS ordering).
    lag_index = macro_cols.index("aove_lag_price")
    scaler = builder.scaler_macro
    lag_mean = float(scaler.mean_[lag_index])
    lag_scale = float(scaler.scale_[lag_index])
    persistence = x_macro_val[:, lag_index] * lag_scale + lag_mean

    # Random walk with drift: the mean weekly change over the training split.
    y_train = builder.inverse_transform_target(y_train_scaled)
    train_lag = x_macro_train[:, lag_index] * lag_scale + lag_mean
    drift = float(np.mean(y_train - train_lag))
    logger.info("Training-split mean weekly change (drift): %+.4f EUR/kg", drift)

    results = [
        compute_metrics("persistence", y_true, persistence),
        compute_metrics("drift", y_true, persistence + drift),
    ]

    lstm_skill: float | None = None
    if checkpoint is not None and checkpoint.exists():
        model = AOVEPricePredictor(
            hf_input_dim=len(HF_COLS), macro_input_dim=len(macro_cols)
        ).to(DEVICE)
        model.load_state_dict(
            torch.load(checkpoint, map_location=DEVICE, weights_only=True)
        )
        model.eval()
        with torch.no_grad():
            preds_scaled = (
                model(
                    torch.from_numpy(x_hf_val).to(DEVICE, dtype=torch.float32),
                    torch.from_numpy(x_macro_val).to(DEVICE, dtype=torch.float32),
                )
                .cpu()
                .numpy()
            )
        lstm = compute_metrics(
            "lstm", y_true, builder.inverse_transform_target(preds_scaled)
        )
        results.append(lstm)
        lstm_skill = skill_score(lstm, results[0])
    elif checkpoint is not None:
        logger.warning(
            "Checkpoint not found at %s — scoring baselines only.", checkpoint
        )

    return BenchmarkReport(
        metrics=results,
        lstm_skill_vs_persistence=lstm_skill,
        validation_repeat_rate=repeat_rate(y_true),
        validation_start=str(val_dates[0].date()),
        validation_end=str(val_dates[-1].date()),
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score the AOVE model against naive forecasting baselines."
    )
    parser.add_argument("--climate-csv", type=Path, default=settings.climate_path)
    parser.add_argument("--macro-csv", type=Path, default=settings.macro_path)
    parser.add_argument("--checkpoint", type=Path, default=settings.checkpoint_path)
    parser.add_argument("--time-steps", type=int, default=settings.time_steps)
    parser.add_argument("--train-ratio", type=float, default=settings.train_ratio)
    parser.add_argument("--aove-lag", type=int, default=settings.aove_lag_weeks)
    parser.add_argument(
        "--json", type=Path, default=None, help="Also write the report to this path."
    )
    return parser


def main() -> None:
    """Console-script entry point (``aove-benchmark``)."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _build_arg_parser().parse_args()

    report = run_benchmark(
        climate_csv=args.climate_csv,
        macro_csv=args.macro_csv,
        checkpoint=args.checkpoint,
        time_steps=args.time_steps,
        train_ratio=args.train_ratio,
        aove_lag_weeks=args.aove_lag,
    )
    print(report.summary())
    if args.json is not None:
        report.to_json(args.json)
        logger.info("Report written to %s", args.json)


if __name__ == "__main__":
    main()
