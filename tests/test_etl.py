"""Tests for the market ETL pipeline on synthetic data."""

import numpy as np
import pandas as pd

from aove.etl import MarketETLPipeline


def _municipal_climate() -> pd.DataFrame:
    dates = pd.date_range("2020-01-06", periods=6, freq="W-MON")
    rows = []
    for station, surface in [("jaen", 100_000), ("cordoba", 50_000)]:
        for i, d in enumerate(dates):
            rows.append(
                {
                    "date": d,
                    "station_id": station,
                    "surface_ha": surface,
                    "rainfall_mm": 10.0 + i,
                    "temp_max_c": 20.0 + i,
                    "water_deficit_mm": -5.0 + i,
                }
            )
    return pd.DataFrame(rows)


def _macro() -> pd.DataFrame:
    dates = pd.date_range("2019-11-01", periods=6, freq="MS")
    return pd.DataFrame(
        {
            "reference_date": dates,
            "stock_delta_pct": np.linspace(-2, 2, 6),
            "surface_delta_pct": np.linspace(0, 1, 6),
            "ipc_monthly": np.linspace(100, 105, 6),
            "diesel_price_eur": np.linspace(0.5, 0.7, 6),
            "aove_price_eur_kg": np.linspace(3.0, 4.0, 6),
        }
    )


def test_aggregate_climate_collapses_to_one_row_per_week() -> None:
    etl = MarketETLPipeline()
    agg = etl.aggregate_climate(_municipal_climate())
    assert agg.index.name == "date"
    assert len(agg) == 6  # 6 unique weeks
    assert {"time_sin", "time_cos"}.issubset(agg.columns)
    # Cyclical encodings stay within the unit range.
    assert agg["time_sin"].abs().max() <= 1.0


def test_align_macro_builds_lag_feature_and_is_float32() -> None:
    etl = MarketETLPipeline()
    agg = etl.aggregate_climate(_municipal_climate())
    final = etl.align_macro_data(agg, _macro(), aove_lag_weeks=1)
    assert "aove_lag_price" in final.columns
    assert final["aove_lag_price"].notna().all()
    assert all(final[c].dtype == np.float32 for c in final.columns)
