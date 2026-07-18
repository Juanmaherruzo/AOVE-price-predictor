"""Market ETL pipeline: spatial climate aggregation and macro alignment.

Single source of truth for the feature-engineering logic shared by the training,
inference, CLI and API layers.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MarketETLPipeline:
    """Spatially weighted climate averaging and time-shifted macro merging."""

    def __init__(self, publish_delay_days: int = 15) -> None:
        self.publish_delay_days = publish_delay_days

    def aggregate_climate(self, df_municipal: pd.DataFrame) -> pd.DataFrame:
        """Collapse municipal climate rows into one weekly sequence.

        Uses productive surface area (ha) as the spatial weight and appends
        cyclical week-of-year encodings (``time_sin`` / ``time_cos``).
        """
        logger.info("Performing spatial weighted aggregation on climate data...")

        climate_cols = ["rainfall_mm", "temp_max_c", "water_deficit_mm"]

        def weighted_avg(group: pd.DataFrame) -> pd.Series:
            weights = group["surface_ha"].to_numpy(dtype=np.float64)
            if weights.sum() == 0:
                return group[climate_cols].mean()
            w_avg = np.average(group[climate_cols].to_numpy(), weights=weights, axis=0)
            return pd.Series(w_avg, index=climate_cols)

        df_agg = (
            df_municipal.groupby("date")[climate_cols + ["surface_ha"]]
            .apply(weighted_avg, include_groups=False)
            .reset_index()
        )

        week_num = df_agg["date"].dt.isocalendar().week.astype(float)
        df_agg["time_sin"] = np.sin(week_num * 2 * np.pi / 52).astype(np.float32)
        df_agg["time_cos"] = np.cos(week_num * 2 * np.pi / 52).astype(np.float32)

        return df_agg.set_index("date")

    def align_macro_data(
        self,
        df_climate: pd.DataFrame,
        df_macro: pd.DataFrame,
        aove_lag_weeks: int = 1,
    ) -> pd.DataFrame:
        """Shift macro data by publication delay and merge it backwards.

        The raw ``aove_price_eur_kg`` is kept only as the target column; an
        explicit ``aove_lag_price`` feature is built from it with a controlled
        weekly offset so the model sees past prices without target leakage.
        """
        logger.info(
            "Applying %d-day publication shift to macro data...",
            self.publish_delay_days,
        )

        df_macro = df_macro.copy()
        df_macro["publish_date"] = df_macro["reference_date"] + pd.DateOffset(
            days=self.publish_delay_days
        )
        df_macro_shifted = df_macro.sort_values("publish_date").set_index(
            "publish_date"
        )

        macro_feature_cols = [
            "stock_delta_pct",
            "surface_delta_pct",
            "ipc_monthly",
            "diesel_price_eur",
        ]

        df_climate = df_climate.sort_index()
        df_final = pd.merge_asof(
            left=df_climate,
            right=df_macro_shifted[macro_feature_cols + ["aove_price_eur_kg"]],
            left_index=True,
            right_index=True,
            direction="backward",
        )

        if aove_lag_weeks > 0:
            # bfill fills the leading NaNs with the oldest observed price, which
            # is semantically correct and harmless to the scaler statistics.
            df_final["aove_lag_price"] = (
                df_final["aove_price_eur_kg"].shift(aove_lag_weeks).bfill()
            )

        df_final = df_final.ffill().fillna(0.0)
        return df_final.astype(np.float32)
