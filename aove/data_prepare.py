"""Data ingestion pipeline: builds the climate and macro CSVs.

Sources: Open-Meteo (climate, no API key), INE (monthly CPI), a MAPA PDF-scraper
CSV (weekly AOVE price), and MITECO/AICA proxies for diesel and stocks.
"""

import argparse
import logging
import time
from datetime import date
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

INE_BASE = "https://servicios.ine.es/wstempus/js/ES"
INE_IPC_SERIES = "IPC251852"

# (display_name, latitude, longitude, productive_surface_ha). Surface: ESYRCE 2022.
OLIVE_LOCATIONS: list[tuple[str, float, float, int]] = [
    ("Jaen capital", 37.779, -3.787, 165_000),
    ("Ubeda", 38.013, -3.370, 98_000),
    ("Baeza", 37.994, -3.469, 72_000),
    ("Linares", 38.095, -3.636, 45_000),
    ("Martos", 37.720, -3.971, 68_000),
    ("Cordoba capital", 37.888, -4.779, 140_000),
    ("Lucena", 37.408, -4.486, 88_000),
    ("Cabra", 37.472, -4.443, 52_000),
    ("Sevilla capital", 37.389, -5.984, 75_000),
    ("Ecija", 37.542, -5.082, 42_000),
    ("Granada capital", 37.177, -3.598, 55_000),
    ("Baza", 37.494, -2.765, 28_000),
    ("Antequera", 37.020, -4.559, 35_000),
]


class OpenMeteoDownloader:
    """Download daily historical weather from the Open-Meteo archive API."""

    ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
    DAILY_VARIABLES = (
        "precipitation_sum,temperature_2m_max,"
        "temperature_2m_min,et0_fao_evapotranspiration"
    )

    def __init__(self, sleep_between_requests: float = 0.5) -> None:
        self.sleep_s = sleep_between_requests
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})

    def _get(
        self, params: dict[str, str | float], retries: int = 3
    ) -> Optional[dict[str, Any]]:
        """GET with exponential backoff."""
        for attempt in range(retries):
            try:
                resp = self.session.get(self.ARCHIVE_URL, params=params, timeout=60)
                if resp.status_code == 429:
                    wait = 60 * (attempt + 1)
                    logger.warning("Open-Meteo rate limit. Waiting %ds...", wait)
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                payload: dict[str, Any] = resp.json()
                return payload
            except requests.RequestException as exc:
                logger.warning("Attempt %d/%d failed: %s", attempt + 1, retries, exc)
                time.sleep(5 * (attempt + 1))
        return None

    def download_location(
        self,
        name: str,
        lat: float,
        lon: float,
        surface_ha: int,
        start: date,
        end: date,
    ) -> pd.DataFrame:
        """Download the full date range for one location in a single API call."""
        params: dict[str, str | float] = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "daily": self.DAILY_VARIABLES,
            "timezone": "Europe/Madrid",
            "models": "era5_land",
        }
        logger.info("Downloading %s (%s, %s)...", name, lat, lon)
        data = self._get(params)
        time.sleep(self.sleep_s)

        if not data or "daily" not in data:
            logger.warning("No data returned for %s", name)
            return pd.DataFrame()

        daily = data["daily"]
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(daily["time"]),
                "rainfall_mm": daily["precipitation_sum"],
                "temp_max_c": daily["temperature_2m_max"],
                "temp_min_c": daily["temperature_2m_min"],
                "etp_mm": daily["et0_fao_evapotranspiration"],
            }
        )
        df["water_deficit_mm"] = df["rainfall_mm"] - df["etp_mm"]
        df["surface_ha"] = surface_ha
        df["station_id"] = name.lower().replace(" ", "_")

        cols = ["rainfall_mm", "temp_max_c", "etp_mm", "water_deficit_mm"]
        df[cols] = df[cols].ffill().fillna(0.0)
        return df[
            [
                "date",
                "station_id",
                "surface_ha",
                "rainfall_mm",
                "temp_max_c",
                "water_deficit_mm",
            ]
        ]

    def download_all_locations(self, start: date, end: date) -> pd.DataFrame:
        """Download every location and aggregate to ISO weekly frequency."""
        frames = [
            df
            for name, lat, lon, surface_ha in OLIVE_LOCATIONS
            if not (
                df := self.download_location(name, lat, lon, surface_ha, start, end)
            ).empty
        ]
        if not frames:
            raise RuntimeError(
                "No climate data retrieved from Open-Meteo. Check your connection."
            )

        df_daily = pd.concat(frames, ignore_index=True)
        df_daily["date"] = df_daily["date"].dt.to_period("W-SUN").dt.start_time
        df_weekly = (
            df_daily.groupby(["date", "station_id", "surface_ha"])[
                ["rainfall_mm", "temp_max_c", "water_deficit_mm"]
            ]
            .agg(
                {
                    "rainfall_mm": "sum",
                    "temp_max_c": "max",
                    "water_deficit_mm": "sum",
                }
            )
            .reset_index()
        )
        logger.info(
            "Climate download complete: %d rows (%d weeks x %d locations)",
            len(df_weekly),
            df_weekly["date"].nunique(),
            df_weekly["station_id"].nunique(),
        )
        return df_weekly


class INEDownloader:
    """Download the Spanish national CPI from the public INE JSON API."""

    @staticmethod
    def download(start: date, end: date) -> pd.DataFrame:
        """Return a monthly DataFrame with ``reference_date`` and ``ipc_monthly``."""
        url = (
            f"{INE_BASE}/DATOS_SERIE/{INE_IPC_SERIES}"
            f"?date={start.strftime('%Y%m%d')}:{end.strftime('%Y%m%d')}&tip=A"
        )
        logger.info("Downloading CPI from INE...")
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        records: list[dict[str, Any]] = resp.json().get("Data", [])
        if not records:
            raise ValueError(f"INE returned no data for series {INE_IPC_SERIES}.")

        rows = [
            {
                "reference_date": pd.Timestamp(r["Fecha"], unit="ms", tz="UTC")
                .tz_convert("Europe/Madrid")
                .replace(day=1, tzinfo=None),
                "ipc_monthly": float(r["Valor"]),
            }
            for r in records
            if r.get("Fecha") is not None and r.get("Valor") is not None
        ]
        df = pd.DataFrame(rows).sort_values("reference_date").reset_index(drop=True)
        logger.info("CPI downloaded: %d monthly records", len(df))
        return df


class PoolRedLoader:
    """Load the weekly AOVE price CSV produced by the MAPA PDF scraper."""

    @staticmethod
    def load(csv_path: Path) -> pd.DataFrame:
        """Return weekly columns: reference_date, aove_price_eur_kg."""
        if not csv_path.exists():
            raise FileNotFoundError(
                f"AOVE price CSV not found: {csv_path}\n"
                "Run the MAPA PDF scraper first to generate it."
            )
        df = pd.read_csv(csv_path, sep=None, engine="python", encoding="utf-8-sig")
        df.columns = [c.strip().lower() for c in df.columns]

        date_col = next(
            (
                c
                for c in df.columns
                if any(k in c for k in ["reference_date", "fecha", "date", "semana"])
            ),
            df.columns[0],
        )
        price_col = next(
            (
                c
                for c in df.columns
                if any(k in c for k in ["aove_price", "precio", "price", "eur_kg"])
            ),
            df.columns[1],
        )
        df["reference_date"] = pd.to_datetime(
            df[date_col], dayfirst=True, errors="coerce"
        )
        df["aove_price_eur_kg"] = pd.to_numeric(
            df[price_col].astype(str).str.replace(",", "."), errors="coerce"
        )
        df = (
            df[["reference_date", "aove_price_eur_kg"]]
            .dropna()
            .sort_values("reference_date")
            .reset_index(drop=True)
        )
        logger.info("AOVE prices loaded: %d weekly records", len(df))
        return df


class DieselLoader:
    """Load agricultural diesel prices (CSV if available, MITECO proxy otherwise)."""

    ANNUAL_PROXY: dict[int, float] = {
        2010: 0.682,
        2011: 0.789,
        2012: 0.820,
        2013: 0.780,
        2014: 0.660,
        2015: 0.548,
        2016: 0.445,
        2017: 0.499,
        2018: 0.596,
        2019: 0.549,
        2020: 0.394,
        2021: 0.598,
        2022: 0.950,
        2023: 0.820,
        2024: 0.760,
        2025: 0.720,
    }

    @staticmethod
    def load(csv_path: Optional[Path], start: date, end: date) -> pd.DataFrame:
        """Return monthly columns: reference_date, diesel_price_eur."""
        if csv_path and csv_path.exists():
            df = pd.read_csv(csv_path, parse_dates=["reference_date"])
            df = df[["reference_date", "diesel_price_eur"]].dropna()
            logger.info("Diesel loaded from CSV: %d records", len(df))
            return df.sort_values("reference_date").reset_index(drop=True)

        logger.warning("No diesel CSV - using MITECO annual proxy (MVP only).")
        rows: list[dict[str, object]] = []
        current = date(start.year, 1, 1)
        end_proxy = date(end.year, 12, 1)
        last_year = max(DieselLoader.ANNUAL_PROXY)
        while current <= end_proxy:
            rows.append(
                {
                    "reference_date": pd.Timestamp(current),
                    "diesel_price_eur": DieselLoader.ANNUAL_PROXY.get(
                        current.year, DieselLoader.ANNUAL_PROXY[last_year]
                    ),
                }
            )
            current = (
                date(current.year + 1, 1, 1)
                if current.month == 12
                else date(current.year, current.month + 1, 1)
            )
        df = pd.DataFrame(rows)
        logger.info("Diesel proxy: %d monthly records", len(df))
        return df


class AICALoader:
    """Load monthly olive oil stocks (AICA CSVs, else a seasonal proxy)."""

    ANNUAL_STOCK_KT: dict[int, int] = {
        2010: 520,
        2011: 480,
        2012: 610,
        2013: 520,
        2014: 350,
        2015: 620,
        2016: 580,
        2017: 490,
        2018: 510,
        2019: 720,
        2020: 840,
        2021: 600,
        2022: 320,
        2023: 240,
        2024: 480,
        2025: 650,
    }
    SEASONAL_OFFSET: dict[int, int] = {
        1: 5,
        2: 3,
        3: 1,
        4: -2,
        5: -4,
        6: -6,
        7: -8,
        8: -6,
        9: -3,
        10: 8,
        11: 15,
        12: 10,
    }

    @staticmethod
    def load(aica_dir: Optional[Path], start: date, end: date) -> pd.DataFrame:
        """Return monthly columns: reference_date, stock_delta_pct."""
        if aica_dir and aica_dir.is_dir() and list(aica_dir.glob("*.csv")):
            return AICALoader._from_dir(aica_dir)
        logger.warning("AICA dir absent or empty - using seasonal stock proxy.")
        return AICALoader._proxy(start, end)

    @staticmethod
    def _from_dir(aica_dir: Path) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for f in sorted(aica_dir.glob("*.csv")):
            try:
                df = pd.read_csv(f, sep=None, engine="python", encoding="latin-1")
                df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
                frames.append(df)
            except (ValueError, UnicodeDecodeError) as exc:
                logger.warning("Could not parse %s: %s", f.name, exc)

        df_all = pd.concat(frames, ignore_index=True)
        date_col = next(
            c for c in df_all.columns if any(k in c for k in ["mes", "fecha", "date"])
        )
        stock_col = next(
            c for c in df_all.columns if any(k in c for k in ["exist", "stock"])
        )
        df_all["reference_date"] = pd.to_datetime(df_all[date_col], errors="coerce")
        df_all["stock_t"] = pd.to_numeric(
            df_all[stock_col].astype(str).str.replace(",", "").str.replace(".", ""),
            errors="coerce",
        )
        df_clean = (
            df_all[["reference_date", "stock_t"]]
            .dropna()
            .sort_values("reference_date")
            .reset_index(drop=True)
        )
        df_clean["stock_delta_pct"] = df_clean["stock_t"].pct_change() * 100
        logger.info("AICA stocks loaded: %d monthly records", len(df_clean))
        return df_clean[["reference_date", "stock_delta_pct"]]

    @staticmethod
    def _proxy(start: date, end: date) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        prev: Optional[float] = None
        current = date(start.year, 1, 1)
        end_proxy = date(end.year, 12, 1)
        last_year = max(AICALoader.ANNUAL_STOCK_KT)
        while current <= end_proxy:
            base = (
                AICALoader.ANNUAL_STOCK_KT.get(
                    current.year, AICALoader.ANNUAL_STOCK_KT[last_year]
                )
                * 1_000
            )
            stock = base * (
                1.0 + AICALoader.SEASONAL_OFFSET.get(current.month, 0) / 100.0
            )
            delta = ((stock - prev) / prev * 100.0) if prev else 0.0
            rows.append(
                {"reference_date": pd.Timestamp(current), "stock_delta_pct": delta}
            )
            prev = stock
            current = (
                date(current.year + 1, 1, 1)
                if current.month == 12
                else date(current.year, current.month + 1, 1)
            )
        df = pd.DataFrame(rows)
        logger.info("Stock proxy: %d monthly records", len(df))
        return df


class DatasetAssembler:
    """Combine all feeds into the two production CSVs (no look-ahead bias)."""

    SURFACE_ANNUAL: dict[int, float] = {
        2010: 0.6,
        2011: 0.7,
        2012: 0.9,
        2013: 1.1,
        2014: 0.8,
        2015: 0.8,
        2016: 1.2,
        2017: 0.5,
        2018: -0.3,
        2019: 0.9,
        2020: 1.1,
        2021: 0.4,
        2022: -0.2,
        2023: 0.6,
        2024: 0.7,
        2025: 0.5,
    }

    @staticmethod
    def build_climate_csv(df_climate: pd.DataFrame, output_path: Path) -> None:
        """Save the weekly-per-location climate CSV."""
        df = df_climate.copy()
        for col in ["rainfall_mm", "temp_max_c", "water_deficit_mm"]:
            df[col] = df[col].astype(np.float32)
        df.to_csv(output_path, index=False)
        logger.info("climate_dataset.csv -> %s (%d rows)", output_path, len(df))

    @staticmethod
    def build_macro_csv(
        df_prices: pd.DataFrame,
        df_ipc: pd.DataFrame,
        df_diesel: pd.DataFrame,
        df_stock: pd.DataFrame,
        output_path: Path,
    ) -> None:
        """Merge weekly prices with monthly macro feeds (backward, no leakage)."""
        backbone = (
            df_prices[["reference_date", "aove_price_eur_kg"]]
            .copy()
            .assign(
                reference_date=lambda d: pd.to_datetime(d["reference_date"]).astype(
                    "datetime64[us]"
                )
            )
            .sort_values("reference_date")
            .reset_index(drop=True)
        )
        for df_feed, col in [
            (df_ipc, "ipc_monthly"),
            (df_diesel, "diesel_price_eur"),
            (df_stock, "stock_delta_pct"),
        ]:
            feed = (
                df_feed[["reference_date", col]]
                .copy()
                .assign(
                    reference_date=lambda d: pd.to_datetime(d["reference_date"]).astype(
                        "datetime64[us]"
                    )
                )
                .sort_values("reference_date")
                .dropna()
            )
            backbone = pd.merge_asof(
                left=backbone, right=feed, on="reference_date", direction="backward"
            )

        backbone["surface_delta_pct"] = backbone["reference_date"].dt.year.map(
            DatasetAssembler.SURFACE_ANNUAL
        )
        backbone = backbone.sort_values("reference_date").reset_index(drop=True)
        backbone = backbone.ffill().fillna(0.0)
        for col in [
            "aove_price_eur_kg",
            "ipc_monthly",
            "diesel_price_eur",
            "stock_delta_pct",
            "surface_delta_pct",
        ]:
            backbone[col] = backbone[col].astype(np.float32)

        backbone.to_csv(output_path, index=False)
        logger.info(
            "macro_dataset.csv -> %s (%d weekly rows)", output_path, len(backbone)
        )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AOVE data collector (Open-Meteo edition)"
    )
    parser.add_argument("--poolred-csv", type=Path, default=None)
    parser.add_argument("--diesel-csv", type=Path, default=None)
    parser.add_argument("--aica-dir", type=Path, default=None)
    parser.add_argument("--start-date", type=str, default="2010-01-01")
    parser.add_argument("--end-date", type=str, default=date.today().isoformat())
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    return parser


def main() -> None:
    """Console-script entry point (``aove-prepare``)."""
    args = _build_arg_parser().parse_args()
    start = date.fromisoformat(args.start_date)
    end = date.fromisoformat(args.end_date)
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("AOVE data collector | %s -> %s | output: %s", start, end, output_dir)

    downloader = OpenMeteoDownloader(sleep_between_requests=6.0)
    df_climate = downloader.download_all_locations(start, end)
    DatasetAssembler.build_climate_csv(df_climate, output_dir / "climate_dataset.csv")

    df_ipc = INEDownloader.download(start, end)

    if not args.poolred_csv:
        logger.error(
            "No AOVE price CSV provided (--poolred-csv) - macro_dataset.csv skipped."
        )
        raise SystemExit(1)

    df_prices = PoolRedLoader.load(args.poolred_csv)
    df_diesel = DieselLoader.load(args.diesel_csv, start, end)
    df_stock = AICALoader.load(args.aica_dir, start, end)
    DatasetAssembler.build_macro_csv(
        df_prices, df_ipc, df_diesel, df_stock, output_dir / "macro_dataset.csv"
    )
    logger.info("Data collector finished. Files in %s", output_dir.resolve())


if __name__ == "__main__":
    main()
