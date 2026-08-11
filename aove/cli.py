"""Interactive command-line interface for weekly AOVE price prediction."""

import logging
from pathlib import Path

import pandas as pd

from aove.config import settings
from aove.inference import build_tensors, load_model, mc_dropout_predict

logger = logging.getLogger(__name__)

# Dedicated stdout logger for user-facing output — a clean, message-only
# formatter keeps the terminal UX while honouring "no print in production".
display = logging.getLogger("aove.display")
if not display.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    display.addHandler(_handler)
    display.setLevel(logging.INFO)
    display.propagate = False


class AOVEOracleCLI:
    """Connect user input -> ETL -> MC-Dropout inference -> formatted output."""

    def __init__(
        self,
        model_path: Path,
        climate_path: Path,
        macro_path: Path,
        time_steps: int = 104,
        train_ratio: float = 0.8,
        mc_samples: int = 200,
    ) -> None:
        self.model_path = model_path
        self.climate_path = climate_path
        self.macro_path = macro_path
        self.time_steps = time_steps
        self.train_ratio = train_ratio
        self.mc_samples = mc_samples

    def _validate_paths(self) -> None:
        for path in (self.model_path, self.climate_path, self.macro_path):
            if not path.exists():
                raise FileNotFoundError(f"Missing required file: {path}")

    def get_user_input(self) -> float:
        """Prompt interactively for the current week's market price."""
        display.info("\n%s", "=" * 52)
        display.info("  AOVE ORACLE - Weekly Price Predictor")
        display.info("=" * 52)
        while True:
            raw = input(
                "\n  Enter this week's EVOO origin price (EUR/kg) [e.g. 4.85]: "
            )
            try:
                price = float(raw.strip().replace(",", "."))
            except ValueError:
                display.info("  Invalid input - please enter a number (e.g. 4.85).")
                continue
            if not 1.0 <= price <= 15.0:
                display.info("  Warning: price outside 1.0-15.0 EUR/kg. Verify it.")
                if input("  Continue anyway? (y/n): ").strip().lower() != "y":
                    continue
            return price

    def run_inference(
        self, current_price: float
    ) -> tuple[float, float, float, object, object]:
        """Full pipeline: load data -> ETL -> model load -> MC-Dropout."""
        display.info("\n  Loading data and running inference...")
        df_climate = pd.read_csv(self.climate_path, parse_dates=["date"])
        df_macro = pd.read_csv(self.macro_path, parse_dates=["reference_date"])

        x_hf, x_macro, sc_tgt, last_date, pred_date = build_tensors(
            current_price,
            df_climate,
            df_macro,
            time_steps=self.time_steps,
            train_ratio=self.train_ratio,
        )
        model = load_model(self.model_path)
        mean, lower, upper = mc_dropout_predict(
            model, x_hf, x_macro, sc_tgt, n_samples=self.mc_samples
        )
        return mean, lower, upper, pred_date, last_date

    def display_results(
        self,
        current_price: float,
        mean: float,
        lower: float,
        upper: float,
        pred_date: object,
        last_date: object,
    ) -> None:
        """Render the prediction to the terminal via the display logger."""
        trend = "UP" if mean > current_price else "DOWN"
        trend_sym = "+" if mean > current_price else "-"
        delta = abs(mean - current_price)
        width = upper - lower
        direction_pct = int(100 * (mean - lower) / width) if width > 0 else 50
        direction_pct = max(5, min(95, direction_pct))

        display.info("\n%s", "=" * 52)
        display.info("  AOVE ORACLE - NEXT WEEK FORECAST")
        display.info("=" * 52)
        display.info("  Data up to       : %s", last_date)
        display.info("  Prediction week  : %s", pred_date)
        display.info("-" * 52)
        display.info("  Current price    :  %.2f EUR/kg", current_price)
        display.info(
            "  Predicted price  :  %.2f EUR/kg  (%s%.2f)", mean, trend_sym, delta
        )
        display.info("  80%% range        :  %.2f - %.2f EUR/kg", lower, upper)
        display.info(
            "  Trend signal     :  %s  (%d%% confidence)", trend, direction_pct
        )
        display.info("=" * 52)
        display.info("  Note: range based on %d MC Dropout samples.", self.mc_samples)

    def run(self) -> None:
        """Main entry point — full interactive session."""
        self._validate_paths()
        current_price = self.get_user_input()
        mean, lower, upper, pred_date, last_date = self.run_inference(current_price)
        self.display_results(current_price, mean, lower, upper, pred_date, last_date)


def main() -> None:
    """Console-script entry point (``aove-cli``)."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    AOVEOracleCLI(
        model_path=settings.checkpoint_path,
        climate_path=settings.climate_path,
        macro_path=settings.macro_path,
        time_steps=settings.time_steps,
        train_ratio=settings.train_ratio,
        mc_samples=settings.mc_samples,
    ).run()


if __name__ == "__main__":
    main()
