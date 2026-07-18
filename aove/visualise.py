"""Post-training visual diagnostics (learning curve, residuals, metrics)."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


class AOVEVisualiser:
    """Self-contained matplotlib toolkit for post-training analysis.

    Generates four publication-ready figures: learning curve, prediction vs
    actual, residuals over time, and a compact metrics dashboard.
    """

    def __init__(self, output_dir: Path = Path("plots")) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._style()

    @staticmethod
    def _style() -> None:
        """Apply a clean, minimal style that renders well in files and notebooks."""
        plt.rcParams.update(
            {
                "figure.facecolor": "white",
                "axes.facecolor": "white",
                "axes.edgecolor": "#cccccc",
                "axes.grid": True,
                "grid.color": "#eeeeee",
                "grid.linewidth": 0.8,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "font.family": "sans-serif",
                "font.size": 11,
                "axes.titlesize": 13,
                "axes.titleweight": "bold",
                "axes.labelsize": 11,
                "legend.frameon": False,
            }
        )

    @staticmethod
    def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        """Return MAE, RMSE, MAPE and R2 for a pair of real-world arrays."""
        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

        mask = np.abs(y_true) >= 0.01
        mape = float(
            np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        )

        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        return {"MAE (EUR/kg)": mae, "RMSE (EUR/kg)": rmse, "MAPE (%)": mape, "R2": r2}

    def _savefig(self, fig: Figure, name: str) -> None:
        path = self.output_dir / name
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Plot saved -> %s", path)

    def learning_curve(
        self,
        train_losses: list[float],
        val_losses: list[float],
        filename: str = "01_learning_curve.png",
    ) -> None:
        """Plot HuberLoss per epoch for train and validation."""
        epochs = range(1, len(train_losses) + 1)
        best_epoch = int(np.argmin(val_losses)) + 1

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(
            epochs, train_losses, label="Train loss", color="#2563eb", linewidth=1.8
        )
        ax.plot(
            epochs, val_losses, label="Validation loss", color="#dc2626", linewidth=1.8
        )
        ax.axvline(
            best_epoch,
            color="#16a34a",
            linestyle="--",
            linewidth=1.2,
            label=f"Best epoch ({best_epoch})",
        )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("HuberLoss (scaled space)")
        ax.set_title("Learning curve")
        ax.legend()
        fig.tight_layout()
        self._savefig(fig, filename)

    def prediction_vs_actual(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        filename: str = "02_pred_vs_actual.png",
    ) -> None:
        """Scatter predicted vs actual EUR/kg on the validation set."""
        abs_err = np.abs(y_true - y_pred)
        vmax = float(np.percentile(abs_err, 95))

        fig, ax = plt.subplots(figsize=(6, 6))
        sc = ax.scatter(
            y_true,
            y_pred,
            c=abs_err,
            cmap="YlOrRd",
            vmin=0,
            vmax=vmax,
            alpha=0.75,
            edgecolors="none",
            s=30,
        )
        lims = (
            float(min(y_true.min(), y_pred.min())) * 0.97,
            float(max(y_true.max(), y_pred.max())) * 1.03,
        )
        ax.plot(lims, lims, color="#16a34a", linewidth=1.5, label="Perfect prediction")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("Actual price (EUR/kg)")
        ax.set_ylabel("Predicted price (EUR/kg)")
        ax.set_title("Prediction vs actual - validation set")
        ax.legend()
        cb = fig.colorbar(sc, ax=ax, shrink=0.8)
        cb.set_label("Absolute error (EUR/kg)")
        fig.tight_layout()
        self._savefig(fig, filename)

    def residuals_over_time(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        dates: pd.DatetimeIndex,
        filename: str = "03_residuals.png",
    ) -> None:
        """Plot signed residuals (pred - actual) over time with a +/-MAE band."""
        residuals = y_pred - y_true
        mae = float(np.mean(np.abs(residuals)))

        fig, ax = plt.subplots(figsize=(11, 4))
        ax.fill_between(
            dates, -mae, mae, color="#2563eb", alpha=0.10, label="+/-MAE band"
        )
        ax.plot(dates, residuals, color="#2563eb", linewidth=1.2, alpha=0.85)
        ax.axhline(0, color="#374151", linewidth=1.0, linestyle="--")
        ax.set_xlabel("Date")
        ax.set_ylabel("Residual (pred - actual, EUR/kg)")
        ax.set_title("Residuals over time - validation set")
        ax.legend()
        fig.tight_layout()
        self._savefig(fig, filename)

    def metrics_dashboard(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        filename: str = "04_metrics_dashboard.png",
    ) -> None:
        """Render MAE, RMSE, MAPE and R2 as large KPI cards."""
        metrics = self._compute_metrics(y_true, y_pred)
        labels = list(metrics.keys())
        values = list(metrics.values())
        fmts = [".3f", ".3f", ".1f", ".4f"]
        colors = ["#2563eb", "#7c3aed", "#db2777", "#16a34a"]

        fig, axes = plt.subplots(1, 4, figsize=(12, 3))
        for ax, label, value, fmt, color in zip(axes, labels, values, fmts, colors):
            ax.set_facecolor(color + "18")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")
            ax.text(
                0.5,
                0.62,
                f"{value:{fmt}}",
                ha="center",
                va="center",
                fontsize=28,
                fontweight="bold",
                color=color,
                transform=ax.transAxes,
            )
            ax.text(
                0.5,
                0.28,
                label,
                ha="center",
                va="center",
                fontsize=12,
                color="#374151",
                transform=ax.transAxes,
            )
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(1.5)
                spine.set_visible(True)

        fig.suptitle(
            "Model performance - validation set", fontsize=14, fontweight="bold", y=1.02
        )
        fig.tight_layout()
        self._savefig(fig, filename)
        logger.info(
            "Metrics: %s", " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        )

    def full_report(
        self,
        train_losses: list[float],
        val_losses: list[float],
        y_true: np.ndarray,
        y_pred: np.ndarray,
        dates: pd.DatetimeIndex,
    ) -> None:
        """Generate all four plots in one call."""
        logger.info("Generating full visual report -> %s/", self.output_dir)
        self.learning_curve(train_losses, val_losses)
        self.prediction_vs_actual(y_true, y_pred)
        self.residuals_over_time(y_true, y_pred, dates)
        self.metrics_dashboard(y_true, y_pred)
        logger.info("Full report complete.")
