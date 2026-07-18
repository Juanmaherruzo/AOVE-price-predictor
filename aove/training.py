"""Training and fine-tuning engine for the AOVE price predictor."""

import argparse
import logging
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from aove.config import DEVICE, HF_COLS, MACRO_COLS, TARGET_COL, settings
from aove.etl import MarketETLPipeline
from aove.features import TemporalSequenceBuilder
from aove.model import AOVEPricePredictor
from aove.visualise import AOVEVisualiser

logger = logging.getLogger(__name__)

# One batch yields (climate window, macro snapshot, target) tensors.
Batch = tuple[Tensor, ...]


class AOVETrainer:
    """Training loop with HuberLoss, LR scheduling, early stopping and checkpoints."""

    def __init__(
        self,
        model: nn.Module,
        sequence_builder: TemporalSequenceBuilder,
        learning_rate: float = 1e-3,
        lr_patience: int = 10,
        es_patience: int = 20,
    ) -> None:
        self.model = model.to(DEVICE)
        self.sequence_builder = sequence_builder
        self.criterion = nn.HuberLoss(delta=1.0)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=1e-5
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", patience=lr_patience, factor=0.5
        )
        self.es_patience = es_patience
        self.history: dict[str, list[float]] = {"train": [], "val": []}

    def _run_epoch(self, loader: DataLoader[Batch], train: bool) -> float:
        self.model.train(train)
        total_loss = 0.0
        with torch.set_grad_enabled(train):
            for x_hf, x_macro, y in loader:
                x_hf = x_hf.to(DEVICE, dtype=torch.float32)
                x_macro = x_macro.to(DEVICE, dtype=torch.float32)
                y = y.to(DEVICE, dtype=torch.float32).view(-1, 1)
                preds = self.model(x_hf, x_macro)
                loss = self.criterion(preds, y)
                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                total_loss += loss.item()
        return total_loss / len(loader)

    def train_and_validate(
        self,
        train_loader: DataLoader[Batch],
        val_loader: DataLoader[Batch],
        epochs: int,
        save_path: Path = Path("best_aove_model.pth"),
    ) -> None:
        """Run the full training loop with checkpointing and early stopping."""
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            avg_train = self._run_epoch(train_loader, train=True)
            avg_val = self._run_epoch(val_loader, train=False)
            self.scheduler.step(avg_val)

            logger.info(
                "Epoch %4d/%d | Train: %.5f | Val: %.5f | LR: %.2e",
                epoch + 1,
                epochs,
                avg_train,
                avg_val,
                self.optimizer.param_groups[0]["lr"],
            )
            self.history["train"].append(avg_train)
            self.history["val"].append(avg_val)

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                patience_counter = 0
                torch.save(self.model.state_dict(), save_path)
                logger.info("  New best model saved -> %s", save_path)
            else:
                patience_counter += 1
                if patience_counter >= self.es_patience:
                    logger.info(
                        "Early stopping after %d epochs (no improvement for %d).",
                        epoch + 1,
                        self.es_patience,
                    )
                    break

        logger.info("Training complete. Best val loss: %.5f", best_val_loss)

    def predict(self, loader: DataLoader[Batch]) -> np.ndarray:
        """Run inference and return predictions in real-world EUR/kg."""
        self.model.eval()
        chunks: list[np.ndarray] = []
        with torch.no_grad():
            for x_hf, x_macro, _ in loader:
                x_hf = x_hf.to(DEVICE, dtype=torch.float32)
                x_macro = x_macro.to(DEVICE, dtype=torch.float32)
                chunks.append(self.model(x_hf, x_macro).cpu().numpy())
        return self.sequence_builder.inverse_transform_target(
            np.concatenate(chunks, axis=0)
        )


class AOVEFineTuner:
    """Fine-tune the FC head of a pre-trained model, freezing the LSTM layers."""

    def __init__(
        self,
        model: nn.Module,
        sequence_builder: TemporalSequenceBuilder,
        checkpoint_path: Path,
        learning_rate: float = 1e-4,
        lr_patience: int = 5,
        es_patience: int = 10,
    ) -> None:
        self.model = model.to(DEVICE)
        self.sequence_builder = sequence_builder
        self.checkpoint_path = checkpoint_path
        self.criterion = nn.HuberLoss(delta=1.0)
        self.lr = learning_rate
        self.lr_patience = lr_patience
        self.es_patience = es_patience
        self.history: dict[str, list[float]] = {"train": [], "val": []}

    def load_and_freeze(self) -> None:
        """Load the checkpoint and freeze every LSTM parameter."""
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {self.checkpoint_path}\n"
                "Train the base model first (run without --finetune)."
            )
        self.model.load_state_dict(
            torch.load(self.checkpoint_path, map_location=DEVICE)
        )
        logger.info("Checkpoint loaded from %s", self.checkpoint_path)

        for name, param in self.model.named_parameters():
            if name.startswith("lstm"):
                param.requires_grad = False

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.model.parameters() if not p.requires_grad)
        logger.info(
            "Layer freeze complete: %d frozen (LSTM) | %d trainable (FC head)",
            frozen,
            trainable,
        )

    def finetune(
        self,
        train_loader: DataLoader[Batch],
        val_loader: DataLoader[Batch],
        epochs: int = 30,
        save_path: Path = Path("finetuned_aove_model.pth"),
    ) -> None:
        """Fine-tune only the unfrozen FC head parameters."""
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.lr,
            weight_decay=1e-5,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=self.lr_patience, factor=0.5
        )
        best_val_loss = float("inf")
        patience_counter = 0
        logger.info(
            "Fine-tuning FC head for up to %d epochs (LR=%.1e)...", epochs, self.lr
        )

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            for x_hf, x_macro, y in train_loader:
                x_hf = x_hf.to(DEVICE, dtype=torch.float32)
                x_macro = x_macro.to(DEVICE, dtype=torch.float32)
                y = y.to(DEVICE, dtype=torch.float32).view(-1, 1)
                loss = self.criterion(self.model(x_hf, x_macro), y)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()
            avg_train = train_loss / len(train_loader)

            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x_hf, x_macro, y in val_loader:
                    x_hf = x_hf.to(DEVICE, dtype=torch.float32)
                    x_macro = x_macro.to(DEVICE, dtype=torch.float32)
                    y = y.to(DEVICE, dtype=torch.float32).view(-1, 1)
                    val_loss += self.criterion(self.model(x_hf, x_macro), y).item()
            avg_val = val_loss / len(val_loader)

            scheduler.step(avg_val)
            self.history["train"].append(avg_train)
            self.history["val"].append(avg_val)
            logger.info(
                "FT Epoch %3d/%d | Train: %.5f | Val: %.5f | LR: %.2e",
                epoch + 1,
                epochs,
                avg_train,
                avg_val,
                optimizer.param_groups[0]["lr"],
            )

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                patience_counter = 0
                torch.save(self.model.state_dict(), save_path)
                logger.info("  Best fine-tuned model saved -> %s", save_path)
            else:
                patience_counter += 1
                if patience_counter >= self.es_patience:
                    logger.info("Fine-tune early stopping at epoch %d.", epoch + 1)
                    break

        logger.info("Fine-tuning complete. Best val loss: %.5f", best_val_loss)

    def predict(self, loader: DataLoader[Batch]) -> np.ndarray:
        """Return fine-tuned predictions in real-world EUR/kg."""
        self.model.eval()
        chunks: list[np.ndarray] = []
        with torch.no_grad():
            for x_hf, x_macro, _ in loader:
                x_hf = x_hf.to(DEVICE, dtype=torch.float32)
                x_macro = x_macro.to(DEVICE, dtype=torch.float32)
                chunks.append(self.model(x_hf, x_macro).cpu().numpy())
        return self.sequence_builder.inverse_transform_target(
            np.concatenate(chunks, axis=0)
        )


def build_dataloaders(
    train_arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
    val_arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
    batch_size: int,
) -> tuple[DataLoader[Batch], DataLoader[Batch]]:
    """Wrap the train/val numpy arrays in PyTorch DataLoaders."""

    def to_loader(
        arrays: tuple[np.ndarray, np.ndarray, np.ndarray], shuffle: bool
    ) -> DataLoader[Batch]:
        tensors = [torch.from_numpy(a) for a in arrays]
        return DataLoader(
            TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle
        )

    return to_loader(train_arrays, True), to_loader(val_arrays, False)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AOVE macro-aware LSTM training engine"
    )
    parser.add_argument("--climate-csv", type=Path, default=settings.climate_path)
    parser.add_argument("--macro-csv", type=Path, default=settings.macro_path)
    parser.add_argument("--time-steps", type=int, default=settings.time_steps)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-ratio", type=float, default=settings.train_ratio)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lr-patience", type=int, default=10)
    parser.add_argument("--es-patience", type=int, default=20)
    parser.add_argument("--aove-lag", type=int, default=settings.aove_lag_weeks)
    parser.add_argument("--save-path", type=Path, default=Path("best_aove_model.pth"))
    parser.add_argument("--finetune", action="store_true", default=False)
    parser.add_argument(
        "--fine-checkpoint", type=Path, default=Path("best_aove_model.pth")
    )
    parser.add_argument("--fine-window-weeks", type=int, default=220)
    parser.add_argument("--fine-epochs", type=int, default=15)
    parser.add_argument("--fine-lr", type=float, default=1e-4)
    parser.add_argument(
        "--fine-save-path", type=Path, default=Path("finetuned_aove_model.pth")
    )
    return parser


def main() -> None:
    """Console-script entry point (``aove-train``)."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _build_arg_parser().parse_args()

    logger.info("=== AOVE deep learning training engine ===  device: %s", DEVICE)
    df_municipal = pd.read_csv(args.climate_csv, parse_dates=["date"])
    df_macro = pd.read_csv(args.macro_csv, parse_dates=["reference_date"])

    etl = MarketETLPipeline(publish_delay_days=settings.publish_delay_days)
    df_climate = etl.aggregate_climate(df_municipal)
    df_final = etl.align_macro_data(df_climate, df_macro, aove_lag_weeks=args.aove_lag)

    macro_cols = list(MACRO_COLS) if args.aove_lag > 0 else MACRO_COLS[:-1]
    builder = TemporalSequenceBuilder(time_steps=args.time_steps)
    splits = builder.build_sequences_split(
        df_final, HF_COLS, macro_cols, TARGET_COL, train_ratio=args.train_ratio
    )
    train_arrays = cast("tuple[np.ndarray, np.ndarray, np.ndarray]", splits["train"])
    val_arrays = cast("tuple[np.ndarray, np.ndarray, np.ndarray]", splits["val"])
    train_loader, val_loader = build_dataloaders(
        train_arrays, val_arrays, args.batch_size
    )

    model = AOVEPricePredictor(
        hf_input_dim=len(HF_COLS), macro_input_dim=len(macro_cols)
    )
    logger.info("Model parameters: %d", sum(p.numel() for p in model.parameters()))

    active_trainer: AOVETrainer | AOVEFineTuner
    if not args.finetune:
        trainer = AOVETrainer(
            model,
            builder,
            learning_rate=args.lr,
            lr_patience=args.lr_patience,
            es_patience=args.es_patience,
        )
        trainer.train_and_validate(
            train_loader, val_loader, epochs=args.epochs, save_path=args.save_path
        )
        active_trainer = trainer
    else:
        logger.info("=== FINE-TUNING MODE ===")
        x_hf_tr, x_macro_tr, y_tr = train_arrays
        ft_start = max(0, len(x_hf_tr) - args.fine_window_weeks)
        ft_train = (x_hf_tr[ft_start:], x_macro_tr[ft_start:], y_tr[ft_start:])
        ft_train_loader, ft_val_loader = build_dataloaders(
            ft_train, val_arrays, args.batch_size
        )
        finetuner = AOVEFineTuner(
            model,
            builder,
            checkpoint_path=args.fine_checkpoint,
            learning_rate=args.fine_lr,
        )
        finetuner.load_and_freeze()
        for module in model.fc.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0.5
        finetuner.finetune(
            ft_train_loader,
            ft_val_loader,
            epochs=args.fine_epochs,
            save_path=args.fine_save_path,
        )
        active_trainer = finetuner

    val_preds = active_trainer.predict(val_loader)
    y_true = builder.inverse_transform_target(val_arrays[2])
    tag = "finetuned" if args.finetune else "base"
    val_dates = cast(pd.DatetimeIndex, splits["val_dates"])
    AOVEVisualiser(output_dir=Path(f"aove_diagnostics_{tag}")).full_report(
        train_losses=active_trainer.history["train"],
        val_losses=active_trainer.history["val"],
        y_true=y_true,
        y_pred=val_preds,
        dates=val_dates,
    )
    logger.info("Pipeline execution finished successfully.")


if __name__ == "__main__":
    main()
