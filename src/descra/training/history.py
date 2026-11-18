"""Training telemetry recorder, CSV/JSON metadata serializer, and loss curve plotting."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from descra.utils.logging import get_logger

logger = get_logger("descra.training.history")


class TrainingHistory:
    """Records epoch-by-epoch training and validation loss, memory usage, and plots metrics."""

    def __init__(self, reports_dir: str = "reports/training", figures_dir: str = "reports/training/figures"):
        self.reports_dir = Path(reports_dir)
        self.figures_dir = Path(figures_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        self.history: List[Dict[str, Any]] = []

    def record_epoch(
        self,
        epoch: int,
        train_loss: float,
        eval_loss: float,
        learning_rate: float,
        train_duration_sec: float,
        eval_duration_sec: float,
        peak_gpu_memory_mb: float,
        global_step: int,
    ) -> None:
        """Appends metrics for a single completed training epoch."""
        entry = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": round(float(train_loss), 4),
            "eval_loss": round(float(eval_loss), 4),
            "loss_gap": round(float(eval_loss - train_loss), 4),
            "learning_rate": round(float(learning_rate), 8),
            "train_duration_sec": round(float(train_duration_sec), 2),
            "eval_duration_sec": round(float(eval_duration_sec), 2),
            "peak_gpu_memory_mb": round(float(peak_gpu_memory_mb), 2),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.history.append(entry)
        logger.info(
            f"Epoch {epoch:02d} Summary | Train Loss: {train_loss:.4f} | "
            f"Eval Loss: {eval_loss:.4f} | LR: {learning_rate:.2e} | "
            f"Peak VRAM: {peak_gpu_memory_mb:.1f} MB | Train Time: {train_duration_sec:.1f}s"
        )

    def save_csv(self, filename: str = "training_history.csv") -> Path:
        """Exports epoch history to tabular CSV format."""
        csv_path = self.reports_dir / filename
        df = pd.DataFrame(self.history)
        df.to_csv(csv_path, index=False)
        logger.info(f"Training history CSV saved to: {csv_path}")
        return csv_path

    def save_metadata(
        self,
        config_dict: Dict[str, Any],
        best_epoch: int,
        best_eval_loss: float,
        total_train_time_sec: float,
        gpu_info: Dict[str, Any],
        filename: str = "training_metadata.json",
    ) -> Path:
        """Serializes training run metadata and summary statistics to JSON."""
        meta_path = self.reports_dir / filename
        metadata = {
            "experiment": "Step 8 Supervised Fine-Tuning (SFT)",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "best_checkpoint": {
                "best_epoch": best_epoch,
                "best_eval_loss": round(float(best_eval_loss), 4),
                "metric_used": "eval_loss",
            },
            "training_summary": {
                "total_epochs_trained": len(self.history),
                "total_train_time_sec": round(float(total_train_time_sec), 2),
                "initial_train_loss": self.history[0]["train_loss"] if self.history else None,
                "final_train_loss": self.history[-1]["train_loss"] if self.history else None,
                "initial_eval_loss": self.history[0]["eval_loss"] if self.history else None,
                "final_eval_loss": self.history[-1]["eval_loss"] if self.history else None,
            },
            "configuration": config_dict,
            "hardware": gpu_info,
            "epoch_records": self.history,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Training metadata saved to: {meta_path}")
        return meta_path

    def generate_plots(self) -> List[Path]:
        """Generates publication-quality metric visualization plots."""
        if not self.history:
            return []

        epochs = [h["epoch"] for h in self.history]
        train_losses = [h["train_loss"] for h in self.history]
        eval_losses = [h["eval_loss"] for h in self.history]
        lrs = [h["learning_rate"] for h in self.history]

        saved_plots: List[Path] = []
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

        # 1. Training Loss Curve
        p1 = self.figures_dir / "01_training_loss_curve.png"
        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
        ax.plot(epochs, train_losses, marker="o", color="#1f77b4", linewidth=2.5, label="Train Loss (Cross-Entropy)")
        ax.set_title("Descra SFT — Training Loss vs. Epoch (FLAN-T5 Small)", fontsize=13, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Cross-Entropy Loss", fontsize=11)
        ax.set_xticks(epochs)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(frameon=True)
        plt.tight_layout()
        plt.savefig(p1)
        plt.close()
        saved_plots.append(p1)

        # 2. Validation Loss Curve
        p2 = self.figures_dir / "02_validation_loss_curve.png"
        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
        ax.plot(epochs, eval_losses, marker="s", color="#ff7f0e", linewidth=2.5, label="Validation Loss")
        ax.set_title("Descra SFT — Validation Loss vs. Epoch", fontsize=13, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Validation Cross-Entropy Loss", fontsize=11)
        ax.set_xticks(epochs)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(frameon=True)
        plt.tight_layout()
        plt.savefig(p2)
        plt.close()
        saved_plots.append(p2)

        # 3. Train vs Validation Loss Comparison
        p3 = self.figures_dir / "03_train_val_loss_comparison.png"
        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
        ax.plot(epochs, train_losses, marker="o", color="#1f77b4", linewidth=2.5, label="Training Loss")
        ax.plot(epochs, eval_losses, marker="s", color="#ff7f0e", linewidth=2.5, label="Validation Loss")
        ax.set_title("Descra SFT — Training vs. Validation Loss Convergence", fontsize=13, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Loss", fontsize=11)
        ax.set_xticks(epochs)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(frameon=True)
        plt.tight_layout()
        plt.savefig(p3)
        plt.close()
        saved_plots.append(p3)

        # 4. Learning Rate Schedule
        p4 = self.figures_dir / "04_learning_rate_schedule.png"
        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
        ax.plot(epochs, lrs, marker="^", color="#2ca02c", linewidth=2.5, label="Learning Rate")
        ax.set_title("Descra SFT — Learning Rate Schedule", fontsize=13, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Learning Rate (AdamW)", fontsize=11)
        ax.set_xticks(epochs)
        ax.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(frameon=True)
        plt.tight_layout()
        plt.savefig(p4)
        plt.close()
        saved_plots.append(p4)

        logger.info(f"Successfully generated 4 training figures in: {self.figures_dir}")
        return saved_plots
