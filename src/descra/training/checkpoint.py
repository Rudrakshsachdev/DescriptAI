"""Checkpoint serialization, restoration, and retention management for Descra SFT."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
from typing import Any, Dict, Optional, Tuple
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from descra.utils.logging import get_logger

logger = get_logger("descra.training.checkpoint")


class CheckpointManager:
    """Handles saving, loading, and retention of model checkpoints and optimizer states."""

    def __init__(self, output_dir: str = "checkpoints/sft", save_total_limit: int = 2):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_total_limit = save_total_limit
        self.best_checkpoint_dir = self.output_dir / "best_checkpoint"
        self.final_checkpoint_dir = self.output_dir / "final_checkpoint"

    def save_checkpoint(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        scaler: Optional[torch.cuda.amp.GradScaler],
        epoch: int,
        global_step: int,
        train_loss: float,
        eval_loss: float,
        is_best: bool = False,
        tag: Optional[str] = None,
    ) -> Path:
        """Saves model weights, tokenizer, optimizer, scheduler, and training state."""
        checkpoint_name = tag if tag else f"checkpoint-epoch-{epoch}-step-{global_step}"
        target_dir = self.output_dir / checkpoint_name
        target_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving checkpoint to: {target_dir} (Epoch {epoch}, Step {global_step}, Eval Loss: {eval_loss:.4f})")

        # Save model & tokenizer using standard HuggingFace methods
        model.save_pretrained(target_dir)
        tokenizer.save_pretrained(target_dir)

        # Save training states
        state_dict = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": train_loss,
            "eval_loss": eval_loss,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_architecture": model.__class__.__name__,
        }
        with open(target_dir / "training_state.json", "w", encoding="utf-8") as f:
            json.dump(state_dict, f, indent=2)

        # Save optimizer, scheduler, and scaler
        torch.save(optimizer.state_dict(), target_dir / "optimizer.pt")
        torch.save(scheduler.state_dict(), target_dir / "scheduler.pt")
        if scaler is not None:
            torch.save(scaler.state_dict(), target_dir / "scaler.pt")

        # Copy to best_checkpoint if this is the best evaluation loss so far
        if is_best:
            logger.info(f"New best model identified (Eval Loss: {eval_loss:.4f})! Updating {self.best_checkpoint_dir}...")
            if self.best_checkpoint_dir.exists():
                shutil.rmtree(self.best_checkpoint_dir)
            shutil.copytree(target_dir, self.best_checkpoint_dir)

        return target_dir

    def load_checkpoint(
        self,
        checkpoint_dir: str,
        model: PreTrainedModel,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
    ) -> Dict[str, Any]:
        """Restores model weights, optimizer, scheduler, and training state from disk."""
        target_dir = Path(checkpoint_dir)
        if not target_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory does not exist: {target_dir}")

        logger.info(f"Restoring training state from checkpoint: {target_dir}")

        # Load weights
        from transformers import AutoModelForSeq2SeqLM
        restored_model = AutoModelForSeq2SeqLM.from_pretrained(target_dir)
        model.load_state_dict(restored_model.state_dict())

        # Load states
        state_file = target_dir / "training_state.json"
        state = {}
        if state_file.exists():
            with open(state_file, "r", encoding="utf-8") as f:
                state = json.load(f)

        if optimizer and (target_dir / "optimizer.pt").exists():
            optimizer.load_state_dict(torch.load(target_dir / "optimizer.pt"))

        if scheduler and (target_dir / "scheduler.pt").exists():
            scheduler.load_state_dict(torch.load(target_dir / "scheduler.pt"))

        if scaler and (target_dir / "scaler.pt").exists():
            scaler.load_state_dict(torch.load(target_dir / "scaler.pt"))

        logger.info(f"Successfully loaded checkpoint state: {state}")
        return state
