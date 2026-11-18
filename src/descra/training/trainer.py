"""PyTorch Seq2Seq Trainer with FP16 AMP, Gradient Accumulation, and Validation Monitoring."""

from datetime import datetime, timezone
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from descra.training.checkpoint import CheckpointManager
from descra.training.config import TrainingConfig
from descra.training.dataset import DataCollatorForSFT, ProductDescriptionDataset
from descra.training.history import TrainingHistory
from descra.training.optimizer import get_optimizer_and_scheduler
from descra.utils.logging import get_logger

logger = get_logger("descra.training.trainer")


class Seq2SeqTrainer:
    """Orchestrates supervised fine-tuning of Seq2Seq Transformers with FP16 mixed precision."""

    def __init__(
        self,
        config: TrainingConfig,
        model: Optional[PreTrainedModel] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        device: Optional[torch.device] = None,
    ):
        self.config = config

        # Set deterministic seeds
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
        np.random.seed(config.seed)

        # Device selection
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device

        logger.info(f"Target compute device: {self.device}")

        # Load Tokenizer
        if tokenizer is None:
            logger.info(f"Loading tokenizer: {config.tokenizer_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name, local_files_only=True)
        else:
            self.tokenizer = tokenizer

        # Load Model
        if model is None:
            logger.info(f"Loading pretrained Seq2Seq model: {config.model_name}")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name, local_files_only=True)
        else:
            self.model = model

        self.model.to(self.device)

        # Dataset & Dataloaders
        self.collator = DataCollatorForSFT(tokenizer=self.tokenizer, pad_to_multiple_of=8)

        logger.info("Initializing SFT Datasets...")
        self.train_dataset = ProductDescriptionDataset(
            jsonl_path=config.train_path,
            tokenizer=self.tokenizer,
            max_input_length=config.max_input_length,
            max_target_length=config.max_target_length,
        )
        self.val_dataset = ProductDescriptionDataset(
            jsonl_path=config.validation_path,
            tokenizer=self.tokenizer,
            max_input_length=config.max_input_length,
            max_target_length=config.max_target_length,
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.per_device_train_batch_size,
            shuffle=True,
            collate_fn=self.collator,
            num_workers=config.dataloader_num_workers,
            pin_memory=(self.device.type == "cuda"),
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=self.collator,
            num_workers=config.dataloader_num_workers,
            pin_memory=(self.device.type == "cuda"),
        )

        # Compute total training steps
        steps_per_epoch = len(self.train_loader) // config.gradient_accumulation_steps
        self.total_training_steps = max(1, steps_per_epoch * config.num_train_epochs)

        # Optimizer & Scheduler
        self.optimizer, self.scheduler = get_optimizer_and_scheduler(
            model=self.model,
            config=config,
            num_training_steps=self.total_training_steps,
        )

        # Mixed Precision Setup
        if self.device.type == "cuda":
            if config.bf16 and torch.cuda.is_bf16_supported():
                self.amp_dtype = torch.bfloat16
                self.use_amp = True
                self.use_scaler = False
                logger.info("Using Native Hardware BFloat16 (BF16) Mixed Precision without GradScaler.")
            elif config.fp16:
                self.amp_dtype = torch.float16
                self.use_amp = True
                self.use_scaler = True
                logger.info("Using Float16 (FP16) Mixed Precision with GradScaler.")
            else:
                self.amp_dtype = torch.float32
                self.use_amp = False
                self.use_scaler = False
                logger.info("Using standard FP32 precision.")
        else:
            self.amp_dtype = torch.float32
            self.use_amp = False
            self.use_scaler = False

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_scaler)

        # Management components
        self.checkpoint_manager = CheckpointManager(output_dir=config.output_dir, save_total_limit=config.save_total_limit)
        self.history = TrainingHistory(reports_dir=config.reports_dir, figures_dir=config.figures_dir)

    def evaluate(self) -> Tuple[float, float]:
        """Evaluates model validation loss across validation batches."""
        self.model.eval()
        total_eval_loss = 0.0
        eval_steps = 0

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                with torch.amp.autocast("cuda", enabled=self.use_amp, dtype=self.amp_dtype):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs.loss

                total_eval_loss += loss.item()
                eval_steps += 1

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        eval_duration = time.perf_counter() - t0

        avg_loss = total_eval_loss / eval_steps if eval_steps > 0 else 0.0
        return avg_loss, eval_duration

    def train(self) -> Dict[str, Any]:
        """Executes full multi-epoch supervised fine-tuning loop."""
        logger.info("=" * 60)
        logger.info("STARTING DESCRA SUPERVISED FINE-TUNING LOOP")
        logger.info(f"Model: {self.config.model_name} on {self.device}")
        logger.info(f"Train samples: {len(self.train_dataset)} | Val samples: {len(self.val_dataset)}")
        logger.info(f"Batch size: {self.config.per_device_train_batch_size} (effective: {self.config.effective_batch_size})")
        logger.info(f"Epochs: {self.config.num_train_epochs} | Total steps: {self.total_training_steps}")
        logger.info("=" * 60)

        best_eval_loss = float("inf")
        best_epoch = -1
        patience_counter = 0
        global_step = 0

        t_train_start = time.perf_counter()

        for epoch in range(1, self.config.num_train_epochs + 1):
            self.model.train()
            total_train_loss = 0.0
            accumulated_loss = 0.0

            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)
                torch.cuda.synchronize()
            t_epoch_start = time.perf_counter()

            self.optimizer.zero_grad()

            for step, batch in enumerate(self.train_loader, 1):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                with torch.amp.autocast("cuda", enabled=self.use_amp, dtype=self.amp_dtype):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs.loss / self.config.gradient_accumulation_steps

                # Backward pass
                if self.use_scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                accumulated_loss += loss.item() * self.config.gradient_accumulation_steps

                # Optimization step after accumulating gradients
                if step % self.config.gradient_accumulation_steps == 0 or step == len(self.train_loader):
                    if self.use_scaler:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                        self.optimizer.step()

                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1

                    total_train_loss += accumulated_loss
                    accumulated_loss = 0.0

                    if global_step % self.config.logging_steps == 0:
                        cur_lr = self.scheduler.get_last_lr()[0]
                        logger.info(f"Epoch {epoch:02d} | Step {global_step:04d}/{self.total_training_steps} | Loss: {loss.item() * self.config.gradient_accumulation_steps:.4f} | LR: {cur_lr:.2e}")

            if self.device.type == "cuda":
                torch.cuda.synchronize()
            train_duration = time.perf_counter() - t_epoch_start

            steps_in_epoch = len(self.train_loader) // self.config.gradient_accumulation_steps
            avg_train_loss = total_train_loss / steps_in_epoch if steps_in_epoch > 0 else 0.0

            # Validation evaluation
            logger.info(f"Running validation evaluation for Epoch {epoch:02d}...")
            avg_eval_loss, eval_duration = self.evaluate()

            # Peak GPU memory
            if self.device.type == "cuda":
                peak_gpu_mb = torch.cuda.max_memory_allocated(self.device) / (1024**2)
            else:
                peak_gpu_mb = 0.0

            # Record history
            cur_lr = self.scheduler.get_last_lr()[0] if self.scheduler.get_last_lr() else self.config.learning_rate
            self.history.record_epoch(
                epoch=epoch,
                train_loss=avg_train_loss,
                eval_loss=avg_eval_loss,
                learning_rate=cur_lr,
                train_duration_sec=train_duration,
                eval_duration_sec=eval_duration,
                peak_gpu_memory_mb=peak_gpu_mb,
                global_step=global_step,
            )

            # Check for best model
            is_best = avg_eval_loss < best_eval_loss
            if is_best:
                best_eval_loss = avg_eval_loss
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1

            # Save checkpoint
            self.checkpoint_manager.save_checkpoint(
                model=self.model,
                tokenizer=self.tokenizer,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                epoch=epoch,
                global_step=global_step,
                train_loss=avg_train_loss,
                eval_loss=avg_eval_loss,
                is_best=is_best,
            )

            # Early stopping check
            if patience_counter >= self.config.early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch} epochs (Patience={self.config.early_stopping_patience}).")
                break

        total_train_time = time.perf_counter() - t_train_start

        # Save final checkpoint
        self.checkpoint_manager.save_checkpoint(
            model=self.model,
            tokenizer=self.tokenizer,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            epoch=epoch,
            global_step=global_step,
            train_loss=avg_train_loss,
            eval_loss=avg_eval_loss,
            is_best=False,
            tag="final_checkpoint",
        )

        # Telemetry & Plotting
        self.history.save_csv()
        self.history.generate_plots()

        gpu_info = {}
        if self.device.type == "cuda":
            props = torch.cuda.get_device_properties(self.device)
            gpu_info = {
                "gpu_name": torch.cuda.get_device_name(self.device),
                "total_vram_mb": round(props.total_memory / (1024**2), 1),
                "peak_training_vram_mb": round(peak_gpu_mb, 1),
            }

        self.history.save_metadata(
            config_dict=self.config.__dict__,
            best_epoch=best_epoch,
            best_eval_loss=best_eval_loss,
            total_train_time_sec=total_train_time,
            gpu_info=gpu_info,
        )

        logger.info("=" * 60)
        logger.info("DESCRA SFT COMPLETED SUCCESSFULLY!")
        logger.info(f"Best Epoch: {best_epoch} | Best Eval Loss: {best_eval_loss:.4f}")
        logger.info(f"Total Time: {total_train_time:.1f} s | Peak VRAM: {peak_gpu_mb:.1f} MB")
        logger.info(f"Best Checkpoint: {self.checkpoint_manager.best_checkpoint_dir}")
        logger.info("=" * 60)

        return {
            "best_epoch": best_epoch,
            "best_eval_loss": best_eval_loss,
            "total_train_time_sec": total_train_time,
            "peak_vram_mb": peak_gpu_mb,
            "best_checkpoint_path": str(self.checkpoint_manager.best_checkpoint_dir),
            "final_checkpoint_path": str(self.checkpoint_manager.final_checkpoint_dir),
            "history": self.history.history,
        }
