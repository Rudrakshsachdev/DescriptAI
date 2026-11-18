"""Master CLI script to execute Supervised Fine-Tuning (SFT) for Descra."""

import argparse
import os
from pathlib import Path
import sys

# Ensure offline mode and paths
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from descra.training.config import TrainingConfig
from descra.training.trainer import Seq2SeqTrainer
from descra.utils.logging import get_logger

logger = get_logger("descra.scripts.train")


def main():
    parser = argparse.ArgumentParser(description="Execute Supervised Fine-Tuning (SFT) for Descra.")
    parser.add_argument("--config", type=str, default="configs/training.yaml", help="Path to training config YAML.")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override per-device training batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate.")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint directory to resume from.")
    args = parser.parse_args()

    print("=" * 70)
    print("DESCRA — STEP 8: SUPERVISED FINE-TUNING (SFT)")
    print("=" * 70)

    # 1. Load configuration
    config = TrainingConfig.from_yaml(args.config)
    if args.epochs is not None:
        config.num_train_epochs = args.epochs
    if args.batch_size is not None:
        config.per_device_train_batch_size = args.batch_size
    if args.lr is not None:
        config.learning_rate = args.lr

    print(f"Model Architecture:       {config.model_name}")
    print(f"Training Partition:       {config.train_path}")
    print(f"Validation Partition:     {config.validation_path}")
    print(f"Epochs:                   {config.num_train_epochs}")
    print(f"Batch Size (Per-Device):  {config.per_device_train_batch_size} (Effective: {config.effective_batch_size})")
    print(f"Learning Rate:            {config.learning_rate:.2e}")
    print(f"FP16 Mixed Precision:     {config.fp16}")
    print(f"Checkpoints Output Dir:   {config.output_dir}")
    print("=" * 70)

    # 2. Initialize Trainer
    trainer = Seq2SeqTrainer(config=config)

    # 3. Optional resume
    if args.resume:
        print(f"Resuming training from checkpoint: {args.resume}")
        trainer.checkpoint_manager.load_checkpoint(
            checkpoint_dir=args.resume,
            model=trainer.model,
            optimizer=trainer.optimizer,
            scheduler=trainer.scheduler,
            scaler=trainer.scaler,
        )

    # 4. Train Model
    results = trainer.train()

    print("\n" + "=" * 70)
    print("SFT EXECUTION SUMMARY")
    print("=" * 70)
    print(f"Best Validation Epoch:    {results['best_epoch']}")
    print(f"Best Validation Loss:     {results['best_eval_loss']:.4f}")
    print(f"Total Training Duration:  {results['total_train_time_sec']:.1f} s")
    print(f"Peak GPU VRAM Allocated:  {results['peak_vram_mb']:.1f} MB")
    print(f"Best Model Checkpoint:    {results['best_checkpoint_path']}")
    print(f"Final Model Checkpoint:   {results['final_checkpoint_path']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
