"""Training configuration dataclass and YAML loader for Descra SFT."""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml


@dataclass
class TrainingConfig:
    """Configuration hyperparameters and environment paths for Supervised Fine-Tuning."""
    model_name: str = "google/flan-t5-small"
    model_revision: str = "main"
    tokenizer_name: str = "google/flan-t5-small"

    train_path: str = "data/processed/sft_train.jsonl"
    validation_path: str = "data/processed/sft_validation.jsonl"
    test_path: str = "data/processed/test.jsonl"

    max_input_length: int = 512
    max_target_length: int = 160

    output_dir: str = "checkpoints/sft"
    reports_dir: str = "reports/training"
    figures_dir: str = "reports/training/figures"

    num_train_epochs: int = 4
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 2
    learning_rate: float = 3.0e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.10
    lr_scheduler_type: str = "linear"
    max_grad_norm: float = 1.0

    fp16: bool = True
    bf16: bool = False
    seed: int = 42

    early_stopping_patience: int = 2
    metric_for_best_model: str = "eval_loss"
    save_total_limit: int = 2
    logging_steps: int = 50
    dataloader_num_workers: int = 0

    eval_sample_size: int = 100
    eval_strategies: List[str] = field(default_factory=lambda: ["greedy", "beam_search_2", "beam_search_4"])

    @property
    def effective_batch_size(self) -> int:
        return self.per_device_train_batch_size * self.gradient_accumulation_steps

    @classmethod
    def from_yaml(cls, yaml_path: str = "configs/training.yaml") -> "TrainingConfig":
        """Loads configuration from YAML file."""
        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"Training config not found at: {yaml_path}")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        m_cfg = data.get("model", {})
        d_cfg = data.get("dataset", {})
        t_cfg = data.get("training", {})
        e_cfg = data.get("evaluation", {})

        return cls(
            model_name=m_cfg.get("name", "google/flan-t5-small"),
            model_revision=m_cfg.get("model_revision", "main"),
            tokenizer_name=m_cfg.get("tokenizer_name", "google/flan-t5-small"),
            train_path=d_cfg.get("train_path", "data/processed/sft_train.jsonl"),
            validation_path=d_cfg.get("validation_path", "data/processed/sft_validation.jsonl"),
            test_path=d_cfg.get("test_path", "data/processed/test.jsonl"),
            max_input_length=d_cfg.get("max_input_length", 512),
            max_target_length=d_cfg.get("max_target_length", 160),
            output_dir=t_cfg.get("output_dir", "checkpoints/sft"),
            reports_dir=t_cfg.get("reports_dir", "reports/training"),
            figures_dir=t_cfg.get("figures_dir", "reports/training/figures"),
            num_train_epochs=t_cfg.get("num_train_epochs", 4),
            per_device_train_batch_size=t_cfg.get("per_device_train_batch_size", 8),
            per_device_eval_batch_size=t_cfg.get("per_device_eval_batch_size", 8),
            gradient_accumulation_steps=t_cfg.get("gradient_accumulation_steps", 2),
            learning_rate=float(t_cfg.get("learning_rate", 3.0e-4)),
            weight_decay=float(t_cfg.get("weight_decay", 0.01)),
            warmup_ratio=float(t_cfg.get("warmup_ratio", 0.10)),
            lr_scheduler_type=t_cfg.get("lr_scheduler_type", "linear"),
            max_grad_norm=float(t_cfg.get("max_grad_norm", 1.0)),
            fp16=t_cfg.get("fp16", True),
            bf16=t_cfg.get("bf16", False),
            seed=t_cfg.get("seed", 42),
            early_stopping_patience=t_cfg.get("early_stopping_patience", 2),
            metric_for_best_model=t_cfg.get("metric_for_best_model", "eval_loss"),
            save_total_limit=t_cfg.get("save_total_limit", 2),
            logging_steps=t_cfg.get("logging_steps", 50),
            dataloader_num_workers=t_cfg.get("dataloader_num_workers", 0),
            eval_sample_size=e_cfg.get("sample_size", 100),
            eval_strategies=e_cfg.get("strategies", ["greedy", "beam_search_2", "beam_search_4"]),
        )
