"""Optimizer and Learning Rate Scheduler setup for Supervised Fine-Tuning."""

from typing import Tuple
import torch
from torch.optim import AdamW
from transformers import PreTrainedModel, get_linear_schedule_with_warmup

from descra.training.config import TrainingConfig
from descra.utils.logging import get_logger

logger = get_logger("descra.training.optimizer")


def get_optimizer_and_scheduler(
    model: PreTrainedModel,
    config: TrainingConfig,
    num_training_steps: int,
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    """Constructs AdamW optimizer with weight decay exemptions and linear warmup scheduler."""
    # Exclude bias and LayerNorm weights from weight decay
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": config.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=config.learning_rate,
        eps=1e-8,
        betas=(0.9, 0.999),
    )

    num_warmup_steps = int(num_training_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    logger.info(
        f"Initialized AdamW (lr={config.learning_rate}, weight_decay={config.weight_decay}) "
        f"with {num_warmup_steps} warmup steps out of {num_training_steps} total steps."
    )
    return optimizer, scheduler
