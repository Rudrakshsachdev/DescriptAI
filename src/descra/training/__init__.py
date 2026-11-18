"""Training package for Descra: Seq2Seq supervised fine-tuning with PyTorch and Hugging Face."""

from descra.training.checkpoint import CheckpointManager
from descra.training.config import TrainingConfig
from descra.training.dataset import DataCollatorForSFT, ProductDescriptionDataset
from descra.training.history import TrainingHistory
from descra.training.optimizer import get_optimizer_and_scheduler
from descra.training.trainer import Seq2SeqTrainer

__all__ = [
    "TrainingConfig",
    "ProductDescriptionDataset",
    "DataCollatorForSFT",
    "CheckpointManager",
    "TrainingHistory",
    "get_optimizer_and_scheduler",
    "Seq2SeqTrainer",
]
