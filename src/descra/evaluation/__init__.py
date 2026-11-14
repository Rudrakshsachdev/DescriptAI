"""Evaluation package for Descra: BLEU, ROUGE, BERTScore, and factual verification."""

from descra.evaluation.evaluator import BaselineEvaluator
from descra.evaluation.factual_checks import verify_factual_adherence
from descra.evaluation.metrics import (
    calculate_bleu,
    calculate_rouge,
    compute_generation_metrics,
)

__all__ = [
    "calculate_bleu",
    "calculate_rouge",
    "compute_generation_metrics",
    "verify_factual_adherence",
    "BaselineEvaluator",
]

