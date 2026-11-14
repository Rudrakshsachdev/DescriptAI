"""Text generation evaluation metrics for Descra (BLEU, ROUGE, and BERTScore)."""

from typing import Any, Dict, List, Optional
import numpy as np
from rouge_score import rouge_scorer
import sacrebleu

from descra.utils.logging import get_logger

logger = get_logger("descra.evaluation.metrics")


def calculate_bleu(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Calculates corpus and sentence-level BLEU scores using SacreBLEU.

    Args:
        predictions: Generated text hypotheses.
        references: Ground truth target strings.

    Returns:
        Dictionary containing corpus_bleu, bleu_1, bleu_2, bleu_3, bleu_4.
    """
    if not predictions or not references:
        return {"corpus_bleu": 0.0, "bleu_1": 0.0, "bleu_2": 0.0, "bleu_3": 0.0, "bleu_4": 0.0}

    # Clean empty strings
    clean_preds = [p if p.strip() else "<empty>" for p in predictions]
    clean_refs = [[r] for r in references]

    bleu = sacrebleu.corpus_bleu(clean_preds, clean_refs)

    # Calculate individual n-gram precisions
    precisions = bleu.precisions if hasattr(bleu, "precisions") else [0.0, 0.0, 0.0, 0.0]
    bleu_1 = precisions[0] if len(precisions) > 0 else 0.0
    bleu_2 = precisions[1] if len(precisions) > 1 else 0.0
    bleu_3 = precisions[2] if len(precisions) > 2 else 0.0
    bleu_4 = precisions[3] if len(precisions) > 3 else 0.0

    return {
        "corpus_bleu": round(float(bleu.score), 2),
        "bleu_1": round(float(bleu_1), 2),
        "bleu_2": round(float(bleu_2), 2),
        "bleu_3": round(float(bleu_3), 2),
        "bleu_4": round(float(bleu_4), 2),
    }


def calculate_rouge(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Calculates ROUGE-1, ROUGE-2, and ROUGE-L F1 scores using rouge_score.

    Args:
        predictions: Generated text hypotheses.
        references: Ground truth target strings.

    Returns:
        Dictionary containing rouge1, rouge2, rougeL precision, recall, and f1.
    """
    if not predictions or not references:
        return {
            "rouge1_f1": 0.0, "rouge1_recall": 0.0, "rouge1_precision": 0.0,
            "rouge2_f1": 0.0, "rouge2_recall": 0.0, "rouge2_precision": 0.0,
            "rougeL_f1": 0.0, "rougeL_recall": 0.0, "rougeL_precision": 0.0,
        }

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    r1_f1, r1_rec, r1_prec = [], [], []
    r2_f1, r2_rec, r2_prec = [], [], []
    rl_f1, rl_rec, rl_prec = [], [], []

    for pred, ref in zip(predictions, references):
        clean_p = pred.strip() if pred.strip() else "<empty>"
        clean_r = ref.strip() if ref.strip() else "<empty>"

        scores = scorer.score(clean_r, clean_p)

        r1 = scores["rouge1"]
        r2 = scores["rouge2"]
        rl = scores["rougeL"]

        r1_f1.append(r1.fmeasure)
        r1_rec.append(r1.recall)
        r1_prec.append(r1.precision)

        r2_f1.append(r2.fmeasure)
        r2_rec.append(r2.recall)
        r2_prec.append(r2.precision)

        rl_f1.append(rl.fmeasure)
        rl_rec.append(rl.recall)
        rl_prec.append(rl.precision)

    return {
        "rouge1_f1": round(float(np.mean(r1_f1)), 4),
        "rouge1_recall": round(float(np.mean(r1_rec)), 4),
        "rouge1_precision": round(float(np.mean(r1_prec)), 4),
        "rouge2_f1": round(float(np.mean(r2_f1)), 4),
        "rouge2_recall": round(float(np.mean(r2_rec)), 4),
        "rouge2_precision": round(float(np.mean(r2_prec)), 4),
        "rougeL_f1": round(float(np.mean(rl_f1)), 4),
        "rougeL_recall": round(float(np.mean(rl_rec)), 4),
        "rougeL_precision": round(float(np.mean(rl_prec)), 4),
    }


def calculate_bertscore(
    predictions: List[str],
    references: List[str],
    device: Optional[str] = None,
) -> Optional[Dict[str, float]]:
    """Calculates BERTScore semantic similarity if offline model checkpoint is available.

    Falls back gracefully if network is isolated and model weights are unindexed.
    """
    try:
        from bert_score import score as bert_score_fn
        P, R, F1 = bert_score_fn(
            predictions,
            references,
            lang="en",
            rescale_with_baseline=True,
            device=device,
            verbose=False,
        )
        return {
            "bertscore_precision": round(float(P.mean().item()), 4),
            "bertscore_recall": round(float(R.mean().item()), 4),
            "bertscore_f1": round(float(F1.mean().item()), 4),
        }
    except Exception as e:
        logger.warning(f"BERTScore computation skipped (offline environment / unindexed weights): {e}")
        return None


def compute_generation_metrics(
    predictions: List[str],
    references: List[str],
    device: Optional[str] = None,
    compute_bertscore_flag: bool = False,
) -> Dict[str, Any]:
    """Computes comprehensive benchmark metrics comparing predictions against references."""
    bleu_results = calculate_bleu(predictions, references)
    rouge_results = calculate_rouge(predictions, references)

    metrics: Dict[str, Any] = {
        "sample_count": len(predictions),
        "bleu": bleu_results,
        "rouge": rouge_results,
    }

    if compute_bertscore_flag:
        bscore = calculate_bertscore(predictions, references, device=device)
        if bscore:
            metrics["bertscore"] = bscore

    return metrics

