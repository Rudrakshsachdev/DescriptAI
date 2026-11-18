"""Paired statistical analysis and bootstrap confidence intervals for Descra test evaluation."""

from typing import Any, Dict, List, Tuple
import numpy as np
import sacrebleu

from descra.utils.logging import get_logger

logger = get_logger("descra.evaluation.statistical")


def compute_paired_bootstrap_ci(
    baseline_scores: List[float],
    finetuned_scores: List[float],
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, Any]:
    """Computes paired bootstrap confidence interval for the difference between two models.

    Args:
        baseline_scores: Metric score for each example from baseline model.
        finetuned_scores: Metric score for each example from fine-tuned model.
        n_bootstrap: Number of bootstrap resamples (default: 1000).
        alpha: Significance level (default 0.05 -> 95% CI).
        seed: Random seed for reproducibility (default: 42).

    Returns:
        Dictionary containing mean difference, 95% CI (lower, upper), p-value estimate,
        and statistical conclusiveness flag.
    """
    if len(baseline_scores) != len(finetuned_scores):
        raise ValueError("Baseline and fine-tuned scores must have identical lengths for paired evaluation.")

    n = len(baseline_scores)
    b_arr = np.array(baseline_scores)
    f_arr = np.array(finetuned_scores)
    diffs = f_arr - b_arr
    mean_diff = float(np.mean(diffs))

    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(n_bootstrap, dtype=float)

    for i in range(n_bootstrap):
        sample_indices = rng.integers(0, n, size=n)
        bootstrap_means[i] = np.mean(diffs[sample_indices])

    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)

    ci_lower = float(np.percentile(bootstrap_means, lower_pct))
    ci_upper = float(np.percentile(bootstrap_means, upper_pct))
    std_err = float(np.std(bootstrap_means))

    # Two-tailed empirical p-value estimate
    # Fraction of bootstrap differences with opposite sign or zero
    if mean_diff > 0:
        p_val = 2.0 * float(np.mean(bootstrap_means <= 0.0))
    elif mean_diff < 0:
        p_val = 2.0 * float(np.mean(bootstrap_means >= 0.0))
    else:
        p_val = 1.0
    p_val = min(1.0, max(0.0, p_val))

    # Improvement is statistically conclusive if 95% CI does not span zero
    is_conclusive = not (ci_lower <= 0.0 <= ci_upper)

    return {
        "n_samples": n,
        "mean_baseline": round(float(np.mean(b_arr)), 4),
        "mean_finetuned": round(float(np.mean(f_arr)), 4),
        "mean_difference": round(mean_diff, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "confidence_level": 1.0 - alpha,
        "std_error": round(std_err, 4),
        "p_value_estimate": round(p_val, 4),
        "statistically_conclusive": is_conclusive,
        "n_bootstrap": n_bootstrap,
        "seed": seed,
    }


def compute_paired_corpus_bleu_ci(
    baseline_predictions: List[str],
    finetuned_predictions: List[str],
    references: List[str],
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
    smooth_method: str = "exp",
) -> Dict[str, Any]:
    """Computes paired bootstrap confidence interval for Corpus SacreBLEU differences.

    Resamples indices with replacement and evaluates corpus BLEU on identical resamples.
    """
    n = len(references)
    if len(baseline_predictions) != n or len(finetuned_predictions) != n:
        raise ValueError("All predictions and references must have identical length.")

    base_corpus = sacrebleu.corpus_bleu(baseline_predictions, [references], smooth_method=smooth_method).score
    ft_corpus = sacrebleu.corpus_bleu(finetuned_predictions, [references], smooth_method=smooth_method).score
    mean_diff = float(ft_corpus - base_corpus)

    rng = np.random.default_rng(seed)
    bootstrap_diffs = np.empty(n_bootstrap, dtype=float)

    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        res_refs = [references[k] for k in idx]
        res_base = [baseline_predictions[k] for k in idx]
        res_ft = [finetuned_predictions[k] for k in idx]

        b_score = sacrebleu.corpus_bleu(res_base, [res_refs], smooth_method=smooth_method).score
        f_score = sacrebleu.corpus_bleu(res_ft, [res_refs], smooth_method=smooth_method).score
        bootstrap_diffs[i] = f_score - b_score

    ci_lower = float(np.percentile(bootstrap_diffs, 100.0 * (alpha / 2.0)))
    ci_upper = float(np.percentile(bootstrap_diffs, 100.0 * (1.0 - alpha / 2.0)))

    if mean_diff > 0:
        p_val = 2.0 * float(np.mean(bootstrap_diffs <= 0.0))
    elif mean_diff < 0:
        p_val = 2.0 * float(np.mean(bootstrap_diffs >= 0.0))
    else:
        p_val = 1.0

    return {
        "baseline_corpus_bleu": round(float(base_corpus), 2),
        "finetuned_corpus_bleu": round(float(ft_corpus), 2),
        "bleu_difference": round(mean_diff, 2),
        "ci_lower": round(ci_lower, 2),
        "ci_upper": round(ci_upper, 2),
        "p_value_estimate": round(min(1.0, max(0.0, p_val)), 4),
        "statistically_conclusive": not (ci_lower <= 0.0 <= ci_upper),
        "n_bootstrap": n_bootstrap,
        "seed": seed,
    }
