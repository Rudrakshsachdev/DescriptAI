"""Publication-quality figure generation for Descra Step 9 test evaluation."""

from pathlib import Path
from typing import Any, Dict, List
import matplotlib.pyplot as plt
import numpy as np

# Set clean scientific plotting style
plt.rcParams.update({
    "font.sans-serif": "Arial",
    "font.family": "sans-serif",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
    "grid.color": "#e0e0e0",
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
})


def generate_evaluation_figures(
    results_summary: Dict[str, Any],
    statistical_analysis: Dict[str, Any],
    failure_data: Dict[str, Any],
    all_predictions: List[Dict[str, Any]],
    output_dir: str = "reports/evaluation/figures",
) -> List[str]:
    """Generates all 10 required academic figures for the final evaluation report."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    generated_figures: List[str] = []

    strategies = ["greedy", "beam_search_2", "beam_search_4"]
    strat_labels = ["Greedy", "Beam Search (2)", "Beam Search (4)"]

    base_rouge1 = [results_summary["baseline"][s]["metrics"]["rouge"]["rouge1_f1"] for s in strategies]
    ft_rouge1 = [results_summary["finetuned"][s]["metrics"]["rouge"]["rouge1_f1"] for s in strategies]

    base_rouge2 = [results_summary["baseline"][s]["metrics"]["rouge"]["rouge2_f1"] for s in strategies]
    ft_rouge2 = [results_summary["finetuned"][s]["metrics"]["rouge"]["rouge2_f1"] for s in strategies]

    base_rougeL = [results_summary["baseline"][s]["metrics"]["rouge"]["rougeL_f1"] for s in strategies]
    ft_rougeL = [results_summary["finetuned"][s]["metrics"]["rouge"]["rougeL_f1"] for s in strategies]

    base_bleu = [results_summary["baseline"][s]["metrics"]["bleu"]["corpus_bleu"] for s in strategies]
    ft_bleu = [results_summary["finetuned"][s]["metrics"]["bleu"]["corpus_bleu"] for s in strategies]

    base_fact = [results_summary["baseline"][s]["factual_adherence"]["mean_consistency_score"] for s in strategies]
    ft_fact = [results_summary["finetuned"][s]["factual_adherence"]["mean_consistency_score"] for s in strategies]

    base_lat = [results_summary["baseline"][s]["performance"]["mean_latency_ms"] for s in strategies]
    ft_lat = [results_summary["finetuned"][s]["performance"]["mean_latency_ms"] for s in strategies]

    x = np.arange(len(strategies))
    width = 0.35

    # 1. ROUGE-1 Comparison
    fig, ax = plt.subplots(figsize=(7, 4.5))
    rects1 = ax.bar(x - width/2, base_rouge1, width, label="Pretrained Baseline", color="#90caf9", edgecolor="#1565c0")
    rects2 = ax.bar(x + width/2, ft_rouge1, width, label="Fine-Tuned (SFT)", color="#1976d2", edgecolor="#0d47a1")
    ax.set_ylabel("ROUGE-1 F1 Score", fontsize=11, fontweight="bold")
    ax.set_title("Held-Out Test Set: ROUGE-1 F1 Score by Decoding Strategy (N=480)", fontsize=12, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(strat_labels, fontsize=10)
    ax.set_ylim(0, max(max(ft_rouge1), max(base_rouge1)) * 1.2)
    ax.grid(axis="y", alpha=0.7)
    ax.legend(frameon=True)
    ax.bar_label(rects1, padding=3, fmt="%.4f", fontsize=9)
    ax.bar_label(rects2, padding=3, fmt="%.4f", fontsize=9, fontweight="bold")
    fig.tight_layout()
    p1 = out_path / "01_rouge1_comparison.png"
    fig.savefig(p1)
    plt.close(fig)
    generated_figures.append(str(p1))

    # 2. ROUGE-2 Comparison
    fig, ax = plt.subplots(figsize=(7, 4.5))
    rects1 = ax.bar(x - width/2, base_rouge2, width, label="Pretrained Baseline", color="#a5d6a7", edgecolor="#2e7d32")
    rects2 = ax.bar(x + width/2, ft_rouge2, width, label="Fine-Tuned (SFT)", color="#388e3c", edgecolor="#1b5e20")
    ax.set_ylabel("ROUGE-2 F1 Score", fontsize=11, fontweight="bold")
    ax.set_title("Held-Out Test Set: ROUGE-2 F1 Score by Decoding Strategy (N=480)", fontsize=12, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(strat_labels, fontsize=10)
    ax.set_ylim(0, max(max(ft_rouge2), max(base_rouge2)) * 1.25)
    ax.grid(axis="y", alpha=0.7)
    ax.legend(frameon=True)
    ax.bar_label(rects1, padding=3, fmt="%.4f", fontsize=9)
    ax.bar_label(rects2, padding=3, fmt="%.4f", fontsize=9, fontweight="bold")
    fig.tight_layout()
    p2 = out_path / "02_rouge2_comparison.png"
    fig.savefig(p2)
    plt.close(fig)
    generated_figures.append(str(p2))

    # 3. ROUGE-L Comparison
    fig, ax = plt.subplots(figsize=(7, 4.5))
    rects1 = ax.bar(x - width/2, base_rougeL, width, label="Pretrained Baseline", color="#ce93d8", edgecolor="#6a1b9a")
    rects2 = ax.bar(x + width/2, ft_rougeL, width, label="Fine-Tuned (SFT)", color="#7b1fa2", edgecolor="#4a148c")
    ax.set_ylabel("ROUGE-L F1 Score", fontsize=11, fontweight="bold")
    ax.set_title("Held-Out Test Set: ROUGE-L F1 Score by Decoding Strategy (N=480)", fontsize=12, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(strat_labels, fontsize=10)
    ax.set_ylim(0, max(max(ft_rougeL), max(base_rougeL)) * 1.2)
    ax.grid(axis="y", alpha=0.7)
    ax.legend(frameon=True)
    ax.bar_label(rects1, padding=3, fmt="%.4f", fontsize=9)
    ax.bar_label(rects2, padding=3, fmt="%.4f", fontsize=9, fontweight="bold")
    fig.tight_layout()
    p3 = out_path / "03_rougeL_comparison.png"
    fig.savefig(p3)
    plt.close(fig)
    generated_figures.append(str(p3))

    # 4. Corpus SacreBLEU Comparison
    fig, ax = plt.subplots(figsize=(7, 4.5))
    rects1 = ax.bar(x - width/2, base_bleu, width, label="Pretrained Baseline", color="#ffcc80", edgecolor="#e65100")
    rects2 = ax.bar(x + width/2, ft_bleu, width, label="Fine-Tuned (SFT)", color="#ef6c00", edgecolor="#b23c00")
    ax.set_ylabel("SacreBLEU Score", fontsize=11, fontweight="bold")
    ax.set_title("Held-Out Test Set: Corpus SacreBLEU by Decoding Strategy (N=480)", fontsize=12, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(strat_labels, fontsize=10)
    ax.set_ylim(0, max(max(ft_bleu), max(base_bleu)) * 1.25)
    ax.grid(axis="y", alpha=0.7)
    ax.legend(frameon=True)
    ax.bar_label(rects1, padding=3, fmt="%.2f", fontsize=9)
    ax.bar_label(rects2, padding=3, fmt="%.2f", fontsize=9, fontweight="bold")
    fig.tight_layout()
    p4 = out_path / "04_corpus_bleu_comparison.png"
    fig.savefig(p4)
    plt.close(fig)
    generated_figures.append(str(p4))

    # 5. Heuristic Factual Adherence
    fig, ax = plt.subplots(figsize=(7, 4.5))
    rects1 = ax.bar(x - width/2, base_fact, width, label="Pretrained Baseline", color="#b0bec5", edgecolor="#37474f")
    rects2 = ax.bar(x + width/2, ft_fact, width, label="Fine-Tuned (SFT)", color="#455a64", edgecolor="#212121")
    ax.set_ylabel("Heuristic Adherence Index (0.0 to 1.0)", fontsize=11, fontweight="bold")
    ax.set_title("Held-Out Test Set: Heuristic Factual & Numerical Adherence Index", fontsize=12, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(strat_labels, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", alpha=0.7)
    ax.legend(frameon=True)
    ax.bar_label(rects1, padding=3, fmt="%.3f", fontsize=9)
    ax.bar_label(rects2, padding=3, fmt="%.3f", fontsize=9, fontweight="bold")
    fig.tight_layout()
    p5 = out_path / "05_heuristic_factual_adherence.png"
    fig.savefig(p5)
    plt.close(fig)
    generated_figures.append(str(p5))

    # 6. Latency Comparison
    fig, ax = plt.subplots(figsize=(7, 4.5))
    rects1 = ax.bar(x - width/2, base_lat, width, label="Pretrained Baseline", color="#ef9a9a", edgecolor="#c62828")
    rects2 = ax.bar(x + width/2, ft_lat, width, label="Fine-Tuned (SFT)", color="#c62828", edgecolor="#b71c1c")
    ax.set_ylabel("Mean Latency (Milliseconds)", fontsize=11, fontweight="bold")
    ax.set_title("Held-Out Test Set: Mean Inference Latency per Record (RTX 4050)", fontsize=12, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(strat_labels, fontsize=10)
    ax.set_ylim(0, max(max(ft_lat), max(base_lat)) * 1.2)
    ax.grid(axis="y", alpha=0.7)
    ax.legend(frameon=True)
    ax.bar_label(rects1, padding=3, fmt="%.1f ms", fontsize=9)
    ax.bar_label(rects2, padding=3, fmt="%.1f ms", fontsize=9, fontweight="bold")
    fig.tight_layout()
    p6 = out_path / "06_latency_comparison.png"
    fig.savefig(p6)
    plt.close(fig)
    generated_figures.append(str(p6))

    # 7. Output Length Distribution (Histogram/KDE)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    base_lens = [p["output_tokens"] for p in all_predictions if p["model_type"] == "baseline" and p["strategy"] == "beam_search_4"]
    ft_lens = [p["output_tokens"] for p in all_predictions if p["model_type"] == "finetuned" and p["strategy"] == "beam_search_4"]
    ax.hist(base_lens, bins=25, alpha=0.6, label="Pretrained Baseline (Beam-4)", color="#1976d2", edgecolor="white")
    ax.hist(ft_lens, bins=25, alpha=0.6, label="Fine-Tuned SFT (Beam-4)", color="#388e3c", edgecolor="white")
    ax.set_xlabel("Generated Token Count", fontsize=11, fontweight="bold")
    ax.set_ylabel("Frequency (Test Examples)", fontsize=11, fontweight="bold")
    ax.set_title("Distribution of Output Sequence Lengths (N=480 Test Records)", fontsize=12, pad=12)
    ax.grid(True, alpha=0.7)
    ax.legend(frameon=True)
    fig.tight_layout()
    p7 = out_path / "07_output_length_distribution.png"
    fig.savefig(p7)
    plt.close(fig)
    generated_figures.append(str(p7))

    # 8. Failure Category Distribution
    fig, ax = plt.subplots(figsize=(8, 4.5))
    cats = list(failure_data.get("baseline", {}).get("category_counts", {}).keys())
    if not cats:
        cats = ["EMPTY_OUTPUT", "BRAND_OR_EXTREMELY_SHORT", "PROMPT_ECHO", "REPETITIVE_OUTPUT", "UNSUPPORTED_NUMBERS", "GENERIC_FILLER"]
    b_counts = [failure_data.get("baseline", {}).get("category_counts", {}).get(c, 0) for c in cats]
    f_counts = [failure_data.get("finetuned", {}).get("category_counts", {}).get(c, 0) for c in cats]
    y = np.arange(len(cats))
    h = 0.35
    ax.barh(y - h/2, b_counts, h, label="Pretrained Baseline", color="#ef5350")
    ax.barh(y + h/2, f_counts, h, label="Fine-Tuned (SFT)", color="#2e7d32")
    ax.set_yticks(y)
    ax.set_yticklabels([c.replace("_", " ").title() for c in cats], fontsize=10)
    ax.set_xlabel("Failure Frequency (Across 1,440 Generations per Model)", fontsize=11, fontweight="bold")
    ax.set_title("Comparison of Model Failure Mode Frequencies on Test Set", fontsize=12, pad=12)
    ax.grid(axis="x", alpha=0.7)
    ax.legend(frameon=True)
    fig.tight_layout()
    p8 = out_path / "08_failure_category_distribution.png"
    fig.savefig(p8)
    plt.close(fig)
    generated_figures.append(str(p8))

    # 9. Per-Example Improvement Distribution (ROUGE-1 Gain Histogram)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    r1_gains = []
    # match baseline and finetuned for beam_search_4
    b_dict = {p["product_id"]: p["sentence_rouge1_f1"] for p in all_predictions if p["model_type"] == "baseline" and p["strategy"] == "beam_search_4"}
    f_dict = {p["product_id"]: p["sentence_rouge1_f1"] for p in all_predictions if p["model_type"] == "finetuned" and p["strategy"] == "beam_search_4"}
    for pid, b_val in b_dict.items():
        if pid in f_dict:
            r1_gains.append(f_dict[pid] - b_val)

    ax.hist(r1_gains, bins=30, color="#5c6bc0", edgecolor="#283593", alpha=0.75)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.2, label="Zero Delta Line")
    ax.axvline(np.mean(r1_gains), color="green", linestyle="-", linewidth=1.5, label=f"Mean Gain ({np.mean(r1_gains):+.4f})")
    ax.set_xlabel("Per-Example ROUGE-1 F1 Gain (Fine-Tuned - Baseline)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Frequency", fontsize=11, fontweight="bold")
    ax.set_title("Distribution of Per-Example ROUGE-1 F1 Gains (Beam-4, N=480)", fontsize=12, pad=12)
    ax.grid(True, alpha=0.7)
    ax.legend(frameon=True)
    fig.tight_layout()
    p9 = out_path / "09_per_example_improvement_distribution.png"
    fig.savefig(p9)
    plt.close(fig)
    generated_figures.append(str(p9))

    # 10. Paired Bootstrap Confidence Intervals Plot
    fig, ax = plt.subplots(figsize=(8, 4.5))
    metrics_ci = ["rouge1_f1", "rouge2_f1", "rougeL_f1", "factual_score"]
    metric_names = ["ROUGE-1 F1", "ROUGE-2 F1", "ROUGE-L F1", "Heuristic Adherence"]
    means = [statistical_analysis[m]["mean_difference"] for m in metrics_ci]
    err_low = [means[i] - statistical_analysis[m]["ci_lower"] for i, m in enumerate(metrics_ci)]
    err_high = [statistical_analysis[m]["ci_upper"] - means[i] for i, m in enumerate(metrics_ci)]

    y_pos = np.arange(len(metrics_ci))
    ax.errorbar(means, y_pos, xerr=[err_low, err_high], fmt="o", color="#1565c0", ecolor="#0d47a1", elinewidth=2, capsize=5, markersize=7)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.2, label="Zero Effect (Null Hypothesis)")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(metric_names, fontsize=11)
    ax.set_xlabel("Mean Paired Difference (Fine-Tuned - Baseline) with 95% Bootstrap CI", fontsize=11, fontweight="bold")
    ax.set_title("Paired Bootstrap 95% Confidence Intervals (1,000 Resamples, N=480)", fontsize=12, pad=12)
    ax.grid(True, alpha=0.7)
    ax.legend(frameon=True)
    fig.tight_layout()
    p10 = out_path / "10_paired_bootstrap_confidence_intervals.png"
    fig.savefig(p10)
    plt.close(fig)
    generated_figures.append(str(p10))

    return generated_figures
