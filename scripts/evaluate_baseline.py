"""Evaluation script for baseline FLAN-T5 model on validation dataset."""

import argparse
import json
import os
from pathlib import Path
import sys

# Ensure offline mode and paths
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from descra.evaluation.evaluator import BaselineEvaluator
from descra.models.baseline import BaselineGenerator
from descra.utils.logging import get_logger

logger = get_logger("descra.scripts.evaluate_baseline")


def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline FLAN-T5 model on validation split.")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/processed/validation.jsonl",
        help="Path to validation dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/baseline",
        help="Directory to save evaluation reports and metrics.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100,
        help="Number of validation records to evaluate (default: 100).",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["greedy", "beam_search"],
        help="Decoding strategies to evaluate.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Compute device ('auto', 'cuda', 'cpu').",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("DESCRA — BASELINE MODEL VALIDATION EVALUATION")
    print("=" * 60)
    print(f"Dataset:         {args.data_path}")
    print(f"Sample Size:     {args.sample_size}")
    print(f"Strategies:      {args.strategies}")
    print(f"Device:          {args.device}")
    print("=" * 60)

    # Initialize generator
    generator = BaselineGenerator(
        model_name="google/flan-t5-small",
        device=args.device,
        use_fp16=True,
    )

    evaluator = BaselineEvaluator(generator=generator)
    eval_results = evaluator.evaluate_split(
        jsonl_path=args.data_path,
        strategies=args.strategies,
        max_samples=args.sample_size,
    )

    # Export metrics JSON
    metrics_path = out_dir / "baseline_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(eval_results["strategies_evaluated"], f, indent=2)
    print(f"\nSaved metrics summary to: {metrics_path}")

    # Export generation examples
    examples_path = out_dir / "generation_examples.json"
    with open(examples_path, "w", encoding="utf-8") as f:
        json.dump(eval_results["examples"], f, indent=2)
    print(f"Saved generation examples to: {examples_path}")

    # Display comparison table
    print("\n" + "=" * 60)
    print("BASELINE EVALUATION RESULTS (Greedy vs. Beam Search)")
    print("=" * 60)

    for strat, data in eval_results["strategies_evaluated"].items():
        m = data["metrics"]
        b = m["bleu"]
        r = m["rouge"]
        f = data["factual_adherence"]
        l = data["latency_benchmark"]

        print(f"\n--- Strategy: {strat.upper()} ---")
        print(f"SacreBLEU:               {b['corpus_bleu']:.2f} (BLEU-1: {b['bleu_1']:.2f}, BLEU-4: {b['bleu_4']:.2f})")
        print(f"ROUGE-1 F1:              {r['rouge1_f1']:.4f} (Recall: {r['rouge1_recall']:.4f})")
        print(f"ROUGE-2 F1:              {r['rouge2_f1']:.4f} (Recall: {r['rouge2_recall']:.4f})")
        print(f"ROUGE-L F1:              {r['rougeL_f1']:.4f} (Recall: {r['rougeL_recall']:.4f})")
        print(f"Factual Score (Mean):    {f['mean_consistency_score']:.3f}")
        print(f"Input Token Preserved:   {f['mean_input_token_preservation']:.1%}")
        print(f"Samples w/ Unsupp. Nums: {f['total_samples_with_unsupported_numbers']} / {data['sample_count']}")
        print(f"Repetitive Outputs:      {f['repetitive_outputs_count']} ({f['repetitive_outputs_pct']}%)")
        print(f"Latency (Mean / P95):    {l['mean_latency_ms']:.2f} ms / {l['p95_latency_ms']:.2f} ms")

    print("\n" + "=" * 60)
    print("BASELINE EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

