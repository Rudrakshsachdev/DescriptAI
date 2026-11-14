"""Baseline model evaluator for Descra.

Orchestrates validation dataset evaluation, compares decoding strategies,
and aggregates BLEU, ROUGE, and factual adherence metrics.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

from descra.evaluation.factual_checks import verify_factual_adherence
from descra.evaluation.metrics import compute_generation_metrics
from descra.models.baseline import BaselineGenerator
from descra.utils.logging import get_logger

logger = get_logger("descra.evaluation.evaluator")


class BaselineEvaluator:
    """Evaluates baseline model generation on validation split data."""

    def __init__(self, generator: Optional[BaselineGenerator] = None, model_name: str = "google/flan-t5-small"):
        if generator is None:
            self.generator = BaselineGenerator(model_name=model_name)
        else:
            self.generator = generator

    def evaluate_split(
        self,
        jsonl_path: str = "data/processed/validation.jsonl",
        strategies: Optional[List[str]] = None,
        max_samples: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Evaluates model across specified decoding strategies on a dataset file.

        Args:
            jsonl_path: Path to evaluation dataset.
            strategies: List of strategies to test (default: ['greedy', 'beam_search']).
            max_samples: Optional cap on evaluated records (e.g., for fast testing).

        Returns:
            Dictionary containing metrics, factual adherence, and sample comparisons.
        """
        if strategies is None:
            strategies = ["greedy", "beam_search"]

        logger.info(f"Loading evaluation dataset from: {jsonl_path}")
        records = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))

        if max_samples and max_samples < len(records):
            logger.info(f"Subsampling evaluation dataset to {max_samples} records.")
            records = records[:max_samples]

        total_records = len(records)
        logger.info(f"Evaluating {total_records} records across strategies: {strategies}")

        references = [r.get("reference_description") or r.get("description", "") for r in records]

        strategy_results: Dict[str, Any] = {}
        all_generation_examples: List[Dict[str, Any]] = []

        for strat in strategies:
            logger.info(f"Executing inference for decoding strategy: '{strat}'...")
            predictions: List[str] = []
            latencies: List[float] = []
            factual_scores: List[float] = []
            unsupported_counts: List[int] = []
            preservation_ratios: List[float] = []
            repetitive_count = 0

            strat_examples: List[Dict[str, Any]] = []

            for i, r in enumerate(records):
                p_name = r.get("product_name") or r.get("title", "")
                cat = r.get("category", "")
                feats = r.get("features", [])
                specs = r.get("specifications", {})
                ref = references[i]

                gen_res = self.generator.generate(
                    product_name=p_name,
                    category=cat,
                    features=feats,
                    specifications=specs,
                    strategy=strat,
                )

                pred = gen_res["description"]
                prompt = gen_res["prompt"]
                meta = gen_res["metadata"]

                predictions.append(pred)
                latencies.append(meta["latency_ms"])

                # Factual verification
                fact_res = verify_factual_adherence(
                    input_text=prompt,
                    generated_text=pred,
                    reference_text=ref,
                )

                factual_scores.append(fact_res["factual_consistency_score"])
                unsupported_counts.append(fact_res["unsupported_number_count"])
                preservation_ratios.append(fact_res["input_token_preservation_ratio"])
                if fact_res["is_repetitive"]:
                    repetitive_count += 1

                # Save first 15 representative examples
                if i < 15:
                    strat_examples.append({
                        "product_id": r.get("product_id", f"sample_{i}"),
                        "product_name": p_name,
                        "strategy": strat,
                        "generated_description": pred,
                        "reference_description": ref,
                        "factual_score": fact_res["factual_consistency_score"],
                        "unsupported_numbers": fact_res["unsupported_numbers"],
                        "preservation_ratio": fact_res["input_token_preservation_ratio"],
                        "latency_ms": meta["latency_ms"],
                        "input_prompt": prompt,
                    })

            # Compute BLEU, ROUGE
            logger.info(f"Computing text generation metrics for '{strat}'...")
            text_metrics = compute_generation_metrics(
                predictions=predictions,
                references=references,
                device=str(self.generator.device),
            )

            strategy_results[strat] = {
                "sample_count": total_records,
                "metrics": text_metrics,
                "factual_adherence": {
                    "mean_consistency_score": round(float(np.mean(factual_scores)), 3),
                    "median_consistency_score": round(float(np.median(factual_scores)), 3),
                    "mean_input_token_preservation": round(float(np.mean(preservation_ratios)), 3),
                    "mean_unsupported_numbers_per_sample": round(float(np.mean(unsupported_counts)), 3),
                    "total_samples_with_unsupported_numbers": int(sum(1 for c in unsupported_counts if c > 0)),
                    "repetitive_outputs_count": repetitive_count,
                    "repetitive_outputs_pct": round(float(repetitive_count / total_records * 100), 2),
                },
                "latency_benchmark": {
                    "mean_latency_ms": round(float(np.mean(latencies)), 2),
                    "median_latency_ms": round(float(np.median(latencies)), 2),
                    "p90_latency_ms": round(float(np.percentile(latencies, 90)), 2),
                    "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2),
                    "min_latency_ms": round(float(np.min(latencies)), 2),
                    "max_latency_ms": round(float(np.max(latencies)), 2),
                },
            }

            all_generation_examples.extend(strat_examples)

        return {
            "model_name": self.generator.model_name,
            "device": str(self.generator.device),
            "strategies_evaluated": strategy_results,
            "examples": all_generation_examples,
        }

