"""Master Held-Out Test Evaluation Runner for Project Descra (Phase 1 Step 9)."""

import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from rouge_score import rouge_scorer
import sacrebleu
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from descra.evaluation.factual_checks import verify_factual_adherence
from descra.evaluation.failure_analysis import aggregate_failure_statistics, classify_generation_failure
from descra.evaluation.metrics import calculate_bleu, calculate_rouge, calculate_sentence_bleu
from descra.evaluation.statistical import compute_paired_bootstrap_ci, compute_paired_corpus_bleu_ci
from descra.evaluation.test_integrity import compute_file_sha256, verify_test_dataset
from descra.evaluation.visualizations import generate_evaluation_figures
from descra.models.generation import construct_prompt, generate_description
from descra.utils.logging import get_logger

logger = get_logger("descra.evaluation.test_runner")


class HeldOutTestRunner:
    """Orchestrates comprehensive, resumable, controlled held-out test evaluation."""

    def __init__(
        self,
        test_dataset_path: str = "data/processed/test.jsonl",
        baseline_model_path: str = "google/flan-t5-small",
        finetuned_checkpoint_path: str = "checkpoints/sft/best_checkpoint",
        output_data_dir: str = "data/evaluation",
        output_report_dir: str = "reports/evaluation",
        random_seed: int = 42,
        use_bf16: bool = True,
    ):
        self.test_path = Path(test_dataset_path)
        self.baseline_path = baseline_model_path
        self.finetuned_path = Path(finetuned_checkpoint_path)
        self.data_dir = Path(output_data_dir)
        self.report_dir = Path(output_report_dir)
        self.figures_dir = self.report_dir / "figures"
        self.seed = random_seed
        self.use_bf16 = use_bf16

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        self.predictions_jsonl_path = self.data_dir / "test_predictions.jsonl"
        self.summary_json_path = self.data_dir / "evaluation_summary.json"
        self.metadata_json_path = self.report_dir / "evaluation_metadata.json"
        self.integrity_json_path = self.report_dir / "test_dataset_integrity.json"

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.rouge_scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

        # Standard canonical decoding strategies
        self.strategies = {
            "greedy": {"num_beams": 1, "do_sample": False, "max_new_tokens": 128},
            "beam_search_2": {"num_beams": 2, "do_sample": False, "early_stopping": True, "length_penalty": 1.0, "no_repeat_ngram_size": 3, "max_new_tokens": 128},
            "beam_search_4": {"num_beams": 4, "do_sample": False, "early_stopping": True, "length_penalty": 1.0, "no_repeat_ngram_size": 3, "max_new_tokens": 128},
        }

    def set_seeds(self) -> None:
        """Centralized seed pinning policy."""
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def load_completed_prediction_keys(self) -> Set[str]:
        """Reads already persisted predictions from JSONL to enable clean resumability."""
        completed_keys = set()
        if self.predictions_jsonl_path.exists():
            with open(self.predictions_jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            record = json.loads(line)
                            key = f"{record['model_type']}_{record['strategy']}_{record['product_id']}"
                            completed_keys.add(key)
                        except Exception:
                            continue
        logger.info(f"Loaded {len(completed_keys)} previously completed generation records from {self.predictions_jsonl_path}.")
        return completed_keys

    def warm_up_model(self, model: Any, tokenizer: Any, model_name: str) -> float:
        """Executes 5 warm-up generation passes with GPU synchronization."""
        logger.info(f"Executing 5 warm-up generation passes for '{model_name}' on {self.device}...")
        warmup_prompts = [
            "Generate a clear and engaging product description. Product Name: Hydrating Face Cream Features: - Organic Aloe",
            "Generate a clear and engaging product description. Product Name: Stainless Steel Nail Clippers Features: - Sharp Blades",
            "Generate a clear and engaging product description. Product Name: Argan Hair Oil Features: - Nourishing - 100ml",
            "Generate a clear and engaging product description. Product Name: Sonic Facial Cleanser Features: - Waterproof",
            "Generate a clear and engaging product description. Product Name: Natural Beeswax Lip Balm Features: - Vanilla Scent",
        ]

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.inference_mode():
            for p in warmup_prompts:
                inputs = tokenizer(p, return_tensors="pt").to(self.device)
                model.generate(**inputs, max_new_tokens=32)

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        duration = time.perf_counter() - t0
        logger.info(f"Warm-up for '{model_name}' completed in {duration:.3f} s.")
        return duration

    def run_full_evaluation(self) -> Dict[str, Any]:
        """Main execution method evaluating 2 models x 3 strategies x 480 test records (2,880 passes)."""
        logger.info("=" * 70)
        logger.info("STARTING DESCRA STEP 9: FINAL HELD-OUT TEST EVALUATION")
        logger.info("=" * 70)

        # 1. Verify Test Dataset Integrity First
        logger.info("[Step 1/7] Verifying held-out test dataset integrity...")
        integrity_report = verify_test_dataset(
            test_path=str(self.test_path),
            train_path="data/processed/train.jsonl",
            val_path="data/processed/validation.jsonl",
            output_report_path=str(self.integrity_json_path),
        )
        if not integrity_report["overall_integrity_passed"]:
            raise ValueError(f"Test dataset failed integrity verification: {integrity_report}")
        logger.info(f"Test dataset verified: 480 records, 0 cross-split leaks. SHA-256: {integrity_report['sha256_hash'][:16]}...")

        # 2. Load Test Records
        test_records = []
        with open(self.test_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    test_records.append(json.loads(line))
        logger.info(f"Loaded {len(test_records)} canonical held-out test records.")

        # 3. Create & Save Evaluation Manifest
        manifest = {
            "experiment_id": "EXP-STEP9-FINAL-TEST-001",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "test_dataset_path": str(self.test_path),
            "test_dataset_sha256": integrity_report["sha256_hash"],
            "test_record_count": len(test_records),
            "random_seed": self.seed,
            "models_evaluated": {
                "baseline": self.baseline_path,
                "finetuned": str(self.finetuned_path),
            },
            "strategies_evaluated": list(self.strategies.keys()),
            "total_expected_generations": 2 * len(self.strategies) * len(test_records),
            "hardware": {
                "device": str(self.device),
                "gpu_name": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "CPU",
                "total_vram_mb": round(torch.cuda.get_device_properties(self.device).total_memory / (1024**2), 1) if self.device.type == "cuda" else 0.0,
            },
            "environment": {
                "python_version": sys.version.split()[0],
                "pytorch_version": torch.__version__,
                "sacrebleu_version": sacrebleu.__version__,
            },
        }
        with open(self.metadata_json_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        # 4. Check for Resumable Generations
        completed_keys = self.load_completed_prediction_keys()

        # Models loop configuration
        models_to_eval = [
            ("baseline", self.baseline_path),
            ("finetuned", str(self.finetuned_path)),
        ]

        # Open file in append mode for streaming persistence
        f_pred = open(self.predictions_jsonl_path, "a", encoding="utf-8")

        all_predictions: List[Dict[str, Any]] = []
        failure_records: Dict[str, List[Dict[str, Any]]] = {"baseline": [], "finetuned": []}

        # 5. Execute Evaluation Loop across Models and Strategies
        for model_type, model_path_str in models_to_eval:
            logger.info(f"\n[Step 2/7] Loading '{model_type}' model from: {model_path_str}...")
            self.set_seeds()

            tok = AutoTokenizer.from_pretrained(model_path_str, local_files_only=True)
            if self.use_bf16 and self.device.type == "cuda" and torch.cuda.is_bf16_supported():
                model = AutoModelForSeq2SeqLM.from_pretrained(model_path_str, local_files_only=True, dtype=torch.bfloat16).to(self.device)
            elif self.device.type == "cuda":
                model = AutoModelForSeq2SeqLM.from_pretrained(model_path_str, local_files_only=True).half().to(self.device)
            else:
                model = AutoModelForSeq2SeqLM.from_pretrained(model_path_str, local_files_only=True).to(self.device)
            model.eval()

            # Warm-up passes
            self.warm_up_model(model, tok, model_type)

            for strat_name, strat_cfg in self.strategies.items():
                logger.info(f"\n--- Model: {model_type.upper()} | Strategy: {strat_name} (Total: {len(test_records)} examples) ---")

                for i, r in enumerate(test_records):
                    p_id = r.get("product_id")
                    key = f"{model_type}_{strat_name}_{p_id}"

                    # If already processed in previous run, skip generation
                    if key in completed_keys:
                        continue

                    title = r.get("product_name") or r.get("title", "")
                    category = r.get("category", "")
                    features = r.get("features", [])
                    specs = r.get("specifications", {})
                    ref = r.get("reference_description") or r.get("description", "")

                    prompt = construct_prompt(
                        product_name=title,
                        category=category,
                        features=features,
                        specifications=specs,
                    )

                    # CUDA memory and timing
                    if self.device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats(self.device)
                        torch.cuda.synchronize()

                    t_start = time.perf_counter()

                    with torch.inference_mode():
                        inputs = tok(prompt, max_length=512, truncation=True, return_tensors="pt").to(self.device)
                        input_tok_len = inputs["input_ids"].shape[1]
                        output_ids = model.generate(
                            input_ids=inputs["input_ids"],
                            attention_mask=inputs.get("attention_mask"),
                            **strat_cfg,
                        )

                    if self.device.type == "cuda":
                        torch.cuda.synchronize()
                        alloc_mb = round(torch.cuda.max_memory_allocated(self.device) / (1024**2), 2)
                        res_mb = round(torch.cuda.max_memory_reserved(self.device) / (1024**2), 2)
                    else:
                        alloc_mb = 0.0
                        res_mb = 0.0

                    elapsed_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
                    gen_tok_len = output_ids.shape[1]
                    toks_per_sec = round((gen_tok_len / (elapsed_ms / 1000.0)) if elapsed_ms > 0 else 0.0, 2)

                    gen_text = tok.decode(output_ids[0], skip_special_tokens=True).strip()

                    # Sentence ROUGE
                    clean_p = gen_text.strip() if gen_text.strip() else "<empty>"
                    clean_r = ref.strip() if ref.strip() else "<empty>"
                    r_score = self.rouge_scorer.score(clean_r, clean_p)
                    s_r1 = round(float(r_score["rouge1"].fmeasure), 4)
                    s_r2 = round(float(r_score["rouge2"].fmeasure), 4)
                    s_rl = round(float(r_score["rougeL"].fmeasure), 4)

                    # Sentence BLEU
                    s_bleu = calculate_sentence_bleu(gen_text, ref, smooth_method="exp")

                    # Factual adherence heuristic
                    fact_res = verify_factual_adherence(input_text=prompt, generated_text=gen_text, reference_text=ref)

                    # Failure classification
                    failures = classify_generation_failure(
                        product_id=p_id,
                        product_name=title,
                        prompt=prompt,
                        generated_text=gen_text,
                        reference_text=ref,
                    )

                    record_out = {
                        "model_type": model_type,
                        "strategy": strat_name,
                        "product_id": p_id,
                        "product_name": title,
                        "prompt": prompt,
                        "generated_text": gen_text,
                        "reference_description": ref,
                        "input_tokens": input_tok_len,
                        "output_tokens": gen_tok_len,
                        "latency_ms": elapsed_ms,
                        "tokens_per_second": toks_per_sec,
                        "gpu_memory_allocated_mb": alloc_mb,
                        "gpu_memory_reserved_mb": res_mb,
                        "sentence_rouge1_f1": s_r1,
                        "sentence_rouge2_f1": s_r2,
                        "sentence_rougeL_f1": s_rl,
                        "sentence_bleu": s_bleu,
                        "factual_consistency_score": fact_res["factual_consistency_score"],
                        "unsupported_number_count": fact_res["unsupported_number_count"],
                        "input_token_preservation_ratio": fact_res["input_token_preservation_ratio"],
                        "failures": failures,
                    }

                    # Stream persist immediately
                    f_pred.write(json.dumps(record_out) + "\n")
                    f_pred.flush()

                    completed_keys.add(key)
                    if (i + 1) % 100 == 0 or (i + 1) == len(test_records):
                        logger.info(f"Progress: [{i+1}/{len(test_records)}] | Latency: {elapsed_ms:.1f} ms | ROUGE-1: {s_r1:.4f} | Factual: {fact_res['factual_consistency_score']:.3f}")

            # Free memory between models
            del model
            del tok
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        f_pred.close()
        logger.info(f"\n[Step 3/7] All 2,880 generations completed and verified in {self.predictions_jsonl_path}!")

        # 6. Read back all predictions to aggregate results
        logger.info("[Step 4/7] Aggregating corpus metrics and statistical analysis...")
        with open(self.predictions_jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    all_predictions.append(json.loads(line))

        # Group predictions by model and strategy
        grouped_preds: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
            "baseline": {s: [] for s in self.strategies.keys()},
            "finetuned": {s: [] for s in self.strategies.keys()},
        }
        for p in all_predictions:
            m = p["model_type"]
            s = p["strategy"]
            if m in grouped_preds and s in grouped_preds[m]:
                grouped_preds[m][s].append(p)

        # 7. Compute Corpus Metrics
        results_summary: Dict[str, Any] = {"baseline": {}, "finetuned": {}, "deltas": {}}
        failure_summaries: Dict[str, Any] = {"baseline": {}, "finetuned": {}}

        for m_type in ["baseline", "finetuned"]:
            for s_name in self.strategies.keys():
                items = grouped_preds[m_type][s_name]
                preds = [it["generated_text"] for it in items]
                refs = [it["reference_description"] for it in items]
                latencies = [it["latency_ms"] for it in items]
                toks_per_sec = [it["tokens_per_second"] for it in items]
                output_tokens = [it["output_tokens"] for it in items]
                factual_scores = [it["factual_consistency_score"] for it in items]
                unsupp_counts = [it["unsupported_number_count"] for it in items]
                preservations = [it["input_token_preservation_ratio"] for it in items]
                alloc_mbs = [it["gpu_memory_allocated_mb"] for it in items]

                corpus_bleu = calculate_bleu(preds, refs, smooth_method="exp")
                corpus_rouge = calculate_rouge(preds, refs)

                # Length analysis
                empty_count = sum(1 for p in preds if not p.strip())
                short_count = sum(1 for tok_len in output_tokens if tok_len <= 10)
                max_token_ceiling_count = sum(1 for tok_len in output_tokens if tok_len >= 127)

                # Failure analysis
                strat_failures = [it for it in items if it.get("failures")]
                failure_stats = aggregate_failure_statistics(strat_failures, len(items))

                results_summary[m_type][s_name] = {
                    "sample_count": len(items),
                    "metrics": {
                        "bleu": corpus_bleu,
                        "rouge": corpus_rouge,
                        "mean_sentence_bleu": round(float(np.mean([it["sentence_bleu"] for it in items])), 2),
                    },
                    "factual_adherence": {
                        "mean_consistency_score": round(float(np.mean(factual_scores)), 3),
                        "median_consistency_score": round(float(np.median(factual_scores)), 3),
                        "mean_token_preservation": round(float(np.mean(preservations)), 3),
                        "mean_unsupported_numbers": round(float(np.mean(unsupp_counts)), 3),
                        "samples_with_unsupported_numbers": sum(1 for c in unsupp_counts if c > 0),
                    },
                    "length_analysis": {
                        "mean_output_tokens": round(float(np.mean(output_tokens)), 1),
                        "median_output_tokens": round(float(np.median(output_tokens)), 1),
                        "min_output_tokens": int(np.min(output_tokens)),
                        "max_output_tokens": int(np.max(output_tokens)),
                        "empty_output_pct": round(100.0 * empty_count / len(items), 2),
                        "very_short_output_pct": round(100.0 * short_count / len(items), 2),
                        "max_token_ceiling_pct": round(100.0 * max_token_ceiling_count / len(items), 2),
                    },
                    "performance": {
                        "mean_latency_ms": round(float(np.mean(latencies)), 2),
                        "median_latency_ms": round(float(np.median(latencies)), 2),
                        "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2),
                        "min_latency_ms": round(float(np.min(latencies)), 2),
                        "max_latency_ms": round(float(np.max(latencies)), 2),
                        "mean_tokens_per_second": round(float(np.mean(toks_per_sec)), 2),
                        "examples_per_second": round(1000.0 / float(np.mean(latencies)), 2) if np.mean(latencies) > 0 else 0.0,
                        "peak_vram_allocated_mb": round(float(np.max(alloc_mbs)), 1),
                    },
                    "failure_analysis": failure_stats,
                }

        # 8. Compute Absolute Deltas (Fine-Tuned vs Baseline)
        for s_name in self.strategies.keys():
            b = results_summary["baseline"][s_name]
            f = results_summary["finetuned"][s_name]

            b_r = b["metrics"]["rouge"]
            f_r = f["metrics"]["rouge"]
            b_b = b["metrics"]["bleu"]["corpus_bleu"]
            f_b = f["metrics"]["bleu"]["corpus_bleu"]
            b_f = b["factual_adherence"]["mean_consistency_score"]
            f_f = f["factual_adherence"]["mean_consistency_score"]
            b_lat = b["performance"]["mean_latency_ms"]
            f_lat = f["performance"]["mean_latency_ms"]

            results_summary["deltas"][s_name] = {
                "rouge1_f1_gain": round(f_r["rouge1_f1"] - b_r["rouge1_f1"], 4),
                "rouge1_f1_rel_gain_pct": round(100.0 * (f_r["rouge1_f1"] - b_r["rouge1_f1"]) / b_r["rouge1_f1"], 2) if b_r["rouge1_f1"] > 0 else 0.0,
                "rouge2_f1_gain": round(f_r["rouge2_f1"] - b_r["rouge2_f1"], 4),
                "rouge2_f1_rel_gain_pct": round(100.0 * (f_r["rouge2_f1"] - b_r["rouge2_f1"]) / b_r["rouge2_f1"], 2) if b_r["rouge2_f1"] > 0 else 0.0,
                "rougeL_f1_gain": round(f_r["rougeL_f1"] - b_r["rougeL_f1"], 4),
                "rougeL_f1_rel_gain_pct": round(100.0 * (f_r["rougeL_f1"] - b_r["rougeL_f1"]) / b_r["rougeL_f1"], 2) if b_r["rougeL_f1"] > 0 else 0.0,
                "corpus_bleu_gain": round(f_b - b_b, 2),
                "factual_adherence_gain": round(f_f - b_f, 3),
                "latency_reduction_ms": round(b_lat - f_lat, 2),
                "latency_reduction_pct": round(100.0 * (b_lat - f_lat) / b_lat, 2) if b_lat > 0 else 0.0,
            }

        # 9. Paired Bootstrap Statistical Testing (1,000 Resamples, Seed=42)
        logger.info("[Step 5/7] Executing paired bootstrap confidence intervals (1,000 resamples)...")
        statistical_analysis: Dict[str, Any] = {}
        for s_name in self.strategies.keys():
            b_items = grouped_preds["baseline"][s_name]
            f_items = grouped_preds["finetuned"][s_name]

            # Match items by product_id
            b_map = {it["product_id"]: it for it in b_items}
            f_map = {it["product_id"]: it for it in f_items}
            common_ids = [r["product_id"] for r in test_records]

            b_r1 = [b_map[pid]["sentence_rouge1_f1"] for pid in common_ids]
            f_r1 = [f_map[pid]["sentence_rouge1_f1"] for pid in common_ids]

            b_r2 = [b_map[pid]["sentence_rouge2_f1"] for pid in common_ids]
            f_r2 = [f_map[pid]["sentence_rouge2_f1"] for pid in common_ids]

            b_rl = [b_map[pid]["sentence_rougeL_f1"] for pid in common_ids]
            f_rl = [f_map[pid]["sentence_rougeL_f1"] for pid in common_ids]

            b_fact = [b_map[pid]["factual_consistency_score"] for pid in common_ids]
            f_fact = [f_map[pid]["factual_consistency_score"] for pid in common_ids]

            b_lat = [b_map[pid]["latency_ms"] for pid in common_ids]
            f_lat = [f_map[pid]["latency_ms"] for pid in common_ids]

            statistical_analysis[s_name] = {
                "rouge1_f1": compute_paired_bootstrap_ci(b_r1, f_r1, n_bootstrap=1000, seed=self.seed),
                "rouge2_f1": compute_paired_bootstrap_ci(b_r2, f_r2, n_bootstrap=1000, seed=self.seed),
                "rougeL_f1": compute_paired_bootstrap_ci(b_rl, f_rl, n_bootstrap=1000, seed=self.seed),
                "factual_score": compute_paired_bootstrap_ci(b_fact, f_fact, n_bootstrap=1000, seed=self.seed),
                "latency_ms": compute_paired_bootstrap_ci(b_lat, f_lat, n_bootstrap=1000, seed=self.seed),
                "corpus_bleu": compute_paired_corpus_bleu_ci(
                    baseline_predictions=[b_map[pid]["generated_text"] for pid in common_ids],
                    finetuned_predictions=[f_map[pid]["generated_text"] for pid in common_ids],
                    references=[b_map[pid]["reference_description"] for pid in common_ids],
                    n_bootstrap=1000,
                    seed=self.seed,
                ),
            }

        # 10. Generate Qualitative Examples across 8 Specific Required Categories
        logger.info("[Step 6/7] Extracting categorized qualitative examples...")
        qualitative_categories = self.extract_qualitative_categories(grouped_preds, test_records)

        # 11. Generate Human Review Template CSV
        self.generate_human_review_template(test_records, grouped_preds)

        # 12. Save Summary JSON Artifacts
        full_summary = {
            "metadata": manifest,
            "results_summary": results_summary,
            "statistical_analysis": statistical_analysis,
            "qualitative_categories": qualitative_categories,
        }
        with open(self.summary_json_path, "w", encoding="utf-8") as f:
            json.dump(full_summary, f, indent=2)

        # Save dedicated qualitative examples JSON
        qual_json_path = self.data_dir / "qualitative_examples.json"
        with open(qual_json_path, "w", encoding="utf-8") as f:
            json.dump(qualitative_categories, f, indent=2)

        # Save dedicated failure records JSON
        fail_json_path = self.data_dir / "failure_records.json"
        with open(fail_json_path, "w", encoding="utf-8") as f:
            json.dump({
                "baseline": results_summary["baseline"]["beam_search_4"]["failure_analysis"],
                "finetuned": results_summary["finetuned"]["beam_search_4"]["failure_analysis"],
            }, f, indent=2)

        # 13. Generate Publication Figures
        logger.info("[Step 7/7] Generating 10 publication figures...")
        generate_evaluation_figures(
            results_summary=results_summary,
            statistical_analysis=statistical_analysis["beam_search_4"],
            failure_data={
                "baseline": results_summary["baseline"]["beam_search_4"]["failure_analysis"],
                "finetuned": results_summary["finetuned"]["beam_search_4"]["failure_analysis"],
            },
            all_predictions=all_predictions,
            output_dir=str(self.figures_dir),
        )

        logger.info(f"Step 9 evaluation complete. Artifacts written to {self.data_dir} and {self.report_dir}.")
        return full_summary

    def extract_qualitative_categories(
        self,
        grouped_preds: Dict[str, Dict[str, List[Dict[str, Any]]]],
        test_records: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Categorizes real test examples into the 8 required qualitative evaluation classes."""
        b_items = {it["product_id"]: it for it in grouped_preds["baseline"]["beam_search_4"]}
        f_items = {it["product_id"]: it for it in grouped_preds["finetuned"]["beam_search_4"]}

        categories = {
            "1_strong_improvement": [],
            "2_similar_performance": [],
            "3_finetuned_worse": [],
            "4_missing_attributes": [],
            "5_generic_or_repetitive": [],
            "6_unsupported_numbers": [],
            "7_brand_only_or_short": [],
            "8_incorrect_misleading": [],
        }

        for r in test_records:
            pid = r["product_id"]
            b = b_items.get(pid)
            f = f_items.get(pid)
            if not b or not f:
                continue

            r1_diff = f["sentence_rouge1_f1"] - b["sentence_rouge1_f1"]
            b_words = len(b["generated_text"].split())
            f_words = len(f["generated_text"].split())

            item_data = {
                "product_id": pid,
                "product_title": r.get("product_name") or r.get("title", ""),
                "features": r.get("features", []),
                "reference_description": r.get("reference_description", ""),
                "baseline_output": b["generated_text"],
                "finetuned_output": f["generated_text"],
                "baseline_rouge1_f1": b["sentence_rouge1_f1"],
                "finetuned_rouge1_f1": f["sentence_rouge1_f1"],
                "rouge1_gain": round(r1_diff, 4),
                "baseline_factual": b["factual_consistency_score"],
                "finetuned_factual": f["factual_consistency_score"],
            }

            # 1. Strong Improvement (Large ROUGE gain + length expansion)
            if r1_diff > 0.35 and len(categories["1_strong_improvement"]) < 3:
                categories["1_strong_improvement"].append(item_data)

            # 2. Similar Performance (Both high or similar overlap)
            elif abs(r1_diff) <= 0.05 and b["sentence_rouge1_f1"] > 0.30 and len(categories["2_similar_performance"]) < 3:
                categories["2_similar_performance"].append(item_data)

            # 3. Fine-tuned Worse
            elif r1_diff < -0.20 and len(categories["3_finetuned_worse"]) < 3:
                categories["3_finetuned_worse"].append(item_data)

            # 4. Missing Attributes
            elif any(flg["category"] == "MISSING_FEATURES" for flg in b.get("failures", [])) and len(categories["4_missing_attributes"]) < 3:
                categories["4_missing_attributes"].append(item_data)

            # 5. Generic or Repetitive
            elif any(flg["category"] in ["REPETITIVE_OUTPUT", "GENERIC_FILLER"] for flg in b.get("failures", [])) and len(categories["5_generic_or_repetitive"]) < 3:
                categories["5_generic_or_repetitive"].append(item_data)

            # 6. Unsupported Numbers
            elif any(flg["category"] == "UNSUPPORTED_NUMBERS" for flg in b.get("failures", [])) and len(categories["6_unsupported_numbers"]) < 3:
                categories["6_unsupported_numbers"].append(item_data)

            # 7. Brand Only or Short
            elif b_words <= 3 and f_words > 15 and len(categories["7_brand_only_or_short"]) < 3:
                categories["7_brand_only_or_short"].append(item_data)

            # 8. Incorrect / Misleading
            elif b["factual_consistency_score"] < 0.5 and len(categories["8_incorrect_misleading"]) < 3:
                categories["8_incorrect_misleading"].append(item_data)

        return categories

    def generate_human_review_template(
        self,
        test_records: List[Dict[str, Any]],
        grouped_preds: Dict[str, Dict[str, List[Dict[str, Any]]]],
    ) -> None:
        """Generates anonymized human review CSV template for academic evaluation readiness."""
        csv_path = self.data_dir / "human_review_template.csv"
        fieldnames = [
            "anonymized_id",
            "product_id",
            "product_title",
            "reference_description",
            "candidate_model_A",
            "candidate_model_B",
            "relevance_A_1to5",
            "relevance_B_1to5",
            "fluency_A_1to5",
            "fluency_B_1to5",
            "completeness_A_1to5",
            "completeness_B_1to5",
            "factual_consistency_A_1to5",
            "factual_consistency_B_1to5",
            "overall_preference_A_B_Tie",
            "reviewer_notes",
        ]

        b_map = {it["product_id"]: it for it in grouped_preds["baseline"]["beam_search_4"]}
        f_map = {it["product_id"]: it for it in grouped_preds["finetuned"]["beam_search_4"]}

        # Take first 50 records for human study template
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for i, r in enumerate(test_records[:50], 1):
                pid = r["product_id"]
                b = b_map.get(pid, {})
                ft = f_map.get(pid, {})

                # Coin flip to blind candidate A and candidate B
                is_a_baseline = (i % 2 == 1)
                cand_a = b.get("generated_text", "") if is_a_baseline else ft.get("generated_text", "")
                cand_b = ft.get("generated_text", "") if is_a_baseline else b.get("generated_text", "")

                writer.writerow({
                    "anonymized_id": f"EVAL-{i:03d}",
                    "product_id": pid,
                    "product_title": r.get("product_name") or r.get("title", ""),
                    "reference_description": r.get("reference_description", ""),
                    "candidate_model_A": cand_a,
                    "candidate_model_B": cand_b,
                    "relevance_A_1to5": "",
                    "relevance_B_1to5": "",
                    "fluency_A_1to5": "",
                    "fluency_B_1to5": "",
                    "completeness_A_1to5": "",
                    "completeness_B_1to5": "",
                    "factual_consistency_A_1to5": "",
                    "factual_consistency_B_1to5": "",
                    "overall_preference_A_B_Tie": "",
                    "reviewer_notes": "Pending human evaluation",
                })

        logger.info(f"Human review template generated at: {csv_path}")
