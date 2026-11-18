"""Master baseline experiment orchestration engine for Descra.

Executes reproducible zero-shot evaluations across multiple decoding strategies,
recording individual generation outputs, GPU telemetry, NLP overlap metrics,
heuristic factual verification, and failure modes.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import random
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
from rouge_score import rouge_scorer
import torch

from descra.evaluation.factual_checks import verify_factual_adherence
from descra.evaluation.metrics import (
    calculate_bleu,
    calculate_rouge,
    calculate_sentence_bleu,
)
from descra.experiments.analysis import ExperimentFailureAnalyzer, FailureRecord
from descra.experiments.config import DecodingStrategyConfig, ExperimentConfig
from descra.experiments.recorder import BaselineOutputRecord, ExperimentRecorder
from descra.models.baseline import BaselineGenerator
from descra.models.generation import construct_prompt, generate_description
from descra.models.loader import load_model_and_tokenizer
from descra.utils.logging import get_logger

logger = get_logger("descra.experiments.runner")


class BaselineExperimentRunner:
    """Orchestrates multi-strategy zero-shot generation experiments and metric evaluation."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.recorder = ExperimentRecorder(output_dir=config.output_dir)
        self.failure_analyzer = ExperimentFailureAnalyzer()

        # Initialize Model & Tokenizer
        logger.info(f"Initializing baseline model '{config.model_name}' on device '{config.device}' (FP16={config.use_fp16})...")
        self.model, self.tokenizer, self.device = load_model_and_tokenizer(
            model_name=config.model_name,
            device=config.device,
            use_fp16=config.use_fp16,
        )

        self.rouge_scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    def warm_up_gpu(self) -> float:
        """Executes kernel warm-up passes to eliminate JIT compilation overhead from benchmarks."""
        logger.info(f"Executing {self.config.num_warmup_iterations} GPU kernel warm-up passes...")
        dummy_prompt = "Warm up prompt for Transformer generation."
        dummy_input = self.tokenizer(dummy_prompt, return_tensors="pt").to(self.device)

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.inference_mode():
            for _ in range(self.config.num_warmup_iterations):
                self.model.generate(**dummy_input, max_new_tokens=16)

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        warmup_duration = time.perf_counter() - t0
        logger.info(f"GPU warm-up completed in {warmup_duration:.3f} s.")
        return warmup_duration

    def load_dataset(self) -> List[Dict[str, Any]]:
        """Loads and deterministically samples evaluation records."""
        path = Path(self.config.dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"Evaluation dataset not found: {path}")

        records: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))

        total_available = len(records)
        logger.info(f"Loaded {total_available} records from {path}.")

        if self.config.sample_size and self.config.sample_size < total_available:
            # Deterministic subsampling matching Step 6 slice
            logger.info(f"Selecting first {self.config.sample_size} records to maintain exact Step 6 alignment.")
            records = records[: self.config.sample_size]

        return records

    def run_all_strategies(self) -> Dict[str, Any]:
        """Executes full experimental matrix across all configured decoding strategies."""
        warmup_time = self.warm_up_gpu()
        records = self.load_dataset()
        num_samples = len(records)
        logger.info(f"Initiating evaluation for {len(self.config.strategies)} strategies on {num_samples} records.")

        # Clean/reset previous output files if new experiment run
        if self.recorder.outputs_path.exists():
            self.recorder.outputs_path.unlink()

        sample_ids = [r.get("product_id", f"sample_{i}") for i, r in enumerate(records)]
        references = [r.get("reference_description") or r.get("description", "") for r in records]

        strategy_results: Dict[str, Any] = {}
        all_failure_records: List[FailureRecord] = []
        samples_per_strategy: Dict[str, int] = {}

        gpu_info = {}
        if self.device.type == "cuda":
            props = torch.cuda.get_device_properties(self.device)
            gpu_info = {
                "gpu_name": torch.cuda.get_device_name(self.device),
                "total_vram_mb": round(props.total_memory / (1024**2), 1),
                "cuda_device_index": self.device.index,
            }

        for strat_idx, strat_cfg in enumerate(self.config.strategies, 1):
            strat_name = strat_cfg.name
            params = strat_cfg.parameters
            logger.info(f"\n[{strat_idx}/{len(self.config.strategies)}] Running strategy: '{strat_name}' ({strat_cfg.description})")

            # Seed reproducibility
            torch.manual_seed(self.config.random_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.config.random_seed)
            random.seed(self.config.random_seed)
            np.random.seed(self.config.random_seed)

            predictions: List[str] = []
            latencies_ms: List[float] = []
            toks_per_sec_list: List[float] = []
            factual_scores: List[float] = []
            unsupported_counts: List[int] = []
            preservation_ratios: List[float] = []
            r1_f1_list, r2_f1_list, rl_f1_list = [], [], []
            sentence_bleu_list: List[float] = []

            for i, r in enumerate(records):
                p_id = r.get("product_id", f"sample_{i}")
                p_name = r.get("product_name") or r.get("title", "")
                cat = r.get("category", "")
                feats = r.get("features", [])
                specs = r.get("specifications", {})
                ref = references[i]

                prompt = construct_prompt(
                    product_name=p_name,
                    category=cat,
                    features=feats,
                    specifications=specs,
                )

                # Reset peak memory before generation
                if self.device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(self.device)

                # Generation
                gen_text, meta = generate_description(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    prompt=prompt,
                    generation_config=params,
                    device=self.device,
                    max_input_length=self.config.max_input_length,
                )

                # Memory Telemetry
                if self.device.type == "cuda":
                    alloc_mb = round(torch.cuda.memory_allocated(self.device) / (1024**2), 2)
                    res_mb = round(torch.cuda.memory_reserved(self.device) / (1024**2), 2)
                else:
                    alloc_mb = 0.0
                    res_mb = 0.0

                # Sentence-level overlap metrics
                s_bleu = calculate_sentence_bleu(gen_text, ref, smooth_method="exp")
                sentence_bleu_list.append(s_bleu)

                # Sentence-level ROUGE
                clean_p = gen_text.strip() if gen_text.strip() else "<empty>"
                clean_r = ref.strip() if ref.strip() else "<empty>"
                r_scores = self.rouge_scorer.score(clean_r, clean_p)
                r1_f1 = round(float(r_scores["rouge1"].fmeasure), 4)
                r1_rec = round(float(r_scores["rouge1"].recall), 4)
                r1_prec = round(float(r_scores["rouge1"].precision), 4)
                r2_f1 = round(float(r_scores["rouge2"].fmeasure), 4)
                r2_rec = round(float(r_scores["rouge2"].recall), 4)
                r2_prec = round(float(r_scores["rouge2"].precision), 4)
                rl_f1 = round(float(r_scores["rougeL"].fmeasure), 4)
                rl_rec = round(float(r_scores["rougeL"].recall), 4)
                rl_prec = round(float(r_scores["rougeL"].precision), 4)

                r1_f1_list.append(r1_f1)
                r2_f1_list.append(r2_f1)
                rl_f1_list.append(rl_f1)

                # Factual verification
                fact_res = verify_factual_adherence(
                    input_text=prompt,
                    generated_text=gen_text,
                    reference_text=ref,
                )
                factual_scores.append(fact_res["factual_consistency_score"])
                unsupported_counts.append(fact_res["unsupported_number_count"])
                preservation_ratios.append(fact_res["input_token_preservation_ratio"])

                # Detailed failure classification
                item_failures = self.failure_analyzer.analyze_single_output(
                    example_id=f"{strat_name}_{i:03d}",
                    product_id=p_id,
                    category=cat,
                    strategy=strat_name,
                    input_prompt=prompt,
                    generated_output=gen_text,
                    product_name=p_name,
                    input_features=feats,
                    input_specifications=specs,
                )
                all_failure_records.extend(item_failures)
                failure_flags = [f.failure_type for f in item_failures]

                # Create and serialize structured output record
                output_record = BaselineOutputRecord(
                    experiment_id=self.config.experiment_id,
                    example_id=f"{strat_name}_{i:03d}",
                    product_id=p_id,
                    product_name=p_name,
                    product_category=cat,
                    input_features=feats,
                    input_specifications=specs,
                    reference_description=ref,
                    exact_prompt=prompt,
                    generation_strategy=strat_name,
                    generation_configuration=params,
                    random_seed=self.config.random_seed,
                    generated_description=gen_text,
                    input_token_count=meta["input_tokens"],
                    output_token_count=meta["output_tokens"],
                    generation_latency_ms=meta["latency_ms"],
                    tokens_per_second=meta["tokens_per_second"],
                    device=str(self.device),
                    gpu_memory_allocated_mb=alloc_mb,
                    gpu_memory_reserved_mb=res_mb,
                    rouge_1_f1=r1_f1,
                    rouge_1_recall=r1_rec,
                    rouge_1_precision=r1_prec,
                    rouge_2_f1=r2_f1,
                    rouge_2_recall=r2_rec,
                    rouge_2_precision=r2_prec,
                    rouge_l_f1=rl_f1,
                    rouge_l_recall=rl_rec,
                    rouge_l_precision=rl_prec,
                    sentence_bleu=s_bleu,
                    bertscore=None,
                    factual_consistency_score=fact_res["factual_consistency_score"],
                    failure_flags=failure_flags,
                    experiment_timestamp=datetime.now(timezone.utc).isoformat(),
                )
                self.recorder.append_output_record(output_record)

                predictions.append(gen_text)
                latencies_ms.append(meta["latency_ms"])
                toks_per_sec_list.append(meta["tokens_per_second"])

            # Compute corpus-level metrics for this strategy
            corpus_bleu_res = calculate_bleu(predictions, references, smooth_method="exp")
            corpus_rouge_res = calculate_rouge(predictions, references)

            samples_per_strategy[strat_name] = num_samples

            strategy_results[strat_name] = {
                "strategy_name": strat_name,
                "description": strat_cfg.description,
                "parameters": params,
                "sample_count": num_samples,
                "metrics": {
                    "bleu": corpus_bleu_res,
                    "rouge": corpus_rouge_res,
                    "mean_sentence_bleu": round(float(np.mean(sentence_bleu_list)), 2),
                },
                "factual_adherence": {
                    "mean_consistency_score": round(float(np.mean(factual_scores)), 3),
                    "median_consistency_score": round(float(np.median(factual_scores)), 3),
                    "mean_input_token_preservation": round(float(np.mean(preservation_ratios)), 3),
                    "mean_unsupported_numbers_per_sample": round(float(np.mean(unsupported_counts)), 3),
                    "samples_with_unsupported_numbers": int(sum(1 for c in unsupported_counts if c > 0)),
                },
                "performance": {
                    "mean_latency_ms": round(float(np.mean(latencies_ms)), 2),
                    "median_latency_ms": round(float(np.median(latencies_ms)), 2),
                    "p90_latency_ms": round(float(np.percentile(latencies_ms, 90)), 2),
                    "p95_latency_ms": round(float(np.percentile(latencies_ms, 95)), 2),
                    "min_latency_ms": round(float(np.min(latencies_ms)), 2),
                    "max_latency_ms": round(float(np.max(latencies_ms)), 2),
                    "mean_tokens_per_second": round(float(np.mean(toks_per_sec_list)), 2),
                },
            }

            logger.info(
                f"Completed '{strat_name}': ROUGE-1 F1={corpus_rouge_res['rouge1_f1']:.4f}, "
                f"Corpus BLEU={corpus_bleu_res['corpus_bleu']:.2f}, Latency={np.mean(latencies_ms):.1f} ms"
            )

        # Tabular CSV export
        self.recorder.export_csv()

        # Failure Aggregation
        failures_summary = self.failure_analyzer.aggregate_failures(all_failure_records, samples_per_strategy)
        failure_report_path = Path(self.config.reports_dir) / "failure_analysis.json"
        failure_report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(failure_report_path, "w", encoding="utf-8") as f:
            json.dump(failures_summary, f, indent=2, ensure_ascii=False)
        logger.info(f"Detailed failure analysis saved to: {failure_report_path}")

        # Save Master Experiment Metadata
        self.recorder.save_experiment_metadata(
            experiment_id=self.config.experiment_id,
            model_name=self.config.model_name,
            model_revision=self.config.model_revision,
            dataset_source=self.config.dataset_path,
            category_coverage=[self.config.active_category],
            sample_ids=sample_ids,
            prompt_template_version="v1.0 (Attribute conditioned with negative constraint)",
            strategies_executed=[s.name for s in self.config.strategies],
            strategy_configurations={s.name: s.parameters for s in self.config.strategies},
            random_seed=self.config.random_seed,
            evaluation_summary=strategy_results,
            device=str(self.device),
            gpu_info=gpu_info,
            warmup_latency_sec=warmup_time,
            status="COMPLETED",
        )

        return {
            "experiment_id": self.config.experiment_id,
            "strategies": strategy_results,
            "failures": failures_summary,
            "warmup_time_sec": warmup_time,
            "gpu_info": gpu_info,
        }
