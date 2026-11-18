"""Experiment recording, streaming JSONL persistence, CSV export, and metadata serialization."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List, Optional
import pandas as pd

from descra.utils.logging import get_logger

logger = get_logger("descra.experiments.recorder")


@dataclass
class BaselineOutputRecord:
    experiment_id: str
    example_id: str
    product_id: str
    product_name: str
    product_category: str
    input_features: List[str]
    input_specifications: Dict[str, str]
    reference_description: str
    exact_prompt: str
    generation_strategy: str
    generation_configuration: Dict[str, Any]
    random_seed: int
    generated_description: str
    input_token_count: int
    output_token_count: int
    generation_latency_ms: float
    tokens_per_second: float
    device: str
    gpu_memory_allocated_mb: float
    gpu_memory_reserved_mb: float
    rouge_1_f1: float
    rouge_1_recall: float
    rouge_1_precision: float
    rouge_2_f1: float
    rouge_2_recall: float
    rouge_2_precision: float
    rouge_l_f1: float
    rouge_l_recall: float
    rouge_l_precision: float
    sentence_bleu: float
    bertscore: Optional[Dict[str, float]]
    factual_consistency_score: float
    failure_flags: List[str]
    experiment_timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ExperimentRecorder:
    """Manages persistent serialization of individual generated outputs and master experiment metadata."""

    def __init__(
        self,
        output_dir: str = "data/experiments",
        metadata_filename: str = "experiment_metadata.json",
        outputs_filename: str = "baseline_outputs.jsonl",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.output_dir / metadata_filename
        self.outputs_path = self.output_dir / outputs_filename

    def append_output_record(self, record: BaselineOutputRecord) -> None:
        """Appends a single generated record to the JSONL output file in a streaming fashion."""
        with open(self.outputs_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

    def export_csv(self, csv_filename: str = "baseline_outputs.csv") -> Path:
        """Exports the accumulated JSONL records to a structured tabular CSV file."""
        csv_path = self.output_dir / csv_filename
        if not self.outputs_path.exists():
            logger.warning(f"No JSONL output file found at {self.outputs_path} to convert to CSV.")
            return csv_path

        records = []
        with open(self.outputs_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))

        if not records:
            return csv_path

        # Flatten records for CSV
        flattened = []
        for r in records:
            flat_item = {
                "experiment_id": r["experiment_id"],
                "example_id": r["example_id"],
                "product_id": r["product_id"],
                "product_name": r["product_name"][:80],
                "product_category": r["product_category"],
                "strategy": r["generation_strategy"],
                "generated_description": r["generated_description"],
                "reference_description": r["reference_description"][:100],
                "input_tokens": r["input_token_count"],
                "output_tokens": r["output_token_count"],
                "latency_ms": r["generation_latency_ms"],
                "tokens_per_sec": r["tokens_per_second"],
                "rouge1_f1": r["rouge_1_f1"],
                "rouge2_f1": r["rouge_2_f1"],
                "rougeL_f1": r["rouge_l_f1"],
                "sentence_bleu": r["sentence_bleu"],
                "factual_score": r["factual_consistency_score"],
                "failure_flags": "|".join(r["failure_flags"]),
                "device": r["device"],
                "gpu_alloc_mb": r["gpu_memory_allocated_mb"],
                "timestamp": r["experiment_timestamp"],
            }
            flattened.append(flat_item)

        df = pd.DataFrame(flattened)
        df.to_csv(csv_path, index=False, encoding="utf-8")
        logger.info(f"Tabular CSV successfully exported to: {csv_path} ({len(df)} rows)")
        return csv_path

    def save_experiment_metadata(
        self,
        experiment_id: str,
        model_name: str,
        model_revision: str,
        dataset_source: str,
        category_coverage: List[str],
        sample_ids: List[str],
        prompt_template_version: str,
        strategies_executed: List[str],
        strategy_configurations: Dict[str, Any],
        random_seed: int,
        evaluation_summary: Dict[str, Any],
        device: str,
        gpu_info: Dict[str, Any],
        warmup_latency_sec: float,
        status: str = "COMPLETED",
    ) -> Path:
        """Serializes full reproducibility metadata and runtime environment fingerprint."""
        # Retrieve git commit hash if available
        git_hash = "UNKNOWN"
        try:
            res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
            if res.returncode == 0:
                git_hash = res.stdout.strip()
        except Exception:
            pass

        import torch
        import transformers

        metadata = {
            "experiment_id": experiment_id,
            "experiment_timestamp": datetime.now(timezone.utc).isoformat(),
            "git_commit_hash": git_hash,
            "model_metadata": {
                "model_name": model_name,
                "model_revision": model_revision,
                "architecture": "Seq2SeqLM (FLAN-T5-small)",
                "precision": "FP16 (torch.float16)",
                "total_parameters": 76961152,
            },
            "dataset_metadata": {
                "source": dataset_source,
                "version": "v1.0 (Amazon Reviews 2023 - All_Beauty)",
                "total_evaluated_samples": len(sample_ids),
                "sample_ids": sample_ids,
                "category_coverage": category_coverage,
            },
            "prompt_metadata": {
                "prompt_template_version": prompt_template_version,
                "negative_constraint_applied": True,
            },
            "strategies_metadata": {
                "strategies_evaluated": strategies_executed,
                "configurations": strategy_configurations,
                "random_seed": random_seed,
            },
            "environment_metadata": {
                "python_version": sys.version.split()[0],
                "pytorch_version": torch.__version__,
                "transformers_version": transformers.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_runtime_version": torch.version.cuda if torch.cuda.is_available() else None,
                "execution_device": device,
                "gpu_hardware": gpu_info,
                "warmup_time_sec": round(warmup_latency_sec, 3),
            },
            "evaluation_results_summary": evaluation_summary,
            "experiment_status": status,
        }

        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.info(f"Experiment metadata saved successfully to: {self.metadata_path}")
        return self.metadata_path
