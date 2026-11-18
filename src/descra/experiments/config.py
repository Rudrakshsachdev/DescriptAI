"""Experiment configuration dataclasses and loader for Descra baseline experimentation."""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml


@dataclass
class DecodingStrategyConfig:
    """Configuration for an individual autoregressive decoding strategy."""
    name: str
    description: str
    parameters: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


@dataclass
class ExperimentConfig:
    """Master configuration for baseline model experimentation."""
    experiment_id: str = "EXP-BASELINE-FLANT5-001"
    model_name: str = "google/flan-t5-small"
    model_revision: str = "main"
    device: str = "auto"
    use_fp16: bool = True
    max_input_length: int = 512
    random_seed: int = 42
    sample_size: int = 100
    dataset_path: str = "data/processed/validation.jsonl"
    output_dir: str = "data/experiments"
    reports_dir: str = "reports/experiments"
    compute_bertscore: bool = False
    num_warmup_iterations: int = 3
    record_cuda_memory: bool = True
    strategies: List[DecodingStrategyConfig] = field(default_factory=list)
    proposed_categories: List[str] = field(default_factory=list)
    active_category: str = "All_Beauty"

    @classmethod
    def from_yaml(cls, yaml_path: str = "configs/experiments.yaml") -> "ExperimentConfig":
        """Loads configuration from a YAML file."""
        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"Experiment config file not found: {yaml_path}")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        exp_data = data.get("experiment", {})
        strat_data = data.get("decoding_strategies", {})
        bench_data = data.get("benchmarking", {})
        cat_data = data.get("categories", {})

        strategies: List[DecodingStrategyConfig] = []
        for s_name, s_cfg in strat_data.items():
            desc = s_cfg.get("description", "")
            params = {k: v for k, v in s_cfg.items() if k != "description"}
            strategies.append(DecodingStrategyConfig(name=s_name, description=desc, parameters=params))

        return cls(
            experiment_id=exp_data.get("experiment_id", "EXP-BASELINE-001"),
            model_name=exp_data.get("model_name", "google/flan-t5-small"),
            model_revision=exp_data.get("model_revision", "main"),
            device=exp_data.get("device", "auto"),
            use_fp16=exp_data.get("use_fp16", True),
            max_input_length=exp_data.get("max_input_length", 512),
            random_seed=exp_data.get("random_seed", 42),
            sample_size=exp_data.get("sample_size", 100),
            dataset_path=exp_data.get("dataset_path", "data/processed/validation.jsonl"),
            output_dir=exp_data.get("output_dir", "data/experiments"),
            reports_dir=exp_data.get("reports_dir", "reports/experiments"),
            compute_bertscore=exp_data.get("compute_bertscore", False),
            num_warmup_iterations=bench_data.get("num_warmup_iterations", 3),
            record_cuda_memory=bench_data.get("record_cuda_memory", True),
            strategies=strategies,
            proposed_categories=cat_data.get("proposed_roadmap_categories", []),
            active_category=cat_data.get("active_dataset_category", "All_Beauty"),
        )
