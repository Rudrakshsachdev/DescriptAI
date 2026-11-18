"""Unit tests for Step 7 baseline experimentation framework."""

import json
from pathlib import Path
import pytest

from descra.experiments.analysis import ExperimentFailureAnalyzer, FailureRecord
from descra.experiments.category_inspector import (
    PROPOSED_CATEGORIES,
    CategoryAuditor,
    CategoryAvailabilityRecord,
)
from descra.experiments.config import DecodingStrategyConfig, ExperimentConfig
from descra.experiments.recorder import BaselineOutputRecord, ExperimentRecorder


def test_experiment_config_from_yaml(tmp_path):
    yaml_content = """
experiment:
  experiment_id: "TEST-EXP-001"
  model_name: "google/flan-t5-small"
  sample_size: 10
  dataset_path: "data/processed/validation.jsonl"
  output_dir: "data/experiments"
  reports_dir: "reports/experiments"

decoding_strategies:
  greedy:
    num_beams: 1
    do_sample: false
    max_new_tokens: 64
    description: "Greedy search"
  beam_search_2:
    num_beams: 2
    do_sample: false
    max_new_tokens: 64
    description: "Beam search 2"

benchmarking:
  num_warmup_iterations: 2
  record_cuda_memory: true

categories:
  proposed_roadmap_categories:
    - "Beauty"
    - "Electronics"
  active_dataset_category: "All_Beauty"
"""
    test_yaml = tmp_path / "test_exp.yaml"
    test_yaml.write_text(yaml_content, encoding="utf-8")

    cfg = ExperimentConfig.from_yaml(str(test_yaml))
    assert cfg.experiment_id == "TEST-EXP-001"
    assert cfg.sample_size == 10
    assert len(cfg.strategies) == 2
    assert cfg.strategies[0].name == "greedy"
    assert cfg.strategies[0].parameters["num_beams"] == 1
    assert cfg.strategies[1].name == "beam_search_2"
    assert cfg.strategies[1].parameters["num_beams"] == 2


def test_category_auditor_records():
    auditor = CategoryAuditor()
    records = auditor.audit_all_categories()

    assert len(records) == len(PROPOSED_CATEGORIES)
    # Beauty & Personal care should be present and eligible
    beauty_rec = next(r for r in records if r.category_name == "Beauty and Personal Care")
    assert beauty_rec.experiment_eligibility is True
    assert beauty_rec.local_availability in ["AVAILABLE_PROCESSED", "AVAILABLE_RAW_ONLY"]
    assert beauty_rec.raw_record_count > 0

    # Electronics should be absent on local disk
    elec_rec = next(r for r in records if r.category_name == "Electronics")
    assert elec_rec.experiment_eligibility is False
    assert elec_rec.local_availability == "ABSENT_NOT_DOWNLOADED"
    assert "absent on local disk" in elec_rec.exclusion_reason.lower()


def test_failure_analyzer_empty_output():
    analyzer = ExperimentFailureAnalyzer()
    failures = analyzer.analyze_single_output(
        example_id="ex_001",
        product_id="P001",
        category="All_Beauty",
        strategy="greedy",
        input_prompt="Prompt text",
        generated_output="   ",
    )
    assert len(failures) == 1
    assert failures[0].failure_type == "EMPTY_OUTPUT"
    assert failures[0].severity == "CRITICAL"


def test_failure_analyzer_output_collapse():
    analyzer = ExperimentFailureAnalyzer()
    failures = analyzer.analyze_single_output(
        example_id="ex_002",
        product_id="P002",
        category="All_Beauty",
        strategy="greedy",
        input_prompt="Product Name: Facial Cream. Features: Hydrating.",
        generated_output="Kemzzone",
    )
    assert any(f.failure_type == "OUTPUT_COLLAPSE" for f in failures)


def test_failure_analyzer_prompt_echo():
    analyzer = ExperimentFailureAnalyzer()
    failures = analyzer.analyze_single_output(
        example_id="ex_003",
        product_id="P003",
        category="All_Beauty",
        strategy="greedy",
        input_prompt="Product Name: Brush. Features: Soft bristles.",
        generated_output="Product Name: Brush Features: - Soft bristles for cats.",
    )
    assert any(f.failure_type == "PROMPT_ECHO" for f in failures)


def test_failure_analyzer_unsupported_numbers():
    analyzer = ExperimentFailureAnalyzer()
    failures = analyzer.analyze_single_output(
        example_id="ex_004",
        product_id="P004",
        category="All_Beauty",
        strategy="greedy",
        input_prompt="Product Name: Lip Balm. Features: Beeswax.",
        generated_output="Organic beeswax lip balm with 500ml volume and 48-hour moisture.",
    )
    assert any(f.failure_type == "UNSUPPORTED_NUMBERS" for f in failures)
    unsupp_fail = next(f for f in failures if f.failure_type == "UNSUPPORTED_NUMBERS")
    assert "500ml" in unsupp_fail.evidence or "48" in unsupp_fail.evidence


def test_recorder_streaming_and_csv_export(tmp_path):
    recorder = ExperimentRecorder(output_dir=str(tmp_path))

    rec = BaselineOutputRecord(
        experiment_id="EXP-TEST",
        example_id="ex_01",
        product_id="B001",
        product_name="Test Product",
        product_category="Beauty",
        input_features=["Feature 1"],
        input_specifications={"Size": "100ml"},
        reference_description="Human reference description text.",
        exact_prompt="Prompt text",
        generation_strategy="greedy",
        generation_configuration={"num_beams": 1},
        random_seed=42,
        generated_description="Generated description.",
        input_token_count=20,
        output_token_count=10,
        generation_latency_ms=250.0,
        tokens_per_second=40.0,
        device="cpu",
        gpu_memory_allocated_mb=0.0,
        gpu_memory_reserved_mb=0.0,
        rouge_1_f1=0.45,
        rouge_1_recall=0.40,
        rouge_1_precision=0.50,
        rouge_2_f1=0.25,
        rouge_2_recall=0.20,
        rouge_2_precision=0.30,
        rouge_l_f1=0.35,
        rouge_l_recall=0.30,
        rouge_l_precision=0.40,
        sentence_bleu=15.5,
        bertscore=None,
        factual_consistency_score=0.9,
        failure_flags=[],
        experiment_timestamp="2026-11-18T00:00:00Z",
    )

    recorder.append_output_record(rec)
    assert recorder.outputs_path.exists()

    # Verify JSONL content
    with open(recorder.outputs_path, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]
    assert len(lines) == 1
    assert lines[0]["product_id"] == "B001"
    assert lines[0]["rouge_1_f1"] == 0.45

    # Test CSV export
    csv_file = recorder.export_csv("test_outputs.csv")
    assert csv_file.exists()
    assert csv_file.stat().st_size > 50
