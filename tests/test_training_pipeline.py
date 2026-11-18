"""Unit tests for Step 8 Supervised Fine-Tuning pipeline, datasets, and checkpointing."""

import json
from pathlib import Path
import pytest
import torch
from transformers import AutoTokenizer

from descra.training.checkpoint import CheckpointManager
from descra.training.config import TrainingConfig
from descra.training.dataset import DataCollatorForSFT, ProductDescriptionDataset
from descra.training.optimizer import get_optimizer_and_scheduler


def test_training_config_from_yaml(tmp_path):
    yaml_content = """
model:
  name: "google/flan-t5-small"
  tokenizer_name: "google/flan-t5-small"
dataset:
  train_path: "data/processed/sft_train.jsonl"
  validation_path: "data/processed/sft_validation.jsonl"
  max_input_length: 256
  max_target_length: 128
training:
  output_dir: "checkpoints/test"
  reports_dir: "reports/test"
  figures_dir: "reports/test/figures"
  num_train_epochs: 2
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 2
  learning_rate: 0.0003
  weight_decay: 0.01
  fp16: false
"""
    cfg_file = tmp_path / "test_training.yaml"
    cfg_file.write_text(yaml_content, encoding="utf-8")

    config = TrainingConfig.from_yaml(str(cfg_file))
    assert config.model_name == "google/flan-t5-small"
    assert config.num_train_epochs == 2
    assert config.per_device_train_batch_size == 4
    assert config.gradient_accumulation_steps == 2
    assert config.effective_batch_size == 8
    assert config.learning_rate == 0.0003


def test_product_description_dataset(tmp_path):
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small", local_files_only=True)

    records = [
        {
            "product_id": "P001",
            "prompt": "Product Name: Lip Balm. Features: Organic beeswax.",
            "target_description": "Natural beeswax lip balm providing smooth moisture.",
        },
        {
            "product_id": "P002",
            "prompt": "Product Name: Body Lotion.",
            "target_description": "Hydrating daily body lotion.",
        },
    ]

    jsonl_path = tmp_path / "test_data.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    dataset = ProductDescriptionDataset(
        jsonl_path=str(jsonl_path),
        tokenizer=tokenizer,
        max_input_length=64,
        max_target_length=32,
    )

    assert len(dataset) == 2
    item = dataset[0]
    assert item["product_id"] == "P001"
    assert isinstance(item["input_ids"], list)
    assert isinstance(item["attention_mask"], list)
    assert isinstance(item["labels"], list)
    assert len(item["input_ids"]) > 0
    assert len(item["labels"]) > 0


def test_data_collator_label_masking():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small", local_files_only=True)
    collator = DataCollatorForSFT(tokenizer=tokenizer, pad_to_multiple_of=8)

    sample_batch = [
        {
            "input_ids": [10, 20, 30],
            "attention_mask": [1, 1, 1],
            "labels": [100, 200],
        },
        {
            "input_ids": [10, 20, 30, 40, 50],
            "attention_mask": [1, 1, 1, 1, 1],
            "labels": [100, 200, 300, 400],
        },
    ]

    batch = collator(sample_batch)

    assert batch["input_ids"].shape[0] == 2
    assert batch["attention_mask"].shape[0] == 2
    assert batch["labels"].shape[0] == 2

    # Check pad multiple of 8
    assert batch["input_ids"].shape[1] % 8 == 0
    assert batch["labels"].shape[1] % 8 == 0

    # Ensure shorter label sequence was padded with -100
    labels = batch["labels"]
    assert -100 in labels[0]
    assert labels[0][-1].item() == -100


def test_checkpoint_manager_save_and_load(tmp_path):
    from transformers import AutoModelForSeq2SeqLM
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small", local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small", local_files_only=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.5, total_iters=5)

    mgr = CheckpointManager(output_dir=str(tmp_path / "checkpoints"))

    saved_path = mgr.save_checkpoint(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        epoch=1,
        global_step=50,
        train_loss=1.85,
        eval_loss=1.75,
        is_best=True,
    )

    assert saved_path.exists()
    assert (saved_path / "model.safetensors").exists()
    assert (saved_path / "training_state.json").exists()
    assert mgr.best_checkpoint_dir.exists()

    # Test reloading state
    loaded_state = mgr.load_checkpoint(
        checkpoint_dir=str(saved_path),
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    assert loaded_state["epoch"] == 1
    assert loaded_state["global_step"] == 50
    assert loaded_state["eval_loss"] == 1.75


def test_optimizer_and_scheduler_grouping():
    from transformers import AutoModelForSeq2SeqLM
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small", local_files_only=True)
    cfg = TrainingConfig(learning_rate=2e-4, weight_decay=0.01, warmup_ratio=0.1)

    opt, sched = get_optimizer_and_scheduler(model, cfg, num_training_steps=100)
    assert len(opt.param_groups) == 2
    # Decay group
    assert opt.param_groups[0]["weight_decay"] == 0.01
    # No decay group (bias, LayerNorm)
    assert opt.param_groups[1]["weight_decay"] == 0.0
