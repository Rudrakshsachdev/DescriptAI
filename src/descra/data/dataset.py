"""Dataset loading and Seq2Seq prompt formatting interface for Descra."""

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union
import torch
from torch.utils.data import Dataset

from descra.utils.logging import get_logger

logger = get_logger("descra.data.dataset")


def format_model_input(
    product_name: str,
    category: str,
    features: List[str],
    specifications: Optional[Dict[str, str]] = None,
    target_audience: Optional[str] = "General Customers",
    writing_style: Optional[str] = "Engaging and factual",
) -> str:
    """Constructs a deterministic text-to-text prompt template for FLAN-T5.

    Guarantees strict factual grounding by explicitly commanding the model
    to rely only on the supplied features and technical specifications.
    """
    feature_bullet_lines = "\n".join(f"- {f.strip()}" for f in features if f.strip())
    specs_lines = ""
    if specifications:
        specs_lines = "\nSpecifications:\n" + "\n".join(f"- {k}: {v}" for k, v in specifications.items() if k and v)

    prompt = (
        f"Generate a concise, engaging, and factually grounded product description using only the supplied product information.\n\n"
        f"Product Name: {product_name.strip()}\n"
        f"Category: {category.strip()}\n"
        f"Target Audience: {target_audience}\n"
        f"Writing Style: {writing_style}\n\n"
        f"Features:\n{feature_bullet_lines}"
        f"{specs_lines}\n\n"
        f"Product Description:"
    )
    return prompt.strip()


def format_model_target(reference_description: str) -> str:
    """Formats and normalizes reference target description for teacher forcing."""
    return reference_description.strip()


class ProductDescriptionDataset(Dataset):
    """PyTorch Dataset for Seq2Seq product description generation."""

    def __init__(
        self,
        jsonl_path: str,
        tokenizer: Optional[Any] = None,
        max_input_length: int = 256,
        max_target_length: int = 128,
    ):
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"Dataset JSONL not found: {jsonl_path}")

        self.jsonl_path = jsonl_path
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

        self.records: List[Dict[str, Any]] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                if line.strip():
                    try:
                        record = json.loads(line)
                        # Validate schema
                        assert "product_id" in record
                        assert "product_name" in record
                        assert "features" in record
                        assert "reference_description" in record
                        self.records.append(record)
                    except Exception as e:
                        logger.warning(f"Skipping malformed record at {jsonl_path}:{line_num}: {e}")

        logger.info(f"Loaded {len(self.records):,} valid records from {jsonl_path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        input_text = format_model_input(
            product_name=rec.get("product_name", ""),
            category=rec.get("category", "General"),
            features=rec.get("features", []),
            specifications=rec.get("specifications", {}),
        )
        target_text = format_model_target(rec.get("reference_description", ""))

        item: Dict[str, Any] = {
            "product_id": rec.get("product_id"),
            "input_text": input_text,
            "target_text": target_text,
        }

        # If tokenizer is supplied, return tokenized tensors
        if self.tokenizer is not None:
            input_enc = self.tokenizer(
                input_text,
                max_length=self.max_input_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            target_enc = self.tokenizer(
                target_text,
                max_length=self.max_target_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            labels = target_enc["input_ids"].squeeze(0).clone()
            # Replace padding token id with -100 for PyTorch CrossEntropyLoss ignore_index
            labels[labels == self.tokenizer.pad_token_id] = -100

            item["input_ids"] = input_enc["input_ids"].squeeze(0)
            item["attention_mask"] = input_enc["attention_mask"].squeeze(0)
            item["labels"] = labels

        return item

