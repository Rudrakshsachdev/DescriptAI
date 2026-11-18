"""PyTorch Dataset and Dynamic Data Collator for Seq2Seq Supervised Fine-Tuning."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from descra.utils.logging import get_logger

logger = get_logger("descra.training.dataset")


class ProductDescriptionDataset(Dataset):
    """PyTorch Dataset for product prompt-to-description Seq2Seq training."""

    def __init__(
        self,
        jsonl_path: str,
        tokenizer: PreTrainedTokenizer,
        max_input_length: int = 512,
        max_target_length: int = 160,
    ):
        self.jsonl_path = Path(jsonl_path)
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

        if not self.jsonl_path.exists():
            raise FileNotFoundError(f"Dataset file does not exist: {self.jsonl_path}")

        self.records: List[Dict[str, Any]] = []
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.records.append(json.loads(line))

        logger.info(f"Loaded {len(self.records)} records from {self.jsonl_path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.records[idx]
        prompt = record.get("prompt", "")
        target = record.get("target_description") or record.get("reference_description", "")

        # Tokenize encoder prompt
        source_encoding = self.tokenizer(
            prompt,
            max_length=self.max_input_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )

        # Tokenize target description
        target_encoding = self.tokenizer(
            target,
            max_length=self.max_target_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )

        return {
            "product_id": record.get("product_id", f"sample_{idx}"),
            "input_ids": source_encoding["input_ids"],
            "attention_mask": source_encoding["attention_mask"],
            "labels": target_encoding["input_ids"],
        }


class DataCollatorForSFT:
    """Dynamically pads input_ids, attention_mask, and labels to batch max sequence lengths.

    Masks label padding tokens with -100 for PyTorch CrossEntropyLoss ignore_index.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, pad_to_multiple_of: Optional[int] = 8):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids_list = [item["input_ids"] for item in batch]
        attention_mask_list = [item["attention_mask"] for item in batch]
        labels_list = [item["labels"] for item in batch]

        # Determine batch max lengths
        max_input_len = max(len(ids) for ids in input_ids_list)
        max_label_len = max(len(lbl) for lbl in labels_list)

        if self.pad_to_multiple_of:
            max_input_len = ((max_input_len + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of
            max_label_len = ((max_label_len + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of

        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []

        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

        for inp_ids, att_mask, lbl in zip(input_ids_list, attention_mask_list, labels_list):
            # Pad input_ids
            inp_pad_len = max_input_len - len(inp_ids)
            padded_input_ids.append(inp_ids + [pad_token_id] * inp_pad_len)
            padded_attention_mask.append(att_mask + [0] * inp_pad_len)

            # Pad labels with -100
            lbl_pad_len = max_label_len - len(lbl)
            padded_labels.append(lbl + [-100] * lbl_pad_len)

        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
        }
