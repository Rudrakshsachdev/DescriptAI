"""Production inference engine for Project Descra (Phase 2 FastAPI Handover).

Enables lightweight, standalone description generation using the fine-tuned
FLAN-T5-small checkpoint or pretrained baseline.
"""

from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Union

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


@dataclass
class GenerationConfig:
    """Production decoding and generation configuration."""
    num_beams: int = 2
    max_new_tokens: int = 128
    min_new_tokens: int = 10
    no_repeat_ngram_size: int = 3
    length_penalty: float = 1.0
    early_stopping: bool = True
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0


@dataclass
class PredictionOutput:
    """Structured output from description predictor."""
    generated_description: str
    input_prompt: str
    output_tokens: int
    latency_ms: float
    device: str
    strategy: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class DescriptionPredictor:
    """Standalone production inference class for serving Descra descriptions."""

    def __init__(
        self,
        checkpoint_path: Union[str, Path] = "checkpoints/sft/best_checkpoint",
        device: Optional[str] = None,
        use_half_precision: bool = True,
    ):
        """Initializes the predictor, loading model weights and tokenizer.

        Args:
            checkpoint_path: Path to checkpoint directory or Hugging Face model ID.
            device: 'cuda', 'cpu', or None (auto-detects CUDA).
            use_half_precision: If True and on CUDA, uses BF16/FP16 for reduced memory.
        """
        self.checkpoint_path = Path(checkpoint_path) if isinstance(checkpoint_path, str) else checkpoint_path
        
        # Resolve device
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load Tokenizer
        tokenizer_source = str(self.checkpoint_path) if self.checkpoint_path.exists() else "google/flan-t5-small"
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, local_files_only=self.checkpoint_path.exists())

        # Determine precision dtype
        if self.device.type == "cuda" and use_half_precision:
            if torch.cuda.is_bf16_supported():
                model_dtype = torch.bfloat16
            else:
                model_dtype = torch.float16
        else:
            model_dtype = torch.float32

        # Load Model weights
        model_source = str(self.checkpoint_path) if self.checkpoint_path.exists() else "google/flan-t5-small"
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_source,
            dtype=model_dtype,
            local_files_only=self.checkpoint_path.exists(),
        ).to(self.device)
        self.model.eval()

        self.default_config = GenerationConfig()

    @staticmethod
    def construct_prompt(
        product_name: str,
        features: Optional[List[str]] = None,
        category: Optional[str] = None,
        brand: Optional[str] = None,
    ) -> str:
        """Constructs canonical e-commerce prompt template."""
        p_name = product_name.strip() if product_name else "Unknown Product"
        parts = ["Generate a clear and engaging product description.", f"Product Name: {p_name}"]

        if category and category.strip():
            parts.append(f"Category: {category.strip()}")
        if brand and brand.strip():
            parts.append(f"Brand: {brand.strip()}")

        if features:
            clean_feats = [f.strip() for f in features if f and f.strip()]
            if clean_feats:
                feat_str = " - " + " - ".join(clean_feats)
                parts.append(f"Features:{feat_str}")

        return " ".join(parts)

    def predict(
        self,
        product_name: str,
        features: Optional[List[str]] = None,
        category: Optional[str] = None,
        brand: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
    ) -> PredictionOutput:
        """Generates a product description for a single product.

        Args:
            product_name: Title/name of the item.
            features: List of bullet point features or specifications.
            category: Optional product category.
            brand: Optional brand name.
            config: Optional custom GenerationConfig (defaults to Beam-2).

        Returns:
            PredictionOutput with generated text, token count, and latency.
        """
        cfg = config or self.default_config
        prompt = self.construct_prompt(product_name=product_name, features=features, category=category, brand=brand)

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        inputs = self.tokenizer(
            prompt,
            max_length=256,
            padding=False,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                num_beams=cfg.num_beams,
                max_new_tokens=cfg.max_new_tokens,
                min_new_tokens=cfg.min_new_tokens,
                no_repeat_ngram_size=cfg.no_repeat_ngram_size,
                length_penalty=cfg.length_penalty,
                early_stopping=cfg.early_stopping,
                do_sample=cfg.do_sample,
                temperature=cfg.temperature if cfg.do_sample else None,
                top_p=cfg.top_p if cfg.do_sample else None,
            )

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        gen_tokens = outputs[0]
        decoded = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        # Sanitize whitespace
        cleaned = " ".join(decoded.split())

        strategy_name = f"beam_search_{cfg.num_beams}" if cfg.num_beams > 1 else ("sampling" if cfg.do_sample else "greedy")

        return PredictionOutput(
            generated_description=cleaned,
            input_prompt=prompt,
            output_tokens=len(gen_tokens),
            latency_ms=round(latency_ms, 2),
            device=str(self.device),
            strategy=strategy_name,
            metadata={
                "num_beams": cfg.num_beams,
                "input_token_count": int(inputs["input_ids"].shape[1]),
                "checkpoint": str(self.checkpoint_path),
            },
        )
