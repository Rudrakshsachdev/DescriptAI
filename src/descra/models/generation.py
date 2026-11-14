"""Prompt construction and autoregressive text generation pipeline for Descra."""

import time
from typing import Any, Dict, List, Optional, Tuple
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from descra.utils.logging import get_logger

logger = get_logger("descra.models.generation")

DEFAULT_GENERATION_CONFIG = {
    "max_new_tokens": 128,
    "num_beams": 4,
    "length_penalty": 1.0,
    "no_repeat_ngram_size": 3,
    "early_stopping": True,
    "do_sample": False,
}

GREEDY_GENERATION_CONFIG = {
    "max_new_tokens": 128,
    "num_beams": 1,
    "do_sample": False,
}


def construct_prompt(
    product_name: str,
    category: str = "",
    features: Optional[List[str]] = None,
    specifications: Optional[Dict[str, str]] = None,
) -> str:
    """Builds a deterministic, structured instruction prompt for Seq2Seq description generation.

    Args:
        product_name: Name/title of the product.
        category: Catalog category (omitted if empty or 'General').
        features: List of bullet point features.
        specifications: Dictionary of key-value specifications.

    Returns:
        Structured text prompt ready for tokenization.
    """
    sections = [
        "Generate a clear and engaging product description using only the information provided below.",
        f"Product Name: {product_name.strip()}",
    ]

    clean_cat = category.strip()
    if clean_cat and clean_cat.lower() not in ["general", "all beauty", "all_beauty", ""]:
        sections.append(f"Category: {clean_cat}")

    if features:
        valid_features = [f.strip() for f in features if f and f.strip()]
        if valid_features:
            sections.append("Features:")
            for feat in valid_features:
                sections.append(f"- {feat}")

    if specifications:
        valid_specs = {k.strip(): str(v).strip() for k, v in specifications.items() if k and str(v).strip()}
        if valid_specs:
            sections.append("Specifications:")
            for k, v in valid_specs.items():
                sections.append(f"- {k}: {v}")

    sections.append(
        "Write a concise product description. "
        "Do not invent specifications, certifications, ingredients, benefits, or claims."
    )

    return "\n\n".join(sections)


def generate_description(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    generation_config: Optional[Dict[str, Any]] = None,
    device: Optional[torch.device] = None,
    max_input_length: int = 512,
) -> Tuple[str, Dict[str, Any]]:
    """Generates a product description from an input prompt using a Transformer model.

    Args:
        model: Loaded PyTorch Seq2Seq Transformer model.
        tokenizer: Matching subword tokenizer.
        prompt: Formatted input string.
        generation_config: Decoding parameters (beams, penalties, tokens).
        device: Target compute device.
        max_input_length: Input sequence truncation limit.

    Returns:
        (generated_text, execution_metadata)
    """
    if device is None:
        device = next(model.parameters()).device

    config = dict(DEFAULT_GENERATION_CONFIG)
    if generation_config:
        config.update(generation_config)

    # Tokenize input
    inputs = tokenizer(
        prompt,
        max_length=max_input_length,
        truncation=True,
        padding=False,
        return_tensors="pt",
    ).to(device)

    input_tokens_count = inputs["input_ids"].shape[1]

    # CUDA timing synchronization
    if device.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            **config,
        )

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_time = time.perf_counter() - start_time

    generated_tokens_count = output_ids.shape[1]
    tokens_per_second = (
        generated_tokens_count / elapsed_time if elapsed_time > 0 else 0.0
    )

    # Decode
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    metadata = {
        "input_tokens": input_tokens_count,
        "output_tokens": generated_tokens_count,
        "latency_ms": round(elapsed_time * 1000, 2),
        "tokens_per_second": round(tokens_per_second, 2),
        "decoding_config": config,
    }

    return generated_text, metadata

