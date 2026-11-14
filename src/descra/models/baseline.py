"""High-level Baseline Generator class for Descra.

Encapsulates model initialization, prompt formatting, batch generation,
and decoding strategy configuration.
"""

from typing import Any, Dict, List, Optional
import torch

from descra.models.generation import (
    DEFAULT_GENERATION_CONFIG,
    GREEDY_GENERATION_CONFIG,
    construct_prompt,
    generate_description,
)
from descra.models.loader import load_model_and_tokenizer
from descra.utils.logging import get_logger

logger = get_logger("descra.models.baseline")


class BaselineGenerator:
    """Manages baseline generation using pretrained Seq2Seq models."""

    def __init__(
        self,
        model_name: str = "google/flan-t5-small",
        device: str = "auto",
        use_fp16: bool = True,
        max_input_length: int = 512,
        local_files_only: bool = False,
    ):
        self.model_name = model_name
        self.max_input_length = max_input_length

        self.model, self.tokenizer, self.device = load_model_and_tokenizer(
            model_name=model_name,
            device=device,
            use_fp16=use_fp16,
            local_files_only=local_files_only,
        )

    def generate(
        self,
        product_name: str,
        category: str = "",
        features: Optional[List[str]] = None,
        specifications: Optional[Dict[str, str]] = None,
        strategy: str = "beam_search",
        custom_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generates description for a single product record.

        Args:
            product_name: Product title.
            category: Product category.
            features: Bullet point list.
            specifications: Specifications dictionary.
            strategy: Decoding strategy ('beam_search', 'greedy', 'custom').
            custom_config: Optional override dictionary.

        Returns:
            Dictionary with 'description', 'prompt', and performance 'metadata'.
        """
        prompt = construct_prompt(
            product_name=product_name,
            category=category,
            features=features,
            specifications=specifications,
        )

        if strategy == "greedy":
            gen_config = dict(GREEDY_GENERATION_CONFIG)
        elif strategy == "beam_search":
            gen_config = dict(DEFAULT_GENERATION_CONFIG)
        else:
            gen_config = dict(DEFAULT_GENERATION_CONFIG)

        if custom_config:
            gen_config.update(custom_config)

        generated_text, metadata = generate_description(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt,
            generation_config=gen_config,
            device=self.device,
            max_input_length=self.max_input_length,
        )

        return {
            "product_name": product_name,
            "prompt": prompt,
            "description": generated_text,
            "strategy": strategy,
            "metadata": metadata,
        }

    def generate_batch(
        self,
        records: List[Dict[str, Any]],
        strategy: str = "beam_search",
        custom_config: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Processes a list of records sequentially, returning generated outputs."""
        results = []
        for r in records:
            p_name = r.get("product_name") or r.get("title", "")
            cat = r.get("category", "")
            feats = r.get("features", [])
            specs = r.get("specifications", {})

            res = self.generate(
                product_name=p_name,
                category=cat,
                features=feats,
                specifications=specs,
                strategy=strategy,
                custom_config=custom_config,
            )
            # Retain ground truth reference if available
            if "reference_description" in r or "description" in r:
                res["reference_description"] = r.get("reference_description") or r.get("description")
            results.append(res)
        return results

