"""Model and tokenizer loader for Descra.

Handles device placement, configuration verification, evaluation mode setting,
and graceful fallback handling for local Transformer models.
"""

import os
from typing import Optional, Tuple
import torch
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from descra.utils.logging import get_logger

logger = get_logger("descra.models.loader")


def get_optimal_device(requested_device: str = "auto") -> torch.device:
    """Selects optimal compute device with automated CUDA detection and fallback."""
    if requested_device == "auto":
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            logger.info(f"CUDA GPU detected: {device_name}. Selecting 'cuda:0'.")
            return torch.device("cuda:0")
        else:
            logger.info("CUDA GPU not detected. Falling back to 'cpu'.")
            return torch.device("cpu")
    elif requested_device == "cuda":
        if not torch.cuda.is_available():
            logger.warning("CUDA explicitly requested but unavailable. Falling back to 'cpu'.")
            return torch.device("cpu")
        return torch.device("cuda:0")
    else:
        return torch.device(requested_device)


def load_model_and_tokenizer(
    model_name: str = "google/flan-t5-small",
    device: str = "auto",
    use_fp16: bool = True,
    local_files_only: bool = False,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer, torch.device]:
    """Loads pretrained Seq2Seq model and tokenizer in evaluation mode.

    Args:
        model_name: Hugging Face model repository or local directory path.
        device: Device string ('auto', 'cuda', 'cpu').
        use_fp16: Whether to cast model weights to float16 on GPU.
        local_files_only: Whether to restrict to offline cache.

    Returns:
        (model, tokenizer, target_device)

    Raises:
        RuntimeError: If model cannot be loaded or initialized.
    """
    target_device = get_optimal_device(device)
    logger.info(f"Loading tokenizer for '{model_name}'...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )
    except Exception as e:
        logger.error(f"Failed to load tokenizer '{model_name}': {e}")
        raise RuntimeError(f"Tokenizer loading failed for '{model_name}': {e}") from e

    logger.info(f"Loading pretrained Seq2Seq model '{model_name}'...")
    try:
        config = AutoConfig.from_pretrained(model_name, local_files_only=local_files_only)
        
        dtype = torch.float16 if (target_device.type == "cuda" and use_fp16) else torch.float32

        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            config=config,
            torch_dtype=dtype,
            local_files_only=local_files_only,
        )
        model.to(target_device)
        model.eval()  # Put in evaluation mode

        param_count = sum(p.numel() for p in model.parameters())
        logger.info(
            f"Successfully loaded '{model_name}' ({param_count:,} parameters) "
            f"on {target_device} [dtype={dtype}]."
        )
        return model, tokenizer, target_device

    except Exception as e:
        logger.error(f"Failed to load model '{model_name}': {e}")
        raise RuntimeError(f"Model loading failed for '{model_name}': {e}") from e

