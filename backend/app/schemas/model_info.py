"""Pydantic response schema for model information endpoint."""

from typing import Any, Dict, List
from pydantic import BaseModel, Field


class ModelInfoData(BaseModel):
    """Safe, sanitized operational metadata for the serving deep learning model."""

    model_id: str = Field(
        ...,
        description="Logical model identifier or base Hugging Face model name.",
        examples=["google/flan-t5-small"],
    )
    architecture: str = Field(
        ...,
        description="Transformer neural network architecture family.",
        examples=["Encoder-Decoder Transformer (Seq2Seq)"],
    )
    checkpoint: str = Field(
        ...,
        description="Logical checkpoint identifier (sanitized, relative path).",
        examples=["checkpoints/sft/best_checkpoint"],
    )
    device: str = Field(
        ...,
        description="Inference execution hardware device.",
        examples=["cuda:0"],
    )
    total_parameters: int = Field(
        ...,
        description="Total trainable and non-trainable model parameter count.",
        examples=[76961152],
    )
    precision: str = Field(
        ...,
        description="Precision format used for inference weights.",
        examples=["bfloat16"],
    )
    supported_strategies: List[str] = Field(
        ...,
        description="List of supported decoding and search strategies.",
        examples=[["beam_search", "greedy", "sampling"]],
    )
    default_config: Dict[str, Any] = Field(
        ...,
        description="Default production generation hyperparameters.",
        examples=[{
            "num_beams": 2,
            "max_new_tokens": 128,
            "min_new_tokens": 10,
            "no_repeat_ngram_size": 3,
            "length_penalty": 1.0,
            "early_stopping": True,
        }],
    )
    is_ready: bool = Field(
        ...,
        description="Operational readiness indicator for servicing inference requests.",
        examples=[True],
    )


class ModelInfoResponse(BaseModel):
    """API envelope for model metadata query."""

    success: bool = Field(default=True, description="Indicates request success.")
    data: ModelInfoData = Field(..., description="Model operational metadata.")
