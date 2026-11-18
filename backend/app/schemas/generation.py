"""Pydantic request and response schemas for product description generation."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class GenerationConfigSchema(BaseModel):
    """Client-configurable decoding parameters with strict bounds."""

    num_beams: Optional[int] = Field(
        default=2,
        ge=1,
        le=5,
        description="Number of beams for beam search decoding (1 to 5). Default is 2.",
        examples=[2],
    )
    max_new_tokens: Optional[int] = Field(
        default=128,
        ge=10,
        le=256,
        description="Maximum number of tokens to generate (10 to 256). Default is 128.",
        examples=[128],
    )
    min_new_tokens: Optional[int] = Field(
        default=10,
        ge=1,
        le=64,
        description="Minimum number of tokens to generate (1 to 64). Default is 10.",
        examples=[10],
    )
    no_repeat_ngram_size: Optional[int] = Field(
        default=3,
        ge=0,
        le=5,
        description="Size of n-grams that cannot be repeated (0 to 5). Default is 3.",
        examples=[3],
    )
    length_penalty: Optional[float] = Field(
        default=1.0,
        ge=0.0,
        le=3.0,
        description="Exponential penalty to length. Default is 1.0.",
        examples=[1.0],
    )
    do_sample: Optional[bool] = Field(
        default=False,
        description="Whether to use multinomial sampling instead of greedy/beam search.",
        examples=[False],
    )
    temperature: Optional[float] = Field(
        default=1.0,
        ge=0.1,
        le=2.0,
        description="Sampling temperature (0.1 to 2.0). Only used when do_sample=True.",
        examples=[1.0],
    )
    top_p: Optional[float] = Field(
        default=1.0,
        ge=0.1,
        le=1.0,
        description="Top-p nucleus sampling probability (0.1 to 1.0). Only used when do_sample=True.",
        examples=[1.0],
    )


class ProductDescriptionRequest(BaseModel):
    """Structured product input for AI description generation."""

    product_name: str = Field(
        ...,
        min_length=2,
        max_length=200,
        description="Title or primary commercial name of the product.",
        examples=["Hydrating Daily Face Moisturizer with Hyaluronic Acid"],
    )
    category: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Product retail category or hierarchy.",
        examples=["Beauty & Personal Care"],
    )
    brand: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Manufacturer or brand name.",
        examples=["DermaPure"],
    )
    features: Optional[List[str]] = Field(
        default=None,
        max_length=20,
        description="Key bullet points or standout features (up to 20 items).",
        examples=[
            "Infused with 2% pure hyaluronic acid and ceramides",
            "Provides 48-hour continuous hydration",
            "Non-comedogenic and fragrance-free",
        ],
    )
    specifications: Optional[Dict[str, str]] = Field(
        default=None,
        description="Key-value specifications (e.g., volume, skin type, finish).",
        examples=[{
            "volume": "50ml / 1.7 fl oz",
            "skin_type": "Sensitive, Dry, Combination",
            "formulation": "Lightweight Cream-Gel",
        }],
    )
    audience: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Target customer segment or intended use case.",
        examples=["Daily skincare routine for sensitive skin"],
    )
    generation_config: Optional[GenerationConfigSchema] = Field(
        default=None,
        description="Optional custom decoding parameters overriding defaults.",
    )

    @field_validator("product_name")
    @classmethod
    def validate_product_name(cls, v: str) -> str:
        """Ensures product name is not empty whitespace and trims extraneous spaces."""
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("product_name cannot be empty or blank whitespace.")
        if len(cleaned) < 2:
            raise ValueError("product_name must be at least 2 characters long.")
        return cleaned

    @field_validator("category", "brand", "audience")
    @classmethod
    def sanitize_optional_strings(cls, v: Optional[str]) -> Optional[str]:
        """Trims whitespace and converts empty strings to None."""
        if v is None:
            return None
        cleaned = v.strip()
        return cleaned if cleaned else None

    @field_validator("features")
    @classmethod
    def sanitize_features(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """Removes empty strings and trims features."""
        if v is None:
            return None
        cleaned = [item.strip() for item in v if item and item.strip()]
        return cleaned if cleaned else None

    @field_validator("specifications")
    @classmethod
    def validate_specifications(cls, v: Optional[Dict[str, str]]) -> Optional[Dict[str, str]]:
        """Validates specifications size and contents."""
        if v is None:
            return None
        if len(v) > 20:
            raise ValueError("specifications dictionary cannot contain more than 20 entries.")
        cleaned = {}
        for k, val in v.items():
            k_clean = str(k).strip()
            v_clean = str(val).strip()
            if k_clean and v_clean:
                cleaned[k_clean] = v_clean
        return cleaned if cleaned else None


class GenerationData(BaseModel):
    """Payload of successful description generation."""

    generated_description: str = Field(
        ...,
        description="Generated marketing and e-commerce product description.",
        examples=["A hydrating daily face moisturizer infused with pure hyaluronic acid..."],
    )
    model: str = Field(
        ...,
        description="Identifier of the serving model checkpoint.",
        examples=["google/flan-t5-small"],
    )
    generation_strategy: str = Field(
        ...,
        description="Decoding strategy employed during generation.",
        examples=["beam_search_2"],
    )
    latency_ms: float = Field(
        ...,
        description="Inference execution latency in milliseconds.",
        examples=[182.4],
    )
    output_tokens: int = Field(
        ...,
        description="Number of tokens generated by the Seq2Seq decoder.",
        examples=[45],
    )
    input_prompt: Optional[str] = Field(
        default=None,
        description="Canonical prompt fed to the encoder model.",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional generation telemetry (num_beams, input_token_count, etc.).",
    )


class ProductDescriptionResponse(BaseModel):
    """Consistent envelope for description generation responses."""

    success: bool = Field(default=True, description="Indicates successful request completion.")
    data: GenerationData = Field(..., description="Generation results and performance telemetry.")
