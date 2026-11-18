"""Generation service coordinating inference requests with the ML predictor."""

from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.app.core.concurrency import InferenceConcurrencyLimiter
from backend.app.core.config import get_settings
from backend.app.core.exceptions import (
    InferenceTimeoutException,
    ModelInferenceException,
    ModelNotReadyException,
    ServiceOverloadedException,
)
from backend.app.core.logging import get_logger
from backend.app.schemas.generation import GenerationConfigSchema, ProductDescriptionRequest
from backend.app.schemas.model_info import ModelInfoData
from descra.inference.predictor import DescriptionPredictor, GenerationConfig, PredictionOutput

logger = get_logger("backend.services.generation")


class GenerationService:
    """Service layer coordinating input transformation, model inference, and output formatting."""

    def __init__(self, predictor: Optional[DescriptionPredictor] = None):
        """Initializes service with an existing predictor instance."""
        self._predictor = predictor
        self._settings = get_settings()
        self._inference_limiter = InferenceConcurrencyLimiter(
            max_concurrent=self._settings.MAX_CONCURRENT_INFERENCE_REQUESTS,
            capacity_wait_seconds=self._settings.INFERENCE_CAPACITY_WAIT_SECONDS,
            inference_timeout_seconds=self._settings.INFERENCE_TIMEOUT_SECONDS,
        )

    @property
    def predictor(self) -> Optional[DescriptionPredictor]:
        return self._predictor

    async def wait_for_inference_idle(self, timeout: Optional[float] = None) -> bool:
        """Waits for active process-local inference before application shutdown."""
        return await self._inference_limiter.wait_for_idle(timeout=timeout)

    @predictor.setter
    def predictor(self, value: DescriptionPredictor) -> None:
        self._predictor = value

    @property
    def is_ready(self) -> bool:
        """Returns True if predictor is loaded and operational."""
        return self._predictor is not None and getattr(self._predictor, "model", None) is not None

    def _build_features_list(
        self,
        features: Optional[List[str]] = None,
        specifications: Optional[Dict[str, str]] = None,
        audience: Optional[str] = None,
    ) -> Optional[List[str]]:
        """Consolidates features, specifications, and audience into a canonical feature list."""
        combined: List[str] = []

        if features:
            for feat in features:
                f_clean = feat.strip()
                if f_clean:
                    combined.append(f_clean)

        if specifications:
            for key, val in specifications.items():
                k_clean = str(key).strip().replace("_", " ").title()
                v_clean = str(val).strip()
                if k_clean and v_clean:
                    combined.append(f"{k_clean}: {v_clean}")

        if audience and audience.strip():
            combined.append(f"Target Audience: {audience.strip()}")

        return combined if combined else None

    def _resolve_generation_config(
        self,
        config_override: Optional[GenerationConfigSchema] = None,
        num_beams: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
    ) -> GenerationConfig:
        """Resolves decoding configuration with precedence: explicit override > args > settings."""
        if config_override is not None:
            return GenerationConfig(
                num_beams=config_override.num_beams or self._settings.DEFAULT_NUM_BEAMS,
                max_new_tokens=config_override.max_new_tokens or self._settings.DEFAULT_MAX_NEW_TOKENS,
                min_new_tokens=config_override.min_new_tokens or self._settings.DEFAULT_MIN_NEW_TOKENS,
                no_repeat_ngram_size=config_override.no_repeat_ngram_size or self._settings.DEFAULT_NO_REPEAT_NGRAM_SIZE,
                length_penalty=config_override.length_penalty or self._settings.DEFAULT_LENGTH_PENALTY,
                early_stopping=True,
                do_sample=config_override.do_sample or False,
                temperature=config_override.temperature or 1.0,
                top_p=config_override.top_p or 1.0,
            )

        # Fallback to individual parameter overrides or application defaults
        beams = num_beams if num_beams is not None else self._settings.DEFAULT_NUM_BEAMS
        tokens = max_new_tokens if max_new_tokens is not None else self._settings.DEFAULT_MAX_NEW_TOKENS

        return GenerationConfig(
            num_beams=beams,
            max_new_tokens=tokens,
            min_new_tokens=self._settings.DEFAULT_MIN_NEW_TOKENS,
            no_repeat_ngram_size=self._settings.DEFAULT_NO_REPEAT_NGRAM_SIZE,
            length_penalty=self._settings.DEFAULT_LENGTH_PENALTY,
            early_stopping=True,
            do_sample=False,
        )

    def generate(
        self,
        product_name: str,
        features: Optional[List[str]] = None,
        category: Optional[str] = None,
        brand: Optional[str] = None,
        num_beams: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        specifications: Optional[Dict[str, str]] = None,
        audience: Optional[str] = None,
        config: Optional[GenerationConfigSchema] = None,
    ) -> PredictionOutput:
        """Processes a generation request through the fine-tuned Seq2Seq model.

        Supports both granular parameter calls and structured request dispatch.
        """
        if not self.is_ready or self._predictor is None:
            logger.error("Inference rejected: ModelPredictor is not initialized.")
            raise ModelNotReadyException("Description generation model is not loaded or ready.")

        gen_config = self._resolve_generation_config(
            config_override=config,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
        )

        all_features = self._build_features_list(
            features=features,
            specifications=specifications,
            audience=audience,
        )

        try:
            logger.debug(
                "Dispatching inference for product: '%s' (beams=%d, max_tokens=%d, do_sample=%s)",
                product_name[:50],
                gen_config.num_beams,
                gen_config.max_new_tokens,
                gen_config.do_sample,
            )
            output = self._predictor.predict(
                product_name=product_name,
                features=all_features,
                category=category,
                brand=brand,
                config=gen_config,
            )
            logger.info(
                "Inference complete: %d tokens generated in %.1f ms (%s)",
                output.output_tokens,
                output.latency_ms,
                output.strategy,
            )
            return output

        except Exception as e:
            logger.error("Model inference failed unexpectedly: %s", e, exc_info=True)
            raise ModelInferenceException(
                message="An error occurred while generating the product description.",
                details={"error_type": type(e).__name__},
            ) from e

    async def generate_async(
        self,
        product_name: str,
        features: Optional[List[str]] = None,
        category: Optional[str] = None,
        brand: Optional[str] = None,
        num_beams: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        specifications: Optional[Dict[str, str]] = None,
        audience: Optional[str] = None,
        config: Optional[GenerationConfigSchema] = None,
    ) -> PredictionOutput:
        """Runs synchronous model inference without blocking or overloading the event loop."""
        if not self.is_ready or self._predictor is None:
            logger.error("Inference rejected: ModelPredictor is not initialized.")
            raise ModelNotReadyException("Description generation model is not loaded or ready.")

        gen_config = self._resolve_generation_config(
            config_override=config,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
        )
        all_features = self._build_features_list(
            features=features,
            specifications=specifications,
            audience=audience,
        )

        logger.debug(
            "Queued description inference for product: '%s'",
            product_name[:50],
        )
        try:
            output = await self._inference_limiter.run(
                self._predictor.predict,
                product_name=product_name,
                features=all_features,
                category=category,
                brand=brand,
                config=gen_config,
            )
        except (ModelNotReadyException, InferenceTimeoutException, ServiceOverloadedException):
            raise
        except Exception as exc:
            logger.error("Model inference failed unexpectedly: %s", exc, exc_info=True)
            raise ModelInferenceException(
                message="An error occurred while generating the product description.",
                details={"error_type": type(exc).__name__},
            ) from exc
        logger.info(
            "Inference complete: %d tokens generated in %.1f ms (%s)",
            output.output_tokens,
            output.latency_ms,
            output.strategy,
        )
        return output

    async def generate_from_request_async(
        self,
        request: ProductDescriptionRequest,
    ) -> PredictionOutput:
        """High-level asynchronous generator preserving the existing request contract."""
        return await self.generate_async(
            product_name=request.product_name,
            features=request.features,
            category=request.category,
            brand=request.brand,
            specifications=request.specifications,
            audience=request.audience,
            config=request.generation_config,
        )

    def generate_from_request(self, request: ProductDescriptionRequest) -> PredictionOutput:
        """High-level generator parsing validated ProductDescriptionRequest."""
        return self.generate(
            product_name=request.product_name,
            features=request.features,
            category=request.category,
            brand=request.brand,
            specifications=request.specifications,
            audience=request.audience,
            config=request.generation_config,
        )

    def get_model_info(self) -> ModelInfoData:
        """Retrieves safe, sanitized operational model metadata."""
        if not self.is_ready or self._predictor is None:
            raise ModelNotReadyException("Model is not loaded or ready.")

        predictor = self._predictor

        # Sanitize checkpoint path (do not expose machine-specific drive letters or user folders)
        raw_checkpoint = str(getattr(predictor, "checkpoint_path", self._settings.MODEL_CHECKPOINT_PATH))
        norm_checkpoint = raw_checkpoint.replace("\\", "/")
        if "checkpoints/" in norm_checkpoint:
            clean_checkpoint = "checkpoints/" + norm_checkpoint.split("checkpoints/", 1)[1]
        else:
            clean_checkpoint = Path(norm_checkpoint).name

        # Calculate parameters safely
        param_count = 76961152
        precision = "bfloat16" if self._settings.USE_HALF_PRECISION else "float32"

        if hasattr(predictor, "model") and predictor.model is not None:
            try:
                param_count = sum(p.numel() for p in predictor.model.parameters())
                first_param = next(predictor.model.parameters(), None)
                if first_param is not None:
                    precision = str(first_param.dtype).replace("torch.", "")
            except Exception:
                pass

        device_name = str(getattr(predictor, "device", self._settings.DEVICE))

        default_cfg = {
            "num_beams": self._settings.DEFAULT_NUM_BEAMS,
            "max_new_tokens": self._settings.DEFAULT_MAX_NEW_TOKENS,
            "min_new_tokens": self._settings.DEFAULT_MIN_NEW_TOKENS,
            "no_repeat_ngram_size": self._settings.DEFAULT_NO_REPEAT_NGRAM_SIZE,
            "length_penalty": self._settings.DEFAULT_LENGTH_PENALTY,
            "early_stopping": True,
            "do_sample": False,
        }

        return ModelInfoData(
            model_id=self._settings.MODEL_FALLBACK_ID,
            architecture="Encoder-Decoder Transformer (Seq2Seq)",
            checkpoint=clean_checkpoint,
            device=device_name,
            total_parameters=param_count,
            precision=precision,
            supported_strategies=["beam_search", "greedy", "sampling"],
            default_config=default_cfg,
            is_ready=True,
        )
