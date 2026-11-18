"""Generation service coordinating inference requests with the ML predictor."""

from typing import Any, Dict, List, Optional

from backend.app.core.config import get_settings
from backend.app.core.exceptions import ModelInferenceException, ModelNotReadyException
from backend.app.core.logging import get_logger
from descra.inference.predictor import DescriptionPredictor, GenerationConfig, PredictionOutput

logger = get_logger("backend.services.generation")


class GenerationService:
    """Service layer coordinating input transformation, model inference, and output formatting."""

    def __init__(self, predictor: Optional[DescriptionPredictor] = None):
        """Initializes service with an existing predictor instance."""
        self._predictor = predictor
        self._settings = get_settings()

    @property
    def predictor(self) -> Optional[DescriptionPredictor]:
        return self._predictor

    @predictor.setter
    def predictor(self, value: DescriptionPredictor) -> None:
        self._predictor = value

    @property
    def is_ready(self) -> bool:
        """Returns True if predictor is loaded and operational."""
        return self._predictor is not None and getattr(self._predictor, "model", None) is not None

    def generate(
        self,
        product_name: str,
        features: Optional[List[str]] = None,
        category: Optional[str] = None,
        brand: Optional[str] = None,
        num_beams: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
    ) -> PredictionOutput:
        """Processes a generation request through the fine-tuned Seq2Seq model.

        Args:
            product_name: Name or title of the product.
            features: List of bullet point specifications or features.
            category: Optional e-commerce category.
            brand: Optional brand name.
            num_beams: Optional beam search beam count (defaults to config: 2).
            max_new_tokens: Optional maximum output tokens (defaults to config: 128).

        Returns:
            PredictionOutput containing description, latency, token count, and metadata.

        Raises:
            ModelNotReadyException: If model is not loaded.
            ModelInferenceException: If an error occurs during model generation.
        """
        if not self.is_ready or self._predictor is None:
            logger.error("Inference rejected: ModelPredictor is not initialized.")
            raise ModelNotReadyException("Description generation model is not loaded or ready.")

        # Resolve decoding parameters
        beams = num_beams if num_beams is not None else self._settings.DEFAULT_NUM_BEAMS
        tokens = max_new_tokens if max_new_tokens is not None else self._settings.DEFAULT_MAX_NEW_TOKENS

        gen_config = GenerationConfig(
            num_beams=beams,
            max_new_tokens=tokens,
            min_new_tokens=self._settings.DEFAULT_MIN_NEW_TOKENS,
            no_repeat_ngram_size=self._settings.DEFAULT_NO_REPEAT_NGRAM_SIZE,
            length_penalty=self._settings.DEFAULT_LENGTH_PENALTY,
            early_stopping=True,
            do_sample=False,
        )

        try:
            logger.debug(
                "Dispatching inference for product: '%s' (beams=%d, max_tokens=%d)",
                product_name[:50],
                beams,
                tokens,
            )
            output = self._predictor.predict(
                product_name=product_name,
                features=features,
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
