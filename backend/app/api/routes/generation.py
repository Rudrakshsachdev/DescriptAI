"""REST API route handlers for description generation and model information."""

from fastapi import APIRouter, Depends, Request, status

from backend.app.core.exceptions import ModelNotReadyException
from backend.app.core.logging import get_logger
from backend.app.schemas.generation import (
    GenerationData,
    ProductDescriptionRequest,
    ProductDescriptionResponse,
)
from backend.app.schemas.model_info import ModelInfoResponse
from backend.app.services.generation import GenerationService

logger = get_logger("backend.routes.generation")

router = APIRouter()


def get_generation_service(request: Request) -> GenerationService:
    """Dependency extracting the operational GenerationService from application state."""
    service: GenerationService | None = getattr(request.app.state, "generation_service", None)
    if service is None or not service.is_ready:
        logger.warning("Generation request received while GenerationService is unready.")
        raise ModelNotReadyException("Description generation model is not loaded or ready.")
    return service


@router.post(
    "/generate",
    response_model=ProductDescriptionResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate Product Description",
    description=(
        "Transforms structured e-commerce product attributes (name, category, brand, features, "
        "specifications, audience) into a fluent, high-converting product description using the "
        "fine-tuned FLAN-T5 Seq2Seq model."
    ),
    tags=["Generation"],
    responses={
        200: {"description": "Product description successfully generated."},
        400: {"description": "Invalid generation configuration or malformed parameters."},
        422: {"description": "Validation error on required input fields."},
        500: {"description": "Internal model inference failure."},
        503: {"description": "Model weights unmounted or service not ready."},
        504: {"description": "Inference exceeded the configured request timeout."},
    },
)
async def generate_product_description(
    request_data: ProductDescriptionRequest,
    service: GenerationService = Depends(get_generation_service),
) -> ProductDescriptionResponse:
    """Receives validated product attributes and returns generated description with metadata."""
    logger.info("Received description generation request for product: '%s'", request_data.product_name)

    prediction = await service.generate_from_request_async(request_data)

    generation_data = GenerationData(
        generated_description=prediction.generated_description,
        model=prediction.metadata.get("model", "google/flan-t5-small"),
        generation_strategy=prediction.strategy,
        latency_ms=prediction.latency_ms,
        output_tokens=prediction.output_tokens,
        input_prompt=prediction.input_prompt,
        metadata=prediction.metadata,
    )

    return ProductDescriptionResponse(
        success=True,
        data=generation_data,
    )


@router.get(
    "/model/info",
    response_model=ModelInfoResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Model Operational Info",
    description=(
        "Returns safe operational metadata for the active Seq2Seq model, including architecture, "
        "parameter count, precision format, execution device, and default generation parameters."
    ),
    tags=["Model"],
    responses={
        200: {"description": "Operational metadata retrieved successfully."},
        503: {"description": "Model weights unmounted or service not ready."},
    },
)
async def get_model_info(
    service: GenerationService = Depends(get_generation_service),
) -> ModelInfoResponse:
    """Returns safe operational metadata for the serving deep learning model."""
    logger.debug("Handling GET /api/v1/model/info request.")
    info_data = service.get_model_info()
    return ModelInfoResponse(
        success=True,
        data=info_data,
    )
