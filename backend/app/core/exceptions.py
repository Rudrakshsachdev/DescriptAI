"""Custom domain exceptions for Descript AI Backend."""

from typing import Any, Dict, Optional


class DescriptAIException(Exception):
    """Base exception for all backend errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ModelNotReadyException(DescriptAIException):
    """Raised when an inference request arrives before the model is initialized."""
    pass


class ModelInferenceException(DescriptAIException):
    """Raised when model inference fails during generation."""
    pass


class InvalidInputException(DescriptAIException):
    """Raised when input attributes fail semantic validation."""
    pass
