"""Domain exceptions for the Context Ingestion Agent."""

class AgenticFlowError(Exception):
    """Base exception for all agentic flow errors."""
    pass


class ConfigurationError(AgenticFlowError):
    """Raised when configuration or required environment variables are invalid or missing."""
    pass


class InvalidInputError(AgenticFlowError):
    """Raised when the caller supplies invalid, empty, or unparseable input."""
    pass


class ImageProcessingError(AgenticFlowError):
    """Raised when an image cannot be read, decoded, or validated."""
    pass


class SSRFSecurityError(ImageProcessingError):
    """Raised when an image URL resolves to a prohibited host or private IP address."""
    pass


class ModelError(AgenticFlowError):
    """Base exception for errors during model inference."""
    pass


class ModelAuthenticationError(ModelError):
    """Raised when model API authentication fails (non-retryable)."""
    pass


class ModelRateLimitError(ModelError):
    """Raised when model API rate limit or quota is exceeded (transient)."""
    pass


class ModelTimeoutError(ModelError):
    """Raised when model API request times out (transient)."""
    pass


class ModelOutputValidationError(ModelError):
    """Raised when model response cannot be parsed or validated against the target schema."""
    pass

