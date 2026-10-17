"""Configuration management for the Context Ingestion Agent."""

import os
from pathlib import Path
from typing import Set
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from agentic_flow.exceptions import ConfigurationError

# Load environment variables from .env if present
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


class AgentConfig(BaseModel):
    """Configuration settings with strict validation and safe defaults."""
    
    gemini_api_key: str = Field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY", ""),
        description="Gemini API Key from environment or .env"
    )
    vision_model_id: str = Field(
        default_factory=lambda: os.getenv("GEMINI_VISION_MODEL", "gemini-3.8-flash"),
        description="Configurable Gemini Vision Model ID"
    )
    summary_model_id: str = Field(
        default_factory=lambda: os.getenv("GEMINI_SUMMARY_MODEL", "gemini-3.1-flash-lite"),
        description="Configurable Gemini Lightweight Summarization Model ID"
    )
    
    # Image constraints
    max_image_bytes: int = Field(
        default=15 * 1024 * 1024,
        ge=128,
        le=50 * 1024 * 1024,
        description="Maximum image size in bytes (128B to 50MB)"
    )
    max_image_dimension: int = Field(
        default=4096,
        ge=32,
        le=8192,
        description="Maximum width/height in pixels (32px to 8192px)"
    )
    allowed_mime_types: Set[str] = Field(
        default_factory=lambda: {"image/jpeg", "image/png", "image/webp"},
        description="Allowed image MIME types"
    )
    
    # Network and retry limits
    request_timeout: float = Field(
        default=60.0,
        ge=1.0,
        le=300.0,
        description="HTTP request timeout in seconds (1.0s to 300.0s)"
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum retries for transient errors (0 to 10)"
    )
    initial_retry_delay: float = Field(
        default=1.0,
        ge=0.01,
        le=30.0,
        description="Initial backoff delay in seconds (0.01s to 30.0s)"
    )

    def validate_api_key(self) -> None:
        """Validate that an API key is present before attempting live model requests."""
        if not self.gemini_api_key or not self.gemini_api_key.strip():
            raise ConfigurationError(
                "Missing GEMINI_API_KEY. Please provide a valid Gemini API key in the environment or .env file."
            )

    def mask_key(self) -> str:
        """Return a safe masked representation of the API key without revealing secrets."""
        if not self.gemini_api_key:
            return "<unset>"
        if len(self.gemini_api_key) <= 8:
            return "***"
        return f"{self.gemini_api_key[:4]}...{self.gemini_api_key[-4:]}"


def get_default_config() -> AgentConfig:
    """Return an AgentConfig loaded from the current environment."""
    return AgentConfig()

