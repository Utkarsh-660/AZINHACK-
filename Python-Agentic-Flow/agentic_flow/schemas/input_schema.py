"""Input schemas for the Context Ingestion Agent."""

from pathlib import Path
from typing import Any, List, Optional, Union
from pydantic import BaseModel, Field, model_validator
from PIL import Image
from agentic_flow.exceptions import InvalidInputError

# Supported raw image representations
ImageInputType = Union[str, bytes, Path, Image.Image]


class IngestionInput(BaseModel):
    """Public validated input schema for the Context Ingestion Agent."""

    model_config = {"arbitrary_types_allowed": True}

    text: Optional[str] = Field(
        default=None,
        description="Clinical text, audio transcription, or PDF-extracted text from the frontend."
    )
    images: List[ImageInputType] = Field(
        default_factory=list,
        description="List of images (local file path, Path object, raw bytes, URL string, or PIL Image)."
    )
    request_id: Optional[str] = Field(
        default=None,
        description="Optional request or tracing ID for correlation across multi-agent logs."
    )
    language: Optional[str] = Field(
        default="en",
        description="Preferred summary output language."
    )

    @model_validator(mode="after")
    def validate_non_empty_payload(self) -> "IngestionInput":
        """Ensure that at least one usable text or image source is provided."""
        has_text = bool(self.text and self.text.strip())
        has_images = bool(self.images and len(self.images) > 0)

        if not has_text and not has_images:
            raise InvalidInputError(
                "Invalid input: At least one non-empty text string or valid image must be provided."
            )
        
        # Clean text if provided
        if self.text is not None:
            cleaned = self.text.strip()
            self.text = cleaned if cleaned else None

        return self

