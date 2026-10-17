"""Core Context Ingestion Agent orchestration."""

import asyncio
from pathlib import Path
from typing import Any, List, Optional, Union
from PIL import Image

from agentic_flow.config import AgentConfig, get_default_config
from agentic_flow.exceptions import InvalidInputError
from agentic_flow.schemas.input_schema import IngestionInput
from agentic_flow.schemas.intermediate_schema import VisionExtractionResult
from agentic_flow.schemas.output_schema import IngestionOutput
from agentic_flow.services.gemini_client import GeminiClient
from agentic_flow.services.image_processor import ImageProcessor, ProcessedImage
from agentic_flow.services.summarizer import ClinicalSummarizer
from agentic_flow.services.vision_extractor import VisionExtractor


class ContextIngestionAgent:
    """Production-grade Multimodal Context Ingestion Agent.

    Routes clinical text and images across a two-model Gemini architecture:
    - Model A (Vision Extractor): gemini-3.8-flash for medical OCR, handwriting, and tables.
    - Model B (Clinical Summarizer): gemini-3.1-flash-lite for structured synthesis.
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        client: Optional[GeminiClient] = None,
        image_processor: Optional[ImageProcessor] = None,
        vision_extractor: Optional[VisionExtractor] = None,
        summarizer: Optional[ClinicalSummarizer] = None
    ):
        self.config = config or get_default_config()
        self.client = client or GeminiClient(self.config)
        self.image_processor = image_processor or ImageProcessor(self.config)
        self.vision_extractor = vision_extractor or VisionExtractor(self.config, self.client)
        self.summarizer = summarizer or ClinicalSummarizer(self.config, self.client)

    def ingest(
        self,
        input_data: Optional[IngestionInput] = None,
        *,
        text: Optional[str] = None,
        images: Optional[List[Union[str, bytes, Path, Image.Image]]] = None,
        request_id: Optional[str] = None,
        language: str = "en"
    ) -> IngestionOutput:
        """Synchronously process clinical text, images, or both into a validated IngestionOutput.

        Args:
            input_data: Pre-constructed IngestionInput model, or supply kwargs below.
            text: Optional clinical text, audio transcription, or PDF text.
            images: Optional list of file paths, bytes, PIL Images, URLs, or base64 strings.
            request_id: Optional tracing identifier.
            language: Target summary language (defaults to 'en').

        Returns:
            Validated IngestionOutput conforming to the target schema.

        Raises:
            InvalidInputError: If both text and images are empty or unparseable.
            ImageProcessingError: If image cannot be decoded, is oversized, or violates SSRF.
            ModelError: If model inference fails after bounded retries.
        """
        resolved_input = self._resolve_input(input_data, text, images, request_id, language)

        has_text = bool(resolved_input.text and resolved_input.text.strip())
        has_images = bool(resolved_input.images and len(resolved_input.images) > 0)

        # Case A: Text Only (Skip Vision Model completely)
        if has_text and not has_images:
            return self.summarizer.summarize(
                text=resolved_input.text,
                vision_extractions=None,
                request_id=resolved_input.request_id,
                language=resolved_input.language
            )

        # Process and validate images for Case B and Case C
        processed_images = self._process_images(resolved_input.images)

        # Extract information using Model A (Vision Extractor)
        vision_extractions: List[VisionExtractionResult] = []
        for img in processed_images:
            extraction = self.vision_extractor.extract(img)
            vision_extractions.append(extraction)

        # Case B: Image Only
        if not has_text and has_images:
            return self.summarizer.summarize(
                text=None,
                vision_extractions=vision_extractions,
                request_id=resolved_input.request_id,
                language=resolved_input.language
            )

        # Case C: Combined Text and Image
        return self.summarizer.summarize(
            text=resolved_input.text,
            vision_extractions=vision_extractions,
            request_id=resolved_input.request_id,
            language=resolved_input.language
        )

    async def aingest(
        self,
        input_data: Optional[IngestionInput] = None,
        *,
        text: Optional[str] = None,
        images: Optional[List[Union[str, bytes, Path, Image.Image]]] = None,
        request_id: Optional[str] = None,
        language: str = "en"
    ) -> IngestionOutput:
        """Asynchronously process clinical text, images, or both into a validated IngestionOutput."""
        resolved_input = self._resolve_input(input_data, text, images, request_id, language)

        has_text = bool(resolved_input.text and resolved_input.text.strip())
        has_images = bool(resolved_input.images and len(resolved_input.images) > 0)

        # Case A: Text Only (Skip Vision Model completely)
        if has_text and not has_images:
            return await self.summarizer.asummarize(
                text=resolved_input.text,
                vision_extractions=None,
                request_id=resolved_input.request_id,
                language=resolved_input.language
            )

        # Process and validate images off the event loop
        processed_images = await asyncio.to_thread(self._process_images, resolved_input.images)

        # Extract information using Model A (Vision Extractor)
        vision_extractions: List[VisionExtractionResult] = []
        for img in processed_images:
            extraction = await self.vision_extractor.aextract(img)
            vision_extractions.append(extraction)

        # Case B: Image Only
        if not has_text and has_images:
            return await self.summarizer.asummarize(
                text=None,
                vision_extractions=vision_extractions,
                request_id=resolved_input.request_id,
                language=resolved_input.language
            )

        # Case C: Combined Text and Image
        return await self.summarizer.asummarize(
            text=resolved_input.text,
            vision_extractions=vision_extractions,
            request_id=resolved_input.request_id,
            language=resolved_input.language
        )

    def _resolve_input(
        self,
        input_data: Optional[Union[IngestionInput, str]],
        text: Optional[str],
        images: Optional[List[Any]],
        request_id: Optional[str],
        language: str
    ) -> IngestionInput:
        """Resolve and validate raw args into a verified IngestionInput model."""
        if isinstance(input_data, str):
            return IngestionInput(
                text=input_data,
                images=images or [],
                request_id=request_id,
                language=language
            )
        if isinstance(input_data, IngestionInput):
            return input_data

        return IngestionInput(
            text=text,
            images=images or [],
            request_id=request_id,
            language=language
        )

    def _process_images(self, raw_images: List[Any]) -> List[ProcessedImage]:
        """Decode and validate each supplied image."""
        processed: List[ProcessedImage] = []
        for index, item in enumerate(raw_images, start=1):
            proc_img = self.image_processor.process(item, source_index=index)
            processed.append(proc_img)
        return processed

