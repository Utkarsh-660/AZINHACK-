"""Pytest fixtures and deterministic synthetic mocks for Context Ingestion Agent tests."""

import io
from typing import Any, Dict, List, Optional, Type
import pytest
from PIL import Image, ImageDraw
from pydantic import BaseModel

from agentic_flow.config import AgentConfig
from agentic_flow.schemas import (
    ClinicalSummary,
    DocumentInfo,
    ExtractionStatus,
    FactCategory,
    IngestionOutput,
    IntermediateFact,
    IntermediateMedication,
    ItemConfidence,
    KeyFact,
    MedicationItem,
    SourceAttribution,
    UncertaintyItem,
    VisionExtractionResult,
)


@pytest.fixture
def test_config() -> AgentConfig:
    """Return a test configuration with deterministic parameters."""
    return AgentConfig(
        gemini_api_key="synthetic_test_key_for_mocking_only",
        vision_model_id="gemini-3.8-flash",
        summary_model_id="gemini-3.1-flash-lite",
        max_image_bytes=5 * 1024 * 1024,
        max_image_dimension=2048,
        max_retries=2,
        initial_retry_delay=0.01
    )


@pytest.fixture
def synthetic_image_bytes() -> bytes:
    """Generate a valid, deterministic synthetic JPEG image in memory."""
    img = Image.new("RGB", (200, 100), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), "Rx: Metformin 500mg", fill=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def synthetic_png_bytes() -> bytes:
    """Generate a valid synthetic PNG image in memory."""
    img = Image.new("RGB", (150, 150), color=(240, 240, 240))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def mock_vision_extraction() -> VisionExtractionResult:
    """Return a deterministic VisionExtractionResult."""
    return VisionExtractionResult(
        image_index=1,
        document_type="prescription",
        content_type="mixed",
        document_date="2026-10-15",
        provider_or_facility="Synthetic Clinic",
        raw_extracted_text="Rx: Metformin 500mg daily. Amox... [illegible dose]",
        facts=[
            IntermediateFact(
                category="SYMPTOM",
                name="Fatigue",
                value="Documented in header note",
                location="top margin",
                is_uncertain=False
            )
        ],
        medications=[
            IntermediateMedication(
                name="Metformin",
                dosage="500mg",
                route="oral",
                frequency="once daily",
                duration="30 days",
                instructions="after food",
                location="line 1",
                is_uncertain=False
            ),
            IntermediateMedication(
                name="Amoxicillin (?)",
                dosage=None,
                route="oral",
                frequency=None,
                duration="5 days",
                instructions=None,
                location="line 2 handwritten",
                is_uncertain=True,
                uncertainty_reason="Handwriting dosage is illegible"
            )
        ],
        unreadable_or_ambiguous_sections=["Line 2 handwritten dosage illegible"]
    )


@pytest.fixture
def mock_summary_output() -> IngestionOutput:
    """Return a deterministic IngestionOutput with synthetic data."""
    return IngestionOutput(
        schema_version="1.0.0",
        request_id="test-req-001",
        summary=ClinicalSummary(
            brief_summary="Synthetic patient record indicating daily Metformin prescription and an ambiguous antibiotic line.",
            detailed_findings="Metformin 500mg was verified. A handwritten prescription for an antibiotic had an illegible dosage.",
            reported_by="Synthetic Clinic"
        ),
        document_info=DocumentInfo(
            document_type="prescription",
            sources=["image"],
            detected_content_types=["mixed"],
            language="en"
        ),
        key_facts=[
            KeyFact(
                fact_id="fact_01",
                category=FactCategory.SYMPTOM,
                name="Fatigue",
                value="Present",
                source=SourceAttribution(source_type="image", source_name="rx.jpg"),
                status=ItemConfidence.VERIFIED
            )
        ],
        medications=[
            MedicationItem(
                name="Metformin",
                dosage="500mg",
                route="oral",
                frequency="once daily",
                duration="30 days",
                instructions="after food",
                source=SourceAttribution(source_type="image", source_name="rx.jpg", section_or_location="line 1"),
                status=ItemConfidence.VERIFIED
            ),
            MedicationItem(
                name="Amoxicillin (?)",
                dosage=None,
                route="oral",
                frequency=None,
                duration="5 days",
                instructions=None,
                source=SourceAttribution(source_type="image", source_name="rx.jpg", section_or_location="line 2"),
                status=ItemConfidence.UNCERTAIN,
                uncertainty_note="Handwriting is illegible for dosage"
            )
        ],
        uncertainties=[
            UncertaintyItem(
                description="Line 2 handwritten dosage is illegible.",
                category="unreadable_handwriting"
            )
        ],
        overall_extraction_status=ExtractionStatus.NEEDS_REVIEW,
        requires_human_review=True
    )


class MockGeminiClient:
    """Mock GeminiClient that tracks model calls and returns configured structured responses."""

    def __init__(
        self,
        config: AgentConfig,
        vision_response: Optional[VisionExtractionResult] = None,
        summary_response: Optional[IngestionOutput] = None
    ):
        self.config = config
        self.vision_response = vision_response
        self.summary_response = summary_response
        self.called_models: List[str] = []
        self.call_history: List[Dict[str, Any]] = []
        self.fail_transient_count = 0
        self.fail_permanent = False

    def generate_structured(
        self,
        model_id: str,
        contents: List[Any],
        target_schema: Type[BaseModel],
        system_instruction: Optional[str] = None
    ) -> Any:
        self.called_models.append(model_id)
        self.call_history.append({
            "model_id": model_id,
            "target_schema": target_schema,
            "contents": contents
        })

        if self.fail_permanent:
            from agentic_flow.exceptions import ModelAuthenticationError
            raise ModelAuthenticationError("Authentication failure: Invalid or unauthorized Gemini API key.")

        if self.fail_transient_count > 0:
            self.fail_transient_count -= 1
            from agentic_flow.exceptions import ModelRateLimitError
            raise ModelRateLimitError("Quota limit hit, retryable.")

        if target_schema == VisionExtractionResult:
            return self.vision_response or VisionExtractionResult(image_index=1)
        if target_schema == IngestionOutput:
            return self.summary_response or IngestionOutput(
                summary=ClinicalSummary(brief_summary="Default", detailed_findings="Default")
            )
        return target_schema()

    async def async_generate_structured(
        self,
        model_id: str,
        contents: List[Any],
        target_schema: Type[BaseModel],
        system_instruction: Optional[str] = None
    ) -> Any:
        return self.generate_structured(model_id, contents, target_schema, system_instruction)

