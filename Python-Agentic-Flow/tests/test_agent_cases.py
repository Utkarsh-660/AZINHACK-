"""Unit tests for the ContextIngestionAgent execution workflow (Cases A, B, C, D)."""

import pytest

from agentic_flow import ContextIngestionAgent, IngestionInput, IngestionOutput
from agentic_flow.config import AgentConfig
from agentic_flow.exceptions import InvalidInputError
from agentic_flow.schemas import ItemConfidence, VisionExtractionResult
from agentic_flow.services import ClinicalSummarizer, ImageProcessor, VisionExtractor
from tests.conftest import MockGeminiClient


def test_case_a_text_only_skips_vision_model(
    test_config: AgentConfig,
    mock_summary_output: IngestionOutput
):
    """Case A: Text only MUST call summarizer and NEVER call the vision model."""
    mock_client = MockGeminiClient(test_config, summary_response=mock_summary_output)
    agent = ContextIngestionAgent(
        config=test_config,
        client=mock_client,
        vision_extractor=VisionExtractor(test_config, mock_client),
        summarizer=ClinicalSummarizer(test_config, mock_client)
    )

    result = agent.ingest(text="Patient presents with persistent cough for 4 days.")

    assert isinstance(result, IngestionOutput)
    # Crucial check: Vision model ID was NEVER called
    assert test_config.vision_model_id not in mock_client.called_models
    # Summarization model was called
    assert test_config.summary_model_id in mock_client.called_models
    assert len(mock_client.called_models) == 1


def test_case_b_image_only_calls_both_models(
    test_config: AgentConfig,
    synthetic_image_bytes: bytes,
    mock_vision_extraction: VisionExtractionResult,
    mock_summary_output: IngestionOutput
):
    """Case B: Image only calls vision model then summarization model."""
    mock_client = MockGeminiClient(
        test_config,
        vision_response=mock_vision_extraction,
        summary_response=mock_summary_output
    )
    agent = ContextIngestionAgent(
        config=test_config,
        client=mock_client,
        vision_extractor=VisionExtractor(test_config, mock_client),
        summarizer=ClinicalSummarizer(test_config, mock_client)
    )

    result = agent.ingest(images=[synthetic_image_bytes])

    assert isinstance(result, IngestionOutput)
    # Both models called in sequence
    assert mock_client.called_models == [
        test_config.vision_model_id,
        test_config.summary_model_id
    ]


def test_case_c_combined_text_and_image(
    test_config: AgentConfig,
    synthetic_image_bytes: bytes,
    mock_vision_extraction: VisionExtractionResult,
    mock_summary_output: IngestionOutput
):
    """Case C: Combined text and image preserves both sources."""
    mock_client = MockGeminiClient(
        test_config,
        vision_response=mock_vision_extraction,
        summary_response=mock_summary_output
    )
    agent = ContextIngestionAgent(
        config=test_config,
        client=mock_client,
        vision_extractor=VisionExtractor(test_config, mock_client),
        summarizer=ClinicalSummarizer(test_config, mock_client)
    )

    input_payload = IngestionInput(
        text="Patient transcript: Complains of fatigue and dry cough.",
        images=[synthetic_image_bytes],
        request_id="combo-req-001"
    )

    result = agent.ingest(input_payload)

    assert isinstance(result, IngestionOutput)
    assert mock_client.called_models == [
        test_config.vision_model_id,
        test_config.summary_model_id
    ]


def test_case_d_empty_input_rejected_zero_model_calls(test_config: AgentConfig):
    """Case D: Empty or whitespace input is rejected with zero model calls."""
    mock_client = MockGeminiClient(test_config)
    agent = ContextIngestionAgent(
        config=test_config,
        client=mock_client,
        vision_extractor=VisionExtractor(test_config, mock_client),
        summarizer=ClinicalSummarizer(test_config, mock_client)
    )

    with pytest.raises(InvalidInputError):
        agent.ingest(text="   ", images=[])

    with pytest.raises(InvalidInputError):
        agent.ingest()

    # Zero model calls were made
    assert len(mock_client.called_models) == 0


@pytest.mark.asyncio
async def test_async_interface_matches_sync(
    test_config: AgentConfig,
    synthetic_image_bytes: bytes,
    mock_vision_extraction: VisionExtractionResult,
    mock_summary_output: IngestionOutput
):
    """Verify async .aingest() produces identical validated output."""
    mock_client = MockGeminiClient(
        test_config,
        vision_response=mock_vision_extraction,
        summary_response=mock_summary_output
    )
    agent = ContextIngestionAgent(
        config=test_config,
        client=mock_client,
        vision_extractor=VisionExtractor(test_config, mock_client),
        summarizer=ClinicalSummarizer(test_config, mock_client)
    )

    result = await agent.aingest(
        text="Sample text",
        images=[synthetic_image_bytes],
        request_id="async-req-001"
    )

    assert isinstance(result, IngestionOutput)
    assert result.requires_human_review is True
    assert mock_client.called_models == [
        test_config.vision_model_id,
        test_config.summary_model_id
    ]


def test_handwriting_uncertainty_flags_human_review(
    test_config: AgentConfig,
    synthetic_image_bytes: bytes,
    mock_vision_extraction: VisionExtractionResult,
    mock_summary_output: IngestionOutput
):
    """Verify that uncertain handwriting lines result in requires_human_review=True."""
    mock_client = MockGeminiClient(
        test_config,
        vision_response=mock_vision_extraction,
        summary_response=mock_summary_output
    )
    agent = ContextIngestionAgent(
        config=test_config,
        client=mock_client,
        vision_extractor=VisionExtractor(test_config, mock_client),
        summarizer=ClinicalSummarizer(test_config, mock_client)
    )

    result = agent.ingest(images=[synthetic_image_bytes])

    assert result.requires_human_review is True
    assert any(m.status == ItemConfidence.UNCERTAIN for m in result.medications)
    assert len(result.uncertainties) > 0


def test_positional_string_input_to_agent(
    test_config: AgentConfig,
    mock_summary_output: IngestionOutput
):
    """Verify that calling agent.ingest('some text') positionally succeeds."""
    mock_client = MockGeminiClient(test_config, summary_response=mock_summary_output)
    agent = ContextIngestionAgent(
        config=test_config,
        client=mock_client,
        summarizer=ClinicalSummarizer(test_config, mock_client)
    )

    result = agent.ingest("Direct patient symptom report")
    assert isinstance(result, IngestionOutput)
    assert len(mock_client.called_models) == 1
    assert test_config.summary_model_id in mock_client.called_models


def test_enum_case_insensitivity():
    """Verify that lowercase status strings match uppercase enum members."""
    from agentic_flow.schemas import ExtractionStatus, FactCategory, ItemConfidence

    assert ExtractionStatus("complete") == ExtractionStatus.COMPLETE
    assert ExtractionStatus("needs_review") == ExtractionStatus.NEEDS_REVIEW
    assert FactCategory("lab_result") == FactCategory.LAB_RESULT
    assert FactCategory("symptom") == FactCategory.SYMPTOM
    assert ItemConfidence("verified") == ItemConfidence.VERIFIED
    assert ItemConfidence("uncertain") == ItemConfidence.UNCERTAIN

