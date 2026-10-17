"""Unit tests for Pydantic input and output schemas."""

import pytest
from pydantic import ValidationError

from agentic_flow.exceptions import InvalidInputError
from agentic_flow.schemas import (
    AllergyItem,
    ClinicalSummary,
    ConflictItem,
    DocumentInfo,
    ExtractionStatus,
    FactCategory,
    IngestionInput,
    IngestionOutput,
    ItemConfidence,
    KeyFact,
    MedicationItem,
    MissingInfoItem,
    SourceAttribution,
    UncertaintyItem,
)


def test_ingestion_input_valid_text():
    """Verify input accepts non-empty text."""
    inp = IngestionInput(text="Patient has mild fever", request_id="req-1")
    assert inp.text == "Patient has mild fever"
    assert inp.images == []
    assert inp.request_id == "req-1"


def test_ingestion_input_valid_images():
    """Verify input accepts image list."""
    inp = IngestionInput(images=[b"fake_bytes"])
    assert len(inp.images) == 1
    assert inp.text is None


def test_ingestion_input_rejects_empty_payload():
    """Verify empty text and empty images are rejected with InvalidInputError."""
    with pytest.raises(InvalidInputError):
        IngestionInput()

    with pytest.raises(InvalidInputError):
        IngestionInput(text="   ", images=[])


def test_ingestion_output_serialization(mock_summary_output: IngestionOutput):
    """Verify IngestionOutput serializes cleanly to JSON and parses back identically."""
    json_str = mock_summary_output.model_dump_json(indent=2)
    assert "Metformin" in json_str
    assert "Amoxicillin (?)" in json_str

    parsed = IngestionOutput.model_validate_json(json_str)
    assert parsed.schema_version == "1.0.0"
    assert parsed.requires_human_review is True
    assert len(parsed.medications) == 2
    assert parsed.medications[1].status == ItemConfidence.UNCERTAIN


def test_conflict_item_structure():
    """Verify ConflictItem fields and provenance capture."""
    conflict = ConflictItem(
        field_or_fact="Medication Dosage",
        value_source_a="Transcript reports 250mg",
        value_source_b="Prescription scan reports 500mg",
        description="Discrepancy between spoken transcript and written script."
    )
    assert "250mg" in conflict.value_source_a
    assert "500mg" in conflict.value_source_b


def test_missing_info_item_structure():
    """Verify MissingInfoItem captures critical fields."""
    missing = MissingInfoItem(
        field_name="Antibiotic Duration",
        importance="critical",
        reason="Required to prevent incomplete treatment course."
    )
    assert missing.importance == "critical"
    assert missing.field_name == "Antibiotic Duration"

