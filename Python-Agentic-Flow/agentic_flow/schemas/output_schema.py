"""Final Pydantic output schemas for the Context Ingestion Agent."""

from enum import Enum
from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class ExtractionStatus(str, Enum):
    """Overall status of clinical extraction."""
    COMPLETE = "COMPLETE"
    PARTIALLY_EXTRACTED = "PARTIALLY_EXTRACTED"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    FAILED = "FAILED"

    @classmethod
    def _missing_(cls, value: object):
        if isinstance(value, str):
            val_upper = value.upper()
            for member in cls:
                if member.value == val_upper:
                    return member
        return None


class FactCategory(str, Enum):
    """Categories for medical and laboratory facts."""
    LAB_RESULT = "LAB_RESULT"
    CLINICAL_OBSERVATION = "CLINICAL_OBSERVATION"
    SYMPTOM = "SYMPTOM"
    DOCUMENTED_DIAGNOSIS = "DOCUMENTED_DIAGNOSIS"
    PROCEDURE = "PROCEDURE"
    VITAL_SIGN = "VITAL_SIGN"
    OTHER = "OTHER"

    @classmethod
    def _missing_(cls, value: object):
        if isinstance(value, str):
            val_upper = value.upper()
            for member in cls:
                if member.value == val_upper:
                    return member
        return None


class ItemConfidence(str, Enum):
    """Confidence level of an individual extracted item."""
    VERIFIED = "VERIFIED"
    UNCERTAIN = "UNCERTAIN"
    AMBIGUOUS = "AMBIGUOUS"

    @classmethod
    def _missing_(cls, value: object):
        if isinstance(value, str):
            val_upper = value.upper()
            for member in cls:
                if member.value == val_upper:
                    return member
        return None


class SourceAttribution(BaseModel):
    """Provenance tracking for an extracted fact or medication."""
    source_type: Literal["text", "image"] = Field(description="Originating source type")
    source_name: Optional[str] = Field(default=None, description="Filename, 'direct_text', or 'audio_transcript'")
    page_or_index: Optional[int] = Field(default=None, description="Image/page index (1-indexed) if applicable")
    section_or_location: Optional[str] = Field(default=None, description="Visual area, table cell, or paragraph")


class ClinicalSummary(BaseModel):
    """Clinical synthesis of patient presentation, findings, and context."""
    brief_summary: str = Field(description="High-level concise clinical overview")
    detailed_findings: str = Field(description="Detailed synthesis of findings, observations, and context")
    reported_by: Optional[str] = Field(default=None, description="Doctor, clinic, hospital, or laboratory name")


class DocumentInfo(BaseModel):
    """Metadata regarding ingested documents and content modalities."""
    document_type: Optional[str] = Field(
        default="unknown",
        description="Prescription, lab_report, discharge_summary, consultation_note, or unknown"
    )
    sources: List[str] = Field(default_factory=list, description="List of modalities used: ['text'], ['image'], or both")
    detected_content_types: List[str] = Field(
        default_factory=list,
        description="Detected styles: ['printed'], ['handwritten'], ['mixed'], or ['unknown']"
    )
    language: Optional[str] = Field(default="en", description="Identified language of input document")
    document_date: Optional[str] = Field(default=None, description="Date on record if documented")


class KeyFact(BaseModel):
    """Single clinical observation, lab value, symptom, or procedure."""
    fact_id: str = Field(description="Unique stable identifier e.g. fact_01")
    category: FactCategory = Field(description="Category of the fact")
    name: str = Field(description="Name of the measurement, test, or symptom")
    value: str = Field(description="Exact value as recorded in source")
    unit: Optional[str] = Field(default=None, description="Unit of measurement if applicable")
    date: Optional[str] = Field(default=None, description="Date recorded if documented")
    source: SourceAttribution = Field(description="Source provenance")
    status: ItemConfidence = Field(default=ItemConfidence.VERIFIED, description="Confidence level")
    uncertainty_note: Optional[str] = Field(default=None, description="Explanation if status is UNCERTAIN/AMBIGUOUS")


class MedicationItem(BaseModel):
    """Medication documented or prescribed in the clinical record."""
    name: str = Field(description="Medication name (brand or generic)")
    dosage: Optional[str] = Field(default=None, description="Strength / dose e.g. 500mg, 1 tablet")
    route: Optional[str] = Field(default=None, description="Route e.g. oral, topical, IV")
    frequency: Optional[str] = Field(default=None, description="Schedule e.g. once daily, TDS, 1-0-1")
    duration: Optional[str] = Field(default=None, description="Length of course e.g. 5 days, 1 month")
    instructions: Optional[str] = Field(default=None, description="Patient directions e.g. after meals")
    source: SourceAttribution = Field(description="Source provenance")
    status: ItemConfidence = Field(default=ItemConfidence.VERIFIED, description="Confidence level")
    uncertainty_note: Optional[str] = Field(default=None, description="Explanation if dose or name is illegible")


class AllergyItem(BaseModel):
    """Allergy explicitly noted in the medical context."""
    allergen: str = Field(description="Substance or medication allergen")
    reaction: Optional[str] = Field(default=None, description="Documented reaction e.g. anaphylaxis, hives")
    status: str = Field(default="documented", description="Status e.g. documented, denied, unknown")
    source: Optional[SourceAttribution] = Field(default=None, description="Source provenance")


class UncertaintyItem(BaseModel):
    """Specific unreadable, ambiguous, or low-confidence segment."""
    description: str = Field(description="Explanation of the uncertainty")
    source: Optional[SourceAttribution] = Field(default=None, description="Location of uncertain text or handwriting")
    category: str = Field(default="unreadable_handwriting", description="unreadable_handwriting, blurry_scan, ambiguous_abbreviation")


class MissingInfoItem(BaseModel):
    """Clinically important data that is absent from the record."""
    field_name: str = Field(description="Missing attribute e.g. 'Medication dosage', 'Patient age'")
    importance: Literal["critical", "informational"] = Field(description="Safety tier")
    reason: str = Field(description="Clinical reason why this field is important")


class ConflictItem(BaseModel):
    """Direct conflict detected between multiple sources or within a source."""
    field_or_fact: str = Field(description="Contradicted attribute or finding")
    value_source_a: str = Field(description="Finding from first source")
    value_source_b: str = Field(description="Contradicting finding from second source")
    description: str = Field(description="Explanation of the conflict")


class IngestionOutput(BaseModel):
    """Comprehensive, versioned structured output of the Context Ingestion Agent."""

    schema_version: str = Field(default="1.0.0", description="Schema semantic version")
    request_id: Optional[str] = Field(default=None, description="Correlation identifier")
    
    summary: ClinicalSummary = Field(description="Clinical overview and synthesis")
    document_info: DocumentInfo = Field(default_factory=DocumentInfo, description="Document and modality metadata")
    
    key_facts: List[KeyFact] = Field(default_factory=list, description="Extracted clinical facts and measurements")
    medications: List[MedicationItem] = Field(default_factory=list, description="Extracted medications")
    allergies: List[AllergyItem] = Field(default_factory=list, description="Reported allergies")
    important_negations: List[str] = Field(default_factory=list, description="Explicitly documented negative findings")
    
    uncertainties: List[UncertaintyItem] = Field(default_factory=list, description="Unreadable or uncertain elements")
    missing_information: List[MissingInfoItem] = Field(default_factory=list, description="Missing safety-critical fields")
    conflicting_information: List[ConflictItem] = Field(default_factory=list, description="Contradictions between sources")
    
    overall_extraction_status: ExtractionStatus = Field(
        default=ExtractionStatus.COMPLETE,
        description="Overall status indicator"
    )
    requires_human_review: bool = Field(
        default=False,
        description="Flag indicating if human-in-the-loop verification is necessary"
    )

