"""Intermediate extraction schemas produced by Model A (Vision Extractor)."""

from typing import List, Optional
from pydantic import BaseModel, Field


class IntermediateFact(BaseModel):
    """Fact extracted directly from an image."""
    category: str = Field(description="Category: LAB_RESULT, VITAL_SIGN, SYMPTOM, CLINICAL_OBSERVATION, PROCEDURE, OTHER")
    name: str = Field(description="Clinical metric, observation, or test name")
    value: str = Field(description="Exact value as written")
    unit: Optional[str] = Field(default=None, description="Unit of measurement if present")
    date: Optional[str] = Field(default=None, description="Date associated with finding")
    location: Optional[str] = Field(default=None, description="Section, row, or visual area")
    is_uncertain: bool = Field(default=False, description="True if handwriting or print is partially illegible")
    uncertainty_reason: Optional[str] = Field(default=None, description="Details of why content is uncertain")


class IntermediateMedication(BaseModel):
    """Medication extracted from an image."""
    name: str = Field(description="Medication name as written")
    dosage: Optional[str] = Field(default=None, description="Dosage as written")
    route: Optional[str] = Field(default=None, description="Route e.g. oral, topical")
    frequency: Optional[str] = Field(default=None, description="Frequency e.g. once daily")
    duration: Optional[str] = Field(default=None, description="Duration e.g. 5 days")
    instructions: Optional[str] = Field(default=None, description="Instructions e.g. after meals")
    location: Optional[str] = Field(default=None, description="Visual location or line number")
    is_uncertain: bool = Field(default=False, description="True if medicine name or dose is unreadable")
    uncertainty_reason: Optional[str] = Field(default=None, description="Reason for uncertainty")


class IntermediateAllergy(BaseModel):
    """Allergy explicitly noted in the document."""
    allergen: str = Field(description="Allergen name")
    reaction: Optional[str] = Field(default=None, description="Reported allergic reaction")
    status: str = Field(default="documented", description="Status: documented, denied, unknown")


class VisionExtractionResult(BaseModel):
    """Structured result returned by Model A (Vision Extraction)."""

    image_index: int = Field(default=0, description="Zero-based index of the processed image")
    document_type: str = Field(default="unknown", description="Prescription, lab_report, discharge_summary, etc.")
    content_type: str = Field(default="unknown", description="printed, handwritten, mixed, unknown")
    document_date: Optional[str] = Field(default=None, description="Document date if visible")
    provider_or_facility: Optional[str] = Field(default=None, description="Clinic, hospital, or doctor name")
    raw_extracted_text: str = Field(default="", description="Verbatim or linearized text read from the image")
    
    facts: List[IntermediateFact] = Field(default_factory=list)
    medications: List[IntermediateMedication] = Field(default_factory=list)
    allergies: List[IntermediateAllergy] = Field(default_factory=list)
    visible_negations: List[str] = Field(default_factory=list)
    unreadable_or_ambiguous_sections: List[str] = Field(default_factory=list)

