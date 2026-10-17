"""Standalone demonstration of the Context Ingestion Agent.

Runs locally using purely synthetic medical test data.
Supports two modes:
1. Default (Mock Mode): Runs locally with zero API token consumption.
2. Live Mode (--live): Invokes configured Gemini models using GEMINI_API_KEY from .env.
"""

import argparse
import io
import sys
from PIL import Image, ImageDraw

from agentic_flow import (
    ClinicalSummary,
    ContextIngestionAgent,
    DocumentInfo,
    ExtractionStatus,
    FactCategory,
    IngestionInput,
    IngestionOutput,
    ItemConfidence,
    KeyFact,
    MedicationItem,
    SourceAttribution,
    UncertaintyItem,
)
from agentic_flow.config import AgentConfig


def create_synthetic_prescription_image() -> bytes:
    """Create a synthetic test image representing a clinical prescription scan."""
    img = Image.new("RGB", (600, 300), color=(250, 250, 245))
    draw = ImageDraw.Draw(img)

    # Header
    draw.text((30, 20), "CITY CLINIC & DIAGNOSTICS - CLINICAL REPORT", fill=(20, 20, 80))
    draw.line([(30, 45), (570, 45)], fill=(180, 180, 180), width=1)

    # Patient & Vital Info
    draw.text((30, 60), "Patient: Synthetic Demo Subject | Date: 2026-10-15", fill=(40, 40, 40))
    draw.text((30, 85), "BP: 128/82 mmHg | Fasting Blood Sugar: 138 mg/dL", fill=(40, 40, 40))

    # Medications
    draw.text((30, 120), "Rx Prescriptions:", fill=(120, 20, 20))
    draw.text((50, 150), "1. Tab Metformin 500mg - 1 tab daily with dinner x 30 days", fill=(20, 20, 20))
    draw.text((50, 180), "2. [Handwritten] Amox... 250/500? - take TID x 5 days", fill=(90, 40, 40))

    # Margin note
    draw.text((30, 230), "Important: Denies penicillin allergy in initial intake.", fill=(60, 60, 60))

    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def run_mock_demo():
    """Run demonstration with mocked models (zero API tokens used)."""
    print("\n" + "=" * 65)
    print("  RUNNING CONTEXT INGESTION AGENT DEMO (MOCK MODE)")
    print("=" * 65)

    config = AgentConfig(
        gemini_api_key="mock_demo_key",
        vision_model_id="gemini-3.8-flash",
        summary_model_id="gemini-3.1-flash-lite"
    )

    from tests.conftest import MockGeminiClient

    mock_output = IngestionOutput(
        schema_version="1.0.0",
        request_id="demo-mock-001",
        summary=ClinicalSummary(
            brief_summary="Synthetic patient presenting with elevated fasting blood glucose and mild fatigue. Prescribed daily Metformin with an ambiguous handwritten antibiotic line.",
            detailed_findings="Metformin 500mg was verified. The second prescription line is handwritten and appears to be Amoxicillin with an ambiguous dosage. Patient intake notes state no known allergies.",
            reported_by="City Clinic & Diagnostics"
        ),
        document_info=DocumentInfo(
            document_type="prescription_and_lab_report",
            sources=["text", "image"],
            detected_content_types=["mixed"],
            language="en",
            document_date="2026-10-15"
        ),
        key_facts=[
            KeyFact(
                fact_id="fact_01",
                category=FactCategory.LAB_RESULT,
                name="Fasting Blood Sugar",
                value="138",
                unit="mg/dL",
                source=SourceAttribution(source_type="image", source_name="prescription.jpg", section_or_location="Vitals row"),
                status=ItemConfidence.VERIFIED
            )
        ],
        medications=[
            MedicationItem(
                name="Metformin",
                dosage="500mg",
                route="oral",
                frequency="once daily with dinner",
                duration="30 days",
                instructions="take with meals",
                source=SourceAttribution(source_type="image", source_name="prescription.jpg", section_or_location="Rx line 1"),
                status=ItemConfidence.VERIFIED
            ),
            MedicationItem(
                name="Amoxicillin (?)",
                dosage=None,
                route="oral",
                frequency="TID",
                duration="5 days",
                instructions=None,
                source=SourceAttribution(source_type="image", source_name="prescription.jpg", section_or_location="Rx line 2 handwritten"),
                status=ItemConfidence.UNCERTAIN,
                uncertainty_note="Handwriting dosage is illegible (possibly 250mg or 500mg). Not guessed."
            )
        ],
        uncertainties=[
            UncertaintyItem(
                description="Handwritten dosage on prescription line 2 is ambiguous.",
                category="unreadable_handwriting"
            )
        ],
        overall_extraction_status=ExtractionStatus.NEEDS_REVIEW,
        requires_human_review=True
    )

    mock_client = MockGeminiClient(config, summary_response=mock_output)
    from agentic_flow.services import ClinicalSummarizer, VisionExtractor
    agent = ContextIngestionAgent(
        config=config,
        client=mock_client,
        vision_extractor=VisionExtractor(config, mock_client),
        summarizer=ClinicalSummarizer(config, mock_client)
    )

    sample_image = create_synthetic_prescription_image()
    sample_text = "Audio Transcript: Patient reports feeling fatigued over the last week. Takes Metformin regularly."

    print("\n[1] Submitting combined input: Extracted Audio Transcript + Synthetic Prescription Scan...")
    result = agent.ingest(
        text=sample_text,
        images=[sample_image],
        request_id="demo-mock-001"
    )

    print("\n[2] Execution Succeeded!")
    print(f"- Overall Status: {result.overall_extraction_status}")
    print(f"- Requires Human Review: {result.requires_human_review}")
    print(f"- Identified Medications: {len(result.medications)}")
    print(f"- Flagged Uncertainties: {len(result.uncertainties)}")

    print("\n[3] Serialized JSON Output:")
    print(result.model_dump_json(indent=2))


def run_live_demo():
    """Run demonstration with live Gemini API calls."""
    print("\n" + "=" * 65)
    print("  RUNNING CONTEXT INGESTION AGENT DEMO (LIVE GEMINI API)")
    print("=" * 65)

    config = AgentConfig()
    try:
        config.validate_api_key()
    except Exception as exc:
        print(f"\n[ERROR] Configuration check failed: {exc}")
        print("Please ensure GEMINI_API_KEY is configured in your .env file.")
        sys.exit(1)

    print(f"- Configured Vision Model: {config.vision_model_id}")
    print(f"- Configured Summary Model: {config.summary_model_id}")
    print(f"- API Key Mask: {config.mask_key()}")

    agent = ContextIngestionAgent(config=config)
    sample_image = create_synthetic_prescription_image()
    sample_text = "Audio Transcript: Patient notes persistent fatigue and states they take metformin with dinner."

    print("\n[1] Sending request to Gemini models...")
    try:
        result = agent.ingest(
            text=sample_text,
            images=[sample_image],
            request_id="demo-live-001"
        )
        print("\n[2] Live Inference Succeeded!")
        print(result.model_dump_json(indent=2))
    except Exception as exc:
        print(f"\n[ERROR] Live inference failed: {exc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Context Ingestion Agent Demo")
    parser.add_argument("--live", action="store_true", help="Run with real Gemini API calls instead of mock")
    args = parser.parse_args()

    if args.live:
        run_live_demo()
    else:
        run_mock_demo()

