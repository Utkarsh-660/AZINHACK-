# Multimodal Context Ingestion Agent

A production-grade, two-model clinical context ingestion agent built for multi-agent healthcare architectures.

This agent ingests medical information across multiple modalities (**text transcripts, PDF extracts, printed medical scans, and handwritten clinical notes**), normalizes the content, flags ambiguities and contradictions, and produces a strictly validated Pydantic JSON structure for downstream agents (e.g. Triage, Drug-Drug Interaction, and RAG).

---

## 1. Two-Model Architecture

```
                         [Incoming Request]
                   (Text, Images, or Combined)
                                │
                                ▼
                       [Input Validation]
                                │
          ┌─────────────────────┼─────────────────────┐
          ▼                     ▼                     ▼
   Case A: Text Only     Case B: Image Only    Case C: Text + Image
          │                     │                     │
          │                     ▼                     ▼
          │             [Model A: Vision]      [Model A: Vision]
          │             (gemini-3.8-flash)     (gemini-3.8-flash)
          │             - Clinical OCR         - Clinical OCR
          │             - Cursive handwriting  - Cursive handwriting
          │             - Lab report tables    - Lab report tables
          │                     │                     │
          └─────────────────────┼─────────────────────┘
                                │
                                ▼
                 [Model B: Clinical Summarizer]
                    (gemini-3.1-flash-lite)
                 - Clinical synthesis & overview
                 - Reconcile sources & detect conflicts
                 - Preserve uncertain handwriting
                 - Flag missing safety-critical info
                 - Enforce strict JSON Pydantic schema
                                │
                                ▼
                   [Validated IngestionOutput]
```

*   **Model A (Strong Vision Model - `gemini-3.8-flash`)**: Extracts text, tables, and handwritten notes from medical documents. Does not guess unreadable text.
*   **Model B (Lightweight Summarization - `gemini-3.1-flash-lite`)**: Synthesizes clinical text, tracks provenance, and outputs structured JSON.
*   **Case A (Text Only)**: Skips Model A completely to save tokens and minimize latency.

---

## 2. Quickstart & Installation

### Requirements
* Python 3.10+
* `google-genai>=2.30.0`
* `pydantic>=2.7.0`
* `pillow>=10.0.0`
* `httpx>=0.27.0`
* `python-dotenv>=1.0.0`

Install dependencies:
```bash
pip install -r requirements.txt
```

### Configuration
Set your Gemini API credentials in `.env` (or via environment variables):
```ini
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_VISION_MODEL=gemini-3.8-flash
GEMINI_SUMMARY_MODEL=gemini-3.1-flash-lite
```

---

## 3. Integration Guide for Teammates

### Basic Usage (Synchronous)
```python
from agentic_flow import ContextIngestionAgent, IngestionOutput

# 1. Initialize the agent
agent = ContextIngestionAgent()

# 2. Ingest clinical context (Text only, Image only, or Combined)
result: IngestionOutput = agent.ingest(
    text="Patient reports mild cough and headache for 3 days. Takes Metformin daily.",
    images=["path/to/prescription.jpg"],  # Accepts paths, bytes, PIL Images, or URLs
    request_id="req-patient-001"
)

# 3. Access validated clinical fields
print("Summary:", result.summary.brief_summary)
print("Requires Doctor Review:", result.requires_human_review)

for med in result.medications:
    print(f"- {med.name} | Dose: {med.dosage} | Status: {med.status}")

# 4. Serialize to JSON for database or downstream agents
json_payload = result.model_dump_json(indent=2)
```

### Async Usage
```python
import asyncio
from agentic_flow import ContextIngestionAgent

async def main():
    agent = ContextIngestionAgent()
    result = await agent.aingest(
        text="Extracted audio transcript...",
        images=[b"...image_bytes..."]
    )
    print(result.overall_extraction_status)

asyncio.run(main())
```

---

## 4. Supported Image Inputs & Security

The `images` parameter accepts any of the following formats:
1. **Local file paths** (`str` or `pathlib.Path`)
2. **Raw bytes** (`bytes`)
3. **PIL Image objects** (`PIL.Image.Image`)
4. **Base64 strings** (`data:image/jpeg;base64,...` or raw base64)
5. **Remote URLs** (`http://` or `https://` — e.g. Cloudinary URLs)

### SSRF Protection
Remote image fetching includes strict Server-Side Request Forgery safeguards:
* Prohibits `localhost`, loopback (`127.0.0.1`), and link-local addresses.
* Blocks internal private network IP ranges (`10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`).
* Disallows non-HTTP schemes (e.g. `file://`).
* Enforces HTTP timeouts (10s) and image file size limits (15MB).

---

## 5. Output Schema Reference (`IngestionOutput`)

| Field | Type | Description |
| :--- | :--- | :--- |
| `schema_version` | `str` | Semantic version of the schema (`"1.0.0"`). |
| `request_id` | `Optional[str]` | Caller-supplied tracing identifier. |
| `summary` | `ClinicalSummary` | `brief_summary`, `detailed_findings`, `reported_by`. |
| `document_info` | `DocumentInfo` | `document_type`, `sources`, `detected_content_types`, `language`, `document_date`. |
| `key_facts` | `List[KeyFact]` | Structured lab results, vitals, symptoms, observations with exact units, values, and provenance. |
| `medications` | `List[MedicationItem]` | Medication name, dosage, route, frequency, duration, directions, status (`VERIFIED`/`UNCERTAIN`), provenance. |
| `allergies` | `List[AllergyItem]` | Documented allergens, reactions, and provenance. |
| `important_negations` | `List[str]` | Explicit negative findings (e.g. *"Denies chest pain"*, *"No allergies"*). |
| `uncertainties` | `List[UncertaintyItem]` | Unreadable handwriting, blurry sections, or ambiguous abbreviations. |
| `missing_information` | `List[MissingInfoItem]` | Absent safety-critical fields (e.g. missing antibiotic dosage). |
| `conflicting_information` | `List[ConflictItem]` | Contradictions between sources (e.g. patient text denies allergy, prescription lists penicillin). |
| `overall_extraction_status`| `ExtractionStatus` | `COMPLETE`, `PARTIALLY_EXTRACTED`, `NEEDS_REVIEW`, `FAILED`. |
| `requires_human_review` | `bool` | `True` if handwriting is illegible, data is contradictory, or dosage is missing. |

---

## 6. Error Handling

All agent exceptions inherit from `AgenticFlowError`:

*   `InvalidInputError`: Raised when both text and image payloads are empty or whitespace.
*   `ImageProcessingError`: Raised if an image is corrupted, unreadable, or exceeds 15MB / 4096px.
*   `SSRFSecurityError`: Raised if an image URL targets a private/local IP address.
*   `ConfigurationError`: Raised if `GEMINI_API_KEY` is missing.
*   `ModelAuthenticationError`: Raised on invalid API credentials (non-retryable).
*   `ModelRateLimitError`: Raised on quota exhaustion (automatically retried with exponential backoff up to 3 times).
*   `ModelTimeoutError`: Raised on network/deadline timeout (automatically retried).
*   `ModelOutputValidationError`: Raised if model response violates the Pydantic schema.

---

## 7. Running Tests & Demonstration

### Automated Unit Tests (100% Mocked, Zero API Calls)
```bash
python -m pytest tests/ -v
```
*Executes all 28 unit tests in under 1 second without network access.*

### Standalone Demonstration
```bash
# Mock mode (Default - uses deterministic synthetic data, no API calls)
python run_demo.py

# Live mode (Executes real inference using your GEMINI_API_KEY from .env)
python run_demo.py --live
```

---

## 8. Medical & Safety Disclaimers
*   **Non-Diagnostic**: This agent is strictly an ingestion and structured extraction component. It does not diagnose clinical conditions, prescribe medications, or alter medical dosages.
*   **Human-In-The-Loop**: Any ambiguous handwriting, missing dosage, or conflicting information triggers `requires_human_review=True`.

