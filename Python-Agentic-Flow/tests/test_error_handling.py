"""Unit tests for error handling: transient retries, permanent auth errors, and configuration failures."""

import pytest

from agentic_flow import ContextIngestionAgent, IngestionOutput
from agentic_flow.config import AgentConfig
from agentic_flow.exceptions import (
    ConfigurationError,
    ModelAuthenticationError,
    ModelOutputValidationError,
    ModelRateLimitError,
)
from agentic_flow.services import ClinicalSummarizer, GeminiClient
from tests.conftest import MockGeminiClient


def test_missing_api_key_raises_configuration_error():
    """Verify missing GEMINI_API_KEY raises a clean ConfigurationError."""
    empty_config = AgentConfig(gemini_api_key="")
    client = GeminiClient(empty_config)

    with pytest.raises(ConfigurationError, match="Missing GEMINI_API_KEY"):
        _ = client.client


def test_auth_error_is_permanent_not_retried(test_config: AgentConfig):
    """Verify ModelAuthenticationError is raised immediately without retrying."""
    mock_client = MockGeminiClient(test_config)
    mock_client.fail_permanent = True

    agent = ContextIngestionAgent(
        config=test_config,
        client=mock_client,
        summarizer=ClinicalSummarizer(test_config, mock_client)
    )

    with pytest.raises(ModelAuthenticationError, match="Authentication failure"):
        agent.ingest(text="Sample test notes")

    # Should have called only once and failed immediately
    assert len(mock_client.called_models) == 1


def test_transient_rate_limit_recovers_after_retry(
    test_config: AgentConfig,
    mock_summary_output: IngestionOutput
):
    """Verify that a transient rate limit error retries and succeeds if under max_retries."""
    class FakeResponse:
        text = mock_summary_output.model_dump_json()
        parsed = None

    class MockModels:
        def __init__(self):
            self.call_count = 0
            self.fail_count = 1

        def generate_content(self, *args, **kwargs):
            self.call_count += 1
            if self.fail_count > 0:
                self.fail_count -= 1
                raise Exception("429 Resource Exhausted: rate limit exceeded")
            return FakeResponse()

    class MockRawClient:
        def __init__(self):
            self.models = MockModels()

    raw_client = MockRawClient()
    gemini_client = GeminiClient(config=test_config, mock_client=raw_client)

    result = gemini_client.generate_structured(
        model_id=test_config.summary_model_id,
        contents=["test prompt"],
        target_schema=IngestionOutput
    )

    assert isinstance(result, IngestionOutput)
    # Called twice: 1 failure + 1 successful retry
    assert raw_client.models.call_count == 2


def test_key_masking_does_not_reveal_secret():
    """Verify API key masking function never leaks sensitive key characters."""
    cfg = AgentConfig(gemini_api_key="MOCK_SECRET_KEY_PORTION_1234567890_XYZ99")
    masked = cfg.mask_key()
    assert masked.startswith("MOCK...")
    assert masked.endswith("...YZ99")
    assert "SECRET" not in masked
    assert len(masked) < 20


def test_client_parse_invalid_json_raises_model_output_validation_error(test_config: AgentConfig):
    """Verify malformed JSON from model raises ModelOutputValidationError."""
    client = GeminiClient(test_config)

    class FakeResponse:
        text = "This is not valid JSON at all: 404 Error"
        parsed = None

    with pytest.raises(ModelOutputValidationError):
        client._parse_and_validate(FakeResponse(), IngestionOutput)


def test_config_validation_constraints_enforced():
    """Verify Pydantic validation rejects invalid timeouts, retries, and bounds."""
    from pydantic import ValidationError

    # Invalid request_timeout (< 1.0 or > 300.0)
    with pytest.raises(ValidationError):
        AgentConfig(request_timeout=0.5)
    with pytest.raises(ValidationError):
        AgentConfig(request_timeout=600.0)

    # Invalid max_retries (< 0 or > 10)
    with pytest.raises(ValidationError):
        AgentConfig(max_retries=-1)
    with pytest.raises(ValidationError):
        AgentConfig(max_retries=15)

    # Invalid initial_retry_delay (< 0.01)
    with pytest.raises(ValidationError):
        AgentConfig(initial_retry_delay=0.0001)

    # Invalid max_image_bytes (< 128)
    with pytest.raises(ValidationError):
        AgentConfig(max_image_bytes=50)


def test_client_applies_timeout_http_options():
    """Verify GeminiClient passes timeout in milliseconds to HttpOptions."""
    from google.genai import types

    cfg = AgentConfig(gemini_api_key="mock", request_timeout=45.0)
    client = GeminiClient(cfg)
    c = client.client

    # Verify Client initialization configured timeout (45000 ms)
    api_client = getattr(c, "_api_client", None)
    assert api_client is not None
    assert getattr(api_client, "_http_options", None) is not None
    assert api_client._http_options.timeout == 45000
