"""Gemini API client wrapper with bounded retries, error mapping, and structured output support."""

import asyncio
import json
import re
import time
from typing import Any, List, Optional, Type, TypeVar
from pydantic import BaseModel, ValidationError
from google import genai
from google.genai import types

from agentic_flow.config import AgentConfig, get_default_config
from agentic_flow.exceptions import (
    ModelAuthenticationError,
    ModelError,
    ModelOutputValidationError,
    ModelRateLimitError,
    ModelTimeoutError,
)

T = TypeVar("T", bound=BaseModel)


class GeminiClient:
    """Resilient client for Google GenAI with bounded retries, timeout management, and mockability."""

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        mock_client: Optional[Any] = None
    ):
        self.config = config or get_default_config()
        self._mock_client = mock_client
        self._client: Optional[genai.Client] = None

    @property
    def client(self) -> genai.Client:
        """Lazy-initialize official Google GenAI client with configured timeout."""
        if self._mock_client is not None:
            return self._mock_client
        if self._client is None:
            self.config.validate_api_key()
            timeout_ms = int(self.config.request_timeout * 1000)
            self._client = genai.Client(
                api_key=self.config.gemini_api_key,
                http_options=types.HttpOptions(timeout=timeout_ms)
            )
        return self._client

    def generate_structured(
        self,
        model_id: str,
        contents: List[Any],
        target_schema: Type[T],
        system_instruction: Optional[str] = None
    ) -> T:
        """Synchronously generate structured output validated against a Pydantic schema."""
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=target_schema,
            system_instruction=system_instruction
        )

        attempts = 0
        backoff = self.config.initial_retry_delay

        while True:
            attempts += 1
            try:
                response = self.client.models.generate_content(
                    model=model_id,
                    contents=contents,
                    config=config
                )
                return self._parse_and_validate(response, target_schema)

            except Exception as exc:
                mapped_error = self._map_exception(exc)
                if self._is_transient(mapped_error) and attempts <= self.config.max_retries:
                    time.sleep(backoff)
                    backoff *= 2.0
                    continue
                raise mapped_error

    async def async_generate_structured(
        self,
        model_id: str,
        contents: List[Any],
        target_schema: Type[T],
        system_instruction: Optional[str] = None
    ) -> T:
        """Asynchronously generate structured output validated against a Pydantic schema."""
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=target_schema,
            system_instruction=system_instruction
        )

        attempts = 0
        backoff = self.config.initial_retry_delay

        while True:
            attempts += 1
            try:
                # If mock client has async or sync methods
                if self._mock_client is not None:
                    aio_models = getattr(getattr(self._mock_client, "aio", None), "models", None)
                    if aio_models and hasattr(aio_models, "generate_content"):
                        response = await aio_models.generate_content(
                            model=model_id,
                            contents=contents,
                            config=config
                        )
                    else:
                        response = self._mock_client.models.generate_content(
                            model=model_id,
                            contents=contents,
                            config=config
                        )
                else:
                    response = await self.client.aio.models.generate_content(
                        model=model_id,
                        contents=contents,
                        config=config
                    )

                return self._parse_and_validate(response, target_schema)

            except Exception as exc:
                mapped_error = self._map_exception(exc)
                if self._is_transient(mapped_error) and attempts <= self.config.max_retries:
                    await asyncio.sleep(backoff)
                    backoff *= 2.0
                    continue
                raise mapped_error

    def _parse_and_validate(self, response: Any, target_schema: Type[T]) -> T:
        """Safely parse model response text or parsed object into target Pydantic schema."""
        # 1. Try directly if SDK parsed it
        if hasattr(response, "parsed") and response.parsed is not None:
            if isinstance(response.parsed, target_schema):
                return response.parsed
            if isinstance(response.parsed, dict):
                try:
                    return target_schema.model_validate(response.parsed)
                except ValidationError as val_err:
                    raise ModelOutputValidationError(f"Schema validation failed: {val_err}")

        # 2. Extract text and validate JSON
        raw_text = getattr(response, "text", None)
        if not raw_text and hasattr(response, "candidates") and response.candidates:
            parts = response.candidates[0].content.parts
            if parts and hasattr(parts[0], "text"):
                raw_text = parts[0].text

        if not raw_text:
            raise ModelOutputValidationError("Model returned empty response content.")

        # Strip markdown json fences if present
        clean_text = raw_text.strip()
        if clean_text.startswith("```json"):
            clean_text = clean_text[7:]
        elif clean_text.startswith("```"):
            clean_text = clean_text[3:]
        if clean_text.endswith("```"):
            clean_text = clean_text[:-3]
        clean_text = clean_text.strip()

        try:
            parsed_data = json.loads(clean_text)
            return target_schema.model_validate(parsed_data)
        except (json.JSONDecodeError, ValidationError) as err:
            raise ModelOutputValidationError(f"Failed to validate response against {target_schema.__name__}: {err}")

    def _map_exception(self, exc: Exception) -> ModelError:
        """Map SDK/HTTP exceptions to domain ModelError classes without exposing secrets."""
        if isinstance(exc, ModelError):
            return exc

        err_msg = str(exc)
        lowered = err_msg.lower()

        # Auth errors
        if (
            "401" in err_msg
            or "403" in err_msg
            or "api_key_invalid" in lowered
            or "permission_denied" in lowered
            or "unauthenticated" in lowered
        ):
            return ModelAuthenticationError("Authentication failure: Invalid or unauthorized Gemini API key.")

        # Rate limits
        if "429" in err_msg or "resource_exhausted" in lowered or "rate limit" in lowered:
            return ModelRateLimitError("Gemini API rate limit exceeded or quota exhausted.")

        # Timeouts
        if "504" in err_msg or "timeout" in lowered or "deadline_exceeded" in lowered:
            return ModelTimeoutError("Gemini API request timed out.")

        # Schema or parse errors
        if isinstance(exc, (ValidationError, json.JSONDecodeError)):
            return ModelOutputValidationError(f"Failed to parse model output: {exc}")

        # Sanitize error message to scrub any API key or sensitive token patterns
        sanitized_msg = err_msg
        if self.config.gemini_api_key and self.config.gemini_api_key in sanitized_msg:
            sanitized_msg = sanitized_msg.replace(self.config.gemini_api_key, "<REDACTED_API_KEY>")
        sanitized_msg = re.sub(r'key=[A-Za-z0-9_\-\.]{8,}', 'key=<REDACTED>', sanitized_msg)

        return ModelError(f"Gemini API inference error: {sanitized_msg}")

    def _is_transient(self, exc: ModelError) -> bool:
        """Check whether an error is transient and safe to retry."""
        return isinstance(exc, (ModelRateLimitError, ModelTimeoutError))

