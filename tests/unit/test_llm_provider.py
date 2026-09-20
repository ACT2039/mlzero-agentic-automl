from unittest.mock import MagicMock, patch

import httpx
import pytest
from pydantic import BaseModel

from mlzero.core.llm import (
    FallbackLLMClient,
    GeminiProviderClient,
    OpenRouterProviderClient,
    PermanentLLMError,
    TransientLLMError,
    get_llm_client,
    sanitize_secrets,
)


class DummySchema(BaseModel):
    decision: str
    reason: str


def test_sanitize_secrets():
    secret_key = "sk-or-v1-secret-key-123456789"
    raw_msg = f"HTTP Error 503: Connection failed with key {secret_key}"
    sanitized = sanitize_secrets(raw_msg, [secret_key])
    assert secret_key not in sanitized
    assert "[REDACTED]" in sanitized


def test_gemini_provider_success():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "Gemini text response"}}]
    }

    with patch("httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value.__enter__.return_value = mock_client

        client = GeminiProviderClient(api_key="gemini-test-key-12345", retry_count=1)
        res = client.generate_text("Hello Gemini")
        assert res == "Gemini text response"


def test_openrouter_provider_success():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "OpenRouter text response"}}]
    }

    with patch("httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value.__enter__.return_value = mock_client

        client = OpenRouterProviderClient(api_key="sk-or-v1-test-key-12345", retry_count=1)
        res = client.generate_text("Hello OpenRouter")
        assert res == "OpenRouter text response"


def test_provider_selection():
    with patch("mlzero.core.config.settings") as mock_settings:
        mock_settings.llm_mode = "real"
        mock_settings.real_llm_provider = "gemini"
        mock_settings.gemini_api_key = "gemini-key-12345"
        mock_settings.openrouter_api_key = "sk-or-v1-key-12345"
        mock_settings.real_llm_model = "gemini-3.8-flash"
        mock_settings.real_llm_base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        mock_settings.openrouter_model = "google/gemini-3.8-flash"
        mock_settings.openrouter_base_url = "https://openrouter.ai/api/v1"

        client_gemini = get_llm_client(mode="real", provider="gemini")
        assert isinstance(client_gemini.primary, GeminiProviderClient)
        assert isinstance(client_gemini.fallback, OpenRouterProviderClient)

        client_or = get_llm_client(mode="real", provider="openrouter")
        assert isinstance(client_or.primary, OpenRouterProviderClient)
        assert isinstance(client_or.fallback, GeminiProviderClient)


def test_fallback_on_http_503():
    primary = MagicMock()
    primary.generate_text.side_effect = TransientLLMError("Primary HTTP 503 Unavailable")

    fallback = MagicMock()
    fallback.generate_text.return_value = "Fallback OpenRouter response"

    client = FallbackLLMClient(primary=primary, fallback=fallback)
    result = client.generate_text("Test prompt")

    assert result == "Fallback OpenRouter response"
    primary.generate_text.assert_called_once()
    fallback.generate_text.assert_called_once()


def test_fallback_on_http_429():
    primary = MagicMock()
    primary.generate_text.side_effect = TransientLLMError("Primary HTTP 429 Rate Limit")

    fallback = MagicMock()
    fallback.generate_text.return_value = "Fallback OpenRouter response"

    client = FallbackLLMClient(primary=primary, fallback=fallback)
    result = client.generate_text("Test prompt")

    assert result == "Fallback OpenRouter response"


def test_no_fallback_on_permanent_401():
    primary = MagicMock()
    primary.generate_text.side_effect = PermanentLLMError("Primary HTTP 401 Unauthorized")

    fallback = MagicMock()

    client = FallbackLLMClient(primary=primary, fallback=fallback)
    with pytest.raises(PermanentLLMError):
        client.generate_text("Test prompt")

    fallback.generate_text.assert_not_called()


def test_structured_generation_openrouter():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": '{"decision": "FINISH", "reason": "Execution succeeded"}'}}]
    }

    with patch("httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value.__enter__.return_value = mock_client

        client = OpenRouterProviderClient(api_key="sk-or-v1-test-key-12345", retry_count=1)
        res = client.generate_structured("Evaluate execution", DummySchema)
        assert res.decision == "FINISH"
        assert res.reason == "Execution succeeded"


def test_api_key_redaction():
    secret_key = "secret-gemini-key-9999999"
    mock_res = MagicMock()
    mock_res.status_code = 401
    mock_res.text = f"Invalid API Key: {secret_key}"
    mock_res.raise_for_status.side_effect = httpx.HTTPStatusError(
        message="401 Client Error: Unauthorized", request=MagicMock(), response=mock_res
    )

    with patch("httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.post.return_value = mock_res
        mock_client_cls.return_value.__enter__.return_value = mock_client

        client = GeminiProviderClient(api_key=secret_key, retry_count=1)
        with pytest.raises(PermanentLLMError) as exc_info:
            client.generate_text("Test prompt")

        msg = str(exc_info.value)
        assert secret_key not in msg
        assert "[REDACTED]" in msg
