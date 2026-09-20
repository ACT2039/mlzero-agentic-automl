"""
Unit tests for the Real LLM Provider and Mock/Real toggle.
All unit tests use mocked HTTP calls or mock clients so no Gemini quota is consumed.
"""
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from mlzero.core.llm import (
    MockLLMClient,
    RealLLMClient,
    get_llm_client,
)
from mlzero.ui.app import submit_task


class SampleSchema(BaseModel):
    value: str
    count: int = 1


def test_get_llm_client_mock():
    """Verify factory returns MockLLMClient for mock mode."""
    client = get_llm_client(mode="mock")
    assert isinstance(client, MockLLMClient)

    client_flag = get_llm_client(use_mock=True)
    assert isinstance(client_flag, MockLLMClient)


def test_get_llm_client_real():
    """Verify factory returns RealLLMClient when mode is 'real'."""
    with patch("mlzero.core.config.settings.gemini_api_key", "dummy-key"):
        client = get_llm_client(mode="real")
        assert isinstance(client, RealLLMClient)

        client_flag = get_llm_client(use_mock=False)
        assert isinstance(client_flag, RealLLMClient)


def test_get_llm_client_invalid_mode():
    """Verify factory raises ValueError for invalid mode."""
    with pytest.raises(ValueError, match="Unknown LLM mode"):
        get_llm_client(mode="unsupported_mode")


def test_mock_llm_client_text_and_structured():
    """Verify MockLLMClient generates deterministic text and structured responses."""
    client = MockLLMClient(mock_responses={"SampleSchema": {"value": "mocked", "count": 42}})
    
    text = client.generate_text("Hello")
    assert isinstance(text, str)
    assert len(text) > 0

    structured = client.generate_structured("test", SampleSchema)
    assert structured.value == "mocked"
    assert structured.count == 42


def test_real_llm_client_missing_key():
    """Verify RealLLMClient raises ValueError if API key is missing."""
    with patch("mlzero.core.config.settings.gemini_api_key", ""), pytest.raises(
        ValueError, match="Gemini API key is required"
    ):
        RealLLMClient(api_key="")


def test_real_llm_client_sanitization():
    """Verify RealLLMClient._sanitize strips the API key from any string."""
    fake_key = "AIzaSySecretTestKey123456789"
    client = RealLLMClient(api_key=fake_key)

    msg = f"HTTP 401 Unauthorized: request with key {fake_key} failed at endpoint."
    sanitized = client._sanitize(msg)
    assert fake_key not in sanitized
    assert "[REDACTED]" in sanitized


@patch("httpx.Client.post")
def test_real_llm_client_generate_text(mock_post: MagicMock):
    """Verify generate_text calls Gemini OpenAI-compatible endpoint with correct payload."""
    fake_response = MagicMock()
    fake_response.raise_for_status = MagicMock()
    fake_response.json.return_value = {
        "choices": [{"message": {"content": "MLZero real LLM connection successful"}}]
    }
    mock_post.return_value = fake_response

    client = RealLLMClient(api_key="test-key", model="gemini-3.8-flash")
    result = client.generate_text("Reply with exactly: MLZero real LLM connection successful")

    assert result == "MLZero real LLM connection successful"
    mock_post.assert_called_once()
    call_kwargs = mock_post.call_args[1]
    assert "Authorization" in call_kwargs["headers"]
    assert call_kwargs["json"]["model"] == "gemini-3.8-flash"


@patch("httpx.Client.post")
def test_real_llm_client_generate_structured(mock_post: MagicMock):
    """Verify generate_structured parses JSON into Pydantic schema."""
    fake_response = MagicMock()
    fake_response.raise_for_status = MagicMock()
    fake_response.json.return_value = {
        "choices": [{"message": {"content": '{"value": "parsed_ok", "count": 10}'}}]
    }
    mock_post.return_value = fake_response

    client = RealLLMClient(api_key="test-key")
    result = client.generate_structured("Provide data", SampleSchema)

    assert result.value == "parsed_ok"
    assert result.count == 10


@patch("httpx.Client.post")
def test_real_llm_client_generate_structured_with_markdown(mock_post: MagicMock):
    """Verify generate_structured strips markdown code fences before parsing."""
    fake_response = MagicMock()
    fake_response.raise_for_status = MagicMock()
    fake_response.json.return_value = {
        "choices": [{"message": {"content": "```json\n{\n  \"value\": \"in_fence\",\n  \"count\": 5\n}\n```"}}]
    }
    mock_post.return_value = fake_response

    client = RealLLMClient(api_key="test-key")
    result = client.generate_structured("Provide data", SampleSchema)

    assert result.value == "in_fence"
    assert result.count == 5


@patch("httpx.Client.post")
def test_real_llm_client_retry_on_invalid_json(mock_post: MagicMock):
    """Verify generate_structured retries when the first attempt produces malformed JSON."""
    bad_response = MagicMock()
    bad_response.raise_for_status = MagicMock()
    bad_response.json.return_value = {
        "choices": [{"message": {"content": "Invalid not-json text"}}]
    }

    good_response = MagicMock()
    good_response.raise_for_status = MagicMock()
    good_response.json.return_value = {
        "choices": [{"message": {"content": '{"value": "after_retry", "count": 1}'}}]
    }

    mock_post.side_effect = [bad_response, good_response]

    client = RealLLMClient(api_key="test-key", retry_count=2)
    result = client.generate_structured("Provide data", SampleSchema)

    assert result.value == "after_retry"
    assert mock_post.call_count == 2


def test_ui_submit_task_mode_resolution():
    """Verify Gradio submit_task correctly parses both Radio strings and legacy booleans."""
    with patch("mlzero.ui.app.run_manager.submit_run") as mock_submit:
        mock_submit.return_value = "mock-run-id"

        # 1. Radio string: Mock mode
        run_id, _, _ = submit_task("dataset/path", "instruction", "Mock LLM (Offline / deterministic)")
        assert run_id == "mock-run-id"
        mock_submit.assert_called_with(
            dataset_path="dataset/path",
            user_instruction="instruction",
            options={"mock_llm": True, "llm_mode": "mock"}
        )

        # 2. Radio string: Real mode
        run_id, _, _ = submit_task("dataset/path", "instruction", "Real LLM (Gemini 3.8 Flash)")
        mock_submit.assert_called_with(
            dataset_path="dataset/path",
            user_instruction="instruction",
            options={"mock_llm": False, "llm_mode": "real"}
        )

        # 3. Legacy boolean: True -> mock
        run_id, _, _ = submit_task("dataset/path", "instruction", True)
        mock_submit.assert_called_with(
            dataset_path="dataset/path",
            user_instruction="instruction",
            options={"mock_llm": True, "llm_mode": "mock"}
        )

        # 4. Legacy boolean: False -> real
        run_id, _, _ = submit_task("dataset/path", "instruction", False)
        mock_submit.assert_called_with(
            dataset_path="dataset/path",
            user_instruction="instruction",
            options={"mock_llm": False, "llm_mode": "real"}
        )
