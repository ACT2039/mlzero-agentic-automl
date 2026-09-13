"""Tests for LLM Abstraction."""

from pydantic import BaseModel

from mlzero.core.llm import MockLLMClient, get_llm_client


class DummySchema(BaseModel):
    value: str


def test_mock_llm_client_custom_response():
    client = MockLLMClient(mock_responses={"DummySchema": {"value": "test"}})
    result = client.generate_structured("prompt", DummySchema)
    assert result.value == "test"


def test_get_llm_client_mock():
    client = get_llm_client(use_mock=True)
    assert isinstance(client, MockLLMClient)
