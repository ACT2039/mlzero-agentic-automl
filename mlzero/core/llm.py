"""
LLM Client Abstraction.
"""
from abc import ABC, abstractmethod
from typing import Any, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMClient(ABC):
    """Abstract interface for LLM interactions."""
    
    @abstractmethod
    def generate_structured(self, prompt: str, schema: type[T]) -> T:
        """
        Generate a structured response from the LLM based on a Pydantic schema.
        
        Args:
            prompt: The instruction and context.
            schema: The Pydantic model to populate.
            
        Returns:
            An instance of the schema populated with LLM output.
        """


class MockLLMClient(LLMClient):
    """Deterministic mock for testing without API keys."""
    
    def __init__(self, mock_responses: dict[str, Any] | None = None):
        """
        Args:
            mock_responses: Optional dictionary to override default mock behavior based on schema name.
        """
        self.mock_responses = mock_responses or {}
        
    def generate_structured(self, prompt: str, schema: type[T]) -> T:
        """Returns deterministic mock data."""
        schema_name = schema.__name__
        
        if schema_name in self.mock_responses:
            return schema(**self.mock_responses[schema_name])
            
        # Default fallbacks based on schema name
        if schema_name == "TaskContext":
            return schema(
                objective="Mock Objective",
                task_type="classification",
                target_column="target",
                input_data_files=["train.csv"],
                explanation="Mock extracted task context."
            )
        elif schema_name == "LibrarySelection":
            return schema(
                selected_library="autogluon.tabular",
                confidence="high",
                explanation="Mock selected library."
            )
            
        elif schema_name == "CodeArtifact":
            if "PREVIOUS FAILURE" in prompt:
                # Iteration 2 (Recovery)
                code = (
                    "import csv\n"
                    "import os\n"
                    "try:\n"
                    "    with open('data.csv', 'r') as f:\n"
                    "        reader = csv.reader(f)\n"
                    "        headers = next(reader)\n"
                    "        total = sum(int(row[1]) for row in reader)\n"
                    "    os.makedirs('out', exist_ok=True)\n"
                    "    with open('out/result.txt', 'w') as f:\n"
                    "        f.write(str(total))\n"
                    "except Exception as e:\n"
                    "    print('Error:', e)\n"
                )
            else:
                # Iteration 1 (Deliberate failure: missing file or syntax error)
                code = (
                    "import csv\n"
                    "import os\n"
                    "# Intentional failure: trying to read a non-existent file\n"
                    "with open('missing_file.csv', 'r') as f:\n"
                    "    pass\n"
                )
            return schema(
                code=code,
                language="python",
                dependencies=[],
                status="generated"
            )
        elif schema_name == "ErrorContext":
            return schema(
                iteration=1,
                error_category="file_not_found",
                error_message="FileNotFoundError: No such file or directory: 'missing_file.csv'",
                stderr_excerpt="Traceback... FileNotFoundError",
                suggested_fix="Change 'missing_file.csv' to the correct data file 'data.csv'."
            )
        elif schema_name == "SummaryResult":
            return schema(summary="Mock summary of documentation.")
        elif schema_name == "CondensationResult":
            return schema(condensed_text="Mock condensed implementation guidance.")
            
        # Generic fallback using field defaults or empty values
        # In a real mock, you'd inspect fields, but for our specific phase, 
        # the above covers the used schemas.
        raise NotImplementedError(f"Mock response not configured for {schema_name}")


def get_llm_client(use_mock: bool = False) -> LLMClient:
    """Factory to get the appropriate LLM client."""
    if use_mock:
        return MockLLMClient()
    
    # In Phase 2 we just use Mock to avoid needing real API keys during perception
    # if the user hasn't configured it, or we could implement a real one.
    # The prompt says: "Do not require an API call for every unit test."
    # But it also says: "Create a clean LLM abstraction. Do NOT scatter OpenAI API calls."
    # We will just return the Mock for now to ensure we don't break without keys,
    # or implement a real one that raises an error if keys are missing.
    
    from mlzero.core.config import settings
    
    # Placeholder for real implementation
    class RealLLMClient(LLMClient):
        def generate_structured(self, prompt: str, schema: type[T]) -> T:
            # Here you would use the OpenAI API.
            # E.g., client.beta.chat.completions.parse(...)
            # For phase 2, if we don't have openai installed, we fail gracefully or use mock.
            if not settings.openai_api_key:
                raise ValueError("OpenAI API key is required for RealLLMClient.")
            raise NotImplementedError("Real LLM client not implemented in Phase 2.")
            
    return RealLLMClient()
