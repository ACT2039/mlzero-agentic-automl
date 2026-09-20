"""
LLM Client Abstraction.
"""
import json
import re
import time
from abc import ABC, abstractmethod
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMClient(ABC):
    """Abstract interface for LLM interactions."""
    
    @abstractmethod
    def generate_text(self, prompt: str) -> str:
        """
        Generate raw text from the LLM.
        
        Args:
            prompt: The instruction and context.
            
        Returns:
            The raw text string response.
        """

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

    def generate_text(self, prompt: str) -> str:
        """Returns deterministic mock text."""
        return "Mock LLM text response."
        
    def generate_structured(self, prompt: str, schema: type[T]) -> T:
        """Returns deterministic mock data."""
        schema_name = schema.__name__
        
        if schema_name in self.mock_responses:
            return schema(**self.mock_responses[schema_name])
            
        # Default fallbacks based on schema name
        if schema_name == "TaskContext":
            prompt_lower = prompt.lower()
            task_type = "classification"
            target_col: str | None = "target"
            ts_col: str | None = None
            id_col: str | None = None

            if "image" in prompt_lower or "multimodal" in prompt_lower or "text classification" in prompt_lower:
                task_type = "multimodal"
                target_col = "label"
            elif "retrieval" in prompt_lower or "corpus" in prompt_lower or "query" in prompt_lower:
                task_type = "retrieval"
                target_col = None
            elif "time_series" in prompt_lower or "forecasting" in prompt_lower or "timestamp" in prompt_lower or "item_id" in prompt_lower:
                task_type = "time_series_forecasting"
                target_col = "target"
                ts_col = "timestamp"
                id_col = "item_id"
            elif "churn" in prompt_lower:
                task_type = "classification"
                target_col = "churn"

            return schema(
                objective="Mock Objective",
                task_type=task_type,
                target_column=target_col,
                timestamp_column=ts_col,
                id_column=id_col,
                input_data_files=["train.csv"],
                explanation="Mock extracted task context."
            )
        elif schema_name == "LibrarySelection":
            import re
            task_match = re.search(r'Task type:\s*([^\n\r]+)', prompt)
            task_type = task_match.group(1).strip().lower() if task_match else "classification"
            
            if "time_series" in task_type or "forecasting" in task_type:
                lib = "autogluon.timeseries"
            elif "retrieval" in task_type or "embedding" in task_type or "search" in task_type:
                lib = "FlagEmbedding"
            elif "multimodal" in task_type or "image classification" in task_type or "text classification" in task_type or "image" in task_type:
                lib = "autogluon.multimodal"
            elif "regression" in task_type:
                lib = "autogluon.tabular"
            else:
                lib = "autogluon.tabular"
                
            return schema(
                selected_library=lib,
                confidence="high",
                explanation=f"Mock selected library based on task {task_type}."
            )
            
        elif schema_name == "CodeArtifact":
            import re
            
            train_path = 'train.csv'
            test_path = 'test.csv'
            files_match = re.search(r'"input_data_files":\s*\[(.*?)\]', prompt)
            if files_match:
                paths = re.findall(r'[\'"]([^\'"]+)[\'"]', files_match.group(1))
                for p in paths:
                    clean_p = p.replace('\\\\', '/').replace('\\', '/')
                    if 'train.csv' in clean_p: train_path = clean_p
                    if 'test.csv' in clean_p: test_path = clean_p
            
            # Check for non-tabular library adapter guidance in prompt
            if "autogluon.timeseries" in prompt:
                code = (
                    "import os, json\n"
                    "import pandas as pd\n"
                    "os.makedirs('out/models', exist_ok=True)\n"
                    f"try:\n"
                    f"    df = pd.read_csv('{train_path}')\n"
                    f"except Exception:\n"
                    f"    df = pd.DataFrame({{'item_id': [1], 'timestamp': ['2025-01-01'], 'target': [1.0]}})\n"
                    "preds = pd.DataFrame({'prediction': [1.0] * len(df)})\n"
                    "preds.to_csv('out/predictions.csv', index=False)\n"
                    "with open('out/summary.json', 'w') as f:\n"
                    "    json.dump({'success': True, 'metrics': {'mae': 0.1}}, f)\n"
                    "print('SUCCESS')\n"
                )
                return schema(code=code, language="python", dependencies=[], status="generated")
                
            elif "autogluon.multimodal" in prompt:
                code = (
                    "import os, json\n"
                    "import pandas as pd\n"
                    "os.makedirs('out/models', exist_ok=True)\n"
                    "preds = pd.DataFrame({'prediction': [0, 1]})\n"
                    "preds.to_csv('out/predictions.csv', index=False)\n"
                    "with open('out/summary.json', 'w') as f:\n"
                    "    json.dump({'success': True, 'metrics': {'accuracy': 0.95}}, f)\n"
                    "print('SUCCESS')\n"
                )
                return schema(code=code, language="python", dependencies=[], status="generated")
                
            elif "FlagEmbedding" in prompt:
                code = (
                    "import os, json\n"
                    "os.makedirs('out', exist_ok=True)\n"
                    "results = {'query': 'test query', 'top_documents': ['doc1', 'doc2']}\n"
                    "with open('out/retrieval_results.json', 'w') as f:\n"
                    "    json.dump(results, f)\n"
                    "with open('out/summary.json', 'w') as f:\n"
                    "    json.dump({'success': True, 'metrics': {'mrr': 0.9}}, f)\n"
                    "print('SUCCESS')\n"
                )
                return schema(code=code, language="python", dependencies=[], status="generated")
                
            elif "General ML" in prompt or "scikit-learn" in prompt or "machine learning" in prompt:
                code = (
                    "import os, json\n"
                    "import pandas as pd\n"
                    "os.makedirs('out', exist_ok=True)\n"
                    "preds = pd.DataFrame({'prediction': [0, 1]})\n"
                    "preds.to_csv('out/predictions.csv', index=False)\n"
                    "with open('out/summary.json', 'w') as f:\n"
                    "    json.dump({'success': True, 'metrics': {'f1': 0.88}}, f)\n"
                    "print('SUCCESS')\n"
                )
                return schema(code=code, language="python", dependencies=[], status="generated")

            # Extract target column (from task specifications or json)
            target_col = "target"
            target_match = re.search(r'Target Column:\s*([^\n\r]+)', prompt)
            if target_match:
                target_col = target_match.group(1).strip()
            else:
                target_json_match = re.search(r'"target_column":\s*"([^"]+)"', prompt)
                if target_json_match:
                    target_col = target_json_match.group(1).strip()
                    
            # Extract task type
            task_type = "classification"
            task_match = re.search(r'Task Type:\s*([^\n\r]+)', prompt)
            if task_match:
                task_type = task_match.group(1).strip()
            else:
                task_json_match = re.search(r'"task_type":\s*"([^"]+)"', prompt)
                if task_json_match:
                    task_type = task_json_match.group(1).strip()
                    
            problem_type_arg = f", problem_type='{task_type}'" if task_type == "regression" else ""
            
            # Check if there are data quality issues in context or previous error
            has_dq_issues = bool(
                re.search(r'"invalid_numeric_values":\s*\[\{', prompt) or
                re.search(r'"malformed_target_values":\s*\["[^"]+"', prompt) or
                re.search(r'"missing_values":\s*\[\{', prompt) or
                "DataQuality" in prompt or
                "Coerce" in prompt
            )
            
            if "PREVIOUS FAILURE" in prompt:
                # Iteration 2+
                if has_dq_issues:
                    # Apply data quality cleaning and train AutoGluon
                    code = (
                        "import pandas as pd\n"
                        "import json\n"
                        "from autogluon.tabular import TabularPredictor\n\n"
                        "import os\n"
                        f"train_df = pd.read_csv('{train_path}')\n"
                        f"test_df = pd.read_csv('{test_path}') if os.path.exists('{test_path}') else train_df\n\n"
                        "# Data quality preprocessing\n"
                        f"train_df['{target_col}'] = pd.to_numeric(train_df['{target_col}'], errors='coerce')\n"
                        f"train_df = train_df.dropna(subset=['{target_col}'])\n\n"
                        f"feature_cols = [c for c in train_df.columns if c != '{target_col}']\n"
                        "for col in feature_cols:\n"
                        "    s_num = pd.to_numeric(train_df[col], errors='coerce')\n"
                        "    if s_num.notna().sum() / len(train_df) >= 0.5:\n"
                        "        train_df[col] = s_num\n"
                        "        median_val = train_df[col].median()\n"
                        "        train_df[col] = train_df[col].fillna(median_val)\n"
                        "        if col in test_df.columns:\n"
                        "            test_df[col] = pd.to_numeric(test_df[col], errors='coerce').fillna(median_val)\n\n"
                        f"test_features = test_df.drop(columns=['{target_col}']) if '{target_col}' in test_df.columns else test_df\n\n"
                        f"predictor = TabularPredictor(label='{target_col}'{problem_type_arg}, path='out/models').fit(train_df, time_limit=10, presets='medium_quality')\n"
                        "preds = predictor.predict(test_features)\n"
                        "preds.to_csv('out/predictions.csv', index=False)\n"
                        "metrics = predictor.evaluate(train_df)\n"
                        "summary = {'success': True, 'model_path': 'out/models', 'metrics': metrics}\n"
                        "with open('out/summary.json', 'w') as f:\n"
                        "    json.dump(summary, f)\n"
                        "print('SUCCESS')\n"
                    )
                else:
                    # Clean dataset (e.g. tiny_classification): use detected target column
                    code = (
                        "import pandas as pd\n"
                        "import os, json\n"
                        "from autogluon.tabular import TabularPredictor\n"
                        f"train_df = pd.read_csv('{train_path}')\n"
                        f"test_df = pd.read_csv('{test_path}') if os.path.exists('{test_path}') else train_df\n"
                        f"predictor = TabularPredictor(label='{target_col}'{problem_type_arg}, path='out/models').fit(train_df, time_limit=10, presets='medium_quality')\n"
                        "preds = predictor.predict(test_df)\n"
                        "preds.to_csv('out/predictions.csv', index=False)\n"
                        "metrics = predictor.evaluate(train_df)\n"
                        "summary = {'success': True, 'model_path': 'out/models', 'metrics': metrics}\n"
                        "with open('out/summary.json', 'w') as f:\n"
                        "    json.dump(summary, f)\n"
                        "print('SUCCESS')\n"
                    )
            elif has_dq_issues:
                # Iteration 1 for datasets with data quality issues:
                # Direct attempt without preprocessing; fails authentically on the dirty data
                code = (
                    "import pandas as pd\n"
                    "import os, json\n"
                    "from autogluon.tabular import TabularPredictor\n"
                    f"train_df = pd.read_csv('{train_path}')\n"
                    f"test_df = pd.read_csv('{test_path}') if os.path.exists('{test_path}') else train_df\n"
                    f"predictor = TabularPredictor(label='{target_col}'{problem_type_arg}, path='out/models').fit(train_df, time_limit=10, presets='medium_quality')\n"
                    "preds = predictor.predict(test_df)\n"
                    "preds.to_csv('out/predictions.csv', index=False)\n"
                    "print('SUCCESS')\n"
                )
            elif "Mock Objective" in prompt or "task_type" in prompt or "Target Column:" in prompt:
                # Iteration 1 for clean dataset (e.g. tiny_classification):
                # Deliberate label typo to test error recovery
                typo_label = "targt" if target_col == "target" else f"{target_col}_typo"
                code = (
                    "import pandas as pd\n"
                    "from autogluon.tabular import TabularPredictor\n"
                    f"train_df = pd.read_csv('{train_path}')\n"
                    "# Intentional error: wrong label\n"
                    f"predictor = TabularPredictor(label='{typo_label}', path='out/models').fit(train_df, time_limit=10)\n"
                )
            else:
                # Generic fallback for empty context unit tests
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
            import re
            
            # Extract target column from perceptual context in prompt if present
            target_col = "target"
            target_match = re.search(r'"target_column":\s*"([^"]+)"', prompt)
            if target_match:
                target_col = target_match.group(1)
            else:
                target_match2 = re.search(r'Target Column:\s*([^\n\r]+)', prompt)
                if target_match2:
                    target_col = target_match2.group(1).strip()

            prompt_lower = prompt.lower()
            
            # 1. Check for KeyError (e.g. label typo like 'targt' or missing column)
            keyerror_match = re.search(r"KeyError:\s*['\"]([^'\"]+)['\"]", prompt)
            if keyerror_match:
                missing_key = keyerror_match.group(1)
                return schema(
                    iteration=1,
                    error_category="KeyError",
                    error_summary=f"Target column '{missing_key}' not found in DataFrame.",
                    error_message=f"KeyError: '{missing_key}' not found in DataFrame.",
                    stderr_excerpt=f"KeyError: '{missing_key}'",
                    suggested_fix=f"Correct the label column name to '{target_col}'."
                )
            
            # 2. Check for DataQuality / ValueError / conversion issues
            if any(err in prompt_lower for err in [
                "could not convert string to float",
                "cannot multiply sequence",
                "trainer has no fit models",
                "dataquality",
                "not_available",
                "valueerror"
            ]):
                return schema(
                    iteration=1,
                    error_category="DataQuality",
                    error_summary="Malformed non-numeric value in numeric/target field.",
                    error_message="Non-numeric or malformed values in numeric/target columns prevent model fitting.",
                    stderr_excerpt="ValueError: could not convert string to float",
                    suggested_fix="Coerce invalid numeric values to NaN, drop rows with invalid target values, and impute missing feature values before training."
                )
                
            # 3. Check for FileNotFoundError
            if "filenotfound" in prompt_lower or "no such file" in prompt_lower:
                return schema(
                    iteration=1,
                    error_category="FileNotFoundError",
                    error_summary="Input dataset file not found.",
                    error_message="Input dataset file not found.",
                    stderr_excerpt="FileNotFoundError",
                    suggested_fix="Verify and provide correct file paths."
                )

            # 4. Check for NameError
            name_err_match = re.search(r"NameError:\s*([^\n]+)", prompt)
            if name_err_match:
                msg = name_err_match.group(0)
                return schema(
                    iteration=1,
                    error_category="NameError",
                    error_summary=f"NameError: {msg}",
                    error_message=msg,
                    stderr_excerpt=msg,
                    suggested_fix="Define missing variables or import required modules."
                )

            # Generic fallback: if KeyError in test_orchestration
            if "keyerror" in prompt_lower:
                return schema(
                    iteration=1,
                    error_category="KeyError",
                    error_summary="KeyError in DataFrame access.",
                    error_message="KeyError in DataFrame access.",
                    stderr_excerpt="KeyError",
                    suggested_fix=f"Correct the label column name to '{target_col}'."
                )

            return schema(
                iteration=1,
                error_category="RuntimeError",
                error_summary="Execution failed due to runtime error.",
                error_message="Execution failed due to runtime error.",
                stderr_excerpt=prompt[-200:],
                suggested_fix="Check logs and resolve runtime failure."
            )
        elif schema_name == "ExecutionDecision":
            prompt_lower = prompt.lower()
            
            # Check for failure indicators in prompt
            has_failure = (
                "success flag: false" in prompt_lower or
                "status: failure" in prompt_lower or
                "status: timeout" in prompt_lower or
                "status: invalid_output" in prompt_lower or
                "keyerror" in prompt_lower or
                "valueerror" in prompt_lower or
                "traceback" in prompt_lower or
                "timed out" in prompt_lower or
                "missing expected" in prompt_lower or
                "exit code: 1" in prompt_lower or
                "exit code: 2" in prompt_lower
            )
            if "stdout:\nsuccess" in prompt_lower or ("exit code: 0" in prompt_lower and not has_failure):
                return schema(
                    decision="FINISH",
                    reason="Execution completed successfully with expected outputs.",
                    confidence=1.0,
                )

            issue = "Execution failure"
            if "keyerror" in prompt_lower:
                issue = "KeyError in DataFrame access"
            elif "could not convert string to float" in prompt_lower or "dataquality" in prompt_lower:
                issue = "Data quality failure in numeric conversion"
            elif "timed out" in prompt_lower:
                issue = "Execution timed out"
            elif "missing expected" in prompt_lower:
                issue = "Missing expected output files"

            return schema(
                decision="FIX",
                reason=f"Execution encountered errors: {issue}.",
                confidence=0.95,
                issue_summary=issue,
            )
        elif schema_name == "SummaryResult":
            if "label" in prompt.lower() or "keyerror" in prompt.lower():
                return schema(summary="AutoGluon Tabular requires exact label column matching. Verify target column exists in DataFrame.")
            return schema(summary="Mock summary of documentation.")
        elif schema_name == "CondensationResult":
            if "label" in prompt.lower() or "keyerror" in prompt.lower():
                return schema(condensed_text="1. Verify label column exists in train_df. 2. Pass exact label name to TabularPredictor(label=target).")
            return schema(condensed_text="Mock condensed implementation guidance.")
            
        # Generic fallback using field defaults or empty values
        # In a real mock, you'd inspect fields, but for our specific phase, 
        # the above covers the used schemas.
        raise NotImplementedError(f"Mock response not configured for {schema_name}")


class RealLLMClient(LLMClient):
    """
    Real LLM Client using Google Gemini via OpenAI-compatible endpoint.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout: int | None = None,
        retry_count: int | None = None,
    ):
        from mlzero.core.config import settings

        self.api_key = api_key or settings.gemini_api_key
        if not self.api_key:
            raise ValueError(
                "Gemini API key is required for RealLLMClient. "
                "Set GEMINI_API_KEY in environment or .env file."
            )
        self.model = model or settings.real_llm_model or settings.llm.model
        raw_url = base_url or settings.real_llm_base_url
        self.base_url = raw_url.rstrip("/") + "/"
        self.timeout = timeout or settings.llm.timeout
        self.retry_count = retry_count or settings.llm.retry_count

    def _sanitize(self, text: str) -> str:
        """Sanitize any occurrence of the API key from text or exceptions."""
        if self.api_key and self.api_key in text:
            return text.replace(self.api_key, "[REDACTED]")
        return text

    def generate_text(self, prompt: str) -> str:
        """Generate text from Gemini via OpenAI-compatible endpoint."""
        url = f"{self.base_url}chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
        }

        last_error: Exception | None = None
        for attempt in range(self.retry_count):
            try:
                with httpx.Client(timeout=float(self.timeout)) as client:
                    response = client.post(url, headers=headers, json=payload)
                    response.raise_for_status()
                    data = response.json()
                    return str(data["choices"][0]["message"]["content"])
            except httpx.HTTPStatusError as e:
                last_error = e
                if e.response.status_code in (429, 500, 502, 503, 504) and attempt < self.retry_count - 1:
                    time.sleep(2 ** attempt)
                    continue
                msg = self._sanitize(f"HTTP error {e.response.status_code}: {e.response.text}")
                raise RuntimeError(f"Real LLM request failed: {msg}") from None
            except Exception as e:  # noqa: BLE001
                last_error = e
                if attempt < self.retry_count - 1:
                    time.sleep(2 ** attempt)
                    continue
                msg = self._sanitize(str(e))
                raise RuntimeError(f"Real LLM request failed: {msg}") from None

        msg = self._sanitize(str(last_error))
        raise RuntimeError(f"Real LLM request failed after {self.retry_count} attempts: {msg}") from None

    def generate_structured(self, prompt: str, schema: type[T]) -> T:
        """
        Generate structured output from Gemini matching a Pydantic schema.
        Retries on JSON decoding, schema validation failure, or transient HTTP errors.
        """
        url = f"{self.base_url}chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        schema_json = json.dumps(schema.model_json_schema(), indent=2)
        system_instruction = (
            "You are a helpful AI assistant that outputs strictly valid JSON matching the following JSON Schema.\n"
            f"JSON Schema:\n{schema_json}\n"
            "Do NOT include any markdown formatting, do NOT wrap the output in ```json ... ```, and output ONLY the raw JSON object."
        )

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
        }

        last_error: Exception | None = None
        for attempt in range(self.retry_count):
            try:
                with httpx.Client(timeout=float(self.timeout)) as client:
                    response = client.post(url, headers=headers, json=payload)
                    response.raise_for_status()
                    data = response.json()
                    raw_content = str(data["choices"][0]["message"]["content"])

                # Clean markdown fences if present
                clean_content = raw_content.strip()
                match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", clean_content, re.DOTALL)
                if match:
                    clean_content = match.group(1)

                return schema.model_validate_json(clean_content)
            except httpx.HTTPStatusError as e:
                last_error = e
                if e.response.status_code in (429, 500, 502, 503, 504) and attempt < self.retry_count - 1:
                    time.sleep(2 ** attempt)
                    continue
                msg = self._sanitize(f"HTTP error {e.response.status_code}: {e.response.text}")
                raise RuntimeError(f"Real LLM structured generation failed: {msg}") from None
            except Exception as e:  # noqa: BLE001
                last_error = e
                # Update messages in retry to remind strict JSON adherence
                if attempt < self.retry_count - 1:
                    time.sleep(1)
                    payload["messages"].append(
                        {
                            "role": "user",
                            "content": f"The previous response failed validation: {self._sanitize(str(e))}. Please output valid JSON matching the schema strictly.",
                        }
                    )

        msg = self._sanitize(str(last_error))
        raise RuntimeError(
            f"Real LLM failed to generate valid structured response after {self.retry_count} attempts: {msg}"
        ) from None


def get_llm_client(use_mock: bool | None = None, mode: str | None = None) -> LLMClient:
    """Factory to get the appropriate LLM client."""
    from mlzero.core.config import settings

    resolved_mode = "mock"
    if mode is not None:
        resolved_mode = mode.lower()
    elif use_mock is not None:
        resolved_mode = "mock" if use_mock else "real"
    elif hasattr(settings, "llm_mode") and settings.llm_mode:
        resolved_mode = settings.llm_mode.lower()

    if resolved_mode == "mock":
        return MockLLMClient()
    elif resolved_mode == "real":
        return RealLLMClient()
    else:
        raise ValueError(f"Unknown LLM mode: '{resolved_mode}'. Must be 'mock' or 'real'.")

