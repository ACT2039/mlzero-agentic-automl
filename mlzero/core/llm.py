"""
LLM Client Abstraction.
"""
import json
import logging
import re
import time
from abc import ABC, abstractmethod
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMClient(ABC):
    """Abstract interface for LLM interactions."""
    
    @abstractmethod
    def generate_text(self, prompt: str, max_tokens: int | None = None) -> str:
        """
        Generate raw text from the LLM.
        
        Args:
            prompt: The instruction and context.
            
        Returns:
            The raw text string response.
        """

    @abstractmethod
    def generate_structured(self, prompt: str, schema: type[T], max_tokens: int | None = None) -> T:
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

    def generate_text(self, prompt: str, max_tokens: int | None = None) -> str:
        """Returns deterministic mock text."""
        return "Mock LLM text response."
        
    def generate_structured(self, prompt: str, schema: type[T], max_tokens: int | None = None) -> T:
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
                    "    json.dump({'success': True, 'metrics': {'mae': 0.1, 'rmse': 0.12, 'r2': 0.85, 'mse': 0.014}}, f)\n"
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
                    "    json.dump({'success': True, 'metrics': {'accuracy': 0.95, 'balanced_accuracy': 0.94, 'f1': 0.95, 'precision': 0.96, 'recall': 0.94, 'mcc': 0.89}}, f)\n"
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
                    "    json.dump({'success': True, 'metrics': {'mrr': 0.9, 'ndcg': 0.88, 'map': 0.85, 'recall_at_k': 0.92}}, f)\n"
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
                    "    json.dump({'success': True, 'metrics': {'accuracy': 0.90, 'balanced_accuracy': 0.89, 'f1': 0.88, 'precision': 0.89, 'recall': 0.87, 'mcc': 0.79}}, f)\n"
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
                        "if len(train_df) > 2500:\n"
                        "    train_df = train_df.sample(n=2500, random_state=42)\n"
                        f"predictor = TabularPredictor(label='{target_col}'{problem_type_arg}, path='out/models').fit(train_df, time_limit=10, presets='medium_quality', excluded_model_types=['NN_TORCH', 'FASTAI'])\n"
                        "preds = predictor.predict(test_features)\n"
                        "preds.to_csv('out/predictions.csv', index=False)\n"
                        "metrics = predictor.evaluate(train_df)\n"
                        "try:\n"
                        "    import numpy as np\n"
                        "    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef, mean_squared_error, mean_absolute_error, r2_score\n"
                        f"    y_true = train_df['{target_col}']\n"
                        "    y_pred = predictor.predict(train_df)\n"
                        f"    if '{task_type}' == 'regression':\n"
                        "        metrics['rmse'] = float(np.sqrt(mean_squared_error(y_true, y_pred)))\n"
                        "        metrics['mae'] = float(mean_absolute_error(y_true, y_pred))\n"
                        "        metrics['r2'] = float(r2_score(y_true, y_pred))\n"
                        "        metrics['mse'] = float(mean_squared_error(y_true, y_pred))\n"
                        "    else:\n"
                        "        metrics['accuracy'] = float(accuracy_score(y_true, y_pred))\n"
                        "        metrics['balanced_accuracy'] = float(balanced_accuracy_score(y_true, y_pred))\n"
                        "        metrics['f1'] = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))\n"
                        "        metrics['precision'] = float(precision_score(y_true, y_pred, average='weighted', zero_division=0))\n"
                        "        metrics['recall'] = float(recall_score(y_true, y_pred, average='weighted', zero_division=0))\n"
                        "        metrics['mcc'] = float(matthews_corrcoef(y_true, y_pred))\n"
                        "except Exception:\n"
                        "    pass\n"
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
                        "if len(train_df) > 2500:\n"
                        "    train_df = train_df.sample(n=2500, random_state=42)\n"
                        f"predictor = TabularPredictor(label='{target_col}'{problem_type_arg}, path='out/models').fit(train_df, time_limit=10, presets='medium_quality', excluded_model_types=['NN_TORCH', 'FASTAI'])\n"
                        "preds = predictor.predict(test_df)\n"
                        "preds.to_csv('out/predictions.csv', index=False)\n"
                        "metrics = predictor.evaluate(train_df)\n"
                        "try:\n"
                        "    import numpy as np\n"
                        "    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef, mean_squared_error, mean_absolute_error, r2_score\n"
                        f"    y_true = train_df['{target_col}']\n"
                        "    y_pred = predictor.predict(train_df)\n"
                        f"    if '{task_type}' == 'regression':\n"
                        "        metrics['rmse'] = float(np.sqrt(mean_squared_error(y_true, y_pred)))\n"
                        "        metrics['mae'] = float(mean_absolute_error(y_true, y_pred))\n"
                        "        metrics['r2'] = float(r2_score(y_true, y_pred))\n"
                        "        metrics['mse'] = float(mean_squared_error(y_true, y_pred))\n"
                        "    else:\n"
                        "        metrics['accuracy'] = float(accuracy_score(y_true, y_pred))\n"
                        "        metrics['balanced_accuracy'] = float(balanced_accuracy_score(y_true, y_pred))\n"
                        "        metrics['f1'] = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))\n"
                        "        metrics['precision'] = float(precision_score(y_true, y_pred, average='weighted', zero_division=0))\n"
                        "        metrics['recall'] = float(recall_score(y_true, y_pred, average='weighted', zero_division=0))\n"
                        "        metrics['mcc'] = float(matthews_corrcoef(y_true, y_pred))\n"
                        "except Exception:\n"
                        "    pass\n"
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
                    "if len(train_df) > 2500:\n"
                    "    train_df = train_df.sample(n=2500, random_state=42)\n"
                    f"predictor = TabularPredictor(label='{target_col}'{problem_type_arg}, path='out/models').fit(train_df, time_limit=10, presets='medium_quality', excluded_model_types=['NN_TORCH', 'FASTAI'])\n"
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
                    f"predictor = TabularPredictor(label='{typo_label}', path='out/models').fit(train_df, time_limit=10, excluded_model_types=['NN_TORCH', 'FASTAI'])\n"
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
                target_match2 = re.search(r'Target(?: Column)?:\s*([^\n\r,]+)', prompt)
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
            
            # 2. Check for FileNotFoundError / invalid path issues
            if any(k in prompt_lower for k in ("filenotfound", "no such file", "invalid dataset path", "dataset path", "directory does not exist")):
                return schema(
                    iteration=1,
                    error_category="FileNotFoundError",
                    error_summary="Input dataset file or directory path not found.",
                    error_message="Input dataset file or directory path not found.",
                    stderr_excerpt="FileNotFoundError",
                    suggested_fix="Verify and provide correct file paths."
                )

            # 3. Check for DataQuality / ValueError / conversion issues
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


class LLMError(RuntimeError):
    """Base exception for LLM client failures."""


class TransientLLMError(LLMError):
    """Exception for transient/retryable provider failures (e.g. 429, 500, 502, 503, 504, timeout)."""


class PermanentLLMError(LLMError):
    """Exception for non-retryable provider failures (e.g. 401 Unauthorized, 400 Bad Request)."""


def sanitize_secrets(text: str, keys: list[str]) -> str:
    """Sanitize occurrences of secret API keys from text or exception messages."""
    result = text
    for key in keys:
        if key and len(key) > 4 and key in result:
            result = result.replace(key, "[REDACTED]")
    return result


class GeminiProviderClient(LLMClient):
    """Direct Google Gemini LLM Client using OpenAI-compatible endpoint."""

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
            raise PermanentLLMError(
                "Gemini API key is required. Set GEMINI_API_KEY in environment or .env file."
            )
        target_model = model or getattr(settings, "gemini_model", None) or settings.llm.model
        if not target_model or not target_model.startswith("gemini"):
            target_model = "gemini-3.6-flash"
        self.model = target_model
        raw_url = base_url or settings.real_llm_base_url
        self.base_url = raw_url.rstrip("/") + "/"
        self.timeout = timeout or settings.llm.timeout
        self.retry_count = retry_count or settings.llm.retry_count

    def _sanitize(self, text: str) -> str:
        return sanitize_secrets(text, [self.api_key])

    def generate_text(self, prompt: str, max_tokens: int | None = None) -> str:
        url = f"{self.base_url}chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
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
                status_code = e.response.status_code
                if status_code in (429, 500, 502, 503, 504) and attempt < self.retry_count - 1:
                    time.sleep(2 ** attempt)
                    continue
                msg = self._sanitize(f"Gemini HTTP {status_code}: {e.response.text}")
                if status_code in (429, 500, 502, 503, 504):
                    raise TransientLLMError(f"Gemini request failed: {msg}") from None
                raise PermanentLLMError(f"Gemini request failed: {msg}") from None
            except Exception as e:  # noqa: BLE001
                last_error = e
                if attempt < self.retry_count - 1:
                    time.sleep(2 ** attempt)
                    continue
                msg = self._sanitize(str(e))
                raise TransientLLMError(f"Gemini request failed: {msg}") from None

        msg = self._sanitize(str(last_error))
        raise TransientLLMError(f"Gemini request failed after {self.retry_count} attempts: {msg}") from None

    def generate_structured(self, prompt: str, schema: type[T], max_tokens: int | None = None) -> T:
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

                clean_content = raw_content.strip()
                match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", clean_content, re.DOTALL)
                if match:
                    clean_content = match.group(1)

                return schema.model_validate_json(clean_content)
            except httpx.HTTPStatusError as e:
                last_error = e
                status_code = e.response.status_code
                if status_code in (429, 500, 502, 503, 504) and attempt < self.retry_count - 1:
                    time.sleep(2 ** attempt)
                    continue
                msg = self._sanitize(f"Gemini HTTP {status_code}: {e.response.text}")
                if status_code in (429, 500, 502, 503, 504):
                    raise TransientLLMError(f"Gemini structured generation failed: {msg}") from None
                raise PermanentLLMError(f"Gemini structured generation failed: {msg}") from None
            except Exception as e:  # noqa: BLE001
                last_error = e
                if attempt < self.retry_count - 1:
                    time.sleep(1)
                    payload["messages"].append(
                        {
                            "role": "user",
                            "content": f"The previous response failed validation: {self._sanitize(str(e))}. Please output valid JSON matching the schema strictly.",
                        }
                    )

        msg = self._sanitize(str(last_error))
        raise TransientLLMError(
            f"Gemini failed to generate valid structured response after {self.retry_count} attempts: {msg}"
        ) from None


class OpenRouterProviderClient(LLMClient):
    """OpenRouter LLM Client using OpenAI-compatible endpoint."""
    
    _cached_free_models: list[str] | None = None

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout: int | None = None,
        retry_count: int | None = None,
    ):
        from mlzero.core.config import settings

        self.api_key = api_key or settings.openrouter_api_key
        if not self.api_key:
            raise PermanentLLMError(
                "OpenRouter API key is required. Set OPENROUTER_API_KEY in environment or .env file."
            )
        self.model = model or settings.openrouter_model or "openrouter/free"
        raw_url = base_url or settings.openrouter_base_url or "https://openrouter.ai/api/v1"
        self.base_url = raw_url.rstrip("/") + "/"
        self.timeout = timeout or settings.llm.timeout
        self.retry_count = retry_count or settings.llm.retry_count

    def _get_fallback_models(self) -> list[str]:
        if OpenRouterProviderClient._cached_free_models is not None:
            return OpenRouterProviderClient._cached_free_models

        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(f"{self.base_url}models")
                resp.raise_for_status()
                data = resp.json()

            free_models = []
            for m in data.get("data", []):
                pricing = m.get("pricing", {})
                if pricing.get("prompt") == "0" and pricing.get("completion") == "0":
                    supported = m.get("supported_parameters", [])
                    if "response_format" in supported or "structured_outputs" in supported:
                        free_models.append(m["id"])
            
            if not free_models:
                for m in data.get("data", []):
                    pricing = m.get("pricing", {})
                    if pricing.get("prompt") == "0" and pricing.get("completion") == "0":
                        free_models.append(m["id"])

            OpenRouterProviderClient._cached_free_models = free_models[:3]
            if not OpenRouterProviderClient._cached_free_models:
                OpenRouterProviderClient._cached_free_models = ["openrouter/free"]
            
            return OpenRouterProviderClient._cached_free_models
        except Exception:  # noqa: BLE001
            return ["openrouter/free"]

    def _sanitize(self, text: str) -> str:
        return sanitize_secrets(text, [self.api_key])

    def generate_text(self, prompt: str, max_tokens: int | None = None) -> str:
        url = f"{self.base_url}chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/mlzero-agentic-automl",
            "X-Title": "MLZero Agentic AutoML",
        }
        from mlzero.core.config import settings as _settings
        payload: dict[str, Any] = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens or _settings.llm.max_tokens,
        }
        if self.model == "openrouter/free":
            payload["models"] = self._get_fallback_models()
        else:
            payload["model"] = self.model

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
                status_code = e.response.status_code
                if status_code in (429, 500, 502, 503, 504) and attempt < self.retry_count - 1:
                    time.sleep(2 ** attempt)
                    continue
                msg = self._sanitize(f"OpenRouter HTTP {status_code}: {e.response.text}")
                if status_code in (429, 500, 502, 503, 504):
                    raise TransientLLMError(f"OpenRouter request failed: {msg}") from None
                raise PermanentLLMError(f"OpenRouter request failed: {msg}") from None
            except Exception as e:  # noqa: BLE001
                last_error = e
                if attempt < self.retry_count - 1:
                    time.sleep(2 ** attempt)
                    continue
                msg = self._sanitize(str(e))
                raise TransientLLMError(f"OpenRouter request failed: {msg}") from None

        msg = self._sanitize(str(last_error))
        raise TransientLLMError(f"OpenRouter request failed after {self.retry_count} attempts: {msg}") from None

    def generate_structured(self, prompt: str, schema: type[T], max_tokens: int | None = None) -> T:
        url = f"{self.base_url}chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/mlzero-agentic-automl",
            "X-Title": "MLZero Agentic AutoML",
        }
        schema_json = json.dumps(schema.model_json_schema(), indent=2)
        system_instruction = (
            "You are a helpful AI assistant that outputs strictly valid JSON matching the following JSON Schema.\n"
            f"JSON Schema:\n{schema_json}\n"
            "Do NOT include any markdown formatting, do NOT wrap the output in ```json ... ```, and output ONLY the raw JSON object.\n"
            "CRITICAL: Escape all newline characters as \\n in strings. Do not use literal newlines inside JSON strings!"
        )

        from mlzero.core.config import settings as _settings
        payload: dict[str, Any] = {
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
            "max_tokens": max_tokens or _settings.llm.max_tokens,
        }
        if self.model == "openrouter/free":
            payload["models"] = self._get_fallback_models()
        else:
            payload["model"] = self.model

        last_error: Exception | None = None
        for attempt in range(self.retry_count):
            try:
                with httpx.Client(timeout=float(self.timeout)) as client:
                    response = client.post(url, headers=headers, json=payload)
                    response.raise_for_status()
                    data = response.json()
                    raw_content = data["choices"][0]["message"]["content"]
                    if raw_content is None or (isinstance(raw_content, str) and raw_content.strip().lower() in ("none", "")):
                        raise ValueError("OpenRouter returned empty/null content")

                clean_content = str(raw_content).strip()
                match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", clean_content, re.DOTALL)
                if match:
                    clean_content = match.group(1)

                return schema.model_validate_json(clean_content)
            except httpx.HTTPStatusError as e:
                last_error = e
                status_code = e.response.status_code
                if status_code in (429, 500, 502, 503, 504) and attempt < self.retry_count - 1:
                    time.sleep(2 ** attempt)
                    continue
                msg = self._sanitize(f"OpenRouter HTTP {status_code}: {e.response.text}")
                if status_code in (429, 500, 502, 503, 504):
                    raise TransientLLMError(f"OpenRouter structured generation failed: {msg}") from None
                raise PermanentLLMError(f"OpenRouter structured generation failed: {msg}") from None
            except Exception as e:  # noqa: BLE001
                last_error = e
                if attempt < self.retry_count - 1:
                    time.sleep(1)
                    payload["messages"].append(
                        {
                            "role": "user",
                            "content": f"The previous response failed validation: {self._sanitize(str(e))}. Please output valid JSON matching the schema strictly.",
                        }
                    )

        msg = self._sanitize(str(last_error))
        raise TransientLLMError(
            f"OpenRouter failed to generate valid structured response after {self.retry_count} attempts: {msg}"
        ) from None


class FallbackLLMClient(LLMClient):
    """
    LLM Client wrapping a primary provider and an optional fallback provider.
    Automatically falls back to secondary provider on transient errors or provider capacity/rate failures.
    """

    def __init__(self, primary: LLMClient, fallback: LLMClient | None = None):
        self.primary = primary
        self.fallback = fallback

    def generate_text(self, prompt: str, max_tokens: int | None = None) -> str:
        try:
            return self.primary.generate_text(prompt, max_tokens=max_tokens)
        except TransientLLMError as e:
            if self.fallback is not None:
                from mlzero.core.logger import setup_logger
                logger = setup_logger(__name__)
                logger.warning(
                    f"Primary LLM provider failed with error: {e}. Falling back to secondary provider..."
                )
                try:
                    return self.fallback.generate_text(prompt, max_tokens=max_tokens)
                except Exception as fallback_err:  # noqa: BLE001
                    raise RuntimeError(
                        f"Both LLM providers failed. Primary error: {e} | Fallback error: {fallback_err}"
                    ) from None
            raise

    def generate_structured(self, prompt: str, schema: type[T], max_tokens: int | None = None) -> T:
        try:
            return self.primary.generate_structured(prompt, schema, max_tokens=max_tokens)
        except TransientLLMError as e:
            if self.fallback is not None:
                from mlzero.core.logger import setup_logger
                logger = setup_logger(__name__)
                logger.warning(
                    f"Primary LLM provider structured generation failed with error: {e}. Falling back to secondary provider..."
                )
                try:
                    return self.fallback.generate_structured(prompt, schema, max_tokens=max_tokens)
                except Exception as fallback_err:  # noqa: BLE001
                    raise RuntimeError(
                        f"Both LLM providers failed structured generation. Primary error: {e} | Fallback error: {fallback_err}"
                    ) from None
            raise


class RealLLMClient(LLMClient):
    """
    Backwards-compatible Real LLM Client supporting Gemini, OpenRouter, and fallback logic.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout: int | None = None,
        retry_count: int | None = None,
        provider: str | None = None,
    ):
        from mlzero.core.config import settings

        selected_provider = (provider or settings.real_llm_provider or "groq").lower()
        
        primary_client: LLMClient
        fallback_client: LLMClient | None = None

        def safe_gemini() -> LLMClient | None:
            if settings.gemini_api_key and settings.gemini_api_key.strip():
                gem_m = getattr(settings, "gemini_model", None) or "gemini-3.6-flash"
                try: return GeminiProviderClient(
                    model=gem_m,
                    timeout=timeout, retry_count=retry_count,
                )
                except Exception: return None  # noqa: BLE001
            return None

        def safe_openrouter() -> LLMClient | None:
            if settings.openrouter_api_key and settings.openrouter_api_key.strip():
                try: return OpenRouterProviderClient(timeout=timeout, retry_count=retry_count)
                except Exception: return None  # noqa: BLE001
            return None

        if selected_provider == "groq":
            primary_client = GroqProviderClient(
                api_key=api_key or settings.groq_api_key or None,
                model=model or settings.real_llm_model or "openai/gpt-oss-120b",
                timeout=timeout,
                retry_count=retry_count,
            )
            g_client = safe_gemini()
            o_client = safe_openrouter()
            if g_client and o_client:
                fallback_client = FallbackLLMClient(primary=g_client, fallback=o_client)
            elif g_client:
                fallback_client = g_client
            else:
                fallback_client = o_client
        elif selected_provider == "openrouter":
            primary_client = OpenRouterProviderClient(
                api_key=api_key or settings.openrouter_api_key or None,
                model=model or settings.openrouter_model or "openrouter/free",
                base_url=base_url or settings.openrouter_base_url,
                timeout=timeout,
                retry_count=retry_count,
            )
            fallback_client = safe_gemini()
        else:
            primary_client = GeminiProviderClient(
                api_key=api_key or settings.gemini_api_key or None,
                model=model or settings.real_llm_model or settings.llm.model,
                base_url=base_url or settings.real_llm_base_url,
                timeout=timeout,
                retry_count=retry_count,
            )
            fallback_client = safe_openrouter()

        self._client = FallbackLLMClient(primary=primary_client, fallback=fallback_client)

    @property
    def primary(self) -> LLMClient:
        return self._client.primary

    @property
    def fallback(self) -> LLMClient | None:
        return self._client.fallback

    def generate_text(self, prompt: str, max_tokens: int | None = None) -> str:
        return self._client.generate_text(prompt, max_tokens=max_tokens)

    def generate_structured(self, prompt: str, schema: type[T], max_tokens: int | None = None) -> T:
        return self._client.generate_structured(prompt, schema, max_tokens=max_tokens)


def get_llm_client(
    use_mock: bool | None = None,
    mode: str | None = None,
    provider: str | None = None,
) -> LLMClient:
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
        return RealLLMClient(provider=provider)
    else:
        raise ValueError(f"Unknown LLM mode: '{resolved_mode}'. Must be 'mock' or 'real'.")


class GroqProviderClient(LLMClient):
    """Groq LLM Client using OpenAI-compatible endpoint."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout: int | None = None,
        retry_count: int | None = None,
    ):
        from mlzero.core.config import settings
        
        self.api_key = api_key or settings.groq_api_key
        if not self.api_key:
            raise PermanentLLMError(
                "Groq API key is required. Set GROQ_API_KEY in environment or .env file."
            )
        self.model = model or settings.real_llm_model or "openai/gpt-oss-20b"
        self.base_url = "https://api.groq.com/openai/v1/"
        self.timeout = timeout or settings.llm.timeout
        self.retry_count = retry_count or settings.llm.retry_count

    def _sanitize(self, text: str) -> str:
        return sanitize_secrets(text, [self.api_key])

    def generate_text(self, prompt: str, max_tokens: int | None = None) -> str:
        url = f"{self.base_url}chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "MLZero/1.0"
        }
        from mlzero.core.config import settings as _settings
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": getattr(_settings.llm, "temperature", 0.0),
            "max_tokens": max_tokens or _settings.llm.max_tokens,
        }

        last_error: Exception | None = None
        for attempt in range(self.retry_count):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(url, headers=headers, json=payload)
                    response.raise_for_status()
                    data = response.json()
                    return str(data["choices"][0]["message"]["content"])
            except httpx.HTTPStatusError as e:
                last_error = e
                status_code = e.response.status_code
                response_text = e.response.text
                msg = self._sanitize(f"Groq HTTP {status_code}: {response_text}")
                
                is_tpm_or_413 = (
                    status_code == 413
                    or "request too large" in response_text.lower()
                )
                if is_tpm_or_413:
                    raw_max = payload.get("max_tokens")
                    curr_max = int(raw_max) if isinstance(raw_max, (int, float)) else 2048
                    if curr_max > 1200 and attempt < self.retry_count - 1:
                        payload["max_tokens"] = 1200
                        time.sleep(1)
                        continue
                    from mlzero.core.config import settings as _settings
                    alt_models = [m for m in getattr(_settings, "groq_fallback_models", []) if m != payload["model"]]
                    if alt_models and attempt < self.retry_count - 1:
                        payload["model"] = alt_models[0]
                        self.model = alt_models[0]
                        time.sleep(1)
                        continue
                    raise TransientLLMError(f"Groq request failed due to token limits: {msg}") from None

                if status_code == 429:
                    import re
                    if "tokens per day" in response_text or "TPD" in response_text:
                        from mlzero.core.config import settings as _settings
                        alt_models = [m for m in getattr(_settings, "groq_fallback_models", []) if m != payload["model"]]
                        if alt_models and attempt < self.retry_count - 1:
                            payload["model"] = alt_models[0]
                            self.model = alt_models[0]
                            time.sleep(1)
                            continue
                    match = re.search(r"Please try again in ([\d\.]+)s", response_text)
                    if match and attempt < self.retry_count - 1:
                        delay = float(match.group(1)) + 0.5
                        if delay <= 5.0:
                            time.sleep(delay)
                            continue
                    raise TransientLLMError(f"Groq request failed: {msg}") from None
                    
                if status_code in (500, 502, 503, 504):
                    if attempt < self.retry_count - 1:
                        time.sleep(2 ** attempt)
                        continue
                    raise TransientLLMError(f"Groq request failed: {msg}") from None
                raise PermanentLLMError(f"Groq request failed: {msg}") from None
            except Exception as e:  # noqa: BLE001
                last_error = e
                if attempt < self.retry_count - 1:
                    time.sleep(2 ** attempt)
                    continue
                msg = self._sanitize(str(e))
                raise TransientLLMError(f"Groq request failed: {msg}") from None

        msg = self._sanitize(str(last_error))
        raise TransientLLMError(f"Groq request failed after {self.retry_count} attempts: {msg}") from None

    def _make_strict(self, schema_obj: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(schema_obj, dict):
            return schema_obj
        if schema_obj.get('type') == 'object' or 'properties' in schema_obj:
            schema_obj['additionalProperties'] = False
            props = schema_obj.get('properties', {})
            if props:
                schema_obj['required'] = list(props.keys())
                for v in props.values():
                    self._make_strict(v)
        elif schema_obj.get('type') == 'array':
            if 'items' in schema_obj:
                self._make_strict(schema_obj['items'])
        elif 'anyOf' in schema_obj:
            for opt in schema_obj['anyOf']:
                self._make_strict(opt)
                
        if '$defs' in schema_obj:
            for v in schema_obj['$defs'].values():
                self._make_strict(v)
        return schema_obj

    def generate_structured(self, prompt: str, schema: type[T], max_tokens: int | None = None) -> T:
        url = f"{self.base_url}chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "MLZero/1.0"
        }
        
        import copy
        schema_dict = copy.deepcopy(schema.model_json_schema())
        schema_dict = self._make_strict(schema_dict)
        
        from mlzero.core.config import settings as _settings
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant that outputs strictly valid JSON matching the requested schema. Do NOT include any markdown formatting, do NOT wrap the output in ```json ... ```, and output ONLY the raw JSON object. CRITICAL: Escape all newline characters as \\n in strings. Do not use literal newlines inside JSON strings!"},
                {"role": "user", "content": prompt},
            ],
            "response_format": {
                "type": "json_schema", 
                "json_schema": {
                    "name": schema.__name__, 
                    "schema": schema_dict, 
                    "strict": True
                }
            },
            "temperature": getattr(_settings.llm, "temperature", 0.0),
            "max_tokens": max_tokens or _settings.llm.max_tokens,
        }

        last_error: Exception | None = None
        for attempt in range(self.retry_count):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(url, headers=headers, json=payload)
                    response.raise_for_status()
                    data = response.json()
                    content = data["choices"][0]["message"]["content"]
                    
                    try:
                        return schema.model_validate_json(content)
                    except ValidationError as e:
                        last_error = e
                        if attempt == self.retry_count - 1:
                            break
                        # Help the model fix it
                        msgs = payload.get("messages")
                        if isinstance(msgs, list):
                            msgs.extend(
                                [
                                    {"role": "assistant", "content": content},
                                    {
                                        "role": "user",
                                        "content": f"The previous response failed validation: {self._sanitize(str(e))}. Please output valid JSON matching the schema strictly.",
                                    }
                                ]
                            )
            except httpx.HTTPStatusError as e:
                last_error = e
                status_code = e.response.status_code
                response_text = e.response.text
                msg = self._sanitize(f"Groq HTTP {status_code}: {response_text}")
                
                # Check for json_validate_failed and recover candidate instance from failed_generation
                if status_code == 400 and ("failed_generation" in response_text or "json_validate_failed" in response_text):
                    try:
                        err_obj = json.loads(response_text)
                        failed_gen = err_obj.get("error", {}).get("failed_generation", "")
                        if failed_gen:
                            import re
                            sub_parts = re.split(r'\}\s*\{', failed_gen)
                            for part in reversed(sub_parts):
                                candidate = part.strip()
                                if not candidate.startswith('{'):
                                    candidate = '{' + candidate
                                if not candidate.endswith('}'):
                                    candidate = candidate + '}'
                                try:
                                    return schema.model_validate_json(candidate)
                                except ValidationError:
                                    continue
                    except Exception as exc:  # noqa: BLE001
                        logger.debug(f"Failed to extract JSON candidate from failed_generation: {exc}")
                    if attempt < self.retry_count - 1:
                        payload["response_format"] = {"type": "json_object"}
                        time.sleep(1)
                        continue

                is_tpm_or_413 = (
                    status_code == 413
                    or "request too large" in response_text.lower()
                    or "tokens per minute" in response_text.lower()
                    or "tpm" in response_text.lower()
                )
                if is_tpm_or_413:
                    raw_max = payload.get("max_tokens")
                    curr_max = int(raw_max) if isinstance(raw_max, (int, float)) else 2048
                    if curr_max > 1200 and attempt < self.retry_count - 1:
                        payload["max_tokens"] = 1200
                        time.sleep(1)
                        continue
                    from mlzero.core.config import settings as _settings
                    alt_models = [m for m in getattr(_settings, "groq_fallback_models", []) if m != payload["model"]]
                    if alt_models and attempt < self.retry_count - 1:
                        payload["model"] = alt_models[0]
                        self.model = alt_models[0]
                        time.sleep(1)
                        continue
                    raise TransientLLMError(f"Groq request failed due to token limits: {msg}") from None

                if status_code == 429:
                    import re
                    if "tokens per day" in response_text or "TPD" in response_text:
                        from mlzero.core.config import settings as _settings
                        alt_models = [m for m in getattr(_settings, "groq_fallback_models", []) if m != payload["model"]]
                        if alt_models and attempt < self.retry_count - 1:
                            payload["model"] = alt_models[0]
                            self.model = alt_models[0]
                            time.sleep(1)
                            continue
                    match = re.search(r"Please try again in ([\d\.]+)s", response_text)
                    if match and attempt < self.retry_count - 1:
                        delay = float(match.group(1)) + 0.5
                        if delay <= 5.0:
                            time.sleep(delay)
                            continue
                    raise TransientLLMError(f"Groq request failed: {msg}") from None

                if status_code in (500, 502, 503, 504):
                    if attempt < self.retry_count - 1:
                        time.sleep(2 ** attempt)
                        continue
                    raise TransientLLMError(f"Groq request failed: {msg}") from None
                raise PermanentLLMError(f"Groq request failed: {msg}") from None
            except Exception as e:  # noqa: BLE001
                last_error = e
                if attempt < self.retry_count - 1:
                    time.sleep(2 ** attempt)
                    continue
                msg = self._sanitize(str(e))
                raise TransientLLMError(f"Groq request failed: {msg}") from None

        msg = self._sanitize(str(last_error))
        raise TransientLLMError(f"Groq failed to generate valid structured response after {self.retry_count} attempts: {msg}") from None
