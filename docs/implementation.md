# Implementation Notes

## Stage Summary

| Stage | Component | Key Files |
|---|---|---|
| 1 | Real LLM Provider Foundation | `mlzero/core/config.py`, `mlzero/core/llm.py`, `mlzero/core/logger.py` |
| 2 | Semantic Memory Architecture | `mlzero/memory/semantic.py`, `mlzero/memory/index.py`, `mlzero/memory/ingestion.py` |
| 3 | Episodic Memory Architecture | `mlzero/memory/episodic.py`, `mlzero/memory/episodic_store.py` |
| 4 | Execution Judge & Retry Loop | `mlzero/agents/judge.py`, `mlzero/agents/coder.py`, `mlzero/orchestration/iterative.py` |
| 5 | Executable ML Library Adapters | `mlzero/tools/adapters/`, `mlzero/tools/python_runner.py` |
| 6 | Multimodal Dataset Perception | `mlzero/agents/perception.py`, `mlzero/schemas/perception.py` |
| 7 | Evaluation Framework | `evaluation/runner.py`, `evaluation/cases.py`, `evaluation/report.py` |

---

## Key Design Decisions

### 1. MockLLMClient for Offline Testing

All unit and integration tests use `MockLLMClient`, which returns deterministic
responses without network access. This ensures tests are:
- Reproducible
- Fast (< 5s for full suite)
- Free (no API cost)

### 2. TF-IDF as Default Semantic Index

FAISS requires native libraries. TF-IDF from scikit-learn works everywhere
and is sufficient for the small knowledge bases used in this project.

### 3. JSON Episodic Store

Episodes are serialised as JSON (not pickle) for safety, readability, and
portability. Each run gets its own `.json` file under `memory/episodic/`.

### 4. Subprocess Execution

Generated code is executed via `subprocess.run()`, not `eval()` or `exec()`.
This provides process-level isolation between the orchestrator and generated
ML code. The subprocess has a configurable timeout.

### 5. Single-Singleton RunManager

`RunManager` is a module-level singleton. This means run state is
process-local and not shared across multiple uvicorn workers or restarts.
This is documented as a known limitation.

### 6. Pydantic v2 for All Schemas

All data transfer objects use Pydantic v2 models. This provides:
- Automatic validation
- JSON serialisation via `.model_dump_json()`
- Clear schema documentation

### 7. Paper-Faithful Semantic Memory (Stage 2)

Semantic Memory aligns with the MLZero NeurIPS 2025 architecture:
- **SummarizationAgent**: Offline summarizer creating concise summaries of documentation chunks.
- **CondensationAgent**: Offline condenser distilling actionable implementation guidance and code patterns.
- **RetrievalAgent**: Online retrieval agent constructing error-aware queries at each coding iteration.
- **Top-5 Retrieval**: Retrieves 5 documents by default (`retrieval_top_k = 5`).
- **Segregated Coder Prompt**: Partitioned into 4 explicit sections: Perceptual Context, User Instruction, Semantic Memory (Top-5 docs), and Episodic / Error Recovery Context.
- **Library Knowledge Registry**: Clean registration and ingestion for `autogluon.tabular`, `autogluon.multimodal`, `autogluon.timeseries`, and `general_ml`.

### 8. Paper-Faithful Episodic Memory (Stage 3)

Episodic Memory aligns with the MLZero NeurIPS 2025 formulation $E_t(P, C_{t-1}, L_{t-1}, G_{t-1}, R_{t-1}) = R_t$:
- **Chronological History**: `get_history(run_id)` returns the ordered sequence of episodes preserving iteration progression.
- **Code & Log Persistence**: Generated code ($C$) is stored for both failed and successful iterations. Execution logs ($L$) capture stdout, stderr, exit codes, and runtime durations with configurable bounded limits.
- **Retrieved Knowledge Storage**: Preserves exact semantic retrieval queries and returned guidance ($G_{t-1}$) for every iteration.
- **Error Context Distillation**: `ErrorAnalyzerAgent` produces structured `ErrorContext` instances ($R_t$) containing category, summary, message, suggested fix, and stderr excerpts.
- **Closed Feedback Loop**: Error context from iteration $t-1$ directly seeds semantic retrieval and Coder prompt generation in iteration $t$.
- **Atomic Persistence**: `EpisodicStore` uses temporary file staging and atomic replacement to prevent disk corruption.

### 9. Paper-Faithful Execution Judge & Iterative Coding (Stage 4)

The iterative coding module follows the MLZero NeurIPS 2025 paper:
- **ExecutionJudgeAgent**: Decides whether execution should `FINISH` or `FIX`. Structured output includes decision, reason, confidence, and issue summary.
- **Strict Separation of Concerns**: `ExecutionJudgeAgent` determines whether to proceed or fix; `ErrorAnalyzerAgent` only runs when `FIX` is chosen to diagnose the issue and propose fixes.
- **Execution Validation**: Evaluates exit code, runtime errors, timeout conditions (`TIMEOUT`), and expected output files (`INVALID_OUTPUT`).
- **Optional Bash Scripts**: `CodeArtifact` supports `bash_script` (stored as `setup.sh`), executed safely in controlled subprocesses.
- **Episodic & Semantic Feedback**: When `FIX` is decided, the error context is recorded in Episodic Memory and used for error-aware Semantic Retrieval, providing targeted guidance for the next Coder iteration.

### 10. Executable ML Library Adapters & Pipeline Expansion (Stage 5)

The executable ML library layer expands beyond `autogluon.tabular` using a modular provider/adapter architecture (`MLLibraryAdapter`):
- **`autogluon.tabular`**: Classification and regression tasks with TabularPredictor. Handles dirty data, missing values, and malformed target values.
- **`autogluon.multimodal`**: Controlled subset of multimodal tasks (image classification, text classification, mixed tabular+text/image). Uses `MultiModalPredictor`.
- **`autogluon.timeseries`**: Time-series forecasting with `TimeSeriesPredictor` and `TimeSeriesDataFrame` over timestamp and item_id columns.
- **`FlagEmbedding`**: Dense retrieval and reranking tasks over document corpus and query text.
- **`general_ml`**: Lightweight `scikit-learn` fallback for generic machine learning tasks.
- **Library Selection & Discovery**: `LibrarySelectorAgent` metadata registry and `AdapterRegistry` map perceptual task context directly to library adapters.
- **Library-Aware Prompts & Memory**: Coder prompt receives adapter-specific coding guidance, and `SemanticMemory` performs library-filtered top-5 knowledge retrieval.

---

## Configuration

All configurable parameters live in `mlzero/core/config.py` via `AppConfig`
(backed by Pydantic Settings). Environment variables or a `.env` file
override defaults.

Key settings:
- `MLZERO_EXECUTION__TIMEOUT_SECONDS` — subprocess timeout
- `MLZERO_APP__ALLOWED_DATA_ROOT` — API path security boundary
- `MLZERO_API__HOST` / `MLZERO_API__PORT` — FastAPI server bind
- `MLZERO_ML__OUTPUT_DIR` — where models/predictions are saved

---

## Testing Strategy

| Test Type | Location | Runs ML? | Requires API Key? |
|---|---|---|---|
| Unit | `tests/unit/` | No | No |
| API | `tests/unit/test_api.py` | No | No |
| Application | `tests/unit/test_application.py` | Mock only | No |
| Integration | `tests/integration/test_ml_pipeline.py` | Yes (tiny) | No |

Integration tests are skipped unless `RUN_ML_INTEGRATION=1` is set.
