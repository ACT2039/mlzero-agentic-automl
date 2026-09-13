# Implementation Notes

## Phase Summary

| Phase | Component | Key Files |
|---|---|---|
| 1 | Foundation | `mlzero/core/config.py`, `mlzero/core/llm.py`, `mlzero/core/logger.py` |
| 2 | Perception | `mlzero/agents/perception.py`, `mlzero/schemas/perception.py` |
| 3 | Coder + Executor | `mlzero/agents/coder.py`, `mlzero/tools/python_runner.py` |
| 4 | Error Analyzer | `mlzero/agents/coder.py::ErrorAnalyzerAgent` |
| 5 | Semantic Memory | `mlzero/memory/semantic.py`, `mlzero/memory/index.py`, `mlzero/memory/ingestion.py` |
| 6 | Episodic Memory | `mlzero/memory/episodic.py`, `mlzero/memory/episodic_store.py` |
| 7 | Real ML | `mlzero/tools/tabular.py`, `mlzero/core/llm.py::MockLLMClient` |
| 8 | API + UI | `mlzero/api/`, `mlzero/ui/`, `mlzero/application/` |
| 9 | Evaluation + Docs | `evaluation/`, `docs/`, `scripts/`, `reports/` |

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
