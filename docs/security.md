# Security Model

## Overview

MLZero implements a layered security model appropriate for a research/academic
prototype. It does **not** claim production-grade hardening.

---

## 1. Code Execution Sandbox

**Component**: `mlzero/tools/python_runner.py`

Generated ML code is executed as a subprocess, **not** via `eval()` or `exec()`.
This isolates the generated code from the main process.

Controls:
- Execution timeout (`settings.execution.timeout_seconds`, default 120 s)
- Maximum stdout size (`settings.execution.max_stdout_size_bytes`)
- Maximum stderr size (`settings.execution.max_stderr_size_bytes`)
- Output written only to a temporary workspace subdirectory

**Known limitation**: The subprocess runs in the same OS user context as the
main process. Full container/VM isolation is not implemented.

---

## 2. Path Traversal Prevention

**Component**: `mlzero/tools/python_runner.py`, `mlzero/api/routes.py`

Input filenames provided to `PythonRunner` are validated before writing:
```python
target_path = (temp_path / fname).resolve()
if temp_path.resolve() not in target_path.parents:
    raise ValueError(f"Invalid path {fname} escapes workspace.")
```

API dataset paths are validated against `settings.app.allowed_data_root`:
```python
resolved = Path(dataset_path).resolve()
allowed  = Path(settings.app.allowed_data_root).resolve()
if allowed not in resolved.parents and resolved != allowed:
    raise HTTPException(400, "Dataset path is outside allowed data root")
```

Attempts to supply `../../.env`, `/etc/passwd`, or absolute paths outside
the allowed root are rejected with HTTP 400.

---

## 3. Secret / Environment Variable Exposure

**Policy**:
- No API keys are committed to source code or configuration files.
- `.env` is listed in `.gitignore`.
- API responses never include environment variables, source code content,
  `.git` history, or model binary blobs.
- The `GET /runs/{run_id}/artifacts` endpoint returns relative path strings
  only — it does not stream file bytes or expose absolute filesystem paths.

**Phase 7 note**: `PythonRunner` uses `os.environ.copy()` to pass the
execution environment, which means the subprocess can access `OPENAI_API_KEY`
if set. This is required for AutoGluon to function when a real LLM key is
present. The `test_executor_security_isolation` test is skipped with an
explanatory note.

---

## 4. No Arbitrary Command Endpoint

The FastAPI API exposes only structured endpoints:

| Method | Path | Action |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/runs` | Submit ML run |
| GET | `/runs/{id}` | Get run status |
| GET | `/runs/{id}/episodes` | Get episodic history |
| GET | `/runs/{id}/artifacts` | Get artifact references |

There is no endpoint that executes arbitrary shell commands or exposes the
filesystem. All ML execution is routed through `MLZeroService` → `IterativeCodingOrchestrator` → `PythonRunner`.

---

## 5. Secrets Not in Reports

The evaluation framework and report generator do not include:
- Environment variables
- API keys
- Absolute filesystem paths (beyond relative artifact references)
- Model binary content

---

## 6. Temporary File Cleanup

`PythonRunner` deletes the temporary workspace after execution:
```python
shutil.rmtree(temp_path, ignore_errors=True)
```
Output artefacts are copied to `outputs/models/<exec_id>/` before cleanup.

---

## Residual Risks (Acknowledged)

| Risk | Status |
|---|---|
| Subprocess not containerised | Known; acceptable for research prototype |
| `os.environ.copy()` in executor | Known; required for AutoGluon |
| No rate limiting on API | Known; not required for academic demo |
| No authentication on API | Known; not required for academic demo |
| Single-user, local deployment | Intended scope |
