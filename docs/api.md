# API Reference

## Base URL

When running locally:
```
http://localhost:8000
```

Configure host/port in `.env` or via CLI flags.

---

## Endpoints

### GET /health

Health check.

**Response 200**:
```json
{"status": "ok", "version": "0.1.0"}
```

---

### POST /runs

Submit a new ML pipeline run.

**Request body** (`application/json`):
```json
{
  "dataset_path": "evaluation/datasets/binary_cls",
  "user_instruction": "Train a binary classifier. Label is 'target'.",
  "options": {
    "mock_llm": true,
    "max_iterations": 10
  }
}
```

**Response 200**:
```json
{"run_id": "uuid-v4", "status": "QUEUED"}
```

**Response 400** (path outside allowed root):
```json
{"detail": "Dataset path is outside allowed data root"}
```

---

### GET /runs/{run_id}

Get the status and result of a run.

**Response 200**:
```json
{
  "run_id": "uuid-v4",
  "status": "SUCCESS",
  "success": true,
  "task_summary": {"objective": "...", "task_type": "classification", ...},
  "selected_library": "autogluon.tabular",
  "iterations": 2,
  "final_metrics": {"accuracy": 0.75, "f1": 0.80},
  "prediction_artifact_reference": "exec_abc123/predictions.csv",
  "model_artifact_reference": "exec_abc123/models",
  "execution_duration": 3.92,
  "final_error": null
}
```

**Status values**: `QUEUED`, `RUNNING`, `SUCCESS`, `FAIL`

---

### GET /runs/{run_id}/episodes

Get the episodic memory history for a run.

**Response 200**:
```json
[
  {
    "iteration": 1,
    "success": false,
    "error_category": "KeyError",
    "suggested_fix": "Use correct label column name"
  },
  {
    "iteration": 2,
    "success": true,
    "error_category": null,
    "suggested_fix": null
  }
]
```

---

### GET /runs/{run_id}/artifacts

Get artifact reference paths for a completed run.

**Response 200**:
```json
{
  "run_id": "uuid-v4",
  "prediction_artifact_reference": "exec_abc123/predictions.csv",
  "model_artifact_reference": "exec_abc123/models"
}
```

**Note**: Only relative reference strings are returned. The API does not
stream file bytes or expose absolute filesystem paths.

---

## Security Notes

- Dataset paths are validated against `allowed_data_root` (default: project root).
- No endpoint exposes secrets, `.env`, `.git`, or source files.
- No arbitrary shell execution endpoint exists.
- No authentication is implemented (academic prototype).

---

## Example: cURL

```bash
# Health
curl http://localhost:8000/health

# Submit run
curl -X POST http://localhost:8000/runs \
  -H "Content-Type: application/json" \
  -d '{"dataset_path":"evaluation/datasets/binary_cls","options":{"mock_llm":true}}'

# Poll status
curl http://localhost:8000/runs/<run_id>

# Get episodes
curl http://localhost:8000/runs/<run_id>/episodes
```
