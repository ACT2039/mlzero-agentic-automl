import os
import shutil
import uuid
import zipfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from mlzero.application.manager import run_manager
from mlzero.core.config import settings
from mlzero.memory.episodic import EpisodicMemory
from mlzero.schemas.application import (
    ErrorResponse,
    RunRequest,
    RunResponse,
    RunStatusResponse,
)

router = APIRouter()

# ── Existing endpoints (backward-compatible, unchanged) ────────────────────

@router.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok", "version": settings.app.version}

@router.post("/runs", response_model=RunResponse, responses={400: {"model": ErrorResponse}})
def create_run(request: RunRequest) -> RunResponse:
    # Validate input
    allowed_root = Path(settings.app.allowed_data_root).resolve()
    input_path = Path(request.dataset_path).resolve()

    try:
        input_path.relative_to(allowed_root)
    except ValueError:
        raise HTTPException(status_code=400, detail="Dataset path is outside allowed data root")

    if not input_path.exists() or not input_path.is_dir():
        raise HTTPException(status_code=400, detail="Invalid dataset directory")

    if request.user_instruction and len(request.user_instruction) > settings.app.max_instruction_length:
        raise HTTPException(status_code=400, detail="User instruction exceeds maximum length")

    run_id = run_manager.submit_run(
        dataset_path=str(input_path),
        user_instruction=request.user_instruction,
        options=request.options
    )

    return RunResponse(run_id=run_id, status="QUEUED")

@router.get("/runs/{run_id}", response_model=RunStatusResponse, responses={404: {"model": ErrorResponse}})
def get_run(run_id: str) -> RunStatusResponse:
    status = run_manager.get_run_status(run_id)
    if not status:
        raise HTTPException(status_code=404, detail="Run not found")
    return status

@router.get("/runs/{run_id}/episodes")
def get_run_episodes(run_id: str) -> dict[str, Any]:
    # Check if run exists at all
    status = run_manager.get_run_status(run_id)
    if not status:
        raise HTTPException(status_code=404, detail="Run not found")

    ep_mem = EpisodicMemory()
    run = ep_mem.store.get_run_history(run_id)
    
    # Run exists but no history recorded yet
    if not run:
        return {"run_id": run_id, "episodes": []}
        
    return run.model_dump()

@router.get("/runs/{run_id}/artifacts")
def get_run_artifacts(run_id: str) -> dict[str, Any]:
    status = run_manager.get_run_status(run_id)
    if not status:
        raise HTTPException(status_code=404, detail="Run not found")

    return {
        "prediction_artifact_reference": status.prediction_artifact_reference,
        "model_artifact_reference": status.model_artifact_reference
    }


@router.get("/runs/{run_id}/predictions/download")
def download_run_predictions(run_id: str) -> FileResponse:
    """Download predictions.csv for a completed run."""
    status = run_manager.get_run_status(run_id)
    if not status:
        raise HTTPException(status_code=404, detail="Run not found")

    pred_path: Path | None = None
    if status.prediction_artifact_reference:
        p = Path(status.prediction_artifact_reference)
        if p.exists() and p.is_file():
            pred_path = p

    if not pred_path:
        fallback = Path(settings.app.artifact_root) / "runs" / run_id / "predictions.csv"
        if fallback.exists() and fallback.is_file():
            pred_path = fallback

    if not pred_path:
        ws_candidates = list(Path(settings.execution.workspace_root).glob(f"*{run_id}*/out/predictions.csv"))
        if not ws_candidates:
            ws_candidates = list(Path(settings.execution.workspace_root).glob("exec_*/out/predictions.csv"))
        if ws_candidates:
            pred_path = max(ws_candidates, key=lambda x: x.stat().st_mtime)

    if not pred_path or not pred_path.exists():
        raise HTTPException(status_code=404, detail="Prediction file not found for this run")

    return FileResponse(
        path=str(pred_path),
        media_type="text/csv",
        filename=f"predictions_{run_id[:8]}.csv",
    )


@router.post("/datasets/upload")
async def upload_dataset(file: UploadFile = File(...)) -> dict[str, Any]:  # noqa: B008
    """
    Securely upload a dataset (CSV, TSV, or ZIP) for agentic execution.
    Saves to an isolated directory under outputs/uploads/upload_<uuid>/.
    Enforces a 25MB file size limit and Zip Slip protection.
    """
    max_size = 25 * 1024 * 1024  # 25MB
    raw_name = file.filename or "dataset.csv"
    clean_filename = Path(raw_name).name
    ext = Path(clean_filename).suffix.lower()

    if ext not in [".csv", ".tsv", ".zip", ".txt"]:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Please upload a .csv, .tsv, or .zip file.",
        )

    upload_id = str(uuid.uuid4())[:8]
    upload_dir = Path("outputs") / "uploads" / f"upload_{upload_id}"
    upload_dir.mkdir(parents=True, exist_ok=True)

    content = await file.read()
    if len(content) > max_size:
        shutil.rmtree(upload_dir, ignore_errors=True)
        raise HTTPException(
            status_code=400,
            detail=f"File exceeds maximum size limit of {max_size // (1024 * 1024)}MB.",
        )

    saved_files: list[str] = []
    if ext == ".zip":
        import io
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as zf:
                resolved_dir = upload_dir.resolve()
                total_uncompressed = 0
                for member in zf.infolist():
                    target = (upload_dir / member.filename).resolve()
                    if not str(target).startswith(str(resolved_dir)):
                        shutil.rmtree(upload_dir, ignore_errors=True)
                        raise HTTPException(status_code=400, detail="Invalid zip archive: path traversal detected.")
                    total_uncompressed += member.file_size
                    if total_uncompressed > max_size * 2:
                        shutil.rmtree(upload_dir, ignore_errors=True)
                        raise HTTPException(status_code=400, detail="Uncompressed zip content exceeds limit.")
                zf.extractall(upload_dir)
                for _, _, files in os.walk(upload_dir):
                    saved_files.extend(files)
        except zipfile.BadZipFile:
            shutil.rmtree(upload_dir, ignore_errors=True)
            raise HTTPException(status_code=400, detail="Corrupted or invalid zip file.")
    else:
        target_file = upload_dir / clean_filename
        target_file.write_bytes(content)
        saved_files.append(clean_filename)

        if clean_filename.lower() != "train.csv" and "test" not in clean_filename.lower():
            train_copy = upload_dir / "train.csv"
            if not train_copy.exists():
                train_copy.write_bytes(content)
                saved_files.append("train.csv")

    rel_path = str(upload_dir).replace("\\", "/")
    return {
        "status": "success",
        "dataset_path": rel_path,
        "files": saved_files,
        "message": f"Successfully uploaded {len(saved_files)} file(s).",
    }

# ── New endpoints (additive, read-only, frontend support) ─────────────────

@router.get("/runs")
def list_runs() -> list[dict[str, Any]]:
    """Return all runs from the in-process RunManager."""
    all_runs = run_manager.get_all_runs()
    result = []
    for r in all_runs:
        result.append({
            "run_id": r.run_id,
            "status": r.status,
            "success": r.success,
            "selected_library": r.selected_library,
            "iterations": r.iterations,
            "execution_duration": r.execution_duration,
            "task_summary": r.task_summary,
            "dataset_name": r.dataset_name,
            "final_metrics": getattr(r, "final_metrics", None),
            "raw_metrics": getattr(r, "raw_metrics", None),
            "execution_backend": getattr(r, "execution_backend", "mock") or "mock",
        })
    # Most-recent first (RunManager appends in order; reverse for display)
    return list(reversed(result))


@router.get("/memory/search")
def search_memory(q: str = "", top_k: int = 5) -> dict[str, Any]:
    """Search semantic memory. Returns knowledge chunks matching the query."""
    if not q.strip():
        return {"query": q, "results": [], "error": None}

    try:
        from mlzero.memory.index import SemanticIndex

        index = SemanticIndex()
        index_path = Path(settings.memory.index_path)
        if not index_path.exists():
            return {"query": q, "results": [], "error": "Knowledge index not built yet."}

        # Load from persisted index file
        import json

        from mlzero.schemas.memory import KnowledgeChunk
        data = json.loads(index_path.read_text(encoding="utf-8"))
        chunks_data = data.get("chunks", [])
        chunks = [KnowledgeChunk(**c) for c in chunks_data]
        if chunks:
            index.add(chunks)

        scored = index.search(query=q, top_k=top_k)
        results = []
        for chunk, score in scored:
            results.append({
                "chunk_id": chunk.chunk_id,
                "source": chunk.source or "",
                "library": chunk.library_name or "",
                "title": (chunk.source or chunk.chunk_id or "").replace("_", " ").title(),
                "score": round(score, 4),
                "summary": chunk.summary or chunk.condensed_guidance or chunk.content[:200],
                "tags": chunk.tags or [],
            })
        return {"query": q, "results": results, "error": None}

    except Exception as exc:  # noqa: BLE001
        return {"query": q, "results": [], "error": str(exc)}


@router.get("/settings")
def get_settings() -> dict[str, Any]:
    """Return non-secret system configuration for the Settings page."""
    return {
        "version": settings.app.version,
        "llm_provider": settings.llm.provider,
        "llm_model": settings.llm.model,
        "llm_mode": settings.llm.mode,
        "llm_temperature": settings.llm.temperature,
        "llm_max_tokens": settings.llm.max_tokens,
        "llm_timeout": settings.llm.timeout,
        "execution_timeout": settings.execution.timeout_seconds,
        "run_timeout": settings.app.run_timeout_seconds,
        "max_concurrent_runs": settings.app.max_concurrent_runs,
        "memory_backend": settings.memory.embedding_backend,
        "retrieval_top_k": settings.memory.retrieval_top_k,
        "artifact_root": settings.app.artifact_root,
        "knowledge_root": settings.memory.knowledge_root,
        "semantic_memory_enabled": settings.memory.semantic_memory_enabled,
        "groq_configured": bool(os.environ.get("GROQ_API_KEY") or settings.groq_api_key),
        "gemini_configured": bool(os.environ.get("GEMINI_API_KEY") or settings.gemini_api_key),
        "openrouter_configured": bool(os.environ.get("OPENROUTER_API_KEY") or settings.openrouter_api_key),
    }


@router.get("/evaluation/summary")
def get_evaluation_summary() -> dict[str, Any]:
    """Return evaluation benchmarks summary if report exists."""
    import json
    candidates = [
        Path("reports/evaluation_summary.json"),
        Path("evaluation_summary.json"),
        Path("evaluation/reports/evaluation_summary.json"),
    ]
    for c in candidates:
        if c.exists():
            try:
                with open(c, encoding="utf-8") as f:
                    data = json.load(f)
                    return dict(data) if isinstance(data, dict) else {"data": data}
            except (OSError, json.JSONDecodeError) as e:
                return {"error": f"Failed to load evaluation report: {e}"}
    return {"error": "No evaluation report found"}


class FourWayAblationRunRequest(BaseModel):
    runs_per_case: int = Field(default=3, ge=1, le=10, description="Repetitions per benchmark case")
    cases: list[str] | None = Field(default=None, description="Optional case IDs to evaluate")
    llm_mode: str = Field(default="mock", description="LLM execution mode (mock or real)")
    output_dir: str = Field(default="reports", description="Report output directory")


@router.get("/evaluation/four-way-ablation")
def get_four_way_ablation() -> dict[str, Any]:
    """Return the latest computed four-way ablation study results."""
    import json
    candidates = [
        Path("reports/four_way_ablation_results.json"),
        Path("four_way_ablation_results.json"),
        Path("evaluation/reports/four_way_ablation_results.json"),
    ]
    for c in candidates:
        if c.exists():
            try:
                with open(c, encoding="utf-8") as f:
                    data = json.load(f)
                    return dict(data) if isinstance(data, dict) else {"data": data}
            except (OSError, json.JSONDecodeError) as e:
                return {"status": "error", "error": f"Failed to load four-way ablation report: {e}"}
    return {"status": "not_run", "message": "No four-way ablation results found."}


@router.post("/evaluation/four-way-ablation/run")
def run_four_way_ablation(request: FourWayAblationRunRequest | None = None) -> dict[str, Any]:
    """Execute four-way ablation study across benchmark cases and return dynamically computed results."""
    from evaluation.cases import get_case
    from evaluation.four_way_ablation import FourWayAblationRunner

    req = request or FourWayAblationRunRequest()
    runner = FourWayAblationRunner(
        use_mock_llm=(req.llm_mode.lower() == "mock"),
        max_iterations=3,
        retrieval_k=5,
        output_dir=req.output_dir,
    )

    selected_cases = None
    if req.cases:
        selected_cases = [c for c in (get_case(cid) for cid in req.cases) if c is not None]

    result = runner.run_experiment(cases=selected_cases, runs_per_case=req.runs_per_case)
    return result


class EvaluationRunRequest(BaseModel):
    mode: str = Field(default="local_formal", description="Evaluation mode (smoke, local_formal)")
    runs_per_case: int = Field(default=1, ge=1, le=10, description="Repetitions per benchmark case")
    llm_mode: str = Field(default="mock", description="LLM execution mode (mock or real)")
    output_dir: str = Field(default="reports", description="Report output directory")


@router.post("/evaluation/run")
def run_evaluation_benchmark(request: EvaluationRunRequest | None = None) -> dict[str, Any]:
    """Execute benchmark evaluation suite (all 10 cases) and update reports/evaluation_summary.json."""
    from evaluation.runner import run_evaluation_suite

    req = request or EvaluationRunRequest()
    result = run_evaluation_suite(
        mode=req.mode,
        runs_per_case=req.runs_per_case,
        use_mock_llm=(req.llm_mode.lower() == "mock"),
        output_dir=req.output_dir,
    )
    if result is not None and hasattr(result, "model_dump"):
        return result.model_dump()
    if isinstance(result, dict):
        return result
    return {"status": "ok", "message": "Evaluation completed successfully"}


