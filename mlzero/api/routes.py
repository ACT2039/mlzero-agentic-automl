from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

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
    ep_mem = EpisodicMemory()
    run = ep_mem.store.get_run_history(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run history not found")
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
