import logging
import threading
import uuid
from typing import Any

from mlzero.application.service import MLZeroService
from mlzero.schemas.application import RunStatusResponse

logger = logging.getLogger(__name__)

class RunManager:
    """
    Process-local Run Manager for background MLZero execution.
    Not intended for distributed production.
    """
    def __init__(self, use_mock_llm: bool = False):
        self._runs: dict[str, RunStatusResponse] = {}
        self._lock = threading.Lock()
        self.use_mock_llm = use_mock_llm

    def submit_run(self, dataset_path: str, user_instruction: str | None = None, options: dict[str, Any] | None = None) -> str:
        run_id = str(uuid.uuid4())
        options = options or {}
        options["run_id"] = run_id
        
        with self._lock:
            self._runs[run_id] = RunStatusResponse(
                run_id=run_id,
                status="QUEUED"
            )
            
        thread = threading.Thread(
            target=self._execute_run,
            args=(run_id, dataset_path, user_instruction, options),
            daemon=True
        )
        thread.start()
        
        return run_id

    def _execute_run(self, run_id: str, dataset_path: str, user_instruction: str | None, options: dict[str, Any]) -> None:
        with self._lock:
            if run_id in self._runs:
                self._runs[run_id].status = "RUNNING"
                
        service = MLZeroService(use_mock_llm=self.use_mock_llm)
        try:
            result = service.run_mlzero(dataset_path, user_instruction, options)
            with self._lock:
                self._runs[run_id] = result
        except Exception as e:  # noqa: BLE001
            logger.error(f"Run {run_id} failed with exception: {e}")
            with self._lock:
                self._runs[run_id].status = "FAIL"
                self._runs[run_id].final_error = str(e)
                self._runs[run_id].success = False

    def get_run_status(self, run_id: str) -> RunStatusResponse | None:
        with self._lock:
            return self._runs.get(run_id)
            
    def get_all_runs(self) -> list[RunStatusResponse]:
        with self._lock:
            return list(self._runs.values())

# Global singleton for FastAPI
run_manager = RunManager()
