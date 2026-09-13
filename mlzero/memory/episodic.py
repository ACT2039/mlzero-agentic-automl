"""
Service interface for Episodic Memory.
"""
import uuid
from datetime import UTC, datetime
from typing import Any

from mlzero.core.config import settings
from mlzero.core.logger import setup_logger
from mlzero.memory.episodic_store import EpisodicStore
from mlzero.schemas.coder import CodeArtifact, ErrorContext, ExecutionResult
from mlzero.schemas.episodic import Episode, RunHistory
from mlzero.schemas.memory import RetrievedKnowledge
from mlzero.schemas.perception import PerceptualContext

logger = setup_logger(__name__)


def _now() -> str:
    return datetime.now(UTC).isoformat()


class EpisodicMemory:
    """Service for managing episodic history of ML automation runs."""
    
    def __init__(self, store: EpisodicStore | None = None) -> None:
        self.store = store or EpisodicStore()
        
    def start_run(self, run_id: str, perceptual_context: PerceptualContext) -> None:
        """Initialize a new run history."""
        run = RunHistory(
            run_id=run_id,
            start_time=_now(),
            perceptual_context_summary=perceptual_context.model_dump(exclude_none=True),
            selected_library=perceptual_context.library.selected_library if perceptual_context.library else None
        )
        self.store.save_run_history(run)
        logger.info(f"Started episodic memory for run {run_id}")
        
    def end_run(self, run_id: str, final_result_status: str) -> None:
        """Mark a run as completed."""
        run = self.store.get_run_history(run_id)
        if run:
            run.end_time = _now()
            run.final_result = final_result_status
            self.store.save_run_history(run)
            logger.info(f"Ended episodic memory for run {run_id} with status {final_result_status}")
            
    def record_iteration(
        self,
        run_id: str,
        iteration: int,
        status: str,
        artifact: CodeArtifact | None = None,
        result: ExecutionResult | None = None,
        error_ctx: ErrorContext | None = None,
        retrieved_knowledge: RetrievedKnowledge | None = None,
        summary: str | None = None
    ) -> None:
        """Record the outcome of a single iteration."""
        # Bound execution result
        result_summary = None
        if result:
            max_chars = settings.episodic.max_stdout_stderr_chars
            stdout = result.stdout[:max_chars] + ("..." if len(result.stdout) > max_chars else "")
            stderr = result.stderr[:max_chars] + ("..." if len(result.stderr) > max_chars else "")
            
            result_summary = {
                "success": result.success,
                "return_code": result.return_code,
                "duration_seconds": result.duration_seconds,
                "stdout_excerpt": stdout,
                "stderr_excerpt": stderr,
                "output_files": result.output_files,
                "error_info": result.error_info
            }
            
        # Bound error context
        error_summary = None
        if error_ctx:
            error_summary = {
                "error_category": error_ctx.error_category,
                "error_message": error_ctx.error_message,
                "suggested_fix": error_ctx.suggested_fix
            }
            
        sources = retrieved_knowledge.sources if retrieved_knowledge else []
        
        episode = Episode(
            episode_id=str(uuid.uuid4()),
            run_id=run_id,
            iteration=iteration,
            timestamp=_now(),
            status=status,
            code_artifact_reference="Code generated in this iteration",
            code_content=artifact.code if artifact else None,
            execution_result_summary=result_summary,
            error_context_summary=error_summary,
            retrieved_knowledge_references=sources,
            summary=summary,
            suggested_fix=error_ctx.suggested_fix if error_ctx else None
        )
        
        self.store.add_episode(episode)
        logger.info(f"Recorded iteration {iteration} for run {run_id}")
        
    def get_latest_failure(self, run_id: str) -> Episode | None:
        """Retrieve the most recent failed iteration."""
        episodes = self.store.list_episodes(run_id)
        for ep in reversed(episodes):
            if ep.status == "FAIL":
                return ep
        return None
        
    def get_latest_success(self, run_id: str) -> Episode | None:
        """Retrieve the most recent successful iteration."""
        episodes = self.store.list_episodes(run_id)
        for ep in reversed(episodes):
            if ep.status == "SUCCESS":
                return ep
        return None

    def get_bounded_context(self, run_id: str) -> dict[str, Any]:
        """Compress run history into a bounded dictionary for the LLM."""
        run = self.store.get_run_history(run_id)
        if not run or not run.episodes:
            return {}
            
        max_episodes = settings.episodic.max_episodes_in_context
        recent_episodes = run.episodes[-max_episodes:]
        
        context: dict[str, Any] = {
            "run_id": run.run_id,
            "total_iterations_so_far": len(run.episodes),
            "recent_history": []
        }
        
        latest_failure = self.get_latest_failure(run_id)
        if latest_failure:
            context["latest_failure"] = {
                "iteration": latest_failure.iteration,
                "error_category": latest_failure.error_context_summary.get("error_category") if latest_failure.error_context_summary else None,
                "suggested_fix": latest_failure.suggested_fix,
                "failed_code": latest_failure.code_content
            }
            
        # Add bounded strings for recent episodes
        for ep in recent_episodes:
            ep_dict = {
                "iteration": ep.iteration,
                "status": ep.status,
            }
            if ep.status == "FAIL" and ep.error_context_summary:
                ep_dict["error_category"] = ep.error_context_summary.get("error_category")
                ep_dict["error_message"] = ep.error_context_summary.get("error_message")
                ep_dict["suggested_fix"] = ep.suggested_fix
            elif ep.status == "SUCCESS":
                ep_dict["success_note"] = "Iteration succeeded."
                
            context["recent_history"].append(ep_dict)
            
        return context
