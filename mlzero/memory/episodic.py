"""
Service interface for Episodic Memory.
Implements chronological recording of iterations according to the MLZero NeurIPS 2025 formulation:
E_t(P, C_{t-1}, L_{t-1}, G_{t-1}, R_{t-1}) = R_t
"""
from __future__ import annotations

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

    def start_run(
        self,
        run_id: str,
        perceptual_context: PerceptualContext,
        user_instruction: str | None = None,
    ) -> None:
        """Initialize a new run history."""
        selected_lib = (
            perceptual_context.library.selected_library
            if perceptual_context.library
            else None
        )
        run = RunHistory(
            run_id=run_id,
            start_time=_now(),
            perceptual_context_summary=perceptual_context.model_dump(exclude_none=True),
            selected_library=selected_lib,
            user_instruction=user_instruction,
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
        retrieval_query: str | None = None,
        error_context_before: ErrorContext | None = None,
        perceptual_context: PerceptualContext | None = None,
        selected_library: str | None = None,
        user_instruction: str | None = None,
        summary: str | None = None,
        suggested_fix: str | None = None,
        judge_decision: str | None = None,
        judge_reason: str | None = None,
        execution_status: str | None = None,
        bash_script: str | None = None,
        missing_expected_files: list[str] | None = None,
    ) -> Episode:
        """
        Record the outcome of a single iteration.
        Preserves generated code, execution logs, retrieved knowledge, and error context.
        Applies configurable bounding limits to avoid unbounded growth.
        """
        # Bounded generated code
        stored_code = None
        if artifact and artifact.code:
            max_code = settings.episodic.max_stored_code_chars
            stored_code = artifact.code[:max_code]
            if len(artifact.code) > max_code:
                stored_code += "\n# ... [code truncated due to length]"

        # Optional bash script
        stored_bash = bash_script or (artifact.bash_script if artifact else None)

        # Bounded execution result
        result_dict: dict[str, Any] | None = None
        stdout_text: str | None = None
        stderr_text: str | None = None
        exit_code: int | None = None
        duration: float | None = None
        exec_status = execution_status
        missing_files = missing_expected_files or []

        if result:
            exec_status = exec_status or getattr(result, "status", None) or ("SUCCESS" if result.success else "FAILURE")
            missing_files = missing_files or getattr(result, "missing_expected_files", [])
            max_stdout = min(
                getattr(settings.episodic, "max_stdout_chars", 5000),
                getattr(settings.episodic, "max_stdout_stderr_chars", 1500),
            )
            max_stderr = min(
                getattr(settings.episodic, "max_stderr_chars", 5000),
                getattr(settings.episodic, "max_stdout_stderr_chars", 1500),
            )
            stdout_text = result.stdout[:max_stdout] + (
                "..." if len(result.stdout) > max_stdout else ""
            )
            stderr_text = result.stderr[:max_stderr] + (
                "..." if len(result.stderr) > max_stderr else ""
            )
            exit_code = result.return_code
            duration = result.duration_seconds

            result_dict = {
                "success": result.success,
                "status": exec_status,
                "return_code": result.return_code,
                "duration_seconds": result.duration_seconds,
                "stdout_excerpt": stdout_text,
                "stderr_excerpt": stderr_text,
                "output_files": result.output_files,
                "missing_expected_files": missing_files,
                "error_info": result.error_info,
            }

        # Bounded retrieved knowledge
        retrieved_list: list[dict[str, Any]] = []
        sources: list[str] = []
        query_recorded = retrieval_query

        if retrieved_knowledge:
            sources = retrieved_knowledge.sources
            if not query_recorded and retrieved_knowledge.query_used:
                query_recorded = retrieved_knowledge.query_used

            max_k_chars = settings.episodic.max_retrieved_knowledge_chars
            curr_chars = 0
            for chunk in retrieved_knowledge.chunks:
                c_text = chunk.condensed_guidance or chunk.content
                c_len = len(c_text)
                if curr_chars + c_len > max_k_chars and curr_chars > 0:
                    break
                retrieved_list.append({
                    "chunk_id": chunk.chunk_id,
                    "source": chunk.source,
                    "library_name": chunk.library_name,
                    "guidance": c_text[: max_k_chars - curr_chars],
                })
                curr_chars += c_len

        # Error diagnosis
        err_cat = None
        err_sum = None
        err_fix = suggested_fix
        error_summary_dict: dict[str, Any] | None = None

        if error_ctx:
            err_cat = error_ctx.error_category
            err_sum = error_ctx.error_summary or f"{error_ctx.error_category}: {error_ctx.error_message}"
            err_fix = err_fix or error_ctx.suggested_fix
            error_summary_dict = {
                "error_category": error_ctx.error_category,
                "error_summary": err_sum,
                "error_message": error_ctx.error_message,
                "suggested_fix": error_ctx.suggested_fix,
                "stderr_excerpt": error_ctx.stderr_excerpt[:500] if error_ctx.stderr_excerpt else "",
            }

        # Perceptual context dict
        p_ctx_dict = None
        if perceptual_context:
            p_ctx_dict = perceptual_context.model_dump(exclude_none=True)

        # Previous error context dict
        err_before_dict = None
        if error_context_before:
            err_before_dict = error_context_before.model_dump(exclude_none=True)

        episode = Episode(
            episode_id=str(uuid.uuid4()),
            run_id=run_id,
            iteration=iteration,
            timestamp=_now(),
            status=status,
            perceptual_context=p_ctx_dict,
            selected_library=selected_library,
            user_instruction=user_instruction,
            error_context_before=err_before_dict,
            generated_code=stored_code,
            code_content=stored_code,
            code_artifact_reference=f"Iteration {iteration} code artifact",
            bash_script=stored_bash,
            execution_status=exec_status,
            execution_result=result_dict,
            execution_result_summary=result_dict,
            stdout=stdout_text,
            stderr=stderr_text,
            exit_code=exit_code,
            execution_duration=duration,
            missing_expected_files=missing_files,
            retrieval_query=query_recorded,
            retrieved_knowledge=retrieved_list if retrieved_list else None,
            retrieved_knowledge_references=sources,
            judge_decision=judge_decision,
            judge_reason=judge_reason,
            error_category=err_cat,
            error_summary=err_sum,
            suggested_fix=err_fix,
            error_context_summary=error_summary_dict,
            summary=summary,
        )

        self.store.add_episode(episode)
        logger.info(f"Recorded iteration {iteration} for run {run_id} (status={status})")
        return episode

    def get_history(self, run_id: str) -> list[Episode]:
        """Retrieve the complete chronological episode sequence for a run."""
        return self.store.list_episodes(run_id)

    def get_latest_context(self, run_id: str) -> dict[str, Any]:
        """
        Retrieve compact structured context of the latest iteration for the orchestrator.
        Provides the most recent error context, generated code, and retrieval info.
        """
        episodes = self.get_history(run_id)
        if not episodes:
            return {}

        latest = episodes[-1]
        latest_fail = self.get_latest_failure(run_id)

        return {
            "run_id": run_id,
            "latest_iteration": latest.iteration,
            "latest_status": latest.status,
            "latest_code": latest.generated_code or latest.code_content,
            "latest_error_context": latest.error_context_summary,
            "latest_error_summary": latest.error_summary,
            "latest_suggested_fix": latest.suggested_fix,
            "latest_retrieval_query": latest.retrieval_query,
            "latest_failure_iteration": latest_fail.iteration if latest_fail else None,
            "latest_failure_code": (latest_fail.generated_code or latest_fail.code_content) if latest_fail else None,
            "total_iterations": len(episodes),
        }

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
        """Compress run history into a bounded dictionary for the Coder / LLM prompt."""
        run = self.store.get_run_history(run_id)
        if not run or not run.episodes:
            return {}

        max_episodes = settings.episodic.max_episodes_in_context
        recent_episodes = run.episodes[-max_episodes:]

        context: dict[str, Any] = {
            "run_id": run.run_id,
            "total_iterations_so_far": len(run.episodes),
            "recent_history": [],
        }

        latest_failure = self.get_latest_failure(run_id)
        if latest_failure:
            context["latest_failure"] = {
                "iteration": latest_failure.iteration,
                "error_category": (
                    latest_failure.error_context_summary.get("error_category")
                    if latest_failure.error_context_summary
                    else latest_failure.error_category
                ),
                "error_summary": latest_failure.error_summary,
                "suggested_fix": latest_failure.suggested_fix,
                "failed_code": latest_failure.generated_code or latest_failure.code_content,
            }

        # Add bounded strings for recent episodes
        for ep in recent_episodes:
            ep_dict: dict[str, Any] = {
                "iteration": ep.iteration,
                "status": ep.status,
            }
            if ep.status == "FAIL":
                if ep.error_context_summary:
                    ep_dict["error_category"] = ep.error_context_summary.get("error_category")
                    ep_dict["error_summary"] = ep.error_summary or ep.error_context_summary.get("error_summary")
                    ep_dict["error_message"] = ep.error_context_summary.get("error_message")
                    ep_dict["suggested_fix"] = ep.suggested_fix
                else:
                    ep_dict["error_category"] = ep.error_category
                    ep_dict["error_summary"] = ep.error_summary
                    ep_dict["suggested_fix"] = ep.suggested_fix
            elif ep.status == "SUCCESS":
                ep_dict["success_note"] = "Iteration succeeded."

            context["recent_history"].append(ep_dict)

        return context
