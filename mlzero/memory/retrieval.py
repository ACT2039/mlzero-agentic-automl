"""
Online Retrieval Agent for Semantic Memory.
"""
from typing import TYPE_CHECKING

from mlzero.core.config import settings
from mlzero.core.logger import setup_logger
from mlzero.schemas.coder import ErrorContext
from mlzero.schemas.memory import KnowledgeChunk, RetrievedKnowledge
from mlzero.schemas.perception import PerceptualContext

if TYPE_CHECKING:
    from mlzero.memory.index import SemanticIndex

logger = setup_logger(__name__)


class RetrievalAgent:
    """
    Agent responsible for online retrieval from Semantic Memory at each coding iteration.
    Constructs an error-aware query from perceptual context, selected ML library,
    and current error context, returning top-k condensed knowledge records.
    Default k = 5, matching the MLZero NeurIPS 2025 paper.
    """

    def __init__(self, index: "SemanticIndex"):
        self.index = index

    def build_query(
        self,
        perceptual_context: PerceptualContext,
        selected_library: str | None = None,
        error_context: ErrorContext | None = None,
        user_instruction: str | None = None,
    ) -> tuple[str, str | None]:
        """
        Construct a semantic search query from contextual signals.
        Returns (query_string, library_filter).
        """
        query_parts: list[str] = []

        # 1. Perceptual Task context
        if perceptual_context.task:
            if perceptual_context.task.objective:
                query_parts.append(perceptual_context.task.objective)
            if perceptual_context.task.task_type:
                query_parts.append(perceptual_context.task.task_type)
            if perceptual_context.task.target_column:
                query_parts.append(f"target {perceptual_context.task.target_column}")

        # 2. User instruction
        if user_instruction and user_instruction.strip():
            query_parts.append(user_instruction.strip())

        # 3. Selected ML Library
        lib_filter = selected_library
        if not lib_filter and perceptual_context.library and perceptual_context.library.selected_library:
            lib_filter = perceptual_context.library.selected_library

        if lib_filter:
            query_parts.append(lib_filter)

        # 4. Error Context (Iteration 2+)
        if error_context:
            if error_context.error_category:
                query_parts.append(error_context.error_category)
            if error_context.error_message:
                query_parts.append(error_context.error_message)
            if error_context.suggested_fix:
                query_parts.append(error_context.suggested_fix)
            if error_context.stderr_excerpt:
                # Include concise keywords from stderr
                query_parts.append(error_context.stderr_excerpt[:150])

        query = " ".join(query_parts).strip()
        return query, lib_filter

    def retrieve(
        self,
        perceptual_context: PerceptualContext,
        selected_library: str | None = None,
        error_context: ErrorContext | None = None,
        user_instruction: str | None = None,
        iteration: int | None = None,
        top_k: int | None = None,
    ) -> RetrievedKnowledge:
        """
        Retrieve relevant condensed knowledge chunks from the semantic index.
        """
        k = top_k or settings.memory.retrieval_top_k
        query, library_filter = self.build_query(
            perceptual_context=perceptual_context,
            selected_library=selected_library,
            error_context=error_context,
            user_instruction=user_instruction,
        )

        if not query:
            return RetrievedKnowledge(
                query_used="",
                selected_library=library_filter,
                iteration=iteration,
            )

        scored_chunks = self.index.search(
            query=query,
            top_k=k,
            library_filter=library_filter,
        )

        max_chars = settings.memory.max_retrieved_context_chars
        current_chars = 0
        final_chunks: list[KnowledgeChunk] = []
        scores: list[float] = []
        sources: set[str] = set()

        for chunk, score in scored_chunks:
            # Measure characters from condensed guidance or content
            display_text = chunk.condensed_guidance or chunk.content
            chunk_len = len(display_text)
            if current_chars + chunk_len > max_chars and current_chars > 0:
                break

            final_chunks.append(chunk)
            scores.append(score)
            if chunk.source:
                sources.add(chunk.source)
            current_chars += chunk_len

        # Observability logging (no secrets or sensitive paths)
        doc_titles = [c.source or c.chunk_id for c in final_chunks]
        logger.info(
            f"SemanticMemory Retrieval: iteration={iteration or 1}, library={library_filter}, "
            f"top_k={k}, retrieved={len(final_chunks)}, doc_titles={doc_titles}, "
            f"query='{query[:120]}...'" if len(query) > 120 else f"query='{query}'"
        )

        return RetrievedKnowledge(
            chunks=final_chunks,
            relevance_scores=scores,
            sources=list(sources),
            query_used=query,
            selected_library=library_filter,
            iteration=iteration,
        )
