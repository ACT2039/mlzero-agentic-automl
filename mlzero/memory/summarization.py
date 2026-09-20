"""
Summarization and condensation agents for Semantic Memory.
"""
from pydantic import BaseModel, Field

from mlzero.core.llm import LLMClient
from mlzero.core.logger import setup_logger
from mlzero.schemas.memory import KnowledgeChunk

logger = setup_logger(__name__)


class SummaryResult(BaseModel):
    summary: str = Field(..., description="Concise summary of the chunk.")


class CondensationResult(BaseModel):
    condensed_text: str = Field(..., description="Focused technical condensation.")


class SummarizationAgent:
    """
    Agent responsible for offline documentation summarization.
    Produces concise, searchable summaries while preserving API names,
    parameters, constraints, and error recovery guidance.
    """

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def summarize(self, target: str | KnowledgeChunk, source: str = "") -> str:
        """
        Summarize raw documentation text or a KnowledgeChunk.
        """
        if isinstance(target, KnowledgeChunk):
            content = target.content
            source_name = target.source or source
            chunk_id = target.chunk_id
        else:
            content = target
            source_name = source
            chunk_id = "ad-hoc"

        prompt = (
            "Summarize the following ML library documentation section concisely.\n"
            "Preserve essential API names, parameters, usage constraints, examples, and failure-related guidance.\n"
            f"Source: {source_name}\n\n"
            f"Content:\n{content}"
        )
        try:
            res = self.llm_client.generate_structured(prompt, SummaryResult)
            return res.summary.strip()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Summarization failed for {chunk_id}: {e}")
            return content[:200].strip()


class CondensationAgent:
    """
    Agent responsible for offline documentation condensation.
    Converts summarized/raw documentation into precise, streamlined guidance
    useful for code generation, stripping unnecessary narrative while preserving
    executable API patterns and parameters.
    """

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def condense(self, target: str | KnowledgeChunk, summary: str = "", source: str = "") -> str:
        """
        Condense technical documentation into actionable implementation guidance.
        """
        if isinstance(target, KnowledgeChunk):
            content = target.content
            source_name = target.source or source
            chunk_summary = target.summary or summary
            chunk_id = target.chunk_id
        else:
            content = target
            source_name = source
            chunk_summary = summary
            chunk_id = "ad-hoc"

        prompt = (
            "Condense the following technical documentation into focused, actionable implementation guidance for code generation.\n"
            "Remove conversational text and narrative. Preserve executable code patterns, API signatures, parameter requirements, and caveats.\n"
            f"Source: {source_name}\n"
        )
        if chunk_summary:
            prompt += f"Summary Context: {chunk_summary}\n"
        prompt += f"\nContent:\n{content}"

        try:
            res = self.llm_client.generate_structured(prompt, CondensationResult)
            return res.condensed_text.strip()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Condensation failed for {chunk_id}: {e}")
            return content.strip()


# Backward-compatible aliases
DocumentationSummarizer = SummarizationAgent
DocumentationCondenser = CondensationAgent
