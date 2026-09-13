"""
Summarization and condensation for Semantic Memory.
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


class DocumentationSummarizer:
    """Creates concise descriptions of documentation sections."""
    
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        
    def summarize(self, chunk: KnowledgeChunk) -> str:
        prompt = (
            "Summarize the following documentation section concisely.\n"
            f"Title: {chunk.source}\n\n"
            f"Content:\n{chunk.content}"
        )
        try:
            res = self.llm_client.generate_structured(prompt, SummaryResult)
            return res.summary
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Summarization failed for chunk {chunk.chunk_id}: {e}")
            return chunk.content[:200]


class DocumentationCondenser:
    """Converts long technical sections into focused implementation guidance."""
    
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        
    def condense(self, chunk: KnowledgeChunk) -> str:
        prompt = (
            "Condense the following technical documentation into focused implementation guidance.\n"
            f"Title: {chunk.source}\n\n"
            f"Content:\n{chunk.content}"
        )
        try:
            res = self.llm_client.generate_structured(prompt, CondensationResult)
            return res.condensed_text
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Condensation failed for chunk {chunk.chunk_id}: {e}")
            return chunk.content
