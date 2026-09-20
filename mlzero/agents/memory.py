"""
Memory agent components for MLZero.
Includes the three paper-faithful agents:
- SummarizationAgent
- CondensationAgent
- RetrievalAgent
"""
from abc import ABC, abstractmethod
from typing import Any

from mlzero.core.logger import setup_logger
from mlzero.memory.retrieval import RetrievalAgent
from mlzero.memory.semantic import SemanticMemory
from mlzero.memory.summarization import CondensationAgent, SummarizationAgent

logger = setup_logger(__name__)


# Legacy abstract interfaces maintained for backwards compatibility
class BaseSemanticMemory(ABC):
    """
    Abstract interface for Semantic Memory.
    """

    @abstractmethod
    def ingest_document(self, document_id: str, content: str) -> bool:
        """Ingest a new document into semantic memory."""

    @abstractmethod
    def retrieve(self, query: str) -> list[dict[str, Any]]:
        """Retrieve relevant documentation based on query."""


class EpisodicMemory(ABC):
    """
    Interface for Episodic Memory (Iteration history, previous code, execution logs).
    """

    @abstractmethod
    def record_episode(self, episode_id: str, data: dict[str, Any]) -> bool:
        """Record an episode (iteration, execution log, error)."""

    @abstractmethod
    def get_episode(self, episode_id: str) -> dict[str, Any] | None:
        """Retrieve a past episode."""

    @abstractmethod
    def get_recent_episodes(self, limit: int = 5) -> list[dict[str, Any]]:
        """Retrieve the most recent episodes."""


class MockSemanticMemory(BaseSemanticMemory):
    """Placeholder implementation of SemanticMemory."""

    def ingest_document(self, document_id: str, content: str) -> bool:
        logger.info(f"Ingesting document: {document_id}")
        return True

    def retrieve(self, query: str) -> list[dict[str, Any]]:
        logger.info(f"Retrieving from semantic memory with query: {query}")
        return [{"message": "Placeholder for Semantic Memory Retrieval"}]


class MockEpisodicMemory(EpisodicMemory):
    """Placeholder implementation of EpisodicMemory."""

    def record_episode(self, episode_id: str, data: dict[str, Any]) -> bool:
        logger.info(f"Recording episode: {episode_id}")
        return True

    def get_episode(self, episode_id: str) -> dict[str, Any] | None:
        logger.info(f"Retrieving episode: {episode_id}")
        return None

    def get_recent_episodes(self, limit: int = 5) -> list[dict[str, Any]]:
        logger.info(f"Retrieving {limit} recent episodes")
        return []


__all__ = [
    "BaseSemanticMemory",
    "CondensationAgent",
    "EpisodicMemory",
    "MockEpisodicMemory",
    "MockSemanticMemory",
    "RetrievalAgent",
    "SemanticMemory",
    "SummarizationAgent",
]

