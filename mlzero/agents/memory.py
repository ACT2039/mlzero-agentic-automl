"""
Memory components for MLZero.
Includes interfaces for Semantic Memory and Episodic Memory.
"""

from abc import ABC, abstractmethod
from typing import Any

from mlzero.core.logger import setup_logger

logger = setup_logger(__name__)


class SemanticMemory(ABC):
    """
    Interface for Semantic Memory (Documentation ingestion, summarization, retrieval).
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


class MockSemanticMemory(SemanticMemory):
    """Placeholder implementation of SemanticMemory for Phase 1."""

    def ingest_document(self, document_id: str, content: str) -> bool:
        logger.info(f"Ingesting document: {document_id}")
        return True

    def retrieve(self, query: str) -> list[dict[str, Any]]:
        logger.info(f"Retrieving from semantic memory with query: {query}")
        return [{"message": "Phase 1 placeholder for Semantic Memory Retrieval"}]


class MockEpisodicMemory(EpisodicMemory):
    """Placeholder implementation of EpisodicMemory for Phase 1."""

    def record_episode(self, episode_id: str, data: dict[str, Any]) -> bool:
        logger.info(f"Recording episode: {episode_id}")
        return True

    def get_episode(self, episode_id: str) -> dict[str, Any] | None:
        logger.info(f"Retrieving episode: {episode_id}")
        return None

    def get_recent_episodes(self, limit: int = 5) -> list[dict[str, Any]]:
        logger.info(f"Retrieving {limit} recent episodes")
        return []
