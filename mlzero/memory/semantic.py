"""
Semantic Memory interface for MLZero.
Coordinates offline ingestion (SummarizationAgent, CondensationAgent)
and online retrieval (RetrievalAgent).
"""
from pathlib import Path
from typing import Any

from mlzero.core.config import settings
from mlzero.core.llm import LLMClient
from mlzero.core.logger import setup_logger
from mlzero.memory.index import SemanticIndex
from mlzero.memory.ingestion import ingest_knowledge_pipeline, load_documents
from mlzero.memory.retrieval import RetrievalAgent
from mlzero.memory.summarization import CondensationAgent, SummarizationAgent
from mlzero.schemas.coder import ErrorContext
from mlzero.schemas.memory import LibraryKnowledgeRegistration, RetrievedKnowledge
from mlzero.schemas.perception import PerceptualContext

logger = setup_logger(__name__)


class SemanticMemory:
    """
    Interface for the Semantic Memory system based on the MLZero NeurIPS 2025 architecture.
    """

    def __init__(self, llm_client: LLMClient | None = None):
        self.llm_client = llm_client
        self.index = SemanticIndex()
        self.index_path = Path(settings.memory.index_path)

        # Three dedicated agents
        self.summarizer = SummarizationAgent(self.llm_client) if self.llm_client else None
        self.condenser = CondensationAgent(self.llm_client) if self.llm_client else None
        self.retrieval_agent = RetrievalAgent(self.index)

        # Library knowledge registry
        self.registry: dict[str, LibraryKnowledgeRegistration] = {}

        if self.index_path.exists():
            try:
                self.index.load(self.index_path)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Failed to load semantic index from {self.index_path}: {e}")

    def register_library_knowledge(
        self,
        registration: LibraryKnowledgeRegistration | dict[str, Any],
    ) -> None:
        """Register an ML library's documentation metadata in the knowledge registry."""
        if isinstance(registration, dict):
            reg = LibraryKnowledgeRegistration(**registration)
        else:
            reg = registration
        self.registry[reg.library_name] = reg
        logger.info(f"Registered library knowledge for: {reg.library_name}")

    def ingest(
        self,
        knowledge_dir: str | Path,
        summarize: bool = False,
        condense: bool = False,
    ) -> None:
        """
        Ingest documents from a directory through the offline pipeline:
        raw docs -> SummarizationAgent -> CondensationAgent -> index -> disk.
        """
        knowledge_dir = Path(knowledge_dir)
        docs = load_documents(knowledge_dir)

        chunks = ingest_knowledge_pipeline(
            documents=docs,
            summarizer=self.summarizer,
            condenser=self.condenser,
            summarize=summarize,
            condense=condense,
        )

        self.index.add(chunks)
        self.index.save(self.index_path)
        logger.info(f"Ingested {len(docs)} documents and {len(chunks)} chunks into semantic index.")

    def ingest_library_docs(
        self,
        library_name: str,
        docs_dir: str | Path,
        version: str | None = None,
        description: str = "",
        summarize: bool = True,
        condense: bool = True,
    ) -> None:
        """Register and ingest documentation for a specific ML library."""
        reg = LibraryKnowledgeRegistration(
            library_name=library_name,
            version=version,
            description=description,
            documentation_path=str(docs_dir),
        )
        self.register_library_knowledge(reg)
        self.ingest(docs_dir, summarize=summarize, condense=condense)

    def retrieve(
        self,
        perceptual_context: PerceptualContext,
        error_context: ErrorContext | None = None,
        user_instruction: str | None = None,
        iteration: int | None = None,
        top_k: int | None = None,
    ) -> RetrievedKnowledge:
        """
        Retrieve top-k relevant condensed knowledge chunks via the RetrievalAgent.
        Default k = 5, matching the paper.
        """
        if not settings.memory.semantic_memory_enabled:
            return RetrievedKnowledge()

        return self.retrieval_agent.retrieve(
            perceptual_context=perceptual_context,
            error_context=error_context,
            user_instruction=user_instruction,
            iteration=iteration,
            top_k=top_k,
        )
