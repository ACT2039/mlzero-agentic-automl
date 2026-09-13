"""
Semantic Memory interface for MLZero.
Supports ingest, index, and retrieve.
"""
from pathlib import Path

from mlzero.core.config import settings
from mlzero.core.llm import LLMClient
from mlzero.core.logger import setup_logger
from mlzero.memory.index import SemanticIndex
from mlzero.memory.ingestion import chunk_document, load_documents
from mlzero.memory.summarization import DocumentationCondenser, DocumentationSummarizer
from mlzero.schemas.coder import ErrorContext
from mlzero.schemas.memory import RetrievedKnowledge
from mlzero.schemas.perception import PerceptualContext

logger = setup_logger(__name__)


class SemanticMemory:
    """Interface for the Semantic Memory system."""
    
    def __init__(self, llm_client: LLMClient | None = None):
        self.llm_client = llm_client
        self.index = SemanticIndex()
        self.index_path = Path(settings.memory.index_path)
        
        if self.index_path.exists():
            try:
                self.index.load(self.index_path)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Failed to load semantic index from {self.index_path}: {e}")

    def ingest(self, knowledge_dir: str | Path, summarize: bool = False, condense: bool = False) -> None:
        """Ingest documents, chunk them, and optionally summarize/condense."""
        knowledge_dir = Path(knowledge_dir)
        docs = load_documents(knowledge_dir)
        
        summarizer = DocumentationSummarizer(self.llm_client) if self.llm_client else None
        condenser = DocumentationCondenser(self.llm_client) if self.llm_client else None
        
        all_chunks = []
        for doc in docs:
            chunks = chunk_document(doc)
            
            if summarize and summarizer:
                for chunk in chunks:
                    chunk.content = f"Summary: {summarizer.summarize(chunk)}\n\nOriginal: {chunk.content}"
                    
            if condense and condenser:
                for chunk in chunks:
                    chunk.content = condenser.condense(chunk)
                    
            all_chunks.extend(chunks)
            
        self.index.add(all_chunks)
        self.index.save(self.index_path)
        logger.info(f"Ingested {len(docs)} documents and {len(all_chunks)} chunks.")

    def retrieve(self, perceptual_context: PerceptualContext, error_context: ErrorContext | None = None) -> RetrievedKnowledge:
        """Retrieve relevant knowledge based on context and errors."""
        query_parts = []
        
        # Build query from task
        if perceptual_context.task:
            if perceptual_context.task.objective:
                query_parts.append(perceptual_context.task.objective)
            if perceptual_context.task.task_type:
                query_parts.append(perceptual_context.task.task_type)
                
        # Build query from error
        if error_context:
            query_parts.append(error_context.error_category)
            query_parts.append(error_context.error_message)
            query_parts.append(error_context.suggested_fix)
            
        query = " ".join(query_parts)
        if not query.strip():
            return RetrievedKnowledge()
            
        library_filter = None
        if perceptual_context.library and perceptual_context.library.selected_library:
            library_filter = perceptual_context.library.selected_library
            # Query often benefits from the library name itself
            query += f" {library_filter}"
            
        top_k = settings.memory.retrieval_top_k
        scored_chunks = self.index.search(query, top_k=top_k, library_filter=library_filter)
        
        max_chars = settings.memory.max_retrieved_context_chars
        current_chars = 0
        final_chunks = []
        scores = []
        sources = set()
        
        for chunk, score in scored_chunks:
            chunk_len = len(chunk.content)
            if current_chars + chunk_len > max_chars and current_chars > 0:
                break
                
            final_chunks.append(chunk)
            scores.append(score)
            sources.add(chunk.source)
            current_chars += chunk_len
            
        return RetrievedKnowledge(
            chunks=final_chunks,
            relevance_scores=scores,
            sources=list(sources)
        )
