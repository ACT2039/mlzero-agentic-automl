"""
Document loading, chunking, and offline ingestion pipeline for Semantic Memory.
"""
import uuid
from pathlib import Path

from mlzero.core.config import settings
from mlzero.core.logger import setup_logger
from mlzero.memory.summarization import CondensationAgent, SummarizationAgent
from mlzero.schemas.memory import KnowledgeChunk, KnowledgeDocument

logger = setup_logger(__name__)


def normalize_library_name(dir_name: str) -> str:
    """Normalize directory name to standard library identifier."""
    mapping = {
        "autogluon_tabular": "autogluon.tabular",
        "autogluon_multimodal": "autogluon.multimodal",
        "autogluon_timeseries": "autogluon.timeseries",
        "general_ml": "general_ml",
    }
    return mapping.get(dir_name, dir_name)


def load_documents(knowledge_dir: Path | str) -> list[KnowledgeDocument]:
    """Load supported text documents from the knowledge directory."""
    knowledge_dir = Path(knowledge_dir)
    documents: list[KnowledgeDocument] = []

    if not knowledge_dir.exists() or not knowledge_dir.is_dir():
        logger.warning(f"Knowledge directory not found: {knowledge_dir}")
        return documents

    supported_extensions = {".md", ".txt", ".rst"}
    max_size = settings.memory.max_document_size_bytes

    for file_path in sorted(knowledge_dir.rglob("*")):
        if not file_path.is_file() or file_path.suffix.lower() not in supported_extensions:
            continue

        try:
            size = file_path.stat().st_size
            if size > max_size:
                logger.warning(f"Skipping {file_path}: size {size} exceeds max {max_size}")
                continue

            content = file_path.read_text(encoding="utf-8")
            rel_path = file_path.relative_to(knowledge_dir)
            raw_lib = rel_path.parts[0] if len(rel_path.parts) > 1 else "general_ml"
            library_name = normalize_library_name(raw_lib)

            doc = KnowledgeDocument(
                document_id=str(uuid.uuid4()),
                source=str(rel_path),
                source_path=str(file_path),
                title=file_path.stem,
                content=content,
                library_name=library_name,
                tags=[library_name, file_path.stem],
                metadata={"file_size": size},
            )
            documents.append(doc)

        except Exception as e:  # noqa: BLE001
            logger.warning(f"Failed to read {file_path}: {e}")

    return documents


def chunk_document(
    doc: KnowledgeDocument, chunk_size: int | None = None, overlap: int | None = None
) -> list[KnowledgeChunk]:
    """Split a document into chunks preserving metadata."""
    c_size = chunk_size or settings.memory.chunk_size
    c_overlap = overlap or settings.memory.chunk_overlap

    chunks: list[KnowledgeChunk] = []
    text = doc.content
    text_len = len(text)

    if text_len == 0:
        return chunks

    start = 0
    while start < text_len:
        end = min(start + c_size, text_len)
        chunk_content = text[start:end]

        chunk = KnowledgeChunk(
            chunk_id=str(uuid.uuid4()),
            document_id=doc.document_id,
            content=chunk_content,
            source=doc.source,
            source_path=doc.source_path,
            library_name=doc.library_name,
            library_version=doc.library_version,
            tags=list(doc.tags),
            metadata={"start_char": start, "end_char": end, "title": doc.title},
        )
        chunks.append(chunk)

        start += (c_size - c_overlap)

    return chunks


def ingest_knowledge_pipeline(
    documents: list[KnowledgeDocument],
    summarizer: SummarizationAgent | None = None,
    condenser: CondensationAgent | None = None,
    summarize: bool = False,
    condense: bool = False,
) -> list[KnowledgeChunk]:
    """
    Offline ingestion pipeline:
    raw docs -> chunking -> SummarizationAgent -> CondensationAgent -> condensed chunks
    """
    all_chunks: list[KnowledgeChunk] = []

    for doc in documents:
        chunks = chunk_document(doc)
        for chunk in chunks:
            # 1. Summarization phase
            if summarize and summarizer:
                summary = summarizer.summarize(chunk)
                chunk.summary = summary
                # Enrich content with summary for broader recall if needed
                chunk.content = f"Summary: {summary}\n\nOriginal: {chunk.content}"

            # 2. Condensation phase
            if condense and condenser:
                condensed = condenser.condense(chunk)
                chunk.condensed_guidance = condensed
                if not summarize:
                    chunk.content = condensed

            all_chunks.append(chunk)

    return all_chunks
