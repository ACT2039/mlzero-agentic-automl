"""
Document loading and chunking for Semantic Memory.
"""
import uuid
from pathlib import Path

from mlzero.core.config import settings
from mlzero.core.logger import setup_logger
from mlzero.schemas.memory import KnowledgeChunk, KnowledgeDocument

logger = setup_logger(__name__)


def load_documents(knowledge_dir: Path) -> list[KnowledgeDocument]:
    """Load supported text documents from the knowledge directory."""
    documents: list[KnowledgeDocument] = []
    
    if not knowledge_dir.exists() or not knowledge_dir.is_dir():
        logger.warning(f"Knowledge directory not found: {knowledge_dir}")
        return documents
        
    supported_extensions = {".md", ".txt", ".rst"}
    max_size = settings.memory.max_document_size_bytes
    
    for file_path in knowledge_dir.rglob("*"):
        if not file_path.is_file() or file_path.suffix.lower() not in supported_extensions:
            continue
            
        try:
            size = file_path.stat().st_size
            if size > max_size:
                logger.warning(f"Skipping {file_path}: size {size} exceeds max {max_size}")
                continue
                
            content = file_path.read_text(encoding="utf-8")
            
            # Infer library name from the directory structure (assumes knowledge/<library_name>/...)
            rel_path = file_path.relative_to(knowledge_dir)
            library_name = rel_path.parts[0] if len(rel_path.parts) > 1 else "general_ml"
            
            doc = KnowledgeDocument(
                document_id=str(uuid.uuid4()),
                source=str(rel_path),
                title=file_path.stem,
                content=content,
                library_name=library_name,
                metadata={"file_size": size}
            )
            documents.append(doc)
            
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Failed to read {file_path}: {e}")
            
    return documents


def chunk_document(doc: KnowledgeDocument, chunk_size: int | None = None, overlap: int | None = None) -> list[KnowledgeChunk]:
    """Split a document into chunks."""
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
            library_name=doc.library_name,
            library_version=doc.library_version,
            metadata={"start_char": start, "end_char": end}
        )
        chunks.append(chunk)
        
        start += (c_size - c_overlap)
        
    return chunks
