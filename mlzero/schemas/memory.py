"""
Schemas for Semantic Memory.
"""
from typing import Any

from pydantic import BaseModel, Field


class KnowledgeDocument(BaseModel):
    """A complete knowledge document."""
    document_id: str = Field(..., description="Unique identifier for the document.")
    source: str = Field(..., description="Source path or URL.")
    source_path: str = Field(default="", description="Relative or absolute file path.")
    title: str = Field(default="", description="Title of the document.")
    content: str = Field(..., description="Full text content.")
    library_name: str | None = Field(default=None, description="Associated ML library name.")
    library_version: str | None = Field(default=None, description="Library version.")
    tags: list[str] = Field(default_factory=list, description="Categorical tags.")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata.")


class KnowledgeChunk(BaseModel):
    """A segment of a knowledge document."""
    chunk_id: str = Field(..., description="Unique identifier for the chunk.")
    document_id: str = Field(..., description="ID of the parent document.")
    content: str = Field(..., description="Chunk text content.")
    source: str = Field(..., description="Source of the chunk.")
    source_path: str = Field(default="", description="Path of source file.")
    library_name: str | None = Field(default=None, description="Associated ML library name.")
    library_version: str | None = Field(default=None, description="Library version.")
    summary: str = Field(default="", description="Concise summary produced by SummarizationAgent.")
    condensed_guidance: str = Field(default="", description="Actionable guidance produced by CondensationAgent.")
    tags: list[str] = Field(default_factory=list, description="Categorical tags.")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata.")
    
    # Simple embedding metadata for our lightweight index
    embedding_metadata: dict[str, float] = Field(default_factory=dict, description="Term frequencies or embedding data.")


class RetrievedKnowledge(BaseModel):
    """Knowledge retrieved from semantic memory."""
    chunks: list[KnowledgeChunk] = Field(default_factory=list, description="Retrieved chunks.")
    relevance_scores: list[float] = Field(default_factory=list, description="Corresponding relevance scores.")
    sources: list[str] = Field(default_factory=list, description="Unique sources included in the retrieval.")
    query_used: str = Field(default="", description="The semantic search query used for retrieval.")
    selected_library: str | None = Field(default=None, description="Library filter applied.")
    iteration: int | None = Field(default=None, description="Iteration number during which retrieval was performed.")


class LibraryKnowledgeRegistration(BaseModel):
    """Registration record for an ML library in the knowledge base."""
    library_name: str = Field(..., description="Registered library name e.g. autogluon.tabular.")
    version: str | None = Field(default=None, description="Registered library version.")
    description: str = Field(default="", description="Short description of library capabilities.")
    documentation_path: str = Field(default="", description="Path to library documentation directory.")
    guidance: str | None = Field(default=None, description="Optional default prompt guidance for coder.")
