"""Tests for Semantic Memory module."""
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from mlzero.agents.coder import CoderAgent
from mlzero.core.llm import get_llm_client
from mlzero.memory.index import SemanticIndex
from mlzero.memory.ingestion import chunk_document, load_documents
from mlzero.memory.semantic import SemanticMemory
from mlzero.memory.summarization import DocumentationCondenser, DocumentationSummarizer
from mlzero.schemas.coder import CodeGenerationRequest
from mlzero.schemas.memory import KnowledgeChunk, KnowledgeDocument
from mlzero.schemas.perception import PerceptualContext, TaskContext


def test_knowledge_document_validation():
    doc = KnowledgeDocument(document_id="1", source="test.md", content="hello")
    assert doc.title == ""
    assert doc.metadata == {}
    with pytest.raises(ValueError):
        KnowledgeDocument(document_id="1", source="test.md")


def test_knowledge_chunk_validation():
    chunk = KnowledgeChunk(chunk_id="c1", document_id="1", content="chunk", source="test.md")
    assert chunk.metadata == {}
    with pytest.raises(ValueError):
        KnowledgeChunk(chunk_id="c1", content="chunk", source="test.md")


def test_document_loading_and_size_limits(monkeypatch):
    from mlzero.core.config import settings
    monkeypatch.setattr(settings.memory, "max_document_size_bytes", 15)
    
    with TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        (tmp / "valid.md").write_text("Small doc", encoding="utf-8")
        (tmp / "too_big.md").write_text("This document is way too big and should be skipped.", encoding="utf-8")
        (tmp / "unsupported.bin").write_text("binary", encoding="utf-8")
        
        docs = load_documents(tmp)
        assert len(docs) == 1
        assert docs[0].title == "valid"
        assert docs[0].content == "Small doc"


def test_chunking_and_metadata_preservation():
    doc = KnowledgeDocument(
        document_id="d1", source="file.md", title="Test", content="abcdefghij",
        library_name="test_lib", library_version="1.0"
    )
    chunks = chunk_document(doc, chunk_size=4, overlap=2)
    
    assert len(chunks) == 5
    assert chunks[0].content == "abcd"
    assert chunks[1].content == "cdef"
    assert chunks[2].content == "efgh"
    assert chunks[3].content == "ghij"
    assert chunks[4].content == "ij"
    assert chunks[0].library_name == "test_lib"
    assert chunks[0].metadata["start_char"] == 0


def test_summarizer_and_condenser():
    client = get_llm_client(use_mock=True)
    summarizer = DocumentationSummarizer(client)
    condenser = DocumentationCondenser(client)
    
    chunk = KnowledgeChunk(chunk_id="c1", document_id="d1", content="Raw content.", source="test.md")
    summary = summarizer.summarize(chunk)
    condensed = condenser.condense(chunk)
    
    assert "Mock summary" in summary
    assert "Mock condensed" in condensed


def test_semantic_index_add_search_persistence():
    index = SemanticIndex()
    chunk1 = KnowledgeChunk(chunk_id="c1", document_id="d1", content="autogluon tabular classification is great", source="test.md", library_name="autogluon")
    chunk2 = KnowledgeChunk(chunk_id="c2", document_id="d2", content="image processing with deep learning", source="test.md")
    
    index.add([chunk1, chunk2])
    
    # Search
    res = index.search("tabular", top_k=1)
    assert len(res) == 1
    assert res[0][0].chunk_id == "c1"
    
    # Library filtering
    res = index.search("deep", library_filter="autogluon")
    assert len(res) == 0
    
    # Persistence
    with TemporaryDirectory() as tmpdir:
        idx_path = Path(tmpdir) / "index.json"
        index.save(idx_path)
        
        new_index = SemanticIndex()
        new_index.load(idx_path)
        assert len(new_index.chunks) == 2
        
        res2 = new_index.search("tabular")
        assert len(res2) == 1
        assert res2[0][0].chunk_id == "c1"


def test_semantic_memory_interface_and_truncation(monkeypatch):
    from mlzero.core.config import settings
    monkeypatch.setattr(settings.memory, "max_retrieved_context_chars", 50)
    
    with TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        monkeypatch.setattr(settings.memory, "index_path", str(tmp / "index.json"))
        
        mem = SemanticMemory()
        
        docs_dir = tmp / "knowledge"
        docs_dir.mkdir()
        (docs_dir / "doc1.md").write_text("a very long document about classification that exceeds fifty characters easily")
        (docs_dir / "doc2.md").write_text("another document about classification")
        
        mem.ingest(docs_dir)
        
        p_ctx = PerceptualContext(task=TaskContext(objective="classification"))
        retrieved = mem.retrieve(p_ctx)
        
        # Max chars is 50, first chunk is 78 chars. It includes at least one chunk before breaking.
        assert len(retrieved.chunks) == 1


def test_coder_receives_retrieved_knowledge():
    client = get_llm_client(use_mock=True)
    coder = CoderAgent(llm_client=client)
    
    req = CodeGenerationRequest(
        perceptual_context_json="{}",
        retrieved_knowledge_json='{"chunks": [{"content": "USE API X"}]}'
    )
    
    # We patch the llm_client to intercept the prompt
    prompts = []
    def fake_gen(prompt, schema):
        prompts.append(prompt)
        from mlzero.schemas.coder import CodeArtifact
        return CodeArtifact(code="pass")
        
    client.generate_structured = fake_gen
    coder.process(req)
    
    assert len(prompts) == 1
    assert "USE API X" in prompts[0]
    assert "EXTERNAL KNOWLEDGE" in prompts[0]
