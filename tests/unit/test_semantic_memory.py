"""
Comprehensive tests for Stage 2: Paper-Faithful Semantic Memory Module.
All tests use MockLLMClient or mocked components (no Gemini quota consumed).
"""
from unittest.mock import MagicMock

from mlzero.agents.coder import CoderAgent, ErrorAnalyzerAgent, ExecutorAgent
from mlzero.agents.memory import CondensationAgent, RetrievalAgent, SummarizationAgent
from mlzero.core.config import settings
from mlzero.core.llm import MockLLMClient, get_llm_client
from mlzero.memory.index import SemanticIndex
from mlzero.memory.ingestion import ingest_knowledge_pipeline
from mlzero.memory.semantic import SemanticMemory
from mlzero.orchestration.iterative import IterativeCodingOrchestrator
from mlzero.schemas.coder import (
    CodeArtifact,
    CodeGenerationRequest,
    ErrorContext,
)
from mlzero.schemas.memory import (
    KnowledgeChunk,
    KnowledgeDocument,
    LibraryKnowledgeRegistration,
)
from mlzero.schemas.perception import (
    LibrarySelection,
    PerceptualContext,
    TaskContext,
)


def test_summarization_agent():
    """Verify SummarizationAgent produces concise summaries via LLM."""
    client = get_llm_client(use_mock=True)
    summarizer = SummarizationAgent(client)

    chunk = KnowledgeChunk(
        chunk_id="c1",
        document_id="d1",
        content="AutoGluon Tabular requires exact label column matching.",
        source="autogluon_tabular.md",
    )
    summary = summarizer.summarize(chunk)
    assert isinstance(summary, str)
    assert len(summary) > 0


def test_condensation_agent():
    """Verify CondensationAgent distills technical text into implementation guidance."""
    client = get_llm_client(use_mock=True)
    condenser = CondensationAgent(client)

    chunk = KnowledgeChunk(
        chunk_id="c1",
        document_id="d1",
        content="Detailed documentation on TabularPredictor parameters and fit kwargs.",
        source="autogluon_tabular.md",
    )
    guidance = condenser.condense(chunk)
    assert isinstance(guidance, str)
    assert len(guidance) > 0


def test_retrieval_agent_default_top_5():
    """Verify RetrievalAgent retrieves top-5 documents by default, matching the paper."""
    index = SemanticIndex()
    chunks = [
        KnowledgeChunk(
            chunk_id=f"c_{i}",
            document_id=f"d_{i}",
            content=f"Tabular predictor documentation chunk number {i}",
            source=f"doc_{i}.md",
            library_name="autogluon.tabular",
        )
        for i in range(10)
    ]
    index.add(chunks)

    agent = RetrievalAgent(index)
    p_ctx = PerceptualContext(
        task=TaskContext(objective="Train classification model", task_type="classification"),
        library=LibrarySelection(selected_library="autogluon.tabular", confidence="high", explanation="Test"),
    )

    retrieved = agent.retrieve(perceptual_context=p_ctx)
    assert len(retrieved.chunks) == 5
    assert settings.memory.retrieval_top_k == 5


def test_retrieval_agent_library_filtering():
    """Verify RetrievalAgent filters documents by the selected ML library."""
    index = SemanticIndex()
    chunks = [
        KnowledgeChunk(
            chunk_id="tab_1",
            document_id="d1",
            content="Classification guide for tabular predictor.",
            source="tabular.md",
            library_name="autogluon.tabular",
        ),
        KnowledgeChunk(
            chunk_id="multi_1",
            document_id="d2",
            content="Classification guide for multimodal predictor.",
            source="multimodal.md",
            library_name="autogluon.multimodal",
        ),
    ]
    index.add(chunks)

    agent = RetrievalAgent(index)
    p_ctx = PerceptualContext(
        task=TaskContext(objective="Classification", task_type="classification"),
        library=LibrarySelection(selected_library="autogluon.tabular", confidence="high", explanation="Test"),
    )

    retrieved = agent.retrieve(perceptual_context=p_ctx, selected_library="autogluon.tabular")
    assert len(retrieved.chunks) == 1
    assert retrieved.chunks[0].chunk_id == "tab_1"


def test_error_aware_query_construction():
    """Verify that the retrieval query differs between iteration 1 and error recovery."""
    index = SemanticIndex()
    agent = RetrievalAgent(index)

    p_ctx = PerceptualContext(
        task=TaskContext(
            objective="Predict customer churn",
            task_type="classification",
            target_column="target",
        ),
        library=LibrarySelection(selected_library="autogluon.tabular", confidence="high", explanation="Test"),
    )

    # Iteration 1 query (no error context)
    q1, _lib1 = agent.build_query(perceptual_context=p_ctx, error_context=None)
    assert "target" in q1
    assert "classification" in q1
    assert "autogluon.tabular" in q1
    assert "KeyError" not in q1

    # Iteration 2 query (with error context)
    err = ErrorContext(
        iteration=1,
        error_category="KeyError",
        error_message="KeyError: 'targt' not found in DataFrame.",
        suggested_fix="Correct label column name to 'target'.",
        stderr_excerpt="KeyError: 'targt'",
    )
    q2, _lib2 = agent.build_query(perceptual_context=p_ctx, error_context=err)
    assert "KeyError" in q2
    assert "targt" in q2
    assert "Correct label" in q2
    assert q1 != q2


def test_offline_ingestion_pipeline():
    """Verify raw docs -> SummarizationAgent -> CondensationAgent -> condensed chunks."""
    client = get_llm_client(use_mock=True)
    summarizer = SummarizationAgent(client)
    condenser = CondensationAgent(client)

    doc = KnowledgeDocument(
        document_id="doc-test-1",
        source="quickstart.md",
        content="This is full documentation for AutoGluon TabularPredictor.",
        library_name="autogluon.tabular",
    )

    chunks = ingest_knowledge_pipeline(
        documents=[doc],
        summarizer=summarizer,
        condenser=condenser,
        summarize=True,
        condense=True,
    )

    assert len(chunks) > 0
    first_chunk = chunks[0]
    assert len(first_chunk.summary) > 0
    assert len(first_chunk.condensed_guidance) > 0


def test_library_knowledge_registry():
    """Verify LibraryKnowledgeRegistration in SemanticMemory."""
    mem = SemanticMemory()
    reg = LibraryKnowledgeRegistration(
        library_name="autogluon.tabular",
        version="1.1.1",
        description="AutoML for tabular data",
        documentation_path="knowledge/autogluon_tabular",
    )
    mem.register_library_knowledge(reg)

    assert "autogluon.tabular" in mem.registry
    assert mem.registry["autogluon.tabular"].version == "1.1.1"


def test_coder_prompt_four_sections():
    """Verify CoderAgent formats prompt with 4 distinct, segregated sections."""
    client = get_llm_client(use_mock=True)
    coder = CoderAgent(llm_client=client)

    req = CodeGenerationRequest(
        perceptual_context_json='{"task": {"objective": "Predict price", "target_column": "price", "task_type": "regression"}}',
        user_instruction="Train a model",
        retrieved_knowledge_json='{"chunks": [{"source": "tabular.md", "condensed_guidance": "Use TabularPredictor"}]}',
        previous_code="predictor = TabularPredictor(label='wrong')",
        error_context_json='{"error_category": "KeyError", "suggested_fix": "Fix label"}',
    )

    captured_prompts = []

    def fake_generate(prompt, schema, **kwargs):
        captured_prompts.append(prompt)
        return CodeArtifact(code="print('SUCCESS')")

    client.generate_structured = fake_generate
    coder.process(req)

    assert len(captured_prompts) == 1
    prompt = captured_prompts[0]

    # Check 4 distinct sections
    assert "=== 1. PERCEPTUAL CONTEXT (DATASET & TASK) ===" in prompt
    assert "=== 2. USER INSTRUCTION ===" in prompt
    assert "=== 3. SEMANTIC MEMORY / EXTERNAL KNOWLEDGE (RETRIEVED KNOWLEDGE - TOP-5 CONDENSED DOCS) ===" in prompt
    assert "=== 4. EPISODIC & ERROR RECOVERY CONTEXT ===" in prompt
    assert "Use TabularPredictor" in prompt


def test_per_iteration_retrieval_in_orchestrator():
    """Verify SemanticMemory retrieval is invoked on every iteration with updated error context."""
    mock_llm = MockLLMClient()
    coder = CoderAgent(mock_llm)
    executor = ExecutorAgent()
    analyzer = ErrorAnalyzerAgent(mock_llm)

    mock_semantic = MagicMock()
    mock_retrieved_1 = MagicMock()
    mock_retrieved_1.chunks = [MagicMock(content="chunk1")]
    mock_retrieved_1.model_dump_json.return_value = '{"chunks": []}'

    mock_retrieved_2 = MagicMock()
    mock_retrieved_2.chunks = [MagicMock(content="chunk2")]
    mock_retrieved_2.model_dump_json.return_value = '{"chunks": []}'

    mock_semantic.retrieve.side_effect = [mock_retrieved_1, mock_retrieved_2]

    orchestrator = IterativeCodingOrchestrator(
        coder=coder,
        executor=executor,
        error_analyzer=analyzer,
        max_iterations=2,
        semantic_memory=mock_semantic,
    )

    p_ctx = PerceptualContext(
        task=TaskContext(
            objective="Train classification",
            task_type="classification",
            target_column="target",
            input_data_files=["tests/data/tiny_classification/train.csv"],
        ),
        library=LibrarySelection(selected_library="autogluon.tabular", confidence="high", explanation="Test"),
    )

    orchestrator.process(perceptual_context=p_ctx)

    # Verify retrieval was called at least twice (Iteration 1 and Iteration 2)
    assert mock_semantic.retrieve.call_count >= 2
    # First call: error_context=None, iteration=1
    first_call_kwargs = mock_semantic.retrieve.call_args_list[0][1]
    assert first_call_kwargs.get("error_context") is None
    assert first_call_kwargs.get("iteration") == 1

    # Second call: error_context is not None, iteration=2
    second_call_kwargs = mock_semantic.retrieve.call_args_list[1][1]
    assert second_call_kwargs.get("error_context") is not None
    assert second_call_kwargs.get("iteration") == 2


def test_semantic_memory_index_on_condensed_guidance():
    """Verify that the SemanticIndex indexes and searches condensed guidance."""
    index = SemanticIndex()
    chunk = KnowledgeChunk(
        chunk_id="chunk-guidance",
        document_id="doc-guidance",
        content="General documentation text.",
        condensed_guidance="Coerce invalid numeric strings to NaN and impute median.",
        source="cleaning.md",
        library_name="autogluon.tabular",
    )
    index.add([chunk])

    # Search for term only in condensed_guidance
    results = index.search("impute median", top_k=5)
    assert len(results) == 1
    assert results[0][0].chunk_id == "chunk-guidance"
