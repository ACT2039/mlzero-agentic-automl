"""
Stage 6 Unit Tests for Multimodal Perception, File Grouping, Bounded Inspection,
Task Perception, and Library Selection.
"""

from pathlib import Path

from mlzero.agents.perception import (
    DataProfiler,
    FileGroupingAgent,
    FilePerceptionAgent,
    LibrarySelectorAgent,
    TaskPerceptionAgent,
)
from mlzero.core.llm import MockLLMClient
from mlzero.schemas.perception import (
    FileGroup,
    PerceptualContext,
    TaskContext,
)


def test_file_type_detection(tmp_path: Path):
    """Test detection of CSV, TSV, JSON, TXT, MD, PNG, WAV, binary files."""
    (tmp_path / "data.csv").write_text("a,b\n1,2")
    (tmp_path / "data.tsv").write_text("a\tb\n1\t2")
    (tmp_path / "meta.json").write_text('{"key": "val"}')
    (tmp_path / "readme.md").write_text("# Task")
    (tmp_path / "notes.txt").write_text("notes")
    (tmp_path / "sample.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (tmp_path / "audio.wav").write_bytes(b"RIFF\x00\x00\x00\x00WAVE")
    (tmp_path / "model.bin").write_bytes(b"\x00\x01\x02\x03")

    agent = FilePerceptionAgent(dataset_dir=tmp_path)
    contexts = agent.process()

    types = {fc.metadata.path: fc.metadata.file_type for fc in contexts}
    assert types["data.csv"] == "tabular"
    assert types["data.tsv"] == "tabular"
    assert types["meta.json"] == "json"
    assert types["readme.md"] == "document"
    assert types["notes.txt"] == "document"
    assert types["sample.png"] == "image"
    assert types["audio.wav"] == "audio"
    assert types["model.bin"] == "binary_or_unsupported"


def test_file_grouping(tmp_path: Path):
    """Test FileGroupingAgent train/test split, image metadata, and multi-table grouping."""
    (tmp_path / "train.csv").write_text("a,b\n1,2")
    (tmp_path / "test.csv").write_text("a,b\n3,4")

    agent_p = FilePerceptionAgent(dataset_dir=tmp_path)
    fctx = agent_p.process()

    grouping_agent = FileGroupingAgent(dataset_dir=tmp_path)
    groups = grouping_agent.process(fctx)

    assert len(groups) >= 1
    split_group = next((g for g in groups if g.relationship == "train_test_pair"), None)
    assert split_group is not None
    assert "train.csv" in split_group.files
    assert "test.csv" in split_group.files


def test_image_perception(tmp_path: Path):
    """Test lightweight Image inspection."""
    img_file = tmp_path / "test_img.png"
    img_file.write_bytes(b"\x89PNG\r\n\x1a\n")

    agent = FilePerceptionAgent(dataset_dir=tmp_path)
    contexts = agent.process()
    img_ctx = next(fc for fc in contexts if fc.metadata.path == "test_img.png")
    assert img_ctx.image_summary is not None
    assert img_ctx.image_summary.format in ("PNG", ".PNG")


def test_text_and_json_perception(tmp_path: Path):
    """Test text and JSON structural inspection."""
    (tmp_path / "info.json").write_text('{"name": "test", "items": [1, 2, 3]}')
    (tmp_path / "readme.txt").write_text("Line 1\nLine 2\nLine 3")

    agent = FilePerceptionAgent(dataset_dir=tmp_path)
    contexts = agent.process()

    json_ctx = next(fc for fc in contexts if fc.metadata.path == "info.json")
    assert json_ctx.json_summary is not None
    assert "name" in json_ctx.json_summary.keys

    txt_ctx = next(fc for fc in contexts if fc.metadata.path == "readme.txt")
    assert txt_ctx.text_summary is not None
    assert "Line 1" in txt_ctx.text_summary.first_lines


def test_readme_and_task_perception(tmp_path: Path):
    """Test TaskPerceptionAgent extracts README context and infers task parameters."""
    (tmp_path / "train.csv").write_text("age,tenure,churn\n25,12,0\n45,3,1")
    (tmp_path / "README.md").write_text("# Customer Churn Task\nPredict customer churn.")

    agent_p = FilePerceptionAgent(dataset_dir=tmp_path)
    fctx = agent_p.process()

    profiler = DataProfiler(tmp_path)
    dq = profiler.profile()

    mock_llm = MockLLMClient()
    task_agent = TaskPerceptionAgent(llm_client=mock_llm, dataset_dir=tmp_path)
    task_ctx = task_agent.process(fctx, data_quality=dq)

    assert task_ctx.target_column == "churn"
    assert task_ctx.task_type == "classification"
    assert task_ctx.relevant_description_files is not None


def test_library_selection_multimodal_perceptual_context():
    """Test LibrarySelectorAgent routes per perceptual context."""
    mock_llm = MockLLMClient()
    lib_agent = LibrarySelectorAgent(llm_client=mock_llm)

    tctx = TaskContext(task_type="multimodal")
    sel = lib_agent.process(tctx, [])
    assert sel.selected_library in ("autogluon.multimodal", "autogluon.tabular")


def test_perceptual_context_schema_backward_compatibility():
    """Test PerceptualContext instantiation and backward compatibility."""
    ctx = PerceptualContext(
        file_groups=[FileGroup(group_id="g1", modality="tabular", relationship="train_test_pair")],
        modalities=["tabular"],
        train_files=["train.csv"],
        test_files=["test.csv"],
        task_type="classification",
        target_column="target",
        selected_library="autogluon.tabular",
    )
    assert ctx.selected_library == "autogluon.tabular"
    assert len(ctx.file_groups) == 1
    assert ctx.file_groups[0].group_id == "g1"


def test_timeseries_task_detection(tmp_path: Path):
    """Test data-aware time series forecasting task detection."""
    (tmp_path / "train.csv").write_text("item_id,timestamp,target\n1,2025-01-01 00:00:00,10.5\n1,2025-01-01 01:00:00,12.0\n")
    profiler = DataProfiler(tmp_path)
    dq = profiler.profile()
    assert dq.task_type == "time_series_forecasting"
    assert dq.target_column == "target"
    assert dq.timestamp_column == "timestamp"
    assert dq.id_column == "item_id"


def test_timeseries_library_selection():
    """Test LibrarySelectorAgent routes time_series_forecasting to autogluon.timeseries."""
    mock_llm = MockLLMClient()
    lib_agent = LibrarySelectorAgent(llm_client=mock_llm)
    tctx = TaskContext(task_type="time_series_forecasting", target_column="target", timestamp_column="timestamp", id_column="item_id")
    sel = lib_agent.process(tctx, [])
    assert sel.selected_library == "autogluon.timeseries"


def test_timeseries_perceptual_context(tmp_path: Path):
    """Test full pipeline perception produces time-series PerceptualContext."""
    ts_dir = Path("tests/data/tiny_timeseries")
    agent_p = FilePerceptionAgent(dataset_dir=ts_dir)
    fctx = agent_p.process()

    grouping_agent = FileGroupingAgent(dataset_dir=ts_dir)
    fgroups = grouping_agent.process(fctx)

    profiler = DataProfiler(ts_dir)
    dq = profiler.profile()

    mock_llm = MockLLMClient()
    task_agent = TaskPerceptionAgent(llm_client=mock_llm, dataset_dir=ts_dir)
    tctx = task_agent.process(fctx, data_quality=dq, file_groups=fgroups)

    lib_agent = LibrarySelectorAgent(llm_client=mock_llm)
    lib_sel = lib_agent.process(tctx, fctx)

    context = PerceptualContext(
        files=fctx,
        file_groups=fgroups,
        task=tctx,
        task_type=tctx.task_type,
        target_column=tctx.target_column,
        timestamp_column=tctx.timestamp_column,
        id_column=tctx.id_column,
        library=lib_sel,
        selected_library=lib_sel.selected_library,
    )

    assert context.task_type == "time_series_forecasting"
    assert context.selected_library == "autogluon.timeseries"
    assert context.target_column == "target"
    assert context.timestamp_column == "timestamp"
    assert context.id_column == "item_id"


def test_retrieval_task_detection(tmp_path: Path):
    """Test structural retrieval task detection from query/corpus JSON."""
    (tmp_path / "corpus.json").write_text('{"query": "What is MLZero?", "documents": ["doc1", "doc2"]}')
    agent_p = FilePerceptionAgent(dataset_dir=tmp_path)
    fctx = agent_p.process()

    grouping_agent = FileGroupingAgent(dataset_dir=tmp_path)
    fgroups = grouping_agent.process(fctx)

    mock_llm = MockLLMClient()
    task_agent = TaskPerceptionAgent(llm_client=mock_llm, dataset_dir=tmp_path)
    tctx = task_agent.process(fctx, file_groups=fgroups)

    assert tctx.task_type == "retrieval"
    assert tctx.target_column is None


def test_retrieval_library_selection():
    """Test LibrarySelectorAgent routes retrieval task to FlagEmbedding."""
    mock_llm = MockLLMClient()
    lib_agent = LibrarySelectorAgent(llm_client=mock_llm)
    tctx = TaskContext(task_type="retrieval")
    sel = lib_agent.process(tctx, [])
    assert sel.selected_library == "FlagEmbedding"


def test_retrieval_perceptual_context(tmp_path: Path):
    """Test full pipeline perception produces retrieval PerceptualContext."""
    ret_dir = Path("tests/data/tiny_retrieval")
    agent_p = FilePerceptionAgent(dataset_dir=ret_dir)
    fctx = agent_p.process()

    grouping_agent = FileGroupingAgent(dataset_dir=ret_dir)
    fgroups = grouping_agent.process(fctx)

    profiler = DataProfiler(ret_dir)
    dq = profiler.profile()

    mock_llm = MockLLMClient()
    task_agent = TaskPerceptionAgent(llm_client=mock_llm, dataset_dir=ret_dir)
    tctx = task_agent.process(fctx, data_quality=dq, file_groups=fgroups)

    lib_agent = LibrarySelectorAgent(llm_client=mock_llm)
    lib_sel = lib_agent.process(tctx, fctx)

    context = PerceptualContext(
        files=fctx,
        file_groups=fgroups,
        task=tctx,
        task_type=tctx.task_type,
        target_column=tctx.target_column,
        library=lib_sel,
        selected_library=lib_sel.selected_library,
    )

    assert context.task_type == "retrieval"
    assert context.selected_library == "FlagEmbedding"

