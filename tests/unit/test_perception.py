"""
Tests for perception agents.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from mlzero.agents.perception import (
    FilePerceptionAgent,
    LibrarySelectorAgent,
    TaskPerceptionAgent,
)
from mlzero.core.llm import get_llm_client
from mlzero.schemas.perception import TaskContext


@pytest.fixture
def temp_dataset():
    with TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        # 1. Create a valid CSV
        (tmp_path / "train.csv").write_text("id,feature1,target\n1,10.5,1\n2,20.1,0\n")
        
        # 2. Create a malformed CSV (but our reader should just read what it can)
        (tmp_path / "bad.csv").write_text("id,feat\n1\n2,3,4,5\n")
        
        # 3. Create a README
        (tmp_path / "README.md").write_text("This is a test dataset.")
        
        # 4. Create an unsupported binary
        (tmp_path / "model.bin").write_bytes(b"\x00\x01\x02")
        
        # 5. Create a nested dir
        nested = tmp_path / "nested"
        nested.mkdir()
        (nested / "test.csv").write_text("id,feature1\n3,30.2\n")
        
        yield tmp_path


def test_file_perception_agent(temp_dataset):
    agent = FilePerceptionAgent(dataset_dir=temp_dataset)
    contexts = agent.process()
    
    assert len(contexts) == 5
    
    # Check CSV
    train_ctx = next(c for c in contexts if c.metadata.path == "train.csv")
    assert train_ctx.metadata.file_type == "tabular"
    assert train_ctx.columns == ["id", "feature1", "target"]
    assert train_ctx.row_count == 3
    assert len(train_ctx.sample_rows) == 2
    
    # Check malformed CSV
    bad_ctx = next(c for c in contexts if c.metadata.path == "bad.csv")
    assert bad_ctx.metadata.file_type == "tabular"
    # Even malformed, it reads headers
    assert bad_ctx.columns == ["id", "feat"]
    
    # Check document
    readme_ctx = next(c for c in contexts if c.metadata.path == "README.md")
    assert readme_ctx.metadata.file_type == "document"
    assert "test dataset" in readme_ctx.extracted_text
    
    # Check binary
    bin_ctx = next(c for c in contexts if c.metadata.path == "model.bin")
    assert bin_ctx.metadata.file_type == "binary_or_unsupported"


def test_file_perception_agent_missing_dir():
    agent = FilePerceptionAgent(dataset_dir="/does/not/exist")
    with pytest.raises(ValueError):
        agent.process()


def test_task_perception_agent(temp_dataset):
    llm = get_llm_client(use_mock=True)
    agent = TaskPerceptionAgent(llm_client=llm)
    
    file_agent = FilePerceptionAgent(dataset_dir=temp_dataset)
    file_contexts = file_agent.process()
    
    task_ctx = agent.process(file_contexts, user_instruction="Build model")
    
    assert isinstance(task_ctx, TaskContext)
    assert task_ctx.task_type == "classification"
    assert task_ctx.target_column == "target"


def test_library_selector_agent(temp_dataset):
    llm = get_llm_client(use_mock=True)
    agent = LibrarySelectorAgent(llm_client=llm)
    
    # Setup dummy contexts
    task_ctx = TaskContext(task_type="classification")
    file_agent = FilePerceptionAgent(dataset_dir=temp_dataset)
    file_contexts = file_agent.process()
    
    selection = agent.process(task_ctx, file_contexts)
    
    assert selection.selected_library == "autogluon.tabular"
    assert selection.confidence == "high"
