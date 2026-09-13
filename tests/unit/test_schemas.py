"""Tests for schemas."""

from mlzero.schemas.base import AgentResponse, AgentTask
from mlzero.schemas.perception import FileContext, FileMetadata


def test_agent_task_creation() -> None:
    """Test creating an AgentTask."""
    task = AgentTask(task_id="t1", description="Test task")
    assert task.task_id == "t1"
    assert task.description == "Test task"
    assert task.metadata == {}


def test_agent_response_creation() -> None:
    """Test creating an AgentResponse."""
    response = AgentResponse(task_id="t1", status="success", output={"result": 42})
    assert response.task_id == "t1"
    assert response.status == "success"
    assert response.output == {"result": 42}
    assert response.errors == []


def test_file_metadata_creation() -> None:
    """Test creating FileMetadata."""
    meta = FileMetadata(
        path="data.csv",
        absolute_path="/tmp/data.csv",
        extension=".csv",
        size_bytes=100,
        file_type="tabular"
    )
    assert meta.path == "data.csv"
    assert meta.file_type == "tabular"

def test_file_context_creation() -> None:
    meta = FileMetadata(
        path="data.csv",
        absolute_path="/tmp/data.csv",
        extension=".csv",
        size_bytes=100,
        file_type="tabular"
    )
    ctx = FileContext(metadata=meta, columns=["a", "b"], row_count=5)
    assert ctx.metadata == meta
    assert ctx.columns == ["a", "b"]
    assert ctx.row_count == 5
