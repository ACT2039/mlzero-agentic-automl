"""Tests for schemas."""

from mlzero.schemas.base import AgentResponse, AgentTask


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
