"""
Perception Agents for MLZero.
Includes interfaces for File Perception, Task Perception, and ML Library Selection.
"""

from abc import ABC, abstractmethod

from mlzero.core.logger import setup_logger
from mlzero.schemas.base import AgentResponse, AgentTask

logger = setup_logger(__name__)


class BasePerceptionAgent(ABC):
    """Base class for all perception agents."""

    @abstractmethod
    def process(self, task: AgentTask) -> AgentResponse:
        """Process a perception task."""


class FilePerceptionAgent(BasePerceptionAgent):
    """Agent responsible for perceiving and understanding files/datasets."""

    def process(self, task: AgentTask) -> AgentResponse:
        """
        Analyze a file or dataset.
        (Placeholder for Phase 2 implementation)
        """
        logger.info(f"FilePerceptionAgent processing task: {task.task_id}")
        return AgentResponse(
            task_id=task.task_id,
            status="not_implemented",
            output={"message": "Phase 1 placeholder for File Perception"}
        )


class TaskPerceptionAgent(BasePerceptionAgent):
    """Agent responsible for understanding the overall user task."""

    def process(self, task: AgentTask) -> AgentResponse:
        """
        Analyze the task description.
        (Placeholder for Phase 2 implementation)
        """
        logger.info(f"TaskPerceptionAgent processing task: {task.task_id}")
        return AgentResponse(
            task_id=task.task_id,
            status="not_implemented",
            output={"message": "Phase 1 placeholder for Task Perception"}
        )


class MLLibrarySelectionAgent(BasePerceptionAgent):
    """Agent responsible for selecting appropriate ML libraries."""

    def process(self, task: AgentTask) -> AgentResponse:
        """
        Select ML libraries based on the perceived task and data.
        (Placeholder for Phase 2 implementation)
        """
        logger.info(f"MLLibrarySelectionAgent processing task: {task.task_id}")
        return AgentResponse(
            task_id=task.task_id,
            status="not_implemented",
            output={"message": "Phase 1 placeholder for ML Library Selection"}
        )
