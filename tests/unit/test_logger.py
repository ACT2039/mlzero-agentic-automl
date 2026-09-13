"""Tests for logger module."""

import logging

from mlzero.core.logger import setup_logger


def test_setup_logger_creates_logger() -> None:
    """Test that setup_logger returns a logging.Logger instance."""
    logger = setup_logger("test_logger")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_logger"
    # Ensure it doesn't duplicate handlers if called twice
    handlers_count = len(logger.handlers)
    logger2 = setup_logger("test_logger")
    assert len(logger2.handlers) == handlers_count
