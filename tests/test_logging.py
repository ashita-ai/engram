"""Tests for Engram structured logging."""

from engram.logging import (
    bind_context,
    clear_context,
    configure_logging,
    get_logger,
    unbind_context,
)


class TestConfigureLogging:
    """Tests for logging configuration."""

    def test_configure_with_defaults(self):
        """Should configure with INFO level and JSON format by default."""
        configure_logging()
        logger = get_logger("test")
        # Should not raise
        logger.info("test message")

    def test_configure_with_debug_level(self):
        """Should accept DEBUG level."""
        configure_logging(level="DEBUG")
        logger = get_logger("test")
        logger.debug("debug message")

    def test_configure_with_text_format(self):
        """Should accept text format for development."""
        configure_logging(level="INFO", format="text")
        logger = get_logger("test")
        logger.info("text format message")

    def test_configure_with_json_format(self):
        """Should accept json format for production."""
        configure_logging(level="INFO", format="json")
        logger = get_logger("test")
        logger.info("json format message")

    def test_configure_multiple_times(self):
        """Should handle multiple configuration calls."""
        configure_logging(level="INFO")
        configure_logging(level="DEBUG")
        logger = get_logger("test")
        logger.info("after reconfigure")


class TestGetLogger:
    """Tests for logger creation."""

    def test_get_logger_with_name(self):
        """Should create logger with specified name."""
        logger = get_logger("my_module")
        assert logger is not None

    def test_get_logger_without_name(self):
        """Should create logger without name."""
        logger = get_logger()
        assert logger is not None

    def test_loggers_are_callable(self):
        """Should return callable logger instances."""
        logger = get_logger("test")
        # structlog returns a lazy proxy that becomes a BoundLogger when used
        assert callable(getattr(logger, "info", None))
        assert callable(getattr(logger, "error", None))
        assert callable(getattr(logger, "debug", None))


class TestContextBinding:
    """Tests for context variable binding."""

    def setup_method(self):
        """Clear context before each test."""
        clear_context()

    def teardown_method(self):
        """Clear context after each test."""
        clear_context()

    def test_bind_context(self):
        """Should bind context variables."""
        bind_context(user_id="user_123", request_id="req_abc")
        # Context is bound - subsequent logs would include these
        logger = get_logger("test")
        logger.info("with context")

    def test_clear_context(self):
        """Should clear all bound context."""
        bind_context(user_id="user_123")
        clear_context()
        # Context is cleared
        logger = get_logger("test")
        logger.info("context cleared")

    def test_unbind_specific_context(self):
        """Should unbind specific context keys."""
        bind_context(user_id="user_123", temp="value")
        unbind_context("temp")
        # Only "temp" is removed, "user_id" remains
        logger = get_logger("test")
        logger.info("partial unbind")


class TestLoggerUsage:
    """Tests for logger usage patterns."""

    def test_log_with_kwargs(self):
        """Should accept keyword arguments for structured data."""
        configure_logging()
        logger = get_logger("test")
        logger.info(
            "operation completed",
            user_id="user_123",
            duration_ms=150,
            success=True,
        )

    def test_log_levels(self):
        """Should support all standard log levels."""
        configure_logging(level="DEBUG")
        logger = get_logger("test")

        logger.debug("debug message")
        logger.info("info message")
        logger.warning("warning message")
        logger.error("error message")

    def test_log_with_exception(self):
        """Should handle exception logging."""
        configure_logging()
        logger = get_logger("test")

        try:
            raise ValueError("test error")
        except ValueError:
            logger.exception("caught an error")


class TestModuleLevelLogger:
    """Tests for the pre-configured module-level logger."""

    def test_import_logger(self):
        """Should be able to import pre-configured logger."""
        from engram.logging import logger

        assert logger is not None
        logger.info("using module logger")
