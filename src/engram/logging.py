"""Structured logging configuration for Engram.

Provides JSON-formatted structured logging using structlog.
Supports both development (colored console) and production (JSON) modes.
"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from structlog.typing import Processor

# Track if logging has been configured
_configured = False


def configure_logging(
    level: str = "INFO",
    format: str = "json",
) -> None:
    """Configure structured logging for Engram.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        format: Output format - "json" for production, "text" for development.

    Example:
        ```python
        from engram.logging import configure_logging, get_logger

        configure_logging(level="DEBUG", format="text")
        logger = get_logger()
        logger.info("Application started", version="0.1.0")
        ```
    """
    global _configured

    # Convert level string to logging constant
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Configure standard library logging to not duplicate structlog output
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    # Shared processors for both formats
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if format.lower() == "json":
        # Production: JSON output
        processors: list[Processor] = [
            *shared_processors,
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]
    else:
        # Development: Colored console output
        processors = [
            *shared_processors,
            structlog.dev.ConsoleRenderer(colors=True),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    _configured = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance.

    Args:
        name: Logger name. Uses calling module name if None.

    Returns:
        A bound structlog logger.

    Example:
        ```python
        from engram.logging import get_logger

        logger = get_logger(__name__)
        logger.info("Processing request", user_id="user_123", action="encode")
        ```
    """
    # Auto-configure with defaults if not already configured
    if not _configured:
        configure_logging()

    return structlog.get_logger(name)  # type: ignore[no-any-return]


def bind_context(**kwargs: object) -> None:
    """Bind context variables to all subsequent log messages.

    Context is thread-local and persists across function calls.
    Useful for adding request-scoped context like user_id or request_id.

    Args:
        **kwargs: Key-value pairs to bind to the logging context.

    Example:
        ```python
        from engram.logging import bind_context, get_logger

        bind_context(user_id="user_123", request_id="req_abc")
        logger = get_logger()
        logger.info("Processing")  # Includes user_id and request_id
        ```
    """
    structlog.contextvars.bind_contextvars(**kwargs)


def clear_context() -> None:
    """Clear all bound context variables.

    Call this at the end of a request to prevent context leaking.

    Example:
        ```python
        from engram.logging import bind_context, clear_context

        bind_context(user_id="user_123")
        # ... handle request ...
        clear_context()  # Clean up
        ```
    """
    structlog.contextvars.clear_contextvars()


def unbind_context(*keys: str) -> None:
    """Remove specific keys from the logging context.

    Args:
        *keys: Keys to remove from context.

    Example:
        ```python
        from engram.logging import bind_context, unbind_context

        bind_context(user_id="user_123", temp="value")
        unbind_context("temp")  # Remove only "temp"
        ```
    """
    structlog.contextvars.unbind_contextvars(*keys)


# Convenience: pre-configured logger for quick imports
logger = get_logger("engram")
