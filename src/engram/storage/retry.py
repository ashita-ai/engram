"""Retry utilities for storage operations.

Provides exponential backoff retry logic for transient network errors
when communicating with Qdrant vector database.
"""

from __future__ import annotations

import logging

import httpx
from qdrant_client.http.exceptions import UnexpectedResponse
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


def _log_retry(retry_state: RetryCallState) -> None:
    """Log retry attempts with context.

    Args:
        retry_state: Current retry state from tenacity.
    """
    if retry_state.attempt_number > 1:
        logger.warning(
            "Retrying Qdrant operation",
            extra={
                "attempt": retry_state.attempt_number,
                "fn_name": retry_state.fn.__name__ if retry_state.fn else "unknown",
                "exception": str(retry_state.outcome.exception()) if retry_state.outcome else None,
            },
        )


# Decorator for retrying transient Qdrant errors
# Only retries network/server errors, not client errors (4xx)
qdrant_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(
        (
            httpx.ConnectError,  # Connection failed
            httpx.TimeoutException,  # Request timed out
            UnexpectedResponse,  # 5xx server errors
        )
    ),
    before_sleep=_log_retry,
    reraise=True,
)
