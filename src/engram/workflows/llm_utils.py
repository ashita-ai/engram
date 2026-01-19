"""LLM agent utilities with retry and timeout support.

Provides a unified interface for running Pydantic AI agents with:
- Configurable timeout to prevent indefinite hangs
- Exponential backoff retry for transient failures
- Proper error classification (retriable vs fatal)
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, TypeVar

from pydantic import BaseModel

if TYPE_CHECKING:
    from pydantic_ai import Agent

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# Default configuration
DEFAULT_TIMEOUT_SECONDS = 60
DEFAULT_MAX_RETRIES = 3
DEFAULT_INITIAL_DELAY = 1.0
DEFAULT_MAX_DELAY = 30.0
DEFAULT_BACKOFF_FACTOR = 2.0


class LLMCallError(Exception):
    """Error from LLM agent call."""

    def __init__(self, message: str, retriable: bool = False) -> None:
        super().__init__(message)
        self.retriable = retriable


async def run_agent_with_retry(
    agent: Agent[None, T],
    prompt: str,
    *,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    initial_delay: float = DEFAULT_INITIAL_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
) -> T:
    """Run a Pydantic AI agent with timeout and retry logic.

    Args:
        agent: The Pydantic AI agent to run.
        prompt: The prompt text to send to the agent.
        timeout_seconds: Maximum time to wait for each attempt.
        max_retries: Maximum number of retry attempts (0 = no retries).
        initial_delay: Initial delay between retries in seconds.
        max_delay: Maximum delay between retries in seconds.
        backoff_factor: Multiplier for delay after each retry.

    Returns:
        The agent's typed output.

    Raises:
        LLMCallError: If all attempts fail or a non-retriable error occurs.
        asyncio.TimeoutError: If the final attempt times out.

    Example:
        >>> from pydantic_ai import Agent
        >>> agent = Agent("openai:gpt-4o-mini", output_type=SummaryOutput)
        >>> result = await run_agent_with_retry(agent, "Summarize this...")
    """
    last_error: Exception | None = None
    delay = initial_delay

    for attempt in range(max_retries + 1):
        try:
            # Run with timeout
            async with asyncio.timeout(timeout_seconds):
                result = await agent.run(prompt)
                return result.output

        except TimeoutError:
            last_error = TimeoutError(
                f"LLM call timed out after {timeout_seconds}s (attempt {attempt + 1}/{max_retries + 1})"
            )
            logger.warning(
                "LLM call timed out after %ss (attempt %d/%d)",
                timeout_seconds,
                attempt + 1,
                max_retries + 1,
            )

        except Exception as e:
            last_error = e
            error_str = str(e).lower()

            # Classify error as retriable or fatal
            retriable = _is_retriable_error(error_str)

            if not retriable:
                logger.error("Non-retriable LLM error: %s", e)
                raise LLMCallError(str(e), retriable=False) from e

            logger.warning(
                "Retriable LLM error (attempt %d/%d): %s",
                attempt + 1,
                max_retries + 1,
                e,
            )

        # Don't sleep after the last attempt
        if attempt < max_retries:
            await asyncio.sleep(delay)
            delay = min(delay * backoff_factor, max_delay)

    # All retries exhausted
    if isinstance(last_error, asyncio.TimeoutError):
        raise last_error
    raise LLMCallError(
        f"LLM call failed after {max_retries + 1} attempts: {last_error}",
        retriable=True,
    ) from last_error


def _is_retriable_error(error_str: str) -> bool:
    """Determine if an error is retriable based on error message.

    Retriable errors include:
    - Rate limits (429)
    - Server errors (500, 502, 503, 504)
    - Timeout errors
    - Connection errors

    Non-retriable errors include:
    - Validation errors (Pydantic)
    - Authentication errors (401, 403)
    - Bad request (400)
    - Model not found (404)

    Args:
        error_str: Lowercase error message string.

    Returns:
        True if the error is likely retriable.
    """
    # Retriable patterns
    retriable_patterns = [
        "rate limit",
        "rate_limit",
        "429",
        "500",
        "502",
        "503",
        "504",
        "timeout",
        "timed out",
        "connection",
        "network",
        "temporarily unavailable",
        "service unavailable",
        "overloaded",
        "capacity",
    ]

    # Non-retriable patterns
    non_retriable_patterns = [
        "validation error",
        "validationerror",
        "401",
        "403",
        "400",
        "invalid",
        "authentication",
        "unauthorized",
        "forbidden",
        "not found",
        "404",
    ]

    # Check for non-retriable first (more specific)
    for pattern in non_retriable_patterns:
        if pattern in error_str:
            return False

    # Check for retriable patterns
    for pattern in retriable_patterns:
        if pattern in error_str:
            return True

    # Default to retriable for unknown errors (be optimistic)
    return True
