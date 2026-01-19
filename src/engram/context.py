"""Memory context manager for Engram.

Provides a convenient async context manager for memory operations
with automatic cleanup and thread-safe context management.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING

from engram.config import Settings
from engram.logging import bind_context, clear_context, get_logger
from engram.service import EngramService

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)

# Thread-safe context variable for current memory instance
_memory_context: ContextVar[EngramService | None] = ContextVar("engram_memory", default=None)


def get_current_memory() -> EngramService | None:
    """Get the current memory instance from context.

    Returns:
        The current EngramService instance, or None if not in a memory context.

    Example:
        ```python
        async with memory_context(user_id="user_123") as mem:
            # Inside context, can get instance from anywhere
            current = get_current_memory()
            assert current is mem
        ```
    """
    return _memory_context.get()


@asynccontextmanager
async def memory_context(
    user_id: str,
    org_id: str | None = None,
    session_id: str | None = None,
    settings: Settings | None = None,
) -> AsyncIterator[EngramService]:
    """Context manager for memory operations.

    Creates an EngramService instance, initializes it, and cleans up on exit.
    Also binds user context to logging for request tracing.

    Args:
        user_id: User identifier for multi-tenancy isolation.
        org_id: Optional organization ID for further isolation.
        session_id: Optional session ID for grouping interactions.
        settings: Optional settings. Uses defaults if None.

    Yields:
        Initialized EngramService instance.

    Example:
        ```python
        async def handle_chat(user_id: str, message: str) -> str:
            async with memory_context(user_id=user_id) as mem:
                # Store the user message
                await mem.encode(
                    content=message,
                    role="user",
                    user_id=user_id,
                )

                # Recall relevant context
                context = await mem.recall(
                    query=message,
                    user_id=user_id,
                    limit=5,
                )

                # Generate response using context
                response = await generate_response(message, context)

                # Store the assistant response
                await mem.encode(
                    content=response,
                    role="assistant",
                    user_id=user_id,
                )

                return response
        ```
    """
    # Use provided settings or create defaults
    if settings is None:
        settings = Settings()

    # Create and initialize service
    service = EngramService.create(settings)
    await service.initialize()

    # Bind logging context
    bind_context(user_id=user_id, org_id=org_id, session_id=session_id)

    # Set context variable
    token = _memory_context.set(service)

    logger.info(
        "Memory context opened",
        user_id=user_id,
        org_id=org_id,
        session_id=session_id,
    )

    try:
        yield service
    finally:
        # Reset context variable
        _memory_context.reset(token)

        # Clear working memory
        service.clear_working_memory()

        # Close service
        await service.close()

        # Clear logging context
        clear_context()

        logger.info(
            "Memory context closed",
            user_id=user_id,
        )


@asynccontextmanager
async def scoped_memory(
    service: EngramService,
    user_id: str,
    org_id: str | None = None,
    session_id: str | None = None,
) -> AsyncIterator[EngramService]:
    """Lightweight context manager using an existing service.

    Use this when you already have an EngramService instance
    (e.g., from FastAPI dependency injection) but want scoped
    logging context and cleanup.

    Args:
        service: Existing EngramService instance.
        user_id: User identifier.
        org_id: Optional organization ID.
        session_id: Optional session ID.

    Yields:
        The same EngramService instance.

    Example:
        ```python
        # In FastAPI endpoint with injected service
        @router.post("/chat")
        async def chat(
            request: ChatRequest,
            service: EngramService = Depends(get_service),
        ):
            async with scoped_memory(service, request.user_id) as mem:
                await mem.encode(content=request.message, ...)
                ...
        ```
    """
    # Bind logging context
    bind_context(user_id=user_id, org_id=org_id, session_id=session_id)

    # Set context variable
    token = _memory_context.set(service)

    try:
        yield service
    finally:
        # Reset context variable
        _memory_context.reset(token)

        # Clear logging context
        clear_context()
