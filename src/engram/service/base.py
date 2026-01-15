"""Core Engram service layer.

This module provides the main EngramService that combines
storage, extraction, and embeddings into a simple encode/recall interface.

Example:
    ```python
    from engram.service import EngramService

    async with EngramService() as engram:
        # Store a memory
        result = await engram.encode(
            content="My email is user@example.com",
            role="user",
            user_id="user_123",
        )
        print(f"Stored episode {result.episode.id}")
        print(f"Extracted: {result.structured.emails}")

        # Recall memories
        memories = await engram.recall(
            query="email address",
            user_id="user_123",
        )
        for memory in memories:
            print(f"{memory.content} (score: {memory.score:.2f})")
    ```
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from engram.config import Settings
from engram.embeddings import Embedder, get_embedder
from engram.models import Episode
from engram.storage import EngramStorage

from .encode import EncodeMixin
from .operations import OperationsMixin
from .recall import RecallMixin

if TYPE_CHECKING:
    pass


@dataclass
class EngramService(EncodeMixin, RecallMixin, OperationsMixin):
    """High-level Engram service for encoding and recalling memories.

    This service provides a simple interface for:
    - encode(): Store text as an episode and extract structured data
    - recall(): Search memories by semantic similarity
    - get_working_memory(): Get current session's episodes
    - clear_working_memory(): Clear session context

    Uses dependency injection for storage and embeddings,
    making it easy to test and configure.

    Attributes:
        storage: Storage backend (Qdrant).
        embedder: Embedding provider (OpenAI or FastEmbed).
        settings: Configuration settings.
    """

    storage: EngramStorage
    embedder: Embedder
    settings: Settings

    # Working memory: in-memory episodes for current session (not persisted separately)
    _working_memory: list[Episode] = field(default_factory=list, init=False, repr=False)

    @classmethod
    def create(cls, settings: Settings | None = None) -> EngramService:
        """Create an EngramService with default dependencies.

        Args:
            settings: Optional settings. Uses defaults if None.

        Returns:
            Configured EngramService instance.
        """
        if settings is None:
            settings = Settings()

        # Create embedder first to get dimensions
        embedder = get_embedder(settings)

        return cls(
            storage=EngramStorage(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key,
                prefix=settings.collection_prefix,
                embedding_dim=embedder.dimensions,
            ),
            embedder=embedder,
            settings=settings,
        )

    async def initialize(self) -> None:
        """Initialize the service (storage collections, etc.)."""
        await self.storage.initialize()

    async def close(self) -> None:
        """Clean up resources and clear working memory."""
        self.clear_working_memory()
        await self.storage.close()

    async def __aenter__(self) -> EngramService:
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()

    def get_working_memory(self) -> list[Episode]:
        """Get current session's working memory.

        Working memory contains episodes from the current session.
        It's volatile (in-memory only) and cleared when the session ends.

        Returns:
            Copy of the working memory episodes list.

        Example:
            ```python
            async with EngramService.create() as engram:
                await engram.encode("Hello", role="user", user_id="u1")
                working = engram.get_working_memory()
                print(f"Session has {len(working)} episodes")
            # Working memory cleared on exit
            ```
        """
        return self._working_memory.copy()

    def clear_working_memory(self) -> None:
        """Clear working memory (typically at end of session).

        This removes all episodes from working memory without
        affecting persisted storage.
        """
        self._working_memory.clear()


__all__ = ["EngramService"]
