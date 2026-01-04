"""Qdrant storage client for Engram memory system.

This module provides the main EngramStorage class that combines
all storage operations through mixins.

Example:
    ```python
    from engram.storage import EngramStorage

    async with EngramStorage() as storage:
        await storage.store_episode(episode)
        results = await storage.search_episodes(query_vector, user_id="user_123")
    ```
"""

from __future__ import annotations

from typing import Any

from .audit import AuditMixin
from .base import COLLECTION_NAMES, DEFAULT_EMBEDDING_DIM, StorageBase
from .crud import CRUDMixin
from .search import SearchMixin
from .store import StoreMixin


class EngramStorage(StoreMixin, SearchMixin, CRUDMixin, AuditMixin, StorageBase):
    """Async Qdrant storage client for Engram memories.

    Handles collection management, multi-tenancy isolation, and CRUD operations
    for all memory types. Uses async Qdrant client for non-blocking I/O.

    This class combines functionality from multiple mixins:
    - StoreMixin: store_episode, store_fact, store_semantic, etc.
    - SearchMixin: search_episodes, search_facts, search_semantic, etc.
    - CRUDMixin: get_episode, delete_episode, get_fact, delete_fact, etc.
    - AuditMixin: log_audit, get_audit_log

    Attributes:
        client: Async Qdrant client instance.

    Example:
        ```python
        storage = EngramStorage()
        await storage.initialize()

        # Store an episode
        episode = Episode(content="Hello", role="user", user_id="user_123", embedding=[...])
        await storage.store_episode(episode)

        # Search for memories
        results = await storage.search_episodes(
            query_vector=[0.1, 0.2, ...],
            user_id="user_123",
            limit=10,
        )
        ```
    """

    async def __aenter__(self) -> EngramStorage:
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()


__all__ = [
    "EngramStorage",
    "COLLECTION_NAMES",
    "DEFAULT_EMBEDDING_DIM",
]
