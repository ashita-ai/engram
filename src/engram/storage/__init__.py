"""Storage backends for Engram.

This module provides the storage layer for persisting memories
to Qdrant vector database with multi-tenancy support.

Example:
    ```python
    from engram.storage import EngramStorage

    async with EngramStorage() as storage:
        await storage.store_episode(episode)
        results = await storage.search_episodes(query_vector, user_id="user_123")
    ```
"""

from .base import COLLECTION_NAMES, DEFAULT_EMBEDDING_DIM
from .client import EngramStorage
from .search import ScoredResult
from .transaction import TransactionContext

__all__ = [
    "EngramStorage",
    "ScoredResult",
    "TransactionContext",
    "COLLECTION_NAMES",
    "DEFAULT_EMBEDDING_DIM",
]
