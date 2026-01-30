"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

# Add tests directory to path so utils can be imported
tests_dir = Path(__file__).parent
if str(tests_dir) not in sys.path:
    sys.path.insert(0, str(tests_dir))

if TYPE_CHECKING:
    pass


class MockTransactionContext:
    """Mock transaction context that delegates to the storage mock.

    This allows tests to use the transaction context pattern without
    changing the underlying mock storage assertions.
    """

    def __init__(self, storage: AsyncMock) -> None:
        """Initialize with a mock storage.

        Args:
            storage: The mock storage to delegate to.
        """
        self._storage = storage
        self.operations: list = []
        self.committed = False

    async def store_episode(self, episode):
        """Delegate to storage.store_episode."""
        result = await self._storage.store_episode(episode)
        self.operations.append(("store_episode", episode.id))
        return result

    async def store_structured(self, memory):
        """Delegate to storage.store_structured."""
        result = await self._storage.store_structured(memory)
        self.operations.append(("store_structured", memory.id))
        return result

    async def update_episode(self, episode):
        """Delegate to storage.update_episode."""
        await self._storage.update_episode(episode)
        self.operations.append(("update_episode", episode.id))

    async def log_audit(self, entry):
        """Delegate to storage.log_audit."""
        result = await self._storage.log_audit(entry)
        self.operations.append(("log_audit", entry.id))
        return result

    def commit(self) -> None:
        """Mark the transaction as committed."""
        self.committed = True
        self.operations.clear()

    async def rollback(self) -> int:
        """Mock rollback - just clear operations."""
        count = len(self.operations)
        self.operations.clear()
        return count

    async def __aenter__(self):
        """Enter the transaction context."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the transaction context."""
        if exc_type is not None and not self.committed:
            await self.rollback()
        return False


def add_transaction_support(storage: AsyncMock) -> AsyncMock:
    """Add transaction support to a mock storage.

    This makes the storage.transaction() method return a MockTransactionContext
    that delegates to the underlying storage mock methods.

    Args:
        storage: The mock storage to enhance.

    Returns:
        The same storage mock with transaction support added.

    Example:
        ```python
        storage = AsyncMock()
        storage.store_episode = AsyncMock(return_value="ep_123")
        add_transaction_support(storage)

        # Now can use storage.transaction() as async context manager
        async with storage.transaction() as txn:
            await txn.store_episode(episode)
            txn.commit()
        ```
    """

    def create_transaction():
        return MockTransactionContext(storage)

    storage.transaction = create_transaction
    return storage
