"""Transaction support for Engram storage.

Provides compensating transaction pattern since Qdrant doesn't support
native transactions. Tracks operations and rolls back on failure.

Example:
    ```python
    async with storage.transaction() as txn:
        await txn.store_episode(episode)
        await txn.store_structured(structured)
        # If any operation fails, previous operations are rolled back
    ```
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from engram.models import AuditEntry, Episode, StructuredMemory
    from engram.storage import EngramStorage

logger = logging.getLogger(__name__)


class OperationType(str, Enum):
    """Types of storage operations that can be tracked and rolled back."""

    STORE_EPISODE = "store_episode"
    STORE_STRUCTURED = "store_structured"
    UPDATE_EPISODE = "update_episode"
    LOG_AUDIT = "log_audit"


@dataclass
class TrackedOperation:
    """A storage operation that can be rolled back.

    Attributes:
        operation: The type of operation performed.
        entity_id: The ID of the entity affected.
        user_id: The user ID for multi-tenancy.
        original_data: Original data for updates (to restore on rollback).
    """

    operation: OperationType
    entity_id: str
    user_id: str
    original_data: dict[str, object] | None = None


@dataclass
class TransactionContext:
    """Context manager for compensating transactions.

    Tracks all storage operations and rolls them back on failure.
    Since Qdrant doesn't support native transactions, this implements
    a compensating transaction pattern where:
    1. Operations are tracked as they execute
    2. On failure, compensating operations (deletes) undo previous work
    3. On success, the tracked operations are committed (no-op)

    Note: This provides eventual consistency, not strict ACID.
    There's a small window where partial data may be visible.

    Attributes:
        storage: The EngramStorage instance.
        operations: List of tracked operations for rollback.
        committed: Whether the transaction has been committed.
    """

    storage: EngramStorage
    operations: list[TrackedOperation] = field(default_factory=list)
    committed: bool = False
    _original_episode_data: dict[str, dict[str, object]] = field(default_factory=dict)

    async def store_episode(self, episode: Episode) -> str:
        """Store an episode with transaction tracking.

        Args:
            episode: The episode to store.

        Returns:
            The episode ID.

        Raises:
            Exception: If the storage operation fails.
        """
        episode_id = await self.storage.store_episode(episode)
        self.operations.append(
            TrackedOperation(
                operation=OperationType.STORE_EPISODE,
                entity_id=episode_id,
                user_id=episode.user_id,
            )
        )
        return episode_id

    async def store_structured(self, memory: StructuredMemory) -> str:
        """Store a structured memory with transaction tracking.

        Args:
            memory: The structured memory to store.

        Returns:
            The memory ID.

        Raises:
            Exception: If the storage operation fails.
        """
        memory_id = await self.storage.store_structured(memory)
        self.operations.append(
            TrackedOperation(
                operation=OperationType.STORE_STRUCTURED,
                entity_id=memory_id,
                user_id=memory.user_id,
            )
        )
        return memory_id

    async def update_episode(self, episode: Episode) -> None:
        """Update an episode with transaction tracking.

        Captures original state for rollback.

        Args:
            episode: The episode to update.

        Raises:
            Exception: If the storage operation fails.
        """
        # Capture original state if not already captured
        if episode.id not in self._original_episode_data:
            original = await self.storage.get_episode(episode.id, episode.user_id)
            if original:
                self._original_episode_data[episode.id] = original.model_dump(mode="json")

        await self.storage.update_episode(episode)
        self.operations.append(
            TrackedOperation(
                operation=OperationType.UPDATE_EPISODE,
                entity_id=episode.id,
                user_id=episode.user_id,
                original_data=self._original_episode_data.get(episode.id),
            )
        )

    async def log_audit(self, entry: AuditEntry) -> str:
        """Log an audit entry with transaction tracking.

        Note: Audit entries are typically not rolled back for forensics,
        but we track them for completeness. On rollback, the audit entry
        may be marked as rolled_back rather than deleted.

        Args:
            entry: The audit entry to log.

        Returns:
            The audit entry ID.

        Raises:
            Exception: If the storage operation fails.
        """
        audit_id: str = await self.storage.log_audit(entry)
        self.operations.append(
            TrackedOperation(
                operation=OperationType.LOG_AUDIT,
                entity_id=audit_id,
                user_id=entry.user_id,
                original_data=entry.model_dump(mode="json"),
            )
        )
        return audit_id

    async def rollback(self) -> int:
        """Roll back all tracked operations in reverse order.

        For store operations, deletes the created entities.
        For update operations, restores the original data.

        Returns:
            Number of operations rolled back.
        """
        rolled_back = 0

        # Roll back in reverse order (LIFO)
        for op in reversed(self.operations):
            try:
                if op.operation == OperationType.STORE_EPISODE:
                    # delete_episode returns dict with 'deleted' key
                    result = await self.storage.delete_episode(
                        episode_id=op.entity_id,
                        user_id=op.user_id,
                    )
                    if result.get("deleted", False):
                        rolled_back += 1
                        logger.debug("Rolled back store_episode: %s", op.entity_id)

                elif op.operation == OperationType.STORE_STRUCTURED:
                    # delete_structured returns bool
                    deleted = await self.storage.delete_structured(
                        memory_id=op.entity_id,
                        user_id=op.user_id,
                    )
                    if deleted:
                        rolled_back += 1
                        logger.debug("Rolled back store_structured: %s", op.entity_id)

                elif op.operation == OperationType.UPDATE_EPISODE:
                    if op.original_data:
                        # Restore original episode data
                        from engram.models import Episode

                        original_episode = Episode.model_validate(op.original_data)
                        await self.storage.update_episode(original_episode)
                        rolled_back += 1
                        logger.debug("Rolled back update_episode: %s", op.entity_id)

                elif op.operation == OperationType.LOG_AUDIT:
                    # Don't delete audit entries - they're forensic records.
                    # Instead, mark them as rolled_back so the paper trail shows
                    # the operation was attempted but did not persist.
                    if op.original_data:
                        from engram.models import AuditEntry

                        original_entry = AuditEntry.model_validate(op.original_data)
                        rolled_back_entry = original_entry.model_copy(update={"rolled_back": True})
                        await self.storage.log_audit(rolled_back_entry)
                        rolled_back += 1
                        logger.info("Marked audit entry %s as rolled_back", op.entity_id)
                    else:
                        logger.warning(
                            "Cannot mark audit entry %s as rolled_back: no original data",
                            op.entity_id,
                        )

            except Exception as e:
                # Log but continue rolling back other operations
                logger.error(
                    "Failed to roll back %s %s: %s",
                    op.operation.value,
                    op.entity_id,
                    e,
                )

        self.operations.clear()
        return rolled_back

    def commit(self) -> None:
        """Mark the transaction as committed.

        After commit, rollback will not be triggered.
        """
        self.committed = True
        self.operations.clear()
        self._original_episode_data.clear()

    async def __aenter__(self) -> TransactionContext:
        """Enter the transaction context."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> bool:
        """Exit the transaction context.

        Rolls back tracked operations if the transaction wasn't committed,
        whether due to an exception or a missing commit() call.

        Returns:
            False to propagate any exception.
        """
        if not self.committed and self.operations:
            if exc_type is not None:
                logger.warning(
                    "Transaction failed with %s: %s. Rolling back %d operations.",
                    exc_type.__name__,
                    exc_val,
                    len(self.operations),
                )
            else:
                logger.warning(
                    "Transaction exited without commit(). Rolling back %d uncommitted operations.",
                    len(self.operations),
                )
            rolled_back = await self.rollback()
            logger.info("Rolled back %d operations", rolled_back)

        return False  # Don't suppress exceptions


class TransactionMixin:
    """Mixin that adds transaction support to EngramStorage."""

    def transaction(self) -> TransactionContext:
        """Create a new transaction context.

        Usage:
            ```python
            async with storage.transaction() as txn:
                await txn.store_episode(episode)
                await txn.store_structured(structured)
                txn.commit()  # Required — uncommitted operations are rolled back on exit
            ```

        Returns:
            A TransactionContext for tracking and rolling back operations.
        """
        # Type assertion for self as EngramStorage
        from engram.storage import EngramStorage

        assert isinstance(self, EngramStorage)
        return TransactionContext(storage=self)


__all__ = [
    "OperationType",
    "TrackedOperation",
    "TransactionContext",
    "TransactionMixin",
]
