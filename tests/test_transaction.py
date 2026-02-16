"""Tests for storage transaction support."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from engram.models import AuditEntry, Episode, StructuredMemory
from engram.storage.transaction import (
    OperationType,
    TrackedOperation,
    TransactionContext,
)


def create_mock_storage() -> MagicMock:
    """Create a mock EngramStorage with all required methods."""
    storage = MagicMock()
    storage.store_episode = AsyncMock(return_value="ep_123")
    storage.store_structured = AsyncMock(return_value="struct_456")
    storage.update_episode = AsyncMock()
    storage.log_audit = AsyncMock(return_value="audit_789")
    storage.get_episode = AsyncMock(return_value=None)
    storage.delete_episode = AsyncMock(return_value={"deleted": True})
    storage.delete_structured = AsyncMock(return_value=True)
    return storage


def create_test_episode() -> Episode:
    """Create a test episode."""
    return Episode(
        id="ep_test",
        content="Test content",
        role="user",
        user_id="user_1",
        embedding=[0.1] * 1536,
    )


def create_test_structured() -> StructuredMemory:
    """Create a test structured memory."""
    return StructuredMemory(
        id="struct_test",
        source_episode_id="ep_test",
        user_id="user_1",
        embedding=[0.1] * 1536,
    )


def create_test_audit() -> AuditEntry:
    """Create a test audit entry."""
    return AuditEntry.for_encode(
        user_id="user_1",
        episode_id="ep_test",
        facts_count=3,
    )


class TestTrackedOperation:
    """Tests for TrackedOperation dataclass."""

    def test_create_store_operation(self) -> None:
        """Should create a tracked store operation."""
        op = TrackedOperation(
            operation=OperationType.STORE_EPISODE,
            entity_id="ep_123",
            user_id="user_1",
        )
        assert op.operation == OperationType.STORE_EPISODE
        assert op.entity_id == "ep_123"
        assert op.user_id == "user_1"
        assert op.original_data is None

    def test_create_update_operation_with_original(self) -> None:
        """Should create a tracked update operation with original data."""
        original = {"id": "ep_123", "content": "original"}
        op = TrackedOperation(
            operation=OperationType.UPDATE_EPISODE,
            entity_id="ep_123",
            user_id="user_1",
            original_data=original,
        )
        assert op.original_data == original


class TestTransactionContext:
    """Tests for TransactionContext."""

    @pytest.mark.asyncio
    async def test_store_episode_tracks_operation(self) -> None:
        """store_episode should track the operation for rollback."""
        storage = create_mock_storage()
        episode = create_test_episode()

        async with TransactionContext(storage=storage) as txn:
            await txn.store_episode(episode)
            assert len(txn.operations) == 1
            assert txn.operations[0].operation == OperationType.STORE_EPISODE
            txn.commit()

        storage.store_episode.assert_called_once_with(episode)

    @pytest.mark.asyncio
    async def test_store_structured_tracks_operation(self) -> None:
        """store_structured should track the operation for rollback."""
        storage = create_mock_storage()
        structured = create_test_structured()

        async with TransactionContext(storage=storage) as txn:
            await txn.store_structured(structured)
            assert len(txn.operations) == 1
            assert txn.operations[0].operation == OperationType.STORE_STRUCTURED
            txn.commit()

        storage.store_structured.assert_called_once_with(structured)

    @pytest.mark.asyncio
    async def test_update_episode_captures_original(self) -> None:
        """update_episode should capture original state for rollback."""
        storage = create_mock_storage()
        original_episode = Episode(
            id="ep_original",
            content="Test content",
            role="user",
            user_id="user_1",
            embedding=[0.1] * 1536,
        )
        storage.get_episode = AsyncMock(return_value=original_episode)

        episode = Episode(
            id="ep_original",
            content="updated content",
            role="user",
            user_id="user_1",
            embedding=[0.1] * 1536,
        )

        async with TransactionContext(storage=storage) as txn:
            await txn.update_episode(episode)
            assert len(txn.operations) == 1
            assert txn.operations[0].operation == OperationType.UPDATE_EPISODE
            assert txn.operations[0].original_data is not None
            txn.commit()

        storage.get_episode.assert_called_once_with("ep_original", episode.user_id)
        storage.update_episode.assert_called_once_with(episode)

    @pytest.mark.asyncio
    async def test_log_audit_tracks_operation(self) -> None:
        """log_audit should track the operation."""
        storage = create_mock_storage()
        audit = create_test_audit()

        async with TransactionContext(storage=storage) as txn:
            await txn.log_audit(audit)
            assert len(txn.operations) == 1
            assert txn.operations[0].operation == OperationType.LOG_AUDIT
            txn.commit()

        storage.log_audit.assert_called_once_with(audit)

    @pytest.mark.asyncio
    async def test_commit_clears_operations(self) -> None:
        """commit should clear tracked operations."""
        storage = create_mock_storage()
        episode = create_test_episode()

        txn = TransactionContext(storage=storage)
        await txn.store_episode(episode)
        assert len(txn.operations) == 1

        txn.commit()
        assert len(txn.operations) == 0
        assert txn.committed is True

    @pytest.mark.asyncio
    async def test_rollback_deletes_stored_entities(self) -> None:
        """rollback should delete stored episode and structured memory."""
        storage = create_mock_storage()
        episode = create_test_episode()
        structured = create_test_structured()

        txn = TransactionContext(storage=storage)
        await txn.store_episode(episode)
        await txn.store_structured(structured)

        rolled_back = await txn.rollback()

        assert rolled_back == 2
        storage.delete_episode.assert_called_once()
        storage.delete_structured.assert_called_once()
        assert len(txn.operations) == 0

    @pytest.mark.asyncio
    async def test_rollback_restores_updated_episode(self) -> None:
        """rollback should restore original episode data."""
        storage = create_mock_storage()
        original_episode = Episode(
            id="ep_restore",
            content="original content",
            role="user",
            user_id="user_1",
            embedding=[0.1] * 1536,
        )
        storage.get_episode = AsyncMock(return_value=original_episode)

        episode = Episode(
            id="ep_restore",
            content="updated content",
            role="user",
            user_id="user_1",
            embedding=[0.1] * 1536,
        )

        txn = TransactionContext(storage=storage)
        await txn.update_episode(episode)

        rolled_back = await txn.rollback()

        assert rolled_back == 1
        # update_episode should be called twice: once for the update, once for restore
        assert storage.update_episode.call_count == 2

    @pytest.mark.asyncio
    async def test_context_manager_rollback_on_exception(self) -> None:
        """Context manager should rollback on exception."""
        storage = create_mock_storage()
        episode = create_test_episode()

        with pytest.raises(ValueError, match="intentional"):
            async with TransactionContext(storage=storage) as txn:
                await txn.store_episode(episode)
                raise ValueError("intentional error")

        # Should have attempted to delete the stored episode
        storage.delete_episode.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_manager_no_rollback_on_commit(self) -> None:
        """Context manager should not rollback if committed."""
        storage = create_mock_storage()
        episode = create_test_episode()

        async with TransactionContext(storage=storage) as txn:
            await txn.store_episode(episode)
            txn.commit()

        # Should not have attempted to delete
        storage.delete_episode.assert_not_called()

    @pytest.mark.asyncio
    async def test_rollback_order_is_lifo(self) -> None:
        """Rollback should happen in reverse order (LIFO)."""
        storage = create_mock_storage()
        call_order: list[str] = []

        async def track_delete_episode(*args, **kwargs) -> dict[str, bool]:
            call_order.append("episode")
            return {"deleted": True}

        async def track_delete_structured(*args, **kwargs) -> bool:
            call_order.append("structured")
            return True

        storage.delete_episode = AsyncMock(side_effect=track_delete_episode)
        storage.delete_structured = AsyncMock(side_effect=track_delete_structured)

        episode = create_test_episode()
        structured = create_test_structured()

        txn = TransactionContext(storage=storage)
        await txn.store_episode(episode)  # First
        await txn.store_structured(structured)  # Second

        await txn.rollback()

        # Structured should be rolled back first (LIFO)
        assert call_order == ["structured", "episode"]

    @pytest.mark.asyncio
    async def test_rollback_continues_on_individual_failure(self) -> None:
        """Rollback should continue even if one operation fails."""
        storage = create_mock_storage()
        storage.delete_episode = AsyncMock(side_effect=Exception("delete failed"))
        storage.delete_structured = AsyncMock(return_value=True)

        episode = create_test_episode()
        structured = create_test_structured()

        txn = TransactionContext(storage=storage)
        await txn.store_episode(episode)
        await txn.store_structured(structured)

        # Should not raise, and should still try to delete structured
        rolled_back = await txn.rollback()

        # Only structured was successfully rolled back
        assert rolled_back == 1
        storage.delete_structured.assert_called_once()

    @pytest.mark.asyncio
    async def test_multiple_operations_tracked_correctly(self) -> None:
        """Multiple operations should all be tracked."""
        storage = create_mock_storage()
        original_episode = create_test_episode()
        storage.get_episode = AsyncMock(return_value=original_episode)

        episode = create_test_episode()
        structured = create_test_structured()
        audit = create_test_audit()

        async with TransactionContext(storage=storage) as txn:
            await txn.store_episode(episode)
            await txn.store_structured(structured)
            await txn.update_episode(episode)
            await txn.log_audit(audit)

            assert len(txn.operations) == 4
            assert txn.operations[0].operation == OperationType.STORE_EPISODE
            assert txn.operations[1].operation == OperationType.STORE_STRUCTURED
            assert txn.operations[2].operation == OperationType.UPDATE_EPISODE
            assert txn.operations[3].operation == OperationType.LOG_AUDIT

            txn.commit()


class TestTransactionMixin:
    """Tests for TransactionMixin on EngramStorage."""

    def test_storage_has_transaction_method(self) -> None:
        """EngramStorage should have transaction() method."""
        from engram.storage import EngramStorage

        storage = EngramStorage()
        assert hasattr(storage, "transaction")
        assert callable(storage.transaction)
