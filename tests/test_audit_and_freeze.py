"""Tests for Episode freeze (BUG-6), mutation_rejected audit (BUG-4), and rollback marking (BUG-3).

These tests verify three related fixes from the code review:
- BUG-6: Episode model is frozen (immutable at Python level)
- BUG-4: Immutable field rejection creates a forensic audit entry
- BUG-3: Transaction rollback marks audit entries as rolled_back
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from engram.models import AuditEntry, Episode
from engram.storage.transaction import OperationType, TransactionContext

# ---------------------------------------------------------------------------
# BUG-6: Episode model is frozen
# ---------------------------------------------------------------------------


class TestEpisodeFrozen:
    """Episode should be immutable after creation."""

    def test_episode_cannot_mutate_content(self) -> None:
        """Direct mutation of content raises ValidationError."""
        ep = Episode(
            content="original",
            role="user",
            user_id="u1",
            embedding=[0.1] * 3,
        )
        with pytest.raises(ValidationError, match="frozen"):
            ep.content = "modified"  # type: ignore[misc]

    def test_episode_cannot_mutate_role(self) -> None:
        """Direct mutation of role raises ValidationError."""
        ep = Episode(
            content="text",
            role="user",
            user_id="u1",
            embedding=[0.1] * 3,
        )
        with pytest.raises(ValidationError, match="frozen"):
            ep.role = "assistant"  # type: ignore[misc]

    def test_episode_cannot_mutate_timestamp(self) -> None:
        """Direct mutation of timestamp raises ValidationError."""
        from datetime import UTC, datetime

        ep = Episode(
            content="text",
            role="user",
            user_id="u1",
            embedding=[0.1] * 3,
        )
        with pytest.raises(ValidationError, match="frozen"):
            ep.timestamp = datetime.now(UTC)  # type: ignore[misc]

    def test_episode_cannot_mutate_importance(self) -> None:
        """Direct mutation of importance raises ValidationError."""
        ep = Episode(
            content="text",
            role="user",
            user_id="u1",
            embedding=[0.1] * 3,
        )
        with pytest.raises(ValidationError, match="frozen"):
            ep.importance = 0.9  # type: ignore[misc]

    def test_episode_cannot_mutate_metadata_fields(self) -> None:
        """Direct mutation of consolidated/summarized/structured raises ValidationError."""
        ep = Episode(
            content="text",
            role="user",
            user_id="u1",
            embedding=[0.1] * 3,
        )
        with pytest.raises(ValidationError, match="frozen"):
            ep.consolidated = True  # type: ignore[misc]
        with pytest.raises(ValidationError, match="frozen"):
            ep.summarized = True  # type: ignore[misc]
        with pytest.raises(ValidationError, match="frozen"):
            ep.structured = True  # type: ignore[misc]

    def test_model_copy_creates_modified_copy(self) -> None:
        """model_copy(update=...) is the approved way to create modified Episodes."""
        ep = Episode(
            content="text",
            role="user",
            user_id="u1",
            embedding=[0.1] * 3,
            importance=0.5,
        )
        updated = ep.model_copy(update={"importance": 0.9, "consolidated": True})

        # Original unchanged
        assert ep.importance == 0.5
        assert ep.consolidated is False

        # Copy has updates
        assert updated.importance == 0.9
        assert updated.consolidated is True
        assert updated.content == "text"  # unchanged fields preserved

    def test_episode_extra_forbid(self) -> None:
        """Episode rejects unknown fields."""
        with pytest.raises(ValidationError, match="extra"):
            Episode(
                content="text",
                role="user",
                user_id="u1",
                embedding=[0.1] * 3,
                unknown_field="value",  # type: ignore[call-arg]
            )


# ---------------------------------------------------------------------------
# BUG-4: mutation_rejected audit entry
# ---------------------------------------------------------------------------


class TestMutationRejectedAudit:
    """AuditEntry.for_mutation_rejected creates proper forensic records."""

    def test_for_mutation_rejected_creates_entry(self) -> None:
        """Factory creates a valid mutation_rejected audit entry."""
        entry = AuditEntry.for_mutation_rejected(
            user_id="u1",
            memory_id="ep_123",
            memory_type="episodic",
            field="content",
            stored_value="original text",
            attempted_value="tampered text",
            org_id="org_1",
        )

        assert entry.event == "mutation_rejected"
        assert entry.user_id == "u1"
        assert entry.org_id == "org_1"
        assert entry.details["memory_id"] == "ep_123"
        assert entry.details["memory_type"] == "episodic"
        assert entry.details["field"] == "content"
        assert entry.details["stored_value"] == "original text"
        assert entry.details["attempted_value"] == "tampered text"

    def test_for_mutation_rejected_truncates_long_values(self) -> None:
        """Stored and attempted values are truncated to 200 chars for safety."""
        long_value = "x" * 500
        entry = AuditEntry.for_mutation_rejected(
            user_id="u1",
            memory_id="ep_123",
            memory_type="episodic",
            field="content",
            stored_value=long_value,
            attempted_value=long_value,
        )

        assert len(entry.details["stored_value"]) == 200
        assert len(entry.details["attempted_value"]) == 200

    def test_for_mutation_rejected_org_id_optional(self) -> None:
        """org_id defaults to None."""
        entry = AuditEntry.for_mutation_rejected(
            user_id="u1",
            memory_id="ep_123",
            memory_type="episodic",
            field="content",
            stored_value="a",
            attempted_value="b",
        )
        assert entry.org_id is None

    def test_mutation_rejected_is_valid_event_type(self) -> None:
        """mutation_rejected is a valid AuditEventType."""
        entry = AuditEntry(
            event="mutation_rejected",
            user_id="u1",
            details={},
        )
        assert entry.event == "mutation_rejected"


# ---------------------------------------------------------------------------
# BUG-3: Transaction rollback marks audit entries as rolled_back
# ---------------------------------------------------------------------------


class TestAuditRolledBack:
    """Audit entries get rolled_back=True when their transaction rolls back."""

    def test_audit_entry_default_rolled_back_false(self) -> None:
        """New audit entries default to rolled_back=False."""
        entry = AuditEntry.for_encode(
            user_id="u1",
            episode_id="ep_1",
            facts_count=3,
        )
        assert entry.rolled_back is False

    def test_audit_entry_rolled_back_serializes(self) -> None:
        """rolled_back field survives serialization round-trip."""
        entry = AuditEntry.for_encode(
            user_id="u1",
            episode_id="ep_1",
            facts_count=3,
        )
        rolled_back_entry = entry.model_copy(update={"rolled_back": True})
        dumped = rolled_back_entry.model_dump(mode="json")
        restored = AuditEntry.model_validate(dumped)
        assert restored.rolled_back is True

    @pytest.mark.asyncio
    async def test_log_audit_stores_original_data_for_rollback(self) -> None:
        """log_audit should store the entry data so rollback can mark it."""
        storage = MagicMock()
        storage.log_audit = AsyncMock(return_value="audit_1")

        audit = AuditEntry.for_encode(
            user_id="u1",
            episode_id="ep_1",
            facts_count=3,
        )

        txn = TransactionContext(storage=storage)
        await txn.log_audit(audit)

        assert len(txn.operations) == 1
        op = txn.operations[0]
        assert op.operation == OperationType.LOG_AUDIT
        assert op.original_data is not None
        assert op.original_data["event"] == "encode"
        assert op.original_data["rolled_back"] is False

    @pytest.mark.asyncio
    async def test_rollback_marks_audit_entry_as_rolled_back(self) -> None:
        """Rolling back a transaction should re-upsert audit with rolled_back=True."""
        storage = AsyncMock()
        storage.store_episode = AsyncMock(return_value="ep_1")
        storage.log_audit = AsyncMock(return_value="audit_1")
        storage.delete_episode = AsyncMock(return_value={"deleted": True})

        audit = AuditEntry.for_encode(
            user_id="u1",
            episode_id="ep_1",
            facts_count=3,
        )

        episode = Episode(
            content="text",
            role="user",
            user_id="u1",
            embedding=[0.1] * 3,
        )

        txn = TransactionContext(storage=storage)
        await txn.store_episode(episode)
        await txn.log_audit(audit)

        rolled_back = await txn.rollback()

        # Both episode delete and audit mark count as rolled back
        assert rolled_back == 2

        # log_audit called twice: once for original, once for rollback marking
        assert storage.log_audit.call_count == 2

        # Second call should have rolled_back=True
        rollback_call = storage.log_audit.call_args_list[1]
        marked_entry = rollback_call[0][0]
        assert isinstance(marked_entry, AuditEntry)
        assert marked_entry.rolled_back is True
        assert marked_entry.event == "encode"
        assert marked_entry.id == audit.id  # Same ID = upsert overwrites

    @pytest.mark.asyncio
    async def test_rollback_with_exception_marks_audit(self) -> None:
        """Context manager rollback on exception should mark audit entries."""
        storage = AsyncMock()
        storage.store_episode = AsyncMock(return_value="ep_1")
        storage.log_audit = AsyncMock(return_value="audit_1")
        storage.delete_episode = AsyncMock(return_value={"deleted": True})

        audit = AuditEntry.for_encode(
            user_id="u1",
            episode_id="ep_1",
            facts_count=3,
        )

        episode = Episode(
            content="text",
            role="user",
            user_id="u1",
            embedding=[0.1] * 3,
        )

        with pytest.raises(RuntimeError, match="boom"):
            async with TransactionContext(storage=storage) as txn:
                await txn.store_episode(episode)
                await txn.log_audit(audit)
                raise RuntimeError("boom")

        # Audit should have been re-upserted with rolled_back=True
        assert storage.log_audit.call_count == 2
        marked_entry = storage.log_audit.call_args_list[1][0][0]
        assert marked_entry.rolled_back is True

    @pytest.mark.asyncio
    async def test_rollback_audit_failure_does_not_stop_rollback(self) -> None:
        """If marking audit as rolled_back fails, other rollbacks still proceed."""
        storage = AsyncMock()
        storage.store_episode = AsyncMock(return_value="ep_1")
        # First call succeeds (original log), second fails (rollback mark)
        storage.log_audit = AsyncMock(
            side_effect=[
                "audit_1",  # original log_audit
                Exception("storage down"),  # rollback mark attempt
            ]
        )
        storage.delete_episode = AsyncMock(return_value={"deleted": True})

        audit = AuditEntry.for_encode(
            user_id="u1",
            episode_id="ep_1",
            facts_count=3,
        )
        episode = Episode(
            content="text",
            role="user",
            user_id="u1",
            embedding=[0.1] * 3,
        )

        txn = TransactionContext(storage=storage)
        await txn.store_episode(episode)
        await txn.log_audit(audit)

        # Rollback should not raise even though audit marking fails
        rolled_back = await txn.rollback()

        # Episode deletion succeeded, audit marking failed
        assert rolled_back == 1
        storage.delete_episode.assert_called_once()
