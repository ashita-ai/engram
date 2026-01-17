"""Unit tests for Engram memory history tracking system."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from engram.models import HistoryEntry


class TestHistoryEntry:
    """Tests for the HistoryEntry model."""

    def test_generates_unique_ids(self):
        """Each HistoryEntry should have a unique ID."""
        entries = [
            HistoryEntry(
                memory_id="sem_123",
                memory_type="semantic",
                user_id="user_1",
                change_type="created",
                trigger="encode",
            )
            for _ in range(10)
        ]
        ids = [e.id for e in entries]
        assert len(ids) == len(set(ids))

    def test_id_prefix(self):
        """HistoryEntry ID should start with hist_."""
        entry = HistoryEntry(
            memory_id="sem_123",
            memory_type="semantic",
            user_id="user_1",
            change_type="created",
            trigger="encode",
        )
        assert entry.id.startswith("hist_")

    def test_default_timestamp(self):
        """HistoryEntry should have a default timestamp."""
        before = datetime.now(UTC)
        entry = HistoryEntry(
            memory_id="sem_123",
            memory_type="semantic",
            user_id="user_1",
            change_type="created",
            trigger="encode",
        )
        after = datetime.now(UTC)
        assert before <= entry.timestamp <= after

    def test_required_fields(self):
        """HistoryEntry should require memory_id, memory_type, user_id, change_type, trigger."""
        with pytest.raises(ValidationError):
            HistoryEntry()  # type: ignore[call-arg]

    def test_extra_fields_forbidden(self):
        """HistoryEntry should reject unknown fields."""
        with pytest.raises(ValidationError):
            HistoryEntry(
                memory_id="sem_123",
                memory_type="semantic",
                user_id="user_1",
                change_type="created",
                trigger="encode",
                unknown_field="value",  # type: ignore[call-arg]
            )


class TestHistoryEntryFactories:
    """Tests for HistoryEntry factory methods."""

    def test_for_create(self):
        """for_create should set change_type=created and capture after_state."""
        after_state = {"id": "sem_123", "content": "User likes Python", "confidence": 0.8}

        entry = HistoryEntry.for_create(
            memory_id="sem_123",
            memory_type="semantic",
            user_id="user_1",
            trigger="encode",
            after_state=after_state,
            reason="Created during encode",
        )

        assert entry.change_type == "created"
        assert entry.trigger == "encode"
        assert entry.before is None
        assert entry.after == after_state
        assert entry.diff == after_state  # Everything is new
        assert entry.reason == "Created during encode"

    def test_for_update(self):
        """for_update should compute diff from before/after states."""
        before = {"id": "sem_123", "content": "Old content", "confidence": 0.7}
        after = {"id": "sem_123", "content": "New content", "confidence": 0.8}

        entry = HistoryEntry.for_update(
            memory_id="sem_123",
            memory_type="semantic",
            user_id="user_1",
            trigger="manual",
            before_state=before,
            after_state=after,
        )

        assert entry.change_type == "updated"
        assert entry.trigger == "manual"
        assert entry.before == before
        assert entry.after == after
        # Diff should capture changed fields
        assert "content" in entry.diff
        assert entry.diff["content"]["old"] == "Old content"
        assert entry.diff["content"]["new"] == "New content"
        assert "confidence" in entry.diff
        # id should not be in diff (unchanged)
        assert "id" not in entry.diff

    def test_for_strengthen(self):
        """for_strengthen should track consolidation strength changes."""
        entry = HistoryEntry.for_strengthen(
            memory_id="sem_123",
            memory_type="semantic",
            user_id="user_1",
            trigger="consolidation",
            old_strength=0.5,
            new_strength=0.6,
            old_passes=2,
            new_passes=3,
        )

        assert entry.change_type == "strengthened"
        assert entry.trigger == "consolidation"
        assert entry.diff["consolidation_strength"]["old"] == 0.5
        assert entry.diff["consolidation_strength"]["new"] == 0.6
        assert entry.diff["consolidation_passes"]["old"] == 2
        assert entry.diff["consolidation_passes"]["new"] == 3

    def test_for_weaken(self):
        """for_weaken should track strength decreases from decay."""
        entry = HistoryEntry.for_weaken(
            memory_id="sem_123",
            memory_type="semantic",
            user_id="user_1",
            trigger="decay",
            old_strength=0.8,
            new_strength=0.7,
        )

        assert entry.change_type == "weakened"
        assert entry.trigger == "decay"
        assert entry.diff["consolidation_strength"]["old"] == 0.8
        assert entry.diff["consolidation_strength"]["new"] == 0.7

    def test_for_archive(self):
        """for_archive should capture final state before archival."""
        before_state = {"id": "sem_123", "content": "Low confidence memory", "confidence": 0.3}

        entry = HistoryEntry.for_archive(
            memory_id="sem_123",
            memory_type="semantic",
            user_id="user_1",
            trigger="decay",
            before_state=before_state,
        )

        assert entry.change_type == "archived"
        assert entry.trigger == "decay"
        assert entry.before == before_state
        assert entry.after is None
        assert "low confidence" in entry.reason.lower()

    def test_for_delete(self):
        """for_delete should capture final state before deletion."""
        before_state = {"id": "sem_123", "content": "Deleted memory"}

        entry = HistoryEntry.for_delete(
            memory_id="sem_123",
            memory_type="semantic",
            user_id="user_1",
            trigger="manual",
            before_state=before_state,
            reason="User requested deletion",
        )

        assert entry.change_type == "deleted"
        assert entry.trigger == "manual"
        assert entry.before == before_state
        assert entry.after is None
        assert entry.reason == "User requested deletion"

    def test_for_retrieval(self):
        """for_retrieval should track retrieval count changes (Testing Effect)."""
        entry = HistoryEntry.for_retrieval(
            memory_id="sem_123",
            memory_type="semantic",
            user_id="user_1",
            old_retrieval_count=5,
            new_retrieval_count=6,
        )

        assert entry.change_type == "updated"
        assert entry.trigger == "retrieval"
        assert entry.diff["retrieval_count"]["old"] == 5
        assert entry.diff["retrieval_count"]["new"] == 6
        assert "Testing Effect" in entry.reason


class TestHistoryEntryWithOrgId:
    """Tests for HistoryEntry with organization isolation."""

    def test_create_with_org_id(self):
        """HistoryEntry should accept org_id."""
        entry = HistoryEntry.for_create(
            memory_id="sem_123",
            memory_type="semantic",
            user_id="user_1",
            trigger="encode",
            after_state={"content": "test"},
            org_id="org_456",
        )

        assert entry.org_id == "org_456"

    def test_default_org_id_is_none(self):
        """org_id should default to None."""
        entry = HistoryEntry(
            memory_id="sem_123",
            memory_type="semantic",
            user_id="user_1",
            change_type="created",
            trigger="encode",
        )
        assert entry.org_id is None


class TestHistoryEntryStringRepresentation:
    """Tests for HistoryEntry string representation."""

    def test_str_representation(self):
        """__str__ should show change type, memory type, and memory id."""
        entry = HistoryEntry(
            memory_id="sem_123",
            memory_type="semantic",
            user_id="user_1",
            change_type="updated",
            trigger="manual",
        )
        s = str(entry)
        assert "updated" in s
        assert "semantic" in s
        assert "sem_123" in s


class TestChangeTypeAndTriggerType:
    """Tests for ChangeType and TriggerType literals."""

    def test_valid_change_types(self):
        """All valid change types should be accepted."""
        valid_types = ["created", "updated", "strengthened", "weakened", "archived", "deleted"]
        for ct in valid_types:
            entry = HistoryEntry(
                memory_id="sem_123",
                memory_type="semantic",
                user_id="user_1",
                change_type=ct,  # type: ignore[arg-type]
                trigger="manual",
            )
            assert entry.change_type == ct

    def test_invalid_change_type(self):
        """Invalid change types should be rejected."""
        with pytest.raises(ValidationError):
            HistoryEntry(
                memory_id="sem_123",
                memory_type="semantic",
                user_id="user_1",
                change_type="invalid",  # type: ignore[arg-type]
                trigger="manual",
            )

    def test_valid_trigger_types(self):
        """All valid trigger types should be accepted."""
        valid_triggers = [
            "encode",
            "consolidation",
            "decay",
            "promotion",
            "manual",
            "retrieval",
            "system",
        ]
        for tt in valid_triggers:
            entry = HistoryEntry(
                memory_id="sem_123",
                memory_type="semantic",
                user_id="user_1",
                change_type="created",
                trigger=tt,  # type: ignore[arg-type]
            )
            assert entry.trigger == tt

    def test_invalid_trigger_type(self):
        """Invalid trigger types should be rejected."""
        with pytest.raises(ValidationError):
            HistoryEntry(
                memory_id="sem_123",
                memory_type="semantic",
                user_id="user_1",
                change_type="created",
                trigger="invalid",  # type: ignore[arg-type]
            )
