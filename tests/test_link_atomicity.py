"""Tests for bidirectional link atomicity (SPEC-004).

Verifies that link_memories() rolls back the forward link when the reverse
link fails, and that unlink_memories() restores the forward link when the
reverse unlink fails — maintaining graph symmetry.
"""

from __future__ import annotations

import logging
from types import MethodType
from unittest.mock import AsyncMock, MagicMock

import pytest

from engram.models import ConfidenceScore, SemanticMemory
from engram.storage.linking import LinkingMixin


def _make_semantic(memory_id: str) -> SemanticMemory:
    """Create a test semantic memory."""
    return SemanticMemory(
        id=memory_id,
        content="test",
        source_episode_ids=["ep_1"],
        user_id="user_1",
        confidence=ConfidenceScore.for_inferred(),
    )


def _make_storage(
    source: SemanticMemory,
    target: SemanticMemory,
    update_side_effects: list[bool] | None = None,
) -> MagicMock:
    """Create a mock storage with LinkingMixin methods bound.

    Args:
        source: Source memory returned by get_semantic for source_id.
        target: Target memory returned by get_semantic for target_id.
        update_side_effects: Sequence of True/False for update_semantic_memory calls.
            If None, all updates succeed.
    """
    storage = MagicMock()

    async def get_semantic_impl(memory_id: str, user_id: str) -> SemanticMemory | None:
        if memory_id == source.id:
            return source
        if memory_id == target.id:
            return target
        return None

    storage.get_semantic = AsyncMock(side_effect=get_semantic_impl)
    storage.get_procedural = AsyncMock(return_value=None)

    if update_side_effects is not None:
        storage.update_semantic_memory = AsyncMock(side_effect=update_side_effects)
    else:
        storage.update_semantic_memory = AsyncMock(return_value=True)
    storage.update_procedural_memory = AsyncMock(return_value=True)

    # Bind LinkingMixin methods
    storage._detect_memory_type = MethodType(LinkingMixin._detect_memory_type, storage)
    storage._get_linkable_memory = MethodType(LinkingMixin._get_linkable_memory, storage)
    storage._update_linkable_memory = MethodType(LinkingMixin._update_linkable_memory, storage)
    storage.link_memories = MethodType(LinkingMixin.link_memories, storage)
    storage.unlink_memories = MethodType(LinkingMixin.unlink_memories, storage)

    return storage


class TestLinkRollbackOnReverseFailure:
    """Tests for link_memories rolling back when reverse link fails."""

    @pytest.mark.asyncio
    async def test_rollback_on_reverse_failure(self) -> None:
        """Forward link should be rolled back when reverse link fails."""
        source = _make_semantic("sem_src")
        target = _make_semantic("sem_tgt")

        # Call sequence: forward=True, reverse-attempt=False, reverse-retry=False, rollback=True
        storage = _make_storage(source, target, update_side_effects=[True, False, False, True])

        result = await storage.link_memories("sem_src", "sem_tgt", "user_1")

        assert result.success is False
        assert result.source_updated is False  # Rollback succeeded
        assert result.target_updated is False
        assert "rollback succeeded" in (result.error or "")

        # Source should NOT have target in related_ids after rollback
        assert "sem_tgt" not in source.related_ids

    @pytest.mark.asyncio
    async def test_retry_succeeds_before_rollback(self) -> None:
        """If the retry of the reverse link succeeds, no rollback needed."""
        source = _make_semantic("sem_src")
        target = _make_semantic("sem_tgt")

        # forward=True, reverse-attempt=False, reverse-retry=True
        storage = _make_storage(source, target, update_side_effects=[True, False, True])

        result = await storage.link_memories("sem_src", "sem_tgt", "user_1")

        assert result.success is True
        assert result.source_updated is True
        assert result.target_updated is True

        # Both should have links
        assert "sem_tgt" in source.related_ids
        assert "sem_src" in target.related_ids

    @pytest.mark.asyncio
    async def test_rollback_failure_logs_critical(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """If rollback also fails, should log CRITICAL for manual cleanup."""
        source = _make_semantic("sem_src")
        target = _make_semantic("sem_tgt")

        # forward=True, reverse=False, retry=False, rollback=False
        storage = _make_storage(source, target, update_side_effects=[True, False, False, False])

        with caplog.at_level(logging.CRITICAL, logger="engram.storage.linking"):
            result = await storage.link_memories("sem_src", "sem_tgt", "user_1")

        assert result.success is False
        assert result.source_updated is True  # Rollback failed, source still has link
        assert "rollback FAILED" in (result.error or "")

        assert any("GRAPH INCONSISTENCY" in rec.message for rec in caplog.records)

    @pytest.mark.asyncio
    async def test_non_bidirectional_no_rollback(self) -> None:
        """Non-bidirectional link should not trigger rollback logic."""
        source = _make_semantic("sem_src")
        target = _make_semantic("sem_tgt")

        # Only one update call (forward link)
        storage = _make_storage(source, target, update_side_effects=[True])

        result = await storage.link_memories("sem_src", "sem_tgt", "user_1", bidirectional=False)

        assert result.success is True
        assert result.source_updated is True
        assert "sem_tgt" in source.related_ids
        # Target should NOT have a link back
        assert "sem_src" not in target.related_ids


class TestUnlinkRollbackOnReverseFailure:
    """Tests for unlink_memories rolling back when reverse unlink fails."""

    @pytest.mark.asyncio
    async def test_rollback_on_reverse_unlink_failure(self) -> None:
        """Forward unlink should be rolled back when reverse unlink fails."""
        source = _make_semantic("sem_src")
        target = _make_semantic("sem_tgt")

        # Pre-link both sides
        source.add_link("sem_tgt", "related")
        target.add_link("sem_src", "related")

        # forward-remove=True, reverse-remove=False, rollback-re-add=True
        storage = _make_storage(source, target, update_side_effects=[True, False, True])

        result = await storage.unlink_memories("sem_src", "sem_tgt", "user_1")

        assert result.success is False
        assert "rollback succeeded" in (result.error or "")

        # Source should still have the link (rollback restored it)
        assert "sem_tgt" in source.related_ids

    @pytest.mark.asyncio
    async def test_no_rollback_when_forward_unlink_failed(self) -> None:
        """If forward unlink didn't persist, no rollback is needed for symmetry."""
        source = _make_semantic("sem_src")
        target = _make_semantic("sem_tgt")

        # Pre-link both sides
        source.add_link("sem_tgt", "related")
        target.add_link("sem_src", "related")

        # forward-remove=False, reverse-remove=False
        storage = _make_storage(source, target, update_side_effects=[False, False])

        result = await storage.unlink_memories("sem_src", "sem_tgt", "user_1")

        # Neither side changed successfully — no rollback needed
        assert result.success is False

    @pytest.mark.asyncio
    async def test_successful_bidirectional_unlink(self) -> None:
        """Both sides should be unlinked when everything succeeds."""
        source = _make_semantic("sem_src")
        target = _make_semantic("sem_tgt")

        source.add_link("sem_tgt", "related")
        target.add_link("sem_src", "related")

        storage = _make_storage(source, target, update_side_effects=[True, True])

        result = await storage.unlink_memories("sem_src", "sem_tgt", "user_1")

        assert result.success is True
        assert result.source_updated is True
        assert result.target_updated is True
        assert "sem_tgt" not in source.related_ids
        assert "sem_src" not in target.related_ids
