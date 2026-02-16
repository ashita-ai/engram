"""Tests for storage linking operations.

Tests the LinkingMixin which provides atomic bidirectional linking at the
storage layer to ensure consistency between related memories.
"""

from __future__ import annotations

from types import MethodType
from unittest.mock import AsyncMock, MagicMock

import pytest

from engram.models import ConfidenceScore, SemanticMemory
from engram.storage.linking import LinkingMixin, LinkResult


def create_test_semantic(memory_id: str = "sem_test") -> SemanticMemory:
    """Create a test semantic memory."""
    return SemanticMemory(
        id=memory_id,
        content="Test semantic memory",
        source_episode_ids=["ep_1"],
        user_id="user_1",
        confidence=ConfidenceScore.for_inferred(),
        embedding=[0.1] * 1536,
    )


def create_mock_storage() -> MagicMock:
    """Create a mock storage with all mixin methods properly bound."""
    storage = MagicMock()

    # Set up base async methods
    storage.get_semantic = AsyncMock(return_value=None)
    storage.get_procedural = AsyncMock(return_value=None)
    storage.update_semantic_memory = AsyncMock(return_value=True)
    storage.update_procedural_memory = AsyncMock(return_value=True)

    # Bind all LinkingMixin methods to the mock as proper bound methods
    storage._detect_memory_type = MethodType(LinkingMixin._detect_memory_type, storage)
    storage._get_linkable_memory = MethodType(LinkingMixin._get_linkable_memory, storage)
    storage._update_linkable_memory = MethodType(LinkingMixin._update_linkable_memory, storage)
    storage.link_memories = MethodType(LinkingMixin.link_memories, storage)
    storage.unlink_memories = MethodType(LinkingMixin.unlink_memories, storage)

    return storage


class TestDetectMemoryType:
    """Tests for memory type detection from ID prefix."""

    def test_detects_semantic_from_prefix(self) -> None:
        """Should detect semantic type from sem_ prefix."""
        mixin = LinkingMixin()
        assert mixin._detect_memory_type("sem_abc123") == "semantic"

    def test_detects_procedural_from_prefix(self) -> None:
        """Should detect procedural type from proc_ prefix."""
        mixin = LinkingMixin()
        assert mixin._detect_memory_type("proc_abc123") == "procedural"

    def test_returns_none_for_unknown_prefix(self) -> None:
        """Should return None for unknown prefix."""
        mixin = LinkingMixin()
        assert mixin._detect_memory_type("ep_abc123") is None
        assert mixin._detect_memory_type("struct_abc123") is None
        assert mixin._detect_memory_type("unknown") is None


class TestLinkMemories:
    """Tests for bidirectional link creation."""

    @pytest.fixture
    def mock_storage(self) -> MagicMock:
        """Create a properly configured mock storage."""
        return create_mock_storage()

    @pytest.mark.asyncio
    async def test_link_semantic_to_semantic(self, mock_storage: MagicMock) -> None:
        """Should create bidirectional link between two semantic memories."""
        source = create_test_semantic("sem_source")
        target = create_test_semantic("sem_target")

        mock_storage.get_semantic = AsyncMock(
            side_effect=lambda id, _: source if id == "sem_source" else target
        )
        mock_storage.update_semantic_memory = AsyncMock(return_value=True)

        result = await mock_storage.link_memories(
            source_id="sem_source",
            target_id="sem_target",
            user_id="user_1",
            link_type="related",
        )

        assert result.success is True
        assert result.source_updated is True
        assert result.target_updated is True
        assert "sem_target" in source.related_ids
        assert "sem_source" in target.related_ids

    @pytest.mark.asyncio
    async def test_link_with_unidirectional(self, mock_storage: MagicMock) -> None:
        """Should create only forward link when bidirectional=False."""
        source = create_test_semantic("sem_source")
        target = create_test_semantic("sem_target")

        mock_storage.get_semantic = AsyncMock(
            side_effect=lambda id, _: source if id == "sem_source" else target
        )
        mock_storage.update_semantic_memory = AsyncMock(return_value=True)

        result = await mock_storage.link_memories(
            source_id="sem_source",
            target_id="sem_target",
            user_id="user_1",
            link_type="related",
            bidirectional=False,
        )

        assert result.success is True
        assert result.source_updated is True
        assert result.target_updated is False
        assert "sem_target" in source.related_ids
        assert "sem_source" not in target.related_ids

    @pytest.mark.asyncio
    async def test_link_preserves_link_type(self, mock_storage: MagicMock) -> None:
        """Should preserve the specified link type."""
        source = create_test_semantic("sem_source")
        target = create_test_semantic("sem_target")

        mock_storage.get_semantic = AsyncMock(
            side_effect=lambda id, _: source if id == "sem_source" else target
        )
        mock_storage.update_semantic_memory = AsyncMock(return_value=True)

        await mock_storage.link_memories(
            source_id="sem_source",
            target_id="sem_target",
            user_id="user_1",
            link_type="supersedes",
        )

        assert source.link_types["sem_target"] == "supersedes"
        assert target.link_types["sem_source"] == "supersedes"

    @pytest.mark.asyncio
    async def test_link_fails_for_missing_source(self, mock_storage: MagicMock) -> None:
        """Should fail gracefully when source memory not found."""
        mock_storage.get_semantic = AsyncMock(return_value=None)

        result = await mock_storage.link_memories(
            source_id="sem_missing",
            target_id="sem_target",
            user_id="user_1",
        )

        assert result.success is False
        assert result.source_updated is False
        assert "Source memory not found" in str(result.error)

    @pytest.mark.asyncio
    async def test_link_fails_for_missing_target(self, mock_storage: MagicMock) -> None:
        """Should fail gracefully when target memory not found."""
        source = create_test_semantic("sem_source")

        mock_storage.get_semantic = AsyncMock(
            side_effect=lambda id, _: source if id == "sem_source" else None
        )

        result = await mock_storage.link_memories(
            source_id="sem_source",
            target_id="sem_missing",
            user_id="user_1",
        )

        assert result.success is False
        assert result.source_updated is False
        assert "Target memory not found" in str(result.error)

    @pytest.mark.asyncio
    async def test_link_fails_for_unknown_source_type(self, mock_storage: MagicMock) -> None:
        """Should fail gracefully when source type cannot be determined."""
        result = await mock_storage.link_memories(
            source_id="unknown_prefix",
            target_id="sem_target",
            user_id="user_1",
        )

        assert result.success is False
        assert "Cannot determine memory type for source" in str(result.error)

    @pytest.mark.asyncio
    async def test_link_partial_failure_reported(self, mock_storage: MagicMock) -> None:
        """Should rollback forward link when reverse link fails."""
        source = create_test_semantic("sem_source")
        target = create_test_semantic("sem_target")

        mock_storage.get_semantic = AsyncMock(
            side_effect=lambda id, _: source if id == "sem_source" else target
        )
        # Source update succeeds, target update fails, rollback succeeds
        # Calls: forward(source)=True, reverse(target)=False, retry(target)=False,
        #        rollback(source)=True
        mock_storage.update_semantic_memory = AsyncMock(
            side_effect=lambda mem: mem.id == "sem_source"
        )

        result = await mock_storage.link_memories(
            source_id="sem_source",
            target_id="sem_target",
            user_id="user_1",
        )

        assert result.success is False
        assert result.source_updated is False  # Rollback succeeded
        assert result.target_updated is False
        assert "rollback succeeded" in str(result.error)


class TestUnlinkMemories:
    """Tests for bidirectional link removal."""

    @pytest.fixture
    def mock_storage(self) -> MagicMock:
        """Create a properly configured mock storage."""
        return create_mock_storage()

    @pytest.mark.asyncio
    async def test_unlink_removes_both_directions(self, mock_storage: MagicMock) -> None:
        """Should remove links in both directions."""
        source = create_test_semantic("sem_source")
        target = create_test_semantic("sem_target")
        # Pre-link them
        source.add_link("sem_target", "related")
        target.add_link("sem_source", "related")

        mock_storage.get_semantic = AsyncMock(
            side_effect=lambda id, _: source if id == "sem_source" else target
        )
        mock_storage.update_semantic_memory = AsyncMock(return_value=True)

        result = await mock_storage.unlink_memories(
            source_id="sem_source",
            target_id="sem_target",
            user_id="user_1",
        )

        assert result.success is True
        assert result.source_updated is True
        assert result.target_updated is True
        assert "sem_target" not in source.related_ids
        assert "sem_source" not in target.related_ids

    @pytest.mark.asyncio
    async def test_unlink_with_unidirectional(self, mock_storage: MagicMock) -> None:
        """Should only remove forward link when bidirectional=False."""
        source = create_test_semantic("sem_source")
        target = create_test_semantic("sem_target")
        source.add_link("sem_target", "related")
        target.add_link("sem_source", "related")

        mock_storage.get_semantic = AsyncMock(
            side_effect=lambda id, _: source if id == "sem_source" else target
        )
        mock_storage.update_semantic_memory = AsyncMock(return_value=True)

        result = await mock_storage.unlink_memories(
            source_id="sem_source",
            target_id="sem_target",
            user_id="user_1",
            bidirectional=False,
        )

        assert result.success is True
        assert result.source_updated is True
        assert result.target_updated is False
        assert "sem_target" not in source.related_ids
        assert "sem_source" in target.related_ids  # Should still exist

    @pytest.mark.asyncio
    async def test_unlink_nonexistent_link_reports_no_update(self, mock_storage: MagicMock) -> None:
        """Should report no updates when links don't exist."""
        source = create_test_semantic("sem_source")
        target = create_test_semantic("sem_target")
        # No pre-existing links

        mock_storage.get_semantic = AsyncMock(
            side_effect=lambda id, _: source if id == "sem_source" else target
        )
        mock_storage.update_semantic_memory = AsyncMock(return_value=True)

        result = await mock_storage.unlink_memories(
            source_id="sem_source",
            target_id="sem_target",
            user_id="user_1",
        )

        # No links existed, so nothing was updated
        assert result.source_updated is False
        assert result.target_updated is False


class TestLinkResult:
    """Tests for LinkResult dataclass."""

    def test_link_result_success(self) -> None:
        """Should create successful LinkResult."""
        result = LinkResult(
            success=True,
            source_updated=True,
            target_updated=True,
            source_id="sem_1",
            target_id="sem_2",
            link_type="related",
        )
        assert result.success is True
        assert result.error is None

    def test_link_result_failure(self) -> None:
        """Should create failed LinkResult with error."""
        result = LinkResult(
            success=False,
            source_updated=False,
            target_updated=False,
            source_id="sem_1",
            target_id="sem_2",
            link_type="related",
            error="Memory not found",
        )
        assert result.success is False
        assert result.error == "Memory not found"


class TestStorageHasLinkMethods:
    """Tests that EngramStorage has link methods."""

    def test_storage_has_link_memories(self) -> None:
        """EngramStorage should have link_memories method."""
        from engram.storage import EngramStorage

        storage = EngramStorage()
        assert hasattr(storage, "link_memories")
        assert callable(storage.link_memories)

    def test_storage_has_unlink_memories(self) -> None:
        """EngramStorage should have unlink_memories method."""
        from engram.storage import EngramStorage

        storage = EngramStorage()
        assert hasattr(storage, "unlink_memories")
        assert callable(storage.unlink_memories)
