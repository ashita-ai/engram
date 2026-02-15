"""Unit tests for cascade deletion functionality."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from engram.models import ProceduralMemory, SemanticMemory
from engram.models.base import ConfidenceScore
from engram.storage.crud import CRUDMixin


class MockStorageWithCRUD(CRUDMixin):
    """Mock storage class that includes CRUD mixin for testing."""

    def __init__(self):
        self.client = AsyncMock()
        self._prefix = "test"
        self.log_audit = AsyncMock(return_value="audit_1")

    def _collection_name(self, memory_type: str) -> str:
        return f"{self._prefix}_{memory_type}"

    def _payload_to_memory(self, payload, memory_class):
        return memory_class.model_validate(payload)

    def _memory_to_payload(self, memory):
        return memory.model_dump(mode="json")


class TestCascadeDeleteEpisode:
    """Tests for cascade deletion of episodes."""

    @pytest.fixture
    def storage(self):
        """Create a mock storage instance with CRUD mixin."""
        return MockStorageWithCRUD()


class TestCascadeModes:
    """Tests for different cascade deletion modes."""

    @pytest.fixture
    def storage(self):
        """Create a mock storage instance."""
        return MockStorageWithCRUD()

    @pytest.mark.asyncio
    async def test_cascade_none_only_deletes_episode(self, storage):
        """Cascade mode 'none' should only delete the episode."""
        # Mock get_episode to return an episode
        mock_episode = MagicMock()
        mock_episode.id = "ep_test"
        storage.get_episode = AsyncMock(return_value=mock_episode)

        # Mock _delete_by_id to succeed
        storage._delete_by_id = AsyncMock(return_value=True)

        result = await storage.delete_episode("ep_test", "user_1", cascade="none")

        assert result["deleted"] is True
        assert result["structured_deleted"] == 0
        assert result["semantic_deleted"] == 0
        # Should only call delete once (for the episode)
        storage._delete_by_id.assert_called_once_with("ep_test", "user_1", "episodic")

    @pytest.mark.asyncio
    async def test_cascade_soft_removes_references(self, storage):
        """Cascade mode 'soft' should remove references and reduce confidence."""
        # Mock episode exists
        mock_episode = MagicMock()
        storage.get_episode = AsyncMock(return_value=mock_episode)

        # Mock structured memory exists
        mock_structured = MagicMock()
        mock_structured.id = "struct_1"
        storage.get_structured_for_episode = AsyncMock(return_value=mock_structured)

        # Mock semantic with 2 sources (will be updated, not deleted)
        # Note: for_inferred caps at 0.6, so initial value is 0.6
        mock_semantic = SemanticMemory(
            id="sem_1",
            content="Test",
            source_episode_ids=["ep_test", "ep_other"],
            user_id="user_1",
            confidence=ConfidenceScore.for_inferred(0.8),
        )
        storage._find_semantics_by_source_episode = AsyncMock(return_value=[mock_semantic])

        # Mock no procedural memories
        storage._find_procedurals_by_source_episode = AsyncMock(return_value=[])

        # Mock delete and update
        storage._delete_by_id = AsyncMock(return_value=True)
        storage.update_semantic_memory = AsyncMock(return_value=True)

        result = await storage.delete_episode("ep_test", "user_1", cascade="soft")

        assert result["deleted"] is True
        assert result["structured_deleted"] == 1
        assert result["semantic_deleted"] == 0
        assert result["semantic_updated"] == 1

        # Verify confidence was reduced
        storage.update_semantic_memory.assert_called_once()
        updated_sem = storage.update_semantic_memory.call_args[0][0]
        # Original: 0.6 (capped from 0.8), after removing 1 of 2 sources: 0.6 * 0.5 = 0.3
        assert updated_sem.confidence.value == pytest.approx(0.3, rel=0.01)

    @pytest.mark.asyncio
    async def test_cascade_soft_deletes_orphaned_semantic(self, storage):
        """Cascade mode 'soft' should delete semantic with no remaining sources."""
        mock_episode = MagicMock()
        storage.get_episode = AsyncMock(return_value=mock_episode)
        storage.get_structured_for_episode = AsyncMock(return_value=None)

        # Semantic with only one source (will be deleted)
        mock_semantic = SemanticMemory(
            id="sem_orphan",
            content="Test",
            source_episode_ids=["ep_test"],  # Only this episode
            user_id="user_1",
        )
        storage._find_semantics_by_source_episode = AsyncMock(return_value=[mock_semantic])
        storage._find_procedurals_by_source_episode = AsyncMock(return_value=[])

        storage._delete_by_id = AsyncMock(return_value=True)

        result = await storage.delete_episode("ep_test", "user_1", cascade="soft")

        assert result["semantic_deleted"] == 1
        assert result["semantic_updated"] == 0

    @pytest.mark.asyncio
    async def test_cascade_hard_deletes_all_derived(self, storage):
        """Cascade mode 'hard' should delete all derived memories."""
        mock_episode = MagicMock()
        storage.get_episode = AsyncMock(return_value=mock_episode)

        mock_structured = MagicMock()
        mock_structured.id = "struct_1"
        storage.get_structured_for_episode = AsyncMock(return_value=mock_structured)

        # Semantic with multiple sources (would be updated in soft, deleted in hard)
        mock_semantic = SemanticMemory(
            id="sem_multi",
            content="Test",
            source_episode_ids=["ep_test", "ep_other", "ep_third"],
            user_id="user_1",
        )
        storage._find_semantics_by_source_episode = AsyncMock(return_value=[mock_semantic])

        mock_procedural = ProceduralMemory(
            id="proc_1",
            content="Pattern",
            source_episode_ids=["ep_test", "ep_other"],
            source_semantic_ids=["sem_x"],
            user_id="user_1",
        )
        storage._find_procedurals_by_source_episode = AsyncMock(return_value=[mock_procedural])

        storage._delete_by_id = AsyncMock(return_value=True)

        result = await storage.delete_episode("ep_test", "user_1", cascade="hard")

        assert result["deleted"] is True
        assert result["structured_deleted"] == 1
        assert result["semantic_deleted"] == 1  # Deleted despite having other sources
        assert result["semantic_updated"] == 0
        assert result["procedural_deleted"] == 1
        assert result["procedural_updated"] == 0

    @pytest.mark.asyncio
    async def test_episode_not_found_returns_not_deleted(self, storage):
        """Should return deleted=False when episode doesn't exist."""
        storage.get_episode = AsyncMock(return_value=None)

        result = await storage.delete_episode("ep_nonexistent", "user_1", cascade="soft")

        assert result["deleted"] is False
        assert result["structured_deleted"] == 0


class TestConfidenceReduction:
    """Tests for confidence reduction during soft cascade."""

    @pytest.fixture
    def storage(self):
        """Create a mock storage instance."""
        return MockStorageWithCRUD()

    @pytest.mark.asyncio
    async def test_confidence_floors_at_minimum(self, storage):
        """Confidence should not go below 0.1."""
        mock_episode = MagicMock()
        storage.get_episode = AsyncMock(return_value=mock_episode)
        storage.get_structured_for_episode = AsyncMock(return_value=None)

        # Semantic with 10 sources, removing 1 would reduce to 0.1 * 0.9 = 0.09
        mock_semantic = SemanticMemory(
            id="sem_1",
            content="Test",
            source_episode_ids=["ep_test"] + [f"ep_{i}" for i in range(9)],
            user_id="user_1",
            confidence=ConfidenceScore.for_inferred(0.1),  # Already at minimum
        )
        storage._find_semantics_by_source_episode = AsyncMock(return_value=[mock_semantic])
        storage._find_procedurals_by_source_episode = AsyncMock(return_value=[])

        storage._delete_by_id = AsyncMock(return_value=True)
        storage.update_semantic_memory = AsyncMock(return_value=True)

        await storage.delete_episode("ep_test", "user_1", cascade="soft")

        updated_sem = storage.update_semantic_memory.call_args[0][0]
        # Should be floored at 0.1
        assert updated_sem.confidence.value >= 0.1

    @pytest.mark.asyncio
    async def test_procedural_confidence_reduction(self, storage):
        """Procedural confidence should be reduced proportionally."""
        mock_episode = MagicMock()
        storage.get_episode = AsyncMock(return_value=mock_episode)
        storage.get_structured_for_episode = AsyncMock(return_value=None)
        storage._find_semantics_by_source_episode = AsyncMock(return_value=[])

        # Procedural with episode sources and semantic sources
        # Note: for_inferred caps at 0.6, so initial value is 0.6
        mock_procedural = ProceduralMemory(
            id="proc_1",
            content="Pattern",
            source_episode_ids=["ep_test", "ep_other"],  # 2 episode sources
            source_semantic_ids=["sem_1"],  # Has semantic source too
            user_id="user_1",
            confidence=ConfidenceScore.for_inferred(0.8),
        )
        storage._find_procedurals_by_source_episode = AsyncMock(return_value=[mock_procedural])

        storage._delete_by_id = AsyncMock(return_value=True)
        storage.update_procedural_memory = AsyncMock(return_value=True)

        result = await storage.delete_episode("ep_test", "user_1", cascade="soft")

        assert result["procedural_updated"] == 1
        updated_proc = storage.update_procedural_memory.call_args[0][0]
        # 2 sources -> 1 source = 50% confidence: 0.6 * 0.5 = 0.3
        assert updated_proc.confidence.value == pytest.approx(0.3, rel=0.01)

    @pytest.mark.asyncio
    async def test_procedural_deleted_when_no_sources_remain(self, storage):
        """Procedural should be deleted when all sources are removed."""
        mock_episode = MagicMock()
        storage.get_episode = AsyncMock(return_value=mock_episode)
        storage.get_structured_for_episode = AsyncMock(return_value=None)
        storage._find_semantics_by_source_episode = AsyncMock(return_value=[])

        # Procedural with only episode source (no semantic sources)
        mock_procedural = ProceduralMemory(
            id="proc_orphan",
            content="Pattern",
            source_episode_ids=["ep_test"],  # Only this episode
            source_semantic_ids=[],  # No semantic sources
            user_id="user_1",
        )
        storage._find_procedurals_by_source_episode = AsyncMock(return_value=[mock_procedural])

        storage._delete_by_id = AsyncMock(return_value=True)

        result = await storage.delete_episode("ep_test", "user_1", cascade="soft")

        assert result["procedural_deleted"] == 1
        assert result["procedural_updated"] == 0


class TestMultipleAffectedMemories:
    """Tests for cascade deletion affecting multiple memories."""

    @pytest.fixture
    def storage(self):
        """Create a mock storage instance."""
        return MockStorageWithCRUD()

    @pytest.mark.asyncio
    async def test_multiple_semantics_affected(self, storage):
        """Should handle multiple semantic memories referencing same episode."""
        mock_episode = MagicMock()
        storage.get_episode = AsyncMock(return_value=mock_episode)
        storage.get_structured_for_episode = AsyncMock(return_value=None)

        # Multiple semantics with different source counts
        semantics = [
            SemanticMemory(
                id="sem_1",
                content="Test 1",
                source_episode_ids=["ep_test"],  # Will be deleted
                user_id="user_1",
            ),
            SemanticMemory(
                id="sem_2",
                content="Test 2",
                source_episode_ids=["ep_test", "ep_other"],  # Will be updated
                user_id="user_1",
                confidence=ConfidenceScore.for_inferred(0.8),
            ),
            SemanticMemory(
                id="sem_3",
                content="Test 3",
                source_episode_ids=["ep_test", "ep_a", "ep_b"],  # Will be updated
                user_id="user_1",
                confidence=ConfidenceScore.for_inferred(0.9),
            ),
        ]
        storage._find_semantics_by_source_episode = AsyncMock(return_value=semantics)
        storage._find_procedurals_by_source_episode = AsyncMock(return_value=[])

        storage._delete_by_id = AsyncMock(return_value=True)
        storage.update_semantic_memory = AsyncMock(return_value=True)

        result = await storage.delete_episode("ep_test", "user_1", cascade="soft")

        assert result["semantic_deleted"] == 1  # sem_1
        assert result["semantic_updated"] == 2  # sem_2 and sem_3
