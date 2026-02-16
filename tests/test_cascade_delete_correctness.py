"""Tests for cascade delete correctness (SPEC-005).

Verifies that:
1. Soft cascade uses recompute() instead of direct confidence multiplication
2. Cascade continues on partial failures (best-effort)
3. Episode is always deleted even if cascade steps fail
4. Errors are reported in the return dict
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from engram.models import ProceduralMemory, SemanticMemory
from engram.models.base import ConfidenceScore
from engram.storage.crud import CRUDMixin


class MockStorageWithCRUD(CRUDMixin):
    """Mock storage class that includes CRUD mixin for testing."""

    def __init__(self) -> None:
        self.client = AsyncMock()
        self._prefix = "test"
        self.log_audit = AsyncMock(return_value="audit_1")

    def _collection_name(self, memory_type: str) -> str:
        return f"{self._prefix}_{memory_type}"

    def _payload_to_memory(self, payload: dict, memory_class: type) -> object:
        return memory_class.model_validate(payload)

    def _memory_to_payload(self, memory: object) -> dict:
        return memory.model_dump(mode="json")  # type: ignore[union-attr]


class TestSoftCascadeUsesRecompute:
    """Tests that soft cascade uses recompute() for confidence adjustment."""

    @pytest.fixture
    def storage(self) -> MockStorageWithCRUD:
        return MockStorageWithCRUD()

    @pytest.mark.asyncio
    async def test_semantic_supporting_episodes_set_correctly(
        self, storage: MockStorageWithCRUD
    ) -> None:
        """Removing a source should set supporting_episodes and call recompute()."""
        mock_episode = MagicMock()
        storage.get_episode = AsyncMock(return_value=mock_episode)
        storage.get_structured_for_episode = AsyncMock(return_value=None)

        # 3 sources, removing 1 leaves 2
        sem = SemanticMemory(
            id="sem_1",
            content="Test fact",
            source_episode_ids=["ep_target", "ep_a", "ep_b"],
            user_id="user_1",
            confidence=ConfidenceScore.for_inferred(0.6, supporting_episodes=3),
        )
        storage._find_semantics_by_source_episode = AsyncMock(return_value=[sem])
        storage._find_procedurals_by_source_episode = AsyncMock(return_value=[])
        storage._delete_by_id = AsyncMock(return_value=True)
        storage.update_semantic_memory = AsyncMock(return_value=True)

        await storage.delete_episode("ep_target", "user_1", cascade="soft")

        updated = storage.update_semantic_memory.call_args[0][0]
        assert updated.confidence.supporting_episodes == 2
        # Verify recompute was used: the value should match a fresh recompute
        expected = ConfidenceScore.for_inferred(0.6, supporting_episodes=2)
        expected.recompute()
        assert updated.confidence.value == pytest.approx(expected.value, rel=0.01)

    @pytest.mark.asyncio
    async def test_semantic_recompute_differs_from_old_multiplication(
        self, storage: MockStorageWithCRUD
    ) -> None:
        """Recompute() should produce different values than the old multiplication."""
        mock_episode = MagicMock()
        storage.get_episode = AsyncMock(return_value=mock_episode)
        storage.get_structured_for_episode = AsyncMock(return_value=None)

        sem = SemanticMemory(
            id="sem_1",
            content="Test fact",
            source_episode_ids=["ep_target", "ep_other"],
            user_id="user_1",
            confidence=ConfidenceScore.for_inferred(0.6, supporting_episodes=2),
        )
        storage._find_semantics_by_source_episode = AsyncMock(return_value=[sem])
        storage._find_procedurals_by_source_episode = AsyncMock(return_value=[])
        storage._delete_by_id = AsyncMock(return_value=True)
        storage.update_semantic_memory = AsyncMock(return_value=True)

        await storage.delete_episode("ep_target", "user_1", cascade="soft")

        updated = storage.update_semantic_memory.call_args[0][0]
        # Old broken behavior: 0.6 * (1/2) = 0.3
        # New recompute behavior: weighted formula with corroboration, recency, etc.
        old_broken_value = 0.3
        assert updated.confidence.value != pytest.approx(old_broken_value, rel=0.01)

    @pytest.mark.asyncio
    async def test_procedural_supporting_episodes_includes_semantic_sources(
        self, storage: MockStorageWithCRUD
    ) -> None:
        """Procedural supporting_episodes should count both episode and semantic sources."""
        mock_episode = MagicMock()
        storage.get_episode = AsyncMock(return_value=mock_episode)
        storage.get_structured_for_episode = AsyncMock(return_value=None)
        storage._find_semantics_by_source_episode = AsyncMock(return_value=[])

        proc = ProceduralMemory(
            id="proc_1",
            content="Pattern",
            source_episode_ids=["ep_target", "ep_other"],  # 2 episodes
            source_semantic_ids=["sem_a", "sem_b"],  # 2 semantics
            user_id="user_1",
            confidence=ConfidenceScore.for_inferred(0.6, supporting_episodes=4),
        )
        storage._find_procedurals_by_source_episode = AsyncMock(return_value=[proc])
        storage._delete_by_id = AsyncMock(return_value=True)
        storage.update_procedural_memory = AsyncMock(return_value=True)

        await storage.delete_episode("ep_target", "user_1", cascade="soft")

        updated = storage.update_procedural_memory.call_args[0][0]
        # 1 remaining episode + 2 semantics = 3 total
        assert updated.confidence.supporting_episodes == 3


class TestCascadeErrorRecovery:
    """Tests that cascade continues on partial failures."""

    @pytest.fixture
    def storage(self) -> MockStorageWithCRUD:
        return MockStorageWithCRUD()

    @pytest.mark.asyncio
    async def test_episode_deleted_even_if_structured_fails(
        self, storage: MockStorageWithCRUD
    ) -> None:
        """Episode should be deleted even when structured cascade raises."""
        mock_episode = MagicMock()
        storage.get_episode = AsyncMock(return_value=mock_episode)
        storage.get_structured_for_episode = AsyncMock(
            side_effect=RuntimeError("structured lookup exploded")
        )
        storage._find_semantics_by_source_episode = AsyncMock(return_value=[])
        storage._find_procedurals_by_source_episode = AsyncMock(return_value=[])
        storage._delete_by_id = AsyncMock(return_value=True)

        result = await storage.delete_episode("ep_test", "user_1", cascade="soft")

        assert result["deleted"] is True
        assert "errors" in result
        assert any("structured" in e for e in result["errors"])

    @pytest.mark.asyncio
    async def test_cascade_continues_when_one_semantic_fails(
        self, storage: MockStorageWithCRUD
    ) -> None:
        """Should process remaining semantics even if one fails."""
        mock_episode = MagicMock()
        storage.get_episode = AsyncMock(return_value=mock_episode)
        storage.get_structured_for_episode = AsyncMock(return_value=None)

        sem_ok = SemanticMemory(
            id="sem_ok",
            content="Test OK",
            source_episode_ids=["ep_test"],
            user_id="user_1",
        )
        sem_fail = SemanticMemory(
            id="sem_fail",
            content="Test FAIL",
            source_episode_ids=["ep_test"],
            user_id="user_1",
        )
        storage._find_semantics_by_source_episode = AsyncMock(return_value=[sem_fail, sem_ok])
        storage._find_procedurals_by_source_episode = AsyncMock(return_value=[])

        # First delete (sem_fail) raises, second delete (sem_ok) succeeds,
        # third delete (episode) succeeds
        call_count = 0

        async def selective_delete(memory_id: str, user_id: str, coll: str) -> bool:
            nonlocal call_count
            call_count += 1
            if memory_id == "sem_fail":
                raise RuntimeError("storage unavailable")
            return True

        storage._delete_by_id = AsyncMock(side_effect=selective_delete)

        result = await storage.delete_episode("ep_test", "user_1", cascade="soft")

        assert result["deleted"] is True
        assert result["semantic_deleted"] == 1  # sem_ok deleted successfully
        assert "errors" in result
        assert any("sem_fail" in e for e in result["errors"])

    @pytest.mark.asyncio
    async def test_episode_deleted_even_if_all_cascade_steps_fail(
        self, storage: MockStorageWithCRUD
    ) -> None:
        """Episode should be deleted even when every cascade step raises."""
        mock_episode = MagicMock()
        storage.get_episode = AsyncMock(return_value=mock_episode)
        storage.get_structured_for_episode = AsyncMock(
            side_effect=RuntimeError("structured exploded")
        )
        storage._find_semantics_by_source_episode = AsyncMock(
            side_effect=RuntimeError("semantic search exploded")
        )
        storage._find_procedurals_by_source_episode = AsyncMock(
            side_effect=RuntimeError("procedural search exploded")
        )
        storage._delete_by_id = AsyncMock(return_value=True)

        result = await storage.delete_episode("ep_test", "user_1", cascade="soft")

        assert result["deleted"] is True
        errors = result["errors"]
        assert len(errors) == 3
        assert any("structured" in e for e in errors)
        assert any("semantic" in e for e in errors)
        assert any("procedural" in e for e in errors)

    @pytest.mark.asyncio
    async def test_errors_reported_in_result(self, storage: MockStorageWithCRUD) -> None:
        """Return dict should include errors list when cascade has failures."""
        mock_episode = MagicMock()
        storage.get_episode = AsyncMock(return_value=mock_episode)

        # Only structured fails
        mock_structured = MagicMock()
        mock_structured.id = "struct_1"
        storage.get_structured_for_episode = AsyncMock(return_value=mock_structured)
        storage._delete_by_id = AsyncMock(side_effect=RuntimeError("delete failed"))
        storage._find_semantics_by_source_episode = AsyncMock(return_value=[])
        storage._find_procedurals_by_source_episode = AsyncMock(return_value=[])

        result = await storage.delete_episode("ep_test", "user_1", cascade="soft")

        # Episode delete also fails (same mock)
        assert result["deleted"] is False
        assert "errors" in result

    @pytest.mark.asyncio
    async def test_no_errors_key_on_clean_cascade(self, storage: MockStorageWithCRUD) -> None:
        """Return dict should NOT include errors key when cascade is clean."""
        mock_episode = MagicMock()
        storage.get_episode = AsyncMock(return_value=mock_episode)
        storage.get_structured_for_episode = AsyncMock(return_value=None)
        storage._find_semantics_by_source_episode = AsyncMock(return_value=[])
        storage._find_procedurals_by_source_episode = AsyncMock(return_value=[])
        storage._delete_by_id = AsyncMock(return_value=True)

        result = await storage.delete_episode("ep_test", "user_1", cascade="soft")

        assert result["deleted"] is True
        assert "errors" not in result

    @pytest.mark.asyncio
    async def test_semantic_update_failure_continues(self, storage: MockStorageWithCRUD) -> None:
        """Should continue processing if update_semantic_memory fails for one."""
        mock_episode = MagicMock()
        storage.get_episode = AsyncMock(return_value=mock_episode)
        storage.get_structured_for_episode = AsyncMock(return_value=None)

        sem1 = SemanticMemory(
            id="sem_1",
            content="Test 1",
            source_episode_ids=["ep_test", "ep_a"],
            user_id="user_1",
            confidence=ConfidenceScore.for_inferred(0.6, supporting_episodes=2),
        )
        sem2 = SemanticMemory(
            id="sem_2",
            content="Test 2",
            source_episode_ids=["ep_test", "ep_b"],
            user_id="user_1",
            confidence=ConfidenceScore.for_inferred(0.6, supporting_episodes=2),
        )
        storage._find_semantics_by_source_episode = AsyncMock(return_value=[sem1, sem2])
        storage._find_procedurals_by_source_episode = AsyncMock(return_value=[])

        # First update fails, second succeeds
        storage.update_semantic_memory = AsyncMock(
            side_effect=[RuntimeError("update failed"), True]
        )
        storage._delete_by_id = AsyncMock(return_value=True)

        result = await storage.delete_episode("ep_test", "user_1", cascade="soft")

        assert result["deleted"] is True
        assert result["semantic_updated"] == 1  # Only sem2 succeeded
        assert "errors" in result
        assert any("sem_1" in e for e in result["errors"])
