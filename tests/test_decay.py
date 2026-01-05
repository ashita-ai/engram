"""Tests for decay workflow."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from engram.config import Settings
from engram.models import SemanticMemory
from engram.models.base import ConfidenceScore, ExtractionMethod
from engram.workflows.decay import DecayResult, run_decay


class TestDecayResult:
    """Tests for DecayResult model."""

    def test_create_result(self) -> None:
        """Test creating decay result."""
        result = DecayResult(
            memories_updated=10,
            memories_archived=5,
            memories_deleted=2,
        )
        assert result.memories_updated == 10
        assert result.memories_archived == 5
        assert result.memories_deleted == 2

    def test_create_empty_result(self) -> None:
        """Test creating empty decay result."""
        result = DecayResult(
            memories_updated=0,
            memories_archived=0,
            memories_deleted=0,
        )
        assert result.memories_updated == 0
        assert result.memories_archived == 0
        assert result.memories_deleted == 0

    def test_counts_non_negative(self) -> None:
        """Test counts cannot be negative."""
        with pytest.raises(ValueError):
            DecayResult(
                memories_updated=-1,
                memories_archived=0,
                memories_deleted=0,
            )

        with pytest.raises(ValueError):
            DecayResult(
                memories_updated=0,
                memories_archived=-1,
                memories_deleted=0,
            )

        with pytest.raises(ValueError):
            DecayResult(
                memories_updated=0,
                memories_archived=0,
                memories_deleted=-1,
            )

    def test_extra_fields_forbidden(self) -> None:
        """Test extra fields are rejected."""
        with pytest.raises(ValueError):
            DecayResult(
                memories_updated=0,
                memories_archived=0,
                memories_deleted=0,
                extra_field=42,  # type: ignore[call-arg]
            )


class TestRunDecay:
    """Tests for run_decay workflow function."""

    @pytest.fixture
    def settings(self) -> Settings:
        """Create settings with known thresholds.

        Thresholds are set to values achievable after recompute:
        - archive_threshold=0.4: Below ~0.575 (fresh, high extraction)
        - delete_threshold=0.2: Below ~0.21 (old, low extraction)
        """
        return Settings(
            openai_api_key="sk-test",
            decay_archive_threshold=0.4,
            decay_delete_threshold=0.2,
        )

    def _create_memory(
        self,
        content: str,
        extraction_base: float = 0.6,
        days_old: int = 0,
        contradictions: int = 0,
        archived: bool = False,
    ) -> SemanticMemory:
        """Create a SemanticMemory with specific confidence factors.

        After recompute_with_weights with default weights (0.5/0.25/0.15/0.10):
        - extraction_base=0.6, days_old=0 → ~0.575 (high confidence)
        - extraction_base=0.1, days_old=730 (2yr) → ~0.21 (for archive)
        - extraction_base=0.1, days_old=730, contradictions=3 → ~0.15 (for delete)
        """
        memory = SemanticMemory(
            content=content,
            user_id="test_user",
            embedding=[0.1] * 384,
            archived=archived,
        )
        # Set confidence with factors that will produce desired post-recompute value
        memory.confidence = ConfidenceScore(
            value=0.5,  # Will be overwritten by recompute
            extraction_method=ExtractionMethod.INFERRED,
            extraction_base=extraction_base,
            supporting_episodes=1,
            verified=False,
            last_confirmed=datetime.now(UTC) - timedelta(days=days_old),
            contradictions=contradictions,
        )
        return memory

    @pytest.mark.asyncio
    async def test_no_memories_returns_empty_result(self, settings: Settings) -> None:
        """Test that empty memory list returns zero counts."""
        mock_storage = AsyncMock()
        mock_storage.list_semantic_memories = AsyncMock(return_value=[])

        result = await run_decay(
            storage=mock_storage,
            settings=settings,
            user_id="test_user",
        )

        assert result.memories_updated == 0
        assert result.memories_archived == 0
        assert result.memories_deleted == 0

    @pytest.mark.asyncio
    async def test_high_confidence_memory_updated(self, settings: Settings) -> None:
        """Test memory above archive threshold gets updated."""
        # Fresh memory with high extraction_base → ~0.575 (above 0.4 archive threshold)
        memory = self._create_memory("User likes Python", extraction_base=0.6, days_old=0)

        mock_storage = AsyncMock()
        mock_storage.list_semantic_memories = AsyncMock(return_value=[memory])
        mock_storage.update_semantic_memory = AsyncMock()

        result = await run_decay(
            storage=mock_storage,
            settings=settings,
            user_id="test_user",
        )

        # High confidence memory may be updated if confidence changed
        assert result.memories_archived == 0
        assert result.memories_deleted == 0
        mock_storage.delete_semantic.assert_not_called()

    @pytest.mark.asyncio
    async def test_low_confidence_memory_archived(self, settings: Settings) -> None:
        """Test memory below archive threshold gets archived."""
        # Older memory with low extraction_base → ~0.25 (below 0.4, above 0.2)
        memory = self._create_memory("Old preference", extraction_base=0.1, days_old=730)

        mock_storage = AsyncMock()
        mock_storage.list_semantic_memories = AsyncMock(return_value=[memory])
        mock_storage.update_semantic_memory = AsyncMock()

        result = await run_decay(
            storage=mock_storage,
            settings=settings,
            user_id="test_user",
        )

        assert result.memories_archived == 1
        assert result.memories_deleted == 0
        mock_storage.update_semantic_memory.assert_called_once()
        # Verify memory was marked as archived
        updated_memory = mock_storage.update_semantic_memory.call_args[0][0]
        assert updated_memory.archived is True

    @pytest.mark.asyncio
    async def test_very_low_confidence_memory_deleted(self, settings: Settings) -> None:
        """Test memory below delete threshold gets deleted."""
        # Very old memory with contradictions → below 0.2 delete threshold
        # Base ~0.21, with 1 contradiction at 10% penalty: 0.21 * 0.9 = 0.189
        memory = self._create_memory(
            "Forgotten info", extraction_base=0.1, days_old=730, contradictions=1
        )

        mock_storage = AsyncMock()
        mock_storage.list_semantic_memories = AsyncMock(return_value=[memory])
        mock_storage.delete_semantic = AsyncMock()

        result = await run_decay(
            storage=mock_storage,
            settings=settings,
            user_id="test_user",
        )

        assert result.memories_deleted == 1
        assert result.memories_archived == 0
        mock_storage.delete_semantic.assert_called_once_with(memory.id, "test_user")

    @pytest.mark.asyncio
    async def test_already_archived_memory_updated(self, settings: Settings) -> None:
        """Test already archived memory just gets confidence updated."""
        # Already archived, between delete and archive thresholds
        memory = self._create_memory(
            "Archived info", extraction_base=0.1, days_old=730, archived=True
        )

        mock_storage = AsyncMock()
        mock_storage.list_semantic_memories = AsyncMock(return_value=[memory])
        mock_storage.update_semantic_memory = AsyncMock()

        result = await run_decay(
            storage=mock_storage,
            settings=settings,
            user_id="test_user",
        )

        # Already archived, so just updated not newly archived
        assert result.memories_updated == 1
        assert result.memories_archived == 0

    @pytest.mark.asyncio
    async def test_mixed_memories_processed_correctly(self, settings: Settings) -> None:
        """Test multiple memories with different confidences are handled correctly."""
        # High confidence: fresh, high extraction → ~0.575
        high_conf = self._create_memory("Recent fact", extraction_base=0.6, days_old=0)
        # Low confidence: old, low extraction → ~0.25 (archive)
        low_conf = self._create_memory("Old preference", extraction_base=0.1, days_old=730)
        # Very low confidence: old + contradiction → ~0.19 (delete)
        very_low = self._create_memory(
            "Forgotten", extraction_base=0.1, days_old=730, contradictions=1
        )

        mock_storage = AsyncMock()
        mock_storage.list_semantic_memories = AsyncMock(
            return_value=[high_conf, low_conf, very_low]
        )
        mock_storage.update_semantic_memory = AsyncMock()
        mock_storage.delete_semantic = AsyncMock()

        result = await run_decay(
            storage=mock_storage,
            settings=settings,
            user_id="test_user",
        )

        # Should have: 1 archived, 1 deleted
        assert result.memories_archived == 1
        assert result.memories_deleted == 1
        mock_storage.delete_semantic.assert_called_once()

    @pytest.mark.asyncio
    async def test_org_id_passed_to_storage(self, settings: Settings) -> None:
        """Test org_id is correctly passed to storage methods."""
        mock_storage = AsyncMock()
        mock_storage.list_semantic_memories = AsyncMock(return_value=[])

        await run_decay(
            storage=mock_storage,
            settings=settings,
            user_id="test_user",
            org_id="test_org",
        )

        mock_storage.list_semantic_memories.assert_called_once_with(
            user_id="test_user",
            org_id="test_org",
            include_archived=True,
        )
