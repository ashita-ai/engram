"""Tests for promotion workflow."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from engram.workflows.promotion import (
    PromotionResult,
    _extract_trigger_context,
    _is_behavioral_pattern,
    _is_duplicate_pattern,
    _should_promote_to_procedural,
    run_promotion,
)


class TestPromotionResult:
    """Tests for PromotionResult model."""

    def test_create_result(self) -> None:
        """Test creating promotion result."""
        result = PromotionResult(
            memories_analyzed=10,
            procedural_created=3,
            patterns_detected=["Pattern A", "Pattern B", "Pattern C"],
        )
        assert result.memories_analyzed == 10
        assert result.procedural_created == 3
        assert len(result.patterns_detected) == 3

    def test_create_empty_result(self) -> None:
        """Test creating empty result."""
        result = PromotionResult(
            memories_analyzed=0,
            procedural_created=0,
        )
        assert result.memories_analyzed == 0
        assert result.procedural_created == 0
        assert result.patterns_detected == []

    def test_counts_non_negative(self) -> None:
        """Test counts cannot be negative."""
        with pytest.raises(ValueError):
            PromotionResult(
                memories_analyzed=-1,
                procedural_created=0,
            )


class TestIsBehavioralPattern:
    """Tests for _is_behavioral_pattern function."""

    def test_detects_prefers(self) -> None:
        """Test detection of 'prefers' keyword."""
        assert _is_behavioral_pattern("User prefers Python")
        assert _is_behavioral_pattern("The user PREFERS dark mode")

    def test_detects_always(self) -> None:
        """Test detection of 'always' keyword."""
        assert _is_behavioral_pattern("User always asks for examples")

    def test_detects_tends_to(self) -> None:
        """Test detection of 'tends to' keyword."""
        assert _is_behavioral_pattern("User tends to work late")

    def test_detects_likes_to(self) -> None:
        """Test detection of 'likes to' keyword."""
        assert _is_behavioral_pattern("User likes to use TypeScript")

    def test_detects_habit(self) -> None:
        """Test detection of 'habit' keyword."""
        assert _is_behavioral_pattern("User has a habit of asking follow-up questions")

    def test_non_behavioral_content(self) -> None:
        """Test non-behavioral content returns False."""
        assert not _is_behavioral_pattern("User's email is test@example.com")
        assert not _is_behavioral_pattern("Meeting scheduled for Monday")
        assert not _is_behavioral_pattern("PostgreSQL is a relational database")


class TestShouldPromoteToProcedural:
    """Tests for _should_promote_to_procedural function."""

    def test_promotes_high_selectivity_behavioral_pattern(self) -> None:
        """Test promotion of well-consolidated behavioral pattern."""
        from engram.models import SemanticMemory

        memory = SemanticMemory(
            content="User prefers concise responses",
            user_id="test_user",
            embedding=[0.1] * 384,
        )
        memory.selectivity_score = 0.6
        memory.consolidation_passes = 3
        memory.confidence.value = 0.8

        assert _should_promote_to_procedural(memory)

    def test_rejects_low_selectivity(self) -> None:
        """Test rejection when selectivity too low."""
        from engram.models import SemanticMemory

        memory = SemanticMemory(
            content="User prefers concise responses",
            user_id="test_user",
            embedding=[0.1] * 384,
        )
        memory.selectivity_score = 0.3  # Too low
        memory.consolidation_passes = 3
        memory.confidence.value = 0.8

        assert not _should_promote_to_procedural(memory)

    def test_rejects_few_consolidation_passes(self) -> None:
        """Test rejection when too few consolidation passes."""
        from engram.models import SemanticMemory

        memory = SemanticMemory(
            content="User prefers concise responses",
            user_id="test_user",
            embedding=[0.1] * 384,
        )
        memory.selectivity_score = 0.6
        memory.consolidation_passes = 1  # Too few
        memory.confidence.value = 0.8

        assert not _should_promote_to_procedural(memory)

    def test_rejects_low_confidence(self) -> None:
        """Test rejection when confidence too low."""
        from engram.models import SemanticMemory

        memory = SemanticMemory(
            content="User prefers concise responses",
            user_id="test_user",
            embedding=[0.1] * 384,
        )
        memory.selectivity_score = 0.6
        memory.consolidation_passes = 3
        memory.confidence.value = 0.5  # Too low

        assert not _should_promote_to_procedural(memory)

    def test_rejects_non_behavioral(self) -> None:
        """Test rejection when content is not behavioral."""
        from engram.models import SemanticMemory

        memory = SemanticMemory(
            content="User's email is test@example.com",  # Not behavioral
            user_id="test_user",
            embedding=[0.1] * 384,
        )
        memory.selectivity_score = 0.6
        memory.consolidation_passes = 3
        memory.confidence.value = 0.8

        assert not _should_promote_to_procedural(memory)


class TestExtractTriggerContext:
    """Tests for _extract_trigger_context function."""

    def test_extracts_when_clause(self) -> None:
        """Test extraction of 'when' context."""
        result = _extract_trigger_context("User prefers Python when writing scripts")
        assert "when writing scripts" in result

    def test_extracts_for_clause(self) -> None:
        """Test extraction of 'for' context."""
        result = _extract_trigger_context("User prefers TypeScript for frontend")
        assert "for frontend" in result

    def test_extracts_in_clause(self) -> None:
        """Test extraction of 'in' context."""
        result = _extract_trigger_context("User always uses dark mode in the IDE")
        assert "in the IDE" in result

    def test_default_context(self) -> None:
        """Test default context when no clause found."""
        result = _extract_trigger_context("User prefers concise responses")
        assert result == "general interaction"


class TestIsDuplicatePattern:
    """Tests for _is_duplicate_pattern function."""

    def test_detects_exact_duplicate(self) -> None:
        """Test detection of exact duplicate."""
        existing = {"user prefers python"}
        assert _is_duplicate_pattern("User prefers Python", existing)

    def test_detects_substring_duplicate(self) -> None:
        """Test detection of substring duplicate."""
        existing = {"user prefers python for scripting"}
        assert _is_duplicate_pattern("prefers python", existing)

    def test_no_duplicate(self) -> None:
        """Test no duplicate found."""
        existing = {"user prefers java"}
        assert not _is_duplicate_pattern("User prefers Python", existing)

    def test_empty_existing(self) -> None:
        """Test empty existing set."""
        assert not _is_duplicate_pattern("any content", set())


class TestRunPromotion:
    """Tests for run_promotion workflow."""

    @pytest.mark.asyncio
    async def test_no_memories_returns_empty_result(self) -> None:
        """Test that empty memory list returns zero counts."""
        mock_storage = AsyncMock()
        mock_storage.list_semantic_memories = AsyncMock(return_value=[])

        mock_embedder = AsyncMock()

        result = await run_promotion(
            storage=mock_storage,
            embedder=mock_embedder,
            user_id="test_user",
        )

        assert result.memories_analyzed == 0
        assert result.procedural_created == 0

    @pytest.mark.asyncio
    async def test_promotes_behavioral_pattern(self) -> None:
        """Test promotion of behavioral pattern to procedural memory."""
        from engram.models import SemanticMemory

        memory = SemanticMemory(
            content="User prefers detailed code explanations",
            user_id="test_user",
            embedding=[0.1] * 384,
        )
        memory.selectivity_score = 0.6
        memory.consolidation_passes = 3
        memory.confidence.value = 0.85

        mock_storage = AsyncMock()
        mock_storage.list_semantic_memories = AsyncMock(return_value=[memory])
        mock_storage.store_procedural = AsyncMock(return_value="proc_123")
        mock_storage.update_semantic_memory = AsyncMock(return_value=True)

        mock_embedder = AsyncMock()
        mock_embedder.embed = AsyncMock(return_value=[0.1] * 384)

        result = await run_promotion(
            storage=mock_storage,
            embedder=mock_embedder,
            user_id="test_user",
        )

        assert result.memories_analyzed == 1
        assert result.procedural_created == 1
        assert len(result.patterns_detected) == 1
        mock_storage.store_procedural.assert_called_once()
        mock_storage.update_semantic_memory.assert_called_once()

    @pytest.mark.asyncio
    async def test_skips_non_promotable_memories(self) -> None:
        """Test that non-promotable memories are skipped."""
        from engram.models import SemanticMemory

        # Non-behavioral memory
        memory1 = SemanticMemory(
            content="User's email is test@example.com",
            user_id="test_user",
            embedding=[0.1] * 384,
        )
        memory1.selectivity_score = 0.6
        memory1.consolidation_passes = 3
        memory1.confidence.value = 0.85

        # Low selectivity
        memory2 = SemanticMemory(
            content="User prefers Python",
            user_id="test_user",
            embedding=[0.1] * 384,
        )
        memory2.selectivity_score = 0.2  # Too low
        memory2.consolidation_passes = 3
        memory2.confidence.value = 0.85

        mock_storage = AsyncMock()
        mock_storage.list_semantic_memories = AsyncMock(return_value=[memory1, memory2])

        mock_embedder = AsyncMock()

        result = await run_promotion(
            storage=mock_storage,
            embedder=mock_embedder,
            user_id="test_user",
        )

        assert result.memories_analyzed == 2
        assert result.procedural_created == 0

    @pytest.mark.asyncio
    async def test_promotes_multiple_patterns(self) -> None:
        """Test promotion of multiple behavioral patterns."""
        from engram.models import SemanticMemory

        memories = []
        for content in [
            "User prefers Python for scripting",
            "User always wants code examples",
            "User tends to ask clarifying questions",
        ]:
            mem = SemanticMemory(
                content=content,
                user_id="test_user",
                embedding=[0.1] * 384,
            )
            mem.selectivity_score = 0.6
            mem.consolidation_passes = 3
            mem.confidence.value = 0.85
            memories.append(mem)

        mock_storage = AsyncMock()
        mock_storage.list_semantic_memories = AsyncMock(return_value=memories)
        mock_storage.store_procedural = AsyncMock(return_value="proc_123")
        mock_storage.update_semantic_memory = AsyncMock(return_value=True)

        mock_embedder = AsyncMock()
        mock_embedder.embed = AsyncMock(return_value=[0.1] * 384)

        result = await run_promotion(
            storage=mock_storage,
            embedder=mock_embedder,
            user_id="test_user",
        )

        assert result.memories_analyzed == 3
        assert result.procedural_created == 3
        assert mock_storage.store_procedural.call_count == 3
