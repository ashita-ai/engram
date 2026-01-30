"""Tests for contradiction detection module."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from engram.models.base import OperationStatus
from engram.service.contradiction import (
    ConflictAnalysis,
    ConflictDetection,
    analyze_pair,
    detect_contradictions,
    get_conflict_agent,
)


class TestConflictDetection:
    """Tests for ConflictDetection model."""

    def test_create_with_required_fields(self):
        """Should create with required fields."""
        conflict = ConflictDetection(
            memory_a_id="mem_1",
            memory_a_content="User likes coffee",
            memory_b_id="mem_2",
            memory_b_content="User hates coffee",
            conflict_type="direct",
            confidence=0.9,
            explanation="Direct contradiction about coffee preference",
            user_id="user_123",
        )
        assert conflict.memory_a_id == "mem_1"
        assert conflict.memory_b_id == "mem_2"
        assert conflict.conflict_type == "direct"
        assert conflict.confidence == 0.9
        assert conflict.resolution is None
        assert conflict.resolved_at is None
        assert conflict.id.startswith("conflict_")

    def test_create_with_optional_fields(self):
        """Should create with optional fields."""
        now = datetime.now(UTC)
        conflict = ConflictDetection(
            memory_a_id="mem_1",
            memory_a_content="Content A",
            memory_b_id="mem_2",
            memory_b_content="Content B",
            conflict_type="temporal",
            confidence=0.7,
            explanation="Temporal conflict",
            user_id="user_123",
            org_id="org_456",
            resolution="newer_wins",
            resolved_at=now,
        )
        assert conflict.org_id == "org_456"
        assert conflict.resolution == "newer_wins"
        assert conflict.resolved_at == now


class TestConflictAnalysis:
    """Tests for ConflictAnalysis model."""

    def test_create_conflict(self):
        """Should create conflict analysis."""
        analysis = ConflictAnalysis(
            is_conflict=True,
            conflict_type="direct",
            confidence=0.85,
            explanation="Direct contradiction about database preference",
            suggested_resolution="newer_wins",
        )
        assert analysis.is_conflict is True
        assert analysis.conflict_type == "direct"
        assert analysis.confidence == 0.85
        assert analysis.status == OperationStatus.SUCCESS  # Default status

    def test_create_no_conflict(self):
        """Should create non-conflict analysis."""
        analysis = ConflictAnalysis(
            is_conflict=False,
            conflict_type="none",
            confidence=0.0,
            explanation="No contradiction found",
        )
        assert analysis.is_conflict is False
        assert analysis.conflict_type == "none"
        assert analysis.status == OperationStatus.SUCCESS

    def test_create_failed_analysis(self):
        """Should create failed analysis with status and error message."""
        analysis = ConflictAnalysis(
            status=OperationStatus.FAILED,
            is_conflict=False,
            conflict_type="none",
            confidence=0.0,
            explanation="Analysis failed due to LLM error",
            error_message="Connection timeout",
        )
        assert analysis.status == OperationStatus.FAILED
        assert analysis.is_conflict is False
        assert analysis.error_message == "Connection timeout"


class TestGetConflictAgent:
    """Tests for conflict agent creation."""

    def test_creates_agent(self, monkeypatch):
        """Should create conflict agent with default model."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-for-agent-creation")
        with patch("engram.service.contradiction._conflict_agent", None):
            agent = get_conflict_agent()
            assert agent is not None

    def test_caches_agent(self, monkeypatch):
        """Should return same agent on repeated calls."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-for-agent-creation")
        with patch("engram.service.contradiction._conflict_agent", None):
            agent1 = get_conflict_agent()
            agent2 = get_conflict_agent()
            assert agent1 is agent2


class TestAnalyzePair:
    """Tests for analyze_pair function."""

    @pytest.mark.asyncio
    async def test_returns_conflict_analysis(self):
        """Should return conflict analysis from LLM."""
        mock_result = MagicMock()
        mock_result.output = ConflictAnalysis(
            is_conflict=True,
            conflict_type="direct",
            confidence=0.9,
            explanation="Direct contradiction",
        )

        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value=mock_result)

        with patch(
            "engram.service.contradiction.get_conflict_agent",
            return_value=mock_agent,
        ):
            result = await analyze_pair(
                "User likes coffee",
                "User hates coffee",
            )
            assert result.is_conflict is True
            assert result.conflict_type == "direct"
            assert result.confidence == 0.9

    @pytest.mark.asyncio
    async def test_handles_error_with_failed_status(self):
        """Should return FAILED status on LLM error, not masquerade as success."""
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(side_effect=Exception("LLM error"))

        with patch(
            "engram.service.contradiction.get_conflict_agent",
            return_value=mock_agent,
        ):
            result = await analyze_pair("Content A", "Content B")
            # Key change: status is FAILED, not SUCCESS
            assert result.status == OperationStatus.FAILED
            assert result.is_conflict is False  # Fallback value
            assert result.error_message is not None
            assert "LLM error" in result.error_message  # May be wrapped in retry message
            assert "failed" in result.explanation.lower()


class TestDetectContradictions:
    """Tests for detect_contradictions function."""

    @pytest.mark.asyncio
    async def test_returns_empty_for_empty_memories(self):
        """Should return empty list when no memories."""
        mock_embedder = MagicMock()

        result = await detect_contradictions(
            new_memories=[],
            existing_memories=[],
            embedder=mock_embedder,
            user_id="user_123",
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_detects_conflicts_above_threshold(self):
        """Should detect conflicts when similarity above threshold."""
        # Create mock semantic memories
        mock_new_mem = MagicMock()
        mock_new_mem.id = "sem_new"
        mock_new_mem.content = "User prefers PostgreSQL"
        mock_new_mem.embedding = [0.9, 0.1, 0.0]

        mock_existing_mem = MagicMock()
        mock_existing_mem.id = "sem_existing"
        mock_existing_mem.content = "User doesn't like SQL databases"
        mock_existing_mem.embedding = [0.85, 0.15, 0.0]

        mock_embedder = MagicMock()
        mock_embedder.embed = AsyncMock(return_value=[0.9, 0.1, 0.0])

        mock_analysis = ConflictAnalysis(
            is_conflict=True,
            conflict_type="implicit",
            confidence=0.75,
            explanation="Implicit contradiction",
        )

        with patch(
            "engram.service.contradiction.analyze_pair",
            return_value=mock_analysis,
        ):
            with patch(
                "engram.service.helpers.cosine_similarity",
                return_value=0.8,  # Above threshold
            ):
                result = await detect_contradictions(
                    new_memories=[mock_new_mem],
                    existing_memories=[mock_existing_mem],
                    embedder=mock_embedder,
                    user_id="user_123",
                    similarity_threshold=0.5,
                )

                assert len(result) == 1
                assert result[0].memory_a_id == "sem_new"
                assert result[0].memory_b_id == "sem_existing"
                assert result[0].conflict_type == "implicit"

    @pytest.mark.asyncio
    async def test_skips_low_similarity_pairs(self):
        """Should skip pairs with low similarity."""
        mock_new_mem = MagicMock()
        mock_new_mem.id = "sem_new"
        mock_new_mem.content = "User likes pizza"
        mock_new_mem.embedding = [1.0, 0.0, 0.0]

        mock_existing_mem = MagicMock()
        mock_existing_mem.id = "sem_existing"
        mock_existing_mem.content = "User works at Company X"
        mock_existing_mem.embedding = [0.0, 1.0, 0.0]

        mock_embedder = MagicMock()

        with patch(
            "engram.service.helpers.cosine_similarity",
            return_value=0.2,  # Below threshold
        ):
            result = await detect_contradictions(
                new_memories=[mock_new_mem],
                existing_memories=[mock_existing_mem],
                embedder=mock_embedder,
                user_id="user_123",
                similarity_threshold=0.5,
            )

            assert len(result) == 0

    @pytest.mark.asyncio
    async def test_skips_self_comparison(self):
        """Should skip comparing memory with itself."""
        mock_mem = MagicMock()
        mock_mem.id = "sem_same"
        mock_mem.content = "Same content"
        mock_mem.embedding = [1.0, 0.0, 0.0]

        mock_embedder = MagicMock()

        # Even with same memory in both lists, should skip
        result = await detect_contradictions(
            new_memories=[mock_mem],
            existing_memories=[mock_mem],
            embedder=mock_embedder,
            user_id="user_123",
        )

        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_skips_failed_analysis(self):
        """Should skip pairs where LLM analysis failed, not treat as 'no conflict'."""
        mock_new_mem = MagicMock()
        mock_new_mem.id = "sem_new"
        mock_new_mem.content = "User prefers PostgreSQL"
        mock_new_mem.embedding = [0.9, 0.1, 0.0]

        mock_existing_mem = MagicMock()
        mock_existing_mem.id = "sem_existing"
        mock_existing_mem.content = "User doesn't like SQL databases"
        mock_existing_mem.embedding = [0.85, 0.15, 0.0]

        mock_embedder = MagicMock()

        # Return a FAILED analysis - this should NOT be treated as "no conflict found"
        failed_analysis = ConflictAnalysis(
            status=OperationStatus.FAILED,
            is_conflict=False,
            confidence=0.0,
            explanation="Analysis failed due to LLM error",
            error_message="Rate limit exceeded",
        )

        with patch(
            "engram.service.contradiction.analyze_pair",
            return_value=failed_analysis,
        ):
            with patch(
                "engram.service.helpers.cosine_similarity",
                return_value=0.8,  # Above threshold - would normally trigger analysis
            ):
                result = await detect_contradictions(
                    new_memories=[mock_new_mem],
                    existing_memories=[mock_existing_mem],
                    embedder=mock_embedder,
                    user_id="user_123",
                    similarity_threshold=0.5,
                )

                # No conflicts returned because analysis failed
                # (not because no conflict was found)
                assert len(result) == 0
