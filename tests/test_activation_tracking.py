"""Tests for activation tracking race condition fix (SPEC-003).

Verifies that _track_semantic_activation and _track_procedural_activation
read retrieval_count from storage before incrementing, rather than using
stale in-memory values from the search result.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from engram.models import ConfidenceScore, ProceduralMemory, SemanticMemory
from engram.storage.search import ScoredResult, SearchMixin


def _make_semantic_memory(
    memory_id: str = "sem_123",
    user_id: str = "test_user",
    retrieval_count: int = 0,
) -> SemanticMemory:
    """Create a SemanticMemory for testing."""
    return SemanticMemory(
        id=memory_id,
        user_id=user_id,
        content="test fact",
        source_episode_ids=["ep_1"],
        confidence=ConfidenceScore.for_inferred(confidence=0.6),
        retrieval_count=retrieval_count,
    )


def _make_procedural_memory(
    memory_id: str = "proc_123",
    user_id: str = "test_user",
    retrieval_count: int = 0,
) -> ProceduralMemory:
    """Create a ProceduralMemory for testing."""
    return ProceduralMemory(
        id=memory_id,
        user_id=user_id,
        content="test rule",
        source_episode_ids=["ep_1"],
        source_semantic_ids=["sem_1"],
        confidence=ConfidenceScore.for_inferred(confidence=0.6),
        retrieval_count=retrieval_count,
    )


def _make_mixin(scroll_payload: dict[str, object] | None = None) -> tuple[SearchMixin, AsyncMock]:
    """Create a SearchMixin with mocked client.

    Args:
        scroll_payload: Payload returned by scroll for _get_current_retrieval_count.
            If None, scroll returns empty results (point not found).
    """
    mixin = SearchMixin()
    mixin._collection_name = MagicMock(side_effect=lambda t: f"engram_{t}")

    mock_point = MagicMock()
    mock_point.payload = scroll_payload

    mock_client = AsyncMock()
    if scroll_payload is not None:
        mock_client.scroll = AsyncMock(return_value=([mock_point], None))
    else:
        mock_client.scroll = AsyncMock(return_value=([], None))
    mock_client.set_payload = AsyncMock()
    mixin.client = mock_client

    return mixin, mock_client


class TestSemanticActivationTracking:
    """Tests for _track_semantic_activation reading from storage."""

    @pytest.mark.asyncio
    async def test_reads_count_from_storage_not_in_memory(self) -> None:
        """Should read retrieval_count from Qdrant, not the stale in-memory value."""
        # In-memory model has count=5 (stale), storage has count=10 (fresh)
        memory = _make_semantic_memory(retrieval_count=5)
        results = [ScoredResult(memory=memory, score=0.9)]

        mixin, mock_client = _make_mixin(scroll_payload={"retrieval_count": 10})

        await mixin._track_semantic_activation(results)

        # Should have written 11 (10 from storage + 1), not 6 (5 from memory + 1)
        set_payload_call = mock_client.set_payload.call_args
        sent_payload = set_payload_call.kwargs.get("payload") or set_payload_call[1].get("payload")
        assert sent_payload["retrieval_count"] == 11

        # In-memory model should also be updated
        assert memory.retrieval_count == 11

    @pytest.mark.asyncio
    async def test_handles_missing_retrieval_count_field(self) -> None:
        """Should default to 0 when retrieval_count is not in the payload."""
        memory = _make_semantic_memory(retrieval_count=5)
        results = [ScoredResult(memory=memory, score=0.9)]

        mixin, mock_client = _make_mixin(scroll_payload={})

        await mixin._track_semantic_activation(results)

        set_payload_call = mock_client.set_payload.call_args
        sent_payload = set_payload_call.kwargs.get("payload") or set_payload_call[1].get("payload")
        assert sent_payload["retrieval_count"] == 1

    @pytest.mark.asyncio
    async def test_handles_point_not_found(self) -> None:
        """Should default to count=0 if the point is not found in storage."""
        memory = _make_semantic_memory(retrieval_count=5)
        results = [ScoredResult(memory=memory, score=0.9)]

        mixin, mock_client = _make_mixin(scroll_payload=None)

        await mixin._track_semantic_activation(results)

        set_payload_call = mock_client.set_payload.call_args
        sent_payload = set_payload_call.kwargs.get("payload") or set_payload_call[1].get("payload")
        assert sent_payload["retrieval_count"] == 1

    @pytest.mark.asyncio
    async def test_updates_last_accessed(self) -> None:
        """Should set last_accessed to current time."""
        memory = _make_semantic_memory()
        assert memory.last_accessed is None  # Default is None
        results = [ScoredResult(memory=memory, score=0.9)]

        mixin, mock_client = _make_mixin(scroll_payload={"retrieval_count": 0})

        await mixin._track_semantic_activation(results)

        set_payload_call = mock_client.set_payload.call_args
        sent_payload = set_payload_call.kwargs.get("payload") or set_payload_call[1].get("payload")
        assert "last_accessed" in sent_payload
        # In-memory model should have been updated from None to a datetime
        assert memory.last_accessed is not None

    @pytest.mark.asyncio
    async def test_tracks_multiple_results(self) -> None:
        """Should track activation for each result independently."""
        mem1 = _make_semantic_memory(memory_id="sem_1", retrieval_count=0)
        mem2 = _make_semantic_memory(memory_id="sem_2", retrieval_count=0)
        results = [
            ScoredResult(memory=mem1, score=0.9),
            ScoredResult(memory=mem2, score=0.8),
        ]

        mixin, mock_client = _make_mixin(scroll_payload={"retrieval_count": 3})

        await mixin._track_semantic_activation(results)

        # scroll called once per result (to read count), set_payload once per result
        assert mock_client.scroll.call_count == 2
        assert mock_client.set_payload.call_count == 2


class TestProceduralActivationTracking:
    """Tests for _track_procedural_activation reading from storage."""

    @pytest.mark.asyncio
    async def test_reads_count_from_storage_not_in_memory(self) -> None:
        """Should read retrieval_count from Qdrant, not the stale in-memory value."""
        memory = _make_procedural_memory(retrieval_count=3)
        results = [ScoredResult(memory=memory, score=0.9)]

        mixin, mock_client = _make_mixin(scroll_payload={"retrieval_count": 7})

        await mixin._track_procedural_activation(results)

        set_payload_call = mock_client.set_payload.call_args
        sent_payload = set_payload_call.kwargs.get("payload") or set_payload_call[1].get("payload")
        assert sent_payload["retrieval_count"] == 8

        assert memory.retrieval_count == 8

    @pytest.mark.asyncio
    async def test_handles_missing_count(self) -> None:
        """Should default to 0 when retrieval_count is absent."""
        memory = _make_procedural_memory(retrieval_count=99)
        results = [ScoredResult(memory=memory, score=0.9)]

        mixin, mock_client = _make_mixin(scroll_payload={})

        await mixin._track_procedural_activation(results)

        set_payload_call = mock_client.set_payload.call_args
        sent_payload = set_payload_call.kwargs.get("payload") or set_payload_call[1].get("payload")
        assert sent_payload["retrieval_count"] == 1

    @pytest.mark.asyncio
    async def test_handles_point_not_found(self) -> None:
        """Should default to count=0 if the point doesn't exist."""
        memory = _make_procedural_memory(retrieval_count=99)
        results = [ScoredResult(memory=memory, score=0.9)]

        mixin, mock_client = _make_mixin(scroll_payload=None)

        await mixin._track_procedural_activation(results)

        set_payload_call = mock_client.set_payload.call_args
        sent_payload = set_payload_call.kwargs.get("payload") or set_payload_call[1].get("payload")
        assert sent_payload["retrieval_count"] == 1


class TestGetCurrentRetrievalCount:
    """Direct tests for the _get_current_retrieval_count helper."""

    @pytest.mark.asyncio
    async def test_returns_stored_count(self) -> None:
        """Should return the integer retrieval_count from storage."""
        mixin, _ = _make_mixin(scroll_payload={"retrieval_count": 42})

        count = await mixin._get_current_retrieval_count("engram_semantic", "sem_1", "user_1")
        assert count == 42

    @pytest.mark.asyncio
    async def test_returns_zero_for_missing_field(self) -> None:
        """Should return 0 when the field is absent."""
        mixin, _ = _make_mixin(scroll_payload={})

        count = await mixin._get_current_retrieval_count("engram_semantic", "sem_1", "user_1")
        assert count == 0

    @pytest.mark.asyncio
    async def test_returns_zero_for_missing_point(self) -> None:
        """Should return 0 when the point doesn't exist."""
        mixin, _ = _make_mixin(scroll_payload=None)

        count = await mixin._get_current_retrieval_count("engram_semantic", "sem_1", "user_1")
        assert count == 0

    @pytest.mark.asyncio
    async def test_coerces_float_to_int(self) -> None:
        """Should handle float values (e.g., from JSON deserialization)."""
        mixin, _ = _make_mixin(scroll_payload={"retrieval_count": 5.0})

        count = await mixin._get_current_retrieval_count("engram_semantic", "sem_1", "user_1")
        assert count == 5
        assert isinstance(count, int)

    @pytest.mark.asyncio
    async def test_returns_zero_for_non_numeric(self) -> None:
        """Should return 0 for unexpected types (defensive)."""
        mixin, _ = _make_mixin(scroll_payload={"retrieval_count": "not_a_number"})

        count = await mixin._get_current_retrieval_count("engram_semantic", "sem_1", "user_1")
        assert count == 0
