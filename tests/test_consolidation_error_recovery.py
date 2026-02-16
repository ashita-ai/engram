"""Tests for consolidation error recovery (SPEC-006).

Verifies that:
1. asyncio.gather uses return_exceptions=True so partial chunk failures
   don't kill the entire batch
2. Empty LLM results leave episodes unmarked for retry
3. Only episodes from successful chunks are marked as summarized
4. Structured consolidation chunks large batches
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from engram.workflows.consolidation import (
    MAX_EPISODES_PER_CHUNK,
    SummaryOutput,
    run_consolidation,
    run_consolidation_from_structured,
)


def _make_mock_episodes(count: int, role: str = "user") -> list[MagicMock]:
    """Create mock Episode objects."""
    from engram.models import Episode

    episodes = []
    for i in range(count):
        ep = MagicMock(spec=Episode)
        ep.id = f"ep_{i}"
        ep.role = role
        ep.content = f"Message {i} content"
        episodes.append(ep)
    return episodes


def _make_mock_storage() -> AsyncMock:
    """Create a mock storage with standard method stubs."""
    storage = AsyncMock()
    storage.store_semantic = AsyncMock(return_value="sem_new")
    storage.mark_episodes_summarized = AsyncMock(return_value=0)
    storage.list_semantic_memories = AsyncMock(return_value=[])
    storage.search_semantic = AsyncMock(return_value=[])
    storage.update_semantic_memory = AsyncMock(return_value=True)
    return storage


def _make_mock_embedder() -> AsyncMock:
    """Create a mock embedder."""
    embedder = AsyncMock()
    embedder.embed = AsyncMock(return_value=[0.1] * 384)
    return embedder


def _make_summary(
    summary: str = "Test summary",
    key_facts: list[str] | None = None,
    keywords: list[str] | None = None,
) -> SummaryOutput:
    """Create a SummaryOutput for mocking."""
    return SummaryOutput(
        summary=summary,
        key_facts=key_facts or ["fact one", "fact two"],
        keywords=keywords or ["test"],
        context="test context",
        confidence=0.7,
        confidence_reasoning="test reasoning",
    )


class TestPartialChunkFailure:
    """Tests for asyncio.gather with return_exceptions=True."""

    @pytest.mark.asyncio
    async def test_partial_chunk_failure_creates_memory_from_successes(self) -> None:
        """When one chunk fails, semantic memory should be created from the rest."""
        # Create enough episodes for 2 chunks
        episodes = _make_mock_episodes(MAX_EPISODES_PER_CHUNK + 5)
        storage = _make_mock_storage()
        storage.get_unsummarized_episodes = AsyncMock(return_value=episodes)
        embedder = _make_mock_embedder()

        call_count = 0

        async def _mock_summarize(chunk: list[dict[str, str]], ctx: str) -> SummaryOutput:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("LLM timeout on chunk 2")
            return _make_summary()

        with patch(
            "engram.workflows.consolidation._summarize_chunk",
            side_effect=_mock_summarize,
        ):
            result = await run_consolidation(
                storage=storage,
                embedder=embedder,
                user_id="test_user",
                org_id="test_org",
            )

        # Should still create a semantic memory from the successful chunk
        assert result.semantic_memories_created == 1
        storage.store_semantic.assert_called_once()

    @pytest.mark.asyncio
    async def test_partial_failure_only_marks_successful_chunk_episodes(self) -> None:
        """Only episodes from successful chunks should be marked as summarized."""
        # Create exactly 2 chunks worth of episodes
        chunk_size = MAX_EPISODES_PER_CHUNK
        episodes = _make_mock_episodes(chunk_size * 2)
        storage = _make_mock_storage()
        storage.get_unsummarized_episodes = AsyncMock(return_value=episodes)
        embedder = _make_mock_embedder()

        call_count = 0

        async def _mock_summarize(chunk: list[dict[str, str]], ctx: str) -> SummaryOutput:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("LLM timeout on chunk 2")
            return _make_summary()

        with patch(
            "engram.workflows.consolidation._summarize_chunk",
            side_effect=_mock_summarize,
        ):
            await run_consolidation(
                storage=storage,
                embedder=embedder,
                user_id="test_user",
                org_id="test_org",
            )

        # Only episodes from chunk 0 should be marked
        mark_call = storage.mark_episodes_summarized.call_args
        marked_ids = mark_call[0][0]
        # Chunk 0 has the first MAX_EPISODES_PER_CHUNK episodes
        expected_ids = [f"ep_{i}" for i in range(chunk_size)]
        assert set(marked_ids) == set(expected_ids)

    @pytest.mark.asyncio
    async def test_all_chunks_fail_marks_nothing(self) -> None:
        """When all chunks fail, no episodes should be marked as summarized."""
        episodes = _make_mock_episodes(MAX_EPISODES_PER_CHUNK + 5)
        storage = _make_mock_storage()
        storage.get_unsummarized_episodes = AsyncMock(return_value=episodes)
        embedder = _make_mock_embedder()

        async def _always_fail(chunk: list[dict[str, str]], ctx: str) -> SummaryOutput:
            raise RuntimeError("LLM is completely down")

        with patch(
            "engram.workflows.consolidation._summarize_chunk",
            side_effect=_always_fail,
        ):
            result = await run_consolidation(
                storage=storage,
                embedder=embedder,
                user_id="test_user",
                org_id="test_org",
            )

        # No semantic memory created, no episodes marked
        assert result.semantic_memories_created == 0
        storage.store_semantic.assert_not_called()
        storage.mark_episodes_summarized.assert_not_called()

    @pytest.mark.asyncio
    async def test_single_chunk_failure_is_total_failure(self) -> None:
        """When there's only 1 chunk and it fails, nothing is created."""
        # Small batch = 1 chunk
        episodes = _make_mock_episodes(5)
        storage = _make_mock_storage()
        storage.get_unsummarized_episodes = AsyncMock(return_value=episodes)
        embedder = _make_mock_embedder()

        async def _fail(chunk: list[dict[str, str]], ctx: str) -> SummaryOutput:
            raise RuntimeError("LLM error")

        with patch(
            "engram.workflows.consolidation._summarize_chunk",
            side_effect=_fail,
        ):
            result = await run_consolidation(
                storage=storage,
                embedder=embedder,
                user_id="test_user",
                org_id="test_org",
            )

        assert result.semantic_memories_created == 0
        storage.mark_episodes_summarized.assert_not_called()


class TestEmptyResultNotMarked:
    """Tests that empty LLM results leave episodes unmarked for retry."""

    @pytest.mark.asyncio
    async def test_empty_summary_leaves_episodes_unmarked(self) -> None:
        """Empty summary should NOT mark episodes as summarized."""
        episodes = _make_mock_episodes(3)
        storage = _make_mock_storage()
        storage.get_unsummarized_episodes = AsyncMock(return_value=episodes)
        embedder = _make_mock_embedder()

        empty_summary = SummaryOutput(
            summary="",
            key_facts=[],
            keywords=[],
            context="",
        )

        with patch(
            "engram.workflows.consolidation._summarize_chunk",
            return_value=empty_summary,
        ):
            result = await run_consolidation(
                storage=storage,
                embedder=embedder,
                user_id="test_user",
                org_id="test_org",
            )

        # No semantic memory created, episodes NOT marked
        assert result.semantic_memories_created == 0
        storage.store_semantic.assert_not_called()
        storage.mark_episodes_summarized.assert_not_called()

    @pytest.mark.asyncio
    async def test_whitespace_only_summary_leaves_episodes_unmarked(self) -> None:
        """Whitespace-only summary should NOT mark episodes as summarized."""
        episodes = _make_mock_episodes(3)
        storage = _make_mock_storage()
        storage.get_unsummarized_episodes = AsyncMock(return_value=episodes)
        embedder = _make_mock_embedder()

        whitespace_summary = SummaryOutput(
            summary="   \n  ",
            key_facts=[],
            keywords=[],
            context="",
        )

        with patch(
            "engram.workflows.consolidation._summarize_chunk",
            return_value=whitespace_summary,
        ):
            result = await run_consolidation(
                storage=storage,
                embedder=embedder,
                user_id="test_user",
                org_id="test_org",
            )

        assert result.semantic_memories_created == 0
        storage.mark_episodes_summarized.assert_not_called()


class TestStructuredConsolidationChunking:
    """Tests for chunking in the structured consolidation path."""

    @pytest.mark.asyncio
    async def test_large_batch_is_chunked(self) -> None:
        """Structured memories exceeding MAX_EPISODES_PER_CHUNK should be chunked."""
        from engram.models import StructuredMemory

        count = MAX_EPISODES_PER_CHUNK + 10
        structured = []
        for i in range(count):
            sm = MagicMock(spec=StructuredMemory)
            sm.id = f"struct_{i}"
            sm.source_episode_id = f"ep_{i}"
            sm.summary = f"Summary {i}"
            sm.keywords = ["test"]
            sm.people = []
            sm.organizations = []
            sm.preferences = []
            sm.negations = []
            structured.append(sm)

        storage = _make_mock_storage()
        storage.get_unconsolidated_structured = AsyncMock(return_value=structured)
        storage.mark_structured_consolidated = AsyncMock()
        embedder = _make_mock_embedder()

        synth_call_count = 0

        async def _mock_synthesize(struct_data: list[dict], ctx: str) -> SummaryOutput:
            nonlocal synth_call_count
            synth_call_count += 1
            return _make_summary(summary=f"Synthesis {synth_call_count}")

        mock_reduce = _make_summary(summary="Reduced summary")

        with (
            patch(
                "engram.workflows.consolidation._synthesize_structured",
                side_effect=_mock_synthesize,
            ),
            patch(
                "engram.workflows.consolidation._reduce_summaries",
                return_value=mock_reduce,
            ),
        ):
            result = await run_consolidation_from_structured(
                storage=storage,
                embedder=embedder,
                user_id="test_user",
                org_id="test_org",
            )

        # Should have chunked into 2 chunks and called synthesize twice
        assert synth_call_count == 2
        assert result.semantic_memories_created == 1

    @pytest.mark.asyncio
    async def test_small_batch_not_chunked(self) -> None:
        """Structured memories within limit should use single call."""
        from engram.models import StructuredMemory

        count = 5  # Well under MAX_EPISODES_PER_CHUNK
        structured = []
        for i in range(count):
            sm = MagicMock(spec=StructuredMemory)
            sm.id = f"struct_{i}"
            sm.source_episode_id = f"ep_{i}"
            sm.summary = f"Summary {i}"
            sm.keywords = ["test"]
            sm.people = []
            sm.organizations = []
            sm.preferences = []
            sm.negations = []
            structured.append(sm)

        storage = _make_mock_storage()
        storage.get_unconsolidated_structured = AsyncMock(return_value=structured)
        storage.mark_structured_consolidated = AsyncMock()
        embedder = _make_mock_embedder()

        synth_call_count = 0

        async def _mock_synthesize(struct_data: list[dict], ctx: str) -> SummaryOutput:
            nonlocal synth_call_count
            synth_call_count += 1
            return _make_summary()

        with patch(
            "engram.workflows.consolidation._synthesize_structured",
            side_effect=_mock_synthesize,
        ):
            result = await run_consolidation_from_structured(
                storage=storage,
                embedder=embedder,
                user_id="test_user",
                org_id="test_org",
            )

        # Single call, no chunking
        assert synth_call_count == 1
        assert result.semantic_memories_created == 1

    @pytest.mark.asyncio
    async def test_structured_all_chunks_fail_returns_empty(self) -> None:
        """When all structured chunks fail, should return empty result."""
        from engram.models import StructuredMemory

        count = MAX_EPISODES_PER_CHUNK + 10
        structured = []
        for i in range(count):
            sm = MagicMock(spec=StructuredMemory)
            sm.id = f"struct_{i}"
            sm.source_episode_id = f"ep_{i}"
            sm.summary = f"Summary {i}"
            sm.keywords = ["test"]
            sm.people = []
            sm.organizations = []
            sm.preferences = []
            sm.negations = []
            structured.append(sm)

        storage = _make_mock_storage()
        storage.get_unconsolidated_structured = AsyncMock(return_value=structured)
        embedder = _make_mock_embedder()

        async def _always_fail(struct_data: list[dict], ctx: str) -> SummaryOutput:
            raise RuntimeError("LLM down")

        with patch(
            "engram.workflows.consolidation._synthesize_structured",
            side_effect=_always_fail,
        ):
            result = await run_consolidation_from_structured(
                storage=storage,
                embedder=embedder,
                user_id="test_user",
                org_id="test_org",
            )

        assert result.semantic_memories_created == 0
        assert result.episodes_processed == 0
        storage.store_semantic.assert_not_called()
