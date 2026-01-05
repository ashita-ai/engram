"""Tests for consolidation workflow."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from engram.workflows.consolidation import (
    ConsolidationResult,
    ExtractedFact,
    IdentifiedLink,
    LLMExtractionResult,
    format_episodes_for_llm,
    run_consolidation,
)


class TestExtractedFact:
    """Tests for ExtractedFact model."""

    def test_create_with_defaults(self) -> None:
        """Test creating extracted fact with defaults."""
        fact = ExtractedFact(content="User's email is test@example.com")
        assert fact.content == "User's email is test@example.com"
        assert fact.confidence == 0.6
        assert fact.source_context == ""

    def test_create_with_all_fields(self) -> None:
        """Test creating extracted fact with all fields."""
        fact = ExtractedFact(
            content="User prefers dark mode",
            confidence=0.8,
            source_context="When the user said 'I like dark themes'",
        )
        assert fact.content == "User prefers dark mode"
        assert fact.confidence == 0.8
        assert "dark themes" in fact.source_context

    def test_confidence_bounds(self) -> None:
        """Test confidence must be between 0 and 1."""
        with pytest.raises(ValueError):
            ExtractedFact(content="test", confidence=1.5)
        with pytest.raises(ValueError):
            ExtractedFact(content="test", confidence=-0.1)


class TestIdentifiedLink:
    """Tests for IdentifiedLink model."""

    def test_create_link(self) -> None:
        """Test creating a link."""
        link = IdentifiedLink(
            source_content="User's email",
            target_content="Work contact information",
            relationship="is_part_of",
        )
        assert link.source_content == "User's email"
        assert link.target_content == "Work contact information"
        assert link.relationship == "is_part_of"


class TestLLMExtractionResult:
    """Tests for LLMExtractionResult model."""

    def test_create_empty(self) -> None:
        """Test creating empty result."""
        result = LLMExtractionResult()
        assert result.semantic_facts == []
        assert result.links == []
        assert result.contradictions == []

    def test_create_with_data(self) -> None:
        """Test creating result with data."""
        result = LLMExtractionResult(
            semantic_facts=[
                ExtractedFact(content="User works at Acme Corp"),
                ExtractedFact(content="User prefers Python", confidence=0.7),
            ],
            links=[
                IdentifiedLink(
                    source_content="Acme Corp",
                    target_content="work email domain",
                    relationship="determines",
                )
            ],
            contradictions=["User previously said they work at Beta Inc"],
        )
        assert len(result.semantic_facts) == 2
        assert len(result.links) == 1
        assert len(result.contradictions) == 1


class TestConsolidationResult:
    """Tests for ConsolidationResult model."""

    def test_create_result(self) -> None:
        """Test creating consolidation result."""
        result = ConsolidationResult(
            episodes_processed=10,
            semantic_memories_created=5,
            links_created=3,
            contradictions_found=["Conflict A", "Conflict B"],
        )
        assert result.episodes_processed == 10
        assert result.semantic_memories_created == 5
        assert result.links_created == 3
        assert len(result.contradictions_found) == 2

    def test_counts_non_negative(self) -> None:
        """Test counts cannot be negative."""
        with pytest.raises(ValueError):
            ConsolidationResult(
                episodes_processed=-1,
                semantic_memories_created=0,
                links_created=0,
            )


class TestFormatEpisodesForLLM:
    """Tests for format_episodes_for_llm function."""

    def test_format_single_episode(self) -> None:
        """Test formatting a single episode."""
        episodes = [{"id": "ep_123", "role": "user", "content": "Hello, world!"}]
        result = format_episodes_for_llm(episodes)
        assert "[USER] (ep_123)" in result
        assert "Hello, world!" in result

    def test_format_multiple_episodes(self) -> None:
        """Test formatting multiple episodes."""
        episodes = [
            {"id": "ep_1", "role": "user", "content": "What's my email?"},
            {"id": "ep_2", "role": "assistant", "content": "Your email is test@example.com"},
            {"id": "ep_3", "role": "user", "content": "Thanks!"},
        ]
        result = format_episodes_for_llm(episodes)
        assert "[USER] (ep_1)" in result
        assert "[ASSISTANT] (ep_2)" in result
        assert "[USER] (ep_3)" in result
        assert "test@example.com" in result

    def test_format_empty_list(self) -> None:
        """Test formatting empty list."""
        result = format_episodes_for_llm([])
        assert "# Conversation Episodes to Analyze" in result


class TestRunConsolidation:
    """Tests for run_consolidation workflow."""

    @pytest.mark.asyncio
    async def test_no_episodes_returns_empty_result(self) -> None:
        """Test that empty episode list returns zero counts."""
        mock_storage = AsyncMock()
        mock_storage.get_unconsolidated_episodes = AsyncMock(return_value=[])

        mock_embedder = AsyncMock()

        result = await run_consolidation(
            storage=mock_storage,
            embedder=mock_embedder,
            user_id="test_user",
        )

        assert result.episodes_processed == 0
        assert result.semantic_memories_created == 0
        assert result.links_created == 0

    @pytest.mark.asyncio
    async def test_fallback_to_non_durable_agent(self) -> None:
        """Test fallback when durable workflows not initialized."""
        from engram.models import Episode

        # Create mock episode
        mock_episode = MagicMock(spec=Episode)
        mock_episode.id = "ep_123"
        mock_episode.role = "user"
        mock_episode.content = "My email is test@example.com"

        mock_storage = AsyncMock()
        mock_storage.get_unconsolidated_episodes = AsyncMock(return_value=[mock_episode])
        mock_storage.store_semantic = AsyncMock(return_value="sem_123")
        mock_storage.mark_episodes_consolidated = AsyncMock(return_value=1)

        mock_embedder = AsyncMock()
        mock_embedder.embed = AsyncMock(return_value=[0.1] * 384)

        # Mock the LLM response
        mock_llm_result = LLMExtractionResult(
            semantic_facts=[ExtractedFact(content="User's email is test@example.com")],
            links=[],
            contradictions=[],
        )

        mock_agent_result = MagicMock()
        mock_agent_result.output = mock_llm_result

        # Patch get_consolidation_agent to raise RuntimeError (workflows not initialized)
        # and patch the fallback Agent to return our mock result
        with patch(
            "engram.workflows.get_consolidation_agent",
            side_effect=RuntimeError("Workflows not initialized"),
        ):
            with patch("pydantic_ai.Agent") as mock_agent_class:
                mock_agent_instance = AsyncMock()
                mock_agent_instance.run = AsyncMock(return_value=mock_agent_result)
                mock_agent_class.return_value = mock_agent_instance

                result = await run_consolidation(
                    storage=mock_storage,
                    embedder=mock_embedder,
                    user_id="test_user",
                )

        assert result.episodes_processed == 1
        assert result.semantic_memories_created == 1
        mock_storage.store_semantic.assert_called_once()
        mock_storage.mark_episodes_consolidated.assert_called_once()

    @pytest.mark.asyncio
    async def test_uses_durable_agent_when_available(self) -> None:
        """Test that durable agent is used when workflows are initialized."""
        from engram.models import Episode

        mock_episode = MagicMock(spec=Episode)
        mock_episode.id = "ep_456"
        mock_episode.role = "user"
        mock_episode.content = "I prefer Python"

        mock_storage = AsyncMock()
        mock_storage.get_unconsolidated_episodes = AsyncMock(return_value=[mock_episode])
        mock_storage.store_semantic = AsyncMock(return_value="sem_456")
        mock_storage.mark_episodes_consolidated = AsyncMock(return_value=1)

        mock_embedder = AsyncMock()
        mock_embedder.embed = AsyncMock(return_value=[0.1] * 384)

        mock_llm_result = LLMExtractionResult(
            semantic_facts=[ExtractedFact(content="User prefers Python")],
            links=[],
            contradictions=[],
        )

        mock_agent_result = MagicMock()
        mock_agent_result.output = mock_llm_result

        mock_durable_agent = AsyncMock()
        mock_durable_agent.run = AsyncMock(return_value=mock_agent_result)

        with patch(
            "engram.workflows.get_consolidation_agent",
            return_value=mock_durable_agent,
        ):
            result = await run_consolidation(
                storage=mock_storage,
                embedder=mock_embedder,
                user_id="test_user",
            )

        assert result.episodes_processed == 1
        assert result.semantic_memories_created == 1
        mock_durable_agent.run.assert_called_once()
