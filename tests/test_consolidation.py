"""Tests for consolidation workflow."""

from __future__ import annotations

import pytest

from engram.workflows.consolidation import (
    ConsolidationResult,
    ExtractedFact,
    IdentifiedLink,
    LLMExtractionResult,
    format_episodes_for_llm,
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
