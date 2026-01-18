"""Unit tests for LLM-driven link discovery.

Tests the linking module which provides A-MEM style relationship discovery
beyond simple embedding similarity.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from engram.linking import (
    LINK_TYPE_DESCRIPTIONS,
    DiscoveredLink,
    LinkDiscoveryResult,
    MemoryEvolution,
    discover_links,
    evolve_memory,
)


class TestLinkTypes:
    """Tests for LinkType definitions."""

    def test_all_link_types_have_descriptions(self):
        """Every LinkType should have a human-readable description."""
        # Get all valid link types from the Literal type
        valid_types = [
            "related",
            "causal",
            "temporal",
            "contradicts",
            "elaborates",
            "supersedes",
            "generalizes",
            "exemplifies",
        ]
        for link_type in valid_types:
            assert link_type in LINK_TYPE_DESCRIPTIONS
            assert len(LINK_TYPE_DESCRIPTIONS[link_type]) > 0


class TestDiscoveredLink:
    """Tests for DiscoveredLink model."""

    def test_valid_link(self):
        """Should create a valid link with all fields."""
        link = DiscoveredLink(
            target_id="mem_123",
            link_type="causal",
            confidence=0.85,
            reasoning="Event A caused Event B",
        )
        assert link.target_id == "mem_123"
        assert link.link_type == "causal"
        assert link.confidence == 0.85
        assert link.bidirectional is True  # Default

    def test_link_with_bidirectional_false(self):
        """Should allow non-bidirectional links."""
        link = DiscoveredLink(
            target_id="mem_456",
            link_type="supersedes",
            confidence=0.9,
            reasoning="Memory A is newer than Memory B",
            bidirectional=False,
        )
        assert link.bidirectional is False

    def test_confidence_bounds(self):
        """Confidence should be between 0.0 and 1.0."""
        # Valid bounds
        link_low = DiscoveredLink(
            target_id="id1",
            link_type="related",
            confidence=0.0,
            reasoning="test",
        )
        assert link_low.confidence == 0.0

        link_high = DiscoveredLink(
            target_id="id2",
            link_type="related",
            confidence=1.0,
            reasoning="test",
        )
        assert link_high.confidence == 1.0

        # Invalid bounds
        with pytest.raises(ValueError):
            DiscoveredLink(
                target_id="id3",
                link_type="related",
                confidence=1.5,  # Too high
                reasoning="test",
            )

        with pytest.raises(ValueError):
            DiscoveredLink(
                target_id="id4",
                link_type="related",
                confidence=-0.1,  # Negative
                reasoning="test",
            )

    def test_invalid_link_type_rejected(self):
        """Should reject invalid link types."""
        with pytest.raises(ValueError):
            DiscoveredLink(
                target_id="id5",
                link_type="invalid_type",  # type: ignore[arg-type]
                confidence=0.5,
                reasoning="test",
            )


class TestMemoryEvolution:
    """Tests for MemoryEvolution model."""

    def test_valid_evolution_tags(self):
        """Should create a valid tag evolution."""
        evolution = MemoryEvolution(
            memory_id="mem_789",
            field="tags",
            new_value="project-alpha",
            reason="New memory mentions project alpha",
        )
        assert evolution.field == "tags"
        assert evolution.new_value == "project-alpha"

    def test_valid_evolution_keywords(self):
        """Should create a valid keyword evolution."""
        evolution = MemoryEvolution(
            memory_id="mem_101",
            field="keywords",
            new_value="machine learning",
            reason="Adds ML context",
        )
        assert evolution.field == "keywords"

    def test_valid_evolution_context(self):
        """Should create a valid context evolution."""
        evolution = MemoryEvolution(
            memory_id="mem_102",
            field="context",
            new_value="work",
            reason="Work-related discussion",
        )
        assert evolution.field == "context"

    def test_invalid_field_rejected(self):
        """Should reject invalid field names."""
        with pytest.raises(ValueError):
            MemoryEvolution(
                memory_id="mem_103",
                field="content",  # type: ignore[arg-type]  # Not allowed
                new_value="new content",
                reason="Trying to change content",
            )


class TestLinkDiscoveryResult:
    """Tests for LinkDiscoveryResult model."""

    def test_empty_result(self):
        """Should create an empty result with reasoning."""
        result = LinkDiscoveryResult(reasoning="No meaningful links found")
        assert result.links == []
        assert result.evolutions == []
        assert result.reasoning == "No meaningful links found"

    def test_result_with_links(self):
        """Should create a result with discovered links."""
        result = LinkDiscoveryResult(
            links=[
                DiscoveredLink(
                    target_id="mem_1",
                    link_type="causal",
                    confidence=0.8,
                    reasoning="Cause and effect",
                ),
                DiscoveredLink(
                    target_id="mem_2",
                    link_type="temporal",
                    confidence=0.7,
                    reasoning="Sequential events",
                ),
            ],
            reasoning="Found 2 relationships",
        )
        assert len(result.links) == 2
        assert result.links[0].link_type == "causal"
        assert result.links[1].link_type == "temporal"

    def test_result_with_evolutions(self):
        """Should create a result with evolution suggestions."""
        result = LinkDiscoveryResult(
            evolutions=[
                MemoryEvolution(
                    memory_id="mem_old",
                    field="tags",
                    new_value="updated",
                    reason="New info available",
                )
            ],
            reasoning="Suggested 1 evolution",
        )
        assert len(result.evolutions) == 1


class TestDiscoverLinks:
    """Tests for discover_links function."""

    @pytest.mark.asyncio
    async def test_empty_candidates_returns_empty(self):
        """Should return empty result when no candidates provided."""
        result = await discover_links(
            new_memory_content="Test memory content",
            new_memory_id="new_mem_1",
            candidate_memories=[],
        )
        assert result.links == []
        assert "No candidate memories" in result.reasoning

    @pytest.mark.asyncio
    async def test_discover_links_calls_llm(self):
        """Should call LLM agent with formatted prompt."""
        mock_result = MagicMock()
        mock_result.output = LinkDiscoveryResult(
            links=[
                DiscoveredLink(
                    target_id="candidate_1",
                    link_type="elaborates",
                    confidence=0.75,
                    reasoning="Provides more detail",
                )
            ],
            reasoning="Found elaboration relationship",
        )

        mock_agent_instance = AsyncMock()
        mock_agent_instance.run = AsyncMock(return_value=mock_result)

        with patch("pydantic_ai.Agent", return_value=mock_agent_instance):
            result = await discover_links(
                new_memory_content="User prefers dark mode in all applications",
                new_memory_id="new_mem_2",
                candidate_memories=[
                    {
                        "id": "candidate_1",
                        "content": "User mentioned preferring dark themes",
                        "keywords": ["dark", "theme"],
                        "tags": ["preference"],
                    }
                ],
            )

        assert len(result.links) == 1
        assert result.links[0].link_type == "elaborates"
        mock_agent_instance.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_discover_links_filters_by_confidence(self):
        """Should filter out links below min_confidence threshold."""
        mock_result = MagicMock()
        mock_result.output = LinkDiscoveryResult(
            links=[
                DiscoveredLink(
                    target_id="high_conf",
                    link_type="causal",
                    confidence=0.9,
                    reasoning="Strong causal link",
                ),
                DiscoveredLink(
                    target_id="low_conf",
                    link_type="related",
                    confidence=0.3,  # Below threshold
                    reasoning="Weak relationship",
                ),
            ],
            reasoning="Found 2 potential links",
        )

        mock_agent_instance = AsyncMock()
        mock_agent_instance.run = AsyncMock(return_value=mock_result)

        with patch("pydantic_ai.Agent", return_value=mock_agent_instance):
            result = await discover_links(
                new_memory_content="Test content",
                new_memory_id="new_mem_3",
                candidate_memories=[
                    {"id": "high_conf", "content": "High confidence target"},
                    {"id": "low_conf", "content": "Low confidence target"},
                ],
                min_confidence=0.6,
            )

        # Only high confidence link should remain
        assert len(result.links) == 1
        assert result.links[0].target_id == "high_conf"

    @pytest.mark.asyncio
    async def test_discover_links_handles_llm_error(self):
        """Should return error result when LLM fails."""
        mock_agent_instance = AsyncMock()
        mock_agent_instance.run = AsyncMock(side_effect=Exception("LLM unavailable"))

        with patch("pydantic_ai.Agent", return_value=mock_agent_instance):
            result = await discover_links(
                new_memory_content="Test content",
                new_memory_id="new_mem_4",
                candidate_memories=[{"id": "c1", "content": "Candidate"}],
            )

        assert result.links == []
        assert "failed" in result.reasoning.lower()


class TestEvolveMemory:
    """Tests for evolve_memory function."""

    @pytest.mark.asyncio
    async def test_evolve_memory_with_evolve_method(self):
        """Should call memory's evolve() method."""
        mock_memory = MagicMock()
        mock_memory.evolve = MagicMock()

        evolution = MemoryEvolution(
            memory_id="mem_to_evolve",
            field="tags",
            new_value="new-tag",
            reason="Adding tag",
        )

        result = await evolve_memory(mock_memory, evolution)

        assert result is True
        mock_memory.evolve.assert_called_once_with(
            trigger_memory_id="mem_to_evolve",
            field="tags",
            new_value="new-tag",
            reason="Adding tag",
        )

    @pytest.mark.asyncio
    async def test_evolve_memory_without_evolve_method(self):
        """Should return False for memories without evolve()."""
        mock_memory = MagicMock(spec=[])  # No evolve method

        evolution = MemoryEvolution(
            memory_id="mem_no_evolve",
            field="keywords",
            new_value="keyword",
            reason="Test",
        )

        result = await evolve_memory(mock_memory, evolution)

        assert result is False

    @pytest.mark.asyncio
    async def test_evolve_memory_handles_error(self):
        """Should return False and log when evolve() fails."""
        mock_memory = MagicMock()
        mock_memory.evolve = MagicMock(side_effect=ValueError("Invalid evolution"))

        evolution = MemoryEvolution(
            memory_id="mem_error",
            field="context",
            new_value="value",
            reason="Test",
        )

        result = await evolve_memory(mock_memory, evolution)

        assert result is False
