"""Tests for query expansion module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from engram.service.query_expansion import (
    ExpandedQuery,
    expand_query,
    get_combined_embedding,
    get_expansion_agent,
)


class TestExpandedQuery:
    """Tests for ExpandedQuery model."""

    def test_create_with_defaults(self):
        """Should create with minimal fields."""
        expanded = ExpandedQuery(original="hello")
        assert expanded.original == "hello"
        assert expanded.expanded_terms == []
        assert expanded.reasoning == ""

    def test_create_with_all_fields(self):
        """Should create with all fields."""
        expanded = ExpandedQuery(
            original="email",
            expanded_terms=["contact", "address", "mail"],
            reasoning="Related contact methods",
        )
        assert expanded.original == "email"
        assert len(expanded.expanded_terms) == 3
        assert "contact" in expanded.expanded_terms
        assert expanded.reasoning == "Related contact methods"

    def test_max_terms_limit(self):
        """Should enforce max 5 expanded terms via validation."""
        # Model allows up to 5 terms
        expanded = ExpandedQuery(
            original="test",
            expanded_terms=["a", "b", "c", "d", "e"],
        )
        assert len(expanded.expanded_terms) == 5


class TestGetExpansionAgent:
    """Tests for expansion agent creation."""

    def test_creates_agent(self, monkeypatch):
        """Should create expansion agent with default model."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-for-agent-creation")
        with patch("engram.service.query_expansion._expansion_agent", None):
            agent = get_expansion_agent()
            assert agent is not None

    def test_caches_agent(self, monkeypatch):
        """Should return same agent on repeated calls."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-for-agent-creation")
        with patch("engram.service.query_expansion._expansion_agent", None):
            agent1 = get_expansion_agent()
            agent2 = get_expansion_agent()
            assert agent1 is agent2


class TestExpandQuery:
    """Tests for expand_query function."""

    @pytest.mark.asyncio
    async def test_returns_expanded_query(self):
        """Should return expanded query from LLM."""
        mock_result = MagicMock()
        mock_result.output = ExpandedQuery(
            original="email",
            expanded_terms=["contact", "address"],
            reasoning="Related terms",
        )

        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value=mock_result)

        with patch(
            "engram.service.query_expansion.get_expansion_agent",
            return_value=mock_agent,
        ):
            result = await expand_query("email")
            assert result.original == "email"
            assert "contact" in result.expanded_terms
            assert "address" in result.expanded_terms

    @pytest.mark.asyncio
    async def test_handles_error_gracefully(self):
        """Should return original query on LLM error."""
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(side_effect=Exception("LLM error"))

        with patch(
            "engram.service.query_expansion.get_expansion_agent",
            return_value=mock_agent,
        ):
            result = await expand_query("email")
            assert result.original == "email"
            assert result.expanded_terms == []
            assert "failed" in result.reasoning.lower()


class TestGetCombinedEmbedding:
    """Tests for get_combined_embedding function."""

    @pytest.mark.asyncio
    async def test_returns_original_when_expand_false(self):
        """Should return original embedding when expand=False."""
        mock_embedder = MagicMock()
        mock_embedder.embed = AsyncMock(return_value=[0.1, 0.2, 0.3])

        result = await get_combined_embedding(
            query="email",
            embedder=mock_embedder,
            expand=False,
        )

        assert result == [0.1, 0.2, 0.3]
        mock_embedder.embed.assert_called_once_with("email")

    @pytest.mark.asyncio
    async def test_returns_original_when_no_expansion(self):
        """Should return original embedding when expansion returns no terms."""
        mock_embedder = MagicMock()
        mock_embedder.embed = AsyncMock(return_value=[0.1, 0.2, 0.3])

        with patch(
            "engram.service.query_expansion.expand_query",
            return_value=ExpandedQuery(original="email", expanded_terms=[]),
        ):
            result = await get_combined_embedding(
                query="email",
                embedder=mock_embedder,
                expand=True,
            )

            assert result == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_combines_embeddings_with_weights(self):
        """Should combine embeddings with weighted average."""
        mock_embedder = MagicMock()
        # Original embedding and 2 expanded term embeddings
        mock_embedder.embed_batch = AsyncMock(
            return_value=[
                [1.0, 0.0, 0.0],  # original (weight 2)
                [0.0, 1.0, 0.0],  # expanded1 (weight 1)
                [0.0, 0.0, 1.0],  # expanded2 (weight 1)
            ]
        )

        with patch(
            "engram.service.query_expansion.expand_query",
            return_value=ExpandedQuery(
                original="email",
                expanded_terms=["contact", "address"],
            ),
        ):
            result = await get_combined_embedding(
                query="email",
                embedder=mock_embedder,
                expand=True,
            )

            # Weighted average: (2*[1,0,0] + 1*[0,1,0] + 1*[0,0,1]) / 4
            # = [0.5, 0.25, 0.25]
            assert result[0] == pytest.approx(0.5)
            assert result[1] == pytest.approx(0.25)
            assert result[2] == pytest.approx(0.25)

    @pytest.mark.asyncio
    async def test_embed_batch_called_with_all_terms(self):
        """Should embed original + expanded terms together."""
        mock_embedder = MagicMock()
        mock_embedder.embed_batch = AsyncMock(
            return_value=[
                [0.1, 0.2],
                [0.3, 0.4],
                [0.5, 0.6],
            ]
        )

        with patch(
            "engram.service.query_expansion.expand_query",
            return_value=ExpandedQuery(
                original="email",
                expanded_terms=["contact", "address"],
            ),
        ):
            await get_combined_embedding(
                query="email",
                embedder=mock_embedder,
                expand=True,
            )

            mock_embedder.embed_batch.assert_called_once_with(["email", "contact", "address"])
