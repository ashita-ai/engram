"""Tests for diversity sampling via MMR reranking."""

import pytest

from engram.service.helpers import cosine_similarity, mmr_rerank


class TestMMRRerank:
    """Tests for mmr_rerank function."""

    def test_empty_candidates(self):
        """Should return empty list for empty candidates."""
        result = mmr_rerank([], limit=10)
        assert result == []

    def test_zero_limit(self):
        """Should return empty list for zero limit."""
        candidates = [(0.9, [1.0, 0.0], 0)]
        result = mmr_rerank(candidates, limit=0)
        assert result == []

    def test_no_diversity(self):
        """Should return by relevance when diversity=0."""
        candidates = [
            (0.9, [1.0, 0.0], 0),  # Most relevant
            (0.8, [0.9, 0.1], 1),
            (0.7, [0.8, 0.2], 2),
        ]
        result = mmr_rerank(candidates, limit=3, diversity=0.0)
        assert result == [0, 1, 2]  # Ordered by relevance

    def test_high_diversity_prefers_different_embeddings(self):
        """Should prefer diverse embeddings with high diversity."""
        # Two very similar embeddings and one different
        candidates = [
            (0.9, [1.0, 0.0], 0),  # Most relevant
            (0.85, [0.99, 0.01], 1),  # Very similar to 0
            (0.8, [0.0, 1.0], 2),  # Very different
        ]
        result = mmr_rerank(candidates, limit=2, diversity=0.5)

        # First should still be most relevant
        assert result[0] == 0
        # Second should be the diverse one, not the similar one
        assert result[1] == 2

    def test_moderate_diversity(self):
        """Should balance relevance and diversity with moderate setting."""
        candidates = [
            (0.9, [1.0, 0.0, 0.0], 0),
            (0.85, [0.9, 0.1, 0.0], 1),
            (0.8, [0.0, 1.0, 0.0], 2),
            (0.75, [0.0, 0.0, 1.0], 3),
        ]
        result = mmr_rerank(candidates, limit=3, diversity=0.3)

        # First should be most relevant
        assert result[0] == 0
        # Results should include diverse options
        assert len(result) == 3

    def test_limit_respected(self):
        """Should respect limit parameter."""
        candidates = [
            (0.9, [1.0, 0.0], 0),
            (0.8, [0.9, 0.1], 1),
            (0.7, [0.8, 0.2], 2),
            (0.6, [0.7, 0.3], 3),
        ]
        result = mmr_rerank(candidates, limit=2, diversity=0.3)
        assert len(result) == 2

    def test_fewer_candidates_than_limit(self):
        """Should return all candidates when fewer than limit."""
        candidates = [
            (0.9, [1.0, 0.0], 0),
            (0.8, [0.9, 0.1], 1),
        ]
        result = mmr_rerank(candidates, limit=10, diversity=0.3)
        assert len(result) == 2

    def test_original_indices_returned(self):
        """Should return original indices, not internal indices."""
        candidates = [
            (0.7, [0.7, 0.3], 5),  # Original index 5
            (0.9, [1.0, 0.0], 3),  # Original index 3
            (0.8, [0.9, 0.1], 7),  # Original index 7
        ]
        result = mmr_rerank(candidates, limit=3, diversity=0.0)

        # Ordered by relevance, should return original indices
        assert result[0] == 3  # Highest relevance
        assert result[1] == 7
        assert result[2] == 5


class TestCosineSimilarity:
    """Tests for cosine_similarity function."""

    def test_identical_vectors(self):
        """Should return 1.0 for identical vectors."""
        vec = [1.0, 2.0, 3.0]
        result = cosine_similarity(vec, vec)
        assert result == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        """Should return 0.0 for orthogonal vectors."""
        vec1 = [1.0, 0.0]
        vec2 = [0.0, 1.0]
        result = cosine_similarity(vec1, vec2)
        assert result == pytest.approx(0.0)

    def test_similar_vectors(self):
        """Should return high similarity for similar vectors."""
        vec1 = [1.0, 0.1]
        vec2 = [1.0, 0.2]
        result = cosine_similarity(vec1, vec2)
        assert result > 0.9

    def test_zero_vector(self):
        """Should return 0.0 when one vector is zero."""
        vec1 = [0.0, 0.0]
        vec2 = [1.0, 2.0]
        result = cosine_similarity(vec1, vec2)
        assert result == 0.0


class TestDiversitySamplingIntegration:
    """Integration tests for diversity sampling in recall."""

    def test_diversity_parameter_in_schema(self):
        """Diversity parameter should be in RecallRequest schema."""
        from engram.api.schemas import RecallRequest

        # Create request with diversity
        request = RecallRequest(
            query="test query",
            user_id="user_123",
            diversity=0.3,
        )
        assert request.diversity == 0.3

    def test_diversity_default_zero(self):
        """Diversity should default to 0.0."""
        from engram.api.schemas import RecallRequest

        request = RecallRequest(
            query="test query",
            user_id="user_123",
        )
        assert request.diversity == 0.0

    def test_diversity_bounds(self):
        """Diversity should be bounded 0.0-1.0."""
        from pydantic import ValidationError

        from engram.api.schemas import RecallRequest

        # Valid values
        RecallRequest(query="test", user_id="u1", diversity=0.0)
        RecallRequest(query="test", user_id="u1", diversity=0.5)
        RecallRequest(query="test", user_id="u1", diversity=1.0)

        # Invalid: below 0
        with pytest.raises(ValidationError):
            RecallRequest(query="test", user_id="u1", diversity=-0.1)

        # Invalid: above 1
        with pytest.raises(ValidationError):
            RecallRequest(query="test", user_id="u1", diversity=1.1)
