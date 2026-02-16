"""Tests for context-aware reranking.

Tests the weighted signal reranking that combines:
- Similarity: Original vector similarity score
- Recency: Time decay (more recent = higher score)
- Confidence: Memory confidence score
- Session match: Bonus for memories from same session
- Access boost: Bonus for frequently accessed memories
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest

from engram.models import Staleness
from engram.service.models import RecallResult


class TestRerankWeightsConfig:
    """Tests for RerankWeights configuration."""

    def test_default_weights(self):
        """Verify default rerank weight values."""
        from engram.config import RerankWeights

        weights = RerankWeights()
        assert weights.similarity == 0.50
        assert weights.recency == 0.20
        assert weights.confidence == 0.15
        assert weights.session == 0.10
        assert weights.access == 0.05
        assert weights.recency_half_life_hours == 24.0
        assert weights.max_access_boost == 0.1

    def test_weights_sum_to_one(self):
        """Default weights should sum to approximately 1.0."""
        from engram.config import RerankWeights

        weights = RerankWeights()
        total = (
            weights.similarity
            + weights.recency
            + weights.confidence
            + weights.session
            + weights.access
        )
        assert abs(total - 1.0) < 0.01

    def test_custom_weights_validation(self):
        """Invalid weights should emit a UserWarning at construction."""
        import warnings

        from engram.config import RerankWeights

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            RerankWeights(similarity=0.3, recency=0.1, confidence=0.1, session=0.1, access=0.1)
        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warnings) == 1
        assert "RerankWeights sum to 0.700" in str(user_warnings[0].message)


class TestSettingsReranking:
    """Tests for reranking settings."""

    def test_default_rerank_enabled(self):
        """Reranking should be enabled by default."""
        from engram.config import Settings

        settings = Settings()
        assert settings.rerank_enabled is True

    def test_settings_from_env(self, monkeypatch):
        """Settings can be overridden via environment variables."""
        monkeypatch.setenv("ENGRAM_RERANK_ENABLED", "false")
        monkeypatch.setenv("ENGRAM_RERANK_WEIGHTS__SIMILARITY", "0.60")
        monkeypatch.setenv("ENGRAM_RERANK_WEIGHTS__RECENCY", "0.15")

        from engram.config import Settings

        settings = Settings()
        assert settings.rerank_enabled is False
        assert settings.rerank_weights.similarity == 0.60
        assert settings.rerank_weights.recency == 0.15


class TestRecencyScoreCalculation:
    """Tests for recency score calculation."""

    @pytest.fixture
    def mock_recall_mixin(self):
        """Create a mock RecallMixin instance."""
        from engram.config import Settings
        from engram.service.recall import RecallMixin

        mixin = MagicMock(spec=RecallMixin)
        mixin.settings = Settings()
        return mixin

    def test_recent_memory_high_recency(self, mock_recall_mixin):
        """Memories created just now should have high recency score."""
        from engram.service.recall import RecallMixin

        now = datetime.now(UTC)
        result = RecallResult(
            memory_type="episodic",
            content="Test",
            score=0.8,
            memory_id="ep_123",
            metadata={"timestamp": now.isoformat()},
        )

        score = RecallMixin._calculate_recency_score(
            mock_recall_mixin, result, now, half_life_hours=24.0
        )
        # Just created, should be nearly 1.0
        assert score > 0.99

    def test_old_memory_low_recency(self, mock_recall_mixin):
        """Memories from days ago should have low recency score."""
        from engram.service.recall import RecallMixin

        now = datetime.now(UTC)
        old_timestamp = now - timedelta(days=3)  # 72 hours ago
        result = RecallResult(
            memory_type="episodic",
            content="Test",
            score=0.8,
            memory_id="ep_123",
            metadata={"timestamp": old_timestamp.isoformat()},
        )

        score = RecallMixin._calculate_recency_score(
            mock_recall_mixin, result, now, half_life_hours=24.0
        )
        # 72 hours / 24 hour half-life = 3 half-lives = 0.5^3 = 0.125
        assert score < 0.2
        assert score > 0.1

    def test_no_timestamp_moderate_recency(self, mock_recall_mixin):
        """Memories without timestamp should get moderate recency score."""
        from engram.service.recall import RecallMixin

        now = datetime.now(UTC)
        result = RecallResult(
            memory_type="episodic",
            content="Test",
            score=0.8,
            memory_id="ep_123",
            metadata={},  # No timestamp
        )

        score = RecallMixin._calculate_recency_score(
            mock_recall_mixin, result, now, half_life_hours=24.0
        )
        assert score == 0.5


class TestSessionScoreCalculation:
    """Tests for session match score calculation."""

    @pytest.fixture
    def mock_recall_mixin(self):
        """Create a mock RecallMixin instance."""
        from engram.service.recall import RecallMixin

        return MagicMock(spec=RecallMixin)

    def test_matching_session_full_score(self, mock_recall_mixin):
        """Memories from same session should get full score."""
        from engram.service.recall import RecallMixin

        result = RecallResult(
            memory_type="episodic",
            content="Test",
            score=0.8,
            memory_id="ep_123",
            metadata={"session_id": "session_abc"},
        )

        score = RecallMixin._calculate_session_score(
            mock_recall_mixin, result, current_session_id="session_abc"
        )
        assert score == 1.0

    def test_different_session_no_score(self, mock_recall_mixin):
        """Memories from different session should get no score."""
        from engram.service.recall import RecallMixin

        result = RecallResult(
            memory_type="episodic",
            content="Test",
            score=0.8,
            memory_id="ep_123",
            metadata={"session_id": "session_abc"},
        )

        score = RecallMixin._calculate_session_score(
            mock_recall_mixin, result, current_session_id="session_xyz"
        )
        assert score == 0.0

    def test_no_current_session_no_score(self, mock_recall_mixin):
        """No current session means no session bonus."""
        from engram.service.recall import RecallMixin

        result = RecallResult(
            memory_type="episodic",
            content="Test",
            score=0.8,
            memory_id="ep_123",
            metadata={"session_id": "session_abc"},
        )

        score = RecallMixin._calculate_session_score(
            mock_recall_mixin, result, current_session_id=None
        )
        assert score == 0.0


class TestAccessScoreCalculation:
    """Tests for access frequency score calculation."""

    @pytest.fixture
    def mock_recall_mixin(self):
        """Create a mock RecallMixin instance."""
        from engram.service.recall import RecallMixin

        return MagicMock(spec=RecallMixin)

    def test_no_access_no_boost(self, mock_recall_mixin):
        """Memories never accessed should get no boost."""
        from engram.service.recall import RecallMixin

        result = RecallResult(
            memory_type="structured",
            content="Test",
            score=0.8,
            memory_id="struct_123",
            metadata={"retrieval_count": 0},
        )

        score = RecallMixin._calculate_access_score(mock_recall_mixin, result, max_boost=0.1)
        assert score == 0.0

    def test_high_access_capped_boost(self, mock_recall_mixin):
        """Frequently accessed memories should get capped boost."""
        from engram.service.recall import RecallMixin

        result = RecallResult(
            memory_type="structured",
            content="Test",
            score=0.8,
            memory_id="struct_123",
            metadata={"retrieval_count": 100},
        )

        score = RecallMixin._calculate_access_score(mock_recall_mixin, result, max_boost=0.1)
        # Should be at max boost
        assert score == pytest.approx(0.1, abs=0.01)

    def test_moderate_access_partial_boost(self, mock_recall_mixin):
        """Moderately accessed memories should get partial boost."""
        from engram.service.recall import RecallMixin

        result = RecallResult(
            memory_type="structured",
            content="Test",
            score=0.8,
            memory_id="struct_123",
            metadata={"retrieval_count": 10},
        )

        score = RecallMixin._calculate_access_score(mock_recall_mixin, result, max_boost=0.1)
        # log(11) / log(101) * 0.1 ≈ 0.052
        assert score > 0.04
        assert score < 0.07


class TestContextReranking:
    """Tests for the full context reranking flow."""

    @pytest.fixture
    def recall_mixin_with_settings(self):
        """Create a RecallMixin-like object with real settings and methods."""
        from engram.config import Settings
        from engram.service.recall import RecallMixin

        # Create a class that inherits the reranking methods
        class TestRecallMixin(RecallMixin):
            def __init__(self):
                self.settings = Settings()
                self._working_memory = []

        return TestRecallMixin()

    def test_reranking_combines_signals(self, recall_mixin_with_settings):
        """Reranking should combine all signal weights."""
        now = datetime.now(UTC)
        results = [
            RecallResult(
                memory_type="episodic",
                content="Recent high-confidence",
                score=0.9,
                confidence=0.9,
                memory_id="ep_1",
                staleness=Staleness.FRESH,
                metadata={
                    "timestamp": now.isoformat(),
                    "session_id": "current_session",
                    "retrieval_count": 10,
                },
            ),
            RecallResult(
                memory_type="episodic",
                content="Old low-confidence",
                score=0.9,
                memory_id="ep_2",
                staleness=Staleness.FRESH,
                metadata={
                    "timestamp": (now - timedelta(days=5)).isoformat(),
                    "session_id": "old_session",
                    "retrieval_count": 0,
                },
            ),
        ]

        reranked = recall_mixin_with_settings._apply_context_reranking(
            results, session_id="current_session"
        )

        # First result should have higher score due to recency, session match, and access
        assert reranked[0].score > reranked[1].score

        # Metadata should include rerank signals
        assert "rerank_signals" in reranked[0].metadata
        assert "original_similarity" in reranked[0].metadata
        assert reranked[0].metadata["original_similarity"] == 0.9

    def test_reranking_preserves_content(self, recall_mixin_with_settings):
        """Reranking should preserve memory content and type."""
        results = [
            RecallResult(
                memory_type="semantic",
                content="Test content",
                score=0.8,
                confidence=0.7,
                memory_id="sem_123",
                staleness=Staleness.FRESH,
                metadata={"timestamp": datetime.now(UTC).isoformat()},
            ),
        ]

        reranked = recall_mixin_with_settings._apply_context_reranking(results, session_id=None)

        assert len(reranked) == 1
        assert reranked[0].memory_type == "semantic"
        assert reranked[0].content == "Test content"
        assert reranked[0].memory_id == "sem_123"
