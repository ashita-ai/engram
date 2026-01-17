"""Tests for surprise-based importance scoring.

Tests the Adaptive Compression framework (Nagy et al. 2025):
- Novel content gets higher importance
- Redundant content stays lower priority
- Cold start (no existing memories) returns moderate surprise
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from engram.models import StructuredMemory
from engram.storage.search import ScoredResult


class TestSurpriseCalculation:
    """Tests for _calculate_surprise method."""

    @pytest.fixture
    def mock_encode_mixin(self):
        """Create a mock EncodeMixin instance."""
        mixin = MagicMock()
        mixin.storage = MagicMock()
        mixin.settings = MagicMock()
        mixin.settings.surprise_search_limit = 5
        return mixin

    @pytest.mark.asyncio
    async def test_high_similarity_low_surprise(self, mock_encode_mixin):
        """High similarity to existing memories = low surprise."""
        from engram.service.encode import EncodeMixin

        # Mock similar memories with high similarity
        mock_results = [
            ScoredResult(memory=MagicMock(), score=0.95),
            ScoredResult(memory=MagicMock(), score=0.85),
        ]
        mock_encode_mixin.storage.search_episodes = AsyncMock(return_value=mock_results)

        # Call the method
        surprise = await EncodeMixin._calculate_surprise(
            mock_encode_mixin,
            embedding=[0.1, 0.2, 0.3],
            user_id="user_123",
            org_id=None,
        )

        # High similarity (0.95) → low surprise (0.05)
        assert surprise == pytest.approx(0.05, abs=0.01)

    @pytest.mark.asyncio
    async def test_low_similarity_high_surprise(self, mock_encode_mixin):
        """Low similarity to existing memories = high surprise."""
        from engram.service.encode import EncodeMixin

        # Mock similar memories with low similarity
        mock_results = [
            ScoredResult(memory=MagicMock(), score=0.2),
            ScoredResult(memory=MagicMock(), score=0.15),
        ]
        mock_encode_mixin.storage.search_episodes = AsyncMock(return_value=mock_results)

        surprise = await EncodeMixin._calculate_surprise(
            mock_encode_mixin,
            embedding=[0.1, 0.2, 0.3],
            user_id="user_123",
            org_id=None,
        )

        # Low similarity (0.2) → high surprise (0.8)
        assert surprise == pytest.approx(0.8, abs=0.01)

    @pytest.mark.asyncio
    async def test_no_memories_moderate_surprise(self, mock_encode_mixin):
        """No existing memories = moderate surprise (cold start)."""
        from engram.service.encode import EncodeMixin

        # No existing memories
        mock_encode_mixin.storage.search_episodes = AsyncMock(return_value=[])

        surprise = await EncodeMixin._calculate_surprise(
            mock_encode_mixin,
            embedding=[0.1, 0.2, 0.3],
            user_id="user_123",
            org_id=None,
        )

        # Cold start → moderate surprise (0.5)
        assert surprise == 0.5

    @pytest.mark.asyncio
    async def test_error_returns_zero_surprise(self, mock_encode_mixin):
        """On error, return zero surprise (don't affect importance)."""
        from engram.service.encode import EncodeMixin

        # Simulate storage error
        mock_encode_mixin.storage.search_episodes = AsyncMock(
            side_effect=Exception("Storage error")
        )

        surprise = await EncodeMixin._calculate_surprise(
            mock_encode_mixin,
            embedding=[0.1, 0.2, 0.3],
            user_id="user_123",
            org_id=None,
        )

        # Error → zero surprise (neutral)
        assert surprise == 0.0


class TestImportanceWithSurprise:
    """Tests for _calculate_importance_with_surprise method."""

    @pytest.fixture
    def mock_encode_mixin(self):
        """Create a mock EncodeMixin instance."""
        mixin = MagicMock()
        mixin.storage = MagicMock()
        mixin.settings = MagicMock()
        mixin.settings.surprise_scoring_enabled = True
        mixin.settings.surprise_weight = 0.15
        mixin.settings.surprise_search_limit = 5
        return mixin

    @pytest.fixture
    def basic_structured(self):
        """Create a basic StructuredMemory for testing."""
        return StructuredMemory(
            source_episode_id="ep_test123",
            user_id="user_123",
            emails=[],
            phones=[],
            urls=[],
            enriched=False,
        )

    @pytest.mark.asyncio
    async def test_surprise_increases_importance(self, mock_encode_mixin, basic_structured):
        """Novel content should increase importance via surprise factor."""
        from engram.service.encode import EncodeMixin

        # Mock _calculate_surprise to return high surprise (0.9) for novel content
        mock_encode_mixin._calculate_surprise = AsyncMock(return_value=0.9)

        importance = await EncodeMixin._calculate_importance_with_surprise(
            mock_encode_mixin,
            content="Hello world",
            role="user",
            structured=basic_structured,
            embedding=[0.1, 0.2, 0.3],
            user_id="user_123",
            org_id=None,
        )

        # Base (0.5) + user role (0.05) + surprise (0.9 * 0.15 = 0.135) = ~0.685
        assert importance > 0.65
        assert importance < 0.75

    @pytest.mark.asyncio
    async def test_redundant_content_lower_importance(self, mock_encode_mixin, basic_structured):
        """Redundant content should have lower importance."""
        from engram.service.encode import EncodeMixin

        # Mock _calculate_surprise to return low surprise (0.05) for redundant content
        mock_encode_mixin._calculate_surprise = AsyncMock(return_value=0.05)

        importance = await EncodeMixin._calculate_importance_with_surprise(
            mock_encode_mixin,
            content="Hello world",
            role="user",
            structured=basic_structured,
            embedding=[0.1, 0.2, 0.3],
            user_id="user_123",
            org_id=None,
        )

        # Base (0.5) + user role (0.05) + surprise (0.05 * 0.15 = 0.0075) = ~0.5575
        assert importance > 0.54
        assert importance < 0.58

    @pytest.mark.asyncio
    async def test_surprise_disabled(self, mock_encode_mixin, basic_structured):
        """When surprise is disabled, importance should not include surprise factor."""
        from engram.service.encode import EncodeMixin

        mock_encode_mixin.settings.surprise_scoring_enabled = False

        importance = await EncodeMixin._calculate_importance_with_surprise(
            mock_encode_mixin,
            content="Hello world",
            role="user",
            structured=basic_structured,
            embedding=[0.1, 0.2, 0.3],
            user_id="user_123",
            org_id=None,
        )

        # Base (0.5) + user role (0.05) = 0.55 (no surprise factor)
        assert importance == pytest.approx(0.55, abs=0.01)
        # Verify search wasn't called
        mock_encode_mixin.storage.search_episodes.assert_not_called()


class TestConfigSettings:
    """Tests for surprise scoring configuration."""

    def test_default_settings(self):
        """Verify default configuration values."""
        from engram.config import Settings

        settings = Settings()
        assert settings.surprise_scoring_enabled is True
        assert settings.surprise_weight == 0.15
        assert settings.surprise_search_limit == 5

    def test_settings_from_env(self, monkeypatch):
        """Settings can be overridden via environment variables."""
        monkeypatch.setenv("ENGRAM_SURPRISE_SCORING_ENABLED", "false")
        monkeypatch.setenv("ENGRAM_SURPRISE_WEIGHT", "0.20")
        monkeypatch.setenv("ENGRAM_SURPRISE_SEARCH_LIMIT", "10")

        from engram.config import Settings

        settings = Settings()
        assert settings.surprise_scoring_enabled is False
        assert settings.surprise_weight == 0.20
        assert settings.surprise_search_limit == 10
