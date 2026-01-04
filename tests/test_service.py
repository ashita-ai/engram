"""Tests for EngramService."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from engram.config import Settings
from engram.models import Episode, Fact
from engram.service import EncodeResult, EngramService, RecallResult


class TestEngramServiceCreate:
    """Tests for EngramService.create factory."""

    def test_create_with_default_settings(self):
        """Should create service with default settings."""
        with patch("engram.service.EngramStorage"):
            with patch("engram.service.get_embedder"):
                service = EngramService.create()
                assert service.settings is not None
                assert service.storage is not None
                assert service.embedder is not None
                assert service.pipeline is not None

    def test_create_with_custom_settings(self):
        """Should create service with custom settings."""
        settings = Settings(embedding_provider="fastembed")
        with patch("engram.service.EngramStorage"):
            with patch("engram.service.get_embedder"):
                service = EngramService.create(settings)
                assert service.settings.embedding_provider == "fastembed"


class TestEngramServiceEncode:
    """Tests for EngramService.encode method."""

    @pytest.fixture
    def mock_service(self):
        """Create a service with mocked dependencies."""
        storage = AsyncMock()
        storage.store_episode = AsyncMock(return_value="ep_123")
        storage.store_fact = AsyncMock(return_value="fact_123")

        embedder = AsyncMock()
        embedder.embed = AsyncMock(return_value=[0.1, 0.2, 0.3])

        pipeline = MagicMock()

        settings = Settings(openai_api_key="sk-test-dummy-key")

        return EngramService(
            storage=storage,
            embedder=embedder,
            pipeline=pipeline,
            settings=settings,
        )

    @pytest.mark.asyncio
    async def test_encode_stores_episode(self, mock_service):
        """Should store episode with embedding."""
        # Mock extraction to return no facts
        mock_service.pipeline.run.return_value = []

        result = await mock_service.encode(
            content="Hello world",
            role="user",
            user_id="user_123",
        )

        assert isinstance(result, EncodeResult)
        assert result.episode.content == "Hello world"
        assert result.episode.role == "user"
        assert result.episode.user_id == "user_123"
        assert result.episode.embedding == [0.1, 0.2, 0.3]

        mock_service.storage.store_episode.assert_called_once()

    @pytest.mark.asyncio
    async def test_encode_runs_extraction(self, mock_service):
        """Should run extraction pipeline and store facts."""
        # Create a mock fact
        mock_fact = Fact(
            content="user@example.com",
            category="email",
            source_episode_id="ep_123",
            user_id="user_123",
        )
        mock_service.pipeline.run.return_value = [mock_fact]

        result = await mock_service.encode(
            content="Email me at user@example.com",
            role="user",
            user_id="user_123",
        )

        assert len(result.facts) == 1
        assert result.facts[0].content == "user@example.com"
        mock_service.storage.store_fact.assert_called_once()

    @pytest.mark.asyncio
    async def test_encode_skips_extraction_when_disabled(self, mock_service):
        """Should skip extraction when run_extraction=False."""
        result = await mock_service.encode(
            content="Email me at user@example.com",
            role="user",
            user_id="user_123",
            run_extraction=False,
        )

        assert result.facts == []
        mock_service.pipeline.run.assert_not_called()

    @pytest.mark.asyncio
    async def test_encode_with_optional_fields(self, mock_service):
        """Should pass optional fields to episode."""
        mock_service.pipeline.run.return_value = []

        result = await mock_service.encode(
            content="Hello",
            role="assistant",
            user_id="user_123",
            org_id="org_456",
            session_id="session_789",
            importance=0.8,
        )

        assert result.episode.org_id == "org_456"
        assert result.episode.session_id == "session_789"
        assert result.episode.importance == 0.8


class TestEngramServiceRecall:
    """Tests for EngramService.recall method."""

    @pytest.fixture
    def mock_service(self):
        """Create a service with mocked dependencies."""
        storage = AsyncMock()

        embedder = AsyncMock()
        embedder.embed = AsyncMock(return_value=[0.1, 0.2, 0.3])

        pipeline = MagicMock()

        settings = Settings(openai_api_key="sk-test-dummy-key")

        return EngramService(
            storage=storage,
            embedder=embedder,
            pipeline=pipeline,
            settings=settings,
        )

    @pytest.mark.asyncio
    async def test_recall_searches_episodes(self, mock_service):
        """Should search episodes and return results."""
        mock_episode = Episode(
            content="Hello world",
            role="user",
            user_id="user_123",
            embedding=[0.1, 0.2, 0.3],
        )
        mock_service.storage.search_episodes.return_value = [mock_episode]
        mock_service.storage.search_facts.return_value = []

        results = await mock_service.recall(
            query="hello",
            user_id="user_123",
        )

        assert len(results) >= 1
        episode_results = [r for r in results if r.memory_type == "episode"]
        assert len(episode_results) == 1
        assert episode_results[0].content == "Hello world"

    @pytest.mark.asyncio
    async def test_recall_searches_facts(self, mock_service):
        """Should search facts and return results."""
        mock_fact = Fact(
            content="user@example.com",
            category="email",
            source_episode_id="ep_123",
            user_id="user_123",
            embedding=[0.1, 0.2, 0.3],
        )
        mock_service.storage.search_episodes.return_value = []
        mock_service.storage.search_facts.return_value = [mock_fact]

        results = await mock_service.recall(
            query="email",
            user_id="user_123",
        )

        fact_results = [r for r in results if r.memory_type == "fact"]
        assert len(fact_results) == 1
        assert fact_results[0].content == "user@example.com"
        assert fact_results[0].source_episode_id == "ep_123"

    @pytest.mark.asyncio
    async def test_recall_excludes_episodes_when_disabled(self, mock_service):
        """Should skip episodes when include_episodes=False."""
        mock_service.storage.search_facts.return_value = []

        await mock_service.recall(
            query="hello",
            user_id="user_123",
            include_episodes=False,
        )

        mock_service.storage.search_episodes.assert_not_called()

    @pytest.mark.asyncio
    async def test_recall_excludes_facts_when_disabled(self, mock_service):
        """Should skip facts when include_facts=False."""
        mock_service.storage.search_episodes.return_value = []

        await mock_service.recall(
            query="hello",
            user_id="user_123",
            include_facts=False,
        )

        mock_service.storage.search_facts.assert_not_called()

    @pytest.mark.asyncio
    async def test_recall_passes_filters(self, mock_service):
        """Should pass filters to storage search."""
        mock_service.storage.search_episodes.return_value = []
        mock_service.storage.search_facts.return_value = []

        await mock_service.recall(
            query="hello",
            user_id="user_123",
            org_id="org_456",
            limit=5,
            min_confidence=0.8,
        )

        mock_service.storage.search_facts.assert_called_once()
        call_kwargs = mock_service.storage.search_facts.call_args.kwargs
        assert call_kwargs["user_id"] == "user_123"
        assert call_kwargs["org_id"] == "org_456"
        assert call_kwargs["limit"] == 5
        assert call_kwargs["min_confidence"] == 0.8


class TestRecallResult:
    """Tests for RecallResult model."""

    def test_recall_result_creation(self):
        """Should create RecallResult with all fields."""
        result = RecallResult(
            memory_type="episode",
            content="Hello world",
            score=0.95,
            memory_id="ep_123",
        )

        assert result.memory_type == "episode"
        assert result.content == "Hello world"
        assert result.score == 0.95
        assert result.memory_id == "ep_123"
        assert result.confidence is None
        assert result.source_episode_id is None

    def test_recall_result_with_optional_fields(self):
        """Should create RecallResult with optional fields."""
        result = RecallResult(
            memory_type="fact",
            content="user@example.com",
            score=0.9,
            memory_id="fact_123",
            confidence=0.85,
            source_episode_id="ep_456",
            metadata={"category": "email"},
        )

        assert result.confidence == 0.85
        assert result.source_episode_id == "ep_456"
        assert result.metadata == {"category": "email"}


class TestEncodeResult:
    """Tests for EncodeResult model."""

    def test_encode_result_creation(self):
        """Should create EncodeResult with episode and facts."""
        episode = Episode(
            content="Hello",
            role="user",
            user_id="user_123",
        )

        result = EncodeResult(episode=episode)
        assert result.episode == episode
        assert result.facts == []

    def test_encode_result_with_facts(self):
        """Should create EncodeResult with facts list."""
        episode = Episode(
            content="Email me at user@example.com",
            role="user",
            user_id="user_123",
        )
        fact = Fact(
            content="user@example.com",
            category="email",
            source_episode_id=episode.id,
            user_id="user_123",
        )

        result = EncodeResult(episode=episode, facts=[fact])
        assert len(result.facts) == 1
        assert result.facts[0].content == "user@example.com"
