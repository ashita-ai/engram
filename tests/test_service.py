"""Tests for EngramService."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from engram.config import Settings
from engram.models import Episode, Fact
from engram.service import EncodeResult, EngramService, RecallResult
from engram.storage import ScoredResult


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
        embedder.embed_batch = AsyncMock(side_effect=lambda texts: [[0.1, 0.2, 0.3] for _ in texts])

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
        mock_service.storage.search_episodes.return_value = [
            ScoredResult(memory=mock_episode, score=0.85)
        ]
        mock_service.storage.search_facts.return_value = []

        results = await mock_service.recall(
            query="hello",
            user_id="user_123",
        )

        assert len(results) >= 1
        episode_results = [r for r in results if r.memory_type == "episode"]
        assert len(episode_results) == 1
        assert episode_results[0].content == "Hello world"
        assert episode_results[0].score == 0.85  # Actual score from Qdrant

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
        mock_service.storage.search_facts.return_value = [
            ScoredResult(memory=mock_fact, score=0.92)
        ]

        results = await mock_service.recall(
            query="email",
            user_id="user_123",
        )

        fact_results = [r for r in results if r.memory_type == "fact"]
        assert len(fact_results) == 1
        assert fact_results[0].content == "user@example.com"
        assert fact_results[0].source_episode_id == "ep_123"
        assert fact_results[0].score == 0.92  # Actual score from Qdrant

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


class TestWorkingMemory:
    """Tests for working memory functionality."""

    @pytest.fixture
    def mock_service(self):
        """Create a service with mocked dependencies."""
        storage = AsyncMock()
        storage.store_episode = AsyncMock(return_value="ep_123")
        storage.store_fact = AsyncMock(return_value="fact_123")
        storage.log_audit = AsyncMock()

        embedder = AsyncMock()
        embedder.embed = AsyncMock(return_value=[0.1, 0.2, 0.3])
        embedder.embed_batch = AsyncMock(side_effect=lambda texts: [[0.1, 0.2, 0.3] for _ in texts])

        pipeline = MagicMock()
        pipeline.run.return_value = []

        settings = Settings(openai_api_key="sk-test-dummy-key")

        return EngramService(
            storage=storage,
            embedder=embedder,
            pipeline=pipeline,
            settings=settings,
        )

    def test_working_memory_starts_empty(self, mock_service):
        """Working memory should start empty."""
        assert mock_service.get_working_memory() == []

    @pytest.mark.asyncio
    async def test_encode_adds_to_working_memory(self, mock_service):
        """Encode should add episode to working memory."""
        await mock_service.encode(
            content="Hello world",
            role="user",
            user_id="user_123",
        )

        working = mock_service.get_working_memory()
        assert len(working) == 1
        assert working[0].content == "Hello world"
        assert working[0].user_id == "user_123"

    @pytest.mark.asyncio
    async def test_working_memory_accumulates(self, mock_service):
        """Multiple encodes should accumulate in working memory."""
        await mock_service.encode(content="First", role="user", user_id="user_123")
        await mock_service.encode(content="Second", role="user", user_id="user_123")
        await mock_service.encode(content="Third", role="assistant", user_id="user_123")

        working = mock_service.get_working_memory()
        assert len(working) == 3
        assert [ep.content for ep in working] == ["First", "Second", "Third"]

    def test_clear_working_memory(self, mock_service):
        """Clear should remove all episodes from working memory."""
        # Manually add an episode (simulate encode without async)
        episode = Episode(content="Test", role="user", user_id="u1", embedding=[0.1, 0.2])
        mock_service._working_memory.append(episode)

        assert len(mock_service.get_working_memory()) == 1
        mock_service.clear_working_memory()
        assert len(mock_service.get_working_memory()) == 0

    def test_get_working_memory_returns_copy(self, mock_service):
        """get_working_memory should return a copy, not the original list."""
        episode = Episode(content="Test", role="user", user_id="u1", embedding=[0.1, 0.2])
        mock_service._working_memory.append(episode)

        working = mock_service.get_working_memory()
        working.clear()  # Modify the returned list

        # Original should be unaffected
        assert len(mock_service.get_working_memory()) == 1

    @pytest.mark.asyncio
    async def test_close_clears_working_memory(self, mock_service):
        """Close should clear working memory."""
        await mock_service.encode(content="Hello", role="user", user_id="user_123")
        assert len(mock_service.get_working_memory()) == 1

        await mock_service.close()
        assert len(mock_service.get_working_memory()) == 0

    @pytest.mark.asyncio
    async def test_recall_includes_working_memory(self, mock_service):
        """Recall should include working memory results by default."""
        # Add episode to working memory
        await mock_service.encode(content="Test content", role="user", user_id="user_123")

        # Mock storage to return empty results
        mock_service.storage.search_episodes.return_value = []
        mock_service.storage.search_facts.return_value = []

        results = await mock_service.recall(query="test", user_id="user_123")

        # Should have working memory result
        working_results = [r for r in results if r.memory_type == "working"]
        assert len(working_results) == 1
        assert working_results[0].content == "Test content"

    @pytest.mark.asyncio
    async def test_recall_excludes_working_when_disabled(self, mock_service):
        """Recall should exclude working memory when include_working=False."""
        await mock_service.encode(content="Test content", role="user", user_id="user_123")

        mock_service.storage.search_episodes.return_value = []
        mock_service.storage.search_facts.return_value = []

        results = await mock_service.recall(
            query="test",
            user_id="user_123",
            include_working=False,
        )

        working_results = [r for r in results if r.memory_type == "working"]
        assert len(working_results) == 0

    @pytest.mark.asyncio
    async def test_recall_filters_working_by_user(self, mock_service):
        """Recall should only return working memory for the specified user."""
        await mock_service.encode(content="User 1 content", role="user", user_id="user_1")
        await mock_service.encode(content="User 2 content", role="user", user_id="user_2")

        mock_service.storage.search_episodes.return_value = []
        mock_service.storage.search_facts.return_value = []

        results = await mock_service.recall(query="content", user_id="user_1")

        working_results = [r for r in results if r.memory_type == "working"]
        assert len(working_results) == 1
        assert working_results[0].content == "User 1 content"
