"""Tests for EngramService."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from engram.config import Settings
from engram.models import Episode, Fact, NegationFact, ProceduralMemory, SemanticMemory
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
        episode_results = [r for r in results if r.memory_type == "episodic"]
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

        fact_results = [r for r in results if r.memory_type == "factual"]
        assert len(fact_results) == 1
        assert fact_results[0].content == "user@example.com"
        assert fact_results[0].source_episode_id == "ep_123"
        assert fact_results[0].score == 0.92  # Actual score from Qdrant

    @pytest.mark.asyncio
    async def test_recall_excludes_episodes_when_not_in_types(self, mock_service):
        """Should skip episodes when not in memory_types."""
        mock_service.storage.search_facts.return_value = []

        await mock_service.recall(
            query="hello",
            user_id="user_123",
            memory_types=["factual"],  # Episode not included
        )

        mock_service.storage.search_episodes.assert_not_called()

    @pytest.mark.asyncio
    async def test_recall_excludes_facts_when_not_in_types(self, mock_service):
        """Should skip facts when not in memory_types."""
        mock_service.storage.search_episodes.return_value = []

        await mock_service.recall(
            query="hello",
            user_id="user_123",
            memory_types=["episodic"],  # Fact not included
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

    @pytest.mark.asyncio
    async def test_recall_searches_procedural(self, mock_service):
        """Should search procedural memories and return results."""
        mock_procedural = ProceduralMemory(
            content="User prefers concise responses",
            user_id="user_123",
            trigger_context="general conversation",
            embedding=[0.1, 0.2, 0.3],
        )
        mock_service.storage.search_episodes.return_value = []
        mock_service.storage.search_facts.return_value = []
        mock_service.storage.search_semantic.return_value = []
        mock_service.storage.search_procedural.return_value = [
            ScoredResult(memory=mock_procedural, score=0.85)
        ]

        results = await mock_service.recall(
            query="response preferences",
            user_id="user_123",
        )

        proc_results = [r for r in results if r.memory_type == "procedural"]
        assert len(proc_results) == 1
        assert proc_results[0].content == "User prefers concise responses"
        assert proc_results[0].score == 0.85
        assert proc_results[0].metadata["trigger_context"] == "general conversation"

    @pytest.mark.asyncio
    async def test_recall_excludes_procedural_when_not_in_types(self, mock_service):
        """Should skip procedural when not in memory_types."""
        mock_service.storage.search_episodes.return_value = []
        mock_service.storage.search_facts.return_value = []
        mock_service.storage.search_semantic.return_value = []

        await mock_service.recall(
            query="hello",
            user_id="user_123",
            memory_types=["episodic", "factual", "semantic"],  # Procedural not included
        )

        mock_service.storage.search_procedural.assert_not_called()

    @pytest.mark.asyncio
    async def test_recall_includes_procedural_metadata(self, mock_service):
        """Should include procedural-specific metadata in results."""
        mock_procedural = ProceduralMemory(
            content="User likes code examples",
            user_id="user_123",
            trigger_context="technical discussion",
            access_count=5,
            embedding=[0.1, 0.2, 0.3],
        )
        mock_service.storage.search_episodes.return_value = []
        mock_service.storage.search_facts.return_value = []
        mock_service.storage.search_semantic.return_value = []
        mock_service.storage.search_procedural.return_value = [
            ScoredResult(memory=mock_procedural, score=0.9)
        ]

        results = await mock_service.recall(
            query="code preferences",
            user_id="user_123",
        )

        proc_results = [r for r in results if r.memory_type == "procedural"]
        assert len(proc_results) == 1
        metadata = proc_results[0].metadata
        assert metadata["trigger_context"] == "technical discussion"
        assert metadata["access_count"] == 5
        assert "derived_at" in metadata

    @pytest.mark.asyncio
    async def test_recall_searches_negation(self, mock_service):
        """Should search negation facts and return results."""
        mock_negation = NegationFact(
            content="User does NOT use MongoDB",
            negates_pattern="mongodb",
            user_id="user_123",
            embedding=[0.1, 0.2, 0.3],
        )
        mock_service.storage.search_episodes.return_value = []
        mock_service.storage.search_facts.return_value = []
        mock_service.storage.search_semantic.return_value = []
        mock_service.storage.search_procedural.return_value = []
        mock_service.storage.search_negation.return_value = [
            ScoredResult(memory=mock_negation, score=0.88)
        ]

        results = await mock_service.recall(
            query="mongodb preferences",
            user_id="user_123",
        )

        neg_results = [r for r in results if r.memory_type == "negation"]
        assert len(neg_results) == 1
        assert neg_results[0].content == "User does NOT use MongoDB"
        assert neg_results[0].score == 0.88
        assert neg_results[0].metadata["negates_pattern"] == "mongodb"

    @pytest.mark.asyncio
    async def test_recall_excludes_negation_when_not_in_types(self, mock_service):
        """Should skip negation when not in memory_types."""
        mock_service.storage.search_episodes.return_value = []
        mock_service.storage.search_facts.return_value = []
        mock_service.storage.search_semantic.return_value = []
        mock_service.storage.search_procedural.return_value = []

        await mock_service.recall(
            query="hello",
            user_id="user_123",
            memory_types=["episodic", "factual"],  # Negation not included
        )

        mock_service.storage.search_negation.assert_not_called()

    @pytest.mark.asyncio
    async def test_recall_includes_negation_metadata(self, mock_service):
        """Should include negation-specific metadata in results."""
        mock_negation = NegationFact(
            content="User does NOT want spam",
            negates_pattern="spam",
            user_id="user_123",
            embedding=[0.1, 0.2, 0.3],
        )
        mock_service.storage.search_episodes.return_value = []
        mock_service.storage.search_facts.return_value = []
        mock_service.storage.search_semantic.return_value = []
        mock_service.storage.search_procedural.return_value = []
        mock_service.storage.search_negation.return_value = [
            ScoredResult(memory=mock_negation, score=0.9)
        ]

        results = await mock_service.recall(
            query="spam preferences",
            user_id="user_123",
        )

        neg_results = [r for r in results if r.memory_type == "negation"]
        assert len(neg_results) == 1
        metadata = neg_results[0].metadata
        assert metadata["negates_pattern"] == "spam"
        assert "derived_at" in metadata

    @pytest.mark.asyncio
    async def test_recall_with_memory_types_array(self, mock_service):
        """Should filter by memory_types array when provided."""
        mock_episode = Episode(
            content="Hello world",
            role="user",
            user_id="user_123",
            embedding=[0.1, 0.2, 0.3],
        )
        mock_fact = Fact(
            content="user@example.com",
            category="email",
            source_episode_id="ep_123",
            user_id="user_123",
            embedding=[0.1, 0.2, 0.3],
        )
        mock_service.storage.search_episodes.return_value = [
            ScoredResult(memory=mock_episode, score=0.85)
        ]
        mock_service.storage.search_facts.return_value = [ScoredResult(memory=mock_fact, score=0.9)]

        # Only request episodes - should not search facts
        results = await mock_service.recall(
            query="hello",
            user_id="user_123",
            memory_types=["episodic"],
        )

        mock_service.storage.search_episodes.assert_called_once()
        mock_service.storage.search_facts.assert_not_called()
        assert len(results) == 1
        assert results[0].memory_type == "episodic"

    @pytest.mark.asyncio
    async def test_recall_memory_types_empty_excludes_all(self, mock_service):
        """Empty memory_types array should exclude all memory types."""
        results = await mock_service.recall(
            query="hello",
            user_id="user_123",
            memory_types=[],  # Empty list
        )

        mock_service.storage.search_episodes.assert_not_called()
        mock_service.storage.search_facts.assert_not_called()
        mock_service.storage.search_semantic.assert_not_called()
        mock_service.storage.search_procedural.assert_not_called()
        mock_service.storage.search_negation.assert_not_called()
        assert len(results) == 0


class TestRecallResult:
    """Tests for RecallResult model."""

    def test_recall_result_creation(self):
        """Should create RecallResult with all fields."""
        result = RecallResult(
            memory_type="episodic",
            content="Hello world",
            score=0.95,
            memory_id="ep_123",
        )

        assert result.memory_type == "episodic"
        assert result.content == "Hello world"
        assert result.score == 0.95
        assert result.memory_id == "ep_123"
        assert result.confidence is None
        assert result.source_episode_id is None

    def test_recall_result_with_optional_fields(self):
        """Should create RecallResult with optional fields."""
        result = RecallResult(
            memory_type="factual",
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
    async def test_recall_excludes_working_when_not_in_types(self, mock_service):
        """Recall should exclude working memory when not in memory_types."""
        await mock_service.encode(content="Test content", role="user", user_id="user_123")

        mock_service.storage.search_episodes.return_value = []
        mock_service.storage.search_facts.return_value = []

        results = await mock_service.recall(
            query="test",
            user_id="user_123",
            memory_types=["episodic", "factual"],  # Working not included
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


class TestGetSources:
    """Tests for EngramService.get_sources method."""

    @pytest.fixture
    def mock_service(self):
        """Create a service with mocked dependencies."""
        storage = AsyncMock()
        embedder = AsyncMock()
        pipeline = MagicMock()
        settings = Settings(openai_api_key="sk-test-dummy-key")

        return EngramService(
            storage=storage,
            embedder=embedder,
            pipeline=pipeline,
            settings=settings,
        )

    @pytest.mark.asyncio
    async def test_get_sources_for_fact(self, mock_service):
        """Should return source episode for a fact."""
        source_episode = Episode(
            id="ep_source_123",
            content="My email is user@example.com",
            role="user",
            user_id="user_123",
        )
        mock_fact = Fact(
            id="fact_abc123",
            content="user@example.com",
            category="email",
            source_episode_id="ep_source_123",
            user_id="user_123",
        )

        mock_service.storage.get_fact.return_value = mock_fact
        mock_service.storage.get_episode.return_value = source_episode

        episodes = await mock_service.get_sources("fact_abc123", "user_123")

        assert len(episodes) == 1
        assert episodes[0].id == "ep_source_123"
        assert episodes[0].content == "My email is user@example.com"
        mock_service.storage.get_fact.assert_called_once_with("fact_abc123", "user_123")

    @pytest.mark.asyncio
    async def test_get_sources_for_semantic(self, mock_service):
        """Should return source episodes for a semantic memory."""
        source_episodes = [
            Episode(id="ep_1", content="First episode", role="user", user_id="user_123"),
            Episode(id="ep_2", content="Second episode", role="user", user_id="user_123"),
        ]
        mock_semantic = SemanticMemory(
            id="sem_xyz789",
            content="User prefers email communication",
            source_episode_ids=["ep_1", "ep_2"],
            user_id="user_123",
        )

        mock_service.storage.get_semantic.return_value = mock_semantic
        mock_service.storage.get_episode.side_effect = source_episodes

        episodes = await mock_service.get_sources("sem_xyz789", "user_123")

        assert len(episodes) == 2
        mock_service.storage.get_semantic.assert_called_once_with("sem_xyz789", "user_123")

    @pytest.mark.asyncio
    async def test_get_sources_for_procedural(self, mock_service):
        """Should return source episodes for a procedural memory."""
        source_episode = Episode(
            id="ep_proc_1", content="Always greet politely", role="user", user_id="user_123"
        )
        mock_procedural = ProceduralMemory(
            id="proc_abc",
            content="When user arrives, say hello",
            source_episode_ids=["ep_proc_1"],
            user_id="user_123",
        )

        mock_service.storage.get_procedural.return_value = mock_procedural
        mock_service.storage.get_episode.return_value = source_episode

        episodes = await mock_service.get_sources("proc_abc", "user_123")

        assert len(episodes) == 1
        mock_service.storage.get_procedural.assert_called_once_with("proc_abc", "user_123")

    @pytest.mark.asyncio
    async def test_get_sources_for_negation(self, mock_service):
        """Should return source episodes for a negation fact."""
        source_episode = Episode(
            id="ep_neg_1", content="I don't like spam", role="user", user_id="user_123"
        )
        mock_negation = NegationFact(
            id="neg_def",
            content="User does not want promotional emails",
            negates_pattern="promotional emails",
            source_episode_ids=["ep_neg_1"],
            user_id="user_123",
        )

        mock_service.storage.get_negation.return_value = mock_negation
        mock_service.storage.get_episode.return_value = source_episode

        episodes = await mock_service.get_sources("neg_def", "user_123")

        assert len(episodes) == 1
        mock_service.storage.get_negation.assert_called_once_with("neg_def", "user_123")

    @pytest.mark.asyncio
    async def test_get_sources_fact_not_found(self, mock_service):
        """Should raise KeyError if fact not found."""
        mock_service.storage.get_fact.return_value = None

        with pytest.raises(KeyError, match="Fact not found"):
            await mock_service.get_sources("fact_nonexistent", "user_123")

    @pytest.mark.asyncio
    async def test_get_sources_semantic_not_found(self, mock_service):
        """Should raise KeyError if semantic memory not found."""
        mock_service.storage.get_semantic.return_value = None

        with pytest.raises(KeyError, match="SemanticMemory not found"):
            await mock_service.get_sources("sem_nonexistent", "user_123")

    @pytest.mark.asyncio
    async def test_get_sources_invalid_prefix(self, mock_service):
        """Should raise ValueError for invalid memory ID prefix."""
        with pytest.raises(ValueError, match="Cannot determine memory type"):
            await mock_service.get_sources("invalid_id", "user_123")

    @pytest.mark.asyncio
    async def test_get_sources_episode_prefix(self, mock_service):
        """Should raise ValueError for episode prefix (not a derived memory)."""
        with pytest.raises(ValueError, match="Cannot determine memory type"):
            await mock_service.get_sources("ep_123", "user_123")

    @pytest.mark.asyncio
    async def test_get_sources_returns_chronological_order(self, mock_service):
        """Should return episodes sorted by timestamp."""
        from datetime import datetime, timedelta

        now = datetime.now()
        older_episode = Episode(
            id="ep_old",
            content="First",
            role="user",
            user_id="user_123",
            timestamp=now - timedelta(hours=1),
        )
        newer_episode = Episode(
            id="ep_new",
            content="Second",
            role="user",
            user_id="user_123",
            timestamp=now,
        )

        mock_semantic = SemanticMemory(
            id="sem_test",
            content="Test semantic",
            source_episode_ids=["ep_new", "ep_old"],  # Out of order
            user_id="user_123",
        )

        mock_service.storage.get_semantic.return_value = mock_semantic
        # Return episodes in wrong order
        mock_service.storage.get_episode.side_effect = [newer_episode, older_episode]

        episodes = await mock_service.get_sources("sem_test", "user_123")

        # Should be sorted chronologically (older first)
        assert len(episodes) == 2
        assert episodes[0].id == "ep_old"
        assert episodes[1].id == "ep_new"
