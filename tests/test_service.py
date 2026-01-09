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

    @pytest.mark.asyncio
    async def test_encode_high_importance_triggers_consolidation(self, mock_service):
        """Should trigger consolidation when importance >= threshold."""
        mock_service.pipeline.run.return_value = []
        # Set threshold to 0.8 (default)
        mock_service.settings = Settings(
            openai_api_key="sk-test-dummy-key",
            high_importance_threshold=0.8,
        )

        with patch(
            "engram.workflows.consolidation.run_consolidation", new_callable=AsyncMock
        ) as mock_consolidation:
            from engram.workflows.consolidation import ConsolidationResult

            mock_consolidation.return_value = ConsolidationResult(
                episodes_processed=1,
                semantic_memories_created=2,
                links_created=1,
            )

            await mock_service.encode(
                content="Critical information",
                role="user",
                user_id="user_123",
                importance=0.9,  # Above threshold
            )

            mock_consolidation.assert_called_once()

    @pytest.mark.asyncio
    async def test_encode_low_importance_skips_consolidation(self, mock_service):
        """Should NOT trigger consolidation when importance < threshold."""
        mock_service.pipeline.run.return_value = []
        mock_service.settings = Settings(
            openai_api_key="sk-test-dummy-key",
            high_importance_threshold=0.8,
        )

        with patch(
            "engram.workflows.consolidation.run_consolidation", new_callable=AsyncMock
        ) as mock_consolidation:
            await mock_service.encode(
                content="Normal information",
                role="user",
                user_id="user_123",
                importance=0.5,  # Below threshold
            )

            mock_consolidation.assert_not_called()

    @pytest.mark.asyncio
    async def test_encode_at_threshold_triggers_consolidation(self, mock_service):
        """Should trigger consolidation when importance == threshold."""
        mock_service.pipeline.run.return_value = []
        mock_service.settings = Settings(
            openai_api_key="sk-test-dummy-key",
            high_importance_threshold=0.8,
        )

        with patch(
            "engram.workflows.consolidation.run_consolidation", new_callable=AsyncMock
        ) as mock_consolidation:
            from engram.workflows.consolidation import ConsolidationResult

            mock_consolidation.return_value = ConsolidationResult(
                episodes_processed=1,
                semantic_memories_created=1,
                links_created=0,
            )

            await mock_service.encode(
                content="Important information",
                role="user",
                user_id="user_123",
                importance=0.8,  # Exactly at threshold
            )

            mock_consolidation.assert_called_once()

    @pytest.mark.asyncio
    async def test_encode_consolidation_failure_doesnt_fail_encode(self, mock_service):
        """Should not fail encode if consolidation raises exception."""
        mock_service.pipeline.run.return_value = []
        mock_service.settings = Settings(
            openai_api_key="sk-test-dummy-key",
            high_importance_threshold=0.8,
        )

        with patch(
            "engram.workflows.consolidation.run_consolidation", new_callable=AsyncMock
        ) as mock_consolidation:
            mock_consolidation.side_effect = Exception("Consolidation failed")

            # Should not raise, encode should succeed
            result = await mock_service.encode(
                content="Critical information",
                role="user",
                user_id="user_123",
                importance=0.9,
            )

            assert result.episode.content == "Critical information"


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

    @pytest.mark.asyncio
    async def test_recall_follow_links_traverses_related_ids(self, mock_service):
        """Should traverse related_ids when follow_links=True."""
        # Create two linked semantic memories
        mock_semantic1 = SemanticMemory(
            content="User prefers Python",
            user_id="user_123",
            embedding=[0.1, 0.2, 0.3],
        )
        mock_semantic1.id = "sem_001"
        mock_semantic1.add_link("sem_002")  # Link to second memory

        mock_semantic2 = SemanticMemory(
            content="User likes type hints",
            user_id="user_123",
            embedding=[0.1, 0.2, 0.3],
        )
        mock_semantic2.id = "sem_002"
        mock_semantic2.add_link("sem_001")  # Bidirectional link

        # Initial search returns only sem_001
        mock_service.storage.search_episodes.return_value = []
        mock_service.storage.search_facts.return_value = []
        mock_service.storage.search_semantic.return_value = [
            ScoredResult(memory=mock_semantic1, score=0.9)
        ]
        mock_service.storage.search_procedural.return_value = []
        mock_service.storage.search_negation.return_value = []

        # Mock get_semantic to return the linked memory
        mock_service.storage.get_semantic = AsyncMock(return_value=mock_semantic2)

        results = await mock_service.recall(
            query="python preferences",
            user_id="user_123",
            follow_links=True,
            max_hops=1,
        )

        # Should have both original and linked memory
        assert len(results) >= 2
        memory_ids = {r.memory_id for r in results}
        assert "sem_001" in memory_ids
        assert "sem_002" in memory_ids

    @pytest.mark.asyncio
    async def test_recall_follow_links_respects_max_hops(self, mock_service):
        """Should limit link traversal depth to max_hops."""
        # Create chain: sem_001 -> sem_002 -> sem_003
        mock_semantic1 = SemanticMemory(
            content="Memory 1",
            user_id="user_123",
            embedding=[0.1, 0.2, 0.3],
        )
        mock_semantic1.id = "sem_001"
        mock_semantic1.add_link("sem_002")

        mock_semantic2 = SemanticMemory(
            content="Memory 2",
            user_id="user_123",
            embedding=[0.1, 0.2, 0.3],
        )
        mock_semantic2.id = "sem_002"
        mock_semantic2.add_link("sem_003")

        mock_semantic3 = SemanticMemory(
            content="Memory 3",
            user_id="user_123",
            embedding=[0.1, 0.2, 0.3],
        )
        mock_semantic3.id = "sem_003"

        mock_service.storage.search_episodes.return_value = []
        mock_service.storage.search_facts.return_value = []
        mock_service.storage.search_semantic.return_value = [
            ScoredResult(memory=mock_semantic1, score=0.9)
        ]
        mock_service.storage.search_procedural.return_value = []
        mock_service.storage.search_negation.return_value = []

        # Mock get_semantic to return memories based on ID
        async def mock_get_semantic(memory_id: str, user_id: str):
            if memory_id == "sem_002":
                return mock_semantic2
            if memory_id == "sem_003":
                return mock_semantic3
            return None

        mock_service.storage.get_semantic = AsyncMock(side_effect=mock_get_semantic)

        # max_hops=1: should get sem_001 and sem_002, but NOT sem_003
        results = await mock_service.recall(
            query="test",
            user_id="user_123",
            follow_links=True,
            max_hops=1,
        )

        memory_ids = {r.memory_id for r in results}
        assert "sem_001" in memory_ids
        assert "sem_002" in memory_ids
        assert "sem_003" not in memory_ids  # Beyond max_hops

    @pytest.mark.asyncio
    async def test_recall_follow_links_disabled_by_default(self, mock_service):
        """Should not traverse links when follow_links=False (default)."""
        mock_semantic = SemanticMemory(
            content="User prefers Python",
            user_id="user_123",
            embedding=[0.1, 0.2, 0.3],
        )
        mock_semantic.id = "sem_001"
        mock_semantic.add_link("sem_002")  # Has a link

        mock_service.storage.search_episodes.return_value = []
        mock_service.storage.search_facts.return_value = []
        mock_service.storage.search_semantic.return_value = [
            ScoredResult(memory=mock_semantic, score=0.9)
        ]
        mock_service.storage.search_procedural.return_value = []
        mock_service.storage.search_negation.return_value = []

        results = await mock_service.recall(
            query="python preferences",
            user_id="user_123",
            # follow_links defaults to False
        )

        # Should only have original memory, not linked
        assert len(results) == 1
        assert results[0].memory_id == "sem_001"
        # get_semantic should not be called for link traversal
        mock_service.storage.get_semantic.assert_not_called()


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


class TestRetrievalInducedForgetting:
    """Tests for Retrieval-Induced Forgetting (RIF).

    Based on Anderson et al. (1994): when memories are retrieved,
    similar but non-retrieved memories are suppressed.
    """

    @pytest.fixture
    def mock_service(self):
        """Create a service with mocked dependencies for RIF testing."""
        storage = AsyncMock()
        storage.search_episodes = AsyncMock(return_value=[])
        storage.search_facts = AsyncMock(return_value=[])
        storage.search_semantic = AsyncMock(return_value=[])
        storage.search_procedural = AsyncMock(return_value=[])
        storage.search_negation = AsyncMock(return_value=[])
        storage.log_audit = AsyncMock()
        storage.get_fact = AsyncMock()
        storage.get_semantic = AsyncMock()
        storage.update_fact = AsyncMock(return_value=True)
        storage.update_semantic_memory = AsyncMock(return_value=True)

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
    async def test_rif_disabled_by_default(self, mock_service):
        """RIF should be disabled by default - no suppression occurs."""
        # Create test facts with different scores
        fact1 = Fact(
            id="fact_1",
            content="Retrieved fact",
            category="test",
            source_episode_id="ep_1",
            user_id="user_123",
        )
        fact1.confidence.value = 0.9

        fact2 = Fact(
            id="fact_2",
            content="Non-retrieved fact",
            category="test",
            source_episode_id="ep_2",
            user_id="user_123",
        )
        fact2.confidence.value = 0.8

        mock_service.storage.search_facts.return_value = [
            ScoredResult(memory=fact1, score=0.9),
            ScoredResult(memory=fact2, score=0.6),
        ]

        # Default recall - no RIF
        await mock_service.recall(
            query="test query",
            user_id="user_123",
            limit=1,  # Only return top result
            memory_types=["factual"],
        )

        # Verify fact2 was NOT suppressed (update_fact not called)
        mock_service.storage.update_fact.assert_not_called()

    @pytest.mark.asyncio
    async def test_rif_enabled_suppresses_competing_facts(self, mock_service):
        """With RIF enabled, similar non-retrieved facts should be suppressed."""
        # Create test facts with different scores
        fact1 = Fact(
            id="fact_1",
            content="Retrieved fact",
            category="test",
            source_episode_id="ep_1",
            user_id="user_123",
        )
        fact1.confidence.value = 0.9

        fact2 = Fact(
            id="fact_2",
            content="Non-retrieved competing fact",
            category="test",
            source_episode_id="ep_2",
            user_id="user_123",
        )
        fact2.confidence.value = 0.8

        mock_service.storage.search_facts.return_value = [
            ScoredResult(memory=fact1, score=0.9),
            ScoredResult(memory=fact2, score=0.6),  # Above threshold
        ]
        mock_service.storage.get_fact.return_value = fact2

        # Recall with RIF enabled
        await mock_service.recall(
            query="test query",
            user_id="user_123",
            limit=1,
            memory_types=["factual"],
            rif_enabled=True,
            rif_threshold=0.5,  # fact2 at 0.6 is above threshold
            rif_decay=0.1,
        )

        # Verify fact2 was suppressed
        mock_service.storage.get_fact.assert_called_with("fact_2", "user_123")
        mock_service.storage.update_fact.assert_called_once()

        # Verify the confidence was decayed
        updated_fact = mock_service.storage.update_fact.call_args[0][0]
        assert updated_fact.confidence.value == pytest.approx(0.7, rel=0.01)

    @pytest.mark.asyncio
    async def test_rif_does_not_suppress_retrieved_memories(self, mock_service):
        """RIF should not suppress memories that were actually retrieved."""
        fact1 = Fact(
            id="fact_1",
            content="Retrieved fact",
            category="test",
            source_episode_id="ep_1",
            user_id="user_123",
        )
        fact1.confidence.value = 0.9

        mock_service.storage.search_facts.return_value = [
            ScoredResult(memory=fact1, score=0.9),
        ]

        await mock_service.recall(
            query="test query",
            user_id="user_123",
            limit=1,
            memory_types=["factual"],
            rif_enabled=True,
        )

        # fact1 was retrieved, so it should NOT be suppressed
        mock_service.storage.update_fact.assert_not_called()

    @pytest.mark.asyncio
    async def test_rif_does_not_suppress_below_threshold(self, mock_service):
        """RIF should not suppress memories below the similarity threshold."""
        fact1 = Fact(
            id="fact_1",
            content="Retrieved fact",
            category="test",
            source_episode_id="ep_1",
            user_id="user_123",
        )
        fact1.confidence.value = 0.9

        fact2 = Fact(
            id="fact_2",
            content="Low similarity fact",
            category="test",
            source_episode_id="ep_2",
            user_id="user_123",
        )
        fact2.confidence.value = 0.8

        mock_service.storage.search_facts.return_value = [
            ScoredResult(memory=fact1, score=0.9),
            ScoredResult(memory=fact2, score=0.3),  # Below threshold
        ]

        await mock_service.recall(
            query="test query",
            user_id="user_123",
            limit=1,
            memory_types=["factual"],
            rif_enabled=True,
            rif_threshold=0.5,  # fact2 at 0.3 is below
        )

        # fact2 is below threshold, so not suppressed
        mock_service.storage.update_fact.assert_not_called()

    @pytest.mark.asyncio
    async def test_rif_does_not_suppress_episodic_memories(self, mock_service):
        """RIF should not suppress episodic memories (ground truth is immutable)."""
        ep1 = Episode(
            id="ep_1",
            content="First episode",
            role="user",
            user_id="user_123",
        )
        ep2 = Episode(
            id="ep_2",
            content="Second episode",
            role="user",
            user_id="user_123",
        )

        mock_service.storage.search_episodes.return_value = [
            ScoredResult(memory=ep1, score=0.9),
            ScoredResult(memory=ep2, score=0.6),
        ]

        await mock_service.recall(
            query="test query",
            user_id="user_123",
            limit=1,
            memory_types=["episodic"],
            rif_enabled=True,
            rif_threshold=0.5,
        )

        # Episodes are immutable ground truth - no updates
        mock_service.storage.update_fact.assert_not_called()
        mock_service.storage.update_semantic_memory.assert_not_called()

    @pytest.mark.asyncio
    async def test_rif_suppresses_semantic_memories(self, mock_service):
        """RIF should suppress competing semantic memories."""
        sem1 = SemanticMemory(
            id="sem_1",
            content="Retrieved semantic",
            source_episode_ids=["ep_1"],
            user_id="user_123",
        )
        sem1.confidence.value = 0.8

        sem2 = SemanticMemory(
            id="sem_2",
            content="Competing semantic",
            source_episode_ids=["ep_2"],
            user_id="user_123",
        )
        sem2.confidence.value = 0.7

        mock_service.storage.search_semantic.return_value = [
            ScoredResult(memory=sem1, score=0.9),
            ScoredResult(memory=sem2, score=0.6),
        ]
        mock_service.storage.get_semantic.return_value = sem2

        await mock_service.recall(
            query="test query",
            user_id="user_123",
            limit=1,
            memory_types=["semantic"],
            rif_enabled=True,
            rif_threshold=0.5,
            rif_decay=0.1,
        )

        # Verify sem2 was suppressed
        mock_service.storage.get_semantic.assert_called_with("sem_2", "user_123")
        mock_service.storage.update_semantic_memory.assert_called_once()

        # Verify the confidence was decayed
        updated_sem = mock_service.storage.update_semantic_memory.call_args[0][0]
        assert updated_sem.confidence.value == pytest.approx(0.6, rel=0.01)

    @pytest.mark.asyncio
    async def test_rif_audit_logging(self, mock_service):
        """RIF suppression count should be logged in audit entry."""
        fact1 = Fact(
            id="fact_1",
            content="Retrieved fact",
            category="test",
            source_episode_id="ep_1",
            user_id="user_123",
        )
        fact1.confidence.value = 0.9

        fact2 = Fact(
            id="fact_2",
            content="Suppressed fact",
            category="test",
            source_episode_id="ep_2",
            user_id="user_123",
        )
        fact2.confidence.value = 0.8

        mock_service.storage.search_facts.return_value = [
            ScoredResult(memory=fact1, score=0.9),
            ScoredResult(memory=fact2, score=0.6),
        ]
        mock_service.storage.get_fact.return_value = fact2

        await mock_service.recall(
            query="test query",
            user_id="user_123",
            limit=1,
            memory_types=["factual"],
            rif_enabled=True,
            rif_threshold=0.5,
        )

        # Check audit entry includes RIF suppression count
        mock_service.storage.log_audit.assert_called_once()
        audit_entry = mock_service.storage.log_audit.call_args[0][0]
        assert audit_entry.details.get("rif_suppressed") == 1

    @pytest.mark.asyncio
    async def test_rif_minimum_confidence_floor(self, mock_service):
        """RIF should not decay confidence below 0.1 floor."""
        fact1 = Fact(
            id="fact_1",
            content="Retrieved fact",
            category="test",
            source_episode_id="ep_1",
            user_id="user_123",
        )
        fact1.confidence.value = 0.9

        fact2 = Fact(
            id="fact_2",
            content="Very low confidence fact",
            category="test",
            source_episode_id="ep_2",
            user_id="user_123",
        )
        fact2.confidence.value = 0.15  # Low confidence

        mock_service.storage.search_facts.return_value = [
            ScoredResult(memory=fact1, score=0.9),
            ScoredResult(memory=fact2, score=0.6),
        ]
        mock_service.storage.get_fact.return_value = fact2

        await mock_service.recall(
            query="test query",
            user_id="user_123",
            limit=1,
            memory_types=["factual"],
            rif_enabled=True,
            rif_threshold=0.5,
            rif_decay=0.1,  # Would bring to 0.05 without floor
        )

        # Verify confidence floored at 0.1
        updated_fact = mock_service.storage.update_fact.call_args[0][0]
        assert updated_fact.confidence.value == 0.1
