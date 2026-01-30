"""Tests for EngramService."""

from unittest.mock import AsyncMock

import pytest
from conftest import add_transaction_support

from engram.config import Settings
from engram.models import (
    Episode,
    ProceduralMemory,
    SemanticMemory,
    StructuredMemory,
)
from engram.service import EncodeResult, EngramService, RecallResult
from engram.service.helpers import calculate_importance
from engram.storage import ScoredResult


class TestEngramServiceCreate:
    """Tests for EngramService.create factory."""

    def test_create_with_default_settings(self):
        """Should create service with default settings."""
        mock_storage = AsyncMock()
        mock_embedder = AsyncMock()
        mock_embedder.dimensions = 384
        mock_backend = AsyncMock()
        settings = Settings()

        # Use model_construct to bypass Pydantic validation for mocks
        service = EngramService.model_construct(
            storage=mock_storage,
            embedder=mock_embedder,
            settings=settings,
            workflow_backend=mock_backend,
            _working_memory=[],
            _conflicts={},
        )
        assert service.settings is not None
        assert service.storage is not None
        assert service.embedder is not None

    def test_create_with_custom_settings(self):
        """Should create service with custom settings."""
        settings = Settings(embedding_provider="fastembed")
        mock_storage = AsyncMock()
        mock_embedder = AsyncMock()
        mock_embedder.dimensions = 384
        mock_backend = AsyncMock()

        # Use model_construct to bypass Pydantic validation for mocks
        service = EngramService.model_construct(
            storage=mock_storage,
            embedder=mock_embedder,
            settings=settings,
            workflow_backend=mock_backend,
            _working_memory=[],
            _conflicts={},
        )
        assert service.settings.embedding_provider == "fastembed"


class TestEngramServiceEncode:
    """Tests for EngramService.encode method."""

    @pytest.fixture
    def mock_service(self):
        """Create a service with mocked dependencies."""
        storage = AsyncMock()
        storage.store_episode = AsyncMock(return_value="ep_123")
        storage.update_episode = AsyncMock()
        storage.store_structured = AsyncMock(return_value="struct_123")
        storage.log_audit = AsyncMock()
        add_transaction_support(storage)

        embedder = AsyncMock()
        embedder.embed = AsyncMock(return_value=[0.1, 0.2, 0.3])
        embedder.embed_batch = AsyncMock(side_effect=lambda texts: [[0.1, 0.2, 0.3] for _ in texts])

        settings = Settings(openai_api_key="sk-test-dummy-key")
        workflow_backend = AsyncMock()

        # Use model_construct to bypass Pydantic validation for mocks
        return EngramService.model_construct(
            storage=storage,
            embedder=embedder,
            settings=settings,
            workflow_backend=workflow_backend,
            _working_memory=[],
            _conflicts={},
        )

    @pytest.mark.asyncio
    async def test_encode_stores_episode(self, mock_service):
        """Should store episode with embedding."""
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
    async def test_encode_creates_structured_memory(self, mock_service):
        """Should create StructuredMemory with regex extracts."""
        result = await mock_service.encode(
            content="Email me at user@example.com",
            role="user",
            user_id="user_123",
        )

        # Check structured memory was created
        assert result.structured is not None
        assert result.structured.source_episode_id == result.episode.id
        assert "user@example.com" in result.structured.emails
        assert result.structured.mode == "fast"
        assert result.structured.enriched is False
        mock_service.storage.store_structured.assert_called_once()

    @pytest.mark.asyncio
    async def test_encode_extracts_multiple_patterns(self, mock_service):
        """Should extract emails, phones, and URLs via regex."""
        result = await mock_service.encode(
            content="Email: test@example.com, Phone: 555-123-4567, URL: https://example.com",
            role="user",
            user_id="user_123",
        )

        assert "test@example.com" in result.structured.emails
        assert any("555" in p for p in result.structured.phones)
        assert "https://example.com" in result.structured.urls

    @pytest.mark.asyncio
    async def test_encode_with_optional_fields(self, mock_service):
        """Should pass optional fields to episode."""
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
        from engram.workflows.consolidation import ConsolidationResult

        # Configure workflow_backend.run_consolidation to return proper result
        mock_service.workflow_backend.run_consolidation = AsyncMock(
            return_value=ConsolidationResult(
                episodes_processed=1,
                semantic_memories_created=2,
                links_created=1,
            )
        )

        await mock_service.encode(
            content="Critical information",
            role="user",
            user_id="user_123",
            importance=0.9,  # Above threshold (default is 0.8)
        )

        mock_service.workflow_backend.run_consolidation.assert_called_once()

    @pytest.mark.asyncio
    async def test_encode_low_importance_skips_consolidation(self, mock_service):
        """Should NOT trigger consolidation when importance < threshold."""
        from engram.workflows.consolidation import ConsolidationResult

        # Configure workflow_backend.run_consolidation but it shouldn't be called
        mock_service.workflow_backend.run_consolidation = AsyncMock(
            return_value=ConsolidationResult(
                episodes_processed=0,
                semantic_memories_created=0,
                links_created=0,
            )
        )

        await mock_service.encode(
            content="Normal information",
            role="user",
            user_id="user_123",
            importance=0.5,  # Below threshold (default is 0.8)
        )

        mock_service.workflow_backend.run_consolidation.assert_not_called()

    @pytest.mark.asyncio
    async def test_encode_at_threshold_triggers_consolidation(self, mock_service):
        """Should trigger consolidation when importance == threshold."""
        from engram.workflows.consolidation import ConsolidationResult

        # Configure workflow_backend.run_consolidation to return proper result
        mock_service.workflow_backend.run_consolidation = AsyncMock(
            return_value=ConsolidationResult(
                episodes_processed=1,
                semantic_memories_created=1,
                links_created=0,
            )
        )

        await mock_service.encode(
            content="Important information",
            role="user",
            user_id="user_123",
            importance=0.8,  # Exactly at threshold (default is 0.8)
        )

        mock_service.workflow_backend.run_consolidation.assert_called_once()

    @pytest.mark.asyncio
    async def test_encode_consolidation_failure_doesnt_fail_encode(self, mock_service):
        """Should not fail encode if consolidation raises exception."""
        # Configure workflow_backend.run_consolidation to raise an exception
        mock_service.workflow_backend.run_consolidation = AsyncMock(
            side_effect=Exception("Consolidation failed")
        )

        # Should not raise, encode should succeed
        result = await mock_service.encode(
            content="Critical information",
            role="user",
            user_id="user_123",
            importance=0.9,  # Above threshold (default is 0.8)
        )

        assert result.episode.content == "Critical information"


class TestEngramServiceRecall:
    """Tests for EngramService.recall method."""

    @pytest.fixture
    def mock_service(self):
        """Create a service with mocked dependencies."""
        storage = AsyncMock()
        # Default all searches to return empty
        storage.search_episodes.return_value = []
        storage.search_structured.return_value = []
        storage.search_semantic.return_value = []
        storage.search_procedural.return_value = []
        storage.list_structured_memories.return_value = []

        embedder = AsyncMock()
        embedder.embed = AsyncMock(return_value=[0.1, 0.2, 0.3])

        settings = Settings(openai_api_key="sk-test-dummy-key")
        workflow_backend = AsyncMock()

        # Use model_construct to bypass Pydantic validation for mocks
        return EngramService.model_construct(
            storage=storage,
            embedder=embedder,
            settings=settings,
            workflow_backend=workflow_backend,
            _working_memory=[],
            _conflicts={},
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

        results = await mock_service.recall(
            query="hello",
            user_id="user_123",
            rerank=False,  # Disable reranking to test raw similarity score
        )

        assert len(results) >= 1
        episode_results = [r for r in results if r.memory_type == "episodic"]
        assert len(episode_results) == 1
        assert episode_results[0].content == "Hello world"
        assert episode_results[0].score == 0.85  # Actual score from Qdrant

    @pytest.mark.asyncio
    async def test_recall_searches_structured(self, mock_service):
        """Should search structured memories and return results."""
        mock_structured = StructuredMemory.from_episode_fast(
            source_episode_id="ep_123",
            user_id="user_123",
            emails=["user@example.com"],
            embedding=[0.1, 0.2, 0.3],
        )
        mock_service.storage.search_structured.return_value = [
            ScoredResult(memory=mock_structured, score=0.92)
        ]

        results = await mock_service.recall(
            query="email",
            user_id="user_123",
            rerank=False,  # Disable reranking to test raw similarity score
        )

        struct_results = [r for r in results if r.memory_type == "structured"]
        assert len(struct_results) == 1
        assert mock_structured.emails[0] in struct_results[0].content
        assert struct_results[0].source_episode_id == "ep_123"
        assert struct_results[0].score == 0.92

    @pytest.mark.asyncio
    async def test_recall_excludes_episodes_when_not_in_types(self, mock_service):
        """Should skip episodes when not in memory_types."""
        await mock_service.recall(
            query="hello",
            user_id="user_123",
            memory_types=["structured"],  # Episode not included
        )

        mock_service.storage.search_episodes.assert_not_called()

    @pytest.mark.asyncio
    async def test_recall_excludes_structured_when_not_in_types(self, mock_service):
        """Should skip structured when not in memory_types."""
        await mock_service.recall(
            query="hello",
            user_id="user_123",
            memory_types=["episodic"],  # Structured not included
        )

        mock_service.storage.search_structured.assert_not_called()

    @pytest.mark.asyncio
    async def test_recall_passes_filters(self, mock_service):
        """Should pass filters to storage search."""
        await mock_service.recall(
            query="hello",
            user_id="user_123",
            org_id="org_456",
            limit=5,
            min_confidence=0.8,
            apply_negation_filter=False,  # Disable to test direct limit passthrough
        )

        mock_service.storage.search_structured.assert_called_once()
        call_kwargs = mock_service.storage.search_structured.call_args.kwargs
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
        mock_service.storage.search_procedural.return_value = [
            ScoredResult(memory=mock_procedural, score=0.85)
        ]

        results = await mock_service.recall(
            query="response preferences",
            user_id="user_123",
            rerank=False,  # Disable reranking to test raw similarity score
        )

        proc_results = [r for r in results if r.memory_type == "procedural"]
        assert len(proc_results) == 1
        assert proc_results[0].content == "User prefers concise responses"
        assert proc_results[0].score == 0.85
        assert proc_results[0].metadata["trigger_context"] == "general conversation"

    @pytest.mark.asyncio
    async def test_recall_excludes_procedural_when_not_in_types(self, mock_service):
        """Should skip procedural when not in memory_types."""
        await mock_service.recall(
            query="hello",
            user_id="user_123",
            memory_types=["episodic", "structured", "semantic"],  # Procedural not included
        )

        mock_service.storage.search_procedural.assert_not_called()

    @pytest.mark.asyncio
    async def test_recall_includes_procedural_metadata(self, mock_service):
        """Should include procedural-specific metadata in results."""
        mock_procedural = ProceduralMemory(
            content="User likes code examples",
            user_id="user_123",
            trigger_context="technical discussion",
            retrieval_count=5,
            embedding=[0.1, 0.2, 0.3],
        )
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
        assert metadata["retrieval_count"] == 5

    @pytest.mark.asyncio
    async def test_recall_with_memory_types_array(self, mock_service):
        """Should filter by memory_types array when provided."""
        mock_episode = Episode(
            content="Hello world",
            role="user",
            user_id="user_123",
            embedding=[0.1, 0.2, 0.3],
        )
        mock_structured = StructuredMemory.from_episode_fast(
            source_episode_id="ep_123",
            user_id="user_123",
            emails=["user@example.com"],
            embedding=[0.1, 0.2, 0.3],
        )
        mock_service.storage.search_episodes.return_value = [
            ScoredResult(memory=mock_episode, score=0.85)
        ]
        mock_service.storage.search_structured.return_value = [
            ScoredResult(memory=mock_structured, score=0.9)
        ]

        # Only request episodes - should not search structured
        results = await mock_service.recall(
            query="hello",
            user_id="user_123",
            memory_types=["episodic"],
        )

        mock_service.storage.search_episodes.assert_called_once()
        mock_service.storage.search_structured.assert_not_called()
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
        mock_service.storage.search_structured.assert_not_called()
        mock_service.storage.search_semantic.assert_not_called()
        mock_service.storage.search_procedural.assert_not_called()
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
        mock_service.storage.search_semantic.return_value = [
            ScoredResult(memory=mock_semantic1, score=0.9)
        ]

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

        mock_service.storage.search_semantic.return_value = [
            ScoredResult(memory=mock_semantic1, score=0.9)
        ]

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

        mock_service.storage.search_semantic.return_value = [
            ScoredResult(memory=mock_semantic, score=0.9)
        ]

        # Mock get_semantic for retrieval strengthening
        mock_service.storage.get_semantic = AsyncMock(return_value=mock_semantic)
        mock_service.storage.update_semantic_memory = AsyncMock(return_value=True)

        results = await mock_service.recall(
            query="python preferences",
            user_id="user_123",
            # follow_links defaults to False
        )

        # Should only have original memory, not linked
        assert len(results) == 1
        assert results[0].memory_id == "sem_001"
        # get_semantic should be called once for retrieval strengthening (sem_001),
        # but NOT for link traversal (sem_002)
        mock_service.storage.get_semantic.assert_called_once_with("sem_001", "user_123")


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
        """Should create EncodeResult with episode and structured memory."""
        episode = Episode(
            content="Hello",
            role="user",
            user_id="user_123",
        )
        structured = StructuredMemory.from_episode_fast(
            source_episode_id=episode.id,
            user_id="user_123",
        )

        result = EncodeResult(episode=episode, structured=structured)
        assert result.episode == episode
        assert result.structured == structured

    def test_encode_result_with_extracts(self):
        """Should create EncodeResult with structured memory containing extracts."""
        episode = Episode(
            content="Email me at user@example.com",
            role="user",
            user_id="user_123",
        )
        structured = StructuredMemory.from_episode_fast(
            source_episode_id=episode.id,
            user_id="user_123",
            emails=["user@example.com"],
        )

        result = EncodeResult(episode=episode, structured=structured)
        assert "user@example.com" in result.structured.emails


class TestWorkingMemory:
    """Tests for working memory functionality."""

    @pytest.fixture
    def mock_service(self):
        """Create a service with mocked dependencies."""
        storage = AsyncMock()
        storage.store_episode = AsyncMock(return_value="ep_123")
        storage.update_episode = AsyncMock()
        storage.store_structured = AsyncMock(return_value="struct_123")
        storage.log_audit = AsyncMock()
        storage.search_episodes.return_value = []
        storage.search_structured.return_value = []
        storage.search_semantic.return_value = []
        storage.search_procedural.return_value = []
        storage.list_structured_memories.return_value = []
        add_transaction_support(storage)

        embedder = AsyncMock()
        embedder.embed = AsyncMock(return_value=[0.1, 0.2, 0.3])
        embedder.embed_batch = AsyncMock(side_effect=lambda texts: [[0.1, 0.2, 0.3] for _ in texts])

        settings = Settings(openai_api_key="sk-test-dummy-key")
        workflow_backend = AsyncMock()

        # Use model_construct to bypass Pydantic validation for mocks
        return EngramService.model_construct(
            storage=storage,
            embedder=embedder,
            settings=settings,
            workflow_backend=workflow_backend,
            _working_memory=[],
            _conflicts={},
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

        results = await mock_service.recall(query="test", user_id="user_123")

        # Should have working memory result
        working_results = [r for r in results if r.memory_type == "working"]
        assert len(working_results) == 1
        assert working_results[0].content == "Test content"

    @pytest.mark.asyncio
    async def test_recall_excludes_working_when_not_in_types(self, mock_service):
        """Recall should exclude working memory when not in memory_types."""
        await mock_service.encode(content="Test content", role="user", user_id="user_123")

        results = await mock_service.recall(
            query="test",
            user_id="user_123",
            memory_types=["episodic", "structured"],  # Working not included
        )

        working_results = [r for r in results if r.memory_type == "working"]
        assert len(working_results) == 0

    @pytest.mark.asyncio
    async def test_recall_filters_working_by_user(self, mock_service):
        """Recall should only return working memory for the specified user."""
        await mock_service.encode(content="User 1 content", role="user", user_id="user_1")
        await mock_service.encode(content="User 2 content", role="user", user_id="user_2")

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
        settings = Settings(openai_api_key="sk-test-dummy-key")
        workflow_backend = AsyncMock()

        # Use model_construct to bypass Pydantic validation for mocks
        return EngramService.model_construct(
            storage=storage,
            embedder=embedder,
            settings=settings,
            workflow_backend=workflow_backend,
            _working_memory=[],
            _conflicts={},
        )

    @pytest.mark.asyncio
    async def test_get_sources_for_structured(self, mock_service):
        """Should return source episode for a structured memory."""
        source_episode = Episode(
            id="ep_source_123",
            content="My email is user@example.com",
            role="user",
            user_id="user_123",
        )
        mock_structured = StructuredMemory.from_episode_fast(
            source_episode_id="ep_source_123",
            user_id="user_123",
            emails=["user@example.com"],
        )
        mock_structured.id = "struct_abc123"

        mock_service.storage.get_structured.return_value = mock_structured
        mock_service.storage.get_episode.return_value = source_episode

        episodes = await mock_service.get_sources("struct_abc123", "user_123")

        assert len(episodes) == 1
        assert episodes[0].id == "ep_source_123"
        assert episodes[0].content == "My email is user@example.com"
        mock_service.storage.get_structured.assert_called_once_with("struct_abc123", "user_123")

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
    async def test_get_sources_structured_not_found(self, mock_service):
        """Should raise NotFoundError if structured memory not found."""
        from engram.exceptions import NotFoundError

        mock_service.storage.get_structured.return_value = None

        with pytest.raises(NotFoundError, match="StructuredMemory not found"):
            await mock_service.get_sources("struct_nonexistent", "user_123")

    @pytest.mark.asyncio
    async def test_get_sources_semantic_not_found(self, mock_service):
        """Should raise NotFoundError if semantic memory not found."""
        from engram.exceptions import NotFoundError

        mock_service.storage.get_semantic.return_value = None

        with pytest.raises(NotFoundError, match="SemanticMemory not found"):
            await mock_service.get_sources("sem_nonexistent", "user_123")

    @pytest.mark.asyncio
    async def test_get_sources_invalid_prefix(self, mock_service):
        """Should raise ValidationError for invalid memory ID prefix."""
        from engram.exceptions import ValidationError

        with pytest.raises(ValidationError, match="Cannot determine memory type"):
            await mock_service.get_sources("invalid_id", "user_123")

    @pytest.mark.asyncio
    async def test_get_sources_episode_prefix(self, mock_service):
        """Should raise ValidationError for episode prefix (not a derived memory)."""
        from engram.exceptions import ValidationError

        with pytest.raises(ValidationError, match="Cannot determine memory type"):
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


class TestImportanceCalculation:
    """Tests for automatic importance calculation."""

    def test_baseline_importance(self):
        """Base importance should be 0.5 with no extra signals."""
        importance = calculate_importance(
            content="Hello world",
            role="assistant",  # assistant doesn't get bonus
        )
        assert importance == 0.5

    def test_user_role_bonus(self):
        """User messages should get +0.05 bonus."""
        importance = calculate_importance(
            content="Hello world",
            role="user",
        )
        assert importance == 0.55

    def test_system_role_penalty(self):
        """System messages should get -0.1 penalty."""
        importance = calculate_importance(
            content="You are a helpful assistant",
            role="system",
        )
        assert importance == 0.4

    def test_regex_extracts_increase_importance(self):
        """Each regex extract should add 0.05 up to 0.15 cap."""
        structured = StructuredMemory.from_episode_fast(
            source_episode_id="ep_1",
            user_id="u1",
            emails=["test@example.com"],
        )
        importance = calculate_importance(
            content="My email is test@example.com",
            role="assistant",
            structured=structured,
        )
        assert importance == 0.55  # 0.5 + 0.05

        # Multiple extracts
        structured = StructuredMemory.from_episode_fast(
            source_episode_id="ep_1",
            user_id="u1",
            emails=["a@b.com", "c@d.com"],
            phones=["555-1234", "555-5678"],
            urls=["https://example.com"],
        )
        importance = calculate_importance(
            content="Lots of contact info here",
            role="assistant",
            structured=structured,
        )
        assert importance == 0.65  # 0.5 + 0.15 (capped at 5 extracts)

    def test_importance_keywords_increase_importance(self):
        """Importance keywords should add 0.05 each up to 0.1 cap."""
        importance = calculate_importance(
            content="Remember this is important",
            role="assistant",
        )
        assert importance == 0.6  # 0.5 + 0.1 (2 keywords: remember, important)

        # Single keyword
        importance = calculate_importance(
            content="This is critical information",
            role="assistant",
        )
        assert importance == 0.55  # 0.5 + 0.05 (1 keyword: critical)

    def test_combined_signals(self):
        """Multiple signals should combine correctly."""
        structured = StructuredMemory.from_episode_fast(
            source_episode_id="ep_1",
            user_id="u1",
            emails=["test@example.com"],
        )
        importance = calculate_importance(
            content="Remember my email is test@example.com",
            role="user",
            structured=structured,
        )
        # 0.5 base + 0.05 (user) + 0.05 (1 email) + 0.05 (1 keyword: remember) = 0.65
        assert importance == pytest.approx(0.65)

    def test_importance_capped_at_one(self):
        """Importance should never exceed 1.0."""
        # Create many signals
        structured = StructuredMemory.from_episode_fast(
            source_episode_id="ep_1",
            user_id="u1",
            emails=["a@b.com", "c@d.com", "e@f.com", "g@h.com", "i@j.com"],
            phones=["555-1234", "555-5678", "555-9012"],
            urls=["https://example.com", "https://test.com"],
        )
        importance = calculate_importance(
            content="Remember this is important and critical and urgent and essential",
            role="user",
            structured=structured,
        )
        assert importance == 0.8  # 0.5 + 0.15 (extracts) + 0.05 (user) + 0.1 (keywords)

    def test_importance_floored_at_zero(self):
        """Importance should never go below 0.0."""
        # System role with negative base
        importance = calculate_importance(
            content="You are a helper",
            role="system",
            base_importance=0.0,  # Start at 0
        )
        assert importance == 0.0  # 0.0 - 0.1 = -0.1 -> clamped to 0.0


class TestWorkingMemoryLimit:
    """Tests for working memory size limit functionality (#169)."""

    @pytest.fixture
    def mock_service_with_limit(self):
        """Create a service with a small working memory limit for testing."""
        storage = AsyncMock()
        storage.store_episode = AsyncMock(return_value="ep_123")
        storage.update_episode = AsyncMock()
        storage.store_structured = AsyncMock(return_value="struct_123")
        storage.log_audit = AsyncMock()
        storage.search_episodes.return_value = []
        add_transaction_support(storage)

        embedder = AsyncMock()
        embedder.embed = AsyncMock(return_value=[0.1, 0.2, 0.3])

        # Small limit for testing (min is 10)
        settings = Settings(openai_api_key="sk-test-dummy-key", working_memory_max_size=10)
        workflow_backend = AsyncMock()

        return EngramService.model_construct(
            storage=storage,
            embedder=embedder,
            settings=settings,
            workflow_backend=workflow_backend,
            _working_memory=[],
            _conflicts={},
        )

    @pytest.mark.asyncio
    async def test_working_memory_enforces_limit(self, mock_service_with_limit):
        """Working memory should evict oldest entries when limit exceeded."""
        # Add 15 episodes (limit is 10)
        for i in range(15):
            await mock_service_with_limit.encode(
                content=f"Episode {i}",
                role="user",
                user_id="user_123",
            )

        working = mock_service_with_limit.get_working_memory()
        # Should only have 10 episodes (oldest 5 evicted)
        assert len(working) == 10
        # Should be the 10 most recent (FIFO eviction)
        assert working[0].content == "Episode 5"
        assert working[9].content == "Episode 14"

    @pytest.mark.asyncio
    async def test_working_memory_preserves_order(self, mock_service_with_limit):
        """Remaining episodes should maintain chronological order."""
        for i in range(20):
            await mock_service_with_limit.encode(
                content=f"Episode {i}",
                role="user",
                user_id="user_123",
            )

        working = mock_service_with_limit.get_working_memory()
        contents = [ep.content for ep in working]
        # Should have episodes 10-19 in order
        expected = [f"Episode {i}" for i in range(10, 20)]
        assert contents == expected

    def test_enforce_limit_method_directly(self, mock_service_with_limit):
        """_enforce_working_memory_limit should work correctly when called directly."""
        # Add episodes directly to working memory
        for i in range(15):
            episode = Episode(
                content=f"Episode {i}",
                role="user",
                user_id="user_123",
                embedding=[0.1, 0.2, 0.3],
            )
            mock_service_with_limit._working_memory.append(episode)

        # Call enforce limit
        mock_service_with_limit._enforce_working_memory_limit()

        assert len(mock_service_with_limit._working_memory) == 10
        assert mock_service_with_limit._working_memory[0].content == "Episode 5"
        assert mock_service_with_limit._working_memory[9].content == "Episode 14"

    def test_under_limit_no_eviction(self, mock_service_with_limit):
        """Episodes under limit should not be evicted."""
        for i in range(8):
            episode = Episode(
                content=f"Episode {i}",
                role="user",
                user_id="user_123",
                embedding=[0.1, 0.2, 0.3],
            )
            mock_service_with_limit._working_memory.append(episode)

        mock_service_with_limit._enforce_working_memory_limit()

        assert len(mock_service_with_limit._working_memory) == 8
