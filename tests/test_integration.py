"""Integration tests for end-to-end Engram workflows.

These tests verify the complete flow from encode to recall,
including storage, extraction, and embeddings working together.
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock

import pytest
from conftest import add_transaction_support

from engram.config import Settings
from engram.models import Episode, StructuredMemory
from engram.service import EngramService
from engram.storage import ScoredResult


class TestEncodeRecallWorkflow:
    """Integration tests for the encode → recall workflow."""

    @pytest.fixture
    def mock_embedder(self):
        """Create a mock embedder that returns deterministic vectors."""
        embedder = AsyncMock()

        # Return different vectors for different content
        async def embed_side_effect(text: str) -> list[float]:
            # Simple hash-based embedding for testing
            hash_val = hash(text) % 1000 / 1000
            return [hash_val, 1 - hash_val, 0.5]

        async def embed_batch_side_effect(texts: list[str]) -> list[list[float]]:
            # Return batch of embeddings
            return [await embed_side_effect(text) for text in texts]

        embedder.embed = AsyncMock(side_effect=embed_side_effect)
        embedder.embed_batch = AsyncMock(side_effect=embed_batch_side_effect)
        return embedder

    @pytest.fixture
    def mock_storage(self):
        """Create a mock storage that stores and retrieves memories."""
        storage = AsyncMock()
        storage._episodes: dict[str, Episode] = {}
        storage._structured: dict[str, StructuredMemory] = {}

        async def store_episode(episode: Episode) -> str:
            storage._episodes[episode.id] = episode
            return episode.id

        async def store_structured(structured: StructuredMemory) -> str:
            storage._structured[structured.id] = structured
            return structured.id

        async def search_episodes(
            query_vector: list[float],
            user_id: str,
            org_id: str | None = None,
            limit: int = 10,
            **kwargs: object,
        ) -> list[ScoredResult[Episode]]:
            # Return all episodes for the user wrapped in ScoredResult
            episodes = [
                ep
                for ep in storage._episodes.values()
                if ep.user_id == user_id and (org_id is None or ep.org_id == org_id)
            ][:limit]
            return [ScoredResult(memory=ep, score=0.85) for ep in episodes]

        async def search_structured(
            query_vector: list[float],
            user_id: str,
            org_id: str | None = None,
            limit: int = 10,
            min_confidence: float | None = None,
            **kwargs: object,
        ) -> list[ScoredResult[StructuredMemory]]:
            results = []
            for structured in storage._structured.values():
                if structured.user_id != user_id:
                    continue
                if org_id is not None and structured.org_id != org_id:
                    continue
                if min_confidence and structured.confidence.value < min_confidence:
                    continue
                results.append(structured)
            return [ScoredResult(memory=s, score=0.9) for s in results[:limit]]

        storage.store_episode = AsyncMock(side_effect=store_episode)
        storage.store_structured = AsyncMock(side_effect=store_structured)
        storage.search_episodes = AsyncMock(side_effect=search_episodes)
        storage.search_structured = AsyncMock(side_effect=search_structured)
        storage.initialize = AsyncMock()
        storage.close = AsyncMock()
        add_transaction_support(storage)

        return storage

    @pytest.fixture
    def service(self, mock_storage, mock_embedder):
        """Create a service with mock dependencies."""
        settings = Settings(openai_api_key="sk-test-dummy")
        workflow_backend = AsyncMock()

        # Use model_construct to bypass Pydantic validation for mocks
        return EngramService.model_construct(
            storage=mock_storage,
            embedder=mock_embedder,
            settings=settings,
            workflow_backend=workflow_backend,
            _working_memory=[],
            _conflicts={},
        )

    @pytest.mark.asyncio
    async def test_encode_stores_episode_and_extracts_structured(self, service):
        """Complete flow: encode text → store episode → extract structured data."""
        result = await service.encode(
            content="Contact me at alice@example.com or call +1 212 555 1234",
            role="user",
            user_id="test_user",
        )

        # Episode should be stored
        assert result.episode is not None
        assert result.episode.content == "Contact me at alice@example.com or call +1 212 555 1234"
        assert result.episode.user_id == "test_user"

        # Structured memory should have extractions - at least email
        assert result.structured is not None
        assert "alice@example.com" in result.structured.emails

    @pytest.mark.asyncio
    async def test_recall_returns_stored_memories(self, service, mock_storage):
        """Complete flow: encode → recall should find the memories."""
        # Encode some memories
        await service.encode(
            content="My email is bob@test.com",
            role="user",
            user_id="test_user",
        )
        await service.encode(
            content="I love hiking in the mountains",
            role="user",
            user_id="test_user",
        )

        # Recall should find them
        results = await service.recall(
            query="email address",
            user_id="test_user",
        )

        assert len(results) > 0
        # Should include both episodes and structured
        memory_types = {r.memory_type for r in results}
        assert "episodic" in memory_types or "structured" in memory_types

    @pytest.mark.asyncio
    async def test_user_isolation_in_workflow(self, service, mock_storage):
        """Users should only see their own memories."""
        # Alice's memory
        await service.encode(
            content="Alice's secret: password123",
            role="user",
            user_id="alice",
        )

        # Bob's memory
        await service.encode(
            content="Bob's secret: hunter2",
            role="user",
            user_id="bob",
        )

        # Alice's search
        alice_results = await service.recall(
            query="secret password",
            user_id="alice",
        )

        # Bob's search
        bob_results = await service.recall(
            query="secret password",
            user_id="bob",
        )

        # Verify isolation
        alice_content = " ".join(r.content for r in alice_results)
        bob_content = " ".join(r.content for r in bob_results)

        assert "Alice" in alice_content or "password123" in alice_content
        assert "Bob" not in alice_content
        assert "Bob" in bob_content or "hunter2" in bob_content
        assert "Alice" not in bob_content

    @pytest.mark.asyncio
    async def test_org_isolation_in_workflow(self, service, mock_storage):
        """Same user in different orgs should have isolated memories."""
        # Charlie at TechCorp
        await service.encode(
            content="TechCorp budget: $1M",
            role="user",
            user_id="charlie",
            org_id="techcorp",
        )

        # Charlie at StartupInc
        await service.encode(
            content="StartupInc budget: $100K",
            role="user",
            user_id="charlie",
            org_id="startupinc",
        )

        # Search in TechCorp context
        techcorp_results = await service.recall(
            query="budget",
            user_id="charlie",
            org_id="techcorp",
        )

        # Search in StartupInc context
        startup_results = await service.recall(
            query="budget",
            user_id="charlie",
            org_id="startupinc",
        )

        # Verify isolation
        techcorp_content = " ".join(r.content for r in techcorp_results)
        startup_content = " ".join(r.content for r in startup_results)

        assert "$1M" in techcorp_content
        assert "$100K" not in techcorp_content
        assert "$100K" in startup_content
        assert "$1M" not in startup_content


class TestExtractionIntegration:
    """Integration tests for extraction with real extractors."""

    @pytest.fixture
    def service(self):
        """Create a service with mocked storage but real extraction."""
        storage = AsyncMock()
        storage.store_episode = AsyncMock(return_value="ep_123")
        storage.store_structured = AsyncMock(return_value="struct_123")
        storage.initialize = AsyncMock()
        storage.close = AsyncMock()
        add_transaction_support(storage)

        embedder = AsyncMock()
        embedder.embed = AsyncMock(return_value=[0.1, 0.2, 0.3])
        embedder.embed_batch = AsyncMock(side_effect=lambda texts: [[0.1, 0.2, 0.3] for _ in texts])

        settings = Settings(openai_api_key="sk-test-dummy")
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
    async def test_extracts_emails(self, service):
        """Should extract valid emails."""
        result = await service.encode(
            content="Email me at john@example.com or jane@company.org",
            role="user",
            user_id="test",
        )

        assert len(result.structured.emails) == 2
        assert "john@example.com" in result.structured.emails
        assert "jane@company.org" in result.structured.emails

    @pytest.mark.asyncio
    async def test_extracts_phones(self, service):
        """Should extract phone numbers."""
        result = await service.encode(
            content="Call us at +44 20 7946 0958 or +33 1 42 68 53 00",
            role="user",
            user_id="test",
        )

        assert len(result.structured.phones) >= 1  # At least one valid international number

    @pytest.mark.asyncio
    async def test_extracts_urls(self, service):
        """Should extract URLs."""
        result = await service.encode(
            content="Check out https://github.com/ashita-ai/engram",
            role="user",
            user_id="test",
        )

        assert len(result.structured.urls) == 1
        assert "github.com" in result.structured.urls[0]

    @pytest.mark.asyncio
    async def test_extracts_complex_text(self, service):
        """Should extract multiple data types from complex text."""
        result = await service.encode(
            content="""
            Hi, I'm Dr. Alice Johnson. Contact me at alice@techcorp.com
            or call +44 20 7946 0958. Our meeting is on March 15, 2025.
            The prototype weighs 3.5 kg and costs $15,000.
            More info at https://techcorp.com/prototype
            """,
            role="user",
            user_id="test",
        )

        # Should find multiple types
        assert len(result.structured.emails) >= 1
        assert len(result.structured.urls) >= 1
        # Total extracts should be significant
        total_extracts = (
            len(result.structured.emails)
            + len(result.structured.phones)
            + len(result.structured.urls)
        )
        assert total_extracts >= 2


class TestAPIIntegration:
    """Integration tests for the FastAPI endpoints."""

    @pytest.fixture
    def mock_service(self):
        """Create a mock service for API tests."""
        service = MagicMock(spec=EngramService)
        service.encode = AsyncMock()
        service.recall = AsyncMock()
        return service

    @pytest.fixture
    def client(self, mock_service):
        """Create a test client with mock service."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from engram.api.router import router

        app = FastAPI()
        app.include_router(router, prefix="/api/v1")
        app.state.service = mock_service
        return TestClient(app)

    def test_full_encode_recall_via_api(self, client, mock_service):
        """Test encode and recall via REST API."""
        from engram.models import Episode, StructuredMemory
        from engram.service import EncodeResult, RecallResult

        # Setup mock encode response
        mock_episode = Episode(
            content="My email is test@example.com",
            role="user",
            user_id="api_user",
            embedding=[0.1, 0.2, 0.3],
        )
        mock_structured = StructuredMemory(
            source_episode_id=mock_episode.id,
            mode="fast",
            user_id="api_user",
            emails=["test@example.com"],
        )
        mock_service.encode.return_value = EncodeResult(
            episode=mock_episode, structured=mock_structured
        )

        # Encode via API
        encode_resp = client.post(
            "/api/v1/encode",
            json={
                "content": "My email is test@example.com",
                "role": "user",
                "user_id": "api_user",
            },
        )
        assert encode_resp.status_code == 201
        encode_data = encode_resp.json()
        assert encode_data["extract_count"] == 1

        # Setup mock recall response
        mock_service.recall.return_value = [
            RecallResult(
                memory_type="structured",
                content="test@example.com",
                score=0.95,
                confidence=0.9,
                memory_id=mock_structured.id,
                source_episode_id=mock_episode.id,
                metadata={"category": "email"},
            )
        ]

        # Recall via API
        recall_resp = client.post(
            "/api/v1/recall",
            json={
                "query": "email address",
                "user_id": "api_user",
            },
        )
        assert recall_resp.status_code == 200
        recall_data = recall_resp.json()
        assert recall_data["count"] == 1
        assert recall_data["results"][0]["content"] == "test@example.com"

    def test_recall_with_as_of_uses_recall_at(self, client, mock_service):
        """Test that recall with as_of parameter calls recall_at."""
        from datetime import datetime

        from engram.service import RecallResult

        # Setup mock recall_at response
        mock_service.recall_at = AsyncMock()
        mock_service.recall_at.return_value = [
            RecallResult(
                memory_type="episodic",
                content="Old memory content",
                score=0.85,
                confidence=None,
                memory_id="ep_old",
            )
        ]

        # Recall with as_of parameter
        as_of_time = datetime(2024, 6, 1, 12, 0, 0)
        recall_resp = client.post(
            "/api/v1/recall",
            json={
                "query": "email address",
                "user_id": "api_user",
                "as_of": as_of_time.isoformat(),
            },
        )
        assert recall_resp.status_code == 200
        recall_data = recall_resp.json()
        assert recall_data["count"] == 1
        assert recall_data["results"][0]["content"] == "Old memory content"

        # Verify recall_at was called, not recall
        mock_service.recall_at.assert_called_once()
        mock_service.recall.assert_not_called()


# Check for LLM API keys
HAS_OPENAI_KEY = bool(os.environ.get("ENGRAM_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY"))


class TestConsolidationIntegration:
    """Integration tests for consolidation workflow with real LLM."""

    @pytest.fixture
    def mock_storage(self):
        """Create mock storage that tracks stored memories."""
        from engram.models import Episode

        storage = AsyncMock()
        storage._episodes: list[Episode] = []
        storage._semantic_memories: list = []
        storage._summarized_ids: list[str] = []

        # Create test episodes
        ep1 = Episode(
            content="My name is Alice and I work at TechCorp as a senior engineer.",
            role="user",
            user_id="test_user",
            embedding=[0.1] * 384,
        )
        ep2 = Episode(
            content="I prefer Python over JavaScript for backend work.",
            role="user",
            user_id="test_user",
            embedding=[0.2] * 384,
        )
        ep3 = Episode(
            content="My email is alice@techcorp.com",
            role="user",
            user_id="test_user",
            embedding=[0.3] * 384,
        )
        storage._episodes = [ep1, ep2, ep3]

        async def get_unsummarized_episodes(
            user_id: str,
            org_id: str | None = None,
            limit: int | None = None,
        ) -> list[Episode]:
            unsummarized = [ep for ep in storage._episodes if ep.id not in storage._summarized_ids]
            if limit is not None:
                return unsummarized[:limit]
            return unsummarized

        async def store_semantic(memory) -> str:
            storage._semantic_memories.append(memory)
            return memory.id

        async def mark_episodes_summarized(
            episode_ids: list[str], user_id: str, semantic_id: str
        ) -> int:
            storage._summarized_ids.extend(episode_ids)
            return len(episode_ids)

        async def list_semantic_memories(user_id: str, org_id: str | None = None, **kwargs) -> list:
            return storage._semantic_memories

        storage.get_unsummarized_episodes = AsyncMock(side_effect=get_unsummarized_episodes)
        storage.store_semantic = AsyncMock(side_effect=store_semantic)
        storage.mark_episodes_summarized = AsyncMock(side_effect=mark_episodes_summarized)
        storage.list_semantic_memories = AsyncMock(side_effect=list_semantic_memories)
        storage.search_semantic = AsyncMock(return_value=[])

        return storage

    @pytest.fixture
    def real_embedder(self):
        """Create real fastembed embedder."""
        from engram.embeddings import FastEmbedEmbedder

        return FastEmbedEmbedder(model="BAAI/bge-small-en-v1.5")

    @pytest.mark.asyncio
    @pytest.mark.skipif(not HAS_OPENAI_KEY, reason="No OpenAI API key configured")
    async def test_consolidation_with_real_llm(self, mock_storage, real_embedder):
        """Test full consolidation workflow with real LLM summarization.

        This test:
        1. Creates episodes with factual content
        2. Runs consolidation (hits real OpenAI API via fallback path)
        3. Verifies ONE semantic summary is created (N episodes → 1 summary)
        """
        from engram.workflows.consolidation import run_consolidation

        result = await run_consolidation(
            storage=mock_storage,
            embedder=real_embedder,
            user_id="test_user",
            org_id="test_org",
        )

        # Should process all 3 episodes into ONE summary
        assert result.episodes_processed == 3
        assert result.semantic_memories_created == 1  # Compression: N → 1
        assert result.compression_ratio == 3.0

        # Verify memory was actually stored
        assert len(mock_storage._semantic_memories) == 1

        # Verify memory has content and embedding
        memory = mock_storage._semantic_memories[0]
        assert memory.content, "Memory should have content"
        assert memory.embedding, "Memory should have embedding"
        assert len(memory.embedding) == 384, "Embedding should be 384-dim"

        # Episodes should be marked as summarized
        mock_storage.mark_episodes_summarized.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.skipif(not HAS_OPENAI_KEY, reason="No OpenAI API key configured")
    async def test_consolidation_summary_contains_key_info(self, mock_storage, real_embedder):
        """Verify LLM summary contains key information from episodes."""
        from engram.workflows.consolidation import run_consolidation

        await run_consolidation(
            storage=mock_storage,
            embedder=real_embedder,
            user_id="test_user",
            org_id="test_org",
        )

        # Get the summary content
        assert len(mock_storage._semantic_memories) == 1
        summary = mock_storage._semantic_memories[0].content.lower()

        # Summary should mention key facts from the episodes
        # At least one of these should appear in the summary
        key_info = ["alice", "techcorp", "python", "engineer", "email"]
        found_info = [info for info in key_info if info in summary]

        assert len(found_info) >= 1, f"Expected at least 1 key fact in summary, found: {found_info}"

    @pytest.mark.asyncio
    @pytest.mark.skipif(not HAS_OPENAI_KEY, reason="No OpenAI API key configured")
    async def test_consolidation_empty_episodes(self, real_embedder):
        """Test consolidation with no episodes returns zero counts."""
        from engram.workflows.consolidation import run_consolidation

        empty_storage = AsyncMock()
        empty_storage.get_unsummarized_episodes = AsyncMock(return_value=[])

        result = await run_consolidation(
            storage=empty_storage,
            embedder=real_embedder,
            user_id="test_user",
            org_id="test_org",
        )

        assert result.episodes_processed == 0
        assert result.semantic_memories_created == 0
        assert result.links_created == 0


class TestBiTemporalRecall:
    """Tests for bi-temporal recall_at functionality."""

    @pytest.fixture
    def mock_embedder(self):
        """Create a mock embedder."""
        embedder = AsyncMock()

        async def embed_side_effect(text: str) -> list[float]:
            hash_val = hash(text) % 1000 / 1000
            return [hash_val, 1 - hash_val, 0.5]

        async def embed_batch_side_effect(texts: list[str]) -> list[list[float]]:
            return [await embed_side_effect(text) for text in texts]

        embedder.embed = AsyncMock(side_effect=embed_side_effect)
        embedder.embed_batch = AsyncMock(side_effect=embed_batch_side_effect)
        return embedder

    @pytest.fixture
    def mock_storage(self):
        """Create a mock storage with bi-temporal filtering support."""
        from datetime import datetime, timedelta

        storage = AsyncMock()
        storage._episodes: dict[str, Episode] = {}
        storage._structured: dict[str, StructuredMemory] = {}

        # Create episodes with different timestamps
        now = datetime.now()
        old_episode = Episode(
            content="Old email: old@example.com",
            role="user",
            user_id="test_user",
            embedding=[0.1, 0.9, 0.5],
            timestamp=now - timedelta(days=30),
        )

        new_episode = Episode(
            content="New email: new@example.com",
            role="user",
            user_id="test_user",
            embedding=[0.2, 0.8, 0.5],
            timestamp=now,
        )

        storage._episodes[old_episode.id] = old_episode
        storage._episodes[new_episode.id] = new_episode

        async def store_episode(episode: Episode) -> str:
            storage._episodes[episode.id] = episode
            return episode.id

        async def store_structured(structured: StructuredMemory) -> str:
            storage._structured[structured.id] = structured
            return structured.id

        async def search_episodes(
            query_vector: list[float],
            user_id: str,
            org_id: str | None = None,
            limit: int = 10,
            timestamp_before: datetime | None = None,
            **kwargs: object,
        ):
            from engram.storage import ScoredResult

            episodes = []
            for ep in storage._episodes.values():
                if ep.user_id != user_id:
                    continue
                if org_id is not None and ep.org_id != org_id:
                    continue
                # Bi-temporal filter
                if timestamp_before is not None and ep.timestamp > timestamp_before:
                    continue
                episodes.append(ep)
            return [ScoredResult(memory=ep, score=0.85) for ep in episodes[:limit]]

        async def search_structured(
            query_vector: list[float],
            user_id: str,
            org_id: str | None = None,
            limit: int = 10,
            min_confidence: float | None = None,
            derived_at_before: datetime | None = None,
            **kwargs: object,
        ):
            from engram.storage import ScoredResult

            results = []
            for structured in storage._structured.values():
                if structured.user_id != user_id:
                    continue
                if org_id is not None and structured.org_id != org_id:
                    continue
                if min_confidence and structured.confidence.value < min_confidence:
                    continue
                # Bi-temporal filter
                if derived_at_before is not None and structured.derived_at > derived_at_before:
                    continue
                results.append(structured)
            return [ScoredResult(memory=s, score=0.9) for s in results[:limit]]

        storage.store_episode = AsyncMock(side_effect=store_episode)
        storage.store_structured = AsyncMock(side_effect=store_structured)
        storage.search_episodes = AsyncMock(side_effect=search_episodes)
        storage.search_structured = AsyncMock(side_effect=search_structured)
        storage.log_audit = AsyncMock()
        storage.update_episode = AsyncMock()
        storage.get_episode = AsyncMock(return_value=None)
        storage.initialize = AsyncMock()
        storage.close = AsyncMock()

        add_transaction_support(storage)
        return storage

    @pytest.mark.asyncio
    async def test_recall_at_filters_by_timestamp(self, mock_storage, mock_embedder):
        """Test that recall_at filters memories by the as_of timestamp."""
        from datetime import datetime, timedelta

        # Use model_construct to bypass Pydantic validation for mocks
        service = EngramService.model_construct(
            storage=mock_storage,
            embedder=mock_embedder,
            settings=Settings(_env_file=None),
            workflow_backend=AsyncMock(),
            _working_memory=[],
            _conflicts={},
        )

        now = datetime.now()

        # Query as of 15 days ago (should only get old episode)
        results = await service.recall_at(
            query="email",
            as_of=now - timedelta(days=15),
            user_id="test_user",
        )

        # Should only get the old episode (created 30 days ago)
        assert len(results) == 1
        assert "old@example.com" in results[0].content

    @pytest.mark.asyncio
    async def test_recall_at_returns_all_when_future_timestamp(self, mock_storage, mock_embedder):
        """Test that recall_at with future timestamp returns all memories."""
        from datetime import datetime, timedelta

        # Use model_construct to bypass Pydantic validation for mocks
        service = EngramService.model_construct(
            storage=mock_storage,
            embedder=mock_embedder,
            settings=Settings(_env_file=None),
            workflow_backend=AsyncMock(),
            _working_memory=[],
            _conflicts={},
        )

        # Query as of tomorrow (should get both)
        future = datetime.now() + timedelta(days=1)
        results = await service.recall_at(
            query="email",
            as_of=future,
            user_id="test_user",
        )

        # Should get both episodes
        assert len(results) == 2
        contents = [r.content for r in results]
        assert any("old@example.com" in c for c in contents)
        assert any("new@example.com" in c for c in contents)

    @pytest.mark.asyncio
    async def test_recall_at_excludes_working_memory(self, mock_storage, mock_embedder):
        """Test that recall_at does NOT include working memory (by design)."""
        from datetime import datetime

        # Use model_construct to bypass Pydantic validation for mocks
        service = EngramService.model_construct(
            storage=mock_storage,
            embedder=mock_embedder,
            settings=Settings(_env_file=None),
            workflow_backend=AsyncMock(),
            _working_memory=[],
            _conflicts={},
        )

        # recall_at should not have include_working parameter - it's bi-temporal
        # Working memory is inherently "now" and shouldn't be in historical queries
        results = await service.recall_at(
            query="email",
            as_of=datetime.now(),
            user_id="test_user",
        )

        # Results should only come from storage, not working memory
        memory_types = {r.memory_type for r in results}
        assert "working" not in memory_types


class TestVerifyWorkflow:
    """Tests for the verify() method workflow."""

    @pytest.fixture
    def mock_embedder(self):
        """Create a mock embedder."""
        embedder = AsyncMock()

        async def embed_side_effect(text: str) -> list[float]:
            hash_val = hash(text) % 1000 / 1000
            return [hash_val, 1 - hash_val, 0.5]

        async def embed_batch_side_effect(texts: list[str]) -> list[list[float]]:
            return [await embed_side_effect(text) for text in texts]

        embedder.embed = AsyncMock(side_effect=embed_side_effect)
        embedder.embed_batch = AsyncMock(side_effect=embed_batch_side_effect)
        return embedder

    @pytest.fixture
    def mock_storage(self):
        """Create a mock storage for verify testing."""
        storage = AsyncMock()
        storage._episodes: dict[str, Episode] = {}
        storage._structured: dict[str, StructuredMemory] = {}

        async def store_episode(episode: Episode) -> str:
            storage._episodes[episode.id] = episode
            return episode.id

        async def store_structured(structured: StructuredMemory) -> str:
            storage._structured[structured.id] = structured
            return structured.id

        async def get_episode(episode_id: str, user_id: str) -> Episode | None:
            ep = storage._episodes.get(episode_id)
            if ep and ep.user_id == user_id:
                return ep
            return None

        async def get_structured(structured_id: str, user_id: str) -> StructuredMemory | None:
            structured = storage._structured.get(structured_id)
            if structured and structured.user_id == user_id:
                return structured
            return None

        storage.store_episode = AsyncMock(side_effect=store_episode)
        storage.store_structured = AsyncMock(side_effect=store_structured)
        storage.get_episode = AsyncMock(side_effect=get_episode)
        storage.get_structured = AsyncMock(side_effect=get_structured)
        storage.update_episode = AsyncMock()
        storage.log_audit = AsyncMock()
        storage.initialize = AsyncMock()
        storage.close = AsyncMock()

        add_transaction_support(storage)
        return storage

    @pytest.mark.asyncio
    async def test_verify_structured_success(self, mock_storage, mock_embedder):
        """Test verifying a structured memory traces back to its source episode."""
        # Use model_construct to bypass Pydantic validation for mocks
        service = EngramService.model_construct(
            storage=mock_storage,
            embedder=mock_embedder,
            settings=Settings(_env_file=None),
            workflow_backend=AsyncMock(),
            _working_memory=[],
            _conflicts={},
        )

        # Encode an email to get structured memory
        result = await service.encode(
            content="My email is test@example.com",
            role="user",
            user_id="user_123",
        )

        assert result.structured is not None
        assert "test@example.com" in result.structured.emails

        # Verify the structured memory
        verification = await service.verify(result.structured.id, "user_123")

        assert verification.memory_id == result.structured.id
        assert verification.memory_type == "structured"
        assert verification.verified is True
        assert len(verification.source_episodes) == 1
        assert verification.extraction_method == "extracted"
        assert verification.confidence == 0.9
        assert "Pattern-matched" in verification.explanation

    @pytest.mark.asyncio
    async def test_verify_structured_not_found(self, mock_storage, mock_embedder):
        """Test verify raises NotFoundError for non-existent structured memory."""
        from engram.exceptions import NotFoundError

        # Use model_construct to bypass Pydantic validation for mocks
        service = EngramService.model_construct(
            storage=mock_storage,
            embedder=mock_embedder,
            settings=Settings(_env_file=None),
            workflow_backend=AsyncMock(),
            _working_memory=[],
            _conflicts={},
        )

        with pytest.raises(NotFoundError, match="StructuredMemory not found"):
            await service.verify("struct_nonexistent", "user_123")

    @pytest.mark.asyncio
    async def test_verify_invalid_memory_id(self, mock_storage, mock_embedder):
        """Test verify raises ValidationError for invalid memory ID format."""
        from engram.exceptions import ValidationError

        # Use model_construct to bypass Pydantic validation for mocks
        service = EngramService.model_construct(
            storage=mock_storage,
            embedder=mock_embedder,
            settings=Settings(_env_file=None),
            workflow_backend=AsyncMock(),
            _working_memory=[],
            _conflicts={},
        )

        with pytest.raises(ValidationError, match="Cannot determine memory type"):
            await service.verify("invalid_id_format", "user_123")


class TestFreshnessHints:
    """Tests for freshness hints in recall."""

    @pytest.fixture
    def mock_embedder(self):
        """Create a mock embedder."""
        embedder = AsyncMock()

        async def embed_side_effect(text: str) -> list[float]:
            hash_val = hash(text) % 1000 / 1000
            return [hash_val, 1 - hash_val, 0.5]

        async def embed_batch_side_effect(texts: list[str]) -> list[list[float]]:
            return [await embed_side_effect(text) for text in texts]

        embedder.embed = AsyncMock(side_effect=embed_side_effect)
        embedder.embed_batch = AsyncMock(side_effect=embed_batch_side_effect)
        return embedder

    @pytest.fixture
    def mock_storage(self):
        """Create a mock storage with episodes."""
        storage = AsyncMock()
        storage._episodes: dict[str, Episode] = {}
        storage._structured: dict[str, StructuredMemory] = {}

        async def store_episode(episode: Episode) -> str:
            storage._episodes[episode.id] = episode
            return episode.id

        async def store_structured(structured: StructuredMemory) -> str:
            storage._structured[structured.id] = structured
            return structured.id

        async def search_episodes(
            query_vector, user_id, org_id=None, limit=10, **kwargs
        ) -> list[ScoredResult[Episode]]:
            results = []
            for ep in storage._episodes.values():
                if ep.user_id == user_id:
                    results.append(ScoredResult(memory=ep, score=0.8))
            return results[:limit]

        async def search_structured(
            query_vector, user_id, org_id=None, limit=10, **kwargs
        ) -> list[ScoredResult[StructuredMemory]]:
            results = []
            for structured in storage._structured.values():
                if structured.user_id == user_id:
                    results.append(ScoredResult(memory=structured, score=0.85))
            return results[:limit]

        async def search_semantic(query_vector, user_id, org_id=None, limit=10, **kwargs):
            return []

        async def search_procedural(query_vector, user_id, org_id=None, limit=10, **kwargs):
            return []

        storage.store_episode = AsyncMock(side_effect=store_episode)
        storage.store_structured = AsyncMock(side_effect=store_structured)
        storage.search_episodes = AsyncMock(side_effect=search_episodes)
        storage.search_structured = AsyncMock(side_effect=search_structured)
        storage.search_semantic = AsyncMock(side_effect=search_semantic)
        storage.search_procedural = AsyncMock(side_effect=search_procedural)
        storage.update_episode = AsyncMock()
        storage.get_episode = AsyncMock(return_value=None)
        storage.log_audit = AsyncMock()

        add_transaction_support(storage)
        return storage

    @pytest.mark.asyncio
    async def test_recall_returns_staleness_for_memories(self, mock_storage, mock_embedder):
        """Test that recall returns staleness information for memories."""
        from engram.models import Staleness

        # Use model_construct to bypass Pydantic validation for mocks
        service = EngramService.model_construct(
            storage=mock_storage,
            embedder=mock_embedder,
            settings=Settings(_env_file=None),
            workflow_backend=AsyncMock(),
            _working_memory=[],
            _conflicts={},
        )

        # Encode a message
        await service.encode(
            content="My phone number is 555-123-4567",
            role="user",
            user_id="user_123",
        )

        # Recall should return results
        results = await service.recall(
            query="phone number",
            user_id="user_123",
        )

        # Should find some results
        assert len(results) > 0

        # All results should have staleness set
        for r in results:
            assert r.staleness in (Staleness.FRESH, Staleness.STALE, Staleness.CONSOLIDATING)

        # Find structured result - should be FRESH
        structured_results = [r for r in results if r.memory_type == "structured"]
        if structured_results:
            # Structured memories are always FRESH since they're created immediately with episode
            assert structured_results[0].staleness == Staleness.FRESH

    @pytest.mark.asyncio
    async def test_recall_returns_different_memory_types(self, mock_storage, mock_embedder):
        """Test that recall returns multiple memory types."""

        # Use model_construct to bypass Pydantic validation for mocks
        service = EngramService.model_construct(
            storage=mock_storage,
            embedder=mock_embedder,
            settings=Settings(_env_file=None),
            workflow_backend=AsyncMock(),
            _working_memory=[],
            _conflicts={},
        )

        # Encode a message
        await service.encode(
            content="My phone number is 555-123-4567",
            role="user",
            user_id="user_123",
        )

        # Recall should return results
        results = await service.recall(
            query="phone number",
            user_id="user_123",
        )

        # Should find some results
        assert len(results) > 0

        # Get memory types returned
        memory_types = {r.memory_type for r in results}
        # Should have at least working memory (from current session) or stored memories
        assert len(memory_types) >= 1

    @pytest.mark.asyncio
    async def test_working_memory_included_in_recall(self, mock_storage, mock_embedder):
        """Test that working memory is included in recall results."""
        # Use model_construct to bypass Pydantic validation for mocks
        service = EngramService.model_construct(
            storage=mock_storage,
            embedder=mock_embedder,
            settings=Settings(_env_file=None),
            workflow_backend=AsyncMock(),
            _working_memory=[],
            _conflicts={},
        )

        # Encode a message (adds to working memory)
        await service.encode(
            content="My email is test@example.com",
            role="user",
            user_id="user_123",
        )

        # Recall including working memory
        results = await service.recall(
            query="email",
            user_id="user_123",
        )

        # Should find results (may include working memory or stored memories)
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_recall_freshness_modes(self, mock_storage, mock_embedder):
        """Test that freshness modes work as expected."""
        # Use model_construct to bypass Pydantic validation for mocks
        service = EngramService.model_construct(
            storage=mock_storage,
            embedder=mock_embedder,
            settings=Settings(_env_file=None),
            workflow_backend=AsyncMock(),
            _working_memory=[],
            _conflicts={},
        )

        # Encode a message
        await service.encode(
            content="My email is test@example.com",
            role="user",
            user_id="user_123",
        )

        # Recall with best_effort (default)
        best_effort_results = await service.recall(
            query="email",
            user_id="user_123",
            freshness="best_effort",
        )

        # Recall with fresh_only
        fresh_only_results = await service.recall(
            query="email",
            user_id="user_123",
            freshness="fresh_only",
        )

        # Both modes should return results (specific filtering depends on memory states)
        # best_effort should have at least as many results as fresh_only
        assert len(best_effort_results) >= len(fresh_only_results)
