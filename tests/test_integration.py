"""Integration tests for end-to-end Engram workflows.

These tests verify the complete flow from encode to recall,
including storage, extraction, and embeddings working together.
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock

import pytest

from engram.config import Settings
from engram.extraction import default_pipeline
from engram.models import Episode, Fact
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
        storage._facts: dict[str, Fact] = {}

        async def store_episode(episode: Episode) -> str:
            storage._episodes[episode.id] = episode
            return episode.id

        async def store_fact(fact: Fact) -> str:
            storage._facts[fact.id] = fact
            return fact.id

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

        async def search_facts(
            query_vector: list[float],
            user_id: str,
            org_id: str | None = None,
            limit: int = 10,
            min_confidence: float | None = None,
            **kwargs: object,
        ) -> list[ScoredResult[Fact]]:
            results = []
            for fact in storage._facts.values():
                if fact.user_id != user_id:
                    continue
                if org_id is not None and fact.org_id != org_id:
                    continue
                if min_confidence and fact.confidence.value < min_confidence:
                    continue
                results.append(fact)
            return [ScoredResult(memory=f, score=0.9) for f in results[:limit]]

        storage.store_episode = AsyncMock(side_effect=store_episode)
        storage.store_fact = AsyncMock(side_effect=store_fact)
        storage.search_episodes = AsyncMock(side_effect=search_episodes)
        storage.search_facts = AsyncMock(side_effect=search_facts)
        storage.initialize = AsyncMock()
        storage.close = AsyncMock()

        return storage

    @pytest.fixture
    def service(self, mock_storage, mock_embedder):
        """Create a service with mock dependencies."""
        settings = Settings(openai_api_key="sk-test-dummy")
        return EngramService(
            storage=mock_storage,
            embedder=mock_embedder,
            pipeline=default_pipeline(),
            settings=settings,
        )

    @pytest.mark.asyncio
    async def test_encode_stores_episode_and_extracts_facts(self, service):
        """Complete flow: encode text → store episode → extract facts."""
        result = await service.encode(
            content="Contact me at alice@example.com or call +1 212 555 1234",
            role="user",
            user_id="test_user",
        )

        # Episode should be stored
        assert result.episode is not None
        assert result.episode.content == "Contact me at alice@example.com or call +1 212 555 1234"
        assert result.episode.user_id == "test_user"

        # Facts should be extracted - at least email
        assert len(result.facts) >= 1
        categories = {f.category for f in result.facts}
        assert "email" in categories

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
        # Should include both episodes and facts
        memory_types = {r.memory_type for r in results}
        assert "episode" in memory_types or "fact" in memory_types

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
        storage.store_fact = AsyncMock(return_value="fact_123")
        storage.initialize = AsyncMock()
        storage.close = AsyncMock()

        embedder = AsyncMock()
        embedder.embed = AsyncMock(return_value=[0.1, 0.2, 0.3])
        embedder.embed_batch = AsyncMock(side_effect=lambda texts: [[0.1, 0.2, 0.3] for _ in texts])

        settings = Settings(openai_api_key="sk-test-dummy")
        return EngramService(
            storage=storage,
            embedder=embedder,
            pipeline=default_pipeline(),
            settings=settings,
        )

    @pytest.mark.asyncio
    async def test_extracts_emails(self, service):
        """Should extract valid emails."""
        result = await service.encode(
            content="Email me at john@example.com or jane@company.org",
            role="user",
            user_id="test",
        )

        emails = [f for f in result.facts if f.category == "email"]
        assert len(emails) == 2
        contents = {e.content for e in emails}
        assert "john@example.com" in contents
        assert "jane@company.org" in contents

    @pytest.mark.asyncio
    async def test_extracts_phones(self, service):
        """Should extract phone numbers."""
        result = await service.encode(
            content="Call us at +44 20 7946 0958 or +33 1 42 68 53 00",
            role="user",
            user_id="test",
        )

        phones = [f for f in result.facts if f.category == "phone"]
        assert len(phones) >= 1  # At least one valid international number

    @pytest.mark.asyncio
    async def test_extracts_urls(self, service):
        """Should extract URLs."""
        result = await service.encode(
            content="Check out https://github.com/ashita-ai/engram",
            role="user",
            user_id="test",
        )

        urls = [f for f in result.facts if f.category == "url"]
        assert len(urls) == 1
        assert "github.com" in urls[0].content

    @pytest.mark.asyncio
    async def test_extracts_dates(self, service):
        """Should extract dates."""
        result = await service.encode(
            content="Meeting scheduled for January 15, 2025 at 3pm",
            role="user",
            user_id="test",
        )

        dates = [f for f in result.facts if f.category == "date"]
        assert len(dates) >= 1

    @pytest.mark.asyncio
    async def test_extracts_quantities(self, service):
        """Should extract physical quantities."""
        result = await service.encode(
            content="The package weighs 5 kg and is 30 cm long",
            role="user",
            user_id="test",
        )

        quantities = [f for f in result.facts if f.category == "quantity"]
        assert len(quantities) >= 2

    @pytest.mark.asyncio
    async def test_extracts_names(self, service):
        """Should extract human names."""
        result = await service.encode(
            content="Please contact Dr. Jane Smith for more information",
            role="user",
            user_id="test",
        )

        names = [f for f in result.facts if f.category == "person"]
        assert len(names) >= 1

    @pytest.mark.asyncio
    async def test_masks_sensitive_ids(self, service):
        """Should mask sensitive IDs like credit cards and SSN."""
        result = await service.encode(
            content="Card: 4532015112830366, SSN: 123-45-6789",
            role="user",
            user_id="test",
        )

        ids = [f for f in result.facts if f.category in ("credit_card", "ssn")]
        for fact in ids:
            # Credit cards should be masked
            if fact.category == "credit_card":
                assert "****" in fact.content
                assert "4532015112830366" not in fact.content
            # SSN should be masked
            if fact.category == "ssn":
                assert "***-**-" in fact.content
                assert "123-45" not in fact.content

    @pytest.mark.asyncio
    async def test_complex_text_extraction(self, service):
        """Should extract multiple fact types from complex text."""
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

        categories = {f.category for f in result.facts}

        # Should find multiple types
        assert "email" in categories
        assert "date" in categories
        assert "url" in categories
        assert "quantity" in categories
        assert "person" in categories
        # At least 5 different categories extracted
        assert len(categories) >= 5


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

        from engram.api.router import router, set_service

        app = FastAPI()
        app.include_router(router, prefix="/api/v1")
        set_service(mock_service)
        yield TestClient(app)
        set_service(None)  # type: ignore[arg-type]

    def test_full_encode_recall_via_api(self, client, mock_service):
        """Test encode and recall via REST API."""
        from engram.models import Episode, Fact
        from engram.service import EncodeResult, RecallResult

        # Setup mock encode response
        mock_episode = Episode(
            content="My email is test@example.com",
            role="user",
            user_id="api_user",
            embedding=[0.1, 0.2, 0.3],
        )
        mock_fact = Fact(
            content="test@example.com",
            category="email",
            source_episode_id=mock_episode.id,
            user_id="api_user",
            embedding=[0.1, 0.2, 0.3],
        )
        mock_service.encode.return_value = EncodeResult(episode=mock_episode, facts=[mock_fact])

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
        assert encode_data["fact_count"] == 1

        # Setup mock recall response
        mock_service.recall.return_value = [
            RecallResult(
                memory_type="fact",
                content="test@example.com",
                score=0.95,
                confidence=0.9,
                memory_id=mock_fact.id,
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
        storage._consolidated_ids: list[str] = []

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

        async def get_unconsolidated_episodes(
            user_id: str,
            org_id: str | None = None,
            limit: int = 20,
        ) -> list[Episode]:
            return [ep for ep in storage._episodes if ep.id not in storage._consolidated_ids][
                :limit
            ]

        async def store_semantic(memory) -> str:
            storage._semantic_memories.append(memory)
            return memory.id

        async def mark_episodes_consolidated(episode_ids: list[str], user_id: str) -> int:
            storage._consolidated_ids.extend(episode_ids)
            return len(episode_ids)

        storage.get_unconsolidated_episodes = AsyncMock(side_effect=get_unconsolidated_episodes)
        storage.store_semantic = AsyncMock(side_effect=store_semantic)
        storage.mark_episodes_consolidated = AsyncMock(side_effect=mark_episodes_consolidated)

        return storage

    @pytest.fixture
    def real_embedder(self):
        """Create real fastembed embedder."""
        from engram.embeddings import FastEmbedEmbedder

        return FastEmbedEmbedder(model="BAAI/bge-small-en-v1.5")

    @pytest.mark.asyncio
    @pytest.mark.skipif(not HAS_OPENAI_KEY, reason="No OpenAI API key configured")
    async def test_consolidation_with_real_llm(self, mock_storage, real_embedder):
        """Test full consolidation workflow with real LLM extraction.

        This test:
        1. Creates episodes with factual content
        2. Runs consolidation (hits real OpenAI API via fallback path)
        3. Verifies semantic memories are extracted and stored
        """
        from engram.workflows.consolidation import run_consolidation

        result = await run_consolidation(
            storage=mock_storage,
            embedder=real_embedder,
            user_id="test_user",
        )

        # Should process all 3 episodes
        assert result.episodes_processed == 3

        # LLM should extract at least some semantic facts
        # (name, workplace, language preference, email)
        assert result.semantic_memories_created >= 1

        # Verify memories were actually stored
        assert len(mock_storage._semantic_memories) == result.semantic_memories_created

        # Verify each memory has content and embedding
        for memory in mock_storage._semantic_memories:
            assert memory.content, "Memory should have content"
            assert memory.embedding, "Memory should have embedding"
            assert len(memory.embedding) == 384, "Embedding should be 384-dim"

        # Episodes should be marked as consolidated
        mock_storage.mark_episodes_consolidated.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.skipif(not HAS_OPENAI_KEY, reason="No OpenAI API key configured")
    async def test_consolidation_extracts_correct_facts(self, mock_storage, real_embedder):
        """Verify LLM extracts semantically correct facts."""
        from engram.workflows.consolidation import run_consolidation

        await run_consolidation(
            storage=mock_storage,
            embedder=real_embedder,
            user_id="test_user",
        )

        # Get all extracted content
        all_content = " ".join(m.content.lower() for m in mock_storage._semantic_memories)

        # LLM should extract key facts from the episodes
        # At least one of these should appear
        key_facts = ["alice", "techcorp", "python", "engineer", "email"]
        found_facts = [f for f in key_facts if f in all_content]

        assert len(found_facts) >= 2, f"Expected at least 2 key facts, found: {found_facts}"

    @pytest.mark.asyncio
    @pytest.mark.skipif(not HAS_OPENAI_KEY, reason="No OpenAI API key configured")
    async def test_consolidation_empty_episodes(self, real_embedder):
        """Test consolidation with no episodes returns zero counts."""
        from engram.workflows.consolidation import run_consolidation

        empty_storage = AsyncMock()
        empty_storage.get_unconsolidated_episodes = AsyncMock(return_value=[])

        result = await run_consolidation(
            storage=empty_storage,
            embedder=real_embedder,
            user_id="test_user",
        )

        assert result.episodes_processed == 0
        assert result.semantic_memories_created == 0
        assert result.links_created == 0
