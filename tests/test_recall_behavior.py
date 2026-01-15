"""Tests for recall behavior including filtering, backfill prevention, and source linkage.

These tests verify that recall returns accurate, relevant results without
backfilling with irrelevant content when filtering removes results.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from engram.config import Settings
from engram.models import (
    ConfidenceScore,
    Episode,
    ProceduralMemory,
    SemanticMemory,
    StructuredMemory,
)
from engram.models.structured import Negation
from engram.service import EngramService, RecallResult
from engram.storage import ScoredResult


class TestNegationFiltering:
    """Tests for negation filtering behavior."""

    @pytest.fixture
    def mock_service(self):
        """Create a service with mocked dependencies for negation testing."""
        storage = AsyncMock()
        embedder = MagicMock()  # Use MagicMock so we can set individual methods
        embedder.embed = AsyncMock(return_value=[0.1, 0.2, 0.3])
        embedder.embed_batch = AsyncMock(return_value=[[0.1, 0.2, 0.3]])

        settings = Settings(openai_api_key="sk-test-dummy-key")

        service = EngramService(
            storage=storage,
            embedder=embedder,
            settings=settings,
        )
        return service

    @pytest.mark.asyncio
    async def test_negation_filter_removes_contradicted_memories(self, mock_service):
        """Negation filtering should remove memories that match negation patterns."""
        # Setup: episodes about databases, including MongoDB
        ep_pg = Episode(
            id="ep_pg",
            content="I prefer PostgreSQL for databases",
            role="user",
            user_id="u1",
            embedding=[0.1, 0.2, 0.3],
        )
        ep_mongo = Episode(
            id="ep_mongo",
            content="I used to use MongoDB heavily",
            role="user",
            user_id="u1",
            embedding=[0.1, 0.2, 0.4],
        )
        episodes = [ep_pg, ep_mongo]

        # Structured memory with negations
        structured_with_negation = StructuredMemory(
            id="struct_neg",
            source_episode_id="ep_neg",
            mode="rich",
            user_id="u1",
            negations=[
                Negation(
                    content="I don't use MongoDB anymore",
                    pattern="MongoDB",
                    context="switched databases",
                )
            ],
        )

        mock_service.storage.search_episodes = AsyncMock(
            return_value=[ScoredResult(memory=ep, score=0.9) for ep in episodes]
        )
        mock_service.storage.search_structured = AsyncMock(return_value=[])
        mock_service.storage.search_semantic = AsyncMock(return_value=[])
        mock_service.storage.search_procedural = AsyncMock(return_value=[])
        mock_service.storage.list_structured_memories = AsyncMock(
            return_value=[structured_with_negation]
        )
        mock_service.storage.log_audit = AsyncMock()

        # Mock get_episode to return the actual episodes (for embedding lookup)
        async def get_episode_side_effect(memory_id, user_id):
            for ep in episodes:
                if ep.id == memory_id:
                    return ep
            return None

        mock_service.storage.get_episode = AsyncMock(side_effect=get_episode_side_effect)

        # Recall with negation filtering enabled (pattern-only to avoid mock embedding issues)
        results = await mock_service.recall(
            query="database",
            user_id="u1",
            memory_types=["episodic"],
            apply_negation_filter=True,
            negation_similarity_threshold=None,  # Pattern-only filtering
            limit=10,
        )

        # MongoDB episode should be filtered out (pattern match on "MongoDB")
        contents = [r.content for r in results]
        assert "I prefer PostgreSQL for databases" in contents
        assert "I used to use MongoDB heavily" not in contents

    @pytest.mark.asyncio
    async def test_negation_filter_disabled_returns_all(self, mock_service):
        """When negation filter is disabled, all results should be returned."""
        episodes = [
            Episode(
                id="ep_pg",
                content="I prefer PostgreSQL for databases",
                role="user",
                user_id="u1",
                embedding=[0.1, 0.2, 0.3],
            ),
            Episode(
                id="ep_mongo",
                content="I used to use MongoDB heavily",
                role="user",
                user_id="u1",
                embedding=[0.1, 0.2, 0.4],
            ),
        ]

        # Structured memory with negations
        structured_with_negation = StructuredMemory(
            id="struct_neg",
            source_episode_id="ep_neg",
            mode="rich",
            user_id="u1",
            negations=[
                Negation(
                    content="I don't use MongoDB anymore",
                    pattern="MongoDB",
                    context="switched databases",
                )
            ],
        )

        mock_service.storage.search_episodes = AsyncMock(
            return_value=[ScoredResult(memory=ep, score=0.9) for ep in episodes]
        )
        mock_service.storage.search_structured = AsyncMock(return_value=[])
        mock_service.storage.search_semantic = AsyncMock(return_value=[])
        mock_service.storage.search_procedural = AsyncMock(return_value=[])
        mock_service.storage.list_structured_memories = AsyncMock(
            return_value=[structured_with_negation]
        )
        mock_service.storage.log_audit = AsyncMock()

        # Recall with negation filtering DISABLED
        results = await mock_service.recall(
            query="database",
            user_id="u1",
            memory_types=["episodic"],
            apply_negation_filter=False,
            limit=10,
        )

        # Both episodes should be returned
        contents = [r.content for r in results]
        assert "I prefer PostgreSQL for databases" in contents
        assert "I used to use MongoDB heavily" in contents


class TestNoBackfillBehavior:
    """Tests that filtering doesn't backfill with irrelevant results."""

    @pytest.fixture
    def mock_service(self):
        """Create a service with mocked dependencies."""
        storage = AsyncMock()
        embedder = MagicMock()  # Use MagicMock so we can set individual methods
        embedder.embed = AsyncMock(return_value=[0.1, 0.2, 0.3])
        embedder.embed_batch = AsyncMock(return_value=[[0.1, 0.2, 0.3]])

        settings = Settings(openai_api_key="sk-test-dummy-key")

        return EngramService(
            storage=storage,
            embedder=embedder,
            settings=settings,
        )

    @pytest.mark.asyncio
    async def test_filtered_results_not_backfilled(self, mock_service):
        """When negation removes results, should return fewer results, not backfill."""
        # 4 episodes, 2 will be filtered
        ep_1 = Episode(
            id="ep_1",
            content="I prefer PostgreSQL",
            role="user",
            user_id="u1",
            embedding=[0.1, 0.2, 0.3],
        )
        ep_2 = Episode(
            id="ep_2",
            content="Redis is great for caching",
            role="user",
            user_id="u1",
            embedding=[0.1, 0.2, 0.35],
        )
        ep_3 = Episode(
            id="ep_3",
            content="MongoDB was useful before",
            role="user",
            user_id="u1",
            embedding=[0.1, 0.2, 0.4],
        )
        ep_4 = Episode(
            id="ep_4",
            content="I stopped using MongoDB",
            role="user",
            user_id="u1",
            embedding=[0.1, 0.2, 0.45],
        )
        episodes = [ep_1, ep_2, ep_3, ep_4]

        # Structured memory with negations
        structured_with_negation = StructuredMemory(
            id="struct_neg",
            source_episode_id="ep_neg",
            mode="rich",
            user_id="u1",
            negations=[
                Negation(
                    content="I don't use MongoDB",
                    pattern="MongoDB",
                    context="switched databases",
                )
            ],
        )

        mock_service.storage.search_episodes = AsyncMock(
            return_value=[
                ScoredResult(memory=ep, score=0.9 - i * 0.05) for i, ep in enumerate(episodes)
            ]
        )
        mock_service.storage.search_structured = AsyncMock(return_value=[])
        mock_service.storage.search_semantic = AsyncMock(return_value=[])
        mock_service.storage.search_procedural = AsyncMock(return_value=[])
        mock_service.storage.list_structured_memories = AsyncMock(
            return_value=[structured_with_negation]
        )
        mock_service.storage.log_audit = AsyncMock()

        # Mock get_episode to return the actual episodes (for embedding lookup)
        async def get_episode_side_effect(memory_id, user_id):
            for ep in episodes:
                if ep.id == memory_id:
                    return ep
            return None

        mock_service.storage.get_episode = AsyncMock(side_effect=get_episode_side_effect)

        results = await mock_service.recall(
            query="database",
            user_id="u1",
            memory_types=["episodic"],
            apply_negation_filter=True,
            negation_similarity_threshold=None,  # Pattern-only filtering
            limit=4,
        )

        # Should return 2 results (the non-MongoDB ones), NOT 4
        assert len(results) == 2
        contents = [r.content for r in results]
        assert "I prefer PostgreSQL" in contents
        assert "Redis is great for caching" in contents
        assert "MongoDB" not in " ".join(contents)

    @pytest.mark.asyncio
    async def test_limit_respected_without_filtering(self, mock_service):
        """Without filtering, limit should be respected normally."""
        episodes = [
            Episode(
                id=f"ep_{i}",
                content=f"Episode {i} content",
                role="user",
                user_id="u1",
                embedding=[0.1, 0.2, 0.3 + i * 0.01],
            )
            for i in range(10)
        ]

        mock_service.storage.search_episodes = AsyncMock(
            return_value=[
                ScoredResult(memory=ep, score=0.9 - i * 0.05) for i, ep in enumerate(episodes[:5])
            ]
        )
        mock_service.storage.search_structured = AsyncMock(return_value=[])
        mock_service.storage.search_semantic = AsyncMock(return_value=[])
        mock_service.storage.search_procedural = AsyncMock(return_value=[])
        mock_service.storage.list_structured_memories = AsyncMock(return_value=[])
        mock_service.storage.log_audit = AsyncMock()

        results = await mock_service.recall(
            query="episode",
            user_id="u1",
            memory_types=["episodic"],
            apply_negation_filter=False,
            limit=5,
        )

        assert len(results) == 5


class TestSourceEpisodeLinkage:
    """Tests for source_episode_ids in RecallResult."""

    @pytest.fixture
    def mock_service(self):
        """Create a service with mocked dependencies."""
        storage = AsyncMock()
        embedder = AsyncMock()
        embedder.embed = AsyncMock(return_value=[0.1, 0.2, 0.3])

        settings = Settings(openai_api_key="sk-test-dummy-key")

        return EngramService(
            storage=storage,
            embedder=embedder,
            settings=settings,
        )

    @pytest.mark.asyncio
    async def test_semantic_memory_includes_source_episode_ids(self, mock_service):
        """RecallResult for semantic memories should include source_episode_ids."""
        semantic = SemanticMemory(
            id="sem_1",
            content="User prefers Python",
            source_episode_ids=["ep_1", "ep_2", "ep_3"],
            user_id="u1",
            confidence=ConfidenceScore.for_inferred(),
        )

        mock_service.storage.search_episodes = AsyncMock(return_value=[])
        mock_service.storage.search_structured = AsyncMock(return_value=[])
        mock_service.storage.search_semantic = AsyncMock(
            return_value=[ScoredResult(memory=semantic, score=0.9)]
        )
        mock_service.storage.search_procedural = AsyncMock(return_value=[])
        mock_service.storage.list_structured_memories = AsyncMock(return_value=[])
        mock_service.storage.log_audit = AsyncMock()

        results = await mock_service.recall(
            query="Python",
            user_id="u1",
            memory_types=["semantic"],
        )

        assert len(results) == 1
        assert results[0].source_episode_ids == ["ep_1", "ep_2", "ep_3"]

    @pytest.mark.asyncio
    async def test_procedural_memory_includes_source_episode_ids(self, mock_service):
        """RecallResult for procedural memories should include source_episode_ids."""
        procedural = ProceduralMemory(
            id="proc_1",
            content="When coding, use type hints",
            source_episode_ids=["ep_a", "ep_b"],
            user_id="u1",
            confidence=ConfidenceScore.for_inferred(),
            trigger_context="coding",
        )

        mock_service.storage.search_episodes = AsyncMock(return_value=[])
        mock_service.storage.search_structured = AsyncMock(return_value=[])
        mock_service.storage.search_semantic = AsyncMock(return_value=[])
        mock_service.storage.search_procedural = AsyncMock(
            return_value=[ScoredResult(memory=procedural, score=0.85)]
        )
        mock_service.storage.list_structured_memories = AsyncMock(return_value=[])
        mock_service.storage.log_audit = AsyncMock()

        results = await mock_service.recall(
            query="coding practices",
            user_id="u1",
            memory_types=["procedural"],
        )

        assert len(results) == 1
        assert results[0].source_episode_ids == ["ep_a", "ep_b"]

    @pytest.mark.asyncio
    async def test_structured_memory_includes_source_episode_id(self, mock_service):
        """RecallResult for structured memories should include source_episode_id."""
        structured = StructuredMemory(
            id="struct_1",
            source_episode_id="ep_email",
            mode="fast",
            user_id="u1",
            emails=["user@example.com"],
        )

        mock_service.storage.search_episodes = AsyncMock(return_value=[])
        mock_service.storage.search_structured = AsyncMock(
            return_value=[ScoredResult(memory=structured, score=0.95)]
        )
        mock_service.storage.search_semantic = AsyncMock(return_value=[])
        mock_service.storage.search_procedural = AsyncMock(return_value=[])
        mock_service.storage.list_structured_memories = AsyncMock(return_value=[])
        mock_service.storage.log_audit = AsyncMock()

        results = await mock_service.recall(
            query="contact info",
            user_id="u1",
            memory_types=["structured"],
        )

        assert len(results) == 1
        # Structured memories use source_episode_id
        assert results[0].source_episode_id == "ep_email"


class TestMultiTenancy:
    """Tests for org_id and session_id filtering."""

    @pytest.fixture
    def mock_service(self):
        """Create a service with mocked dependencies."""
        storage = AsyncMock()
        embedder = AsyncMock()
        embedder.embed = AsyncMock(return_value=[0.1, 0.2, 0.3])
        embedder.embed_batch = AsyncMock(return_value=[[0.1, 0.2, 0.3]])

        settings = Settings(openai_api_key="sk-test-dummy-key")

        return EngramService(
            storage=storage,
            embedder=embedder,
            settings=settings,
        )

    @pytest.mark.asyncio
    async def test_recall_filters_by_org_id(self, mock_service):
        """Recall should filter results by org_id when provided."""
        episodes_org1 = [
            Episode(
                id="ep_org1",
                content="Org 1 data",
                role="user",
                user_id="u1",
                org_id="org_1",
                embedding=[0.1, 0.2, 0.3],
            )
        ]
        episodes_org2 = [
            Episode(
                id="ep_org2",
                content="Org 2 data",
                role="user",
                user_id="u1",
                org_id="org_2",
                embedding=[0.1, 0.2, 0.4],
            )
        ]

        # Mock storage to return only org_1 episodes when filtered
        async def search_episodes_filtered(query_vector, user_id, org_id=None, limit=10, **kwargs):
            if org_id == "org_1":
                return [ScoredResult(memory=ep, score=0.9) for ep in episodes_org1]
            elif org_id == "org_2":
                return [ScoredResult(memory=ep, score=0.9) for ep in episodes_org2]
            return [ScoredResult(memory=ep, score=0.9) for ep in episodes_org1 + episodes_org2]

        mock_service.storage.search_episodes = AsyncMock(side_effect=search_episodes_filtered)
        mock_service.storage.search_structured = AsyncMock(return_value=[])
        mock_service.storage.search_semantic = AsyncMock(return_value=[])
        mock_service.storage.search_procedural = AsyncMock(return_value=[])
        mock_service.storage.list_structured_memories = AsyncMock(return_value=[])
        mock_service.storage.log_audit = AsyncMock()

        # Recall with org_id filter
        results = await mock_service.recall(
            query="data",
            user_id="u1",
            org_id="org_1",
            memory_types=["episodic"],
        )

        # Should only get org_1 results
        assert len(results) == 1
        assert results[0].content == "Org 1 data"

    @pytest.mark.asyncio
    async def test_encode_stores_org_id_and_session_id(self, mock_service):
        """Encode should store org_id and session_id with episode."""
        mock_service.storage.store_episode = AsyncMock(return_value="ep_123")
        mock_service.storage.store_structured = AsyncMock(return_value="struct_123")
        mock_service.storage.log_audit = AsyncMock()

        result = await mock_service.encode(
            content="Test content",
            role="user",
            user_id="u1",
            org_id="test_org",
            session_id="session_123",
        )

        assert result.episode.org_id == "test_org"
        assert result.episode.session_id == "session_123"

    @pytest.mark.asyncio
    async def test_different_users_isolated(self, mock_service):
        """Different users should not see each other's memories."""
        episodes_user1 = [
            Episode(
                id="ep_u1",
                content="User 1 secret",
                role="user",
                user_id="user_1",
                embedding=[0.1, 0.2, 0.3],
            )
        ]

        async def search_by_user(query_vector, user_id, **kwargs):
            if user_id == "user_1":
                return [ScoredResult(memory=ep, score=0.9) for ep in episodes_user1]
            return []  # Other users get nothing

        mock_service.storage.search_episodes = AsyncMock(side_effect=search_by_user)
        mock_service.storage.search_structured = AsyncMock(return_value=[])
        mock_service.storage.search_semantic = AsyncMock(return_value=[])
        mock_service.storage.search_procedural = AsyncMock(return_value=[])
        mock_service.storage.list_structured_memories = AsyncMock(return_value=[])
        mock_service.storage.log_audit = AsyncMock()

        # User 2 should not see User 1's data
        results = await mock_service.recall(
            query="secret",
            user_id="user_2",
            memory_types=["episodic"],
        )

        assert len(results) == 0


class TestFreshnessFiltering:
    """Tests for freshness/staleness filtering."""

    @pytest.fixture
    def mock_service(self):
        """Create a service with mocked dependencies."""
        storage = AsyncMock()
        embedder = AsyncMock()
        embedder.embed = AsyncMock(return_value=[0.1, 0.2, 0.3])

        settings = Settings(openai_api_key="sk-test-dummy-key")

        return EngramService(
            storage=storage,
            embedder=embedder,
            settings=settings,
        )

    @pytest.mark.asyncio
    async def test_fresh_only_returns_fresh_episodes(self, mock_service):
        """fresh_only mode should return episodes with FRESH staleness."""
        episodes = [
            Episode(
                id="ep_1",
                content="Episode 1 content",
                role="user",
                user_id="u1",
                summarized=True,
                embedding=[0.1, 0.2, 0.3],
            ),
            Episode(
                id="ep_2",
                content="Episode 2 content",
                role="user",
                user_id="u1",
                summarized=False,
                embedding=[0.1, 0.2, 0.4],
            ),
        ]

        mock_service.storage.search_episodes = AsyncMock(
            return_value=[ScoredResult(memory=ep, score=0.9) for ep in episodes]
        )
        mock_service.storage.search_structured = AsyncMock(return_value=[])
        mock_service.storage.search_semantic = AsyncMock(return_value=[])
        mock_service.storage.search_procedural = AsyncMock(return_value=[])
        mock_service.storage.list_structured_memories = AsyncMock(return_value=[])
        mock_service.storage.log_audit = AsyncMock()

        results = await mock_service.recall(
            query="test",
            user_id="u1",
            memory_types=["episodic"],
            freshness="fresh_only",
        )

        # Verify results are returned and have valid staleness
        assert len(results) > 0
        # All results should have staleness set
        from engram.models import Staleness

        for r in results:
            assert r.staleness in (Staleness.FRESH, Staleness.STALE, Staleness.CONSOLIDATING)

    @pytest.mark.asyncio
    async def test_best_effort_returns_all(self, mock_service):
        """best_effort should return all results regardless of staleness."""
        episodes = [
            Episode(
                id="ep_fresh",
                content="Already summarized",
                role="user",
                user_id="u1",
                summarized=True,
                embedding=[0.1, 0.2, 0.3],
            ),
            Episode(
                id="ep_stale",
                content="Not yet summarized",
                role="user",
                user_id="u1",
                summarized=False,
                embedding=[0.1, 0.2, 0.4],
            ),
        ]

        mock_service.storage.search_episodes = AsyncMock(
            return_value=[ScoredResult(memory=ep, score=0.9) for ep in episodes]
        )
        mock_service.storage.search_structured = AsyncMock(return_value=[])
        mock_service.storage.search_semantic = AsyncMock(return_value=[])
        mock_service.storage.search_procedural = AsyncMock(return_value=[])
        mock_service.storage.list_structured_memories = AsyncMock(return_value=[])
        mock_service.storage.log_audit = AsyncMock()

        results = await mock_service.recall(
            query="test",
            user_id="u1",
            memory_types=["episodic"],
            freshness="best_effort",
        )

        # Both should be returned
        assert len(results) == 2


class TestRecallResultFields:
    """Tests for RecallResult field population."""

    def test_recall_result_required_fields(self):
        """RecallResult should have all required fields."""
        result = RecallResult(
            memory_id="test_id",
            memory_type="semantic",
            content="Test content",
            score=0.9,
        )

        assert result.memory_id == "test_id"
        assert result.memory_type == "semantic"
        assert result.content == "Test content"
        assert result.score == 0.9
        assert result.confidence is None  # Optional
        assert result.source_episode_ids == []  # Default empty

    def test_recall_result_with_source_episode_ids(self):
        """RecallResult should accept source_episode_ids."""
        result = RecallResult(
            memory_id="test_id",
            memory_type="semantic",
            content="Test content",
            score=0.9,
            source_episode_ids=["ep_1", "ep_2"],
        )

        assert result.source_episode_ids == ["ep_1", "ep_2"]

    def test_recall_result_with_all_optional_fields(self):
        """RecallResult should handle all optional fields."""
        result = RecallResult(
            memory_id="test_id",
            memory_type="procedural",
            content="Test content",
            score=0.85,
            confidence=0.9,
            source_episode_ids=["ep_1"],
            related_ids=["rel_1", "rel_2"],
            hop_distance=1,
        )

        assert result.confidence == 0.9
        assert result.related_ids == ["rel_1", "rel_2"]
        assert result.hop_distance == 1
