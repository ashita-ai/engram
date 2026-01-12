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
    Fact,
    NegationFact,
    ProceduralMemory,
    SemanticMemory,
)
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
            pipeline=MagicMock(),
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

        # Negation: user doesn't use MongoDB anymore
        negations = [
            NegationFact(
                id="neg_mongo",
                content="I don't use MongoDB anymore",
                negates_pattern="MongoDB",
                source_episode_ids=["ep_neg"],
                user_id="u1",
                confidence=ConfidenceScore.for_extracted(),
            )
        ]

        mock_service.storage.search_episodes = AsyncMock(
            return_value=[ScoredResult(memory=ep, score=0.9) for ep in episodes]
        )
        mock_service.storage.search_facts = AsyncMock(return_value=[])
        mock_service.storage.search_semantic = AsyncMock(return_value=[])
        mock_service.storage.search_procedural = AsyncMock(return_value=[])
        mock_service.storage.search_negation = AsyncMock(return_value=[])
        mock_service.storage.list_negation_facts = AsyncMock(return_value=negations)
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

        negations = [
            NegationFact(
                id="neg_mongo",
                content="I don't use MongoDB anymore",
                negates_pattern="MongoDB",
                source_episode_ids=["ep_neg"],
                user_id="u1",
                confidence=ConfidenceScore.for_extracted(),
            )
        ]

        mock_service.storage.search_episodes = AsyncMock(
            return_value=[ScoredResult(memory=ep, score=0.9) for ep in episodes]
        )
        mock_service.storage.search_facts = AsyncMock(return_value=[])
        mock_service.storage.search_semantic = AsyncMock(return_value=[])
        mock_service.storage.search_procedural = AsyncMock(return_value=[])
        mock_service.storage.search_negation = AsyncMock(return_value=[])
        mock_service.storage.list_negation_facts = AsyncMock(return_value=negations)
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
            pipeline=MagicMock(),
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

        negations = [
            NegationFact(
                id="neg_mongo",
                content="I don't use MongoDB",
                negates_pattern="MongoDB",
                source_episode_ids=["ep_neg"],
                user_id="u1",
                confidence=ConfidenceScore.for_extracted(),
            )
        ]

        mock_service.storage.search_episodes = AsyncMock(
            return_value=[
                ScoredResult(memory=ep, score=0.9 - i * 0.05) for i, ep in enumerate(episodes)
            ]
        )
        mock_service.storage.search_facts = AsyncMock(return_value=[])
        mock_service.storage.search_semantic = AsyncMock(return_value=[])
        mock_service.storage.search_procedural = AsyncMock(return_value=[])
        mock_service.storage.search_negation = AsyncMock(return_value=[])
        mock_service.storage.list_negation_facts = AsyncMock(return_value=negations)
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
        mock_service.storage.search_facts = AsyncMock(return_value=[])
        mock_service.storage.search_semantic = AsyncMock(return_value=[])
        mock_service.storage.search_procedural = AsyncMock(return_value=[])
        mock_service.storage.search_negation = AsyncMock(return_value=[])
        mock_service.storage.list_negation_facts = AsyncMock(return_value=[])
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
            pipeline=MagicMock(),
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
        mock_service.storage.search_facts = AsyncMock(return_value=[])
        mock_service.storage.search_semantic = AsyncMock(
            return_value=[ScoredResult(memory=semantic, score=0.9)]
        )
        mock_service.storage.search_procedural = AsyncMock(return_value=[])
        mock_service.storage.search_negation = AsyncMock(return_value=[])
        mock_service.storage.list_negation_facts = AsyncMock(return_value=[])
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
        mock_service.storage.search_facts = AsyncMock(return_value=[])
        mock_service.storage.search_semantic = AsyncMock(return_value=[])
        mock_service.storage.search_procedural = AsyncMock(
            return_value=[ScoredResult(memory=procedural, score=0.85)]
        )
        mock_service.storage.search_negation = AsyncMock(return_value=[])
        mock_service.storage.list_negation_facts = AsyncMock(return_value=[])
        mock_service.storage.log_audit = AsyncMock()

        results = await mock_service.recall(
            query="coding practices",
            user_id="u1",
            memory_types=["procedural"],
        )

        assert len(results) == 1
        assert results[0].source_episode_ids == ["ep_a", "ep_b"]

    @pytest.mark.asyncio
    async def test_negation_memory_includes_source_episode_ids(self, mock_service):
        """RecallResult for negation facts should include source_episode_ids."""
        negation = NegationFact(
            id="neg_1",
            content="User doesn't use Windows",
            negates_pattern="Windows",
            source_episode_ids=["ep_win"],
            user_id="u1",
            confidence=ConfidenceScore.for_extracted(),
        )

        mock_service.storage.search_episodes = AsyncMock(return_value=[])
        mock_service.storage.search_facts = AsyncMock(return_value=[])
        mock_service.storage.search_semantic = AsyncMock(return_value=[])
        mock_service.storage.search_procedural = AsyncMock(return_value=[])
        mock_service.storage.search_negation = AsyncMock(
            return_value=[ScoredResult(memory=negation, score=0.88)]
        )
        mock_service.storage.list_negation_facts = AsyncMock(return_value=[])
        mock_service.storage.log_audit = AsyncMock()

        results = await mock_service.recall(
            query="operating system",
            user_id="u1",
            memory_types=["negation"],
        )

        assert len(results) == 1
        assert results[0].source_episode_ids == ["ep_win"]

    @pytest.mark.asyncio
    async def test_fact_includes_source_episode_id(self, mock_service):
        """RecallResult for facts should include source_episode_id."""
        fact = Fact(
            id="fact_1",
            content="user@example.com",
            category="email",
            source_episode_id="ep_email",
            user_id="u1",
            confidence=ConfidenceScore.for_extracted(),
        )

        mock_service.storage.search_episodes = AsyncMock(return_value=[])
        mock_service.storage.search_facts = AsyncMock(
            return_value=[ScoredResult(memory=fact, score=0.95)]
        )
        mock_service.storage.search_semantic = AsyncMock(return_value=[])
        mock_service.storage.search_procedural = AsyncMock(return_value=[])
        mock_service.storage.search_negation = AsyncMock(return_value=[])
        mock_service.storage.list_negation_facts = AsyncMock(return_value=[])
        mock_service.storage.log_audit = AsyncMock()

        results = await mock_service.recall(
            query="contact info",
            user_id="u1",
            memory_types=["factual"],
        )

        assert len(results) == 1
        # Facts use source_episode_id (singular) not source_episode_ids
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
            pipeline=MagicMock(),
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
        mock_service.storage.search_facts = AsyncMock(return_value=[])
        mock_service.storage.search_semantic = AsyncMock(return_value=[])
        mock_service.storage.search_procedural = AsyncMock(return_value=[])
        mock_service.storage.search_negation = AsyncMock(return_value=[])
        mock_service.storage.list_negation_facts = AsyncMock(return_value=[])
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
        mock_service.pipeline.run = MagicMock(return_value=[])
        mock_service.storage.store_episode = AsyncMock(return_value="ep_123")
        mock_service.storage.store_negation = AsyncMock(return_value="neg_123")
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
        mock_service.storage.search_facts = AsyncMock(return_value=[])
        mock_service.storage.search_semantic = AsyncMock(return_value=[])
        mock_service.storage.search_procedural = AsyncMock(return_value=[])
        mock_service.storage.search_negation = AsyncMock(return_value=[])
        mock_service.storage.list_negation_facts = AsyncMock(return_value=[])
        mock_service.storage.log_audit = AsyncMock()

        # User 2 should not see User 1's data
        results = await mock_service.recall(
            query="secret",
            user_id="user_2",
            memory_types=["episodic"],
        )

        assert len(results) == 0


class TestRIFEdgeCases:
    """Edge case tests for Retrieval-Induced Forgetting."""

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
            pipeline=MagicMock(),
            settings=settings,
        )

    @pytest.mark.asyncio
    async def test_rif_does_not_suppress_episodic(self, mock_service):
        """RIF should not suppress episodic memories (immutable ground truth)."""
        episodes = [
            Episode(
                id="ep_1",
                content="First episode",
                role="user",
                user_id="u1",
                embedding=[0.1, 0.2, 0.3],
            ),
            Episode(
                id="ep_2",
                content="Second episode",
                role="user",
                user_id="u1",
                embedding=[0.1, 0.2, 0.35],
            ),
        ]

        mock_service.storage.search_episodes = AsyncMock(
            return_value=[
                ScoredResult(memory=ep, score=0.9 - i * 0.1) for i, ep in enumerate(episodes)
            ]
        )
        mock_service.storage.search_facts = AsyncMock(return_value=[])
        mock_service.storage.search_semantic = AsyncMock(return_value=[])
        mock_service.storage.search_procedural = AsyncMock(return_value=[])
        mock_service.storage.search_negation = AsyncMock(return_value=[])
        mock_service.storage.list_negation_facts = AsyncMock(return_value=[])
        mock_service.storage.log_audit = AsyncMock()
        mock_service.storage.update_confidence = AsyncMock()

        # Enable RIF with limit=1 (should suppress ep_2 if it were semantic)
        await mock_service.recall(
            query="episode",
            user_id="u1",
            memory_types=["episodic"],
            limit=1,
            rif_enabled=True,
            rif_threshold=0.5,
        )

        # update_confidence should NOT be called for episodic memories
        mock_service.storage.update_confidence.assert_not_called()

    @pytest.mark.asyncio
    async def test_rif_suppresses_semantic_competitors(self, mock_service):
        """RIF should suppress semantic memories that compete but weren't retrieved."""
        sem_1 = SemanticMemory(
            id="sem_1",
            content="User likes Python",
            source_episode_ids=["ep_1"],
            user_id="u1",
            confidence=ConfidenceScore.for_inferred(),
        )
        sem_2 = SemanticMemory(
            id="sem_2",
            content="User also likes JavaScript",
            source_episode_ids=["ep_2"],
            user_id="u1",
            confidence=ConfidenceScore.for_inferred(),
        )
        semantics = [sem_1, sem_2]

        mock_service.storage.search_episodes = AsyncMock(return_value=[])
        mock_service.storage.search_facts = AsyncMock(return_value=[])
        mock_service.storage.search_semantic = AsyncMock(
            return_value=[
                ScoredResult(memory=sem, score=0.9 - i * 0.1) for i, sem in enumerate(semantics)
            ]
        )
        mock_service.storage.search_procedural = AsyncMock(return_value=[])
        mock_service.storage.search_negation = AsyncMock(return_value=[])
        mock_service.storage.list_negation_facts = AsyncMock(return_value=[])
        mock_service.storage.log_audit = AsyncMock()

        # Mock get_semantic to return the actual semantic memory for RIF
        async def get_semantic_side_effect(memory_id, user_id):
            if memory_id == "sem_2":
                return sem_2
            return None

        mock_service.storage.get_semantic = AsyncMock(side_effect=get_semantic_side_effect)
        mock_service.storage.update_semantic = AsyncMock()

        # Enable RIF with limit=1 (sem_2 should be suppressed)
        results = await mock_service.recall(
            query="programming languages",
            user_id="u1",
            memory_types=["semantic"],
            limit=1,
            rif_enabled=True,
            rif_threshold=0.5,
            rif_decay=0.1,
        )

        # Should return only sem_1
        assert len(results) == 1
        assert results[0].memory_id == "sem_1"

        # get_semantic should be called for sem_2 (the competitor)
        mock_service.storage.get_semantic.assert_called()

    @pytest.mark.asyncio
    async def test_rif_threshold_respected(self, mock_service):
        """RIF should only suppress memories above the similarity threshold."""
        semantics = [
            SemanticMemory(
                id="sem_high",
                content="High relevance",
                source_episode_ids=["ep_1"],
                user_id="u1",
                confidence=ConfidenceScore.for_inferred(),
            ),
            SemanticMemory(
                id="sem_low",
                content="Low relevance",
                source_episode_ids=["ep_2"],
                user_id="u1",
                confidence=ConfidenceScore.for_inferred(),
            ),
        ]

        mock_service.storage.search_episodes = AsyncMock(return_value=[])
        mock_service.storage.search_facts = AsyncMock(return_value=[])
        mock_service.storage.search_semantic = AsyncMock(
            return_value=[
                ScoredResult(memory=semantics[0], score=0.9),  # Above threshold
                ScoredResult(memory=semantics[1], score=0.3),  # Below threshold
            ]
        )
        mock_service.storage.search_procedural = AsyncMock(return_value=[])
        mock_service.storage.search_negation = AsyncMock(return_value=[])
        mock_service.storage.list_negation_facts = AsyncMock(return_value=[])
        mock_service.storage.log_audit = AsyncMock()
        mock_service.storage.update_confidence = AsyncMock()

        await mock_service.recall(
            query="test",
            user_id="u1",
            memory_types=["semantic"],
            limit=1,
            rif_enabled=True,
            rif_threshold=0.5,  # sem_low (0.3) is below this
        )

        # Check that update_confidence was not called for sem_low
        # (it's below threshold so shouldn't be suppressed)
        for call in mock_service.storage.update_confidence.call_args_list:
            memory_id = call[1].get("memory_id") or call[0][0]
            assert memory_id != "sem_low"

    @pytest.mark.asyncio
    async def test_rif_disabled_by_default(self, mock_service):
        """RIF should be disabled by default."""
        semantics = [
            SemanticMemory(
                id="sem_1",
                content="Semantic 1",
                source_episode_ids=["ep_1"],
                user_id="u1",
                confidence=ConfidenceScore.for_inferred(),
            ),
        ]

        mock_service.storage.search_episodes = AsyncMock(return_value=[])
        mock_service.storage.search_facts = AsyncMock(return_value=[])
        mock_service.storage.search_semantic = AsyncMock(
            return_value=[ScoredResult(memory=sem, score=0.9) for sem in semantics]
        )
        mock_service.storage.search_procedural = AsyncMock(return_value=[])
        mock_service.storage.search_negation = AsyncMock(return_value=[])
        mock_service.storage.list_negation_facts = AsyncMock(return_value=[])
        mock_service.storage.log_audit = AsyncMock()
        mock_service.storage.update_confidence = AsyncMock()

        # Don't pass rif_enabled (should default to False)
        await mock_service.recall(
            query="test",
            user_id="u1",
            memory_types=["semantic"],
        )

        # update_confidence should not be called
        mock_service.storage.update_confidence.assert_not_called()


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
            pipeline=MagicMock(),
            settings=settings,
        )

    @pytest.mark.asyncio
    async def test_fresh_only_excludes_unconsolidated(self, mock_service):
        """fresh_only should exclude episodes not yet consolidated."""
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
        mock_service.storage.search_facts = AsyncMock(return_value=[])
        mock_service.storage.search_semantic = AsyncMock(return_value=[])
        mock_service.storage.search_procedural = AsyncMock(return_value=[])
        mock_service.storage.search_negation = AsyncMock(return_value=[])
        mock_service.storage.list_negation_facts = AsyncMock(return_value=[])
        mock_service.storage.log_audit = AsyncMock()

        results = await mock_service.recall(
            query="test",
            user_id="u1",
            memory_types=["episodic"],
            freshness="fresh_only",
        )

        # Only the summarized episode should be returned
        assert len(results) == 1
        assert results[0].memory_id == "ep_fresh"

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
        mock_service.storage.search_facts = AsyncMock(return_value=[])
        mock_service.storage.search_semantic = AsyncMock(return_value=[])
        mock_service.storage.search_procedural = AsyncMock(return_value=[])
        mock_service.storage.search_negation = AsyncMock(return_value=[])
        mock_service.storage.list_negation_facts = AsyncMock(return_value=[])
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
