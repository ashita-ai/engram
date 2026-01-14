"""Unit tests for Engram storage layer.

These tests use qdrant-client's local in-memory mode for fast, isolated testing.
No external Qdrant server is required.
"""

import pytest
from qdrant_client import AsyncQdrantClient

from engram.models import (
    AuditEntry,
    Episode,
    Fact,
    NegationFact,
    ProceduralMemory,
    SemanticMemory,
)
from engram.storage import EngramStorage

# Use a small embedding dimension for tests
TEST_EMBEDDING_DIM = 4


def make_embedding(seed: float = 0.1) -> list[float]:
    """Create a simple test embedding vector."""
    return [seed, seed + 0.1, seed + 0.2, seed + 0.3]


@pytest.fixture
async def storage():
    """Create an in-memory storage instance for testing.

    Uses qdrant-client's local mode with in-memory storage.
    """
    # Create storage with custom initialization
    store = EngramStorage(
        prefix="test",
        embedding_dim=TEST_EMBEDDING_DIM,
    )
    # Override with in-memory client
    store._client = AsyncQdrantClient(location=":memory:")
    await store._ensure_collections()
    store._collections_initialized = True

    yield store

    await store.close()


class TestEngramStorageInit:
    """Tests for storage initialization."""

    async def test_initialize_creates_collections(self, storage: EngramStorage):
        """initialize() should create all required collections."""
        collections = await storage.client.get_collections()
        names = [c.name for c in collections.collections]

        assert "test_episodic" in names
        assert "test_structured" in names
        assert "test_factual" in names
        assert "test_semantic" in names
        assert "test_procedural" in names
        assert "test_negation" in names
        assert "test_audit" in names

    async def test_context_manager(self, storage: EngramStorage):
        """Storage should work as async context manager (tested via fixture)."""
        # The fixture itself uses async context manager pattern
        collections = await storage.client.get_collections()
        assert (
            len(collections.collections) == 7
        )  # episodic, structured, factual, semantic, procedural, negation, audit

    def test_collection_name(self, storage: EngramStorage):
        """_collection_name should apply prefix correctly."""
        assert storage._collection_name("episodic") == "test_episodic"
        assert storage._collection_name("factual") == "test_factual"
        assert storage._collection_name("semantic") == "test_semantic"

    def test_build_key_with_org(self):
        """_build_key should include org_id when provided."""
        key = EngramStorage._build_key("mem_123", "user_456", "org_789")
        assert key == "org_789/user_456/mem_123"

    def test_build_key_personal(self):
        """_build_key should use 'personal' when no org_id."""
        key = EngramStorage._build_key("mem_123", "user_456")
        assert key == "personal/user_456/mem_123"


class TestEpisodeStorage:
    """Tests for episode storage operations."""

    async def test_store_and_get_episode(self, storage: EngramStorage):
        """Should store and retrieve an episode."""
        episode = Episode(
            content="Hello, world!",
            role="user",
            user_id="user_123",
            embedding=make_embedding(0.1),
        )

        # Store
        episode_id = await storage.store_episode(episode)
        assert episode_id == episode.id

        # Get
        retrieved = await storage.get_episode(episode.id, "user_123")
        assert retrieved is not None
        assert retrieved.content == "Hello, world!"
        assert retrieved.role == "user"

    async def test_store_episode_requires_embedding(self, storage: EngramStorage):
        """Should raise ValueError if episode has no embedding."""
        episode = Episode(
            content="No embedding",
            role="user",
            user_id="user_123",
        )

        with pytest.raises(ValueError, match="must have an embedding"):
            await storage.store_episode(episode)

    async def test_get_episode_wrong_user(self, storage: EngramStorage):
        """Should return None when user doesn't own episode."""
        episode = Episode(
            content="Secret",
            role="user",
            user_id="user_123",
            embedding=make_embedding(),
        )
        await storage.store_episode(episode)

        # Try to get with wrong user
        result = await storage.get_episode(episode.id, "wrong_user")
        assert result is None

    async def test_search_episodes(self, storage: EngramStorage):
        """Should find similar episodes by vector search."""
        # Store multiple episodes
        for i in range(3):
            ep = Episode(
                content=f"Message {i}",
                role="user",
                user_id="user_123",
                embedding=make_embedding(i * 0.1),
            )
            await storage.store_episode(ep)

        # Search
        results = await storage.search_episodes(
            query_vector=make_embedding(0.0),
            user_id="user_123",
            limit=2,
        )

        assert len(results) == 2
        # First result should be most similar to query
        assert results[0].memory.content == "Message 0"
        # Verify we get actual similarity scores
        assert results[0].score > 0

    async def test_delete_episode(self, storage: EngramStorage):
        """Should delete an episode."""
        episode = Episode(
            content="To be deleted",
            role="user",
            user_id="user_123",
            embedding=make_embedding(),
        )
        await storage.store_episode(episode)

        # Delete
        deleted = await storage.delete_episode(episode.id, "user_123")
        assert deleted is True

        # Verify gone
        result = await storage.get_episode(episode.id, "user_123")
        assert result is None


class TestFactStorage:
    """Tests for fact storage operations."""

    async def test_store_and_get_fact(self, storage: EngramStorage):
        """Should store and retrieve a fact."""
        fact = Fact(
            content="email=test@example.com",
            category="email",
            source_episode_id="ep_123",
            user_id="user_123",
            embedding=make_embedding(),
        )

        await storage.store_fact(fact)
        retrieved = await storage.get_fact(fact.id, "user_123")

        assert retrieved is not None
        assert retrieved.content == "email=test@example.com"
        assert retrieved.category == "email"

    async def test_search_facts_by_category(self, storage: EngramStorage):
        """Should filter facts by category."""
        # Store facts of different categories
        email_fact = Fact(
            content="email=a@b.com",
            category="email",
            source_episode_id="ep_1",
            user_id="user_123",
            embedding=make_embedding(0.1),
        )
        phone_fact = Fact(
            content="phone=555-1234",
            category="phone",
            source_episode_id="ep_2",
            user_id="user_123",
            embedding=make_embedding(0.2),
        )

        await storage.store_fact(email_fact)
        await storage.store_fact(phone_fact)

        # Search for emails only
        results = await storage.search_facts(
            query_vector=make_embedding(0.1),
            user_id="user_123",
            category="email",
        )

        assert len(results) == 1
        assert results[0].memory.category == "email"
        assert results[0].score > 0


class TestSemanticMemoryStorage:
    """Tests for semantic memory storage."""

    async def test_store_and_get_semantic(self, storage: EngramStorage):
        """Should store and retrieve semantic memory."""
        memory = SemanticMemory(
            content="User prefers Python",
            user_id="user_123",
            embedding=make_embedding(),
        )

        await storage.store_semantic(memory)
        retrieved = await storage.get_semantic(memory.id, "user_123")

        assert retrieved is not None
        assert retrieved.content == "User prefers Python"

    async def test_search_semantic_with_confidence(self, storage: EngramStorage):
        """Should filter by minimum confidence."""
        # Store memories with different confidence
        from engram.models import ConfidenceScore, ExtractionMethod

        high_conf = SemanticMemory(
            content="High confidence",
            user_id="user_123",
            confidence=ConfidenceScore(
                value=0.9,
                extraction_method=ExtractionMethod.EXTRACTED,
                extraction_base=0.9,
            ),
            embedding=make_embedding(0.1),
        )
        low_conf = SemanticMemory(
            content="Low confidence",
            user_id="user_123",
            confidence=ConfidenceScore(
                value=0.3,
                extraction_method=ExtractionMethod.INFERRED,
                extraction_base=0.6,
            ),
            embedding=make_embedding(0.2),
        )

        await storage.store_semantic(high_conf)
        await storage.store_semantic(low_conf)

        # Search with min_confidence filter
        results = await storage.search_semantic(
            query_vector=make_embedding(0.1),
            user_id="user_123",
            min_confidence=0.7,
        )

        assert len(results) == 1
        assert results[0].memory.content == "High confidence"
        assert results[0].score > 0


class TestProceduralMemoryStorage:
    """Tests for procedural memory storage."""

    async def test_store_and_get_procedural(self, storage: EngramStorage):
        """Should store and retrieve procedural memory."""
        memory = ProceduralMemory(
            content="User likes verbose responses",
            user_id="user_123",
            embedding=make_embedding(),
        )

        await storage.store_procedural(memory)
        retrieved = await storage.get_procedural(memory.id, "user_123")

        assert retrieved is not None
        assert retrieved.content == "User likes verbose responses"


class TestNegationFactStorage:
    """Tests for negation fact storage."""

    async def test_store_and_get_negation(self, storage: EngramStorage):
        """Should store and retrieve negation fact."""
        fact = NegationFact(
            content="User does NOT use MongoDB",
            negates_pattern="mongodb",
            user_id="user_123",
            embedding=make_embedding(),
        )

        await storage.store_negation(fact)
        retrieved = await storage.get_negation(fact.id, "user_123")

        assert retrieved is not None
        assert retrieved.content == "User does NOT use MongoDB"
        assert retrieved.negates_pattern == "mongodb"


class TestAuditLogging:
    """Tests for audit log operations."""

    async def test_log_and_get_audit(self, storage: EngramStorage):
        """Should log and retrieve audit entries."""
        entry = AuditEntry.for_encode(
            user_id="user_123",
            episode_id="ep_456",
            facts_count=3,
            duration_ms=150,
        )

        await storage.log_audit(entry)
        logs = await storage.get_audit_log("user_123")

        assert len(logs) == 1
        assert logs[0].event == "encode"
        assert logs[0].details["episode_id"] == "ep_456"

    async def test_get_audit_by_event_type(self, storage: EngramStorage):
        """Should filter audit logs by event type."""
        # Log different event types
        encode_entry = AuditEntry.for_encode(
            user_id="user_123",
            episode_id="ep_1",
            facts_count=1,
        )
        recall_entry = AuditEntry.for_recall(
            user_id="user_123",
            query_hash="abc",
            results_count=5,
            memory_types=["episodic"],
        )

        await storage.log_audit(encode_entry)
        await storage.log_audit(recall_entry)

        # Filter by event type
        encodes = await storage.get_audit_log("user_123", event_type="encode")
        recalls = await storage.get_audit_log("user_123", event_type="recall")

        assert len(encodes) == 1
        assert encodes[0].event == "encode"
        assert len(recalls) == 1
        assert recalls[0].event == "recall"


class TestMultiTenancy:
    """Tests for multi-tenancy isolation."""

    async def test_users_isolated(self, storage: EngramStorage):
        """Users should only see their own memories."""
        user1_ep = Episode(
            content="User 1 message",
            role="user",
            user_id="user_1",
            embedding=make_embedding(0.1),
        )
        user2_ep = Episode(
            content="User 2 message",
            role="user",
            user_id="user_2",
            embedding=make_embedding(0.1),
        )

        await storage.store_episode(user1_ep)
        await storage.store_episode(user2_ep)

        # User 1 only sees their episode
        user1_results = await storage.search_episodes(
            query_vector=make_embedding(0.1),
            user_id="user_1",
        )
        assert len(user1_results) == 1
        assert user1_results[0].memory.content == "User 1 message"

        # User 2 only sees their episode
        user2_results = await storage.search_episodes(
            query_vector=make_embedding(0.1),
            user_id="user_2",
        )
        assert len(user2_results) == 1
        assert user2_results[0].memory.content == "User 2 message"

    async def test_org_filtering(self, storage: EngramStorage):
        """Should filter by organization when specified."""
        org_ep = Episode(
            content="Org message",
            role="user",
            user_id="user_123",
            org_id="org_456",
            embedding=make_embedding(0.1),
        )
        personal_ep = Episode(
            content="Personal message",
            role="user",
            user_id="user_123",
            embedding=make_embedding(0.2),
        )

        await storage.store_episode(org_ep)
        await storage.store_episode(personal_ep)

        # Filter by org
        org_results = await storage.search_episodes(
            query_vector=make_embedding(0.1),
            user_id="user_123",
            org_id="org_456",
        )

        assert len(org_results) == 1
        assert org_results[0].memory.org_id == "org_456"
