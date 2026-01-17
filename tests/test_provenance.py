"""Tests for memory provenance system.

Tests the complete provenance tracking from source episodes
through derived memories (StructuredMemory, SemanticMemory, ProceduralMemory).
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from engram.config import Settings
from engram.models import (
    Episode,
    ProceduralMemory,
    ProvenanceChain,
    ProvenanceEvent,
    SemanticMemory,
    StructuredMemory,
)
from engram.service import EngramService


class TestProvenanceModels:
    """Tests for provenance models."""

    def test_provenance_event_creation(self):
        """Should create provenance event with all fields."""
        event = ProvenanceEvent(
            timestamp=datetime.now(UTC),
            event_type="stored",
            description="Episode stored",
            memory_id="ep_123",
            metadata={"key": "value"},
        )

        assert event.event_type == "stored"
        assert event.description == "Episode stored"
        assert event.memory_id == "ep_123"
        assert event.metadata == {"key": "value"}

    def test_provenance_chain_creation(self):
        """Should create provenance chain with all fields."""
        chain = ProvenanceChain(
            memory_id="sem_123",
            memory_type="semantic",
            derivation_method="consolidation:openai:gpt-4o-mini",
            derivation_reasoning="Synthesized from 3 episodes",
            derived_at=datetime.now(UTC),
            source_episodes=[
                {
                    "id": "ep_1",
                    "content": "Hello",
                    "role": "user",
                    "timestamp": "2026-01-15T10:00:00Z",
                },
            ],
            intermediate_memories=[
                {
                    "id": "struct_1",
                    "type": "structured",
                    "summary": "Test",
                    "derivation_method": "rich:llm:gpt-4o-mini",
                    "derived_at": "2026-01-15T10:01:00Z",
                },
            ],
            timeline=[
                ProvenanceEvent(
                    timestamp=datetime.now(UTC),
                    event_type="stored",
                    description="Episode stored",
                ),
            ],
        )

        assert chain.memory_id == "sem_123"
        assert chain.memory_type == "semantic"
        assert chain.derivation_method == "consolidation:openai:gpt-4o-mini"
        assert len(chain.source_episodes) == 1
        assert len(chain.intermediate_memories) == 1
        assert len(chain.timeline) == 1


class TestDerivationMethod:
    """Tests for derivation_method field on memory models."""

    def test_structured_memory_fast_mode_default(self):
        """Fast mode StructuredMemory should have derivation_method='fast:regex'."""
        structured = StructuredMemory.from_episode_fast(
            source_episode_id="ep_123",
            user_id="user_1",
            emails=["test@example.com"],
        )

        assert structured.derivation_method == "fast:regex"

    def test_structured_memory_rich_mode_default(self):
        """Rich mode StructuredMemory should use provided derivation_method."""
        structured = StructuredMemory.from_episode(
            source_episode_id="ep_123",
            user_id="user_1",
            summary="Test summary",
            derivation_method="rich:llm:gpt-4o-mini",
        )

        assert structured.derivation_method == "rich:llm:gpt-4o-mini"

    def test_structured_memory_rich_mode_defaults_to_unknown(self):
        """Rich mode without derivation_method should default to unknown."""
        structured = StructuredMemory.from_episode(
            source_episode_id="ep_123",
            user_id="user_1",
            summary="Test summary",
        )

        assert structured.derivation_method == "rich:llm:unknown"

    def test_semantic_memory_derivation_method_default(self):
        """SemanticMemory should have default derivation_method."""
        semantic = SemanticMemory(
            content="Test content",
            source_episode_ids=["ep_123"],
            user_id="user_1",
        )

        assert semantic.derivation_method == "llm:unknown"

    def test_semantic_memory_derivation_method_set(self):
        """SemanticMemory derivation_method can be set explicitly."""
        semantic = SemanticMemory(
            content="Test content",
            source_episode_ids=["ep_123"],
            user_id="user_1",
            derivation_method="consolidation:openai:gpt-4o-mini",
        )

        assert semantic.derivation_method == "consolidation:openai:gpt-4o-mini"

    def test_procedural_memory_derivation_method_default(self):
        """ProceduralMemory should have default derivation_method."""
        procedural = ProceduralMemory(
            content="Test behavioral pattern",
            user_id="user_1",
        )

        assert procedural.derivation_method == "synthesis:unknown"

    def test_procedural_memory_derivation_method_set(self):
        """ProceduralMemory derivation_method can be set explicitly."""
        procedural = ProceduralMemory(
            content="Test behavioral pattern",
            user_id="user_1",
            derivation_method="synthesis:openai:gpt-4o-mini",
        )

        assert procedural.derivation_method == "synthesis:openai:gpt-4o-mini"


class TestGetProvenance:
    """Tests for EngramService.get_provenance method."""

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
    async def test_get_provenance_structured(self, mock_service):
        """Should return provenance for StructuredMemory."""
        # Setup mocks
        episode = Episode(
            id="ep_123",
            content="Test content",
            role="user",
            user_id="user_1",
            embedding=[0.1, 0.2, 0.3],
        )

        structured = StructuredMemory.from_episode_fast(
            source_episode_id="ep_123",
            user_id="user_1",
            emails=["test@example.com"],
        )

        mock_service.storage.get_structured = AsyncMock(return_value=structured)
        mock_service.storage.get_episode = AsyncMock(return_value=episode)

        # Get provenance
        chain = await mock_service.get_provenance("struct_123", "user_1")

        assert chain.memory_type == "structured"
        assert chain.derivation_method == "fast:regex"
        assert len(chain.source_episodes) == 1
        assert chain.source_episodes[0]["id"] == "ep_123"
        assert len(chain.timeline) >= 2  # stored + extracted

    @pytest.mark.asyncio
    async def test_get_provenance_semantic(self, mock_service):
        """Should return provenance for SemanticMemory with intermediates."""
        # Setup mocks
        episode = Episode(
            id="ep_123",
            content="Test content",
            role="user",
            user_id="user_1",
            embedding=[0.1, 0.2, 0.3],
        )

        structured = StructuredMemory.from_episode(
            source_episode_id="ep_123",
            user_id="user_1",
            summary="Test summary",
            derivation_method="rich:llm:gpt-4o-mini",
        )

        semantic = SemanticMemory(
            content="Consolidated content",
            source_episode_ids=["ep_123"],
            user_id="user_1",
            derivation_method="consolidation:openai:gpt-4o-mini",
        )

        mock_service.storage.get_semantic = AsyncMock(return_value=semantic)
        mock_service.storage.get_episode = AsyncMock(return_value=episode)
        mock_service.storage.get_structured_for_episode = AsyncMock(return_value=structured)

        # Get provenance
        chain = await mock_service.get_provenance("sem_123", "user_1")

        assert chain.memory_type == "semantic"
        assert chain.derivation_method == "consolidation:openai:gpt-4o-mini"
        assert len(chain.source_episodes) == 1
        assert len(chain.intermediate_memories) == 1
        assert chain.intermediate_memories[0]["type"] == "structured"

    @pytest.mark.asyncio
    async def test_get_provenance_procedural(self, mock_service):
        """Should return provenance for ProceduralMemory with semantic intermediates."""
        # Setup mocks
        episode = Episode(
            id="ep_123",
            content="Test content",
            role="user",
            user_id="user_1",
            embedding=[0.1, 0.2, 0.3],
        )

        semantic = SemanticMemory(
            id="sem_456",
            content="Semantic content",
            source_episode_ids=["ep_123"],
            user_id="user_1",
            derivation_method="consolidation:openai:gpt-4o-mini",
        )

        procedural = ProceduralMemory(
            content="Behavioral pattern",
            source_episode_ids=["ep_123"],
            source_semantic_ids=["sem_456"],
            user_id="user_1",
            derivation_method="synthesis:openai:gpt-4o-mini",
        )

        mock_service.storage.get_procedural = AsyncMock(return_value=procedural)
        mock_service.storage.get_semantic = AsyncMock(return_value=semantic)
        mock_service.storage.get_episode = AsyncMock(return_value=episode)

        # Get provenance
        chain = await mock_service.get_provenance("proc_123", "user_1")

        assert chain.memory_type == "procedural"
        assert chain.derivation_method == "synthesis:openai:gpt-4o-mini"
        assert len(chain.source_episodes) == 1
        assert len(chain.intermediate_memories) == 1
        assert chain.intermediate_memories[0]["type"] == "semantic"

    @pytest.mark.asyncio
    async def test_get_provenance_not_found(self, mock_service):
        """Should raise KeyError if memory not found."""
        mock_service.storage.get_structured = AsyncMock(return_value=None)

        with pytest.raises(KeyError):
            await mock_service.get_provenance("struct_notfound", "user_1")

    @pytest.mark.asyncio
    async def test_get_provenance_invalid_prefix(self, mock_service):
        """Should raise ValueError for invalid memory ID prefix."""
        with pytest.raises(ValueError):
            await mock_service.get_provenance("invalid_123", "user_1")


class TestProvenanceTimeline:
    """Tests for provenance timeline construction."""

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
    async def test_timeline_sorted_chronologically(self, mock_service):
        """Timeline events should be sorted by timestamp."""
        # Setup mocks
        episode = Episode(
            id="ep_123",
            content="Test content",
            role="user",
            user_id="user_1",
            embedding=[0.1, 0.2, 0.3],
        )

        structured = StructuredMemory.from_episode_fast(
            source_episode_id="ep_123",
            user_id="user_1",
            emails=["test@example.com"],
        )

        mock_service.storage.get_structured = AsyncMock(return_value=structured)
        mock_service.storage.get_episode = AsyncMock(return_value=episode)

        # Get provenance
        chain = await mock_service.get_provenance("struct_123", "user_1")

        # Check timeline is sorted
        timestamps = [e.timestamp for e in chain.timeline]
        assert timestamps == sorted(timestamps)

    @pytest.mark.asyncio
    async def test_timeline_includes_event_types(self, mock_service):
        """Timeline should include correct event types."""
        # Setup mocks
        episode = Episode(
            id="ep_123",
            content="Test content",
            role="user",
            user_id="user_1",
            embedding=[0.1, 0.2, 0.3],
        )

        structured = StructuredMemory.from_episode_fast(
            source_episode_id="ep_123",
            user_id="user_1",
            emails=["test@example.com"],
        )

        mock_service.storage.get_structured = AsyncMock(return_value=structured)
        mock_service.storage.get_episode = AsyncMock(return_value=episode)

        # Get provenance
        chain = await mock_service.get_provenance("struct_123", "user_1")

        event_types = [e.event_type for e in chain.timeline]
        assert "stored" in event_types
        assert "extracted" in event_types
