"""Unit tests for entity resolution module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from engram.entities import (
    Entity,
    EntityCluster,
    EntityMention,
    extract_entities,
    resolve_entities,
)


class TestEntityModel:
    """Tests for Entity model."""

    def test_create_entity(self):
        """Should create an entity with required fields."""
        entity = Entity(
            canonical_name="John Smith",
            entity_type="person",
            user_id="user_123",
        )
        assert entity.canonical_name == "John Smith"
        assert entity.entity_type == "person"
        assert entity.aliases == []
        assert entity.memory_ids == []

    def test_add_alias(self):
        """Should add aliases without duplicates."""
        entity = Entity(
            canonical_name="John Smith",
            entity_type="person",
            user_id="user_123",
        )
        entity.add_alias("John")
        entity.add_alias("Mr. Smith")
        entity.add_alias("John")  # Duplicate

        assert len(entity.aliases) == 2
        assert "John" in entity.aliases
        assert "Mr. Smith" in entity.aliases

    def test_add_alias_skips_canonical(self):
        """Should not add canonical name as alias."""
        entity = Entity(
            canonical_name="John Smith",
            entity_type="person",
            user_id="user_123",
        )
        entity.add_alias("John Smith")
        assert len(entity.aliases) == 0

    def test_add_memory(self):
        """Should add memory IDs without duplicates."""
        entity = Entity(
            canonical_name="Acme Corp",
            entity_type="organization",
            user_id="user_123",
        )
        entity.add_memory("mem_1")
        entity.add_memory("mem_2")
        entity.add_memory("mem_1")  # Duplicate

        assert len(entity.memory_ids) == 2
        assert "mem_1" in entity.memory_ids
        assert "mem_2" in entity.memory_ids

    def test_matches_name(self):
        """Should match canonical name and aliases."""
        entity = Entity(
            canonical_name="John Smith",
            entity_type="person",
            user_id="user_123",
            aliases=["John", "Mr. Smith"],
        )

        assert entity.matches_name("John Smith")
        assert entity.matches_name("john smith")  # Case insensitive
        assert entity.matches_name("John")
        assert entity.matches_name("Mr. Smith")
        assert not entity.matches_name("Jane Doe")

    def test_merge_from(self):
        """Should merge another entity's data."""
        entity1 = Entity(
            canonical_name="John Smith",
            entity_type="person",
            user_id="user_123",
            aliases=["John"],
            memory_ids=["mem_1"],
            attributes={"role": "developer"},
        )
        entity2 = Entity(
            canonical_name="J. Smith",
            entity_type="person",
            user_id="user_123",
            aliases=["JS"],
            memory_ids=["mem_2", "mem_3"],
            attributes={"company": "Acme"},
        )

        entity1.merge_from(entity2)

        assert "J. Smith" in entity1.aliases
        assert "JS" in entity1.aliases
        assert "John" in entity1.aliases
        assert "mem_2" in entity1.memory_ids
        assert "mem_3" in entity1.memory_ids
        assert entity1.attributes["company"] == "Acme"
        assert entity1.attributes["role"] == "developer"
        assert entity1.merge_count == 2


class TestEntityMention:
    """Tests for EntityMention model."""

    def test_create_mention(self):
        """Should create a valid mention."""
        mention = EntityMention(
            text="John",
            entity_type="person",
            memory_id="mem_123",
            context="John mentioned he likes Python",
        )
        assert mention.text == "John"
        assert mention.entity_type == "person"
        assert mention.confidence == 0.8  # Default

    def test_mention_confidence_bounds(self):
        """Confidence should be bounded 0.0-1.0."""
        with pytest.raises(ValueError):
            EntityMention(
                text="Test",
                entity_type="person",
                memory_id="mem_1",
                confidence=1.5,
            )


class TestEntityCluster:
    """Tests for EntityCluster model."""

    def test_create_cluster(self):
        """Should create a valid cluster."""
        mentions = [
            EntityMention(
                text="John",
                entity_type="person",
                memory_id="mem_1",
            ),
            EntityMention(
                text="John Smith",
                entity_type="person",
                memory_id="mem_2",
            ),
        ]
        cluster = EntityCluster(
            mentions=mentions,
            suggested_canonical="John Smith",
            suggested_type="person",
            confidence=0.85,
            reasoning="Same person based on context",
        )
        assert len(cluster.mentions) == 2
        assert cluster.suggested_canonical == "John Smith"


class TestExtractEntities:
    """Tests for extract_entities function."""

    @pytest.mark.asyncio
    async def test_extract_entities_calls_llm(self):
        """Should call LLM with formatted prompt."""
        mock_result = MagicMock()
        mock_result.output = MagicMock()
        mock_result.output.mentions = [
            EntityMention(
                text="John Smith",
                entity_type="person",
                memory_id="mem_123",
                context="works as a developer",
            )
        ]

        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value=mock_result)

        with patch("pydantic_ai.Agent", return_value=mock_agent):
            result = await extract_entities(
                content="John Smith works as a developer at Acme Corp",
                memory_id="mem_123",
            )

        assert len(result) == 1
        assert result[0].text == "John Smith"
        assert result[0].memory_id == "mem_123"


class TestResolveEntities:
    """Tests for resolve_entities function."""

    @pytest.mark.asyncio
    async def test_resolve_empty_memories(self):
        """Should return empty result for no memories."""
        result = await resolve_entities(
            memories=[],
            user_id="user_123",
        )
        assert result.entities == []
        assert "No memories" in result.reasoning

    @pytest.mark.asyncio
    async def test_resolve_entities_integration(self):
        """Should extract and cluster entities."""
        # Mock extraction
        mock_extract_result = MagicMock()
        mock_extract_result.output = MagicMock()
        mock_extract_result.output.mentions = [
            EntityMention(
                text="John",
                entity_type="person",
                memory_id="mem_1",
                context="John likes Python",
            )
        ]

        # Mock clustering
        mock_cluster_result = MagicMock()
        mock_cluster_result.output = MagicMock()
        mock_cluster_result.output.clusters = [
            EntityCluster(
                mentions=[
                    EntityMention(
                        text="John",
                        entity_type="person",
                        memory_id="mem_1",
                    )
                ],
                suggested_canonical="John",
                suggested_type="person",
                confidence=0.8,
            )
        ]
        mock_cluster_result.output.reasoning = "Single person mention"

        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(side_effect=[mock_extract_result, mock_cluster_result])

        with patch("pydantic_ai.Agent", return_value=mock_agent):
            result = await resolve_entities(
                memories=[{"id": "mem_1", "content": "John likes Python"}],
                user_id="user_123",
            )

        assert result.new_entities == 1
        assert len(result.entities) == 1
        assert result.entities[0].canonical_name == "John"
