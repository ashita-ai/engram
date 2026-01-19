"""Unit tests for entity extraction and resolution."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from engram.entities.models import (
    Entity,
    EntityCluster,
    EntityMention,
)
from engram.entities.resolution import (
    ClusteringResult,
    ExtractedEntities,
    _infer_relationship,
    cluster_mentions,
    extract_entities,
    find_entity_links,
    resolve_entities,
)


class TestExtractedEntitiesModel:
    """Tests for ExtractedEntities Pydantic model."""

    def test_default_values(self) -> None:
        """Model should have empty defaults."""
        result = ExtractedEntities()
        assert result.mentions == []
        assert result.reasoning == ""

    def test_with_mentions(self) -> None:
        """Model should accept mentions."""
        mention = EntityMention(
            text="John Smith",
            entity_type="person",
            memory_id="mem_123",
        )
        result = ExtractedEntities(mentions=[mention], reasoning="Found name")
        assert len(result.mentions) == 1
        assert result.reasoning == "Found name"


class TestClusteringResultModel:
    """Tests for ClusteringResult Pydantic model."""

    def test_default_values(self) -> None:
        """Model should have empty defaults."""
        result = ClusteringResult()
        assert result.clusters == []
        assert result.reasoning == ""

    def test_with_clusters(self) -> None:
        """Model should accept clusters."""
        mention = EntityMention(
            text="John",
            entity_type="person",
            memory_id="mem_123",
        )
        cluster = EntityCluster(
            mentions=[mention],
            suggested_canonical="John Smith",
            suggested_type="person",
            confidence=0.9,
        )
        result = ClusteringResult(clusters=[cluster], reasoning="Clustered")
        assert len(result.clusters) == 1


class TestExtractEntities:
    """Tests for extract_entities function."""

    @pytest.mark.asyncio
    async def test_extracts_entities_from_content(self) -> None:
        """Should extract entities using LLM."""
        mock_extracted = ExtractedEntities(
            mentions=[
                EntityMention(
                    text="John Smith",
                    entity_type="person",
                    memory_id="",  # Will be set by function
                    context="John Smith is a developer",
                    confidence=0.95,
                ),
                EntityMention(
                    text="Acme Corp",
                    entity_type="organization",
                    memory_id="",
                    context="works at Acme Corp",
                    confidence=0.9,
                ),
            ],
            reasoning="Found person and organization",
        )

        with (
            patch("pydantic_ai.Agent") as mock_agent_class,
            patch(
                "engram.workflows.llm_utils.run_agent_with_retry",
                new_callable=AsyncMock,
            ) as mock_run,
        ):
            mock_agent_class.return_value = MagicMock()
            mock_run.return_value = mock_extracted

            mentions = await extract_entities(
                content="John Smith is a developer at Acme Corp",
                memory_id="mem_test123",
            )

        assert len(mentions) == 2
        # Memory IDs should be set to the provided memory_id
        assert all(m.memory_id == "mem_test123" for m in mentions)
        assert mentions[0].text == "John Smith"
        assert mentions[1].text == "Acme Corp"

    @pytest.mark.asyncio
    async def test_returns_empty_on_error(self) -> None:
        """Should return empty list on LLM error."""
        with (
            patch("pydantic_ai.Agent") as mock_agent_class,
            patch(
                "engram.workflows.llm_utils.run_agent_with_retry",
                new_callable=AsyncMock,
            ) as mock_run,
        ):
            mock_agent_class.return_value = MagicMock()
            mock_run.side_effect = Exception("LLM error")

            mentions = await extract_entities(
                content="Some text with entities",
                memory_id="mem_123",
            )

        assert mentions == []

    @pytest.mark.asyncio
    async def test_empty_content_still_calls_llm(self) -> None:
        """Should still call LLM even with empty content."""
        mock_extracted = ExtractedEntities(mentions=[], reasoning="No content")

        with (
            patch("pydantic_ai.Agent") as mock_agent_class,
            patch(
                "engram.workflows.llm_utils.run_agent_with_retry",
                new_callable=AsyncMock,
            ) as mock_run,
        ):
            mock_agent_class.return_value = MagicMock()
            mock_run.return_value = mock_extracted

            mentions = await extract_entities(content="", memory_id="mem_123")

        assert mentions == []
        mock_run.assert_called_once()


class TestClusterMentions:
    """Tests for cluster_mentions function."""

    @pytest.mark.asyncio
    async def test_returns_empty_for_no_mentions(self) -> None:
        """Should return empty result for no mentions."""
        result = await cluster_mentions([])

        assert result.clusters == []
        assert "No mentions" in result.reasoning

    @pytest.mark.asyncio
    async def test_clusters_mentions(self) -> None:
        """Should cluster related mentions."""
        mentions = [
            EntityMention(
                text="John",
                entity_type="person",
                memory_id="mem_1",
                context="John said hello",
            ),
            EntityMention(
                text="John Smith",
                entity_type="person",
                memory_id="mem_2",
                context="John Smith works here",
            ),
        ]

        mock_clustering = ClusteringResult(
            clusters=[
                EntityCluster(
                    mentions=mentions,
                    suggested_canonical="John Smith",
                    suggested_type="person",
                    confidence=0.9,
                    reasoning="Same person based on name",
                )
            ],
            reasoning="Clustered by name",
        )

        with (
            patch("pydantic_ai.Agent") as mock_agent_class,
            patch(
                "engram.workflows.llm_utils.run_agent_with_retry",
                new_callable=AsyncMock,
            ) as mock_run,
        ):
            mock_agent_class.return_value = MagicMock()
            mock_run.return_value = mock_clustering

            result = await cluster_mentions(mentions)

        assert len(result.clusters) == 1
        assert result.clusters[0].suggested_canonical == "John Smith"

    @pytest.mark.asyncio
    async def test_includes_existing_entities(self) -> None:
        """Should consider existing entities for merging."""
        mentions = [
            EntityMention(
                text="JS",
                entity_type="person",
                memory_id="mem_1",
                context="JS mentioned",
            ),
        ]

        existing = [
            Entity(
                canonical_name="John Smith",
                entity_type="person",
                aliases=["JS", "John"],
                user_id="user_1",
            )
        ]

        mock_clustering = ClusteringResult(
            clusters=[],
            reasoning="Considered existing entities",
        )

        with (
            patch("pydantic_ai.Agent") as mock_agent_class,
            patch(
                "engram.workflows.llm_utils.run_agent_with_retry",
                new_callable=AsyncMock,
            ) as mock_run,
        ):
            mock_agent_class.return_value = MagicMock()
            mock_run.return_value = mock_clustering

            await cluster_mentions(mentions, existing_entities=existing)

        # Check that the prompt included existing entities
        call_args = mock_run.call_args
        prompt = call_args[0][1]
        assert "John Smith" in prompt
        assert "JS" in prompt

    @pytest.mark.asyncio
    async def test_handles_llm_error(self) -> None:
        """Should return error result on LLM failure."""
        mentions = [
            EntityMention(
                text="Test",
                entity_type="person",
                memory_id="mem_1",
                context="Test context",
            )
        ]

        with (
            patch("pydantic_ai.Agent") as mock_agent_class,
            patch(
                "engram.workflows.llm_utils.run_agent_with_retry",
                new_callable=AsyncMock,
            ) as mock_run,
        ):
            mock_agent_class.return_value = MagicMock()
            mock_run.side_effect = Exception("Clustering failed")

            result = await cluster_mentions(mentions)

        assert result.clusters == []
        assert "failed" in result.reasoning.lower()


class TestResolveEntities:
    """Tests for resolve_entities function."""

    @pytest.mark.asyncio
    async def test_returns_empty_for_no_memories(self) -> None:
        """Should return empty result for no memories."""
        result = await resolve_entities([])

        assert result.entities == []
        assert "No memories" in result.reasoning

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_entities_found(self) -> None:
        """Should return empty result when extraction finds nothing."""
        with patch(
            "engram.entities.resolution.extract_entities",
            new_callable=AsyncMock,
        ) as mock_extract:
            mock_extract.return_value = []

            result = await resolve_entities([{"id": "mem_1", "content": "No entities here"}])

        assert result.entities == []
        assert "No entities found" in result.reasoning

    @pytest.mark.asyncio
    async def test_creates_new_entities(self) -> None:
        """Should create new entities from high-confidence clusters."""
        memories = [
            {"id": "mem_1", "content": "John Smith works at Acme"},
        ]

        mention = EntityMention(
            text="John Smith",
            entity_type="person",
            memory_id="mem_1",
            context="John Smith works at Acme",
            confidence=0.9,
        )

        cluster = EntityCluster(
            mentions=[mention],
            suggested_canonical="John Smith",
            suggested_type="person",
            confidence=0.9,
        )

        with (
            patch(
                "engram.entities.resolution.extract_entities",
                new_callable=AsyncMock,
            ) as mock_extract,
            patch(
                "engram.entities.resolution.cluster_mentions",
                new_callable=AsyncMock,
            ) as mock_cluster,
        ):
            mock_extract.return_value = [mention]
            mock_cluster.return_value = ClusteringResult(
                clusters=[cluster],
                reasoning="Found one person",
            )

            result = await resolve_entities(memories, user_id="user_1")

        assert result.new_entities == 1
        assert len(result.entities) == 1
        assert result.entities[0].canonical_name == "John Smith"
        assert result.entities[0].user_id == "user_1"

    @pytest.mark.asyncio
    async def test_merges_with_existing_entities(self) -> None:
        """Should merge mentions into existing entities."""
        memories = [
            {"id": "mem_2", "content": "John mentioned the project"},
        ]

        existing = Entity(
            id="ent_existing",
            canonical_name="John Smith",
            entity_type="person",
            aliases=["John"],
            memory_ids=["mem_1"],
            user_id="user_1",
        )

        mention = EntityMention(
            text="John",
            entity_type="person",
            memory_id="mem_2",
            context="John mentioned the project",
        )

        cluster = EntityCluster(
            mentions=[mention],
            suggested_canonical="John Smith",
            suggested_type="person",
            confidence=0.85,
        )

        with (
            patch(
                "engram.entities.resolution.extract_entities",
                new_callable=AsyncMock,
            ) as mock_extract,
            patch(
                "engram.entities.resolution.cluster_mentions",
                new_callable=AsyncMock,
            ) as mock_cluster,
        ):
            mock_extract.return_value = [mention]
            mock_cluster.return_value = ClusteringResult(
                clusters=[cluster],
                reasoning="Matches existing",
            )

            result = await resolve_entities(
                memories,
                existing_entities=[existing],
                user_id="user_1",
            )

        assert result.merged_entities == 1
        assert result.new_entities == 0

    @pytest.mark.asyncio
    async def test_low_confidence_clusters_go_to_unresolved(self) -> None:
        """Should mark low-confidence clusters as unresolved."""
        memories = [
            {"id": "mem_1", "content": "Someone did something"},
        ]

        mention = EntityMention(
            text="Someone",
            entity_type="person",
            memory_id="mem_1",
            context="Someone did something",
        )

        cluster = EntityCluster(
            mentions=[mention],
            suggested_canonical="Someone",
            suggested_type="person",
            confidence=0.4,  # Below 0.6 threshold
        )

        with (
            patch(
                "engram.entities.resolution.extract_entities",
                new_callable=AsyncMock,
            ) as mock_extract,
            patch(
                "engram.entities.resolution.cluster_mentions",
                new_callable=AsyncMock,
            ) as mock_cluster,
        ):
            mock_extract.return_value = [mention]
            mock_cluster.return_value = ClusteringResult(
                clusters=[cluster],
                reasoning="Low confidence",
            )

            result = await resolve_entities(memories, user_id="user_1")

        assert result.new_entities == 0
        assert len(result.unresolved_mentions) == 1

    @pytest.mark.asyncio
    async def test_matches_existing_by_alias(self) -> None:
        """Should match existing entities by alias."""
        memories = [
            {"id": "mem_2", "content": "JS said hi"},
        ]

        existing = Entity(
            id="ent_existing",
            canonical_name="John Smith",
            entity_type="person",
            aliases=["JS", "Johnny"],
            memory_ids=["mem_1"],
            user_id="user_1",
        )

        mention = EntityMention(
            text="JS",
            entity_type="person",
            memory_id="mem_2",
            context="JS said hi",
        )

        cluster = EntityCluster(
            mentions=[mention],
            suggested_canonical="JS",  # Different from existing canonical
            suggested_type="person",
            confidence=0.9,
        )

        with (
            patch(
                "engram.entities.resolution.extract_entities",
                new_callable=AsyncMock,
            ) as mock_extract,
            patch(
                "engram.entities.resolution.cluster_mentions",
                new_callable=AsyncMock,
            ) as mock_cluster,
        ):
            mock_extract.return_value = [mention]
            mock_cluster.return_value = ClusteringResult(
                clusters=[cluster],
                reasoning="Matched by alias",
            )

            result = await resolve_entities(
                memories,
                existing_entities=[existing],
                user_id="user_1",
            )

        assert result.merged_entities == 1


class TestFindEntityLinks:
    """Tests for find_entity_links function."""

    @pytest.mark.asyncio
    async def test_finds_co_occurring_entities(self) -> None:
        """Should find entities that share memories."""
        entity_a = Entity(
            id="ent_a",
            canonical_name="John Smith",
            entity_type="person",
            memory_ids=["mem_1", "mem_2", "mem_3"],
            user_id="user_1",
        )

        entity_b = Entity(
            id="ent_b",
            canonical_name="Acme Corp",
            entity_type="organization",
            memory_ids=["mem_1", "mem_2"],  # Overlaps with entity_a
            user_id="user_1",
        )

        entity_c = Entity(
            id="ent_c",
            canonical_name="Other Project",
            entity_type="project",
            memory_ids=["mem_99"],  # No overlap
            user_id="user_1",
        )

        all_entities = [entity_a, entity_b, entity_c]

        links = await find_entity_links(entity_a, all_entities, min_memory_overlap=1)

        assert len(links) == 1
        assert links[0][0] == "ent_b"
        assert links[0][1] == "works_at"  # person -> organization

    @pytest.mark.asyncio
    async def test_respects_min_overlap(self) -> None:
        """Should respect minimum memory overlap threshold."""
        entity_a = Entity(
            id="ent_a",
            canonical_name="John",
            entity_type="person",
            memory_ids=["mem_1", "mem_2"],
            user_id="user_1",
        )

        entity_b = Entity(
            id="ent_b",
            canonical_name="Acme",
            entity_type="organization",
            memory_ids=["mem_1"],  # Only 1 overlap
            user_id="user_1",
        )

        links = await find_entity_links(entity_a, [entity_a, entity_b], min_memory_overlap=2)

        assert len(links) == 0

    @pytest.mark.asyncio
    async def test_excludes_self(self) -> None:
        """Should not link entity to itself."""
        entity = Entity(
            id="ent_self",
            canonical_name="Self",
            entity_type="person",
            memory_ids=["mem_1"],
            user_id="user_1",
        )

        links = await find_entity_links(entity, [entity], min_memory_overlap=1)

        assert len(links) == 0


class TestInferRelationship:
    """Tests for _infer_relationship function."""

    def test_person_organization_relationship(self) -> None:
        """Person and organization should infer works_at."""
        assert _infer_relationship("person", "organization") == "works_at"

    def test_person_project_relationship(self) -> None:
        """Person and project should infer works_on."""
        assert _infer_relationship("person", "project") == "works_on"

    def test_person_technology_relationship(self) -> None:
        """Person and technology should infer uses."""
        assert _infer_relationship("person", "technology") == "uses"

    def test_organization_project_relationship(self) -> None:
        """Organization and project should infer owns."""
        assert _infer_relationship("organization", "project") == "owns"

    def test_organization_technology_relationship(self) -> None:
        """Organization and technology should infer uses."""
        assert _infer_relationship("organization", "technology") == "uses"

    def test_project_technology_relationship(self) -> None:
        """Project and technology should infer uses."""
        assert _infer_relationship("project", "technology") == "uses"

    def test_reverse_relationship(self) -> None:
        """Reverse order should prefix with has_."""
        assert _infer_relationship("organization", "person") == "has_works_at"
        assert _infer_relationship("project", "person") == "has_works_on"
        assert _infer_relationship("technology", "person") == "has_uses"

    def test_unknown_relationship(self) -> None:
        """Unknown pairs should return related_to."""
        assert _infer_relationship("location", "concept") == "related_to"
        assert _infer_relationship("product", "concept") == "related_to"

    def test_same_type_relationship(self) -> None:
        """Same type should return related_to."""
        assert _infer_relationship("person", "person") == "related_to"
        assert _infer_relationship("organization", "organization") == "related_to"


class TestEntityModel:
    """Tests for Entity model methods."""

    def test_add_alias(self) -> None:
        """add_alias should add new aliases."""
        entity = Entity(
            canonical_name="John Smith",
            entity_type="person",
            user_id="user_1",
        )
        entity.add_alias("JS")
        entity.add_alias("Johnny")

        assert "JS" in entity.aliases
        assert "Johnny" in entity.aliases

    def test_add_alias_ignores_canonical(self) -> None:
        """add_alias should not add the canonical name as alias."""
        entity = Entity(
            canonical_name="John Smith",
            entity_type="person",
            user_id="user_1",
        )
        entity.add_alias("John Smith")

        assert entity.aliases == []

    def test_add_alias_ignores_duplicates(self) -> None:
        """add_alias should not add duplicate aliases."""
        entity = Entity(
            canonical_name="John Smith",
            entity_type="person",
            user_id="user_1",
        )
        entity.add_alias("JS")
        entity.add_alias("JS")

        assert entity.aliases.count("JS") == 1

    def test_add_memory(self) -> None:
        """add_memory should add memory IDs."""
        entity = Entity(
            canonical_name="John",
            entity_type="person",
            user_id="user_1",
        )
        entity.add_memory("mem_1")
        entity.add_memory("mem_2")

        assert "mem_1" in entity.memory_ids
        assert "mem_2" in entity.memory_ids

    def test_add_memory_ignores_duplicates(self) -> None:
        """add_memory should not add duplicate IDs."""
        entity = Entity(
            canonical_name="John",
            entity_type="person",
            user_id="user_1",
        )
        entity.add_memory("mem_1")
        entity.add_memory("mem_1")

        assert entity.memory_ids.count("mem_1") == 1

    def test_matches_name_canonical(self) -> None:
        """matches_name should match canonical name."""
        entity = Entity(
            canonical_name="John Smith",
            entity_type="person",
            user_id="user_1",
        )
        assert entity.matches_name("John Smith") is True
        assert entity.matches_name("john smith") is True  # Case insensitive

    def test_matches_name_alias(self) -> None:
        """matches_name should match aliases."""
        entity = Entity(
            canonical_name="John Smith",
            entity_type="person",
            aliases=["JS", "Johnny"],
            user_id="user_1",
        )
        assert entity.matches_name("JS") is True
        assert entity.matches_name("Johnny") is True

    def test_matches_name_no_match(self) -> None:
        """matches_name should return False for non-matches."""
        entity = Entity(
            canonical_name="John Smith",
            entity_type="person",
            aliases=["JS"],
            user_id="user_1",
        )
        assert entity.matches_name("Jane Doe") is False

    def test_merge_from(self) -> None:
        """merge_from should combine two entities."""
        entity_a = Entity(
            canonical_name="John Smith",
            entity_type="person",
            aliases=["JS"],
            memory_ids=["mem_1"],
            attributes={"role": "developer"},
            user_id="user_1",
            merge_count=2,
        )

        entity_b = Entity(
            canonical_name="John D Smith",
            entity_type="person",
            aliases=["JDS"],
            memory_ids=["mem_2"],
            attributes={"title": "Senior Dev"},
            user_id="user_1",
            merge_count=3,
        )

        entity_a.merge_from(entity_b)

        # Should have both aliases plus entity_b's canonical
        assert "JDS" in entity_a.aliases
        assert "John D Smith" in entity_a.aliases
        # JS should still be there
        assert "JS" in entity_a.aliases

        # Should have both memory IDs
        assert "mem_1" in entity_a.memory_ids
        assert "mem_2" in entity_a.memory_ids

        # Should have both attributes
        assert entity_a.attributes["role"] == "developer"
        assert entity_a.attributes["title"] == "Senior Dev"

        # Merge count should be combined
        assert entity_a.merge_count == 5
