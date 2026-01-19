"""LLM-driven entity extraction and resolution.

Uses Pydantic AI to extract entity mentions from memories and
resolve them to canonical entities with aliases.

References:
- Named Entity Recognition (NER)
- Entity Linking/Resolution research
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .models import (
    Entity,
    EntityCluster,
    EntityMention,
    EntityResolutionResult,
    EntityType,
)

logger = logging.getLogger(__name__)


class ExtractedEntities(BaseModel):
    """LLM output for entity extraction."""

    model_config = ConfigDict(extra="forbid")

    mentions: list[EntityMention] = Field(default_factory=list)
    reasoning: str = Field(default="")


class ClusteringResult(BaseModel):
    """LLM output for entity clustering."""

    model_config = ConfigDict(extra="forbid")

    clusters: list[EntityCluster] = Field(default_factory=list)
    reasoning: str = Field(default="")


# System prompt for entity extraction
EXTRACTION_PROMPT = """You are extracting named entities from text content.

Extract mentions of:
- PERSON: People's names (full names, first names, titles like "Mr. Smith")
- ORGANIZATION: Companies, institutions, teams
- PROJECT: Software projects, products, initiatives
- TECHNOLOGY: Programming languages, frameworks, tools
- LOCATION: Cities, countries, addresses
- PRODUCT: Commercial products, services
- CONCEPT: Abstract ideas, methodologies

For each entity mention:
1. Extract the exact text as it appears
2. Classify the entity type
3. Include surrounding context (1-2 sentences) for disambiguation
4. Rate confidence in the classification (0.0-1.0)

IMPORTANT:
- Extract EVERY entity mention, even if the same entity appears multiple times
- Include partial names (e.g., "John" even if "John Smith" appears elsewhere)
- Don't merge mentions - just extract what you see
"""

# System prompt for entity clustering
CLUSTERING_PROMPT = """You are clustering entity mentions that refer to the same real-world entity.

Given a list of entity mentions from different memories, determine which ones
refer to the same entity.

CLUSTERING RULES:
1. Names that are subsets (e.g., "John" and "John Smith") likely refer to same person
2. Consider context clues (same role, same company, etc.)
3. Different entities with similar names should NOT be clustered
4. Use the memory context to disambiguate

For each cluster:
1. Suggest a canonical name (most complete/formal version)
2. Suggest the entity type
3. Rate confidence that all mentions are the same entity
4. Explain your reasoning

Be CONSERVATIVE - only cluster when confident. Separate clusters are fine.
"""


async def extract_entities(
    content: str,
    memory_id: str,
) -> list[EntityMention]:
    """Extract entity mentions from text content.

    Uses LLM to identify named entities and their types.

    Args:
        content: Text content to extract from.
        memory_id: ID of the source memory.

    Returns:
        List of extracted entity mentions.
    """
    from pydantic_ai import Agent

    from engram.config import settings

    agent: Agent[None, ExtractedEntities] = Agent(
        settings.llm_model,
        output_type=ExtractedEntities,
        instructions=EXTRACTION_PROMPT,
    )

    from engram.workflows.llm_utils import run_agent_with_retry

    user_prompt = f"""Extract all named entities from this text:

{content}

Memory ID: {memory_id}

List every entity mention you find."""

    try:
        extracted = await run_agent_with_retry(agent, user_prompt)

        # Ensure memory_id is set on all mentions
        for mention in extracted.mentions:
            mention.memory_id = memory_id

        logger.info(
            "Extracted %d entity mentions from memory %s",
            len(extracted.mentions),
            memory_id,
        )
        return extracted.mentions

    except Exception as e:
        logger.exception("Entity extraction failed: %s", e)
        return []


async def cluster_mentions(
    mentions: list[EntityMention],
    existing_entities: list[Entity] | None = None,
) -> ClusteringResult:
    """Cluster entity mentions into groups representing the same entity.

    Uses LLM to analyze mentions and determine which ones refer to
    the same real-world entity.

    Args:
        mentions: List of entity mentions to cluster.
        existing_entities: Optional existing entities to consider for merging.

    Returns:
        ClusteringResult with grouped mentions.
    """
    from pydantic_ai import Agent

    from engram.config import settings

    if not mentions:
        return ClusteringResult(reasoning="No mentions to cluster")

    # Format mentions for LLM
    mentions_text = "\n".join(
        f'{i + 1}. [{m.entity_type}] "{m.text}" (memory: {m.memory_id})\n'
        f"   Context: {m.context[:100]}..."
        if len(m.context) > 100
        else f"   Context: {m.context}"
        for i, m in enumerate(mentions)
    )

    # Format existing entities if provided
    existing_text = ""
    if existing_entities:
        existing_text = "\n\nEXISTING ENTITIES (consider merging with these):\n"
        existing_text += "\n".join(
            f"- {e.canonical_name} (aliases: {', '.join(e.aliases[:3])})"
            for e in existing_entities[:10]
        )

    agent: Agent[None, ClusteringResult] = Agent(
        settings.llm_model,
        output_type=ClusteringResult,
        instructions=CLUSTERING_PROMPT,
    )

    from engram.workflows.llm_utils import run_agent_with_retry

    user_prompt = f"""Cluster these entity mentions by same real-world entity:

{mentions_text}
{existing_text}

Group mentions that refer to the same entity. Be conservative."""

    try:
        return await run_agent_with_retry(agent, user_prompt)

    except Exception as e:
        logger.exception("Entity clustering failed: %s", e)
        return ClusteringResult(reasoning=f"Clustering failed: {e}")


async def resolve_entities(
    memories: list[dict[str, Any]],
    existing_entities: list[Entity] | None = None,
    user_id: str = "",
    org_id: str | None = None,
) -> EntityResolutionResult:
    """Resolve entities from a batch of memories.

    Full pipeline: extract mentions -> cluster -> create/merge entities.

    Args:
        memories: List of memory dicts with 'id' and 'content' keys.
        existing_entities: Optional existing entities to consider.
        user_id: User ID for new entities.
        org_id: Optional organization ID.

    Returns:
        EntityResolutionResult with resolved entities.
    """
    if not memories:
        return EntityResolutionResult(reasoning="No memories provided")

    # Step 1: Extract mentions from all memories
    all_mentions: list[EntityMention] = []
    for memory in memories:
        mem_id = memory.get("id", "")
        content = memory.get("content", "")
        if content:
            mentions = await extract_entities(content, mem_id)
            all_mentions.extend(mentions)

    if not all_mentions:
        return EntityResolutionResult(reasoning="No entities found in memories")

    logger.info("Extracted %d total mentions from %d memories", len(all_mentions), len(memories))

    # Step 2: Cluster mentions
    clustering = await cluster_mentions(all_mentions, existing_entities)

    if not clustering.clusters:
        return EntityResolutionResult(
            unresolved_mentions=all_mentions,
            reasoning="Could not cluster any mentions",
        )

    # Step 3: Create or merge entities from clusters
    result = EntityResolutionResult(reasoning=clustering.reasoning)
    existing_map = {e.canonical_name.lower(): e for e in (existing_entities or [])}

    for cluster in clustering.clusters:
        if cluster.confidence < 0.6:
            # Low confidence - mark as unresolved
            result.unresolved_mentions.extend(cluster.mentions)
            continue

        # Check if matches existing entity
        existing_entity = existing_map.get(cluster.suggested_canonical.lower())
        if not existing_entity:
            # Check aliases
            for existing in existing_entities or []:
                if existing.matches_name(cluster.suggested_canonical):
                    existing_entity = existing
                    break

        if existing_entity:
            # Merge into existing
            for mention in cluster.mentions:
                existing_entity.add_alias(mention.text)
                existing_entity.add_memory(mention.memory_id)
            existing_entity.merge_count += len(cluster.mentions)
            result.merged_entities += 1
            result.entities.append(existing_entity)
        else:
            # Create new entity
            entity = Entity(
                canonical_name=cluster.suggested_canonical,
                entity_type=cluster.suggested_type,
                user_id=user_id,
                org_id=org_id,
            )
            for mention in cluster.mentions:
                entity.add_alias(mention.text)
                entity.add_memory(mention.memory_id)
            entity.merge_count = len(cluster.mentions)
            result.new_entities += 1
            result.entities.append(entity)

    logger.info(
        "Entity resolution: %d new, %d merged, %d unresolved",
        result.new_entities,
        result.merged_entities,
        len(result.unresolved_mentions),
    )

    return result


async def find_entity_links(
    entity: Entity,
    all_entities: list[Entity],
    min_memory_overlap: int = 1,
) -> list[tuple[str, str]]:
    """Find other entities that co-occur with this entity in memories.

    Entities appearing in the same memories are likely related.

    Args:
        entity: Entity to find links for.
        all_entities: All entities to search.
        min_memory_overlap: Minimum shared memories to create link.

    Returns:
        List of (entity_id, relationship) tuples.
    """
    links: list[tuple[str, str]] = []
    entity_mems = set(entity.memory_ids)

    for other in all_entities:
        if other.id == entity.id:
            continue

        other_mems = set(other.memory_ids)
        overlap = entity_mems & other_mems

        if len(overlap) >= min_memory_overlap:
            # Determine relationship based on entity types
            relationship = _infer_relationship(entity.entity_type, other.entity_type)
            links.append((other.id, relationship))

    return links


def _infer_relationship(type_a: EntityType, type_b: EntityType) -> str:
    """Infer relationship type between two entity types."""
    relationships = {
        ("person", "organization"): "works_at",
        ("person", "project"): "works_on",
        ("person", "technology"): "uses",
        ("organization", "project"): "owns",
        ("organization", "technology"): "uses",
        ("project", "technology"): "uses",
    }

    key = (type_a, type_b)
    if key in relationships:
        return relationships[key]

    reverse_key = (type_b, type_a)
    if reverse_key in relationships:
        return f"has_{relationships[reverse_key]}"

    return "related_to"
