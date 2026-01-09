"""Consolidation workflow for extracting semantic knowledge from episodes.

This workflow runs in the background to:
1. Fetch unconsolidated episodes
2. Run LLM extraction via Pydantic AI
3. Store semantic memories
4. Build links between memories

The workflow is durable - it survives crashes and can be retried on failure.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from engram.embeddings import Embedder
    from engram.models import SemanticMemory
    from engram.storage import EngramStorage

logger = logging.getLogger(__name__)


class ExtractedFact(BaseModel):
    """A semantic fact extracted by the LLM.

    A-MEM inspired: includes keywords, tags, and context for richer
    memory organization and linking.
    """

    model_config = ConfigDict(extra="forbid")

    content: str = Field(description="The semantic knowledge extracted")
    confidence: float = Field(
        ge=0.0, le=1.0, default=0.6, description="Confidence in this extraction"
    )
    source_context: str = Field(default="", description="Original context this was extracted from")
    # A-MEM inspired fields
    keywords: list[str] = Field(
        default_factory=list,
        description="Key terms for this fact (3-5 words)",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Category labels (e.g., preference, technical, personal, behavioral)",
    )
    context: str = Field(
        default="",
        description="Domain or theme (e.g., programming, communication style, work preferences)",
    )


class IdentifiedLink(BaseModel):
    """A relationship between memories identified by the LLM."""

    model_config = ConfigDict(extra="forbid")

    source_content: str = Field(description="Content of source memory")
    target_content: str = Field(description="Content of related memory")
    relationship: str = Field(description="Nature of the relationship")


class MemoryEvolution(BaseModel):
    """Suggested update to an existing memory (A-MEM style evolution).

    When new information arrives, it may provide context that enriches
    existing memories. This captures suggested metadata updates.
    """

    model_config = ConfigDict(extra="forbid")

    target_content: str = Field(
        description="Content of the existing memory to update (for matching)"
    )
    add_tags: list[str] = Field(
        default_factory=list,
        description="Tags to add to the existing memory",
    )
    add_keywords: list[str] = Field(
        default_factory=list,
        description="Keywords to add to the existing memory",
    )
    update_context: str = Field(
        default="",
        description="Additional context to append",
    )
    reason: str = Field(
        default="",
        description="Why this evolution is suggested",
    )


class LLMExtractionResult(BaseModel):
    """Structured output from the consolidation LLM agent.

    A-MEM inspired: includes memory evolution suggestions for updating
    existing memories based on new information.
    """

    model_config = ConfigDict(extra="forbid")

    semantic_facts: list[ExtractedFact] = Field(
        default_factory=list, description="Extracted semantic knowledge"
    )
    links: list[IdentifiedLink] = Field(
        default_factory=list, description="Relationships between facts"
    )
    contradictions: list[str] = Field(
        default_factory=list, description="Contradictions with existing knowledge"
    )
    evolutions: list[MemoryEvolution] = Field(
        default_factory=list,
        description="Suggested updates to existing memories based on new context",
    )


class ConsolidationResult(BaseModel):
    """Result of a consolidation workflow run.

    Attributes:
        episodes_processed: Number of episodes that were processed.
        semantic_memories_created: Number of semantic memories extracted.
        links_created: Number of memory links built.
        evolutions_applied: Number of memory evolution updates applied.
        contradictions_found: List of detected contradictions.
    """

    model_config = ConfigDict(extra="forbid")

    episodes_processed: int = Field(ge=0)
    semantic_memories_created: int = Field(ge=0)
    links_created: int = Field(ge=0)
    evolutions_applied: int = Field(ge=0, default=0)
    contradictions_found: list[str] = Field(default_factory=list)


def format_episodes_for_llm(episodes: list[dict[str, str]]) -> str:
    """Format episodes for LLM processing.

    Args:
        episodes: List of dicts with 'id', 'role', 'content' keys.

    Returns:
        Formatted text for LLM input.
    """
    lines = ["# Conversation Episodes to Analyze\n"]
    for ep in episodes:
        lines.append(f"[{ep['role'].upper()}] ({ep['id']})")
        lines.append(ep["content"])
        lines.append("")
    return "\n".join(lines)


def _find_matching_memory(
    content: str,
    memories: dict[str, SemanticMemory],
) -> SemanticMemory | None:
    """Find a memory matching the given content (fallback string matching).

    Uses exact match first, then substring matching if not found.
    This is a fallback for when semantic similarity isn't available.

    Args:
        content: Content string to match.
        memories: Dict mapping content to SemanticMemory.

    Returns:
        Matching SemanticMemory or None.
    """
    # Exact match
    if content in memories:
        return memories[content]

    # Normalize and try again
    normalized = content.strip().lower()
    for mem_content, memory in memories.items():
        if mem_content.strip().lower() == normalized:
            return memory

    # Substring match (content is contained in memory or vice versa)
    for mem_content, memory in memories.items():
        mem_normalized = mem_content.strip().lower()
        if normalized in mem_normalized or mem_normalized in normalized:
            return memory

    return None


async def _find_semantically_similar_memories(
    query_embedding: list[float],
    storage: EngramStorage,
    user_id: str,
    org_id: str | None,
    limit: int = 5,
    min_score: float = 0.7,
) -> list[SemanticMemory]:
    """Find semantically similar memories using embedding search.

    A-MEM style: uses vector similarity to find related memories,
    not just string matching.

    Args:
        query_embedding: Embedding vector to search with.
        storage: Storage instance.
        user_id: User ID for isolation.
        org_id: Optional org ID.
        limit: Max results to return.
        min_score: Minimum similarity score threshold.

    Returns:
        List of SemanticMemory objects above the similarity threshold.
    """
    results = await storage.search_semantic(
        query_vector=query_embedding,
        user_id=user_id,
        org_id=org_id,
        limit=limit,
    )

    # Filter by minimum score
    return [r.memory for r in results if r.score >= min_score]


def _format_existing_memories_for_llm(memories: list[SemanticMemory]) -> str:
    """Format existing memories for LLM context.

    Provides the LLM with existing memories so it can suggest
    evolutions and identify meaningful links.

    Args:
        memories: List of existing semantic memories.

    Returns:
        Formatted text for LLM input.
    """
    if not memories:
        return ""

    lines = ["\n# Existing Memories (for linking and evolution)\n"]
    for mem in memories[:20]:  # Limit to avoid token overflow
        tags_str = ", ".join(mem.tags) if mem.tags else "none"
        lines.append(f"- [{mem.id}] {mem.content}")
        lines.append(f"  Tags: {tags_str}, Context: {mem.context or 'general'}")
    return "\n".join(lines)


async def run_consolidation(
    storage: EngramStorage,
    embedder: Embedder,
    user_id: str,
    org_id: str | None = None,
    batch_size: int = 20,
    similarity_threshold: float = 0.7,
) -> ConsolidationResult:
    """Run the consolidation workflow.

    This is the main entry point for consolidation. A-MEM inspired features:
    1. Fetches unconsolidated episodes
    2. Gets existing memories for context (enables evolution)
    3. Runs LLM extraction with keywords/tags/context
    4. Stores semantic memories with rich metadata
    5. Builds links using semantic similarity (not just string matching)
    6. Applies memory evolutions to existing memories
    7. Marks episodes as consolidated

    Args:
        storage: EngramStorage instance.
        embedder: Embedder for generating vectors.
        user_id: User ID for multi-tenancy.
        org_id: Optional organization ID.
        batch_size: Number of episodes to process at once.
        similarity_threshold: Min similarity score for automatic linking.

    Returns:
        ConsolidationResult with processing statistics.
    """
    from engram.models import SemanticMemory

    # 1. Fetch unconsolidated episodes
    episodes = await storage.get_unconsolidated_episodes(
        user_id=user_id,
        org_id=org_id,
        limit=batch_size,
    )

    if not episodes:
        logger.info("No unconsolidated episodes found")
        return ConsolidationResult(
            episodes_processed=0,
            semantic_memories_created=0,
            links_created=0,
            evolutions_applied=0,
        )

    logger.info(f"Processing {len(episodes)} unconsolidated episodes")

    # 2. Get existing memories to provide context for linking and evolution
    existing_memories = await storage.list_semantic_memories(user_id, org_id)
    existing_memories_context = _format_existing_memories_for_llm(existing_memories)
    existing_by_content = {m.content: m for m in existing_memories}

    # 3. Format episodes for LLM
    episode_data = [{"id": ep.id, "role": ep.role, "content": ep.content} for ep in episodes]
    formatted_text = format_episodes_for_llm(episode_data) + existing_memories_context

    # 4. Run LLM extraction using the durable agent
    from engram.workflows import get_consolidation_agent

    # A-MEM inspired prompt with keywords/tags/context extraction and evolution
    amem_prompt = """You are analyzing conversation episodes to extract lasting knowledge.

For each semantic fact you extract, provide:
- content: The knowledge extracted
- confidence: How certain (0.6-0.9)
- keywords: 3-5 key terms for this fact
- tags: Category labels like "preference", "technical", "personal", "behavioral", "factual"
- context: Domain/theme like "programming", "communication", "work habits"

Extract facts that are:
- Personally identifying (names, preferences, habits)
- Temporally stable (unlikely to change soon)
- Explicitly stated or clearly implied

For LINKS between facts:
- Identify meaningful relationships (not just any mention)
- Link new facts to existing memories when relevant

For EVOLUTIONS (updates to existing memories):
- If new information adds context to an existing memory, suggest updates
- Only suggest tag/keyword/context updates, not content changes
- Explain why the evolution is suggested

Be conservative. When uncertain, don't extract.
Focus on quality over quantity."""

    try:
        agent = get_consolidation_agent()
        result = await agent.run(formatted_text)
        extraction: LLMExtractionResult = result.output
    except RuntimeError:
        # Workflows not initialized, run without durability
        from pydantic_ai import Agent

        from engram.config import settings

        logger.warning("Durable workflows not initialized, running non-durable extraction")
        temp_agent: Agent[None, LLMExtractionResult] = Agent(
            settings.consolidation_model,
            output_type=LLMExtractionResult,
            instructions=amem_prompt,
        )
        result = await temp_agent.run(formatted_text)
        extraction = result.output

    # 5. Store semantic memories with A-MEM metadata
    episode_ids = [ep.id for ep in episodes]
    memories_created = 0
    links_created = 0
    evolutions_applied = 0

    # Map content to newly created memory for link building
    content_to_memory: dict[str, SemanticMemory] = {}

    for fact in extraction.semantic_facts:
        # Generate embedding for the fact
        embedding = await embedder.embed(fact.content)

        memory = SemanticMemory(
            content=fact.content,
            source_episode_ids=episode_ids,
            user_id=user_id,
            org_id=org_id,
            embedding=embedding,
            # A-MEM inspired fields
            keywords=fact.keywords,
            tags=fact.tags,
            context=fact.context,
        )
        memory.confidence.value = fact.confidence

        await storage.store_semantic(memory)
        content_to_memory[fact.content] = memory
        memories_created += 1
        logger.debug(f"Created semantic memory: {memory.id} with tags={fact.tags}")

    # 6. Build links using SEMANTIC SIMILARITY (A-MEM style)
    # For each new memory, find semantically similar existing memories
    all_memories = dict(existing_by_content)
    all_memories.update(content_to_memory)

    for new_memory in content_to_memory.values():
        if new_memory.embedding is None:
            continue

        # Find semantically similar memories
        similar_memories = await _find_semantically_similar_memories(
            query_embedding=new_memory.embedding,
            storage=storage,
            user_id=user_id,
            org_id=org_id,
            limit=5,
            min_score=similarity_threshold,
        )

        for similar in similar_memories:
            # Don't link to self
            if similar.id == new_memory.id:
                continue

            # Check if already linked
            if similar.id in new_memory.related_ids:
                continue

            # INHIBITORY PLASTICITY (Tomé et al. Nature Neuroscience 2024)
            # During consolidation, overlapping memories compete.
            # Winner (more sources/higher confidence) becomes more selective.
            # Loser becomes less selective (may be pruned in decay workflow).
            new_sources = len(new_memory.source_episode_ids)
            existing_sources = len(similar.source_episode_ids)

            if existing_sources > new_sources:
                # Existing memory wins - it has more corroboration
                similar.increase_selectivity(delta=0.1)
                # New memory starts with default selectivity (0.0)
                logger.debug(
                    f"Inhibitory plasticity: {similar.id} wins over {new_memory.id} "
                    f"({existing_sources} vs {new_sources} sources)"
                )
            elif new_sources > existing_sources:
                # New memory wins - decrease existing selectivity
                new_memory.increase_selectivity(delta=0.1)
                similar.decrease_selectivity(delta=0.05)
                logger.debug(
                    f"Inhibitory plasticity: {new_memory.id} wins over {similar.id} "
                    f"({new_sources} vs {existing_sources} sources)"
                )
            else:
                # Tie - use confidence as tiebreaker
                if similar.confidence.value >= new_memory.confidence.value:
                    similar.increase_selectivity(delta=0.05)
                else:
                    new_memory.increase_selectivity(delta=0.05)

            # Add bidirectional links
            new_memory.add_link(similar.id)
            similar.add_link(new_memory.id)

            # Persist the updated memories
            await storage.update_semantic_memory(new_memory)
            await storage.update_semantic_memory(similar)

            links_created += 1
            logger.info(f"Semantic link: {new_memory.id} <--> {similar.id}")

    # Also process LLM-identified links (as additional signal)
    for link in extraction.links:
        source_memory = _find_matching_memory(link.source_content, all_memories)
        target_memory = _find_matching_memory(link.target_content, all_memories)

        if source_memory and target_memory and source_memory.id != target_memory.id:
            # Check if already linked
            if target_memory.id not in source_memory.related_ids:
                source_memory.add_link(target_memory.id)
                target_memory.add_link(source_memory.id)
                await storage.update_semantic_memory(source_memory)
                await storage.update_semantic_memory(target_memory)
                links_created += 1
                logger.info(
                    f"LLM link: {source_memory.id} --[{link.relationship}]--> {target_memory.id}"
                )

    # 7. Apply memory evolutions (A-MEM style)
    for evolution in extraction.evolutions:
        target_memory = _find_matching_memory(evolution.target_content, existing_by_content)

        if target_memory is None:
            logger.debug(f"Could not find memory for evolution: {evolution.target_content[:30]}...")
            continue

        # Get a trigger ID (use first new memory if available)
        trigger_id = next(iter(content_to_memory.values())).id if content_to_memory else "unknown"

        # Apply tag updates
        if evolution.add_tags:
            target_memory.evolve(
                trigger_memory_id=trigger_id,
                field="tags",
                new_value=",".join(evolution.add_tags),
                reason=evolution.reason,
            )

        # Apply keyword updates
        if evolution.add_keywords:
            target_memory.evolve(
                trigger_memory_id=trigger_id,
                field="keywords",
                new_value=",".join(evolution.add_keywords),
                reason=evolution.reason,
            )

        # Apply context updates
        if evolution.update_context:
            target_memory.evolve(
                trigger_memory_id=trigger_id,
                field="context",
                new_value=evolution.update_context,
                reason=evolution.reason,
            )

        # Persist the evolved memory
        await storage.update_semantic_memory(target_memory)
        evolutions_applied += 1
        logger.info(f"Evolved memory {target_memory.id}: {evolution.reason}")

    # 8. Mark episodes as consolidated
    await storage.mark_episodes_consolidated(episode_ids, user_id)

    return ConsolidationResult(
        episodes_processed=len(episodes),
        semantic_memories_created=memories_created,
        links_created=links_created,
        evolutions_applied=evolutions_applied,
        contradictions_found=extraction.contradictions,
    )


__all__ = [
    "ConsolidationResult",
    "ExtractedFact",
    "IdentifiedLink",
    "LLMExtractionResult",
    "MemoryEvolution",
    "format_episodes_for_llm",
    "run_consolidation",
]
