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
    """A semantic fact extracted by the LLM."""

    model_config = ConfigDict(extra="forbid")

    content: str = Field(description="The semantic knowledge extracted")
    confidence: float = Field(
        ge=0.0, le=1.0, default=0.6, description="Confidence in this extraction"
    )
    source_context: str = Field(default="", description="Original context this was extracted from")


class IdentifiedLink(BaseModel):
    """A relationship between memories identified by the LLM."""

    model_config = ConfigDict(extra="forbid")

    source_content: str = Field(description="Content of source memory")
    target_content: str = Field(description="Content of related memory")
    relationship: str = Field(description="Nature of the relationship")


class LLMExtractionResult(BaseModel):
    """Structured output from the consolidation LLM agent."""

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


class ConsolidationResult(BaseModel):
    """Result of a consolidation workflow run.

    Attributes:
        episodes_processed: Number of episodes that were processed.
        semantic_memories_created: Number of semantic memories extracted.
        links_created: Number of memory links built.
        contradictions_found: List of detected contradictions.
    """

    model_config = ConfigDict(extra="forbid")

    episodes_processed: int = Field(ge=0)
    semantic_memories_created: int = Field(ge=0)
    links_created: int = Field(ge=0)
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
    """Find a memory matching the given content.

    Uses exact match first, then substring matching if not found.

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


async def run_consolidation(
    storage: EngramStorage,
    embedder: Embedder,
    user_id: str,
    org_id: str | None = None,
    batch_size: int = 20,
) -> ConsolidationResult:
    """Run the consolidation workflow.

    This is the main entry point for consolidation. It:
    1. Fetches unconsolidated episodes
    2. Runs LLM extraction via the durable agent
    3. Stores semantic memories
    4. Builds links between memories
    5. Marks episodes as consolidated

    Args:
        storage: EngramStorage instance.
        embedder: Embedder for generating vectors.
        user_id: User ID for multi-tenancy.
        org_id: Optional organization ID.
        batch_size: Number of episodes to process at once.

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
        )

    logger.info(f"Processing {len(episodes)} unconsolidated episodes")

    # 2. Format episodes for LLM
    episode_data = [{"id": ep.id, "role": ep.role, "content": ep.content} for ep in episodes]
    formatted_text = format_episodes_for_llm(episode_data)

    # 3. Run LLM extraction using the durable agent
    from engram.workflows import get_consolidation_agent

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
            instructions="""You are analyzing conversation episodes to extract lasting knowledge.

Extract facts that are:
- Personally identifying (names, emails, preferences)
- Temporally stable (unlikely to change soon)
- Explicitly stated (not inferred)

Identify relationships between concepts.
Flag any contradictions with previously known facts.

Be conservative. When uncertain, don't extract.""",
        )
        result = await temp_agent.run(formatted_text)
        extraction = result.output

    # 4. Store semantic memories and track content → memory mapping
    episode_ids = [ep.id for ep in episodes]
    memories_created = 0
    links_created = 0

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
        )
        memory.confidence.value = fact.confidence

        await storage.store_semantic(memory)
        content_to_memory[fact.content] = memory
        memories_created += 1
        logger.debug(f"Created semantic memory: {memory.id}")

    # 5. Build links between memories
    # First, get existing semantic memories for the user
    existing_memories = await storage.list_semantic_memories(user_id, org_id)
    all_memories = {m.content: m for m in existing_memories}
    all_memories.update(content_to_memory)

    for link in extraction.links:
        source_memory = _find_matching_memory(link.source_content, all_memories)
        target_memory = _find_matching_memory(link.target_content, all_memories)

        if source_memory and target_memory and source_memory.id != target_memory.id:
            # Add bidirectional links
            source_memory.add_link(target_memory.id)
            target_memory.add_link(source_memory.id)

            # Persist the updated memories
            await storage.update_semantic_memory(source_memory)
            await storage.update_semantic_memory(target_memory)

            links_created += 1
            logger.info(f"Linked: {source_memory.id} --[{link.relationship}]--> {target_memory.id}")
        else:
            logger.debug(
                f"Could not find memories for link: {link.source_content[:30]}... "
                f"--[{link.relationship}]--> {link.target_content[:30]}..."
            )

    # 6. Mark episodes as consolidated
    await storage.mark_episodes_consolidated(episode_ids, user_id)

    return ConsolidationResult(
        episodes_processed=len(episodes),
        semantic_memories_created=memories_created,
        links_created=links_created,
        contradictions_found=extraction.contradictions,
    )


__all__ = [
    "ConsolidationResult",
    "ExtractedFact",
    "IdentifiedLink",
    "LLMExtractionResult",
    "format_episodes_for_llm",
    "run_consolidation",
]
