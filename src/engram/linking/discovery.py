"""LLM-driven link discovery using Pydantic AI.

Implements A-MEM style relationship discovery that goes beyond embedding
similarity to find semantic connections like causality, temporality,
contradictions, and elaborations.

References:
- A-MEM Paper: https://arxiv.org/abs/2502.12110
- Zettelkasten Method: https://zettelkasten.de/introduction/
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from engram.models.base import OperationStatus

logger = logging.getLogger(__name__)

# Relationship types for memory links
# Extends the base types (related, supersedes, contradicts) with richer semantics
LinkType = Literal[
    "related",  # General semantic similarity
    "causal",  # A caused or led to B
    "temporal",  # A happened before B (temporal sequence)
    "contradicts",  # A conflicts with B
    "elaborates",  # B provides more detail about A
    "supersedes",  # A is newer/more accurate than B
    "generalizes",  # A is a more general form of B
    "exemplifies",  # B is a specific example of A
]

# Human-readable descriptions for each link type
LINK_TYPE_DESCRIPTIONS: dict[str, str] = {
    "related": "General semantic similarity - the memories share common themes or topics",
    "causal": "Causal relationship - one memory describes something that caused or led to what the other describes",
    "temporal": "Temporal sequence - one memory describes something that happened before the other",
    "contradicts": "Contradiction - the memories contain conflicting or incompatible information",
    "elaborates": "Elaboration - one memory provides more detail, context, or explanation for the other",
    "supersedes": "Supersession - one memory contains newer or more accurate information than the other",
    "generalizes": "Generalization - one memory is a more abstract or general form of the other",
    "exemplifies": "Exemplification - one memory is a specific instance or example of the other",
}


class DiscoveredLink(BaseModel):
    """A discovered relationship between two memories.

    Attributes:
        target_id: ID of the memory to link to.
        link_type: Type of relationship discovered.
        confidence: How confident the LLM is in this relationship (0.0-1.0).
        reasoning: Brief explanation of why this link was identified.
        bidirectional: Whether the relationship applies in both directions.
    """

    model_config = ConfigDict(extra="forbid")

    target_id: str = Field(description="ID of the memory to link to")
    link_type: LinkType = Field(description="Type of relationship")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in this relationship (0.0-1.0)",
    )
    reasoning: str = Field(description="Brief explanation of the relationship")
    bidirectional: bool = Field(
        default=True,
        description="Whether relationship applies both ways",
    )


class MemoryEvolution(BaseModel):
    """Suggested evolution for an existing memory based on new information.

    A-MEM style memory updates: when new memories are added, existing
    memories may need their metadata updated to maintain coherence.

    Attributes:
        memory_id: ID of the memory to evolve.
        field: Which field to update (tags, keywords, context).
        new_value: Value to add/set.
        reason: Why this evolution is suggested.
    """

    model_config = ConfigDict(extra="forbid")

    memory_id: str = Field(description="ID of the memory to evolve")
    field: Literal["tags", "keywords", "context"] = Field(
        description="Which metadata field to update"
    )
    new_value: str = Field(description="Value to add or set")
    reason: str = Field(description="Why this evolution is suggested")


class LinkDiscoveryResult(BaseModel):
    """Result of LLM-driven link discovery.

    Contains discovered links and optional evolution suggestions.

    IMPORTANT: Always check `status` before trusting results. When `status`
    is FAILED, empty `links` is a fallback, not a finding that no links exist.

    Attributes:
        status: Operation status - check this before trusting results.
        links: List of discovered relationships.
        evolutions: Suggested updates to existing memories.
        reasoning: Overall analysis reasoning.
        error_message: Error details when status is FAILED.
    """

    model_config = ConfigDict(extra="forbid")

    status: OperationStatus = Field(
        default=OperationStatus.SUCCESS,
        description="Operation status - check this before trusting results",
    )
    links: list[DiscoveredLink] = Field(
        default_factory=list,
        description="Discovered relationships",
    )
    evolutions: list[MemoryEvolution] = Field(
        default_factory=list,
        description="Suggested memory evolutions",
    )
    reasoning: str = Field(
        default="",
        description="Overall analysis reasoning",
    )
    error_message: str | None = Field(
        default=None,
        description="Error details when status is FAILED",
    )


# System prompt for the link discovery agent
LINK_DISCOVERY_PROMPT = """You are analyzing memories to discover meaningful relationships beyond simple similarity.

For each candidate memory, determine if there is a meaningful relationship with the new memory.

RELATIONSHIP TYPES (use exactly these names):
- "related": General semantic similarity - shared themes or topics
- "causal": One caused or led to the other
- "temporal": One happened before the other (time sequence)
- "contradicts": Conflicting or incompatible information
- "elaborates": One provides more detail about the other
- "supersedes": One is newer/more accurate than the other
- "generalizes": One is a more abstract form of the other
- "exemplifies": One is a specific example of the other

IMPORTANT GUIDELINES:
1. Only identify relationships you are confident about (>0.6 confidence)
2. Prefer specific relationship types over "related" when applicable
3. "contradicts" should only be used for genuine conflicts, not just different topics
4. Consider temporal context when determining supersedes vs related
5. Keep reasoning brief but specific

For evolutions, suggest metadata updates when:
- The new memory adds relevant tags/keywords to existing memories
- The new memory provides context that enriches existing memories
- Do NOT suggest content changes (content is immutable)
"""


async def discover_links(
    new_memory_content: str,
    new_memory_id: str,
    candidate_memories: list[dict[str, Any]],
    min_confidence: float = 0.6,
) -> LinkDiscoveryResult:
    """Discover relationships between a new memory and candidates using LLM.

    Uses Pydantic AI to analyze memories and identify semantic relationships
    that go beyond embedding similarity.

    Args:
        new_memory_content: Content of the new memory.
        new_memory_id: ID of the new memory.
        candidate_memories: List of candidate memories to analyze.
            Each dict should have: id, content, keywords (optional), tags (optional).
        min_confidence: Minimum confidence threshold for links.

    Returns:
        LinkDiscoveryResult with discovered links and evolution suggestions.
    """
    from pydantic_ai import Agent

    from engram.config import settings

    if not candidate_memories:
        return LinkDiscoveryResult(reasoning="No candidate memories to analyze")

    # Build context for the LLM
    def format_candidate(idx: int, m: dict[str, Any]) -> str:
        keywords = m.get("keywords", [])
        tags = m.get("tags", [])
        keywords_str = ", ".join(keywords) if isinstance(keywords, list) else ""
        tags_str = ", ".join(tags) if isinstance(tags, list) else ""
        return (
            f"CANDIDATE {idx + 1} (ID: {m['id']}):\n"
            f"Content: {m['content']}\n"
            f"Keywords: {keywords_str}\n"
            f"Tags: {tags_str}"
        )

    candidates_text = "\n\n".join(format_candidate(i, m) for i, m in enumerate(candidate_memories))

    user_prompt = f"""NEW MEMORY (ID: {new_memory_id}):
{new_memory_content}

CANDIDATE MEMORIES TO ANALYZE:
{candidates_text}

Analyze the relationships between the new memory and each candidate.
For each meaningful relationship found, specify the link type, confidence, and reasoning.
Also suggest any metadata evolutions for existing memories if appropriate."""

    from engram.workflows.llm_utils import run_agent_with_retry

    agent: Agent[None, LinkDiscoveryResult] = Agent(
        settings.llm_model,
        output_type=LinkDiscoveryResult,
        instructions=LINK_DISCOVERY_PROMPT,
    )

    try:
        discovered = await run_agent_with_retry(agent, user_prompt)

        # Filter by confidence threshold
        discovered.links = [link for link in discovered.links if link.confidence >= min_confidence]

        logger.info(
            "Link discovery found %d links (min_confidence=%.2f) for memory %s",
            len(discovered.links),
            min_confidence,
            new_memory_id,
        )

        return discovered

    except Exception as e:
        logger.exception("Link discovery failed: %s", e)
        return LinkDiscoveryResult(
            status=OperationStatus.FAILED,
            reasoning="Link discovery failed due to LLM error",
            error_message=str(e),
        )


async def evolve_memory(
    memory: object,
    evolution: MemoryEvolution,
) -> bool:
    """Apply an evolution to a memory.

    Calls the memory's evolve() method to update metadata fields.

    Args:
        memory: SemanticMemory to evolve (has evolve() method).
        evolution: Evolution to apply.

    Returns:
        True if evolution was applied, False otherwise.
    """
    try:
        # Use duck typing - memory should have evolve() method
        if hasattr(memory, "evolve"):
            memory.evolve(
                trigger_memory_id=evolution.memory_id,
                field=evolution.field,
                new_value=evolution.new_value,
                reason=evolution.reason,
            )
            logger.debug(
                "Evolved memory %s: %s = %s",
                evolution.memory_id,
                evolution.field,
                evolution.new_value,
            )
            return True
        else:
            logger.warning("Memory does not support evolution: %s", type(memory))
            return False
    except Exception as e:
        logger.exception("Failed to evolve memory: %s", e)
        return False


async def discover_and_apply_links(
    new_memory: object,
    candidate_memories: list[object],
    storage: object,
    min_confidence: float = 0.6,
    apply_evolutions: bool = True,
) -> int:
    """Discover links and apply them to memories.

    Convenience function that runs link discovery and applies the results.

    Args:
        new_memory: The new SemanticMemory (has add_link, id, content).
        candidate_memories: List of SemanticMemory candidates.
        storage: EngramStorage for persisting changes.
        min_confidence: Minimum confidence for links.
        apply_evolutions: Whether to apply suggested evolutions.

    Returns:
        Number of links created.
    """
    # Build candidate data for LLM
    candidates_data = [
        {
            "id": getattr(m, "id", ""),
            "content": getattr(m, "content", ""),
            "keywords": getattr(m, "keywords", []),
            "tags": getattr(m, "tags", []),
        }
        for m in candidate_memories
    ]

    result = await discover_links(
        new_memory_content=getattr(new_memory, "content", ""),
        new_memory_id=getattr(new_memory, "id", ""),
        candidate_memories=candidates_data,
        min_confidence=min_confidence,
    )

    # Check for failed discovery and log appropriately
    if result.status == OperationStatus.FAILED:
        logger.warning(
            "Link discovery failed for memory %s: %s. No links will be created.",
            getattr(new_memory, "id", "unknown"),
            result.error_message,
        )
        return 0

    links_created = 0

    # Apply discovered links
    for link in result.links:
        # Find the target memory
        target = next(
            (m for m in candidate_memories if getattr(m, "id", None) == link.target_id),
            None,
        )
        if target is None:
            continue

        # Add bidirectional links
        if hasattr(new_memory, "add_link"):
            new_memory.add_link(link.target_id, link.link_type)

        if link.bidirectional and hasattr(target, "add_link"):
            target.add_link(getattr(new_memory, "id", ""), link.link_type)

            # Strengthen the target memory (Testing Effect)
            if hasattr(target, "strengthen"):
                target.strengthen(delta=0.1)

            # Persist target changes
            if hasattr(storage, "update_semantic_memory"):
                await storage.update_semantic_memory(target)

        links_created += 1
        logger.debug(
            "Created %s link: %s <-> %s (confidence: %.2f)",
            link.link_type,
            getattr(new_memory, "id", ""),
            link.target_id,
            link.confidence,
        )

    # Apply evolutions if enabled
    if apply_evolutions and result.evolutions:
        memory_map = {getattr(m, "id", ""): m for m in candidate_memories}

        for evolution in result.evolutions:
            target = memory_map.get(evolution.memory_id)
            if target:
                if await evolve_memory(target, evolution):
                    if hasattr(storage, "update_semantic_memory"):
                        await storage.update_semantic_memory(target)

    return links_created
