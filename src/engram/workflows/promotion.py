"""Promotion workflow for promoting memories through the hierarchy.

This workflow runs during decay to:
1. Find semantic memories with high selectivity (well-consolidated)
2. Detect behavioral patterns suitable for procedural memory
3. Promote patterns to procedural memories

Buffer promotion hierarchy:
    Working (volatile) → Episodic → Semantic → Procedural
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


# Keywords that indicate behavioral patterns suitable for procedural memory
BEHAVIORAL_PATTERNS = [
    "prefers",
    "preference",
    "always",
    "usually",
    "tends to",
    "likes to",
    "wants",
    "expects",
    "style",
    "habit",
    "pattern",
]


class PromotionResult(BaseModel):
    """Result of a promotion workflow run.

    Attributes:
        memories_analyzed: Number of semantic memories analyzed.
        procedural_created: Number of procedural memories created.
        patterns_detected: List of pattern descriptions that were promoted.
    """

    model_config = ConfigDict(extra="forbid")

    memories_analyzed: int = Field(ge=0)
    procedural_created: int = Field(ge=0)
    patterns_detected: list[str] = Field(default_factory=list)


def _is_behavioral_pattern(content: str) -> bool:
    """Check if content describes a behavioral pattern.

    Args:
        content: Memory content to check.

    Returns:
        True if content appears to describe a behavioral pattern.
    """
    content_lower = content.lower()
    return any(pattern in content_lower for pattern in BEHAVIORAL_PATTERNS)


def _should_promote_to_procedural(memory: SemanticMemory) -> bool:
    """Determine if a semantic memory should be promoted to procedural.

    Promotion criteria:
    - High selectivity score (>= 0.5) - well-consolidated
    - Multiple consolidation passes (>= 2) - stable over time
    - High confidence (>= 0.7) - reliable information
    - Contains behavioral pattern keywords

    Args:
        memory: SemanticMemory to evaluate.

    Returns:
        True if memory should be promoted.
    """
    # Must have high selectivity (survived consolidation)
    if memory.selectivity_score < 0.5:
        return False

    # Must have gone through multiple consolidation passes
    if memory.consolidation_passes < 2:
        return False

    # Must have reasonable confidence
    if memory.confidence.value < 0.7:
        return False

    # Must describe a behavioral pattern
    return _is_behavioral_pattern(memory.content)


def _extract_trigger_context(content: str) -> str:
    """Extract the trigger context from a behavioral pattern.

    Args:
        content: The behavioral pattern content.

    Returns:
        A brief trigger context description.
    """
    content_lower = content.lower()

    # Try to identify what triggers this pattern
    if "when" in content_lower:
        # Extract context after "when"
        idx = content_lower.find("when")
        return content[idx:].strip()

    if "for" in content_lower:
        # "prefers X for Y"
        idx = content_lower.find("for")
        return content[idx:].strip()

    if "in" in content_lower:
        # "always X in Y context"
        idx = content_lower.find("in")
        return content[idx:].strip()

    # Default: general context
    return "general interaction"


async def run_promotion(
    storage: EngramStorage,
    embedder: Embedder,
    user_id: str,
    org_id: str | None = None,
) -> PromotionResult:
    """Run the promotion workflow.

    This workflow:
    1. Fetches semantic memories with high selectivity
    2. Identifies behavioral patterns
    3. Creates procedural memories from patterns
    4. Links new procedural memories to source semantics

    Args:
        storage: EngramStorage instance.
        embedder: Embedder for generating vectors.
        user_id: User ID for multi-tenancy.
        org_id: Optional organization ID.

    Returns:
        PromotionResult with processing statistics.
    """
    from engram.models import ProceduralMemory

    # 1. Fetch semantic memories
    semantics = await storage.list_semantic_memories(
        user_id=user_id,
        org_id=org_id,
        include_archived=False,
    )

    if not semantics:
        logger.info("No semantic memories found for promotion")
        return PromotionResult(
            memories_analyzed=0,
            procedural_created=0,
        )

    logger.info(f"Analyzing {len(semantics)} semantic memories for promotion")

    # 2. Check existing procedural memories to avoid duplicates
    existing_procedurals = await _get_existing_procedural_contents(storage, user_id, org_id)

    promoted = 0
    patterns: list[str] = []

    for memory in semantics:
        # 3. Check if should be promoted
        if not _should_promote_to_procedural(memory):
            continue

        # 4. Check for duplicates
        if _is_duplicate_pattern(memory.content, existing_procedurals):
            logger.debug(f"Skipping duplicate pattern: {memory.content[:50]}...")
            continue

        # 5. Create procedural memory
        embedding = await embedder.embed(memory.content)

        procedural = ProceduralMemory(
            content=memory.content,
            trigger_context=_extract_trigger_context(memory.content),
            source_episode_ids=memory.source_episode_ids,
            related_ids=[memory.id],  # Link back to source semantic
            user_id=user_id,
            org_id=org_id,
            embedding=embedding,
        )
        procedural.confidence.value = memory.confidence.value

        await storage.store_procedural(procedural)
        promoted += 1
        patterns.append(memory.content)
        existing_procedurals.add(memory.content.lower())

        logger.info(f"Promoted to procedural: {memory.content[:50]}...")

        # 6. Add link from semantic to procedural
        memory.add_link(procedural.id)
        await storage.update_semantic_memory(memory)

    logger.info(
        f"Promotion complete: {promoted} procedural memories created from {len(semantics)} analyzed"
    )

    return PromotionResult(
        memories_analyzed=len(semantics),
        procedural_created=promoted,
        patterns_detected=patterns,
    )


async def _get_existing_procedural_contents(
    storage: EngramStorage,
    user_id: str,
    org_id: str | None,
) -> set[str]:
    """Get content of existing procedural memories for deduplication.

    Args:
        storage: Storage instance.
        user_id: User ID.
        org_id: Optional org ID.

    Returns:
        Set of lowercase content strings.
    """
    procedurals = await storage.list_procedural_memories(user_id, org_id)
    return {p.content.lower() for p in procedurals}


def _is_duplicate_pattern(content: str, existing: set[str]) -> bool:
    """Check if content is a duplicate of existing patterns.

    Args:
        content: New pattern content.
        existing: Set of existing pattern contents (lowercase).

    Returns:
        True if duplicate found.
    """
    content_lower = content.lower()

    # Exact match
    if content_lower in existing:
        return True

    # Substring match (new is subset of existing or vice versa)
    for existing_content in existing:
        if content_lower in existing_content or existing_content in content_lower:
            return True

    return False


__all__ = [
    "PromotionResult",
    "run_promotion",
]
