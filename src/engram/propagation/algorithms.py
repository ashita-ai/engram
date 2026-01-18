"""PageRank-style confidence propagation algorithms.

Implements belief propagation for memory confidence scores,
where linked memories can boost each other's confidence.

References:
- Pearl 1982: Belief propagation on graphical models
- PageRank: Iterative ranking with damping
- TrustRank: Trust from seed nodes
- GBR: Good/Bad rank for distrust propagation

Design principles:
1. Episodes are trusted seeds (confidence = 1.0, immutable)
2. Extraction method weights trust (VERBATIM > EXTRACTED > INFERRED)
3. Bidirectional propagation between derived memories
4. Damping factor prevents runaway propagation
5. Distrust propagation for contradictions
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from engram.models import SemanticMemory
    from engram.storage import EngramStorage

logger = logging.getLogger(__name__)


# Default configuration
DEFAULT_DAMPING_FACTOR = 0.85  # Classic PageRank value
DEFAULT_MAX_ITERATIONS = 10
DEFAULT_CONVERGENCE_THRESHOLD = 0.001
DEFAULT_MAX_BOOST_PER_CYCLE = 0.15
DEFAULT_DISTRUST_PENALTY = 0.1


@dataclass
class PropagationConfig:
    """Configuration for confidence propagation.

    Attributes:
        damping_factor: Fraction of confidence that propagates (0.0-1.0).
        max_iterations: Maximum propagation iterations.
        convergence_threshold: Stop when max change < this.
        max_boost_per_cycle: Maximum confidence boost per memory per cycle.
        distrust_penalty: Penalty applied from low-confidence memories.
        min_confidence_for_propagation: Memories below this don't propagate.
    """

    damping_factor: float = DEFAULT_DAMPING_FACTOR
    max_iterations: int = DEFAULT_MAX_ITERATIONS
    convergence_threshold: float = DEFAULT_CONVERGENCE_THRESHOLD
    max_boost_per_cycle: float = DEFAULT_MAX_BOOST_PER_CYCLE
    distrust_penalty: float = DEFAULT_DISTRUST_PENALTY
    min_confidence_for_propagation: float = 0.5


class PropagationResult(BaseModel):
    """Result of confidence propagation.

    Attributes:
        memories_updated: Number of memories whose confidence changed.
        iterations: Number of iterations until convergence.
        converged: Whether propagation converged.
        total_boost_applied: Sum of all confidence boosts.
        total_distrust_applied: Sum of all distrust penalties.
        max_change_final: Maximum confidence change in final iteration.
    """

    model_config = ConfigDict(extra="forbid")

    memories_updated: int = Field(default=0, ge=0)
    iterations: int = Field(default=0, ge=0)
    converged: bool = Field(default=False)
    total_boost_applied: float = Field(default=0.0, ge=0.0)
    total_distrust_applied: float = Field(default=0.0, ge=0.0)
    max_change_final: float = Field(default=0.0, ge=0.0)


def compute_link_strength(
    source: SemanticMemory,
    target: SemanticMemory,
) -> float:
    """Compute strength of link between two memories.

    Link strength affects how much confidence propagates.
    Factors:
    - Link type (contradicts = negative, related = positive)
    - Source confidence
    - Extraction method of source

    Args:
        source: Memory propagating confidence.
        target: Memory receiving confidence.

    Returns:
        Link strength from -1.0 to 1.0.
    """
    from engram.models import ExtractionMethod

    # Base strength from extraction method
    extraction_weights = {
        ExtractionMethod.VERBATIM: 1.0,
        ExtractionMethod.EXTRACTED: 0.9,
        ExtractionMethod.INFERRED: 0.6,
    }
    base_strength = extraction_weights.get(
        source.confidence.extraction_method,
        0.6,
    )

    # Adjust for link type
    link_type = source.link_types.get(target.id, "related")
    type_multipliers = {
        "related": 1.0,
        "elaborates": 1.2,  # Elaboration strengthens
        "causal": 1.1,
        "temporal": 0.9,
        "exemplifies": 1.1,
        "generalizes": 0.9,
        "supersedes": 0.8,  # Newer info, less propagation to old
        "contradicts": -1.0,  # Negative propagation
    }
    type_mult = type_multipliers.get(link_type, 1.0)

    return base_strength * type_mult


async def propagate_confidence(
    memories: list[SemanticMemory],
    config: PropagationConfig | None = None,
) -> PropagationResult:
    """Run PageRank-style confidence propagation.

    Iteratively propagates confidence between linked memories
    until convergence or max iterations.

    Args:
        memories: List of memories to propagate between.
        config: Propagation configuration.

    Returns:
        PropagationResult with statistics.
    """
    if config is None:
        config = PropagationConfig()

    if not memories:
        return PropagationResult(converged=True)

    # Build memory lookup
    memory_map = {m.id: m for m in memories}

    result = PropagationResult()
    memories_changed: set[str] = set()

    for iteration in range(config.max_iterations):
        max_change = 0.0

        for memory in memories:
            if memory.confidence.value < config.min_confidence_for_propagation:
                continue

            # Get linked memories
            linked_ids = memory.related_ids
            if not linked_ids:
                continue

            # Compute incoming trust from links
            incoming_trust = 0.0
            incoming_distrust = 0.0

            for linked_id in linked_ids:
                linked_memory = memory_map.get(linked_id)
                if linked_memory is None:
                    continue

                link_strength = compute_link_strength(linked_memory, memory)

                if link_strength > 0:
                    # Positive propagation (trust)
                    trust_contribution = (
                        linked_memory.confidence.value * config.damping_factor * link_strength
                    )
                    incoming_trust += trust_contribution
                elif link_strength < 0:
                    # Negative propagation (distrust from contradictions)
                    distrust_contribution = (
                        linked_memory.confidence.value
                        * config.distrust_penalty
                        * abs(link_strength)
                    )
                    incoming_distrust += distrust_contribution

            # Apply boost (capped)
            old_confidence = memory.confidence.value

            if incoming_trust > 0:
                # Apply boost by increasing supporting episodes and recomputing
                memory.confidence.supporting_episodes += 1
                old_value = memory.confidence.value
                memory.confidence.recompute()
                actual_boost = memory.confidence.value - old_value
                result.total_boost_applied += max(0, actual_boost)

            if incoming_distrust > 0:
                # Apply distrust as confidence reduction
                penalty = min(incoming_distrust, config.max_boost_per_cycle)
                new_conf = max(0.1, memory.confidence.value - penalty)
                memory.confidence.value = new_conf
                result.total_distrust_applied += penalty

            # Track change
            change = abs(memory.confidence.value - old_confidence)
            if change > 0:
                memories_changed.add(memory.id)
            max_change = max(max_change, change)

        result.iterations = iteration + 1
        result.max_change_final = max_change

        # Check convergence
        if max_change < config.convergence_threshold:
            result.converged = True
            break

    result.memories_updated = len(memories_changed)

    logger.info(
        "Confidence propagation: %d iterations, %d memories updated, converged=%s",
        result.iterations,
        result.memories_updated,
        result.converged,
    )

    return result


async def propagate_distrust(
    contradicted_memory: SemanticMemory,
    linked_memories: list[SemanticMemory],
    penalty: float = DEFAULT_DISTRUST_PENALTY,
) -> int:
    """Propagate distrust from a contradicted/low-confidence memory.

    When a memory is contradicted or has very low confidence,
    linked memories should also be penalized.

    Args:
        contradicted_memory: The source of distrust.
        linked_memories: Memories to potentially penalize.
        penalty: Base penalty to apply.

    Returns:
        Number of memories penalized.
    """
    penalized = 0

    source_confidence = contradicted_memory.confidence.value

    # Only propagate distrust if source is low confidence
    if source_confidence >= 0.5:
        return 0

    for memory in linked_memories:
        # Check if linked
        if contradicted_memory.id not in memory.related_ids:
            continue

        # Check link type
        link_type = memory.link_types.get(contradicted_memory.id, "related")

        # Contradictions propagate more distrust
        if link_type == "contradicts":
            effective_penalty = penalty * 2
        else:
            effective_penalty = penalty * (1 - source_confidence)

        # Apply penalty
        old_conf = memory.confidence.value
        new_conf = max(0.1, old_conf - effective_penalty)
        memory.confidence.value = new_conf
        penalized += 1

        logger.debug(
            "Distrust propagation: %s -> %s (%.2f -> %.2f)",
            contradicted_memory.id,
            memory.id,
            old_conf,
            new_conf,
        )

    return penalized


async def run_propagation_cycle(
    storage: EngramStorage,
    user_id: str,
    org_id: str | None = None,
    config: PropagationConfig | None = None,
) -> PropagationResult:
    """Run a full propagation cycle for a user's memories.

    Fetches all semantic memories, runs propagation, and persists changes.

    Args:
        storage: Storage instance.
        user_id: User ID.
        org_id: Optional organization ID.
        config: Propagation configuration.

    Returns:
        PropagationResult with statistics.
    """
    # Fetch all semantic memories
    memories = await storage.list_semantic_memories(user_id, org_id)

    if not memories:
        return PropagationResult(converged=True)

    # Run propagation
    result = await propagate_confidence(memories, config)

    # Persist changes
    if result.memories_updated > 0:
        for memory in memories:
            await storage.update_semantic_memory(memory)

    logger.info(
        "Propagation cycle complete for user %s: %d memories updated",
        user_id,
        result.memories_updated,
    )

    return result
