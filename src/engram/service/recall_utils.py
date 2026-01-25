"""Shared recall utilities for EngramService.

Contains utility functions used across operations.py and recall.py:
- enrich_with_sources: Add source episode details to results
- follow_links: Multi-hop reasoning via related_ids
- apply_negation_filtering: Filter memories matching negated patterns
- get_result_embedding: Fetch embedding for a recall result
- fetch_memory_by_id: Fetch memory and convert to RecallResult

These functions are extracted to avoid code duplication (DRY principle).
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from engram.models import Episode, Staleness

from .helpers import cosine_similarity
from .models import RecallResult, SourceEpisodeSummary

if TYPE_CHECKING:
    from engram.embeddings import Embedder
    from engram.storage import EngramStorage

logger = logging.getLogger(__name__)


async def enrich_with_sources(
    results: list[RecallResult],
    user_id: str,
    storage: EngramStorage,
) -> list[RecallResult]:
    """Enrich recall results with source episode details.

    Args:
        results: List of recall results to enrich.
        user_id: User ID for isolation.
        storage: Storage backend for fetching episodes.

    Returns:
        List of RecallResult with source_episodes populated.
    """
    enriched: list[RecallResult] = []

    for result in results:
        source_episodes: list[SourceEpisodeSummary] = []

        # Structured memories have a single source episode
        if result.memory_type == "structured" and result.source_episode_id:
            ep = await storage.get_episode(result.source_episode_id, user_id)
            if ep:
                source_episodes.append(
                    SourceEpisodeSummary(
                        id=ep.id,
                        content=ep.content,
                        role=ep.role,
                        timestamp=ep.timestamp.isoformat(),
                    )
                )

        # Semantic memories have multiple source episodes
        elif result.memory_type == "semantic":
            sem = await storage.get_semantic(result.memory_id, user_id)
            if sem:
                for ep_id in sem.source_episode_ids:
                    ep = await storage.get_episode(ep_id, user_id)
                    if ep:
                        source_episodes.append(
                            SourceEpisodeSummary(
                                id=ep.id,
                                content=ep.content,
                                role=ep.role,
                                timestamp=ep.timestamp.isoformat(),
                            )
                        )

        # Procedural memories have multiple source episodes
        elif result.memory_type == "procedural":
            proc = await storage.get_procedural(result.memory_id, user_id)
            if proc:
                for ep_id in proc.source_episode_ids:
                    ep = await storage.get_episode(ep_id, user_id)
                    if ep:
                        source_episodes.append(
                            SourceEpisodeSummary(
                                id=ep.id,
                                content=ep.content,
                                role=ep.role,
                                timestamp=ep.timestamp.isoformat(),
                            )
                        )

        # Create enriched result with sources
        enriched.append(
            RecallResult(
                memory_type=result.memory_type,
                content=result.content,
                score=result.score,
                confidence=result.confidence,
                memory_id=result.memory_id,
                source_episode_id=result.source_episode_id,
                source_episode_ids=result.source_episode_ids,
                source_episodes=source_episodes,
                related_ids=result.related_ids,
                hop_distance=result.hop_distance,
                staleness=result.staleness,
                consolidated_at=result.consolidated_at,
                metadata=result.metadata,
            )
        )

    return enriched


async def follow_links(
    results: list[RecallResult],
    user_id: str,
    max_hops: int,
    limit: int,
    storage: EngramStorage,
) -> list[RecallResult]:
    """Follow related_ids links to discover connected memories.

    Implements multi-hop reasoning by traversing related_ids from
    semantic and procedural memories.

    Args:
        results: Initial recall results.
        user_id: User ID for isolation.
        max_hops: Maximum link traversal depth.
        limit: Maximum total results.
        storage: Storage backend for fetching memories.

    Returns:
        Extended list of RecallResult including linked memories.
    """
    all_results = list(results)
    seen_ids: set[str] = {r.memory_id for r in results}
    current_frontier = results

    for hop in range(1, max_hops + 1):
        # Collect all related_ids from current frontier
        related_ids_to_fetch: set[str] = set()
        for r in current_frontier:
            for related_id in r.related_ids:
                if related_id not in seen_ids:
                    related_ids_to_fetch.add(related_id)

        if not related_ids_to_fetch:
            break  # No more links to follow

        # Fetch related memories
        next_frontier: list[RecallResult] = []
        for related_id in related_ids_to_fetch:
            if len(all_results) >= limit:
                break

            memory_result = await fetch_memory_by_id(related_id, user_id, hop, storage)
            if memory_result:
                next_frontier.append(memory_result)
                all_results.append(memory_result)
                seen_ids.add(related_id)

        current_frontier = next_frontier

    return all_results[:limit]


async def fetch_memory_by_id(
    memory_id: str,
    user_id: str,
    hop_distance: int,
    storage: EngramStorage,
) -> RecallResult | None:
    """Fetch a memory by ID and convert to RecallResult.

    Args:
        memory_id: ID of the memory to fetch.
        user_id: User ID for isolation.
        hop_distance: How many hops from original query.
        storage: Storage backend for fetching memories.

    Returns:
        RecallResult or None if not found.
    """
    # Determine memory type from ID prefix
    if memory_id.startswith("sem_"):
        sem = await storage.get_semantic(memory_id, user_id)
        if sem:
            return RecallResult(
                memory_type="semantic",
                content=sem.content,
                score=0.0,  # No similarity score for linked memories
                confidence=sem.confidence.value,
                memory_id=sem.id,
                source_episode_ids=sem.source_episode_ids,
                related_ids=sem.related_ids,
                hop_distance=hop_distance,
                staleness=Staleness.FRESH,
                consolidated_at=sem.derived_at.isoformat(),
                metadata={
                    "consolidation_strength": sem.consolidation_strength,
                    "derived_at": sem.derived_at.isoformat(),
                    "linked": True,
                },
            )

    elif memory_id.startswith("proc_"):
        proc = await storage.get_procedural(memory_id, user_id)
        if proc:
            return RecallResult(
                memory_type="procedural",
                content=proc.content,
                score=0.0,
                confidence=proc.confidence.value,
                memory_id=proc.id,
                source_episode_ids=proc.source_episode_ids,
                related_ids=proc.related_ids,
                hop_distance=hop_distance,
                staleness=Staleness.FRESH,
                consolidated_at=proc.derived_at.isoformat(),
                metadata={
                    "trigger_context": proc.trigger_context,
                    "linked": True,
                },
            )

    elif memory_id.startswith("struct_"):
        struct = await storage.get_structured(memory_id, user_id)
        if struct:
            return RecallResult(
                memory_type="structured",
                content=struct.to_embedding_text() or struct.summary,
                score=0.0,
                confidence=struct.confidence.value,
                memory_id=struct.id,
                source_episode_id=struct.source_episode_id,
                source_episode_ids=[struct.source_episode_id],
                hop_distance=hop_distance,
                staleness=Staleness.FRESH,
                consolidated_at=struct.derived_at.isoformat(),
                metadata={
                    "mode": struct.mode,
                    "enriched": struct.enriched,
                    "linked": True,
                },
            )

    return None


async def apply_negation_filtering(
    results: list[RecallResult],
    user_id: str,
    storage: EngramStorage,
    embedder: Embedder,
    working_memory: list[Episode],
    org_id: str | None = None,
    similarity_threshold: float | None = 0.75,
) -> list[RecallResult]:
    """Filter out memories that match negated patterns.

    Uses a two-pronged approach:
    1. Pattern-based (substring): Fast, catches exact keyword matches
    2. Embedding-based (semantic): Catches related terms like "Mongo" ≈ "MongoDB"

    Args:
        results: List of recall results to filter.
        user_id: User ID for multi-tenancy.
        storage: Storage backend for fetching memories.
        embedder: Embedder for computing embeddings.
        working_memory: List of working memory episodes.
        org_id: Optional organization ID filter.
        similarity_threshold: Threshold for semantic similarity filtering.
            If None, only pattern-based filtering is used.

    Returns:
        Filtered list of results with negated items removed.
    """
    # Collect negation patterns from StructuredMemory.negations
    negated_patterns: set[str] = set()

    # Get from StructuredMemory.negations
    structured_memories = await storage.list_structured_memories(
        user_id, org_id, with_negations_only=True
    )
    for struct in structured_memories:
        for negation in struct.negations:
            negated_patterns.add(negation.pattern.lower())

    if not negated_patterns:
        return results

    # Prepare embedding-based filtering if threshold is set
    pattern_embeddings: dict[str, list[float]] = {}
    if similarity_threshold is not None:
        # Embed all unique negated patterns (batch for efficiency)
        unique_patterns = list(negated_patterns)
        if unique_patterns:
            embeddings = await embedder.embed_batch(unique_patterns)
            pattern_embeddings = dict(zip(unique_patterns, embeddings, strict=True))

    filtered_results: list[RecallResult] = []
    pattern_filtered = 0
    semantic_filtered = 0

    for result in results:
        # Phase 1: Pattern-based filtering (fast, word boundary match)
        # Use word boundaries for short patterns (<=3 chars) to avoid false positives
        # e.g., "r" should match "R language" but not "Jordan" or "programmer"
        content_lower = result.content.lower()
        is_pattern_negated = False
        for pattern in negated_patterns:
            if len(pattern) <= 3:
                # Use word boundary regex for short patterns
                if re.search(rf"\b{re.escape(pattern)}\b", content_lower):
                    is_pattern_negated = True
                    break
            else:
                # Substring match for longer patterns
                if pattern in content_lower:
                    is_pattern_negated = True
                    break

        if is_pattern_negated:
            pattern_filtered += 1
            logger.debug(
                f"Pattern-filtered {result.memory_type} {result.memory_id}: matches negated pattern"
            )
            continue

        # Phase 2: Embedding-based filtering (semantic similarity)
        is_semantic_negated = False
        if similarity_threshold is not None and pattern_embeddings:
            # Get embedding for this result's content
            result_embedding = await get_result_embedding(result, user_id, storage, working_memory)

            if result_embedding is not None:
                # Check similarity against all negated pattern embeddings
                for pattern, pattern_emb in pattern_embeddings.items():
                    similarity = cosine_similarity(result_embedding, pattern_emb)
                    if similarity >= similarity_threshold:
                        is_semantic_negated = True
                        semantic_filtered += 1
                        logger.debug(
                            f"Semantic-filtered {result.memory_type} {result.memory_id}: "
                            f"similarity {similarity:.2f} to '{pattern}'"
                        )
                        break

        if not is_semantic_negated:
            filtered_results.append(result)

    total_filtered = pattern_filtered + semantic_filtered
    if total_filtered > 0:
        logger.info(
            f"Negation filter: {pattern_filtered} pattern-based, {semantic_filtered} semantic-based"
        )

    return filtered_results


async def get_result_embedding(
    result: RecallResult,
    user_id: str,
    storage: EngramStorage,
    working_memory: list[Episode],
) -> list[float] | None:
    """Get embedding for a recall result from storage.

    Args:
        result: The recall result.
        user_id: User ID for multi-tenancy.
        storage: Storage backend for fetching memories.
        working_memory: List of working memory episodes.

    Returns:
        Embedding vector or None if not found.
    """
    if result.memory_type == "episodic":
        ep = await storage.get_episode(result.memory_id, user_id)
        return ep.embedding if ep else None

    elif result.memory_type == "structured":
        struct = await storage.get_structured(result.memory_id, user_id)
        return struct.embedding if struct else None

    elif result.memory_type == "semantic":
        sem = await storage.get_semantic(result.memory_id, user_id)
        return sem.embedding if sem else None

    elif result.memory_type == "procedural":
        proc = await storage.get_procedural(result.memory_id, user_id)
        return proc.embedding if proc else None

    elif result.memory_type == "working":
        # Working memory is in-memory, find by ID
        for ep in working_memory:
            if ep.id == result.memory_id:
                return ep.embedding
        return None

    return None


__all__ = [
    "apply_negation_filtering",
    "enrich_with_sources",
    "fetch_memory_by_id",
    "follow_links",
    "get_result_embedding",
]
