"""Recall mixin for EngramService.

Provides recall() and recall_at() methods for searching memories.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from datetime import datetime
from typing import TYPE_CHECKING

from engram.models import AuditEntry, Episode, Staleness

from .helpers import cosine_similarity, mmr_rerank
from .models import RecallResult, SourceEpisodeSummary
from .query_expansion import get_combined_embedding

if TYPE_CHECKING:
    from engram.config import Settings
    from engram.embeddings import Embedder
    from engram.storage import EngramStorage

logger = logging.getLogger(__name__)


class RecallMixin:
    """Mixin providing recall functionality.

    Expects these attributes from the base class:
    - storage: EngramStorage
    - embedder: Embedder
    - settings: Settings
    - _working_memory: list[Episode]
    """

    storage: EngramStorage
    embedder: Embedder
    settings: Settings
    _working_memory: list[Episode]

    async def recall(
        self,
        query: str,
        user_id: str,
        org_id: str | None = None,
        limit: int = 10,
        min_confidence: float | None = None,
        min_selectivity: float = 0.0,
        memory_types: list[str] | None = None,
        include_sources: bool = False,
        follow_links: bool = False,
        max_hops: int = 2,
        freshness: str = "best_effort",
        apply_negation_filter: bool = True,
        negation_similarity_threshold: float | None = 0.75,
        include_system_prompts: bool = False,
        diversity: float = 0.0,
        expand_query: bool = False,
    ) -> list[RecallResult]:
        """Recall memories by semantic similarity.

        Searches across memory types and returns unified results
        sorted by similarity score, with optional diversity reranking.

        Memory types:
        - episodic: Raw interactions (ground truth)
        - structured: Per-episode intelligence (fast or enriched)
        - semantic: Cross-episode summaries
        - procedural: Behavioral patterns
        - working: Current session (in-memory)

        Args:
            query: Natural language query.
            user_id: User ID for multi-tenancy isolation.
            org_id: Optional organization ID filter.
            limit: Maximum results per memory type.
            min_confidence: Minimum confidence for structured/semantic.
            min_selectivity: Minimum selectivity for semantic memories (0.0-1.0).
            memory_types: List of memory types to search. None means all types.
                Valid types: episodic, structured, semantic, procedural, working.
            include_sources: Whether to include source episodes in results.
            follow_links: Enable multi-hop reasoning via related_ids.
            max_hops: Maximum link traversal depth when follow_links=True.
            freshness: Freshness mode - "best_effort" returns all, "fresh_only" only
                returns fully consolidated memories.
            apply_negation_filter: Filter out memories that match negated patterns.
            negation_similarity_threshold: Semantic similarity threshold for negation filtering.
            include_system_prompts: Include system prompt episodes in results (default False).
            diversity: Diversity parameter for MMR reranking (0.0-1.0). Higher values
                return more diverse results at the cost of some relevance. 0.0 = no
                diversity reranking (default), 0.3 = recommended balance.
            expand_query: Expand query with LLM-generated related terms for better recall.

        Returns:
            List of RecallResult sorted by similarity score (or MMR score if diversity > 0).
        """
        start_time = time.monotonic()

        # Valid memory types
        all_types = {"episodic", "structured", "semantic", "procedural", "working"}
        types_to_search = set(memory_types) if memory_types is not None else all_types

        # Generate query embedding (with optional expansion)
        query_vector = await get_combined_embedding(
            query=query,
            embedder=self.embedder,
            expand=expand_query,
        )

        # Use larger search limit when negation filtering is enabled
        search_limit = limit * 3 if apply_negation_filter else limit

        results: list[RecallResult] = []

        # Search each memory type
        results.extend(
            await self._search_episodic(
                query_vector, user_id, org_id, search_limit, types_to_search, include_system_prompts
            )
        )
        results.extend(
            await self._search_structured(
                query_vector, user_id, org_id, search_limit, min_confidence, types_to_search
            )
        )
        results.extend(
            await self._search_semantic(
                query_vector,
                user_id,
                org_id,
                search_limit,
                min_confidence,
                min_selectivity,
                types_to_search,
            )
        )
        results.extend(
            await self._search_procedural(
                query_vector, user_id, org_id, search_limit, min_confidence, types_to_search
            )
        )
        results.extend(await self._search_working(query_vector, user_id, org_id, types_to_search))

        # Sort by score descending
        results.sort(key=lambda r: r.score, reverse=True)

        # Apply negation filtering
        if apply_negation_filter:
            results = await self._apply_negation_filtering(
                results, user_id, org_id, negation_similarity_threshold
            )

        # Apply freshness filtering
        if freshness == "fresh_only":
            results = [r for r in results if r.staleness == Staleness.FRESH]

        # Apply MMR diversity reranking if diversity > 0
        if diversity > 0 and len(results) > limit:
            final_results = await self._apply_diversity_reranking(
                results=results,
                user_id=user_id,
                limit=limit,
                diversity=diversity,
            )
        else:
            final_results = results[:limit]

        # Multi-hop reasoning: follow links to related memories
        if follow_links and max_hops > 0:
            final_results = await self._follow_links(
                results=final_results,
                user_id=user_id,
                max_hops=max_hops,
                limit=limit,
            )

        # Include source episodes if requested
        if include_sources:
            final_results = await self._enrich_with_sources(final_results, user_id)

        # Testing Effect: strengthen retrieved memories
        if self.settings.retrieval_strengthening_enabled:
            await self._apply_retrieval_strengthening(final_results, user_id)

        # Log audit entry
        duration_ms = int((time.monotonic() - start_time) * 1000)
        query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]
        result_memory_types = list({r.memory_type for r in final_results})
        audit_entry = AuditEntry.for_recall(
            user_id=user_id,
            query_hash=query_hash,
            results_count=len(final_results),
            memory_types=result_memory_types,
            org_id=org_id,
            duration_ms=duration_ms,
        )
        await self.storage.log_audit(audit_entry)

        return final_results

    async def recall_at(
        self,
        query: str,
        as_of: datetime,
        user_id: str,
        org_id: str | None = None,
        limit: int = 10,
        min_confidence: float | None = None,
        memory_types: list[str] | None = None,
    ) -> list[RecallResult]:
        """Recall memories as they existed at a specific point in time.

        This is a bi-temporal query that only returns memories that were
        derived before the `as_of` timestamp.

        Args:
            query: Natural language query.
            as_of: Point in time to query (only memories derived before this).
            user_id: User ID for multi-tenancy isolation.
            org_id: Optional organization ID filter.
            limit: Maximum results per memory type.
            min_confidence: Minimum confidence for structured.
            memory_types: List of memory types to search. None means all types.

        Returns:
            List of RecallResult sorted by similarity score.
        """
        start_time = time.monotonic()

        # recall_at supports episodic and structured
        all_types = {"episodic", "structured"}
        types_to_search = set(memory_types) & all_types if memory_types is not None else all_types

        query_vector = await self.embedder.embed(query)
        results: list[RecallResult] = []

        # Search episodes (filter by timestamp <= as_of)
        if "episodic" in types_to_search:
            scored_episodes = await self.storage.search_episodes(
                query_vector=query_vector,
                user_id=user_id,
                org_id=org_id,
                limit=limit,
                timestamp_before=as_of,
            )
            for scored_ep in scored_episodes:
                ep = scored_ep.memory
                results.append(
                    RecallResult(
                        memory_type="episodic",
                        content=ep.content,
                        score=scored_ep.score,
                        memory_id=ep.id,
                        metadata={
                            "role": ep.role,
                            "importance": ep.importance,
                            "timestamp": ep.timestamp.isoformat(),
                        },
                    )
                )

        # Search structured (filter by derived_at <= as_of)
        if "structured" in types_to_search:
            scored_structured = await self.storage.search_structured(
                query_vector=query_vector,
                user_id=user_id,
                org_id=org_id,
                limit=limit,
                min_confidence=min_confidence,
                derived_at_before=as_of,
            )
            for scored in scored_structured:
                struct = scored.memory
                results.append(
                    RecallResult(
                        memory_type="structured",
                        content=struct.summary or struct.to_embedding_text(),
                        score=scored.score,
                        confidence=struct.confidence.value,
                        memory_id=struct.id,
                        source_episode_id=struct.source_episode_id,
                        metadata={
                            "mode": struct.mode,
                            "enriched": struct.enriched,
                            "derived_at": struct.derived_at.isoformat(),
                            "emails": struct.emails,
                            "phones": struct.phones,
                            "urls": struct.urls,
                        },
                    )
                )

        results.sort(key=lambda r: r.score, reverse=True)
        final_results = results[:limit]

        # Log audit entry
        duration_ms = int((time.monotonic() - start_time) * 1000)
        query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]
        result_types = list({r.memory_type for r in final_results})
        audit_entry = AuditEntry.for_recall(
            user_id=user_id,
            query_hash=query_hash,
            results_count=len(final_results),
            memory_types=result_types,
            org_id=org_id,
            duration_ms=duration_ms,
        )
        await self.storage.log_audit(audit_entry)

        return final_results

    # --- Private search methods for each memory type ---

    async def _search_episodic(
        self,
        query_vector: list[float],
        user_id: str,
        org_id: str | None,
        limit: int,
        types_to_search: set[str],
        include_system_prompts: bool = False,
    ) -> list[RecallResult]:
        """Search episodic memories."""
        if "episodic" not in types_to_search:
            return []

        scored_episodes = await self.storage.search_episodes(
            query_vector=query_vector,
            user_id=user_id,
            org_id=org_id,
            limit=limit,
        )

        results: list[RecallResult] = []
        for scored_ep in scored_episodes:
            ep = scored_ep.memory

            # Skip system prompts unless explicitly requested
            if not include_system_prompts and ep.role == "system":
                continue

            results.append(
                RecallResult(
                    memory_type="episodic",
                    content=ep.content,
                    score=scored_ep.score,
                    memory_id=ep.id,
                    staleness=Staleness.FRESH,
                    metadata={
                        "role": ep.role,
                        "importance": ep.importance,
                        "timestamp": ep.timestamp.isoformat(),
                        "summarized": ep.summarized,
                        "structured": ep.structured,
                    },
                )
            )

        return results

    async def _search_structured(
        self,
        query_vector: list[float],
        user_id: str,
        org_id: str | None,
        limit: int,
        min_confidence: float | None,
        types_to_search: set[str],
    ) -> list[RecallResult]:
        """Search structured memories."""
        if "structured" not in types_to_search:
            return []

        scored_structured = await self.storage.search_structured(
            query_vector=query_vector,
            user_id=user_id,
            org_id=org_id,
            limit=limit,
            min_confidence=min_confidence,
        )

        results: list[RecallResult] = []
        for scored in scored_structured:
            struct = scored.memory
            results.append(
                RecallResult(
                    memory_type="structured",
                    content=struct.summary or struct.to_embedding_text(),
                    score=scored.score,
                    confidence=struct.confidence.value,
                    memory_id=struct.id,
                    source_episode_id=struct.source_episode_id,
                    staleness=Staleness.FRESH,
                    metadata={
                        "mode": struct.mode,
                        "enriched": struct.enriched,
                        "emails": struct.emails,
                        "phones": struct.phones,
                        "urls": struct.urls,
                        "people": [p.model_dump() for p in struct.people] if struct.people else [],
                        "preferences": [p.model_dump() for p in struct.preferences]
                        if struct.preferences
                        else [],
                        "negations": [n.model_dump() for n in struct.negations]
                        if struct.negations
                        else [],
                        "derived_at": struct.derived_at.isoformat(),
                    },
                )
            )

        return results

    async def _search_semantic(
        self,
        query_vector: list[float],
        user_id: str,
        org_id: str | None,
        limit: int,
        min_confidence: float | None,
        min_selectivity: float,
        types_to_search: set[str],
    ) -> list[RecallResult]:
        """Search semantic memories."""
        if "semantic" not in types_to_search:
            return []

        scored_semantics = await self.storage.search_semantic(
            query_vector=query_vector,
            user_id=user_id,
            org_id=org_id,
            limit=limit,
            min_confidence=min_confidence,
            min_selectivity=min_selectivity,
        )

        results: list[RecallResult] = []
        for scored in scored_semantics:
            sem = scored.memory
            results.append(
                RecallResult(
                    memory_type="semantic",
                    content=sem.content,
                    score=scored.score,
                    confidence=sem.confidence.value,
                    memory_id=sem.id,
                    source_episode_ids=sem.source_episode_ids,
                    related_ids=sem.related_ids,
                    staleness=Staleness.FRESH,
                    consolidated_at=sem.derived_at.isoformat(),
                    metadata={
                        "consolidation_strength": sem.consolidation_strength,
                        "derived_at": sem.derived_at.isoformat(),
                    },
                )
            )

        return results

    async def _search_procedural(
        self,
        query_vector: list[float],
        user_id: str,
        org_id: str | None,
        limit: int,
        min_confidence: float | None,
        types_to_search: set[str],
    ) -> list[RecallResult]:
        """Search procedural memories."""
        if "procedural" not in types_to_search:
            return []

        scored_procedural = await self.storage.search_procedural(
            query_vector=query_vector,
            user_id=user_id,
            org_id=org_id,
            limit=limit,
            min_confidence=min_confidence,
        )

        results: list[RecallResult] = []
        for scored in scored_procedural:
            proc = scored.memory
            results.append(
                RecallResult(
                    memory_type="procedural",
                    content=proc.content,
                    score=scored.score,
                    confidence=proc.confidence.value,
                    memory_id=proc.id,
                    source_episode_ids=proc.source_episode_ids,
                    related_ids=proc.related_ids,
                    staleness=Staleness.FRESH,
                    consolidated_at=proc.derived_at.isoformat(),
                    metadata={
                        "trigger_context": proc.trigger_context,
                        "retrieval_count": proc.retrieval_count,
                    },
                )
            )

        return results

    async def _search_working(
        self,
        query_vector: list[float],
        user_id: str,
        org_id: str | None,
        types_to_search: set[str],
    ) -> list[RecallResult]:
        """Search working memory (in-memory episodes from current session)."""
        if "working" not in types_to_search:
            return []

        results: list[RecallResult] = []
        for ep in self._working_memory:
            # Filter by user/org
            if ep.user_id != user_id:
                continue
            if org_id is not None and ep.org_id != org_id:
                continue

            # Calculate similarity
            if ep.embedding:
                score = cosine_similarity(query_vector, ep.embedding)
                results.append(
                    RecallResult(
                        memory_type="working",
                        content=ep.content,
                        score=score,
                        memory_id=ep.id,
                        staleness=Staleness.FRESH,
                        metadata={
                            "role": ep.role,
                            "importance": ep.importance,
                        },
                    )
                )

        return results

    async def _apply_negation_filtering(
        self,
        results: list[RecallResult],
        user_id: str,
        org_id: str | None = None,
        similarity_threshold: float | None = 0.75,
    ) -> list[RecallResult]:
        """Filter out memories that match negated patterns.

        Uses negations from StructuredMemory.negations field.

        Args:
            results: List of recall results to filter.
            user_id: User ID for multi-tenancy.
            org_id: Optional organization ID filter.
            similarity_threshold: Threshold for semantic similarity filtering.

        Returns:
            Filtered list of results.
        """
        # Collect negation patterns from StructuredMemory.negations
        negated_patterns: set[str] = set()

        # Get structured memories with negations
        structured_with_negations = await self.storage.list_structured_memories(
            user_id, org_id, with_negations_only=True
        )
        for struct in structured_with_negations:
            for negation in struct.negations:
                negated_patterns.add(negation.pattern.lower())

        if not negated_patterns:
            return results

        # Prepare embedding-based filtering if threshold is set
        pattern_embeddings: dict[str, list[float]] = {}
        if similarity_threshold is not None:
            unique_patterns = list(negated_patterns)
            if unique_patterns:
                embeddings = await self.embedder.embed_batch(unique_patterns)
                pattern_embeddings = dict(zip(unique_patterns, embeddings, strict=True))

        filtered_results: list[RecallResult] = []
        pattern_filtered = 0
        semantic_filtered = 0

        for result in results:
            # Phase 1: Pattern-based filtering (word boundary match)
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
                logger.debug(f"Pattern-filtered {result.memory_type} {result.memory_id}")
                continue

            # Phase 2: Embedding-based filtering (semantic similarity)
            is_semantic_negated = False
            if similarity_threshold is not None and pattern_embeddings:
                result_embedding = await self._get_result_embedding(result, user_id)

                if result_embedding is not None:
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
                f"Negation filter: {pattern_filtered} pattern-based, "
                f"{semantic_filtered} semantic-based"
            )

        return filtered_results

    async def _get_result_embedding(
        self,
        result: RecallResult,
        user_id: str,
    ) -> list[float] | None:
        """Get embedding for a recall result from storage."""
        if result.memory_type == "episodic":
            ep = await self.storage.get_episode(result.memory_id, user_id)
            return ep.embedding if ep else None

        elif result.memory_type == "structured":
            struct = await self.storage.get_structured(result.memory_id, user_id)
            return struct.embedding if struct else None

        elif result.memory_type == "semantic":
            sem = await self.storage.get_semantic(result.memory_id, user_id)
            return sem.embedding if sem else None

        elif result.memory_type == "procedural":
            proc = await self.storage.get_procedural(result.memory_id, user_id)
            return proc.embedding if proc else None

        elif result.memory_type == "working":
            for ep in self._working_memory:
                if ep.id == result.memory_id:
                    return ep.embedding
            return None

        return None

    async def _follow_links(
        self,
        results: list[RecallResult],
        user_id: str,
        max_hops: int,
        limit: int,
    ) -> list[RecallResult]:
        """Follow related_ids links to discover connected memories."""
        all_results = list(results)
        seen_ids: set[str] = {r.memory_id for r in results}
        current_frontier = results

        for hop in range(1, max_hops + 1):
            related_ids_to_fetch: set[str] = set()
            for r in current_frontier:
                for related_id in r.related_ids:
                    if related_id not in seen_ids:
                        related_ids_to_fetch.add(related_id)

            if not related_ids_to_fetch:
                break

            next_frontier: list[RecallResult] = []
            for related_id in related_ids_to_fetch:
                if len(all_results) >= limit:
                    break

                memory_result = await self._fetch_memory_by_id(related_id, user_id, hop)
                if memory_result:
                    next_frontier.append(memory_result)
                    all_results.append(memory_result)
                    seen_ids.add(related_id)

            current_frontier = next_frontier

        return all_results[:limit]

    async def _fetch_memory_by_id(
        self,
        memory_id: str,
        user_id: str,
        hop_distance: int,
    ) -> RecallResult | None:
        """Fetch a memory by ID and convert to RecallResult."""
        if memory_id.startswith("sem_"):
            sem = await self.storage.get_semantic(memory_id, user_id)
            if sem:
                return RecallResult(
                    memory_type="semantic",
                    content=sem.content,
                    score=0.0,
                    confidence=sem.confidence.value,
                    memory_id=sem.id,
                    source_episode_ids=sem.source_episode_ids,
                    related_ids=sem.related_ids,
                    hop_distance=hop_distance,
                    staleness=Staleness.FRESH,
                    consolidated_at=sem.derived_at.isoformat(),
                    metadata={"linked": True},
                )

        elif memory_id.startswith("proc_"):
            proc = await self.storage.get_procedural(memory_id, user_id)
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
                    metadata={"linked": True},
                )

        elif memory_id.startswith("struct_"):
            struct = await self.storage.get_structured(memory_id, user_id)
            if struct:
                return RecallResult(
                    memory_type="structured",
                    content=struct.summary or struct.to_embedding_text(),
                    score=0.0,
                    confidence=struct.confidence.value,
                    memory_id=struct.id,
                    source_episode_id=struct.source_episode_id,
                    hop_distance=hop_distance,
                    staleness=Staleness.FRESH,
                    metadata={"linked": True},
                )

        return None

    async def _enrich_with_sources(
        self,
        results: list[RecallResult],
        user_id: str,
    ) -> list[RecallResult]:
        """Enrich recall results with source episode details."""
        enriched: list[RecallResult] = []

        for result in results:
            source_episodes: list[SourceEpisodeSummary] = []

            # Structured memories have a single source episode
            if result.memory_type == "structured" and result.source_episode_id:
                ep = await self.storage.get_episode(result.source_episode_id, user_id)
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
                sem = await self.storage.get_semantic(result.memory_id, user_id)
                if sem:
                    for ep_id in sem.source_episode_ids:
                        ep = await self.storage.get_episode(ep_id, user_id)
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
                proc = await self.storage.get_procedural(result.memory_id, user_id)
                if proc:
                    for ep_id in proc.source_episode_ids:
                        ep = await self.storage.get_episode(ep_id, user_id)
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

    async def _apply_retrieval_strengthening(
        self,
        results: list[RecallResult],
        user_id: str,
    ) -> None:
        """Apply Testing Effect: strengthen memories that were retrieved.

        Based on Roediger & Karpicke (2006): retrieval strengthens memory
        more than restudying. Retrieved memories get a small boost to their
        consolidation_strength.

        Only applies to:
        - SemanticMemory (has consolidation_strength)
        - ProceduralMemory (has consolidation_strength)
        - StructuredMemory (tracks retrieval_count)

        Episodic memories are immutable and not strengthened.

        Args:
            results: The recall results to strengthen.
            user_id: User ID for storage access.
        """
        delta = self.settings.retrieval_strengthening_delta

        for result in results:
            try:
                if result.memory_type == "semantic":
                    semantic_mem = await self.storage.get_semantic(result.memory_id, user_id)
                    if semantic_mem:
                        semantic_mem.strengthen(delta)
                        semantic_mem.record_access()
                        await self.storage.update_semantic_memory(semantic_mem)
                        logger.debug(
                            f"Strengthened semantic {result.memory_id}: "
                            f"strength={semantic_mem.consolidation_strength:.2f}"
                        )

                elif result.memory_type == "procedural":
                    procedural_mem = await self.storage.get_procedural(result.memory_id, user_id)
                    if procedural_mem:
                        procedural_mem.strengthen(delta)
                        procedural_mem.record_access()
                        await self.storage.update_procedural_memory(procedural_mem)
                        logger.debug(
                            f"Strengthened procedural {result.memory_id}: "
                            f"strength={procedural_mem.consolidation_strength:.2f}"
                        )

                elif result.memory_type == "structured":
                    structured_mem = await self.storage.get_structured(result.memory_id, user_id)
                    if structured_mem:
                        structured_mem.record_access()
                        await self.storage.update_structured_memory(structured_mem)
                        logger.debug(
                            f"Recorded access for structured {result.memory_id}: "
                            f"count={structured_mem.retrieval_count}"
                        )

                # Episodic memories are immutable - no strengthening
                # Working memories are volatile - no strengthening

            except Exception as e:
                logger.warning(f"Failed to strengthen {result.memory_type} {result.memory_id}: {e}")

    async def _apply_diversity_reranking(
        self,
        results: list[RecallResult],
        user_id: str,
        limit: int,
        diversity: float,
    ) -> list[RecallResult]:
        """Apply MMR diversity reranking to results.

        Uses Maximal Marginal Relevance to select results that balance
        relevance with diversity, preventing redundant/similar results.

        Args:
            results: List of recall results to rerank.
            user_id: User ID for fetching embeddings.
            limit: Maximum results to return.
            diversity: Diversity parameter (0.0-1.0).

        Returns:
            Reranked list of results with diverse selection.
        """
        if not results:
            return []

        # Fetch embeddings for each result
        candidates: list[tuple[float, list[float], int]] = []

        for i, result in enumerate(results):
            embedding = await self._get_result_embedding(result, user_id)
            if embedding:
                candidates.append((result.score, embedding, i))
            else:
                # If no embedding, use original index with zero embedding
                # This ensures we don't lose results that lack embeddings
                candidates.append((result.score, [0.0] * 1536, i))
                logger.debug(
                    f"No embedding for {result.memory_type} {result.memory_id}, "
                    "using placeholder for diversity"
                )

        # Apply MMR reranking
        selected_indices = mmr_rerank(candidates, limit=limit, diversity=diversity)

        # Return results in MMR order
        return [results[i] for i in selected_indices]


__all__ = ["RecallMixin"]
