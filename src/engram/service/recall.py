"""Recall mixin for EngramService.

Provides recall() and recall_at() methods for searching memories.
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from engram.models import AuditEntry, Episode, Staleness

from .helpers import cosine_similarity, mmr_rerank
from .models import RecallResult
from .query_expansion import get_combined_embedding
from .recall_utils import (
    apply_negation_filtering,
    enrich_with_sources,
    get_result_embedding,
)
from .recall_utils import (
    follow_links as traverse_memory_links,
)

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
        session_id: str | None = None,
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
        rerank: bool | None = None,
    ) -> list[RecallResult]:
        """Recall memories by semantic similarity.

        Searches across memory types and returns unified results
        sorted by similarity score, with optional context-aware reranking.

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
            session_id: Current session ID for session-aware reranking bonus.
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
            rerank: Enable context-aware reranking using weighted signals (similarity,
                recency, confidence, session match, access frequency). None uses
                settings.rerank_enabled (default True).

        Returns:
            List of RecallResult sorted by final score (reranked if enabled).
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
            results = await apply_negation_filtering(
                results=results,
                user_id=user_id,
                storage=self.storage,
                embedder=self.embedder,
                working_memory=self._working_memory,
                org_id=org_id,
                similarity_threshold=negation_similarity_threshold,
            )

        # Apply freshness filtering
        if freshness == "fresh_only":
            results = [r for r in results if r.staleness == Staleness.FRESH]

        # Apply context-aware reranking
        should_rerank = rerank if rerank is not None else self.settings.rerank_enabled
        if should_rerank and results:
            results = self._apply_context_reranking(results, session_id)
            # Re-sort after reranking
            results.sort(key=lambda r: r.score, reverse=True)

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
            final_results = await traverse_memory_links(
                results=final_results,
                user_id=user_id,
                max_hops=max_hops,
                limit=limit,
                storage=self.storage,
            )

        # Include source episodes if requested
        if include_sources:
            final_results = await enrich_with_sources(final_results, user_id, self.storage)

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
                        "session_id": ep.session_id,
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
                        "timestamp": struct.derived_at.isoformat(),
                        "derived_at": struct.derived_at.isoformat(),
                        "retrieval_count": struct.retrieval_count,
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

    def _apply_context_reranking(
        self,
        results: list[RecallResult],
        session_id: str | None = None,
    ) -> list[RecallResult]:
        """Apply context-aware reranking using weighted signals.

        Combines multiple signals to compute a final score:
        - Similarity: Original vector similarity score
        - Recency: Time decay (more recent = higher score)
        - Confidence: Memory confidence score
        - Session match: Bonus for memories from same session
        - Access boost: Bonus for frequently accessed memories

        Args:
            results: List of recall results to rerank.
            session_id: Current session ID for session match bonus.

        Returns:
            Results with updated scores based on weighted signals.
        """
        weights = self.settings.rerank_weights
        now = datetime.now(UTC)

        reranked: list[RecallResult] = []
        for result in results:
            # Signal 1: Similarity (already normalized 0-1)
            similarity_score = result.score

            # Signal 2: Recency (exponential decay)
            recency_score = self._calculate_recency_score(
                result, now, weights.recency_half_life_hours
            )

            # Signal 3: Confidence (normalized 0-1, default 0.5 for episodic)
            confidence_score = result.confidence if result.confidence is not None else 0.5

            # Signal 4: Session match (binary 0 or 1)
            session_score = self._calculate_session_score(result, session_id)

            # Signal 5: Access boost (logarithmic scaling)
            access_score = self._calculate_access_score(result, weights.max_access_boost)

            # Compute weighted final score
            final_score = (
                similarity_score * weights.similarity
                + recency_score * weights.recency
                + confidence_score * weights.confidence
                + session_score * weights.session
                + access_score * weights.access
            )

            # Clamp to [0, 1]
            final_score = max(0.0, min(1.0, final_score))

            # Create updated result with new score
            reranked.append(
                RecallResult(
                    memory_type=result.memory_type,
                    content=result.content,
                    score=final_score,
                    confidence=result.confidence,
                    memory_id=result.memory_id,
                    source_episode_id=result.source_episode_id,
                    source_episode_ids=result.source_episode_ids,
                    source_episodes=result.source_episodes,
                    related_ids=result.related_ids,
                    hop_distance=result.hop_distance,
                    staleness=result.staleness,
                    consolidated_at=result.consolidated_at,
                    metadata={
                        **result.metadata,
                        "original_similarity": similarity_score,
                        "rerank_signals": {
                            "similarity": similarity_score,
                            "recency": recency_score,
                            "confidence": confidence_score,
                            "session": session_score,
                            "access": access_score,
                        },
                    },
                )
            )

        return reranked

    def _calculate_recency_score(
        self,
        result: RecallResult,
        now: datetime,
        half_life_hours: float,
    ) -> float:
        """Calculate recency score using exponential decay.

        Args:
            result: The recall result.
            now: Current timestamp.
            half_life_hours: Hours for score to halve.

        Returns:
            Recency score between 0.0 and 1.0.
        """
        # Get timestamp from metadata
        timestamp_str = result.metadata.get("timestamp")
        if not timestamp_str:
            # For memories without timestamp, use moderate recency
            return 0.5

        try:
            # Parse ISO format timestamp
            timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=UTC)

            # Calculate age in hours
            age_hours = (now - timestamp).total_seconds() / 3600.0

            # Exponential decay: score = 0.5^(age / half_life)
            # This gives 1.0 for age=0, 0.5 for age=half_life, 0.25 for age=2*half_life, etc.
            decay_factor = math.pow(0.5, age_hours / half_life_hours)
            return max(0.0, min(1.0, decay_factor))

        except (ValueError, TypeError):
            return 0.5

    def _calculate_session_score(
        self,
        result: RecallResult,
        current_session_id: str | None,
    ) -> float:
        """Calculate session match score.

        Args:
            result: The recall result.
            current_session_id: Current session ID.

        Returns:
            1.0 if sessions match, 0.0 otherwise.
        """
        if current_session_id is None:
            return 0.0

        # Check metadata for session_id
        memory_session_id = result.metadata.get("session_id")
        if memory_session_id and memory_session_id == current_session_id:
            return 1.0

        return 0.0

    def _calculate_access_score(
        self,
        result: RecallResult,
        max_boost: float,
    ) -> float:
        """Calculate access frequency score using logarithmic scaling.

        Args:
            result: The recall result.
            max_boost: Maximum access boost.

        Returns:
            Access score between 0.0 and max_boost.
        """
        # Get retrieval count from metadata
        retrieval_count = result.metadata.get("retrieval_count", 0)
        if not isinstance(retrieval_count, int) or retrieval_count <= 0:
            return 0.0

        # Logarithmic scaling: log(1 + count) / log(1 + 100)
        # This gives a smooth curve that caps around 100 accesses
        # log(1) = 0, log(11) ≈ 2.4, log(101) ≈ 4.6
        normalized = math.log(1 + retrieval_count) / math.log(101)
        return min(max_boost, normalized * max_boost)

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
            embedding = await get_result_embedding(
                result, user_id, self.storage, self._working_memory
            )
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
