"""Operations mixin for EngramService.

Provides consolidation, verification, and source tracing methods.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import TYPE_CHECKING, Any

from engram.models import Episode, HistoryEntry, ProvenanceChain, ProvenanceEvent, Staleness

from .helpers import cosine_similarity
from .models import RecallResult, SourceEpisodeSummary, VerificationResult

if TYPE_CHECKING:
    from engram.embeddings import Embedder
    from engram.storage import EngramStorage
    from engram.workflows.backend import WorkflowBackend
    from engram.workflows.consolidation import ConsolidationResult
    from engram.workflows.promotion import SynthesisResult


class OperationsMixin:
    """Mixin providing consolidation and verification functionality.

    Expects these attributes from the base class:
    - storage: EngramStorage
    - embedder: Embedder
    - workflow_backend: WorkflowBackend
    - _working_memory: list[Episode]
    """

    storage: EngramStorage
    embedder: Embedder
    workflow_backend: WorkflowBackend | None  # Always set after __post_init__
    _working_memory: list[Episode]

    async def get_sources(
        self,
        memory_id: str,
        user_id: str,
    ) -> list[Episode]:
        """Get source episodes for a derived memory.

        Traces a derived memory (StructuredMemory, SemanticMemory, ProceduralMemory)
        back to its source Episode(s).

        Args:
            memory_id: ID of the derived memory.
            user_id: User ID for multi-tenancy isolation.

        Returns:
            List of source Episodes in chronological order.

        Raises:
            ValueError: If memory type cannot be determined from ID prefix.
            KeyError: If memory not found.

        Example:
            ```python
            # Get a structured memory and trace it back to source
            memories = await engram.recall("email", user_id="u1")
            struct = next(m for m in memories if m.memory_type == "structured")
            sources = await engram.get_sources(struct.memory_id, user_id="u1")
            for ep in sources:
                print(f"{ep.timestamp}: {ep.content}")
            ```
        """
        # Determine memory type from ID prefix
        source_episode_ids: list[str] = []

        if memory_id.startswith("struct_"):
            structured = await self.storage.get_structured(memory_id, user_id)
            if structured is None:
                raise KeyError(f"StructuredMemory not found: {memory_id}")
            source_episode_ids = [structured.source_episode_id]

        elif memory_id.startswith("sem_"):
            semantic = await self.storage.get_semantic(memory_id, user_id)
            if semantic is None:
                raise KeyError(f"SemanticMemory not found: {memory_id}")
            source_episode_ids = semantic.source_episode_ids

        elif memory_id.startswith("proc_"):
            procedural = await self.storage.get_procedural(memory_id, user_id)
            if procedural is None:
                raise KeyError(f"ProceduralMemory not found: {memory_id}")
            source_episode_ids = procedural.source_episode_ids

        else:
            raise ValueError(
                f"Cannot determine memory type from ID: {memory_id}. "
                "Expected prefix: struct_, sem_, or proc_"
            )

        # Fetch source episodes
        episodes: list[Episode] = []
        for ep_id in source_episode_ids:
            ep = await self.storage.get_episode(ep_id, user_id)
            if ep is not None:
                episodes.append(ep)

        # Sort by timestamp (chronological order)
        episodes.sort(key=lambda e: e.timestamp)

        return episodes

    async def get_provenance(
        self,
        memory_id: str,
        user_id: str,
    ) -> ProvenanceChain:
        """Get complete provenance chain for a derived memory.

        Traces a derived memory back through its entire derivation chain,
        from source episodes through intermediate memories, recording
        how and when each derivation occurred.

        Args:
            memory_id: ID of the memory to trace.
            user_id: User ID for multi-tenancy isolation.

        Returns:
            ProvenanceChain with complete derivation history.

        Raises:
            ValueError: If memory type cannot be determined from ID prefix.
            KeyError: If memory not found.

        Example:
            ```python
            # Get full provenance for a semantic memory
            chain = await engram.get_provenance("sem_abc123", user_id="u1")
            print(f"Derived via: {chain.derivation_method}")
            print(f"Source episodes: {len(chain.source_episodes)}")
            for event in chain.timeline:
                print(f"  {event.timestamp}: {event.description}")
            ```
        """
        timeline: list[ProvenanceEvent] = []
        source_episodes: list[dict[str, Any]] = []
        intermediate_memories: list[dict[str, Any]] = []

        if memory_id.startswith("struct_"):
            structured = await self.storage.get_structured(memory_id, user_id)
            if structured is None:
                raise KeyError(f"StructuredMemory not found: {memory_id}")

            # Get source episode
            episode = await self.storage.get_episode(structured.source_episode_id, user_id)
            if episode:
                source_episodes.append(
                    {
                        "id": episode.id,
                        "content": episode.content,
                        "role": episode.role,
                        "timestamp": episode.timestamp.isoformat(),
                    }
                )
                timeline.append(
                    ProvenanceEvent(
                        timestamp=episode.timestamp,
                        event_type="stored",
                        description=f"Episode stored: {episode.id}",
                        memory_id=episode.id,
                    )
                )

            # Extraction event
            timeline.append(
                ProvenanceEvent(
                    timestamp=structured.derived_at,
                    event_type="extracted",
                    description=f"StructuredMemory extracted via {structured.derivation_method}",
                    memory_id=structured.id,
                    metadata={
                        "mode": structured.mode,
                        "enriched": structured.enriched,
                        "extract_counts": {
                            "emails": len(structured.emails),
                            "phones": len(structured.phones),
                            "urls": len(structured.urls),
                            "people": len(structured.people),
                            "preferences": len(structured.preferences),
                            "negations": len(structured.negations),
                        },
                    },
                )
            )

            return ProvenanceChain(
                memory_id=memory_id,
                memory_type="structured",
                derivation_method=structured.derivation_method,
                derivation_reasoning=None,
                derived_at=structured.derived_at,
                source_episodes=source_episodes,
                intermediate_memories=[],
                timeline=sorted(timeline, key=lambda e: e.timestamp),
            )

        elif memory_id.startswith("sem_"):
            semantic = await self.storage.get_semantic(memory_id, user_id)
            if semantic is None:
                raise KeyError(f"SemanticMemory not found: {memory_id}")

            # Get source episodes
            for ep_id in semantic.source_episode_ids:
                episode = await self.storage.get_episode(ep_id, user_id)
                if episode:
                    source_episodes.append(
                        {
                            "id": episode.id,
                            "content": episode.content,
                            "role": episode.role,
                            "timestamp": episode.timestamp.isoformat(),
                        }
                    )
                    timeline.append(
                        ProvenanceEvent(
                            timestamp=episode.timestamp,
                            event_type="stored",
                            description=f"Episode stored: {episode.id}",
                            memory_id=episode.id,
                        )
                    )

            # Check if there are intermediate structured memories
            for ep_id in semantic.source_episode_ids:
                struct = await self.storage.get_structured_for_episode(ep_id, user_id)
                if struct:
                    intermediate_memories.append(
                        {
                            "id": struct.id,
                            "type": "structured",
                            "summary": struct.summary,
                            "derivation_method": struct.derivation_method,
                            "derived_at": struct.derived_at.isoformat(),
                        }
                    )
                    timeline.append(
                        ProvenanceEvent(
                            timestamp=struct.derived_at,
                            event_type="extracted",
                            description=f"StructuredMemory extracted: {struct.id}",
                            memory_id=struct.id,
                        )
                    )

            # Consolidation event
            timeline.append(
                ProvenanceEvent(
                    timestamp=semantic.derived_at,
                    event_type="inferred",
                    description=f"SemanticMemory consolidated via {semantic.derivation_method}",
                    memory_id=semantic.id,
                    metadata={
                        "source_count": len(semantic.source_episode_ids),
                        "consolidation_strength": semantic.consolidation_strength,
                    },
                )
            )

            return ProvenanceChain(
                memory_id=memory_id,
                memory_type="semantic",
                derivation_method=semantic.derivation_method,
                derivation_reasoning=semantic.derivation_reasoning,
                derived_at=semantic.derived_at,
                source_episodes=source_episodes,
                intermediate_memories=intermediate_memories,
                timeline=sorted(timeline, key=lambda e: e.timestamp),
            )

        elif memory_id.startswith("proc_"):
            procedural = await self.storage.get_procedural(memory_id, user_id)
            if procedural is None:
                raise KeyError(f"ProceduralMemory not found: {memory_id}")

            # Get source episodes
            for ep_id in procedural.source_episode_ids:
                episode = await self.storage.get_episode(ep_id, user_id)
                if episode:
                    source_episodes.append(
                        {
                            "id": episode.id,
                            "content": episode.content,
                            "role": episode.role,
                            "timestamp": episode.timestamp.isoformat(),
                        }
                    )
                    timeline.append(
                        ProvenanceEvent(
                            timestamp=episode.timestamp,
                            event_type="stored",
                            description=f"Episode stored: {episode.id}",
                            memory_id=episode.id,
                        )
                    )

            # Get intermediate semantic memories
            for sem_id in procedural.source_semantic_ids:
                semantic = await self.storage.get_semantic(sem_id, user_id)
                if semantic:
                    intermediate_memories.append(
                        {
                            "id": semantic.id,
                            "type": "semantic",
                            "content": semantic.content,
                            "derivation_method": semantic.derivation_method,
                            "derived_at": semantic.derived_at.isoformat(),
                        }
                    )
                    timeline.append(
                        ProvenanceEvent(
                            timestamp=semantic.derived_at,
                            event_type="inferred",
                            description=f"SemanticMemory consolidated: {semantic.id}",
                            memory_id=semantic.id,
                        )
                    )

            # Synthesis event
            timeline.append(
                ProvenanceEvent(
                    timestamp=procedural.derived_at,
                    event_type="synthesized",
                    description=f"ProceduralMemory synthesized via {procedural.derivation_method}",
                    memory_id=procedural.id,
                    metadata={
                        "source_semantic_count": len(procedural.source_semantic_ids),
                        "consolidation_strength": procedural.consolidation_strength,
                    },
                )
            )

            return ProvenanceChain(
                memory_id=memory_id,
                memory_type="procedural",
                derivation_method=procedural.derivation_method,
                derivation_reasoning=procedural.derivation_reasoning,
                derived_at=procedural.derived_at,
                source_episodes=source_episodes,
                intermediate_memories=intermediate_memories,
                timeline=sorted(timeline, key=lambda e: e.timestamp),
            )

        else:
            raise ValueError(
                f"Cannot determine memory type from ID: {memory_id}. "
                "Expected prefix: struct_, sem_, or proc_"
            )

    async def verify(
        self,
        memory_id: str,
        user_id: str,
    ) -> VerificationResult:
        """Verify a memory against its source episodes.

        Traces a derived memory back to its source episode(s) and
        generates a human-readable explanation of how it was derived.

        Args:
            memory_id: ID of the memory to verify.
            user_id: User ID for multi-tenancy isolation.

        Returns:
            VerificationResult with source traceability and explanation.

        Raises:
            ValueError: If memory type cannot be determined from ID prefix.
            KeyError: If memory not found.

        Example:
            ```python
            # Verify a structured memory and see its derivation
            result = await engram.verify("struct_abc123", user_id="u1")
            print(result.explanation)
            # "Pattern-matched from source episode(s).
            #  Confidence: 0.90: extracted, 1 source, confirmed just now"

            for ep in result.source_episodes:
                print(f"Source: {ep['content']}")
            ```
        """
        from engram.models import ConfidenceScore, ExtractionMethod

        # Get memory and its metadata based on type
        content: str
        confidence: ConfidenceScore
        memory_type: str
        category: str = ""

        if memory_id.startswith("struct_"):
            memory_type = "structured"
            structured = await self.storage.get_structured(memory_id, user_id)
            if structured is None:
                raise KeyError(f"StructuredMemory not found: {memory_id}")
            content = structured.to_embedding_text() or structured.summary
            confidence = structured.confidence
            category = "structured" if structured.enriched else "regex-extracted"

        elif memory_id.startswith("sem_"):
            memory_type = "semantic"
            semantic = await self.storage.get_semantic(memory_id, user_id)
            if semantic is None:
                raise KeyError(f"SemanticMemory not found: {memory_id}")
            content = semantic.content
            confidence = semantic.confidence

        elif memory_id.startswith("proc_"):
            memory_type = "procedural"
            procedural = await self.storage.get_procedural(memory_id, user_id)
            if procedural is None:
                raise KeyError(f"ProceduralMemory not found: {memory_id}")
            content = procedural.content
            confidence = procedural.confidence

        else:
            raise ValueError(
                f"Cannot determine memory type from ID: {memory_id}. "
                "Expected prefix: struct_, sem_, or proc_"
            )

        # Get source episodes
        source_episodes = await self.get_sources(memory_id, user_id)

        # Build source episode summaries
        episode_details: list[dict[str, Any]] = []
        for ep in source_episodes:
            episode_details.append(
                {
                    "id": ep.id,
                    "content": ep.content,
                    "role": ep.role,
                    "timestamp": ep.timestamp.isoformat(),
                }
            )

        # Generate explanation
        explanation_parts: list[str] = []

        # Describe extraction
        if confidence.extraction_method == ExtractionMethod.EXTRACTED:
            if category:
                explanation_parts.append(f"Pattern-matched {category} from source episode(s).")
            else:
                explanation_parts.append("Pattern-matched from source episode(s).")
        elif confidence.extraction_method == ExtractionMethod.INFERRED:
            explanation_parts.append("LLM-inferred from source episode(s).")
        else:  # VERBATIM
            explanation_parts.append("Verbatim quote from source episode.")

        # Source info
        if source_episodes:
            if len(source_episodes) == 1:
                ep = source_episodes[0]
                explanation_parts.append(
                    f"Source: {ep.id} ({ep.timestamp.strftime('%Y-%m-%d %H:%M')})."
                )
            else:
                explanation_parts.append(f"Sources: {len(source_episodes)} episodes.")

        # Confidence explanation
        explanation_parts.append(f"Confidence: {confidence.explain()}")

        explanation = " ".join(explanation_parts)

        return VerificationResult(
            memory_id=memory_id,
            memory_type=memory_type,
            content=content,
            verified=len(source_episodes) > 0,
            source_episodes=episode_details,
            extraction_method=confidence.extraction_method.value,
            confidence=confidence.value,
            explanation=explanation,
        )

    async def consolidate(
        self,
        user_id: str,
        org_id: str | None = None,
    ) -> ConsolidationResult:
        """Consolidate unsummarized episodes into a semantic summary.

        Implements hierarchical compression: N episodes → 1 semantic memory.
        Uses map-reduce for large batches that exceed token limits.

        This method:
        1. Fetches ALL unsummarized episodes for the user
        2. Creates ONE coherent semantic summary via LLM
        3. Marks episodes as summarized with link to the summary
        4. Links new summary to similar existing memories

        Based on Complementary Learning Systems (McClelland et al., 1995):
        hippocampus (episodic) → neocortex (semantic) transfer with compression.

        Uses the configured workflow backend for execution, which may provide
        durability guarantees depending on the backend (DBOS, Temporal, Prefect).

        Args:
            user_id: User ID for multi-tenancy isolation.
            org_id: Optional organization ID for further isolation.

        Returns:
            ConsolidationResult with processing statistics:
            - episodes_processed: Number of episodes summarized
            - semantic_memories_created: Number of summaries created (typically 1)
            - links_created: Number of memory links built
            - compression_ratio: Ratio of input episodes to output memories

        Example:
            ```python
            # Consolidate all unsummarized episodes
            result = await engram.consolidate(user_id="u1")
            print(f"{result.episodes_processed} episodes → {result.semantic_memories_created} summary")
            print(f"Compression ratio: {result.compression_ratio:.1f}:1")
            ```
        """
        assert self.workflow_backend is not None, "workflow_backend not initialized"
        return await self.workflow_backend.run_consolidation(
            storage=self.storage,
            embedder=self.embedder,
            user_id=user_id,
            org_id=org_id,
        )

    async def create_procedural(
        self,
        user_id: str,
        org_id: str | None = None,
    ) -> SynthesisResult:
        """Synthesize a procedural memory from all semantic memories.

        Creates or updates ONE procedural memory that captures the user's
        behavioral patterns, preferences, and communication style.

        This method:
        1. Fetches ALL semantic memories for the user
        2. Uses LLM to synthesize a behavioral profile
        3. Creates or replaces the user's procedural memory
        4. Links procedural to all source semantic IDs

        Design decision: ONE procedural per user (replaces existing).

        Uses the configured workflow backend for execution, which may provide
        durability guarantees depending on the backend (DBOS, Temporal, Prefect).

        Args:
            user_id: User ID for multi-tenancy isolation.
            org_id: Optional organization ID for further isolation.

        Returns:
            SynthesisResult with processing statistics:
            - semantics_analyzed: Number of semantic memories analyzed
            - procedural_created: True if new procedural was created
            - procedural_updated: True if existing procedural was updated
            - procedural_id: ID of the created/updated procedural

        Example:
            ```python
            # Create/update the user's behavioral profile
            result = await engram.create_procedural(user_id="u1")
            if result.procedural_created:
                print(f"Created procedural: {result.procedural_id}")
            elif result.procedural_updated:
                print(f"Updated procedural: {result.procedural_id}")
            ```
        """
        assert self.workflow_backend is not None, "workflow_backend not initialized"
        return await self.workflow_backend.run_promotion(
            storage=self.storage,
            embedder=self.embedder,
            user_id=user_id,
            org_id=org_id,
        )

    async def _enrich_with_sources(
        self,
        results: list[RecallResult],
        user_id: str,
    ) -> list[RecallResult]:
        """Enrich recall results with source episode details.

        Args:
            results: List of recall results to enrich.
            user_id: User ID for isolation.

        Returns:
            List of RecallResult with source_episodes populated.
        """
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

    async def _follow_links(
        self,
        results: list[RecallResult],
        user_id: str,
        max_hops: int,
        limit: int,
    ) -> list[RecallResult]:
        """Follow related_ids links to discover connected memories.

        Implements multi-hop reasoning by traversing related_ids from
        semantic and procedural memories.

        Args:
            results: Initial recall results.
            user_id: User ID for isolation.
            max_hops: Maximum link traversal depth.
            limit: Maximum total results.

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
        """Fetch a memory by ID and convert to RecallResult.

        Args:
            memory_id: ID of the memory to fetch.
            user_id: User ID for isolation.
            hop_distance: How many hops from original query.

        Returns:
            RecallResult or None if not found.
        """
        # Determine memory type from ID prefix
        if memory_id.startswith("sem_"):
            sem = await self.storage.get_semantic(memory_id, user_id)
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
                    metadata={
                        "trigger_context": proc.trigger_context,
                        "linked": True,
                    },
                )

        elif memory_id.startswith("struct_"):
            struct = await self.storage.get_structured(memory_id, user_id)
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

    async def _apply_negation_filtering(
        self,
        results: list[RecallResult],
        user_id: str,
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
            org_id: Optional organization ID filter.
            similarity_threshold: Threshold for semantic similarity filtering.
                If None, only pattern-based filtering is used.

        Returns:
            Filtered list of results with negated items removed.
        """
        import logging

        logger = logging.getLogger(__name__)

        # Collect negation patterns from StructuredMemory.negations
        negated_patterns: set[str] = set()

        # Get from StructuredMemory.negations
        structured_memories = await self.storage.list_structured_memories(
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
                embeddings = await self.embedder.embed_batch(unique_patterns)
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
                    f"Pattern-filtered {result.memory_type} {result.memory_id}: "
                    "matches negated pattern"
                )
                continue

            # Phase 2: Embedding-based filtering (semantic similarity)
            is_semantic_negated = False
            if similarity_threshold is not None and pattern_embeddings:
                # Get embedding for this result's content
                result_embedding = await self._get_result_embedding(result, user_id)

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
                f"Negation filter: {pattern_filtered} pattern-based, "
                f"{semantic_filtered} semantic-based"
            )

        return filtered_results

    async def _get_result_embedding(
        self,
        result: RecallResult,
        user_id: str,
    ) -> list[float] | None:
        """Get embedding for a recall result from storage.

        Args:
            result: The recall result.
            user_id: User ID for multi-tenancy.

        Returns:
            Embedding vector or None if not found.
        """
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
            # Working memory is in-memory, find by ID
            for ep in self._working_memory:
                if ep.id == result.memory_id:
                    return ep.embedding
            return None

        return None

    async def get_memory_history(
        self,
        memory_id: str,
        user_id: str,
        since: datetime | None = None,
        limit: int = 100,
    ) -> list[HistoryEntry]:
        """Get change history for a specific memory.

        Returns all changes made to a memory over time, enabling full
        audit trail and debugging. Each entry shows before/after state,
        what triggered the change, and when.

        Args:
            memory_id: ID of the memory to get history for.
            user_id: User ID for multi-tenancy isolation.
            since: Optional timestamp to filter entries after.
            limit: Maximum entries to return (default 100).

        Returns:
            List of HistoryEntry sorted by timestamp (newest first).

        Example:
            ```python
            history = await engram.get_memory_history("sem_abc123", user_id="u1")
            for entry in history:
                print(f"{entry.timestamp}: {entry.change_type} ({entry.trigger})")
                if entry.diff:
                    print(f"  Changed: {list(entry.diff.keys())}")
            ```
        """
        return await self.storage.get_memory_history(
            memory_id=memory_id,
            user_id=user_id,
            since=since,
            limit=limit,
        )

    async def get_user_history(
        self,
        user_id: str,
        org_id: str | None = None,
        memory_type: str | None = None,
        change_type: str | None = None,
        since: datetime | None = None,
        limit: int = 100,
    ) -> list[HistoryEntry]:
        """Get change history for all memories of a user.

        Returns a timeline of all changes across all memory types,
        useful for understanding how the user's memory store evolved.

        Args:
            user_id: User to get history for.
            org_id: Optional organization filter.
            memory_type: Optional filter by type (structured, semantic, procedural).
            change_type: Optional filter by change type (created, updated, etc.).
            since: Optional timestamp to filter entries after.
            limit: Maximum entries to return (default 100).

        Returns:
            List of HistoryEntry sorted by timestamp (newest first).

        Example:
            ```python
            # Get all recent changes
            history = await engram.get_user_history(user_id="u1", limit=50)

            # Get only consolidation-triggered changes
            consolidations = await engram.get_user_history(
                user_id="u1",
                memory_type="semantic",
                change_type="strengthened",
            )
            ```
        """
        return await self.storage.get_user_history(
            user_id=user_id,
            org_id=org_id,
            memory_type=memory_type,
            change_type=change_type,
            since=since,
            limit=limit,
        )


__all__ = ["OperationsMixin"]
