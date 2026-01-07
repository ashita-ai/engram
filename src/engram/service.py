"""Core Engram service layer.

This module provides the main EngramService that combines
storage, extraction, and embeddings into a simple encode/recall interface.

Example:
    ```python
    from engram.service import EngramService

    async with EngramService() as engram:
        # Store a memory
        result = await engram.encode(
            content="My email is user@example.com",
            role="user",
            user_id="user_123",
        )
        print(f"Stored episode {result.episode.id}")
        print(f"Extracted {len(result.facts)} facts")

        # Recall memories
        memories = await engram.recall(
            query="email address",
            user_id="user_123",
        )
        for memory in memories:
            print(f"{memory.content} (score: {memory.score:.2f})")
    ```
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from engram.config import Settings
from engram.embeddings import Embedder, get_embedder
from engram.extraction import ExtractionPipeline, default_pipeline
from engram.models import AuditEntry, Episode, Fact, Staleness
from engram.storage import EngramStorage


def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Returns a value clamped to [0.0, 1.0] to handle floating point precision.
    """
    dot_product = sum(a * b for a, b in zip(vec1, vec2, strict=True))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    # Clamp to handle floating point precision (e.g., 1.0000000000000002)
    return max(0.0, min(1.0, dot_product / (norm1 * norm2)))


class EncodeResult(BaseModel):
    """Result of encoding a memory.

    Attributes:
        episode: The stored episode.
        facts: List of facts extracted from the episode.
    """

    model_config = ConfigDict(extra="forbid")

    episode: Episode
    facts: list[Fact] = Field(default_factory=list)


class SourceEpisodeSummary(BaseModel):
    """Lightweight summary of a source episode."""

    model_config = ConfigDict(extra="forbid")

    id: str
    content: str
    role: str
    timestamp: str


class RecallResult(BaseModel):
    """A single recalled memory with similarity score.

    Attributes:
        memory_type: Type of memory (episode, fact, semantic, etc.).
        content: The memory content.
        score: Similarity score (0.0-1.0).
        confidence: Confidence score for facts/semantic memories.
        source_episode_id: Source episode ID for facts (single source).
        source_episodes: Source episode details (when include_sources=True).
        related_ids: IDs of related memories (for multi-hop).
        hop_distance: Distance from original query result (0=direct, 1=1-hop, etc.).
        staleness: Freshness state (fresh, consolidating, stale).
        consolidated_at: When this memory was last consolidated.
        metadata: Additional memory-specific metadata.
    """

    model_config = ConfigDict(extra="forbid")

    memory_type: str
    content: str
    score: float = Field(ge=0.0, le=1.0)
    confidence: float | None = None
    memory_id: str
    source_episode_id: str | None = None
    source_episodes: list[SourceEpisodeSummary] = Field(default_factory=list)
    related_ids: list[str] = Field(default_factory=list)
    hop_distance: int = Field(default=0, ge=0)
    staleness: Staleness = Field(default=Staleness.FRESH)
    consolidated_at: str | None = Field(default=None)
    metadata: dict[str, Any] = Field(default_factory=dict)


class VerificationResult(BaseModel):
    """Result of verifying a memory against its sources.

    Provides full traceability from a derived memory back to
    the source episode(s) it was extracted from.

    Attributes:
        memory_id: ID of the verified memory.
        memory_type: Type of memory (fact, semantic, etc.).
        content: The memory content.
        verified: Whether sources were found and content matches.
        source_episodes: Source episode contents.
        extraction_method: How the memory was extracted.
        confidence: Current confidence score.
        explanation: Human-readable derivation trace.
    """

    model_config = ConfigDict(extra="forbid")

    memory_id: str = Field(description="ID of the verified memory")
    memory_type: str = Field(description="Type: fact, semantic, procedural, negation")
    content: str = Field(description="The memory content")
    verified: bool = Field(description="True if sources found and traceable")
    source_episodes: list[dict[str, Any]] = Field(
        default_factory=list, description="Source episode details"
    )
    extraction_method: str = Field(description="How memory was extracted")
    confidence: float = Field(ge=0.0, le=1.0, description="Current confidence score")
    explanation: str = Field(description="Human-readable derivation trace")


@dataclass
class EngramService:
    """High-level Engram service for encoding and recalling memories.

    This service provides a simple interface for:
    - encode(): Store text as an episode and extract facts
    - recall(): Search memories by semantic similarity
    - get_working_memory(): Get current session's episodes
    - clear_working_memory(): Clear session context

    Uses dependency injection for storage, embeddings, and extraction,
    making it easy to test and configure.

    Attributes:
        storage: Storage backend (Qdrant).
        embedder: Embedding provider (OpenAI or FastEmbed).
        pipeline: Extraction pipeline for fact extraction.
        settings: Configuration settings.
    """

    storage: EngramStorage
    embedder: Embedder
    pipeline: ExtractionPipeline
    settings: Settings

    # Working memory: in-memory episodes for current session (not persisted separately)
    _working_memory: list[Episode] = field(default_factory=list, init=False, repr=False)

    @classmethod
    def create(cls, settings: Settings | None = None) -> EngramService:
        """Create an EngramService with default dependencies.

        Args:
            settings: Optional settings. Uses defaults if None.

        Returns:
            Configured EngramService instance.
        """
        if settings is None:
            settings = Settings()

        # Create embedder first to get dimensions
        embedder = get_embedder(settings)

        return cls(
            storage=EngramStorage(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key,
                prefix=settings.collection_prefix,
                embedding_dim=embedder.dimensions,
            ),
            embedder=embedder,
            pipeline=default_pipeline(),
            settings=settings,
        )

    async def initialize(self) -> None:
        """Initialize the service (storage collections, etc.)."""
        await self.storage.initialize()

    async def close(self) -> None:
        """Clean up resources and clear working memory."""
        self.clear_working_memory()
        await self.storage.close()

    async def __aenter__(self) -> EngramService:
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()

    def get_working_memory(self) -> list[Episode]:
        """Get current session's working memory.

        Working memory contains episodes from the current session.
        It's volatile (in-memory only) and cleared when the session ends.

        Returns:
            Copy of the working memory episodes list.

        Example:
            ```python
            async with EngramService.create() as engram:
                await engram.encode("Hello", role="user", user_id="u1")
                working = engram.get_working_memory()
                print(f"Session has {len(working)} episodes")
            # Working memory cleared on exit
            ```
        """
        return self._working_memory.copy()

    def clear_working_memory(self) -> None:
        """Clear working memory (typically at end of session).

        This removes all episodes from working memory without
        affecting persisted storage.
        """
        self._working_memory.clear()

    async def encode(
        self,
        content: str,
        role: str,
        user_id: str,
        org_id: str | None = None,
        session_id: str | None = None,
        importance: float = 0.5,
        run_extraction: bool = True,
    ) -> EncodeResult:
        """Encode content as an episode and optionally extract facts.

        This is the primary method for storing memories. It:
        1. Generates an embedding for the content
        2. Creates and stores an Episode
        3. Runs extraction pipeline to find facts (emails, phones, dates, etc.)
        4. Stores extracted facts with links to source episode

        Args:
            content: The text content to encode.
            role: Role of the speaker (user, assistant, system).
            user_id: User ID for multi-tenancy isolation.
            org_id: Optional organization ID.
            session_id: Optional session ID for grouping.
            importance: Importance score (0.0-1.0).
            run_extraction: Whether to run fact extraction.

        Returns:
            EncodeResult with the stored episode and extracted facts.

        Example:
            ```python
            result = await engram.encode(
                content="Call me at 555-123-4567",
                role="user",
                user_id="user_123",
            )
            # result.episode is the stored Episode
            # result.facts contains the extracted phone number
            ```
        """
        start_time = time.monotonic()

        # Generate embedding
        embedding = await self.embedder.embed(content)

        # Create episode
        episode = Episode(
            content=content,
            role=role,
            user_id=user_id,
            org_id=org_id,
            session_id=session_id,
            importance=importance,
            embedding=embedding,
        )

        # Store episode
        await self.storage.store_episode(episode)

        # Add to working memory for current session
        self._working_memory.append(episode)

        # Extract and store facts
        facts: list[Fact] = []
        if run_extraction:
            extracted_facts = self.pipeline.run(episode)
            if extracted_facts:
                # Batch embed all fact contents at once for efficiency
                fact_contents = [fact.content for fact in extracted_facts]
                fact_embeddings = await self.embedder.embed_batch(fact_contents)

                for fact, fact_embedding in zip(extracted_facts, fact_embeddings, strict=True):
                    fact.embedding = fact_embedding
                    await self.storage.store_fact(fact)
                    facts.append(fact)

        # Log audit entry
        duration_ms = int((time.monotonic() - start_time) * 1000)
        audit_entry = AuditEntry.for_encode(
            user_id=user_id,
            episode_id=episode.id,
            facts_count=len(facts),
            org_id=org_id,
            session_id=session_id,
            duration_ms=duration_ms,
        )
        await self.storage.log_audit(audit_entry)

        return EncodeResult(episode=episode, facts=facts)

    async def recall(
        self,
        query: str,
        user_id: str,
        org_id: str | None = None,
        limit: int = 10,
        min_confidence: float | None = None,
        min_selectivity: float = 0.0,
        include_episodes: bool = True,
        include_facts: bool = True,
        include_semantic: bool = True,
        include_procedural: bool = True,
        include_working: bool = True,
        include_sources: bool = False,
        follow_links: bool = False,
        max_hops: int = 2,
        freshness: str = "best_effort",
    ) -> list[RecallResult]:
        """Recall memories by semantic similarity.

        Searches across memory types and returns unified results
        sorted by similarity score.

        Args:
            query: Natural language query.
            user_id: User ID for multi-tenancy isolation.
            org_id: Optional organization ID filter.
            limit: Maximum results per memory type.
            min_confidence: Minimum confidence for facts.
            min_selectivity: Minimum selectivity for semantic memories (0.0-1.0).
            include_episodes: Whether to search episodes.
            include_facts: Whether to search facts.
            include_semantic: Whether to search semantic memories.
            include_procedural: Whether to search procedural memories.
            include_working: Whether to include working memory (current session).
            include_sources: Whether to include source episodes in results.
            follow_links: Enable multi-hop reasoning via related_ids.
            max_hops: Maximum link traversal depth when follow_links=True.
            freshness: Freshness mode - "best_effort" returns all, "fresh_only" only
                returns fully consolidated memories.

        Returns:
            List of RecallResult sorted by similarity score, with staleness metadata.

        Example:
            ```python
            memories = await engram.recall(
                query="phone numbers",
                user_id="user_123",
                limit=5,
                freshness="fresh_only",  # Only consolidated memories
            )
            for m in memories:
                print(f"{m.content} (staleness: {m.staleness})")
            ```
        """
        start_time = time.monotonic()

        # Generate query embedding
        query_vector = await self.embedder.embed(query)

        results: list[RecallResult] = []

        # Search episodes
        if include_episodes:
            scored_episodes = await self.storage.search_episodes(
                query_vector=query_vector,
                user_id=user_id,
                org_id=org_id,
                limit=limit,
            )
            for scored_ep in scored_episodes:
                ep = scored_ep.memory
                # Episode staleness: FRESH if consolidated, STALE otherwise
                ep_staleness = Staleness.FRESH if ep.consolidated else Staleness.STALE
                results.append(
                    RecallResult(
                        memory_type="episode",
                        content=ep.content,
                        score=scored_ep.score,
                        memory_id=ep.id,
                        staleness=ep_staleness,
                        metadata={
                            "role": ep.role,
                            "importance": ep.importance,
                            "timestamp": ep.timestamp.isoformat(),
                            "consolidated": ep.consolidated,
                        },
                    )
                )

        # Search facts
        if include_facts:
            scored_facts = await self.storage.search_facts(
                query_vector=query_vector,
                user_id=user_id,
                org_id=org_id,
                limit=limit,
                min_confidence=min_confidence,
            )
            for scored_fact in scored_facts:
                fact = scored_fact.memory
                # Facts are extracted immediately, so they're always fresh
                results.append(
                    RecallResult(
                        memory_type="fact",
                        content=fact.content,
                        score=scored_fact.score,
                        confidence=fact.confidence.value,
                        memory_id=fact.id,
                        source_episode_id=fact.source_episode_id,
                        staleness=Staleness.FRESH,
                        consolidated_at=fact.derived_at.isoformat(),
                        metadata={
                            "category": fact.category,
                            "derived_at": fact.derived_at.isoformat(),
                        },
                    )
                )

        # Search semantic memories
        if include_semantic:
            scored_semantics = await self.storage.search_semantic(
                query_vector=query_vector,
                user_id=user_id,
                org_id=org_id,
                limit=limit,
                min_confidence=min_confidence,
            )
            for scored_sem in scored_semantics:
                sem = scored_sem.memory
                # Filter by selectivity
                if sem.selectivity_score < min_selectivity:
                    continue
                # Semantic memories are created by consolidation, so they're fresh
                results.append(
                    RecallResult(
                        memory_type="semantic",
                        content=sem.content,
                        score=scored_sem.score,
                        confidence=sem.confidence.value,
                        memory_id=sem.id,
                        related_ids=sem.related_ids,
                        staleness=Staleness.FRESH,
                        consolidated_at=sem.derived_at.isoformat(),
                        metadata={
                            "selectivity": sem.selectivity_score,
                            "derived_at": sem.derived_at.isoformat(),
                            "consolidation_passes": sem.consolidation_passes,
                        },
                    )
                )

        # Search procedural memories
        if include_procedural:
            scored_procedurals = await self.storage.search_procedural(
                query_vector=query_vector,
                user_id=user_id,
                org_id=org_id,
                limit=limit,
                min_confidence=min_confidence,
            )
            for scored_proc in scored_procedurals:
                proc = scored_proc.memory
                # Procedural memories are created by consolidation, so they're fresh
                results.append(
                    RecallResult(
                        memory_type="procedural",
                        content=proc.content,
                        score=scored_proc.score,
                        confidence=proc.confidence.value,
                        memory_id=proc.id,
                        related_ids=proc.related_ids,
                        staleness=Staleness.FRESH,
                        consolidated_at=proc.derived_at.isoformat(),
                        metadata={
                            "trigger_context": proc.trigger_context,
                            "access_count": proc.access_count,
                            "derived_at": proc.derived_at.isoformat(),
                        },
                    )
                )

        # Search working memory (in-memory, no DB round-trip)
        if include_working and self._working_memory:
            # Filter by user_id (and org_id if specified)
            for ep in self._working_memory:
                if ep.user_id != user_id:
                    continue
                if org_id is not None and ep.org_id != org_id:
                    continue
                if ep.embedding is None:
                    continue

                # Compute similarity score
                score = _cosine_similarity(query_vector, ep.embedding)
                # Working memory is always stale (not yet consolidated)
                results.append(
                    RecallResult(
                        memory_type="working",
                        content=ep.content,
                        score=score,
                        memory_id=ep.id,
                        staleness=Staleness.STALE,
                        metadata={
                            "role": ep.role,
                            "importance": ep.importance,
                            "timestamp": ep.timestamp.isoformat(),
                        },
                    )
                )

        # Sort by score descending
        results.sort(key=lambda r: r.score, reverse=True)

        # Apply freshness filtering
        if freshness == "fresh_only":
            results = [r for r in results if r.staleness == Staleness.FRESH]

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

        # Log audit entry (hash query to avoid PII in logs)
        duration_ms = int((time.monotonic() - start_time) * 1000)
        query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]
        memory_types = list({r.memory_type for r in final_results})
        audit_entry = AuditEntry.for_recall(
            user_id=user_id,
            query_hash=query_hash,
            results_count=len(final_results),
            memory_types=memory_types,
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
        include_episodes: bool = True,
        include_facts: bool = True,
    ) -> list[RecallResult]:
        """Recall memories as they existed at a specific point in time.

        This is a bi-temporal query that only returns memories that were
        derived before the `as_of` timestamp. Useful for debugging,
        auditing, and understanding historical system state.

        Args:
            query: Natural language query.
            as_of: Point in time to query (only memories derived before this).
            user_id: User ID for multi-tenancy isolation.
            org_id: Optional organization ID filter.
            limit: Maximum results per memory type.
            min_confidence: Minimum confidence for facts.
            include_episodes: Whether to search episodes.
            include_facts: Whether to search facts.

        Returns:
            List of RecallResult sorted by similarity score.

        Example:
            ```python
            # What did we know about the user on June 1st?
            memories = await engram.recall_at(
                query="email address",
                as_of=datetime(2024, 6, 1),
                user_id="user_123",
            )
            ```
        """
        start_time = time.monotonic()

        # Generate query embedding
        query_vector = await self.embedder.embed(query)

        results: list[RecallResult] = []

        # Search episodes (filter by timestamp <= as_of)
        if include_episodes:
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
                        memory_type="episode",
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

        # Search facts (filter by derived_at <= as_of)
        if include_facts:
            scored_facts = await self.storage.search_facts(
                query_vector=query_vector,
                user_id=user_id,
                org_id=org_id,
                limit=limit,
                min_confidence=min_confidence,
                derived_at_before=as_of,
            )
            for scored_fact in scored_facts:
                fact = scored_fact.memory
                results.append(
                    RecallResult(
                        memory_type="fact",
                        content=fact.content,
                        score=scored_fact.score,
                        confidence=fact.confidence.value,
                        memory_id=fact.id,
                        source_episode_id=fact.source_episode_id,
                        metadata={
                            "category": fact.category,
                            "derived_at": fact.derived_at.isoformat(),
                            "event_at": fact.event_at.isoformat(),
                        },
                    )
                )

        # Sort by score descending
        results.sort(key=lambda r: r.score, reverse=True)

        final_results = results[:limit]

        # Log audit entry
        duration_ms = int((time.monotonic() - start_time) * 1000)
        query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]
        memory_types = list({r.memory_type for r in final_results})
        audit_entry = AuditEntry.for_recall(
            user_id=user_id,
            query_hash=query_hash,
            results_count=len(final_results),
            memory_types=memory_types,
            org_id=org_id,
            duration_ms=duration_ms,
        )
        await self.storage.log_audit(audit_entry)

        return final_results

    async def get_sources(
        self,
        memory_id: str,
        user_id: str,
    ) -> list[Episode]:
        """Get source episodes for a derived memory.

        Traces a derived memory (Fact, SemanticMemory, ProceduralMemory,
        NegationFact) back to its source Episode(s).

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
            # Get a fact and trace it back to source
            memories = await engram.recall("email", user_id="u1")
            fact = next(m for m in memories if m.memory_type == "fact")
            sources = await engram.get_sources(fact.memory_id, user_id="u1")
            for ep in sources:
                print(f"{ep.timestamp}: {ep.content}")
            ```
        """
        # Determine memory type from ID prefix
        source_episode_ids: list[str] = []

        if memory_id.startswith("fact_"):
            fact = await self.storage.get_fact(memory_id, user_id)
            if fact is None:
                raise KeyError(f"Fact not found: {memory_id}")
            source_episode_ids = [fact.source_episode_id]

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

        elif memory_id.startswith("neg_"):
            negation = await self.storage.get_negation(memory_id, user_id)
            if negation is None:
                raise KeyError(f"NegationFact not found: {memory_id}")
            source_episode_ids = negation.source_episode_ids

        else:
            raise ValueError(
                f"Cannot determine memory type from ID: {memory_id}. "
                "Expected prefix: fact_, sem_, proc_, or neg_"
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
            # Verify a fact and see its derivation
            result = await engram.verify("fact_abc123", user_id="u1")
            print(result.explanation)
            # "Extracted from episode ep_xyz on 2024-01-15.
            #  Pattern match: email
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

        if memory_id.startswith("fact_"):
            memory_type = "fact"
            fact = await self.storage.get_fact(memory_id, user_id)
            if fact is None:
                raise KeyError(f"Fact not found: {memory_id}")
            content = fact.content
            confidence = fact.confidence
            category = fact.category

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

        elif memory_id.startswith("neg_"):
            memory_type = "negation"
            negation = await self.storage.get_negation(memory_id, user_id)
            if negation is None:
                raise KeyError(f"NegationFact not found: {memory_id}")
            content = negation.content
            confidence = negation.confidence

        else:
            raise ValueError(
                f"Cannot determine memory type from ID: {memory_id}. "
                "Expected prefix: fact_, sem_, proc_, or neg_"
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

            # Facts have a single source episode
            if result.memory_type == "fact" and result.source_episode_id:
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

            # Create enriched result with sources
            enriched.append(
                RecallResult(
                    memory_type=result.memory_type,
                    content=result.content,
                    score=result.score,
                    confidence=result.confidence,
                    memory_id=result.memory_id,
                    source_episode_id=result.source_episode_id,
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
                    related_ids=sem.related_ids,
                    hop_distance=hop_distance,
                    staleness=Staleness.FRESH,
                    consolidated_at=sem.derived_at.isoformat(),
                    metadata={
                        "selectivity": sem.selectivity_score,
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
                    related_ids=proc.related_ids,
                    hop_distance=hop_distance,
                    staleness=Staleness.FRESH,
                    consolidated_at=proc.derived_at.isoformat(),
                    metadata={
                        "trigger_context": proc.trigger_context,
                        "linked": True,
                    },
                )

        elif memory_id.startswith("fact_"):
            fact = await self.storage.get_fact(memory_id, user_id)
            if fact:
                return RecallResult(
                    memory_type="fact",
                    content=fact.content,
                    score=0.0,
                    confidence=fact.confidence.value,
                    memory_id=fact.id,
                    source_episode_id=fact.source_episode_id,
                    hop_distance=hop_distance,
                    staleness=Staleness.FRESH,
                    consolidated_at=fact.derived_at.isoformat(),
                    metadata={
                        "category": fact.category,
                        "linked": True,
                    },
                )

        return None
