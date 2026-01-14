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
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from engram.config import Settings
from engram.embeddings import Embedder, get_embedder
from engram.extraction import ExtractionPipeline, NegationDetector, default_pipeline
from engram.models import AuditEntry, Episode, Fact, NegationFact, QuickExtracts, Staleness
from engram.storage import EngramStorage

if TYPE_CHECKING:
    from engram.workflows.consolidation import ConsolidationResult
    from engram.workflows.promotion import SynthesisResult


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


# Keywords that indicate important content worth remembering
_IMPORTANCE_KEYWORDS = frozenset(
    [
        "remember",
        "important",
        "don't forget",
        "always",
        "never",
        "critical",
        "key",
        "must",
        "essential",
        "priority",
        "urgent",
        "note that",
        "keep in mind",
        "fyi",
        "heads up",
    ]
)


def _calculate_importance(
    content: str,
    role: str,
    facts: list[Fact],
    negations: list[NegationFact],
    base_importance: float = 0.5,
) -> float:
    """Calculate episode importance based on content and extraction results.

    Uses heuristics to determine how important an episode is:
    - Extracted facts indicate concrete information worth remembering
    - Negations are corrections, which are very important
    - Certain keywords suggest the user wants something remembered
    - User messages are slightly more important than assistant responses

    Args:
        content: The episode content.
        role: Role of the speaker (user, assistant, system).
        facts: Facts extracted from the episode.
        negations: Negations detected from the episode.
        base_importance: Starting importance value (default 0.5).

    Returns:
        Importance score clamped to [0.0, 1.0].
    """
    score = base_importance

    # Facts indicate concrete info worth remembering
    # Each fact adds 0.05, capped at 0.15 (3+ facts)
    score += min(0.15, len(facts) * 0.05)

    # Negations are corrections - very important for accuracy
    # Each negation adds 0.1, capped at 0.2 (2+ negations)
    score += min(0.2, len(negations) * 0.1)

    # Check for importance keywords
    content_lower = content.lower()
    keyword_matches = sum(1 for kw in _IMPORTANCE_KEYWORDS if kw in content_lower)
    # Each keyword match adds 0.05, capped at 0.1 (2+ matches)
    score += min(0.1, keyword_matches * 0.05)

    # User messages are slightly more important than assistant responses
    # (user is providing info, assistant is often just responding)
    if role == "user":
        score += 0.05

    # System messages are usually setup/instructions, less important for recall
    if role == "system":
        score -= 0.1

    return max(0.0, min(1.0, score))


class EncodeResult(BaseModel):
    """Result of encoding a memory.

    Attributes:
        episode: The stored episode.
        facts: List of facts extracted from the episode.
        negations: List of negations detected from the episode.
    """

    model_config = ConfigDict(extra="forbid")

    episode: Episode
    facts: list[Fact] = Field(default_factory=list)
    negations: list[NegationFact] = Field(default_factory=list)


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
        memory_type: Type of memory (episodic, factual, semantic, etc.).
        content: The memory content.
        score: Similarity score (0.0-1.0).
        confidence: Confidence score for facts/semantic memories.
        source_episode_id: Source episode ID for facts (single source).
        source_episode_ids: Source episode IDs for memories with multiple sources.
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
    source_episode_ids: list[str] = Field(default_factory=list)
    source_episodes: list[SourceEpisodeSummary] = Field(default_factory=list)
    related_ids: list[str] = Field(default_factory=list)
    hop_distance: int = Field(default=0, ge=0)
    staleness: Staleness = Field(default=Staleness.FRESH)
    consolidated_at: str | None = Field(default=None)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("score", mode="before")
    @classmethod
    def clamp_score(cls, v: float) -> float:
        """Clamp score to [0, 1] to handle floating point precision errors."""
        return max(0.0, min(1.0, v))


class VerificationResult(BaseModel):
    """Result of verifying a memory against its sources.

    Provides full traceability from a derived memory back to
    the source episode(s) it was extracted from.

    Attributes:
        memory_id: ID of the verified memory.
        memory_type: Type of memory (factual, semantic, etc.).
        content: The memory content.
        verified: Whether sources were found and content matches.
        source_episodes: Source episode contents.
        extraction_method: How the memory was extracted.
        confidence: Current confidence score.
        explanation: Human-readable derivation trace.
    """

    model_config = ConfigDict(extra="forbid")

    memory_id: str = Field(description="ID of the verified memory")
    memory_type: str = Field(description="Type: factual, semantic, procedural, negation")
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
    negation_detector: NegationDetector = field(default_factory=NegationDetector)

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
        importance: float | None = None,
        run_extraction: bool = True,
        deduplicate_facts: bool = True,
        dedup_threshold: float = 0.95,
    ) -> EncodeResult:
        """Encode content as an episode and optionally extract facts.

        This is the primary method for storing memories. It:
        1. Generates an embedding for the content
        2. Creates and stores an Episode
        3. Runs extraction pipeline to find facts (emails, phones, dates, etc.)
        4. Stores extracted facts with links to source episode
        5. Calculates importance based on extracted content (if not provided)

        Args:
            content: The text content to encode.
            role: Role of the speaker (user, assistant, system).
            user_id: User ID for multi-tenancy isolation.
            org_id: Optional organization ID.
            session_id: Optional session ID for grouping.
            importance: Importance score (0.0-1.0). If None, automatically
                calculated based on content and extraction results.
            run_extraction: Whether to run fact extraction.
            deduplicate_facts: Whether to skip storing facts that are semantically
                similar to existing facts (default True).
            dedup_threshold: Similarity threshold for deduplication (default 0.95).
                Higher values require more exact matches.

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
            # result.episode.importance is auto-calculated
            ```
        """
        start_time = time.monotonic()

        # Generate embedding
        embedding = await self.embedder.embed(content)

        # Use provided importance or default to 0.5 (will be recalculated after extraction)
        initial_importance = importance if importance is not None else 0.5

        # Create episode first (we need it for extractors)
        episode = Episode(
            content=content,
            role=role,
            user_id=user_id,
            org_id=org_id,
            session_id=session_id,
            importance=initial_importance,
            embedding=embedding,
        )

        # Run quick deterministic extraction (emails, phones, URLs) for immediate access
        # Extractors return Facts with .content field
        from engram.extraction import EmailExtractor, PhoneExtractor, URLExtractor

        emails: list[str] = [f.content for f in EmailExtractor().extract(episode)]
        phones: list[str] = [f.content for f in PhoneExtractor().extract(episode)]
        urls: list[str] = [f.content for f in URLExtractor().extract(episode)]

        quick_extracts = QuickExtracts(emails=emails, phones=phones, urls=urls)
        episode.quick_extracts = quick_extracts

        # Store episode
        await self.storage.store_episode(episode)

        # Add to working memory for current session
        self._working_memory.append(episode)

        # Extract and store facts
        facts: list[Fact] = []
        dedup_skipped = 0
        if run_extraction:
            extracted_facts = self.pipeline.run(episode)
            if extracted_facts:
                # Batch embed all fact contents at once for efficiency
                fact_contents = [fact.content for fact in extracted_facts]
                fact_embeddings = await self.embedder.embed_batch(fact_contents)

                for fact, fact_embedding in zip(extracted_facts, fact_embeddings, strict=True):
                    fact.embedding = fact_embedding

                    # Semantic deduplication: check for existing similar facts
                    if deduplicate_facts:
                        is_duplicate = await self._is_duplicate_fact(
                            fact, fact_embedding, user_id, org_id, dedup_threshold
                        )
                        if is_duplicate:
                            dedup_skipped += 1
                            continue

                    await self.storage.store_fact(fact)
                    facts.append(fact)

            # Detect and store negations (separate from regular facts)
            negation_facts = self.negation_detector.detect(episode)
            if negation_facts:
                # Batch embed negation contents
                negation_contents = [neg.content for neg in negation_facts]
                negation_embeddings = await self.embedder.embed_batch(negation_contents)

                for negation, neg_embedding in zip(
                    negation_facts, negation_embeddings, strict=True
                ):
                    negation.embedding = neg_embedding
                    await self.storage.store_negation(negation)
            else:
                negation_facts = []
        else:
            negation_facts = []

        # Calculate importance from extraction results (if not explicitly provided)
        if importance is None:
            calculated_importance = _calculate_importance(
                content=content,
                role=role,
                facts=facts,
                negations=negation_facts,
                base_importance=0.5,
            )
            # Update episode with calculated importance
            episode.importance = calculated_importance
            await self.storage.update_episode(episode)
        else:
            calculated_importance = importance

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

        # High-importance episodes trigger immediate consolidation (A-MEM style)
        if calculated_importance >= self.settings.high_importance_threshold:
            await self._trigger_consolidation(user_id=user_id, org_id=org_id)

        return EncodeResult(episode=episode, facts=facts, negations=negation_facts)

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
    ) -> list[RecallResult]:
        """Recall memories by semantic similarity.

        Searches across memory types and returns unified results
        sorted by similarity score.

        Supports negation filtering: when enabled, memories that match negated
        patterns (e.g., "I don't use MongoDB") are filtered out to prevent
        returning outdated or contradicted information.

        Args:
            query: Natural language query.
            user_id: User ID for multi-tenancy isolation.
            org_id: Optional organization ID filter.
            limit: Maximum results per memory type.
            min_confidence: Minimum confidence for facts.
            min_selectivity: Minimum selectivity for semantic memories (0.0-1.0).
            memory_types: List of memory types to search. None means all types.
                Valid types: episodic, factual, semantic, procedural, negation, working.
            include_sources: Whether to include source episodes in results.
            follow_links: Enable multi-hop reasoning via related_ids.
            max_hops: Maximum link traversal depth when follow_links=True.
            freshness: Freshness mode - "best_effort" returns all, "fresh_only" only
                returns fully consolidated memories.
            apply_negation_filter: Filter out memories that match negated patterns (default True).
            include_system_prompts: Include system prompt episodes in results (default False).
                System prompts are operational metadata, not user content, so they're
                excluded by default to keep results focused on actual conversation.
            negation_similarity_threshold: Semantic similarity threshold for negation filtering.
                If set (default 0.75), uses embedding similarity to filter semantically related
                memories. Set to None to use only pattern-based (substring) filtering.

        Returns:
            List of RecallResult sorted by similarity score, with staleness metadata.

        Example:
            ```python
            memories = await engram.recall(
                query="phone numbers",
                user_id="user_123",
                limit=5,
                memory_types=["factual", "episodic"],
            )
            for m in memories:
                print(f"{m.content} (staleness: {m.staleness})")
            ```
        """
        start_time = time.monotonic()

        # Determine which memory types to search (cognitive science terms)
        all_types = {
            "episodic",
            "structured",
            "factual",
            "semantic",
            "procedural",
            "negation",
            "working",
        }
        types_to_search = set(memory_types) if memory_types is not None else all_types

        # Generate query embedding
        query_vector = await self.embedder.embed(query)

        # Use larger search limit when negation filtering is enabled
        # since filtering may remove some results
        search_limit = limit * 3 if apply_negation_filter else limit

        results: list[RecallResult] = []

        # Search episodes
        if "episodic" in types_to_search:
            scored_episodes = await self.storage.search_episodes(
                query_vector=query_vector,
                user_id=user_id,
                org_id=org_id,
                limit=search_limit,
            )
            for scored_ep in scored_episodes:
                ep = scored_ep.memory
                # Filter out system prompts unless explicitly requested
                # System prompts are operational metadata, not user content
                if ep.role == "system" and not include_system_prompts:
                    continue
                # Episode staleness: FRESH if summarized into semantic memory, STALE otherwise
                ep_staleness = Staleness.FRESH if ep.summarized else Staleness.STALE
                results.append(
                    RecallResult(
                        memory_type="episodic",
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

        # Search structured memories (per-episode LLM extraction)
        if "structured" in types_to_search:
            scored_structured = await self.storage.search_structured(
                query_vector=query_vector,
                user_id=user_id,
                org_id=org_id,
                limit=search_limit,
                min_confidence=min_confidence,
            )
            for scored_struct in scored_structured:
                struct = scored_struct.memory
                # Build metadata including extracted entities
                struct_metadata: dict[str, Any] = {
                    "source_episode_id": struct.source_episode_id,
                    "derived_at": struct.derived_at.isoformat(),
                    "keywords": struct.keywords,
                }
                if struct.emails:
                    struct_metadata["emails"] = struct.emails
                if struct.phones:
                    struct_metadata["phones"] = struct.phones
                if struct.people:
                    struct_metadata["people"] = [p.name for p in struct.people]
                if struct.organizations:
                    struct_metadata["organizations"] = struct.organizations
                if struct.preferences:
                    struct_metadata["preferences"] = [
                        {"topic": p.topic, "value": p.value} for p in struct.preferences
                    ]
                if struct.negations:
                    struct_metadata["negations"] = [n.content for n in struct.negations]

                results.append(
                    RecallResult(
                        memory_type="structured",
                        content=struct.summary,
                        score=scored_struct.score,
                        confidence=struct.confidence.value,
                        memory_id=struct.id,
                        source_episode_id=struct.source_episode_id,
                        source_episode_ids=[struct.source_episode_id],
                        staleness=Staleness.FRESH,
                        consolidated_at=struct.derived_at.isoformat(),
                        metadata=struct_metadata,
                    )
                )

        # Search facts
        if "factual" in types_to_search:
            scored_facts = await self.storage.search_facts(
                query_vector=query_vector,
                user_id=user_id,
                org_id=org_id,
                limit=search_limit,
                min_confidence=min_confidence,
            )
            for scored_fact in scored_facts:
                fact = scored_fact.memory
                # Facts are extracted immediately, so they're always fresh
                results.append(
                    RecallResult(
                        memory_type="factual",
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
        if "semantic" in types_to_search:
            scored_semantics = await self.storage.search_semantic(
                query_vector=query_vector,
                user_id=user_id,
                org_id=org_id,
                limit=search_limit,
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
                        source_episode_ids=sem.source_episode_ids,
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
        if "procedural" in types_to_search:
            scored_procedurals = await self.storage.search_procedural(
                query_vector=query_vector,
                user_id=user_id,
                org_id=org_id,
                limit=search_limit,
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
                        source_episode_ids=proc.source_episode_ids,
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

        # Search negation facts
        if "negation" in types_to_search:
            scored_negations = await self.storage.search_negation(
                query_vector=query_vector,
                user_id=user_id,
                org_id=org_id,
                limit=search_limit,
            )
            for scored_neg in scored_negations:
                neg = scored_neg.memory
                # Negation facts are created by consolidation, so they're fresh
                results.append(
                    RecallResult(
                        memory_type="negation",
                        content=neg.content,
                        score=scored_neg.score,
                        confidence=neg.confidence.value,
                        memory_id=neg.id,
                        source_episode_ids=neg.source_episode_ids,
                        staleness=Staleness.FRESH,
                        consolidated_at=neg.derived_at.isoformat(),
                        metadata={
                            "negates_pattern": neg.negates_pattern,
                            "derived_at": neg.derived_at.isoformat(),
                        },
                    )
                )

        # Search working memory (in-memory, no DB round-trip)
        if "working" in types_to_search and self._working_memory:
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

        # Apply negation filtering: remove memories that match negated patterns
        if apply_negation_filter:
            results = await self._apply_negation_filtering(
                results, user_id, org_id, negation_similarity_threshold
            )

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
        derived before the `as_of` timestamp. Useful for debugging,
        auditing, and understanding historical system state.

        Args:
            query: Natural language query.
            as_of: Point in time to query (only memories derived before this).
            user_id: User ID for multi-tenancy isolation.
            org_id: Optional organization ID filter.
            limit: Maximum results per memory type.
            min_confidence: Minimum confidence for facts.
            memory_types: List of memory types to search. None means all types.

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

        # Determine which memory types to search (recall_at only supports episodic and factual)
        all_types = {"episodic", "factual"}
        types_to_search = set(memory_types) & all_types if memory_types is not None else all_types

        # Generate query embedding
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

        # Search facts (filter by derived_at <= as_of)
        if "factual" in types_to_search:
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
                        memory_type="factual",
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
            fact = next(m for m in memories if m.memory_type == "factual")
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
            memory_type = "factual"
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
        from engram.workflows.consolidation import run_consolidation

        return await run_consolidation(
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
        from engram.workflows.promotion import run_synthesis

        return await run_synthesis(
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

            # Facts have a single source episode
            if result.memory_type == "factual" and result.source_episode_id:
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

            # Negation facts have multiple source episodes
            elif result.memory_type == "negation":
                neg = await self.storage.get_negation(result.memory_id, user_id)
                if neg:
                    for ep_id in neg.source_episode_ids:
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
                    memory_type="factual",
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

    async def _trigger_consolidation(
        self,
        user_id: str,
        org_id: str | None = None,
    ) -> None:
        """Trigger consolidation for high-importance episodes.

        This runs consolidation to extract semantic memories from
        recent unconsolidated episodes. Called automatically when
        an episode with importance >= high_importance_threshold is stored.

        Args:
            user_id: User ID for multi-tenancy.
            org_id: Optional organization ID.
        """
        import logging

        from engram.workflows.consolidation import run_consolidation

        logger = logging.getLogger(__name__)

        try:
            result = await run_consolidation(
                storage=self.storage,
                embedder=self.embedder,
                user_id=user_id,
                org_id=org_id,
            )
            if result.semantic_memories_created > 0:
                logger.info(
                    f"High-importance consolidation: {result.episodes_processed} episodes → "
                    f"{result.semantic_memories_created} summary ({result.compression_ratio:.1f}:1)"
                )
        except Exception as e:
            # Log but don't fail the encode operation
            logger.warning(f"High-importance consolidation failed: {e}")

    async def _is_duplicate_fact(
        self,
        fact: Fact,
        embedding: list[float],
        user_id: str,
        org_id: str | None,
        threshold: float,
    ) -> bool:
        """Check if a semantically similar fact already exists.

        Uses vector search to find existing facts with high similarity.
        Only considers facts in the same category to avoid false positives.

        Args:
            fact: The new fact to check.
            embedding: Pre-computed embedding for the fact.
            user_id: User ID for multi-tenancy.
            org_id: Optional organization ID.
            threshold: Similarity threshold for deduplication.

        Returns:
            True if a duplicate exists, False otherwise.
        """
        import logging

        logger = logging.getLogger(__name__)

        # Search for similar existing facts
        similar_facts = await self.storage.search_facts(
            query_vector=embedding,
            user_id=user_id,
            org_id=org_id,
            limit=3,  # Only need to find one match
        )

        for scored_fact in similar_facts:
            existing = scored_fact.memory
            # Must be same category (email vs email, phone vs phone)
            if existing.category != fact.category:
                continue

            # Check similarity threshold
            if scored_fact.score >= threshold:
                logger.debug(
                    f"Duplicate fact skipped: '{fact.content}' similar to "
                    f"'{existing.content}' (score: {scored_fact.score:.3f})"
                )
                return True

        return False

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

        # Collect negation patterns from two sources:
        # 1. Legacy NegationFact collection (deprecated)
        # 2. StructuredMemory.negations (preferred)
        negated_patterns: set[str] = set()

        # Get from legacy NegationFact collection
        legacy_negations = await self.storage.list_negation_facts(user_id, org_id)
        for neg in legacy_negations:
            negated_patterns.add(neg.negates_pattern.lower())

        # Get from StructuredMemory.negations (the new approach)
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
            # Don't filter negation facts themselves
            if result.memory_type == "negation":
                filtered_results.append(result)
                continue

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
                        similarity = _cosine_similarity(result_embedding, pattern_emb)
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

        elif result.memory_type == "factual":
            fact = await self.storage.get_fact(result.memory_id, user_id)
            return fact.embedding if fact else None

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
