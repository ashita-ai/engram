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
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from engram.config import Settings
from engram.embeddings import Embedder, get_embedder
from engram.extraction import ExtractionPipeline, default_pipeline
from engram.models import AuditEntry, Episode, Fact
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


class RecallResult(BaseModel):
    """A single recalled memory with similarity score.

    Attributes:
        memory_type: Type of memory (episode, fact, semantic, etc.).
        content: The memory content.
        score: Similarity score (0.0-1.0).
        confidence: Confidence score for facts/semantic memories.
        metadata: Additional memory-specific metadata.
    """

    model_config = ConfigDict(extra="forbid")

    memory_type: str
    content: str
    score: float = Field(ge=0.0, le=1.0)
    confidence: float | None = None
    memory_id: str
    source_episode_id: str | None = None
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
    memory_type: str = Field(description="Type: fact, semantic, procedural, inhibitory")
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
        include_episodes: bool = True,
        include_facts: bool = True,
        include_working: bool = True,
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
            include_episodes: Whether to search episodes.
            include_facts: Whether to search facts.
            include_working: Whether to include working memory (current session).

        Returns:
            List of RecallResult sorted by similarity score.

        Example:
            ```python
            memories = await engram.recall(
                query="phone numbers",
                user_id="user_123",
                limit=5,
            )
            for m in memories:
                print(f"{m.content} (score: {m.score:.2f})")
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
                results.append(
                    RecallResult(
                        memory_type="working",
                        content=ep.content,
                        score=score,
                        memory_id=ep.id,
                        metadata={
                            "role": ep.role,
                            "importance": ep.importance,
                            "timestamp": ep.timestamp.isoformat(),
                        },
                    )
                )

        # Sort by score descending
        results.sort(key=lambda r: r.score, reverse=True)

        final_results = results[:limit]

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

    async def get_sources(
        self,
        memory_id: str,
        user_id: str,
    ) -> list[Episode]:
        """Get source episodes for a derived memory.

        Traces a derived memory (Fact, SemanticMemory, ProceduralMemory,
        InhibitoryFact) back to its source Episode(s).

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

        elif memory_id.startswith("inh_"):
            inhibitory = await self.storage.get_inhibitory(memory_id, user_id)
            if inhibitory is None:
                raise KeyError(f"InhibitoryFact not found: {memory_id}")
            source_episode_ids = inhibitory.source_episode_ids

        else:
            raise ValueError(
                f"Cannot determine memory type from ID: {memory_id}. "
                "Expected prefix: fact_, sem_, proc_, or inh_"
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

        elif memory_id.startswith("inh_"):
            memory_type = "inhibitory"
            inhibitory = await self.storage.get_inhibitory(memory_id, user_id)
            if inhibitory is None:
                raise KeyError(f"InhibitoryFact not found: {memory_id}")
            content = inhibitory.content
            confidence = inhibitory.confidence

        else:
            raise ValueError(
                f"Cannot determine memory type from ID: {memory_id}. "
                "Expected prefix: fact_, sem_, proc_, or inh_"
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
