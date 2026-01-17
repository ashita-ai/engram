"""Encode mixin for EngramService.

Provides the encode() method for storing memories.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Literal

from engram.models import AuditEntry, Episode, QuickExtracts, StructuredMemory

from .models import EncodeResult

if TYPE_CHECKING:
    from engram.embeddings import Embedder
    from engram.storage import EngramStorage
    from engram.workflows.backend import WorkflowBackend

logger = logging.getLogger(__name__)


class EncodeMixin:
    """Mixin providing encode functionality.

    Expects these attributes from the base class:
    - storage: EngramStorage
    - embedder: Embedder
    - settings: Settings
    - workflow_backend: WorkflowBackend
    - _working_memory: list[Episode]
    """

    storage: EngramStorage
    embedder: Embedder
    settings: Any
    workflow_backend: WorkflowBackend | None  # Always set after __post_init__
    _working_memory: list[Episode]

    async def encode(
        self,
        content: str,
        role: str,
        user_id: str,
        org_id: str | None = None,
        session_id: str | None = None,
        importance: float | None = None,
        enrich: bool | Literal["background"] = False,
    ) -> EncodeResult:
        """Encode content as an episode with structured extraction.

        This is the primary method for storing memories. It:
        1. Generates an embedding for the content
        2. Creates and stores an Episode (immutable ground truth)
        3. Creates a StructuredMemory (fast mode by default)
        4. Optionally enriches with LLM extraction (enrich=True or "background")

        Args:
            content: The text content to encode.
            role: Role of the speaker (user, assistant, system).
            user_id: User ID for multi-tenancy isolation.
            org_id: Optional organization ID.
            session_id: Optional session ID for grouping.
            importance: Importance score (0.0-1.0). If None, automatically
                calculated based on content analysis.
            enrich: LLM enrichment mode:
                - False (default): Fast mode, regex only, no LLM
                - True: Rich mode, regex + LLM (blocks until complete)
                - "background": Rich mode, LLM runs in background task

        Returns:
            EncodeResult with the stored episode and structured memory.

        Example:
            ```python
            # Fast encode (default) - regex only, immediate
            result = await engram.encode(
                content="My email is user@example.com",
                role="user",
                user_id="user_123",
            )

            # Rich encode - regex + LLM extraction (blocks)
            result = await engram.encode(
                content="I'm Alex, prefer PostgreSQL, work with Sarah",
                role="user",
                user_id="user_123",
                enrich=True,
            )

            # Background enrichment - returns fast, LLM runs async
            result = await engram.encode(
                content="Meeting with John next Tuesday about the API",
                role="user",
                user_id="user_123",
                enrich="background",
            )
            ```
        """
        start_time = time.monotonic()

        # Generate embedding
        embedding = await self.embedder.embed(content)

        # Use provided importance or default to 0.5
        initial_importance = importance if importance is not None else 0.5

        # Create episode (immutable ground truth)
        episode = Episode(
            content=content,
            role=role,
            user_id=user_id,
            org_id=org_id,
            session_id=session_id,
            importance=initial_importance,
            embedding=embedding,
        )

        # Run quick deterministic extraction (emails, phones, URLs)
        from engram.extraction import EmailExtractor, PhoneExtractor, URLExtractor

        emails = EmailExtractor().extract(episode)
        phones = PhoneExtractor().extract(episode)
        urls = URLExtractor().extract(episode)

        # Store quick extracts on episode for fast access
        quick_extracts = QuickExtracts(emails=emails, phones=phones, urls=urls)
        episode.quick_extracts = quick_extracts

        # Store episode
        await self.storage.store_episode(episode)

        # Add to working memory for current session
        self._working_memory.append(episode)

        # Create StructuredMemory (always, fast mode by default)
        structured = StructuredMemory.from_episode_fast(
            source_episode_id=episode.id,
            user_id=user_id,
            org_id=org_id,
            emails=emails,
            phones=phones,
            urls=urls,
            embedding=embedding,  # Reuse episode embedding for fast mode
        )

        # Store structured memory
        await self.storage.store_structured(structured)

        # Link episode to structured
        episode.structured = True
        episode.structured_into = structured.id
        await self.storage.update_episode(episode)

        # Calculate importance based on extracts + surprise
        if importance is None:
            # Use structured extracts and surprise for importance calculation
            calculated_importance = await self._calculate_importance_with_surprise(
                content=content,
                role=role,
                structured=structured,
                embedding=embedding,
                user_id=user_id,
                org_id=org_id,
            )
            episode.importance = calculated_importance
            await self.storage.update_episode(episode)
        else:
            calculated_importance = importance

        # Log audit entry
        duration_ms = int((time.monotonic() - start_time) * 1000)
        audit_entry = AuditEntry.for_encode(
            user_id=user_id,
            episode_id=episode.id,
            facts_count=len(emails) + len(phones) + len(urls),
            org_id=org_id,
            session_id=session_id,
            duration_ms=duration_ms,
        )
        await self.storage.log_audit(audit_entry)

        # Handle LLM enrichment
        if enrich is True:
            # Synchronous enrichment - blocks until complete, returns enriched version
            structured = await self._enrich_structured_sync(episode, structured)
        elif enrich == "background":
            # Durable background enrichment - uses DBOS/Temporal/Prefect
            # If process crashes, enrichment will be retried automatically
            from engram.workflows.structure import schedule_durable_enrichment

            await schedule_durable_enrichment(
                episode_id=episode.id,
                user_id=user_id,
                org_id=org_id,
            )

        # High-importance episodes trigger immediate consolidation
        if calculated_importance >= self.settings.high_importance_threshold:
            await self._trigger_consolidation(user_id=user_id, org_id=org_id)

        return EncodeResult(episode=episode, structured=structured)

    async def _calculate_importance_with_surprise(
        self,
        content: str,
        role: str,
        structured: StructuredMemory,
        embedding: list[float],
        user_id: str,
        org_id: str | None = None,
    ) -> float:
        """Calculate episode importance including surprise/novelty factor.

        Uses the Adaptive Compression framework (Nagy et al. 2025):
        - Novel/surprising content gets higher importance
        - Redundant/predictable content stays lower priority
        - Surprise is computed as 1 - max_similarity to existing memories

        Args:
            content: The episode content.
            role: Role of the speaker.
            structured: The structured memory with extracts.
            embedding: The episode's embedding vector.
            user_id: User ID for memory search.
            org_id: Optional organization ID.

        Returns:
            Importance score clamped to [0.0, 1.0].
        """
        from .helpers import IMPORTANCE_KEYWORDS

        score = 0.5  # Base importance

        # Regex extracts indicate concrete info
        regex_count = len(structured.emails) + len(structured.phones) + len(structured.urls)
        score += min(0.15, regex_count * 0.05)

        # LLM extracts (if enriched)
        if structured.enriched:
            llm_count = (
                len(structured.people) + len(structured.preferences) + len(structured.negations)
            )
            score += min(0.15, llm_count * 0.05)

            # Negations are very important
            score += min(0.2, len(structured.negations) * 0.1)

        # Check for importance keywords
        content_lower = content.lower()
        keyword_matches = sum(1 for kw in IMPORTANCE_KEYWORDS if kw in content_lower)
        score += min(0.1, keyword_matches * 0.05)

        # Role adjustments
        if role == "user":
            score += 0.05
        elif role == "system":
            score -= 0.1

        # Surprise factor (Adaptive Compression)
        if self.settings.surprise_scoring_enabled:
            surprise_score = await self._calculate_surprise(
                embedding=embedding,
                user_id=user_id,
                org_id=org_id,
            )
            score += surprise_score * self.settings.surprise_weight
            logger.debug(
                f"Surprise score: {surprise_score:.2f} (weighted: {surprise_score * self.settings.surprise_weight:.3f})"
            )

        return max(0.0, min(1.0, score))

    async def _calculate_surprise(
        self,
        embedding: list[float],
        user_id: str,
        org_id: str | None = None,
    ) -> float:
        """Calculate surprise/novelty score for new content.

        Based on Adaptive Compression (Nagy et al. 2025):
        - High similarity to existing memories = low surprise (predictable)
        - Low similarity to existing memories = high surprise (novel)
        - No existing memories = moderate surprise (cold start)

        Args:
            embedding: The new content's embedding vector.
            user_id: User ID for memory search.
            org_id: Optional organization ID.

        Returns:
            Surprise score between 0.0 (predictable) and 1.0 (highly novel).
        """
        try:
            # Search for similar existing memories (episodic)
            similar = await self.storage.search_episodes(
                query_vector=embedding,
                user_id=user_id,
                org_id=org_id,
                limit=self.settings.surprise_search_limit,
            )

            if similar:
                # High similarity = low surprise
                max_similarity = max(r.score for r in similar)
                surprise = 1.0 - max_similarity
                return max(0.0, min(1.0, surprise))
            else:
                # Cold start: no existing memories = moderate surprise
                # Not maximum (1.0) since we don't know if this is truly novel
                return 0.5

        except Exception as e:
            logger.warning(f"Surprise calculation failed: {e}")
            # On error, return neutral surprise (don't affect importance)
            return 0.0

    async def _enrich_structured_sync(
        self,
        episode: Episode,
        structured: StructuredMemory,
    ) -> StructuredMemory:
        """Enrich a structured memory with LLM extraction (synchronous).

        Uses the configured workflow backend for execution.

        Args:
            episode: The source episode.
            structured: The structured memory to enrich.

        Returns:
            The enriched StructuredMemory (or original if enrichment fails).
        """
        assert self.workflow_backend is not None, "workflow_backend not initialized"
        try:
            result = await self.workflow_backend.run_structure(
                episode=episode,
                storage=self.storage,
                embedder=self.embedder,
                skip_if_structured=False,  # Force re-enrichment
            )
            if result and result.structured:
                logger.info(
                    f"Enriched {episode.id}: "
                    f"{result.deterministic_count} regex + {result.llm_count} LLM "
                    f"({result.processing_time_ms:.0f}ms)"
                )
                enriched: StructuredMemory = result.structured
                return enriched
        except Exception as e:
            logger.warning(f"Enrichment failed for {episode.id}: {e}")
        return structured

    async def _enrich_structured_background(
        self,
        episode: Episode,
        structured: StructuredMemory,
    ) -> None:
        """Enrich a structured memory with LLM extraction (background task).

        Args:
            episode: The source episode.
            structured: The structured memory to enrich.
        """
        try:
            await self._enrich_structured_sync(episode, structured)
        except Exception as e:
            logger.error(f"Background enrichment failed for {episode.id}: {e}")

    async def enrich_all(
        self,
        user_id: str,
        org_id: str | None = None,
        limit: int | None = None,
    ) -> int:
        """Enrich all fast-mode structured memories with LLM extraction.

        Processes all structured memories that haven't been enriched yet.
        Call this at end of session or as a batch job.

        Uses the configured workflow backend for execution, which may provide
        durability guarantees depending on the backend (DBOS, Temporal, Prefect).

        Args:
            user_id: User ID for multi-tenancy.
            org_id: Optional organization ID filter.
            limit: Maximum memories to process (None = all).

        Returns:
            Number of memories enriched.

        Example:
            ```python
            # At end of session, enrich all fast-mode memories
            count = await engram.enrich_all(user_id="user_123")
            print(f"Enriched {count} memories")
            ```
        """
        assert self.workflow_backend is not None, "workflow_backend not initialized"
        results = await self.workflow_backend.run_structure_batch(
            storage=self.storage,
            embedder=self.embedder,
            user_id=user_id,
            org_id=org_id,
            limit=limit,
        )

        if results:
            total_extracts = sum(r.extracts_count for r in results)
            logger.info(
                f"Batch enrichment: {len(results)} memories, {total_extracts} total extracts"
            )

        return len(results)

    async def _trigger_consolidation(
        self,
        user_id: str,
        org_id: str | None = None,
    ) -> None:
        """Trigger consolidation for high-importance episodes.

        Uses the configured workflow backend for execution.

        Args:
            user_id: User ID for multi-tenancy.
            org_id: Optional organization ID.
        """
        assert self.workflow_backend is not None, "workflow_backend not initialized"
        try:
            result = await self.workflow_backend.run_consolidation(
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
            logger.warning(f"High-importance consolidation failed: {e}")


__all__ = ["EncodeMixin"]
