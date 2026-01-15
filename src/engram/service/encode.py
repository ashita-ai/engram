"""Encode mixin for EngramService.

Provides the encode() method for storing memories.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any, Literal

from engram.models import AuditEntry, Episode, QuickExtracts, StructuredMemory

from .models import EncodeResult

if TYPE_CHECKING:
    from engram.embeddings import Embedder
    from engram.storage import EngramStorage

logger = logging.getLogger(__name__)


class EncodeMixin:
    """Mixin providing encode functionality.

    Expects these attributes from the base class:
    - storage: EngramStorage
    - embedder: Embedder
    - settings: Settings
    - _working_memory: list[Episode]
    """

    storage: EngramStorage
    embedder: Embedder
    settings: Any
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

        # Calculate importance based on extracts
        if importance is None:
            # Use structured extracts for importance calculation
            calculated_importance = self._calculate_importance_from_structured(
                content=content,
                role=role,
                structured=structured,
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
            # Synchronous enrichment - blocks until complete
            await self._enrich_structured_sync(episode, structured)
        elif enrich == "background":
            # Background enrichment - fire and forget
            asyncio.create_task(self._enrich_structured_background(episode, structured))

        # High-importance episodes trigger immediate consolidation
        if calculated_importance >= self.settings.high_importance_threshold:
            await self._trigger_consolidation(user_id=user_id, org_id=org_id)

        return EncodeResult(episode=episode, structured=structured)

    def _calculate_importance_from_structured(
        self,
        content: str,
        role: str,
        structured: StructuredMemory,
    ) -> float:
        """Calculate episode importance from structured extracts.

        Args:
            content: The episode content.
            role: Role of the speaker.
            structured: The structured memory with extracts.

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

        return max(0.0, min(1.0, score))

    async def _enrich_structured_sync(
        self,
        episode: Episode,
        structured: StructuredMemory,
    ) -> None:
        """Enrich a structured memory with LLM extraction (synchronous).

        Args:
            episode: The source episode.
            structured: The structured memory to enrich.
        """
        from engram.workflows.structure import run_structure

        try:
            result = await run_structure(
                episode=episode,
                storage=self.storage,
                embedder=self.embedder,
                skip_if_structured=False,  # Force re-enrichment
            )
            if result:
                logger.info(
                    f"Enriched {episode.id}: "
                    f"{result.deterministic_count} regex + {result.llm_count} LLM "
                    f"({result.processing_time_ms:.0f}ms)"
                )
        except Exception as e:
            logger.warning(f"Enrichment failed for {episode.id}: {e}")

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
        from engram.workflows.structure import run_structure_batch

        results = await run_structure_batch(
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

        Args:
            user_id: User ID for multi-tenancy.
            org_id: Optional organization ID.
        """
        from engram.workflows.consolidation import run_consolidation

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
            logger.warning(f"High-importance consolidation failed: {e}")


__all__ = ["EncodeMixin"]
