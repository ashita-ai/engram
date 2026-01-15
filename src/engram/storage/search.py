"""Search operations for Engram storage.

Provides vector similarity search methods for all memory types.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from qdrant_client import models

if TYPE_CHECKING:
    from collections.abc import Sequence

    from engram.models import (
        Episode,
        ProceduralMemory,
        SemanticMemory,
        StructuredMemory,
    )

MemoryT = TypeVar("MemoryT")


@dataclass
class ScoredResult(Generic[MemoryT]):
    """A search result with similarity score.

    Attributes:
        memory: The matched memory object.
        score: Similarity score from vector search (0.0-1.0).
    """

    memory: MemoryT
    score: float


class SearchMixin:
    """Mixin providing search operations for EngramStorage.

    This mixin expects the following attributes/methods from the base class:
    - _build_filters(user_id, org_id, min_confidence) -> list[FieldCondition]
    - _search(collection, query_vector, filters, limit) -> list[ScoredPoint]
    - _payload_to_memory(payload, memory_class) -> MemoryT
    - _collection_name(memory_type) -> str
    - client: AsyncQdrantClient
    """

    # These will be provided by the base class
    _payload_to_memory: Any
    _collection_name: Any
    client: Any

    async def search_episodes(
        self,
        query_vector: Sequence[float],
        user_id: str,
        org_id: str | None = None,
        limit: int = 10,
        min_importance: float | None = None,
        timestamp_before: datetime | None = None,
    ) -> list[ScoredResult[Episode]]:
        """Search for similar episodes.

        Args:
            query_vector: Query embedding vector.
            user_id: User ID for isolation.
            org_id: Optional org ID filter.
            limit: Maximum results to return.
            min_importance: Minimum importance threshold.
            timestamp_before: Only include episodes from before this time.

        Returns:
            List of ScoredResult[Episode] sorted by similarity.
        """
        from engram.models import Episode

        filters = self._build_filters(user_id, org_id, timestamp_before=timestamp_before)
        if min_importance is not None:
            filters.append(
                models.FieldCondition(
                    key="importance",
                    range=models.Range(gte=min_importance),
                )
            )

        results = await self._search(
            collection="episodic",
            query_vector=query_vector,
            filters=filters,
            limit=limit,
        )

        return [
            ScoredResult(
                memory=self._payload_to_memory(r.payload, Episode),
                score=r.score,
            )
            for r in results
            if r.payload is not None
        ]

    async def search_structured(
        self,
        query_vector: Sequence[float],
        user_id: str,
        org_id: str | None = None,
        limit: int = 10,
        min_confidence: float | None = None,
        source_episode_id: str | None = None,
        derived_at_before: datetime | None = None,
    ) -> list[ScoredResult[StructuredMemory]]:
        """Search for similar structured memories.

        Args:
            query_vector: Query embedding vector.
            user_id: User ID for isolation.
            org_id: Optional org ID filter.
            limit: Maximum results to return.
            min_confidence: Minimum confidence threshold.
            source_episode_id: Filter by source episode.
            derived_at_before: Only include memories derived before this time.

        Returns:
            List of ScoredResult[StructuredMemory] sorted by similarity.
        """
        from engram.models import StructuredMemory

        filters = self._build_filters(
            user_id, org_id, min_confidence, derived_at_before=derived_at_before
        )
        if source_episode_id is not None:
            filters.append(
                models.FieldCondition(
                    key="source_episode_id",
                    match=models.MatchValue(value=source_episode_id),
                )
            )

        results = await self._search(
            collection="structured",
            query_vector=query_vector,
            filters=filters,
            limit=limit,
        )

        return [
            ScoredResult(
                memory=self._payload_to_memory(r.payload, StructuredMemory),
                score=r.score,
            )
            for r in results
            if r.payload is not None
        ]

    async def search_semantic(
        self,
        query_vector: Sequence[float],
        user_id: str,
        org_id: str | None = None,
        limit: int = 10,
        min_confidence: float | None = None,
        min_selectivity: float = 0.0,
        derived_at_before: datetime | None = None,
        track_activation: bool = False,
    ) -> list[ScoredResult[SemanticMemory]]:
        """Search for similar semantic memories.

        Args:
            query_vector: Query embedding vector.
            user_id: User ID for isolation.
            org_id: Optional org ID filter.
            limit: Maximum results to return.
            min_confidence: Minimum confidence threshold.
            min_selectivity: Minimum selectivity score (0.0-1.0).
            derived_at_before: Only include memories derived before this time.
            track_activation: If True, update retrieval_count and last_accessed
                for returned memories (A-MEM style activation tracking).

        Returns:
            List of ScoredResult[SemanticMemory] sorted by similarity.
        """
        from engram.models import SemanticMemory

        filters = self._build_filters(
            user_id, org_id, min_confidence, derived_at_before=derived_at_before
        )

        # Fetch extra results to account for selectivity filtering
        fetch_limit = limit if min_selectivity <= 0.0 else limit * 2

        results = await self._search(
            collection="semantic",
            query_vector=query_vector,
            filters=filters,
            limit=fetch_limit,
        )

        scored_results = [
            ScoredResult(
                memory=self._payload_to_memory(r.payload, SemanticMemory),
                score=r.score,
            )
            for r in results
            if r.payload is not None
        ]

        # Filter by selectivity if specified
        if min_selectivity > 0.0:
            scored_results = [
                sr for sr in scored_results if sr.memory.selectivity_score >= min_selectivity
            ][:limit]

        # A-MEM style activation tracking
        if track_activation and scored_results:
            await self._track_semantic_activation(scored_results)

        return scored_results

    async def _track_semantic_activation(
        self,
        results: list[ScoredResult[SemanticMemory]],
    ) -> None:
        """Update activation metadata for searched memories.

        Records retrieval_count and last_accessed for A-MEM style
        activation-based strengthening.

        Args:
            results: Search results to track activation for.
        """
        from datetime import UTC, datetime

        collection = self._collection_name("semantic")

        for scored in results:
            memory = scored.memory
            memory.retrieval_count += 1
            memory.last_accessed = datetime.now(UTC)

            # Update just the activation fields in storage
            await self.client.set_payload(
                collection_name=collection,
                payload={
                    "retrieval_count": memory.retrieval_count,
                    "last_accessed": memory.last_accessed.isoformat(),
                },
                points=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="id",
                                match=models.MatchValue(value=memory.id),
                            ),
                        ]
                    )
                ),
            )

    async def search_procedural(
        self,
        query_vector: Sequence[float],
        user_id: str,
        org_id: str | None = None,
        limit: int = 10,
        min_confidence: float | None = None,
        track_activation: bool = False,
    ) -> list[ScoredResult[ProceduralMemory]]:
        """Search for similar procedural memories.

        Args:
            query_vector: Query embedding vector.
            user_id: User ID for isolation.
            org_id: Optional org ID filter.
            limit: Maximum results to return.
            min_confidence: Minimum confidence threshold.
            track_activation: If True, update access_count and last_accessed
                for returned memories (A-MEM style activation tracking).

        Returns:
            List of ScoredResult[ProceduralMemory] sorted by similarity.
        """
        from engram.models import ProceduralMemory

        filters = self._build_filters(user_id, org_id, min_confidence)

        results = await self._search(
            collection="procedural",
            query_vector=query_vector,
            filters=filters,
            limit=limit,
        )

        scored_results = [
            ScoredResult(
                memory=self._payload_to_memory(r.payload, ProceduralMemory),
                score=r.score,
            )
            for r in results
            if r.payload is not None
        ]

        # A-MEM style activation tracking
        if track_activation and scored_results:
            await self._track_procedural_activation(scored_results)

        return scored_results

    async def _track_procedural_activation(
        self,
        results: list[ScoredResult[ProceduralMemory]],
    ) -> None:
        """Update activation metadata for searched procedural memories.

        Records access_count and last_accessed for A-MEM style
        activation-based strengthening.

        Args:
            results: Search results to track activation for.
        """
        from datetime import UTC, datetime

        collection = self._collection_name("procedural")

        for scored in results:
            memory = scored.memory
            memory.access_count += 1
            memory.last_accessed = datetime.now(UTC)

            # Update just the activation fields in storage
            await self.client.set_payload(
                collection_name=collection,
                payload={
                    "access_count": memory.access_count,
                    "last_accessed": memory.last_accessed.isoformat(),
                },
                points=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="id",
                                match=models.MatchValue(value=memory.id),
                            ),
                        ]
                    )
                ),
            )

    def _build_filters(
        self,
        user_id: str,
        org_id: str | None = None,
        min_confidence: float | None = None,
        derived_at_before: datetime | None = None,
        timestamp_before: datetime | None = None,
    ) -> list[models.FieldCondition]:
        """Build Qdrant filter conditions.

        Args:
            user_id: User ID for isolation.
            org_id: Optional org ID filter.
            min_confidence: Minimum confidence threshold.
            derived_at_before: Filter for memories derived before this time (bi-temporal).
            timestamp_before: Filter for episodes with timestamp before this time.
        """
        filters: list[models.FieldCondition] = [
            models.FieldCondition(
                key="user_id",
                match=models.MatchValue(value=user_id),
            )
        ]

        if org_id is not None:
            filters.append(
                models.FieldCondition(
                    key="org_id",
                    match=models.MatchValue(value=org_id),
                )
            )

        if min_confidence is not None:
            filters.append(
                models.FieldCondition(
                    key="confidence_value",
                    range=models.Range(gte=min_confidence),
                )
            )

        if derived_at_before is not None:
            # Filter for memories derived before this timestamp
            filters.append(
                models.FieldCondition(
                    key="derived_at",
                    range=models.DatetimeRange(lte=derived_at_before),
                )
            )

        if timestamp_before is not None:
            # Filter for episodes with timestamp before this
            filters.append(
                models.FieldCondition(
                    key="timestamp",
                    range=models.DatetimeRange(lte=timestamp_before),
                )
            )

        return filters

    async def _search(
        self,
        collection: str,
        query_vector: Sequence[float],
        filters: list[models.FieldCondition],
        limit: int,
    ) -> list[models.ScoredPoint]:
        """Execute a vector search with filters."""
        collection_name = self._collection_name(collection)

        results = await self.client.query_points(
            collection_name=collection_name,
            query=list(query_vector),
            query_filter=models.Filter(must=filters) if filters else None,
            limit=limit,
            with_payload=True,
        )
        return list(results.points)
