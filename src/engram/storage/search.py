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

    from engram.models import Episode, Fact, NegationFact, ProceduralMemory, SemanticMemory

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

    async def search_facts(
        self,
        query_vector: Sequence[float],
        user_id: str,
        org_id: str | None = None,
        limit: int = 10,
        min_confidence: float | None = None,
        category: str | None = None,
        derived_at_before: datetime | None = None,
    ) -> list[ScoredResult[Fact]]:
        """Search for similar facts.

        Args:
            query_vector: Query embedding vector.
            user_id: User ID for isolation.
            org_id: Optional org ID filter.
            limit: Maximum results to return.
            min_confidence: Minimum confidence threshold.
            category: Filter by fact category.
            derived_at_before: Only include facts derived before this time.

        Returns:
            List of ScoredResult[Fact] sorted by similarity.
        """
        from engram.models import Fact

        filters = self._build_filters(
            user_id, org_id, min_confidence, derived_at_before=derived_at_before
        )
        if category is not None:
            filters.append(
                models.FieldCondition(
                    key="category",
                    match=models.MatchValue(value=category),
                )
            )

        results = await self._search(
            collection="factual",
            query_vector=query_vector,
            filters=filters,
            limit=limit,
        )

        return [
            ScoredResult(
                memory=self._payload_to_memory(r.payload, Fact),
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
        derived_at_before: datetime | None = None,
    ) -> list[ScoredResult[SemanticMemory]]:
        """Search for similar semantic memories.

        Args:
            query_vector: Query embedding vector.
            user_id: User ID for isolation.
            org_id: Optional org ID filter.
            limit: Maximum results to return.
            min_confidence: Minimum confidence threshold.
            derived_at_before: Only include memories derived before this time.

        Returns:
            List of ScoredResult[SemanticMemory] sorted by similarity.
        """
        from engram.models import SemanticMemory

        filters = self._build_filters(
            user_id, org_id, min_confidence, derived_at_before=derived_at_before
        )

        results = await self._search(
            collection="semantic",
            query_vector=query_vector,
            filters=filters,
            limit=limit,
        )

        return [
            ScoredResult(
                memory=self._payload_to_memory(r.payload, SemanticMemory),
                score=r.score,
            )
            for r in results
            if r.payload is not None
        ]

    async def search_procedural(
        self,
        query_vector: Sequence[float],
        user_id: str,
        org_id: str | None = None,
        limit: int = 10,
        min_confidence: float | None = None,
    ) -> list[ScoredResult[ProceduralMemory]]:
        """Search for similar procedural memories.

        Args:
            query_vector: Query embedding vector.
            user_id: User ID for isolation.
            org_id: Optional org ID filter.
            limit: Maximum results to return.
            min_confidence: Minimum confidence threshold.

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

        return [
            ScoredResult(
                memory=self._payload_to_memory(r.payload, ProceduralMemory),
                score=r.score,
            )
            for r in results
            if r.payload is not None
        ]

    async def search_negation(
        self,
        query_vector: Sequence[float],
        user_id: str,
        org_id: str | None = None,
        limit: int = 10,
    ) -> list[ScoredResult[NegationFact]]:
        """Search for similar negation facts.

        Args:
            query_vector: Query embedding vector.
            user_id: User ID for isolation.
            org_id: Optional org ID filter.
            limit: Maximum results to return.

        Returns:
            List of ScoredResult[NegationFact] sorted by similarity.
        """
        from engram.models import NegationFact

        filters = self._build_filters(user_id, org_id)

        results = await self._search(
            collection="negation",
            query_vector=query_vector,
            filters=filters,
            limit=limit,
        )

        return [
            ScoredResult(
                memory=self._payload_to_memory(r.payload, NegationFact),
                score=r.score,
            )
            for r in results
            if r.payload is not None
        ]

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
