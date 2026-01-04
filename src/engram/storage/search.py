"""Search operations for Engram storage.

Provides vector similarity search methods for all memory types.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from qdrant_client import models

if TYPE_CHECKING:
    from collections.abc import Sequence

    from engram.models import Episode, Fact, InhibitoryFact, ProceduralMemory, SemanticMemory


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
    ) -> list[Episode]:
        """Search for similar episodes.

        Args:
            query_vector: Query embedding vector.
            user_id: User ID for isolation.
            org_id: Optional org ID filter.
            limit: Maximum results to return.
            min_importance: Minimum importance threshold.

        Returns:
            List of matching Episodes sorted by similarity.
        """
        from engram.models import Episode

        filters = self._build_filters(user_id, org_id)
        if min_importance is not None:
            filters.append(
                models.FieldCondition(
                    key="importance",
                    range=models.Range(gte=min_importance),
                )
            )

        results = await self._search(
            collection="episode",
            query_vector=query_vector,
            filters=filters,
            limit=limit,
        )

        return [
            self._payload_to_memory(r.payload, Episode) for r in results if r.payload is not None
        ]

    async def search_facts(
        self,
        query_vector: Sequence[float],
        user_id: str,
        org_id: str | None = None,
        limit: int = 10,
        min_confidence: float | None = None,
        category: str | None = None,
    ) -> list[Fact]:
        """Search for similar facts.

        Args:
            query_vector: Query embedding vector.
            user_id: User ID for isolation.
            org_id: Optional org ID filter.
            limit: Maximum results to return.
            min_confidence: Minimum confidence threshold.
            category: Filter by fact category.

        Returns:
            List of matching Facts sorted by similarity.
        """
        from engram.models import Fact

        filters = self._build_filters(user_id, org_id, min_confidence)
        if category is not None:
            filters.append(
                models.FieldCondition(
                    key="category",
                    match=models.MatchValue(value=category),
                )
            )

        results = await self._search(
            collection="fact",
            query_vector=query_vector,
            filters=filters,
            limit=limit,
        )

        return [self._payload_to_memory(r.payload, Fact) for r in results if r.payload is not None]

    async def search_semantic(
        self,
        query_vector: Sequence[float],
        user_id: str,
        org_id: str | None = None,
        limit: int = 10,
        min_confidence: float | None = None,
    ) -> list[SemanticMemory]:
        """Search for similar semantic memories.

        Args:
            query_vector: Query embedding vector.
            user_id: User ID for isolation.
            org_id: Optional org ID filter.
            limit: Maximum results to return.
            min_confidence: Minimum confidence threshold.

        Returns:
            List of matching SemanticMemory sorted by similarity.
        """
        from engram.models import SemanticMemory

        filters = self._build_filters(user_id, org_id, min_confidence)

        results = await self._search(
            collection="semantic",
            query_vector=query_vector,
            filters=filters,
            limit=limit,
        )

        return [
            self._payload_to_memory(r.payload, SemanticMemory)
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
    ) -> list[ProceduralMemory]:
        """Search for similar procedural memories.

        Args:
            query_vector: Query embedding vector.
            user_id: User ID for isolation.
            org_id: Optional org ID filter.
            limit: Maximum results to return.
            min_confidence: Minimum confidence threshold.

        Returns:
            List of matching ProceduralMemory sorted by similarity.
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
            self._payload_to_memory(r.payload, ProceduralMemory)
            for r in results
            if r.payload is not None
        ]

    async def search_inhibitory(
        self,
        query_vector: Sequence[float],
        user_id: str,
        org_id: str | None = None,
        limit: int = 10,
    ) -> list[InhibitoryFact]:
        """Search for similar inhibitory facts.

        Args:
            query_vector: Query embedding vector.
            user_id: User ID for isolation.
            org_id: Optional org ID filter.
            limit: Maximum results to return.

        Returns:
            List of matching InhibitoryFact sorted by similarity.
        """
        from engram.models import InhibitoryFact

        filters = self._build_filters(user_id, org_id)

        results = await self._search(
            collection="inhibitory",
            query_vector=query_vector,
            filters=filters,
            limit=limit,
        )

        return [
            self._payload_to_memory(r.payload, InhibitoryFact)
            for r in results
            if r.payload is not None
        ]

    def _build_filters(
        self,
        user_id: str,
        org_id: str | None = None,
        min_confidence: float | None = None,
    ) -> list[models.FieldCondition]:
        """Build Qdrant filter conditions."""
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
