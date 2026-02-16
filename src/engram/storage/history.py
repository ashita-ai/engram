"""History logging operations for Engram storage.

Provides methods to log and query memory change history.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from qdrant_client import models

from engram.storage.retry import qdrant_retry

if TYPE_CHECKING:
    from engram.models import HistoryEntry


class HistoryMixin:
    """Mixin providing history operations for EngramStorage.

    This mixin expects the following attributes/methods from the base class:
    - _collection_name(memory_type) -> str
    - _build_key(memory_id, user_id, org_id) -> str
    - _key_to_point_id(key) -> str
    - _payload_to_memory(payload, memory_class) -> MemoryT
    - _embedding_dim: int
    - client: AsyncQdrantClient
    """

    # These will be provided by the base class
    _collection_name: Any
    _build_key: Any
    _key_to_point_id: Any
    _payload_to_memory: Any
    _embedding_dim: int
    client: Any

    @qdrant_retry
    async def log_history(self, entry: HistoryEntry) -> str:
        """Log a history entry for a memory change.

        History entries don't require embeddings - they're stored with
        a zero vector for schema compatibility.

        Args:
            entry: HistoryEntry to log.

        Returns:
            The history entry ID.
        """
        collection = self._collection_name("history")
        key = self._build_key(entry.id, entry.user_id, entry.org_id)
        payload = entry.model_dump(mode="json")

        # Use zero vector for history (no semantic search needed)
        zero_vector = [0.0] * self._embedding_dim

        await self.client.upsert(
            collection_name=collection,
            points=[
                models.PointStruct(
                    id=self._key_to_point_id(key),
                    vector=zero_vector,
                    payload=payload,
                )
            ],
        )

        return entry.id

    @qdrant_retry
    async def get_memory_history(
        self,
        memory_id: str,
        user_id: str,
        since: datetime | None = None,
        limit: int = 100,
    ) -> list[HistoryEntry]:
        """Get change history for a specific memory.

        Args:
            memory_id: ID of the memory to get history for.
            user_id: User ID for multi-tenancy.
            since: Optional timestamp to filter entries after.
            limit: Maximum entries to return.

        Returns:
            List of HistoryEntry sorted by timestamp (newest first).
        """
        from engram.models import HistoryEntry

        collection = self._collection_name("history")

        filters: list[models.FieldCondition] = [
            models.FieldCondition(
                key="memory_id",
                match=models.MatchValue(value=memory_id),
            ),
            models.FieldCondition(
                key="user_id",
                match=models.MatchValue(value=user_id),
            ),
        ]

        if since is not None:
            filters.append(
                models.FieldCondition(
                    key="timestamp",
                    range=models.Range(gte=since.isoformat()),
                )
            )

        results, _ = await self.client.scroll(
            collection_name=collection,
            scroll_filter=models.Filter(must=filters),
            limit=limit,
            with_payload=True,
        )

        entries: list[HistoryEntry] = [
            self._payload_to_memory(r.payload, HistoryEntry)
            for r in results
            if r.payload is not None
        ]

        # Sort by timestamp descending (newest first)
        entries.sort(key=lambda e: e.timestamp, reverse=True)

        return entries

    @qdrant_retry
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

        Args:
            user_id: User to get history for.
            org_id: Optional org filter.
            memory_type: Optional filter by memory type (structured, semantic, procedural).
            change_type: Optional filter by change type (created, updated, etc.).
            since: Optional timestamp to filter entries after.
            limit: Maximum entries to return.

        Returns:
            List of HistoryEntry sorted by timestamp (newest first).
        """
        from engram.models import HistoryEntry

        collection = self._collection_name("history")

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

        if memory_type is not None:
            filters.append(
                models.FieldCondition(
                    key="memory_type",
                    match=models.MatchValue(value=memory_type),
                )
            )

        if change_type is not None:
            filters.append(
                models.FieldCondition(
                    key="change_type",
                    match=models.MatchValue(value=change_type),
                )
            )

        if since is not None:
            filters.append(
                models.FieldCondition(
                    key="timestamp",
                    range=models.Range(gte=since.isoformat()),
                )
            )

        results, _ = await self.client.scroll(
            collection_name=collection,
            scroll_filter=models.Filter(must=filters),
            limit=limit,
            with_payload=True,
        )

        entries: list[HistoryEntry] = [
            self._payload_to_memory(r.payload, HistoryEntry)
            for r in results
            if r.payload is not None
        ]

        # Sort by timestamp descending (newest first)
        entries.sort(key=lambda e: e.timestamp, reverse=True)

        return entries
