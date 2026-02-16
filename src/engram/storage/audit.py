"""Audit logging operations for Engram storage.

Provides methods to log and query audit entries.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from qdrant_client import models

from engram.storage.retry import qdrant_retry

if TYPE_CHECKING:
    from engram.models import AuditEntry


class AuditMixin:
    """Mixin providing audit operations for EngramStorage.

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
    async def log_audit(self, entry: AuditEntry) -> str:
        """Log an audit entry.

        Audit entries don't require embeddings - they're stored with
        a zero vector for schema compatibility.

        Args:
            entry: AuditEntry to log.

        Returns:
            The audit entry ID.
        """
        collection = self._collection_name("audit")
        key = self._build_key(entry.id, entry.user_id, entry.org_id)
        payload = entry.model_dump(mode="json")

        # Use zero vector for audit (no semantic search needed)
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
    async def get_audit_log(
        self,
        user_id: str,
        org_id: str | None = None,
        event_type: str | None = None,
        limit: int = 100,
    ) -> list[AuditEntry]:
        """Get audit log entries for a user.

        Args:
            user_id: User to get logs for.
            org_id: Optional org filter.
            event_type: Optional event type filter (encode, recall, etc.)
            limit: Maximum entries to return.

        Returns:
            List of AuditEntry sorted by timestamp (newest first).
        """
        from engram.models import AuditEntry

        collection = self._collection_name("audit")

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

        if event_type is not None:
            filters.append(
                models.FieldCondition(
                    key="event",
                    match=models.MatchValue(value=event_type),
                )
            )

        results, _ = await self.client.scroll(
            collection_name=collection,
            scroll_filter=models.Filter(must=filters),
            limit=limit,
            with_payload=True,
        )

        entries: list[AuditEntry] = [
            self._payload_to_memory(r.payload, AuditEntry) for r in results if r.payload is not None
        ]

        # Sort by timestamp descending
        entries.sort(key=lambda e: e.timestamp, reverse=True)

        return entries
