"""AuditEntry model - operation logging for auditability."""

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .base import generate_id


class AuditEntry(BaseModel):
    """Audit log entry for tracking all Engram operations.

    Every encode, recall, consolidate, and decay operation is logged
    for auditability and debugging. Stored in engram_audit collection.

    Attributes:
        id: Unique identifier for this audit entry.
        timestamp: When the operation occurred.
        event: Type of operation (encode, recall, consolidate, decay).
        user_id: User who triggered the operation.
        org_id: Organization (optional).
        session_id: Session context (optional).
        details: Event-specific data (varies by event type).
        duration_ms: How long the operation took (optional).
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: generate_id("audit"))
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When the operation occurred",
    )
    event: str = Field(description="Operation type: encode, recall, consolidate, decay")
    user_id: str = Field(description="User who triggered the operation")
    org_id: str | None = Field(default=None, description="Organization (optional)")
    session_id: str | None = Field(default=None, description="Session context (optional)")
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Event-specific data",
    )
    duration_ms: int | None = Field(
        default=None,
        description="Operation duration in milliseconds",
    )

    @classmethod
    def for_encode(
        cls,
        user_id: str,
        episode_id: str,
        facts_count: int,
        org_id: str | None = None,
        session_id: str | None = None,
        duration_ms: int | None = None,
    ) -> "AuditEntry":
        """Create audit entry for an encode operation."""
        return cls(
            event="encode",
            user_id=user_id,
            org_id=org_id,
            session_id=session_id,
            details={
                "episode_id": episode_id,
                "facts_count": facts_count,
            },
            duration_ms=duration_ms,
        )

    @classmethod
    def for_recall(
        cls,
        user_id: str,
        query_hash: str,
        results_count: int,
        memory_types: list[str],
        org_id: str | None = None,
        session_id: str | None = None,
        duration_ms: int | None = None,
    ) -> "AuditEntry":
        """Create audit entry for a recall operation.

        Args:
            user_id: User who triggered the operation.
            query_hash: SHA256 hash of query (for privacy).
            results_count: Number of results returned.
            memory_types: Memory types in results.
            org_id: Organization (optional).
            session_id: Session context (optional).
            duration_ms: Operation duration.
        """
        return cls(
            event="recall",
            user_id=user_id,
            org_id=org_id,
            session_id=session_id,
            details={
                "query_hash": query_hash,
                "results_count": results_count,
                "memory_types": memory_types,
            },
            duration_ms=duration_ms,
        )

    @classmethod
    def for_consolidate(
        cls,
        user_id: str,
        episode_ids: list[str],
        facts_created: int,
        links_created: int,
        org_id: str | None = None,
        duration_ms: int | None = None,
    ) -> "AuditEntry":
        """Create audit entry for a consolidate operation."""
        return cls(
            event="consolidate",
            user_id=user_id,
            org_id=org_id,
            details={
                "episode_ids": episode_ids,
                "facts_created": facts_created,
                "links_created": links_created,
            },
            duration_ms=duration_ms,
        )

    @classmethod
    def for_decay(
        cls,
        user_id: str,
        memories_updated: int,
        memories_archived: int,
        org_id: str | None = None,
        duration_ms: int | None = None,
    ) -> "AuditEntry":
        """Create audit entry for a decay operation."""
        return cls(
            event="decay",
            user_id=user_id,
            org_id=org_id,
            details={
                "memories_updated": memories_updated,
                "memories_archived": memories_archived,
            },
            duration_ms=duration_ms,
        )

    @classmethod
    def for_delete(
        cls,
        user_id: str,
        memory_id: str,
        memory_type: str,
        org_id: str | None = None,
        duration_ms: int | None = None,
    ) -> "AuditEntry":
        """Create audit entry for a delete operation."""
        return cls(
            event="delete",
            user_id=user_id,
            org_id=org_id,
            details={
                "memory_id": memory_id,
                "memory_type": memory_type,
            },
            duration_ms=duration_ms,
        )

    @classmethod
    def for_update(
        cls,
        user_id: str,
        memory_id: str,
        memory_type: str,
        changes: list[dict[str, str]],
        org_id: str | None = None,
        duration_ms: int | None = None,
    ) -> "AuditEntry":
        """Create audit entry for a memory update operation.

        Args:
            user_id: User who performed the update.
            memory_id: ID of the updated memory.
            memory_type: Type of memory (structured, semantic, procedural).
            changes: List of changes, each with field, old, new values.
            org_id: Optional organization ID.
            duration_ms: Processing duration in milliseconds.
        """
        return cls(
            event="update",
            user_id=user_id,
            org_id=org_id,
            details={
                "memory_id": memory_id,
                "memory_type": memory_type,
                "changes": changes,
                "change_count": len(changes),
            },
            duration_ms=duration_ms,
        )

    @classmethod
    def for_bulk_delete(
        cls,
        user_id: str,
        deleted_counts: dict[str, int],
        org_id: str | None = None,
        duration_ms: int | None = None,
    ) -> "AuditEntry":
        """Create audit entry for a bulk delete operation (GDPR erasure)."""
        return cls(
            event="bulk_delete",
            user_id=user_id,
            org_id=org_id,
            details={
                "deleted_counts": deleted_counts,
                "total_deleted": sum(deleted_counts.values()),
            },
            duration_ms=duration_ms,
        )

    def __str__(self) -> str:
        """String representation showing event and user."""
        return f"AuditEntry({self.event} by {self.user_id})"
