"""HistoryEntry model - memory change log for auditability.

Tracks the complete history of changes to mutable memories,
enabling audit trails, debugging, and compliance requirements.
"""

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .base import generate_id

# Change types for history entries
ChangeType = Literal["created", "updated", "strengthened", "weakened", "archived", "deleted"]

# Trigger types - what caused the change
TriggerType = Literal[
    "encode",  # Created during encode
    "consolidation",  # Changed during consolidation
    "decay",  # Changed during decay workflow
    "promotion",  # Changed during procedural promotion
    "manual",  # Manual API update
    "retrieval",  # Changed due to retrieval (Testing Effect)
    "system",  # System-initiated change
]


class HistoryEntry(BaseModel):
    """History log entry for tracking memory state changes.

    Every change to a mutable memory (SemanticMemory, ProceduralMemory,
    StructuredMemory) is logged with before/after state for full auditability.

    Attributes:
        id: Unique identifier for this history entry.
        memory_id: ID of the memory that changed.
        memory_type: Type of memory (structured, semantic, procedural).
        user_id: User who owns the memory.
        org_id: Organization (optional).
        timestamp: When the change occurred.
        change_type: Type of change (created, updated, archived, deleted).
        trigger: What caused the change (encode, consolidation, decay, manual).
        before: Previous state (null for create).
        after: New state (null for delete).
        diff: What specifically changed.
        reason: Human-readable explanation (optional).
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: generate_id("hist"))
    memory_id: str = Field(description="ID of the memory that changed")
    memory_type: str = Field(description="Type: structured, semantic, procedural")
    user_id: str = Field(description="User who owns the memory")
    org_id: str | None = Field(default=None, description="Organization (optional)")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When the change occurred",
    )
    change_type: ChangeType = Field(description="Type of change")
    trigger: TriggerType = Field(description="What caused the change")
    before: dict[str, Any] | None = Field(
        default=None,
        description="Previous state (null for create)",
    )
    after: dict[str, Any] | None = Field(
        default=None,
        description="New state (null for delete)",
    )
    diff: dict[str, Any] = Field(
        default_factory=dict,
        description="What specifically changed",
    )
    reason: str | None = Field(
        default=None,
        description="Human-readable explanation",
    )

    @classmethod
    def for_create(
        cls,
        memory_id: str,
        memory_type: str,
        user_id: str,
        trigger: TriggerType,
        after_state: dict[str, Any],
        org_id: str | None = None,
        reason: str | None = None,
    ) -> "HistoryEntry":
        """Create history entry for a new memory creation."""
        return cls(
            memory_id=memory_id,
            memory_type=memory_type,
            user_id=user_id,
            org_id=org_id,
            change_type="created",
            trigger=trigger,
            before=None,
            after=after_state,
            diff=after_state,  # Everything is new
            reason=reason,
        )

    @classmethod
    def for_update(
        cls,
        memory_id: str,
        memory_type: str,
        user_id: str,
        trigger: TriggerType,
        before_state: dict[str, Any],
        after_state: dict[str, Any],
        org_id: str | None = None,
        reason: str | None = None,
    ) -> "HistoryEntry":
        """Create history entry for a memory update."""
        # Compute diff - only fields that changed
        diff: dict[str, Any] = {}
        for key in set(before_state.keys()) | set(after_state.keys()):
            old_val = before_state.get(key)
            new_val = after_state.get(key)
            if old_val != new_val:
                diff[key] = {"old": old_val, "new": new_val}

        return cls(
            memory_id=memory_id,
            memory_type=memory_type,
            user_id=user_id,
            org_id=org_id,
            change_type="updated",
            trigger=trigger,
            before=before_state,
            after=after_state,
            diff=diff,
            reason=reason,
        )

    @classmethod
    def for_strengthen(
        cls,
        memory_id: str,
        memory_type: str,
        user_id: str,
        trigger: TriggerType,
        old_strength: float,
        new_strength: float,
        old_passes: int,
        new_passes: int,
        org_id: str | None = None,
        reason: str | None = None,
    ) -> "HistoryEntry":
        """Create history entry for memory strengthening."""
        before = {"consolidation_strength": old_strength, "consolidation_passes": old_passes}
        after = {"consolidation_strength": new_strength, "consolidation_passes": new_passes}

        return cls(
            memory_id=memory_id,
            memory_type=memory_type,
            user_id=user_id,
            org_id=org_id,
            change_type="strengthened",
            trigger=trigger,
            before=before,
            after=after,
            diff={
                "consolidation_strength": {"old": old_strength, "new": new_strength},
                "consolidation_passes": {"old": old_passes, "new": new_passes},
            },
            reason=reason or f"Strengthened from {old_strength:.2f} to {new_strength:.2f}",
        )

    @classmethod
    def for_weaken(
        cls,
        memory_id: str,
        memory_type: str,
        user_id: str,
        trigger: TriggerType,
        old_strength: float,
        new_strength: float,
        org_id: str | None = None,
        reason: str | None = None,
    ) -> "HistoryEntry":
        """Create history entry for memory weakening."""
        return cls(
            memory_id=memory_id,
            memory_type=memory_type,
            user_id=user_id,
            org_id=org_id,
            change_type="weakened",
            trigger=trigger,
            before={"consolidation_strength": old_strength},
            after={"consolidation_strength": new_strength},
            diff={"consolidation_strength": {"old": old_strength, "new": new_strength}},
            reason=reason or f"Weakened from {old_strength:.2f} to {new_strength:.2f}",
        )

    @classmethod
    def for_archive(
        cls,
        memory_id: str,
        memory_type: str,
        user_id: str,
        trigger: TriggerType,
        before_state: dict[str, Any],
        org_id: str | None = None,
        reason: str | None = None,
    ) -> "HistoryEntry":
        """Create history entry for memory archival."""
        return cls(
            memory_id=memory_id,
            memory_type=memory_type,
            user_id=user_id,
            org_id=org_id,
            change_type="archived",
            trigger=trigger,
            before=before_state,
            after=None,
            diff={},
            reason=reason or "Memory archived due to low confidence",
        )

    @classmethod
    def for_delete(
        cls,
        memory_id: str,
        memory_type: str,
        user_id: str,
        trigger: TriggerType,
        before_state: dict[str, Any],
        org_id: str | None = None,
        reason: str | None = None,
    ) -> "HistoryEntry":
        """Create history entry for memory deletion."""
        return cls(
            memory_id=memory_id,
            memory_type=memory_type,
            user_id=user_id,
            org_id=org_id,
            change_type="deleted",
            trigger=trigger,
            before=before_state,
            after=None,
            diff={},
            reason=reason or "Memory deleted",
        )

    @classmethod
    def for_retrieval(
        cls,
        memory_id: str,
        memory_type: str,
        user_id: str,
        old_retrieval_count: int,
        new_retrieval_count: int,
        org_id: str | None = None,
    ) -> "HistoryEntry":
        """Create history entry for retrieval access tracking."""
        return cls(
            memory_id=memory_id,
            memory_type=memory_type,
            user_id=user_id,
            org_id=org_id,
            change_type="updated",
            trigger="retrieval",
            before={"retrieval_count": old_retrieval_count},
            after={"retrieval_count": new_retrieval_count},
            diff={"retrieval_count": {"old": old_retrieval_count, "new": new_retrieval_count}},
            reason="Memory accessed during retrieval (Testing Effect)",
        )

    def __str__(self) -> str:
        """String representation showing change type and memory."""
        return f"HistoryEntry({self.change_type} {self.memory_type} {self.memory_id})"


__all__ = [
    "ChangeType",
    "HistoryEntry",
    "TriggerType",
]
