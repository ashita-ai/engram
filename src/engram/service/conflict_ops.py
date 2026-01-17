"""Conflict operations mixin for EngramService.

Provides methods for detecting, storing, and managing memory conflicts.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from .contradiction import (
    ConflictDetection,
    detect_contradictions,
    detect_contradictions_in_structured,
)

if TYPE_CHECKING:
    from engram.embeddings import Embedder
    from engram.storage import EngramStorage

logger = logging.getLogger(__name__)


class ConflictMixin:
    """Mixin providing conflict detection and management functionality.

    Expects these attributes from the base class:
    - storage: EngramStorage
    - embedder: Embedder
    - _conflicts: dict[str, ConflictDetection]
    """

    storage: EngramStorage
    embedder: Embedder
    _conflicts: dict[str, ConflictDetection]

    def get_conflicts(
        self,
        user_id: str,
        org_id: str | None = None,
        include_resolved: bool = False,
    ) -> list[ConflictDetection]:
        """Get detected conflicts for a user.

        Args:
            user_id: User ID for multi-tenancy.
            org_id: Optional organization ID filter.
            include_resolved: Whether to include resolved conflicts.

        Returns:
            List of ConflictDetection objects.
        """
        conflicts = [
            c
            for c in self._conflicts.values()
            if c.user_id == user_id and (org_id is None or c.org_id == org_id)
        ]

        if not include_resolved:
            conflicts = [c for c in conflicts if c.resolution is None]

        return sorted(conflicts, key=lambda c: c.detected_at, reverse=True)

    def get_conflict(self, conflict_id: str) -> ConflictDetection | None:
        """Get a specific conflict by ID.

        Args:
            conflict_id: The conflict ID.

        Returns:
            ConflictDetection or None if not found.
        """
        return self._conflicts.get(conflict_id)

    def resolve_conflict(
        self,
        conflict_id: str,
        resolution: str,
    ) -> ConflictDetection | None:
        """Resolve a conflict with a given resolution.

        Args:
            conflict_id: The conflict ID.
            resolution: Resolution type (newer_wins, flag_for_review, lower_confidence, create_negation).

        Returns:
            Updated ConflictDetection or None if not found.
        """
        conflict = self._conflicts.get(conflict_id)
        if conflict is None:
            return None

        conflict.resolution = resolution
        conflict.resolved_at = datetime.now(UTC)
        return conflict

    def clear_conflicts(
        self,
        user_id: str,
        org_id: str | None = None,
    ) -> int:
        """Clear all conflicts for a user.

        Args:
            user_id: User ID for multi-tenancy.
            org_id: Optional organization ID filter.

        Returns:
            Number of conflicts cleared.
        """
        to_remove = [
            cid
            for cid, c in self._conflicts.items()
            if c.user_id == user_id and (org_id is None or c.org_id == org_id)
        ]
        for cid in to_remove:
            del self._conflicts[cid]
        return len(to_remove)

    async def detect_conflicts_in_semantic(
        self,
        user_id: str,
        org_id: str | None = None,
        similarity_threshold: float = 0.5,
        model: str = "openai:gpt-4o-mini",
    ) -> list[ConflictDetection]:
        """Detect conflicts among all semantic memories.

        Compares all semantic memories to find contradictions.

        Args:
            user_id: User ID for multi-tenancy.
            org_id: Optional organization ID filter.
            similarity_threshold: Minimum similarity to check (0.0-1.0).
            model: LLM model for conflict analysis.

        Returns:
            List of detected conflicts.
        """
        # Get all semantic memories for the user
        all_memories = await self.storage.list_semantic_memories(user_id, org_id)

        if len(all_memories) < 2:
            return []

        # Compare all pairs (using first half vs second half to avoid N^2 comparisons)
        mid = len(all_memories) // 2
        new_memories = all_memories[:mid] if mid > 0 else all_memories[:1]
        existing_memories = all_memories[mid:] if mid > 0 else all_memories[1:]

        conflicts = await detect_contradictions(
            new_memories=new_memories,
            existing_memories=existing_memories,
            embedder=self.embedder,
            user_id=user_id,
            org_id=org_id,
            similarity_threshold=similarity_threshold,
            model=model,
        )

        # Store conflicts
        for conflict in conflicts:
            self._conflicts[conflict.id] = conflict
            logger.info(f"Stored conflict {conflict.id}: {conflict.conflict_type}")

        return conflicts

    async def detect_conflicts_in_structured(
        self,
        user_id: str,
        org_id: str | None = None,
        similarity_threshold: float = 0.5,
        model: str = "openai:gpt-4o-mini",
    ) -> list[ConflictDetection]:
        """Detect conflicts among all structured memories.

        Compares structured memory summaries to find contradictions.

        Args:
            user_id: User ID for multi-tenancy.
            org_id: Optional organization ID filter.
            similarity_threshold: Minimum similarity to check (0.0-1.0).
            model: LLM model for conflict analysis.

        Returns:
            List of detected conflicts.
        """
        # Get all structured memories for the user
        all_memories = await self.storage.list_structured_memories(user_id, org_id)

        if len(all_memories) < 2:
            return []

        # Compare all pairs
        mid = len(all_memories) // 2
        new_memories = all_memories[:mid] if mid > 0 else all_memories[:1]
        existing_memories = all_memories[mid:] if mid > 0 else all_memories[1:]

        conflicts = await detect_contradictions_in_structured(
            new_memories=new_memories,
            existing_memories=existing_memories,
            embedder=self.embedder,
            user_id=user_id,
            org_id=org_id,
            similarity_threshold=similarity_threshold,
            model=model,
        )

        # Store conflicts
        for conflict in conflicts:
            self._conflicts[conflict.id] = conflict

        return conflicts


__all__ = ["ConflictMixin"]
