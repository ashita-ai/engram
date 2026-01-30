"""Bidirectional linking operations for Engram storage.

Provides atomic methods to create and remove bidirectional links between
SemanticMemory and ProceduralMemory instances. This ensures A-MEM multi-hop
reasoning works correctly by maintaining consistent graph traversal.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

if TYPE_CHECKING:
    from engram.models import ProceduralMemory, SemanticMemory

logger = logging.getLogger(__name__)

# Type alias for linkable memory types
LinkableMemoryType = Literal["semantic", "procedural"]


@dataclass
class LinkResult:
    """Result of a link operation.

    Attributes:
        success: Whether both sides were successfully linked.
        source_updated: Whether the source memory was updated.
        target_updated: Whether the target memory was updated.
        source_id: ID of the source memory.
        target_id: ID of the target memory.
        link_type: Type of link created.
        error: Error message if linking failed.
    """

    success: bool
    source_updated: bool
    target_updated: bool
    source_id: str
    target_id: str
    link_type: str
    error: str | None = None


class LinkingMixin:
    """Mixin providing bidirectional linking operations for EngramStorage.

    This mixin expects the following methods from the base class:
    - get_semantic(memory_id, user_id) -> SemanticMemory | None
    - get_procedural(memory_id, user_id) -> ProceduralMemory | None
    - update_semantic_memory(memory) -> bool
    - update_procedural_memory(memory) -> bool
    """

    # These will be provided by the base class
    get_semantic: Any
    get_procedural: Any
    update_semantic_memory: Any
    update_procedural_memory: Any

    async def link_memories(
        self,
        source_id: str,
        target_id: str,
        user_id: str,
        link_type: str = "related",
        source_type: LinkableMemoryType | None = None,
        target_type: LinkableMemoryType | None = None,
        bidirectional: bool = True,
    ) -> LinkResult:
        """Create a bidirectional link between two memories.

        This is the recommended way to create links, as it ensures both
        memories are updated atomically (best-effort since Qdrant doesn't
        support transactions).

        Args:
            source_id: ID of the source memory.
            target_id: ID of the target memory.
            user_id: User ID for multi-tenancy verification.
            link_type: Type of link (related, supersedes, contradicts).
            source_type: Type of source memory (semantic/procedural).
                        If None, auto-detected from ID prefix.
            target_type: Type of target memory (semantic/procedural).
                        If None, auto-detected from ID prefix.
            bidirectional: If True (default), create reverse link too.

        Returns:
            LinkResult with success status and details.

        Example:
            ```python
            result = await storage.link_memories(
                source_id="sem_abc123",
                target_id="sem_def456",
                user_id="user_1",
                link_type="related",
            )
            if result.success:
                print("Both memories linked")
            ```
        """
        # Auto-detect types from ID prefixes
        source_type = source_type or self._detect_memory_type(source_id)
        target_type = target_type or self._detect_memory_type(target_id)

        if source_type is None:
            return LinkResult(
                success=False,
                source_updated=False,
                target_updated=False,
                source_id=source_id,
                target_id=target_id,
                link_type=link_type,
                error=f"Cannot determine memory type for source: {source_id}",
            )

        if target_type is None:
            return LinkResult(
                success=False,
                source_updated=False,
                target_updated=False,
                source_id=source_id,
                target_id=target_id,
                link_type=link_type,
                error=f"Cannot determine memory type for target: {target_id}",
            )

        # Fetch both memories
        source = await self._get_linkable_memory(source_id, user_id, source_type)
        target = await self._get_linkable_memory(target_id, user_id, target_type)

        if source is None:
            return LinkResult(
                success=False,
                source_updated=False,
                target_updated=False,
                source_id=source_id,
                target_id=target_id,
                link_type=link_type,
                error=f"Source memory not found: {source_id}",
            )

        if target is None:
            return LinkResult(
                success=False,
                source_updated=False,
                target_updated=False,
                source_id=source_id,
                target_id=target_id,
                link_type=link_type,
                error=f"Target memory not found: {target_id}",
            )

        # Add forward link (source → target)
        source.add_link(target_id, link_type)
        source_updated = await self._update_linkable_memory(source, source_type)

        if not source_updated:
            return LinkResult(
                success=False,
                source_updated=False,
                target_updated=False,
                source_id=source_id,
                target_id=target_id,
                link_type=link_type,
                error="Failed to update source memory",
            )

        # Add reverse link (target → source) if bidirectional
        target_updated = True
        if bidirectional:
            target.add_link(source_id, link_type)
            target_updated = await self._update_linkable_memory(target, target_type)

            if not target_updated:
                # Best-effort: source was updated but target failed
                # Log warning but don't rollback source (partial success)
                logger.warning(
                    "Bidirectional link partially failed: source %s updated, "
                    "target %s failed to update",
                    source_id,
                    target_id,
                )
                return LinkResult(
                    success=False,
                    source_updated=True,
                    target_updated=False,
                    source_id=source_id,
                    target_id=target_id,
                    link_type=link_type,
                    error="Failed to update target memory (source was updated)",
                )

        logger.debug(
            "Created %slink: %s ↔ %s (type=%s)",
            "bidirectional " if bidirectional else "",
            source_id,
            target_id,
            link_type,
        )

        return LinkResult(
            success=True,
            source_updated=True,
            target_updated=target_updated if bidirectional else False,
            source_id=source_id,
            target_id=target_id,
            link_type=link_type,
        )

    async def unlink_memories(
        self,
        source_id: str,
        target_id: str,
        user_id: str,
        source_type: LinkableMemoryType | None = None,
        target_type: LinkableMemoryType | None = None,
        bidirectional: bool = True,
    ) -> LinkResult:
        """Remove a bidirectional link between two memories.

        Args:
            source_id: ID of the source memory.
            target_id: ID of the target memory.
            user_id: User ID for multi-tenancy verification.
            source_type: Type of source memory (semantic/procedural).
                        If None, auto-detected from ID prefix.
            target_type: Type of target memory (semantic/procedural).
                        If None, auto-detected from ID prefix.
            bidirectional: If True (default), remove reverse link too.

        Returns:
            LinkResult with success status and details.
        """
        # Auto-detect types from ID prefixes
        source_type = source_type or self._detect_memory_type(source_id)
        target_type = target_type or self._detect_memory_type(target_id)

        if source_type is None:
            return LinkResult(
                success=False,
                source_updated=False,
                target_updated=False,
                source_id=source_id,
                target_id=target_id,
                link_type="",
                error=f"Cannot determine memory type for source: {source_id}",
            )

        if target_type is None:
            return LinkResult(
                success=False,
                source_updated=False,
                target_updated=False,
                source_id=source_id,
                target_id=target_id,
                link_type="",
                error=f"Cannot determine memory type for target: {target_id}",
            )

        # Fetch both memories
        source = await self._get_linkable_memory(source_id, user_id, source_type)
        target = await self._get_linkable_memory(target_id, user_id, target_type)

        if source is None:
            return LinkResult(
                success=False,
                source_updated=False,
                target_updated=False,
                source_id=source_id,
                target_id=target_id,
                link_type="",
                error=f"Source memory not found: {source_id}",
            )

        # Remove forward link (source → target)
        removed = source.remove_link(target_id)
        source_updated = False
        if removed:
            source_updated = await self._update_linkable_memory(source, source_type)

        # Remove reverse link (target → source) if bidirectional
        target_updated = False
        if bidirectional and target is not None:
            removed_reverse = target.remove_link(source_id)
            if removed_reverse:
                target_updated = await self._update_linkable_memory(target, target_type)

        success = source_updated or target_updated  # At least one side changed

        logger.debug(
            "Removed %slink: %s ↔ %s",
            "bidirectional " if bidirectional else "",
            source_id,
            target_id,
        )

        return LinkResult(
            success=success,
            source_updated=source_updated,
            target_updated=target_updated,
            source_id=source_id,
            target_id=target_id,
            link_type="",
        )

    def _detect_memory_type(self, memory_id: str) -> LinkableMemoryType | None:
        """Detect memory type from ID prefix.

        Args:
            memory_id: Memory ID with type prefix.

        Returns:
            Memory type or None if unknown.
        """
        if memory_id.startswith("sem_"):
            return "semantic"
        elif memory_id.startswith("proc_"):
            return "procedural"
        return None

    async def _get_linkable_memory(
        self,
        memory_id: str,
        user_id: str,
        memory_type: LinkableMemoryType,
    ) -> SemanticMemory | ProceduralMemory | None:
        """Get a linkable memory by type.

        Args:
            memory_id: Memory ID.
            user_id: User ID for multi-tenancy.
            memory_type: Type of memory to fetch.

        Returns:
            Memory instance or None if not found.
        """
        if memory_type == "semantic":
            result = await self.get_semantic(memory_id, user_id)
            return cast("SemanticMemory | None", result)
        elif memory_type == "procedural":
            result = await self.get_procedural(memory_id, user_id)
            return cast("ProceduralMemory | None", result)
        return None

    async def _update_linkable_memory(
        self,
        memory: SemanticMemory | ProceduralMemory,
        memory_type: LinkableMemoryType,
    ) -> bool:
        """Update a linkable memory by type.

        Args:
            memory: Memory instance to update.
            memory_type: Type of memory.

        Returns:
            True if updated successfully.
        """
        if memory_type == "semantic":
            result = await self.update_semantic_memory(memory)
            return cast(bool, result)
        elif memory_type == "procedural":
            result = await self.update_procedural_memory(memory)
            return cast(bool, result)
        return False


__all__ = ["LinkingMixin", "LinkResult", "LinkableMemoryType"]
