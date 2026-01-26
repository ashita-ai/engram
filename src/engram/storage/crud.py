"""Get and delete operations for Engram storage.

Provides methods to retrieve and delete individual memories.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, TypeVar

from qdrant_client import models

from engram.config import settings
from engram.storage.retry import qdrant_retry

if TYPE_CHECKING:
    from engram.models import (
        AuditEntry,
        Episode,
        ProceduralMemory,
        SemanticMemory,
        StructuredMemory,
    )

MemoryT = TypeVar(
    "MemoryT",
    "Episode",
    "StructuredMemory",
    "SemanticMemory",
    "ProceduralMemory",
    "AuditEntry",
)


class CRUDMixin:
    """Mixin providing get and delete operations for EngramStorage.

    This mixin expects the following attributes/methods from the base class:
    - _collection_name(memory_type) -> str
    - _payload_to_memory(payload, memory_class) -> MemoryT
    - client: AsyncQdrantClient
    """

    # These will be provided by the base class
    _collection_name: Any
    _payload_to_memory: Any
    _memory_to_payload: Any
    client: Any

    async def get_episode(self, episode_id: str, user_id: str) -> Episode | None:
        """Get an episode by ID.

        Args:
            episode_id: Episode identifier.
            user_id: User ID for verification.

        Returns:
            Episode if found and owned by user, None otherwise.
        """
        from engram.models import Episode

        return await self._get_by_id(episode_id, user_id, "episodic", Episode)

    async def get_structured(self, memory_id: str, user_id: str) -> StructuredMemory | None:
        """Get a structured memory by ID."""
        from engram.models import StructuredMemory

        return await self._get_by_id(memory_id, user_id, "structured", StructuredMemory)

    async def get_semantic(self, memory_id: str, user_id: str) -> SemanticMemory | None:
        """Get a semantic memory by ID."""
        from engram.models import SemanticMemory

        return await self._get_by_id(memory_id, user_id, "semantic", SemanticMemory)

    async def get_procedural(self, memory_id: str, user_id: str) -> ProceduralMemory | None:
        """Get a procedural memory by ID."""
        from engram.models import ProceduralMemory

        return await self._get_by_id(memory_id, user_id, "procedural", ProceduralMemory)

    @qdrant_retry
    async def _get_by_id(
        self,
        memory_id: str,
        user_id: str,
        memory_type: str,
        memory_class: type[MemoryT],
    ) -> MemoryT | None:
        """Get a memory by ID with user verification.

        Args:
            memory_id: Memory identifier.
            user_id: User ID for ownership check.
            memory_type: Collection type key.
            memory_class: Memory class for deserialization.

        Returns:
            Memory if found and owned by user, None otherwise.
        """
        collection = self._collection_name(memory_type)

        results = await self.client.scroll(
            collection_name=collection,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="id",
                        match=models.MatchValue(value=memory_id),
                    ),
                    models.FieldCondition(
                        key="user_id",
                        match=models.MatchValue(value=user_id),
                    ),
                ]
            ),
            limit=1,
            with_payload=True,
        )

        points, _ = results
        if not points or points[0].payload is None:
            return None

        result: MemoryT = self._payload_to_memory(points[0].payload, memory_class)
        return result

    async def delete_episode(
        self,
        episode_id: str,
        user_id: str,
        cascade: str = "soft",
    ) -> dict[str, object]:
        """Delete an episode with optional cascade to derived memories.

        Args:
            episode_id: Episode to delete.
            user_id: User ID for ownership verification.
            cascade: Cascade mode:
                - "none": Delete only the episode, leave derived memories orphaned
                - "soft": Remove episode reference from derived memories, reduce confidence,
                          delete derived memories that have no remaining sources
                - "hard": Delete all derived memories that reference this episode

        Returns:
            Dict with deletion statistics:
                - deleted: Whether the episode was deleted
                - structured_deleted: Number of structured memories deleted
                - semantic_deleted: Number of semantic memories deleted
                - semantic_updated: Number of semantic memories updated (reference removed)
                - procedural_deleted: Number of procedural memories deleted
                - procedural_updated: Number of procedural memories updated (reference removed)
        """
        # Use typed counters for cascade statistics
        structured_deleted = 0
        semantic_deleted = 0
        semantic_updated = 0
        procedural_deleted = 0
        procedural_updated = 0

        # First verify the episode exists
        episode = await self.get_episode(episode_id, user_id)
        if not episode:
            return {
                "deleted": False,
                "structured_deleted": 0,
                "semantic_deleted": 0,
                "semantic_updated": 0,
                "procedural_deleted": 0,
                "procedural_updated": 0,
            }

        if cascade != "none":
            # Handle StructuredMemory (1:1 with episode, always delete)
            structured = await self.get_structured_for_episode(episode_id, user_id)
            if structured:
                await self._delete_by_id(structured.id, user_id, "structured")
                structured_deleted = 1

            # Handle SemanticMemory
            semantics = await self._find_semantics_by_source_episode(episode_id, user_id)
            for sem in semantics:
                if cascade == "hard":
                    # Hard cascade: delete any semantic that references this episode
                    await self._delete_by_id(sem.id, user_id, "semantic")
                    semantic_deleted += 1
                else:
                    # Soft cascade: remove reference, reduce confidence
                    original_count = len(sem.source_episode_ids)
                    sem.source_episode_ids = [
                        eid for eid in sem.source_episode_ids if eid != episode_id
                    ]

                    if not sem.source_episode_ids:
                        # No sources left, delete
                        await self._delete_by_id(sem.id, user_id, "semantic")
                        semantic_deleted += 1
                    else:
                        # Reduce confidence proportionally
                        remaining_count = len(sem.source_episode_ids)
                        confidence_factor = remaining_count / original_count
                        sem.confidence.value *= confidence_factor
                        sem.confidence.value = max(0.1, sem.confidence.value)  # Floor at 0.1
                        await self.update_semantic_memory(sem)
                        semantic_updated += 1

            # Handle ProceduralMemory
            procedurals = await self._find_procedurals_by_source_episode(episode_id, user_id)
            for proc in procedurals:
                if cascade == "hard":
                    await self._delete_by_id(proc.id, user_id, "procedural")
                    procedural_deleted += 1
                else:
                    # Soft cascade: remove reference
                    original_count = len(proc.source_episode_ids)
                    proc.source_episode_ids = [
                        eid for eid in proc.source_episode_ids if eid != episode_id
                    ]

                    if not proc.source_episode_ids and not proc.source_semantic_ids:
                        # No sources left at all, delete
                        await self._delete_by_id(proc.id, user_id, "procedural")
                        procedural_deleted += 1
                    else:
                        # Reduce confidence proportionally (only if episode sources changed)
                        if original_count > 0:
                            remaining_count = len(proc.source_episode_ids)
                            confidence_factor = (
                                remaining_count / original_count if original_count > 0 else 1.0
                            )
                            proc.confidence.value *= confidence_factor
                            proc.confidence.value = max(0.1, proc.confidence.value)
                        await self.update_procedural_memory(proc)
                        procedural_updated += 1

        # Delete the episode itself
        deleted = await self._delete_by_id(episode_id, user_id, "episodic")

        return {
            "deleted": deleted,
            "structured_deleted": structured_deleted,
            "semantic_deleted": semantic_deleted,
            "semantic_updated": semantic_updated,
            "procedural_deleted": procedural_deleted,
            "procedural_updated": procedural_updated,
        }

    async def _find_semantics_by_source_episode(
        self,
        episode_id: str,
        user_id: str,
    ) -> list[SemanticMemory]:
        """Find all SemanticMemory records that reference a given episode.

        Args:
            episode_id: The episode ID to search for.
            user_id: User ID for isolation.

        Returns:
            List of SemanticMemory objects containing the episode in source_episode_ids.
        """
        from engram.models import SemanticMemory

        collection = self._collection_name("semantic")

        # Qdrant supports array contains via MatchAny
        results, _ = await self.client.scroll(
            collection_name=collection,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="source_episode_ids",
                        match=models.MatchAny(any=[episode_id]),
                    ),
                    models.FieldCondition(
                        key="user_id",
                        match=models.MatchValue(value=user_id),
                    ),
                ]
            ),
            limit=settings.storage_max_scroll_limit,
            with_payload=True,
            with_vectors=True,
        )

        memories: list[SemanticMemory] = []
        for point in results:
            if point.payload is not None:
                memory = self._payload_to_memory(point.payload, SemanticMemory)
                if isinstance(point.vector, list):
                    memory.embedding = point.vector
                memories.append(memory)

        return memories

    async def _find_procedurals_by_source_episode(
        self,
        episode_id: str,
        user_id: str,
    ) -> list[ProceduralMemory]:
        """Find all ProceduralMemory records that reference a given episode.

        Args:
            episode_id: The episode ID to search for.
            user_id: User ID for isolation.

        Returns:
            List of ProceduralMemory objects containing the episode in source_episode_ids.
        """
        from engram.models import ProceduralMemory

        collection = self._collection_name("procedural")

        results, _ = await self.client.scroll(
            collection_name=collection,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="source_episode_ids",
                        match=models.MatchAny(any=[episode_id]),
                    ),
                    models.FieldCondition(
                        key="user_id",
                        match=models.MatchValue(value=user_id),
                    ),
                ]
            ),
            limit=settings.storage_max_scroll_limit,
            with_payload=True,
            with_vectors=True,
        )

        memories: list[ProceduralMemory] = []
        for point in results:
            if point.payload is not None:
                memory = self._payload_to_memory(point.payload, ProceduralMemory)
                if isinstance(point.vector, list):
                    memory.embedding = point.vector
                memories.append(memory)

        return memories

    async def delete_structured(self, memory_id: str, user_id: str) -> bool:
        """Delete a structured memory."""
        return await self._delete_by_id(memory_id, user_id, "structured")

    async def delete_semantic(self, memory_id: str, user_id: str) -> bool:
        """Delete a semantic memory."""
        return await self._delete_by_id(memory_id, user_id, "semantic")

    async def delete_procedural(self, memory_id: str, user_id: str) -> bool:
        """Delete a procedural memory."""
        return await self._delete_by_id(memory_id, user_id, "procedural")

    @qdrant_retry
    async def _delete_by_id(
        self,
        memory_id: str,
        user_id: str,
        memory_type: str,
    ) -> bool:
        """Delete a memory by ID with user verification.

        Args:
            memory_id: Memory to delete.
            user_id: User ID for ownership check.
            memory_type: Collection type key.

        Returns:
            True if deleted, False if not found.
        """
        collection = self._collection_name(memory_type)

        result = await self.client.delete(
            collection_name=collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="id",
                            match=models.MatchValue(value=memory_id),
                        ),
                        models.FieldCondition(
                            key="user_id",
                            match=models.MatchValue(value=user_id),
                        ),
                    ]
                )
            ),
        )

        deleted: bool = result.status == models.UpdateStatus.COMPLETED
        return deleted

    async def delete_all_user_memories(
        self,
        user_id: str,
        org_id: str | None = None,
    ) -> dict[str, int]:
        """Delete all memories for a user (GDPR right to erasure).

        Deletes all memories across all collections for a given user.
        This is an irreversible operation intended for GDPR compliance.

        Args:
            user_id: User whose memories should be deleted.
            org_id: Optional org filter (if provided, only deletes within org).

        Returns:
            Dict mapping memory type to count of deleted memories.
        """
        deleted_counts: dict[str, int] = {}
        collections = ["episodic", "structured", "semantic", "procedural"]

        for memory_type in collections:
            collection = self._collection_name(memory_type)

            # Build filter
            must_conditions = [
                models.FieldCondition(
                    key="user_id",
                    match=models.MatchValue(value=user_id),
                ),
            ]
            if org_id is not None:
                must_conditions.append(
                    models.FieldCondition(
                        key="org_id",
                        match=models.MatchValue(value=org_id),
                    )
                )

            # Count before deletion
            count_result = await self.client.count(
                collection_name=collection,
                count_filter=models.Filter(must=must_conditions),
            )

            # Delete all matching
            await self.client.delete(
                collection_name=collection,
                points_selector=models.FilterSelector(filter=models.Filter(must=must_conditions)),
            )

            deleted_counts[memory_type] = count_result.count

        return deleted_counts

    async def get_unconsolidated_episodes(
        self,
        user_id: str,
        org_id: str | None = None,
        limit: int = 100,
    ) -> list[Episode]:
        """Get episodes that haven't been processed by consolidation.

        Args:
            user_id: User ID for isolation.
            org_id: Optional org ID filter.
            limit: Maximum episodes to return.

        Returns:
            List of unconsolidated Episodes.
        """
        from engram.models import Episode

        collection = self._collection_name("episodic")

        filters: list[models.FieldCondition] = [
            models.FieldCondition(
                key="user_id",
                match=models.MatchValue(value=user_id),
            ),
            models.FieldCondition(
                key="consolidated",
                match=models.MatchValue(value=False),
            ),
        ]

        if org_id is not None:
            filters.append(
                models.FieldCondition(
                    key="org_id",
                    match=models.MatchValue(value=org_id),
                )
            )

        results, _ = await self.client.scroll(
            collection_name=collection,
            scroll_filter=models.Filter(must=filters),
            limit=limit,
            with_payload=True,
            with_vectors=True,
        )

        episodes: list[Episode] = []
        for point in results:
            if point.payload is not None:
                ep = self._payload_to_memory(point.payload, Episode)
                # Restore the embedding from the vector
                if isinstance(point.vector, list):
                    ep.embedding = point.vector
                episodes.append(ep)

        return episodes

    async def mark_episodes_consolidated(
        self,
        episode_ids: list[str],
        user_id: str,
    ) -> int:
        """Mark episodes as consolidated.

        Args:
            episode_ids: IDs of episodes to mark.
            user_id: User ID for ownership verification.

        Returns:
            Number of episodes updated.
        """
        if not episode_ids:
            return 0

        collection = self._collection_name("episodic")

        # Batch fetch all episodes in a single query
        results, _ = await self.client.scroll(
            collection_name=collection,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="id",
                        match=models.MatchAny(any=episode_ids),
                    ),
                    models.FieldCondition(
                        key="user_id",
                        match=models.MatchValue(value=user_id),
                    ),
                ]
            ),
            limit=len(episode_ids),
            with_payload=True,
        )

        if not results:
            return 0

        # Batch update all points at once
        point_ids = [point.id for point in results]
        await self.client.set_payload(
            collection_name=collection,
            payload={"consolidated": True},
            points=point_ids,
        )

        return len(point_ids)

    async def list_structured_memories(
        self,
        user_id: str,
        org_id: str | None = None,
        with_negations_only: bool = False,
        limit: int = 1000,
    ) -> list[StructuredMemory]:
        """List all structured memories for a user.

        Args:
            user_id: User ID for isolation.
            org_id: Optional org ID filter.
            with_negations_only: If True, only return memories that have negations.
            limit: Maximum memories to return.

        Returns:
            List of StructuredMemory objects.
        """
        from engram.models import StructuredMemory

        collection = self._collection_name("structured")

        filters: list[models.FieldCondition] = [
            models.FieldCondition(
                key="user_id",
                match=models.MatchValue(value=user_id),
            ),
        ]

        if org_id is not None:
            filters.append(
                models.FieldCondition(
                    key="org_id",
                    match=models.MatchValue(value=org_id),
                )
            )

        results, _ = await self.client.scroll(
            collection_name=collection,
            scroll_filter=models.Filter(must=filters),
            limit=limit,
            with_payload=True,
            with_vectors=True,
        )

        memories: list[StructuredMemory] = []
        for point in results:
            if point.payload is not None:
                memory = self._payload_to_memory(point.payload, StructuredMemory)
                if isinstance(point.vector, list):
                    memory.embedding = point.vector
                # Filter for negations if requested
                if with_negations_only and not memory.has_negations():
                    continue
                memories.append(memory)

        return memories

    async def get_unstructured_episodes(
        self,
        user_id: str,
        org_id: str | None = None,
        limit: int | None = 100,
    ) -> list[Episode]:
        """Get episodes that haven't been processed into StructuredMemory.

        Used by the structure() workflow to find episodes needing processing.

        Args:
            user_id: User ID for isolation.
            org_id: Optional org ID filter.
            limit: Maximum episodes to return.

        Returns:
            List of unstructured Episodes, ordered by timestamp.
        """
        from engram.models import Episode

        collection = self._collection_name("episodic")

        filters: list[models.FieldCondition] = [
            models.FieldCondition(
                key="user_id",
                match=models.MatchValue(value=user_id),
            ),
            models.FieldCondition(
                key="structured",
                match=models.MatchValue(value=False),
            ),
        ]

        if org_id is not None:
            filters.append(
                models.FieldCondition(
                    key="org_id",
                    match=models.MatchValue(value=org_id),
                )
            )

        # Use a high limit if none specified to get all
        scroll_limit = limit if limit is not None else 10000

        results, _ = await self.client.scroll(
            collection_name=collection,
            scroll_filter=models.Filter(must=filters),
            limit=scroll_limit,
            with_payload=True,
            with_vectors=True,
        )

        episodes: list[Episode] = []
        for point in results:
            if point.payload is not None:
                ep = self._payload_to_memory(point.payload, Episode)
                if isinstance(point.vector, list):
                    ep.embedding = point.vector
                episodes.append(ep)

        # Sort by timestamp for consistent ordering
        episodes.sort(key=lambda e: e.timestamp)

        return episodes

    async def mark_episodes_structured(
        self,
        episode_ids: list[str],
        user_id: str,
        structured_id: str,
    ) -> int:
        """Mark episodes as structured and link to the StructuredMemory.

        Called after structure() creates a StructuredMemory from an episode.
        Sets `structured=True` and records which StructuredMemory was created.

        Args:
            episode_ids: IDs of episodes to mark.
            user_id: User ID for ownership verification.
            structured_id: ID of the StructuredMemory created from these episodes.

        Returns:
            Number of episodes updated.
        """
        if not episode_ids:
            return 0

        collection = self._collection_name("episodic")

        # Batch fetch all episodes in a single query
        results, _ = await self.client.scroll(
            collection_name=collection,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="id",
                        match=models.MatchAny(any=episode_ids),
                    ),
                    models.FieldCondition(
                        key="user_id",
                        match=models.MatchValue(value=user_id),
                    ),
                ]
            ),
            limit=len(episode_ids),
            with_payload=True,
        )

        if not results:
            return 0

        # Batch update all points at once
        point_ids = [point.id for point in results]
        await self.client.set_payload(
            collection_name=collection,
            payload={
                "structured": True,
                "structured_into": structured_id,
            },
            points=point_ids,
        )

        return len(point_ids)

    async def get_structured_for_episode(
        self,
        episode_id: str,
        user_id: str,
    ) -> StructuredMemory | None:
        """Get the StructuredMemory for a specific episode.

        Args:
            episode_id: The source episode ID.
            user_id: User ID for isolation.

        Returns:
            StructuredMemory if found, None otherwise.
        """
        from engram.models import StructuredMemory

        collection = self._collection_name("structured")

        results, _ = await self.client.scroll(
            collection_name=collection,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="source_episode_id",
                        match=models.MatchValue(value=episode_id),
                    ),
                    models.FieldCondition(
                        key="user_id",
                        match=models.MatchValue(value=user_id),
                    ),
                ]
            ),
            limit=1,
            with_payload=True,
            with_vectors=True,
        )

        if not results or results[0].payload is None:
            return None

        memory: StructuredMemory = self._payload_to_memory(results[0].payload, StructuredMemory)
        if isinstance(results[0].vector, list):
            memory.embedding = results[0].vector
        return memory

    async def get_unconsolidated_structured(
        self,
        user_id: str,
        org_id: str | None = None,
        limit: int | None = 100,
    ) -> list[StructuredMemory]:
        """Get StructuredMemories that haven't been consolidated into SemanticMemory.

        Used by the consolidation workflow to find structured extractions
        that need to be synthesized into semantic memories.

        Args:
            user_id: User ID for isolation.
            org_id: Optional org ID filter.
            limit: Maximum results (None for all).

        Returns:
            List of StructuredMemory objects that haven't been consolidated.
        """
        from engram.models import StructuredMemory

        collection = self._collection_name("structured")

        filters: list[models.FieldCondition] = [
            models.FieldCondition(
                key="user_id",
                match=models.MatchValue(value=user_id),
            ),
            models.FieldCondition(
                key="consolidated",
                match=models.MatchValue(value=False),
            ),
        ]

        if org_id is not None:
            filters.append(
                models.FieldCondition(
                    key="org_id",
                    match=models.MatchValue(value=org_id),
                )
            )

        scroll_limit = limit if limit is not None else 10000

        results, _ = await self.client.scroll(
            collection_name=collection,
            scroll_filter=models.Filter(must=filters),
            limit=scroll_limit,
            with_payload=True,
            with_vectors=True,
        )

        memories: list[StructuredMemory] = []
        for point in results:
            if point.payload is not None:
                mem = self._payload_to_memory(point.payload, StructuredMemory)
                if isinstance(point.vector, list):
                    mem.embedding = point.vector
                memories.append(mem)

        # Sort by derived_at for consistent ordering
        memories.sort(key=lambda m: m.derived_at)

        return memories

    async def mark_structured_consolidated(
        self,
        structured_ids: list[str],
        user_id: str,
        semantic_id: str,
    ) -> int:
        """Mark StructuredMemories as consolidated into a SemanticMemory.

        Args:
            structured_ids: List of StructuredMemory IDs to mark.
            user_id: User ID for isolation.
            semantic_id: ID of the SemanticMemory they were consolidated into.

        Returns:
            Number of memories updated.
        """
        if not structured_ids:
            return 0

        collection = self._collection_name("structured")

        # Batch fetch all structured memories in a single query
        results, _ = await self.client.scroll(
            collection_name=collection,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="id",
                        match=models.MatchAny(any=structured_ids),
                    ),
                    models.FieldCondition(
                        key="user_id",
                        match=models.MatchValue(value=user_id),
                    ),
                ]
            ),
            limit=len(structured_ids),
            with_payload=True,
        )

        if not results:
            return 0

        # Batch update all points at once
        point_ids = [point.id for point in results]
        await self.client.set_payload(
            collection_name=collection,
            payload={
                "consolidated": True,
                "consolidated_into": semantic_id,
            },
            points=point_ids,
        )

        return len(point_ids)

    async def list_semantic_memories(
        self,
        user_id: str,
        org_id: str | None = None,
        include_archived: bool = False,
        limit: int = 1000,
    ) -> list[SemanticMemory]:
        """List all semantic memories for a user.

        Args:
            user_id: User ID for isolation.
            org_id: Optional org ID filter.
            include_archived: Whether to include archived memories.
            limit: Maximum memories to return.

        Returns:
            List of SemanticMemory objects.
        """
        from engram.models import SemanticMemory

        collection = self._collection_name("semantic")

        filters: list[models.FieldCondition] = [
            models.FieldCondition(
                key="user_id",
                match=models.MatchValue(value=user_id),
            ),
        ]

        if org_id is not None:
            filters.append(
                models.FieldCondition(
                    key="org_id",
                    match=models.MatchValue(value=org_id),
                )
            )

        if not include_archived:
            filters.append(
                models.FieldCondition(
                    key="archived",
                    match=models.MatchValue(value=False),
                )
            )

        results, _ = await self.client.scroll(
            collection_name=collection,
            scroll_filter=models.Filter(must=filters),
            limit=limit,
            with_payload=True,
            with_vectors=True,
        )

        memories: list[SemanticMemory] = []
        for point in results:
            if point.payload is not None:
                memory = self._payload_to_memory(point.payload, SemanticMemory)
                if isinstance(point.vector, list):
                    memory.embedding = point.vector
                memories.append(memory)

        return memories

    async def list_procedural_memories(
        self,
        user_id: str,
        org_id: str | None = None,
        limit: int = 1000,
    ) -> list[ProceduralMemory]:
        """List all procedural memories for a user.

        Args:
            user_id: User ID for isolation.
            org_id: Optional org ID filter.
            limit: Maximum memories to return.

        Returns:
            List of ProceduralMemory objects.
        """
        from engram.models import ProceduralMemory

        collection = self._collection_name("procedural")

        filters: list[models.FieldCondition] = [
            models.FieldCondition(
                key="user_id",
                match=models.MatchValue(value=user_id),
            ),
        ]

        if org_id is not None:
            filters.append(
                models.FieldCondition(
                    key="org_id",
                    match=models.MatchValue(value=org_id),
                )
            )

        results, _ = await self.client.scroll(
            collection_name=collection,
            scroll_filter=models.Filter(must=filters),
            limit=limit,
            with_payload=True,
            with_vectors=True,
        )

        memories: list[ProceduralMemory] = []
        for point in results:
            if point.payload is not None:
                memory = self._payload_to_memory(point.payload, ProceduralMemory)
                if isinstance(point.vector, list):
                    memory.embedding = point.vector
                memories.append(memory)

        return memories

    @qdrant_retry
    async def update_semantic_memory(
        self,
        memory: SemanticMemory,
    ) -> bool:
        """Update a semantic memory.

        Args:
            memory: SemanticMemory with updated fields.

        Returns:
            True if updated, False if not found.
        """
        collection = self._collection_name("semantic")

        # Find the point
        results, _ = await self.client.scroll(
            collection_name=collection,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="id",
                        match=models.MatchValue(value=memory.id),
                    ),
                    models.FieldCondition(
                        key="user_id",
                        match=models.MatchValue(value=memory.user_id),
                    ),
                ]
            ),
            limit=1,
            with_payload=True,
        )

        if not results:
            return False

        point = results[0]
        payload = self._memory_to_payload(memory)

        await self.client.set_payload(
            collection_name=collection,
            payload=payload,
            points=[point.id],
        )

        return True

    @qdrant_retry
    async def update_procedural_memory(
        self,
        memory: ProceduralMemory,
    ) -> bool:
        """Update a procedural memory.

        Args:
            memory: ProceduralMemory with updated fields.

        Returns:
            True if updated, False if not found.
        """
        collection = self._collection_name("procedural")

        # Find the point
        results, _ = await self.client.scroll(
            collection_name=collection,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="id",
                        match=models.MatchValue(value=memory.id),
                    ),
                    models.FieldCondition(
                        key="user_id",
                        match=models.MatchValue(value=memory.user_id),
                    ),
                ]
            ),
            limit=1,
            with_payload=True,
        )

        if not results:
            return False

        point = results[0]
        payload = self._memory_to_payload(memory)

        await self.client.set_payload(
            collection_name=collection,
            payload=payload,
            points=[point.id],
        )

        return True

    @qdrant_retry
    async def update_structured_memory(
        self,
        memory: StructuredMemory,
    ) -> bool:
        """Update a structured memory.

        Args:
            memory: StructuredMemory with updated fields.

        Returns:
            True if updated, False if not found.
        """
        collection = self._collection_name("structured")

        # Find the point
        results, _ = await self.client.scroll(
            collection_name=collection,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="id",
                        match=models.MatchValue(value=memory.id),
                    ),
                    models.FieldCondition(
                        key="user_id",
                        match=models.MatchValue(value=memory.user_id),
                    ),
                ]
            ),
            limit=1,
            with_payload=True,
        )

        if not results:
            return False

        point = results[0]
        payload = self._memory_to_payload(memory)

        await self.client.set_payload(
            collection_name=collection,
            payload=payload,
            points=[point.id],
        )

        return True

    async def get_unsummarized_episodes(
        self,
        user_id: str,
        org_id: str | None = None,
        limit: int | None = None,
    ) -> list[Episode]:
        """Get episodes that haven't been included in a semantic summary.

        Used by consolidation to find episodes that need summarization.
        The `summarized` field tracks whether an episode has been
        compressed into a semantic memory.

        Args:
            user_id: User ID for isolation.
            org_id: Optional org ID filter.
            limit: Maximum episodes to return. None means all.

        Returns:
            List of unsummarized Episodes, ordered by timestamp.
        """
        from engram.models import Episode

        collection = self._collection_name("episodic")

        filters: list[models.FieldCondition] = [
            models.FieldCondition(
                key="user_id",
                match=models.MatchValue(value=user_id),
            ),
            models.FieldCondition(
                key="summarized",
                match=models.MatchValue(value=False),
            ),
        ]

        if org_id is not None:
            filters.append(
                models.FieldCondition(
                    key="org_id",
                    match=models.MatchValue(value=org_id),
                )
            )

        # Use a high limit if none specified to get all
        scroll_limit = limit if limit is not None else 10000

        results, _ = await self.client.scroll(
            collection_name=collection,
            scroll_filter=models.Filter(must=filters),
            limit=scroll_limit,
            with_payload=True,
            with_vectors=True,
        )

        episodes: list[Episode] = []
        for point in results:
            if point.payload is not None:
                ep = self._payload_to_memory(point.payload, Episode)
                # Restore the embedding from the vector
                if isinstance(point.vector, list):
                    ep.embedding = point.vector
                episodes.append(ep)

        # Sort by timestamp for consistent ordering
        episodes.sort(key=lambda e: e.timestamp)

        return episodes

    async def mark_episodes_summarized(
        self,
        episode_ids: list[str],
        user_id: str,
        semantic_id: str,
    ) -> int:
        """Mark episodes as summarized and link to the semantic memory.

        Called after consolidation creates a semantic summary from
        a batch of episodes. Sets `summarized=True` and records
        which semantic memory contains the summary.

        Args:
            episode_ids: IDs of episodes to mark.
            user_id: User ID for ownership verification.
            semantic_id: ID of the semantic memory that summarizes these episodes.

        Returns:
            Number of episodes updated.
        """
        if not episode_ids:
            return 0

        collection = self._collection_name("episodic")

        # Batch fetch all episodes in a single query
        results, _ = await self.client.scroll(
            collection_name=collection,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="id",
                        match=models.MatchAny(any=episode_ids),
                    ),
                    models.FieldCondition(
                        key="user_id",
                        match=models.MatchValue(value=user_id),
                    ),
                ]
            ),
            limit=len(episode_ids),
            with_payload=True,
        )

        if not results:
            return 0

        # Batch update all points at once
        point_ids = [point.id for point in results]
        await self.client.set_payload(
            collection_name=collection,
            payload={
                "summarized": True,
                "summarized_into": semantic_id,
            },
            points=point_ids,
        )

        return len(point_ids)

    async def update_episode(
        self,
        episode: Episode,
    ) -> bool:
        """Update an episode's mutable fields.

        Note: Episode content is immutable. This only updates metadata
        fields like `consolidated`, `summarized`, `summarized_into`.

        Args:
            episode: Episode with updated fields.

        Returns:
            True if updated, False if not found.
        """
        collection = self._collection_name("episodic")

        # Find the point
        results, _ = await self.client.scroll(
            collection_name=collection,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="id",
                        match=models.MatchValue(value=episode.id),
                    ),
                    models.FieldCondition(
                        key="user_id",
                        match=models.MatchValue(value=episode.user_id),
                    ),
                ]
            ),
            limit=1,
            with_payload=True,
        )

        if not results:
            return False

        point = results[0]
        payload = self._memory_to_payload(episode)

        await self.client.set_payload(
            collection_name=collection,
            payload=payload,
            points=[point.id],
        )

        return True

    async def search_memories(
        self,
        user_id: str,
        org_id: str | None = None,
        session_id: str | None = None,
        memory_types: list[str] | None = None,
        created_after: datetime | None = None,
        created_before: datetime | None = None,
        min_confidence: float | None = None,
        sort_by: str = "created_at",
        sort_order: str = "desc",
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[dict[str, object]], int]:
        """Search memories by metadata filters.

        Supports filtering across all memory types without semantic search.
        Results are sorted and paginated in Python after retrieval.

        Args:
            user_id: User ID for multi-tenancy isolation (required).
            org_id: Optional organization ID filter.
            session_id: Filter by session ID (episodes only).
            memory_types: Filter by memory types. Options: episodic, structured,
                semantic, procedural. None means all types.
            created_after: Filter to memories created after this timestamp.
            created_before: Filter to memories created before this timestamp.
            min_confidence: Minimum confidence score (derived memories only).
            sort_by: Sort field ("created_at" or "confidence").
            sort_order: Sort direction ("asc" or "desc").
            limit: Maximum results to return (after offset).
            offset: Number of results to skip.

        Returns:
            Tuple of (list of memory dicts, total count before pagination).
        """
        # Determine which memory types to search
        types_to_search = memory_types or ["episodic", "structured", "semantic", "procedural"]

        all_memories: list[dict[str, object]] = []

        # Search each collection
        for mem_type in types_to_search:
            if mem_type == "episodic":
                memories = await self._search_episodic(
                    user_id=user_id,
                    org_id=org_id,
                    session_id=session_id,
                    created_after=created_after,
                    created_before=created_before,
                )
                all_memories.extend(memories)

            elif mem_type == "structured":
                memories = await self._search_structured(
                    user_id=user_id,
                    org_id=org_id,
                    created_after=created_after,
                    created_before=created_before,
                    min_confidence=min_confidence,
                )
                all_memories.extend(memories)

            elif mem_type == "semantic":
                memories = await self._search_semantic(
                    user_id=user_id,
                    org_id=org_id,
                    created_after=created_after,
                    created_before=created_before,
                    min_confidence=min_confidence,
                )
                all_memories.extend(memories)

            elif mem_type == "procedural":
                memories = await self._search_procedural(
                    user_id=user_id,
                    org_id=org_id,
                    created_after=created_after,
                    created_before=created_before,
                    min_confidence=min_confidence,
                )
                all_memories.extend(memories)

        # Sort results
        reverse = sort_order == "desc"
        if sort_by == "confidence":
            # Sort by confidence (None values go to end)
            def confidence_key(m: dict[str, object]) -> tuple[bool, float]:
                conf = m.get("confidence")
                return (conf is None, float(conf) if conf is not None else 0.0)  # type: ignore[arg-type]

            all_memories.sort(key=confidence_key, reverse=reverse)
        else:
            # Sort by created_at (default)
            def created_at_key(m: dict[str, object]) -> str:
                val = m.get("created_at")
                return str(val) if val is not None else ""

            all_memories.sort(key=created_at_key, reverse=reverse)

        # Calculate total before pagination
        total = len(all_memories)

        # Apply pagination
        paginated = all_memories[offset : offset + limit]

        return paginated, total

    async def _search_episodic(
        self,
        user_id: str,
        org_id: str | None = None,
        session_id: str | None = None,
        created_after: datetime | None = None,
        created_before: datetime | None = None,
    ) -> list[dict[str, object]]:
        """Search episodic memories with filters."""
        from engram.models import Episode

        collection = self._collection_name("episodic")

        filters: list[models.FieldCondition] = [
            models.FieldCondition(
                key="user_id",
                match=models.MatchValue(value=user_id),
            ),
        ]

        if org_id is not None:
            filters.append(
                models.FieldCondition(
                    key="org_id",
                    match=models.MatchValue(value=org_id),
                )
            )

        if session_id is not None:
            filters.append(
                models.FieldCondition(
                    key="session_id",
                    match=models.MatchValue(value=session_id),
                )
            )

        results, _ = await self.client.scroll(
            collection_name=collection,
            scroll_filter=models.Filter(must=filters),
            limit=settings.storage_max_scroll_limit,  # Configurable via ENGRAM_STORAGE_MAX_SCROLL_LIMIT
            with_payload=True,
            with_vectors=False,
        )

        memories: list[dict[str, object]] = []
        for point in results:
            if point.payload is not None:
                episode = self._payload_to_memory(point.payload, Episode)
                timestamp_str = episode.timestamp.isoformat()

                # Apply date filters
                if created_after and episode.timestamp < created_after:
                    continue
                if created_before and episode.timestamp > created_before:
                    continue

                memories.append(
                    {
                        "id": episode.id,
                        "memory_type": "episodic",
                        "content": episode.content,
                        "user_id": episode.user_id,
                        "org_id": episode.org_id,
                        "session_id": episode.session_id,
                        "confidence": None,
                        "created_at": timestamp_str,
                    }
                )

        return memories

    async def _search_structured(
        self,
        user_id: str,
        org_id: str | None = None,
        created_after: datetime | None = None,
        created_before: datetime | None = None,
        min_confidence: float | None = None,
    ) -> list[dict[str, object]]:
        """Search structured memories with filters."""
        from engram.models import StructuredMemory

        collection = self._collection_name("structured")

        filters: list[models.FieldCondition] = [
            models.FieldCondition(
                key="user_id",
                match=models.MatchValue(value=user_id),
            ),
        ]

        if org_id is not None:
            filters.append(
                models.FieldCondition(
                    key="org_id",
                    match=models.MatchValue(value=org_id),
                )
            )

        results, _ = await self.client.scroll(
            collection_name=collection,
            scroll_filter=models.Filter(must=filters),
            limit=settings.storage_max_scroll_limit,
            with_payload=True,
            with_vectors=False,
        )

        memories: list[dict[str, object]] = []
        for point in results:
            if point.payload is not None:
                memory = self._payload_to_memory(point.payload, StructuredMemory)
                timestamp_str = memory.derived_at.isoformat()

                # Apply date filters
                if created_after and memory.derived_at < created_after:
                    continue
                if created_before and memory.derived_at > created_before:
                    continue

                # Apply confidence filter
                if min_confidence and memory.confidence.value < min_confidence:
                    continue

                memories.append(
                    {
                        "id": memory.id,
                        "memory_type": "structured",
                        "content": memory.summary or "",
                        "user_id": memory.user_id,
                        "org_id": memory.org_id,
                        "session_id": None,
                        "confidence": memory.confidence.value,
                        "created_at": timestamp_str,
                    }
                )

        return memories

    async def _search_semantic(
        self,
        user_id: str,
        org_id: str | None = None,
        created_after: datetime | None = None,
        created_before: datetime | None = None,
        min_confidence: float | None = None,
    ) -> list[dict[str, object]]:
        """Search semantic memories with filters."""
        from engram.models import SemanticMemory

        collection = self._collection_name("semantic")

        filters: list[models.FieldCondition] = [
            models.FieldCondition(
                key="user_id",
                match=models.MatchValue(value=user_id),
            ),
        ]

        if org_id is not None:
            filters.append(
                models.FieldCondition(
                    key="org_id",
                    match=models.MatchValue(value=org_id),
                )
            )

        # Exclude archived by default
        filters.append(
            models.FieldCondition(
                key="archived",
                match=models.MatchValue(value=False),
            )
        )

        results, _ = await self.client.scroll(
            collection_name=collection,
            scroll_filter=models.Filter(must=filters),
            limit=settings.storage_max_scroll_limit,
            with_payload=True,
            with_vectors=False,
        )

        memories: list[dict[str, object]] = []
        for point in results:
            if point.payload is not None:
                memory = self._payload_to_memory(point.payload, SemanticMemory)
                timestamp_str = memory.derived_at.isoformat()

                # Apply date filters
                if created_after and memory.derived_at < created_after:
                    continue
                if created_before and memory.derived_at > created_before:
                    continue

                # Apply confidence filter
                if min_confidence and memory.confidence.value < min_confidence:
                    continue

                memories.append(
                    {
                        "id": memory.id,
                        "memory_type": "semantic",
                        "content": memory.content,
                        "user_id": memory.user_id,
                        "org_id": memory.org_id,
                        "session_id": None,
                        "confidence": memory.confidence.value,
                        "created_at": timestamp_str,
                    }
                )

        return memories

    async def _search_procedural(
        self,
        user_id: str,
        org_id: str | None = None,
        created_after: datetime | None = None,
        created_before: datetime | None = None,
        min_confidence: float | None = None,
    ) -> list[dict[str, object]]:
        """Search procedural memories with filters."""
        from engram.models import ProceduralMemory

        collection = self._collection_name("procedural")

        filters: list[models.FieldCondition] = [
            models.FieldCondition(
                key="user_id",
                match=models.MatchValue(value=user_id),
            ),
        ]

        if org_id is not None:
            filters.append(
                models.FieldCondition(
                    key="org_id",
                    match=models.MatchValue(value=org_id),
                )
            )

        results, _ = await self.client.scroll(
            collection_name=collection,
            scroll_filter=models.Filter(must=filters),
            limit=settings.storage_max_scroll_limit,
            with_payload=True,
            with_vectors=False,
        )

        memories: list[dict[str, object]] = []
        for point in results:
            if point.payload is not None:
                memory = self._payload_to_memory(point.payload, ProceduralMemory)
                timestamp_str = memory.derived_at.isoformat()

                # Apply date filters
                if created_after and memory.derived_at < created_after:
                    continue
                if created_before and memory.derived_at > created_before:
                    continue

                # Apply confidence filter
                if min_confidence and memory.confidence.value < min_confidence:
                    continue

                memories.append(
                    {
                        "id": memory.id,
                        "memory_type": "procedural",
                        "content": memory.content,
                        "user_id": memory.user_id,
                        "org_id": memory.org_id,
                        "session_id": None,
                        "confidence": memory.confidence.value,
                        "created_at": timestamp_str,
                    }
                )

        return memories

    # ========================================================================
    # Session Management
    # ========================================================================

    async def list_sessions(
        self,
        user_id: str,
        org_id: str | None = None,
    ) -> list[dict[str, object]]:
        """List all sessions for a user with episode counts and date ranges.

        Args:
            user_id: User ID for isolation.
            org_id: Optional org ID filter.

        Returns:
            List of session summaries with session_id, episode_count,
            first_episode_at, and last_episode_at.
        """
        from engram.models import Episode

        collection = self._collection_name("episodic")

        filters: list[models.FieldCondition] = [
            models.FieldCondition(
                key="user_id",
                match=models.MatchValue(value=user_id),
            ),
        ]

        if org_id is not None:
            filters.append(
                models.FieldCondition(
                    key="org_id",
                    match=models.MatchValue(value=org_id),
                )
            )

        results, _ = await self.client.scroll(
            collection_name=collection,
            scroll_filter=models.Filter(must=filters),
            limit=settings.storage_max_scroll_limit,
            with_payload=True,
            with_vectors=False,
        )

        # Group episodes by session_id
        sessions: dict[str, list[Episode]] = {}
        for point in results:
            if point.payload is not None:
                episode = self._payload_to_memory(point.payload, Episode)
                if episode.session_id is not None:
                    if episode.session_id not in sessions:
                        sessions[episode.session_id] = []
                    sessions[episode.session_id].append(episode)

        # Build session summaries
        session_list: list[dict[str, object]] = []
        for session_id, episodes in sessions.items():
            episodes.sort(key=lambda e: e.timestamp)
            session_list.append(
                {
                    "session_id": session_id,
                    "episode_count": len(episodes),
                    "first_episode_at": episodes[0].timestamp.isoformat(),
                    "last_episode_at": episodes[-1].timestamp.isoformat(),
                }
            )

        # Sort by last_episode_at descending (most recent first)
        session_list.sort(key=lambda s: str(s["last_episode_at"]), reverse=True)

        return session_list

    async def get_session_episodes(
        self,
        session_id: str,
        user_id: str,
        org_id: str | None = None,
    ) -> list[Episode]:
        """Get all episodes for a specific session.

        Args:
            session_id: Session identifier.
            user_id: User ID for isolation.
            org_id: Optional org ID filter.

        Returns:
            List of Episodes in the session, ordered by timestamp.
        """
        from engram.models import Episode

        collection = self._collection_name("episodic")

        filters: list[models.FieldCondition] = [
            models.FieldCondition(
                key="user_id",
                match=models.MatchValue(value=user_id),
            ),
            models.FieldCondition(
                key="session_id",
                match=models.MatchValue(value=session_id),
            ),
        ]

        if org_id is not None:
            filters.append(
                models.FieldCondition(
                    key="org_id",
                    match=models.MatchValue(value=org_id),
                )
            )

        results, _ = await self.client.scroll(
            collection_name=collection,
            scroll_filter=models.Filter(must=filters),
            limit=settings.storage_max_scroll_limit,
            with_payload=True,
            with_vectors=True,
        )

        episodes: list[Episode] = []
        for point in results:
            if point.payload is not None:
                episode = self._payload_to_memory(point.payload, Episode)
                if isinstance(point.vector, list):
                    episode.embedding = point.vector
                episodes.append(episode)

        # Sort by timestamp
        episodes.sort(key=lambda e: e.timestamp)

        return episodes

    async def delete_session(
        self,
        session_id: str,
        user_id: str,
        org_id: str | None = None,
        cascade: str = "soft",
    ) -> dict[str, object]:
        """Delete all episodes in a session with optional cascade.

        Args:
            session_id: Session identifier.
            user_id: User ID for isolation.
            org_id: Optional org ID filter.
            cascade: Cascade mode:
                - "none": Delete only episodes, leave derived memories orphaned
                - "soft": Remove episode references, reduce confidence, delete empty
                - "hard": Delete all derived memories that reference session episodes

        Returns:
            Dict with deletion statistics.
        """
        # Get all episodes in the session
        episodes = await self.get_session_episodes(session_id, user_id, org_id)

        if not episodes:
            return {
                "deleted": False,
                "episodes_deleted": 0,
                "structured_deleted": 0,
                "semantic_deleted": 0,
                "semantic_updated": 0,
                "procedural_deleted": 0,
                "procedural_updated": 0,
            }

        # Track cascade statistics
        total_structured_deleted = 0
        total_semantic_deleted = 0
        total_semantic_updated = 0
        total_procedural_deleted = 0
        total_procedural_updated = 0

        # Delete each episode with cascade
        for episode in episodes:
            result = await self.delete_episode(episode.id, user_id, cascade=cascade)
            if result.get("deleted"):
                # Cast values from dict[str, object] to int
                structured = result.get("structured_deleted", 0)
                semantic_del = result.get("semantic_deleted", 0)
                semantic_upd = result.get("semantic_updated", 0)
                procedural_del = result.get("procedural_deleted", 0)
                procedural_upd = result.get("procedural_updated", 0)
                total_structured_deleted += structured if isinstance(structured, int) else 0
                total_semantic_deleted += semantic_del if isinstance(semantic_del, int) else 0
                total_semantic_updated += semantic_upd if isinstance(semantic_upd, int) else 0
                total_procedural_deleted += procedural_del if isinstance(procedural_del, int) else 0
                total_procedural_updated += procedural_upd if isinstance(procedural_upd, int) else 0

        return {
            "deleted": True,
            "episodes_deleted": len(episodes),
            "structured_deleted": total_structured_deleted,
            "semantic_deleted": total_semantic_deleted,
            "semantic_updated": total_semantic_updated,
            "procedural_deleted": total_procedural_deleted,
            "procedural_updated": total_procedural_updated,
        }
