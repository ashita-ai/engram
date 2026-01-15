"""Get and delete operations for Engram storage.

Provides methods to retrieve and delete individual memories.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

from qdrant_client import models

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

    async def delete_episode(self, episode_id: str, user_id: str) -> bool:
        """Delete an episode.

        Args:
            episode_id: Episode to delete.
            user_id: User ID for ownership verification.

        Returns:
            True if deleted, False if not found.
        """
        return await self._delete_by_id(episode_id, user_id, "episodic")

    async def delete_structured(self, memory_id: str, user_id: str) -> bool:
        """Delete a structured memory."""
        return await self._delete_by_id(memory_id, user_id, "structured")

    async def delete_semantic(self, memory_id: str, user_id: str) -> bool:
        """Delete a semantic memory."""
        return await self._delete_by_id(memory_id, user_id, "semantic")

    async def delete_procedural(self, memory_id: str, user_id: str) -> bool:
        """Delete a procedural memory."""
        return await self._delete_by_id(memory_id, user_id, "procedural")

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
        collection = self._collection_name("episodic")
        updated = 0

        for episode_id in episode_ids:
            # Find and update each episode
            results, _ = await self.client.scroll(
                collection_name=collection,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="id",
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
            )

            if results:
                point = results[0]
                await self.client.set_payload(
                    collection_name=collection,
                    payload={"consolidated": True},
                    points=[point.id],
                )
                updated += 1

        return updated

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
        collection = self._collection_name("episodic")
        updated = 0

        for episode_id in episode_ids:
            # Find and update each episode
            results, _ = await self.client.scroll(
                collection_name=collection,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="id",
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
            )

            if results:
                point = results[0]
                await self.client.set_payload(
                    collection_name=collection,
                    payload={
                        "structured": True,
                        "structured_into": structured_id,
                    },
                    points=[point.id],
                )
                updated += 1

        return updated

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
        collection = self._collection_name("structured")
        updated = 0

        for struct_id in structured_ids:
            results, _ = await self.client.scroll(
                collection_name=collection,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="id",
                            match=models.MatchValue(value=struct_id),
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

            if results:
                point = results[0]
                await self.client.set_payload(
                    collection_name=collection,
                    payload={
                        "consolidated": True,
                        "consolidated_into": semantic_id,
                    },
                    points=[point.id],
                )
                updated += 1

        return updated

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
        collection = self._collection_name("episodic")
        updated = 0

        for episode_id in episode_ids:
            # Find and update each episode
            results, _ = await self.client.scroll(
                collection_name=collection,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="id",
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
            )

            if results:
                point = results[0]
                await self.client.set_payload(
                    collection_name=collection,
                    payload={
                        "summarized": True,
                        "summarized_into": semantic_id,
                    },
                    points=[point.id],
                )
                updated += 1

        return updated

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
