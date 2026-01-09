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
        Fact,
        NegationFact,
        ProceduralMemory,
        SemanticMemory,
    )

MemoryT = TypeVar(
    "MemoryT",
    "Episode",
    "Fact",
    "SemanticMemory",
    "ProceduralMemory",
    "NegationFact",
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

    async def get_fact(self, fact_id: str, user_id: str) -> Fact | None:
        """Get a fact by ID."""
        from engram.models import Fact

        return await self._get_by_id(fact_id, user_id, "factual", Fact)

    async def get_semantic(self, memory_id: str, user_id: str) -> SemanticMemory | None:
        """Get a semantic memory by ID."""
        from engram.models import SemanticMemory

        return await self._get_by_id(memory_id, user_id, "semantic", SemanticMemory)

    async def get_procedural(self, memory_id: str, user_id: str) -> ProceduralMemory | None:
        """Get a procedural memory by ID."""
        from engram.models import ProceduralMemory

        return await self._get_by_id(memory_id, user_id, "procedural", ProceduralMemory)

    async def get_negation(self, fact_id: str, user_id: str) -> NegationFact | None:
        """Get a negation fact by ID."""
        from engram.models import NegationFact

        return await self._get_by_id(fact_id, user_id, "negation", NegationFact)

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

    async def delete_fact(self, fact_id: str, user_id: str) -> bool:
        """Delete a fact."""
        return await self._delete_by_id(fact_id, user_id, "factual")

    async def delete_semantic(self, memory_id: str, user_id: str) -> bool:
        """Delete a semantic memory."""
        return await self._delete_by_id(memory_id, user_id, "semantic")

    async def delete_procedural(self, memory_id: str, user_id: str) -> bool:
        """Delete a procedural memory."""
        return await self._delete_by_id(memory_id, user_id, "procedural")

    async def delete_negation(self, fact_id: str, user_id: str) -> bool:
        """Delete a negation fact."""
        return await self._delete_by_id(fact_id, user_id, "negation")

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

    async def list_negation_facts(
        self,
        user_id: str,
        org_id: str | None = None,
        limit: int = 1000,
    ) -> list[NegationFact]:
        """List all negation facts for a user.

        Used for filtering during recall to exclude memories that
        match negated patterns.

        Args:
            user_id: User ID for isolation.
            org_id: Optional org ID filter.
            limit: Maximum facts to return.

        Returns:
            List of NegationFact objects.
        """
        from engram.models import NegationFact

        collection = self._collection_name("negation")

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

        facts: list[NegationFact] = []
        for point in results:
            if point.payload is not None:
                fact = self._payload_to_memory(point.payload, NegationFact)
                if isinstance(point.vector, list):
                    fact.embedding = point.vector
                facts.append(fact)

        return facts

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

    async def update_fact(
        self,
        fact: Fact,
    ) -> bool:
        """Update a fact.

        Args:
            fact: Fact with updated fields.

        Returns:
            True if updated, False if not found.
        """
        collection = self._collection_name("factual")

        # Find the point
        results, _ = await self.client.scroll(
            collection_name=collection,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="id",
                        match=models.MatchValue(value=fact.id),
                    ),
                    models.FieldCondition(
                        key="user_id",
                        match=models.MatchValue(value=fact.user_id),
                    ),
                ]
            ),
            limit=1,
            with_payload=True,
        )

        if not results:
            return False

        point = results[0]
        payload = self._memory_to_payload(fact)

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

    async def update_negation_fact(
        self,
        negation: NegationFact,
    ) -> bool:
        """Update a negation fact.

        Args:
            negation: NegationFact with updated fields.

        Returns:
            True if updated, False if not found.
        """
        collection = self._collection_name("negation")

        # Find the point
        results, _ = await self.client.scroll(
            collection_name=collection,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="id",
                        match=models.MatchValue(value=negation.id),
                    ),
                    models.FieldCondition(
                        key="user_id",
                        match=models.MatchValue(value=negation.user_id),
                    ),
                ]
            ),
            limit=1,
            with_payload=True,
        )

        if not results:
            return False

        point = results[0]
        payload = self._memory_to_payload(negation)

        await self.client.set_payload(
            collection_name=collection,
            payload=payload,
            points=[point.id],
        )

        return True
