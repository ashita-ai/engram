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
        InhibitoryFact,
        ProceduralMemory,
        SemanticMemory,
    )

MemoryT = TypeVar(
    "MemoryT",
    "Episode",
    "Fact",
    "SemanticMemory",
    "ProceduralMemory",
    "InhibitoryFact",
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

        return await self._get_by_id(episode_id, user_id, "episode", Episode)

    async def get_fact(self, fact_id: str, user_id: str) -> Fact | None:
        """Get a fact by ID."""
        from engram.models import Fact

        return await self._get_by_id(fact_id, user_id, "fact", Fact)

    async def get_semantic(self, memory_id: str, user_id: str) -> SemanticMemory | None:
        """Get a semantic memory by ID."""
        from engram.models import SemanticMemory

        return await self._get_by_id(memory_id, user_id, "semantic", SemanticMemory)

    async def get_procedural(self, memory_id: str, user_id: str) -> ProceduralMemory | None:
        """Get a procedural memory by ID."""
        from engram.models import ProceduralMemory

        return await self._get_by_id(memory_id, user_id, "procedural", ProceduralMemory)

    async def get_inhibitory(self, fact_id: str, user_id: str) -> InhibitoryFact | None:
        """Get an inhibitory fact by ID."""
        from engram.models import InhibitoryFact

        return await self._get_by_id(fact_id, user_id, "inhibitory", InhibitoryFact)

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
        return await self._delete_by_id(episode_id, user_id, "episode")

    async def delete_fact(self, fact_id: str, user_id: str) -> bool:
        """Delete a fact."""
        return await self._delete_by_id(fact_id, user_id, "fact")

    async def delete_semantic(self, memory_id: str, user_id: str) -> bool:
        """Delete a semantic memory."""
        return await self._delete_by_id(memory_id, user_id, "semantic")

    async def delete_procedural(self, memory_id: str, user_id: str) -> bool:
        """Delete a procedural memory."""
        return await self._delete_by_id(memory_id, user_id, "procedural")

    async def delete_inhibitory(self, fact_id: str, user_id: str) -> bool:
        """Delete an inhibitory fact."""
        return await self._delete_by_id(fact_id, user_id, "inhibitory")

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
