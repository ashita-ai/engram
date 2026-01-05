"""Store operations for Engram storage.

Provides methods to store memories in Qdrant collections.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from qdrant_client import models

if TYPE_CHECKING:
    from engram.models import Episode, Fact, NegationFact, ProceduralMemory, SemanticMemory


class StoreMixin:
    """Mixin providing store operations for EngramStorage.

    This mixin expects the following attributes/methods from the base class:
    - _collection_name(memory_type) -> str
    - _build_key(memory_id, user_id, org_id) -> str
    - _key_to_point_id(key) -> str
    - _memory_to_payload(memory) -> dict
    - client: AsyncQdrantClient
    """

    # These will be provided by the base class
    _collection_name: Any
    _build_key: Any
    _key_to_point_id: Any
    _memory_to_payload: Any
    client: Any

    async def store_episode(self, episode: Episode) -> str:
        """Store an episode in the episodic collection.

        Args:
            episode: Episode to store.

        Returns:
            The episode ID.

        Raises:
            ValueError: If episode has no embedding.
        """
        return await self._store_memory(episode, "episode")

    async def store_fact(self, fact: Fact) -> str:
        """Store a fact in the factual collection.

        Args:
            fact: Fact to store.

        Returns:
            The fact ID.
        """
        return await self._store_memory(fact, "fact")

    async def store_semantic(self, memory: SemanticMemory) -> str:
        """Store a semantic memory.

        Args:
            memory: SemanticMemory to store.

        Returns:
            The memory ID.
        """
        return await self._store_memory(memory, "semantic")

    async def store_procedural(self, memory: ProceduralMemory) -> str:
        """Store a procedural memory.

        Args:
            memory: ProceduralMemory to store.

        Returns:
            The memory ID.
        """
        return await self._store_memory(memory, "procedural")

    async def store_negation(self, fact: NegationFact) -> str:
        """Store a negation fact.

        Args:
            fact: NegationFact to store.

        Returns:
            The fact ID.
        """
        return await self._store_memory(fact, "negation")

    async def _store_memory(
        self,
        memory: Episode | Fact | SemanticMemory | ProceduralMemory | NegationFact,
        memory_type: str,
    ) -> str:
        """Store a memory in the appropriate collection.

        Args:
            memory: Memory model to store.
            memory_type: Type key for collection lookup.

        Returns:
            The memory ID.

        Raises:
            ValueError: If memory has no embedding.
        """
        if memory.embedding is None:
            raise ValueError(f"{memory_type} must have an embedding before storage")

        collection = self._collection_name(memory_type)
        key = self._build_key(memory.id, memory.user_id, memory.org_id)
        payload = self._memory_to_payload(memory)

        await self.client.upsert(
            collection_name=collection,
            points=[
                models.PointStruct(
                    id=self._key_to_point_id(key),
                    vector=memory.embedding,
                    payload=payload,
                )
            ],
        )

        return memory.id
