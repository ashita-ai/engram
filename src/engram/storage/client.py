"""Qdrant storage client for Engram memory system."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient, models

from engram.config import settings
from engram.models import (
    AuditEntry,
    Episode,
    Fact,
    InhibitoryFact,
    ProceduralMemory,
    SemanticMemory,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

# Type variable for memory models
MemoryT = TypeVar(
    "MemoryT",
    Episode,
    Fact,
    SemanticMemory,
    ProceduralMemory,
    InhibitoryFact,
    AuditEntry,
)

# Collection names by memory type
COLLECTION_NAMES = {
    "episode": "episodic",
    "fact": "factual",
    "semantic": "semantic",
    "procedural": "procedural",
    "inhibitory": "inhibitory",
    "audit": "audit",
}

# Default embedding dimension (text-embedding-3-small)
DEFAULT_EMBEDDING_DIM = 1536


class EngramStorage:
    """Async Qdrant storage client for Engram memories.

    Handles collection management, multi-tenancy isolation, and CRUD operations
    for all memory types. Uses async Qdrant client for non-blocking I/O.

    Attributes:
        client: Async Qdrant client instance.
        prefix: Collection name prefix (default: "engram").
        embedding_dim: Dimension of embedding vectors.

    Example:
        ```python
        storage = EngramStorage()
        await storage.initialize()

        # Store an episode
        episode = Episode(content="Hello", role="user", user_id="user_123")
        await storage.store_episode(episode)

        # Search for memories
        results = await storage.search_episodes(
            query_vector=[0.1, 0.2, ...],
            user_id="user_123",
            limit=10,
        )
        ```
    """

    def __init__(
        self,
        url: str | None = None,
        api_key: str | None = None,
        prefix: str | None = None,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    ) -> None:
        """Initialize storage client.

        Args:
            url: Qdrant server URL. Defaults to settings.qdrant_url.
            api_key: Qdrant API key. Defaults to settings.qdrant_api_key.
            prefix: Collection name prefix. Defaults to settings.collection_prefix.
            embedding_dim: Embedding vector dimension. Defaults to 1536.
        """
        self._url = url or settings.qdrant_url
        self._api_key = api_key or settings.qdrant_api_key
        self._prefix = prefix or settings.collection_prefix
        self._embedding_dim = embedding_dim
        self._client: AsyncQdrantClient | None = None
        self._collections_initialized = False

    @property
    def client(self) -> AsyncQdrantClient:
        """Get the Qdrant client, raising if not initialized."""
        if self._client is None:
            raise RuntimeError("Storage not initialized. Call initialize() first.")
        return self._client

    async def initialize(self) -> None:
        """Initialize the storage client and ensure collections exist.

        Creates the async Qdrant client and ensures all required collections
        are created with proper schemas.
        """
        self._client = AsyncQdrantClient(
            url=self._url,
            api_key=self._api_key,
        )
        await self._ensure_collections()
        self._collections_initialized = True

    async def close(self) -> None:
        """Close the storage client connection."""
        if self._client is not None:
            await self._client.close()
            self._client = None
            self._collections_initialized = False

    async def __aenter__(self) -> EngramStorage:
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()

    def _collection_name(self, memory_type: str) -> str:
        """Get full collection name with prefix.

        Args:
            memory_type: Short name (episode, fact, semantic, etc.)

        Returns:
            Full collection name like "engram_episodic".
        """
        suffix = COLLECTION_NAMES.get(memory_type, memory_type)
        return f"{self._prefix}_{suffix}"

    @staticmethod
    def _build_key(
        memory_id: str,
        user_id: str,
        org_id: str | None = None,
    ) -> str:
        """Build a multi-tenancy key for storage.

        Keys isolate data by organization and user:
        - With org: {org_id}/{user_id}/{memory_id}
        - Personal: personal/{user_id}/{memory_id}

        Args:
            memory_id: Unique memory identifier.
            user_id: User identifier.
            org_id: Optional organization identifier.

        Returns:
            Formatted storage key.
        """
        if org_id:
            return f"{org_id}/{user_id}/{memory_id}"
        return f"personal/{user_id}/{memory_id}"

    @staticmethod
    def _key_to_point_id(key: str) -> str:
        """Convert a storage key to a valid Qdrant point ID.

        Qdrant requires point IDs to be UUIDs or unsigned integers.
        We hash the key to create a deterministic UUID-format string.

        Args:
            key: Storage key from _build_key().

        Returns:
            UUID-format string suitable for Qdrant point ID.
        """
        # Create a deterministic hash of the key
        hash_bytes = hashlib.sha256(key.encode()).hexdigest()[:32]
        # Format as UUID: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
        return f"{hash_bytes[:8]}-{hash_bytes[8:12]}-{hash_bytes[12:16]}-{hash_bytes[16:20]}-{hash_bytes[20:32]}"

    async def _ensure_collections(self) -> None:
        """Ensure all required collections exist with proper schemas.

        Creates collections if they don't exist. Each collection uses:
        - Cosine distance for vector similarity
        - Payload indexing for user_id, org_id, confidence filters
        """
        for memory_type in COLLECTION_NAMES:
            collection_name = self._collection_name(memory_type)

            # Check if collection exists
            collections = await self.client.get_collections()
            existing = [c.name for c in collections.collections]

            if collection_name not in existing:
                # Create collection with vector config
                await self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=self._embedding_dim,
                        distance=models.Distance.COSINE,
                    ),
                )

                # Create payload indexes for filtering
                await self._create_indexes(collection_name)

    async def _create_indexes(self, collection_name: str) -> None:
        """Create payload indexes for efficient filtering.

        Args:
            collection_name: Name of the collection to index.
        """
        # Index for user isolation (required for all queries)
        await self.client.create_payload_index(
            collection_name=collection_name,
            field_name="user_id",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )

        # Index for organization filtering
        await self.client.create_payload_index(
            collection_name=collection_name,
            field_name="org_id",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )

        # Index for confidence filtering (except audit)
        if "audit" not in collection_name:
            await self.client.create_payload_index(
                collection_name=collection_name,
                field_name="confidence_value",
                field_schema=models.PayloadSchemaType.FLOAT,
            )

    def _memory_to_payload(self, memory: BaseModel) -> dict[str, Any]:
        """Convert a memory model to Qdrant payload.

        Args:
            memory: Any memory model (Episode, Fact, etc.)

        Returns:
            Dictionary payload for Qdrant storage.
        """
        data = memory.model_dump(mode="json")

        # Extract confidence value for indexing if present
        if "confidence" in data and isinstance(data["confidence"], dict):
            data["confidence_value"] = data["confidence"].get("value", 0.0)

        # Remove embedding from payload (stored separately as vector)
        data.pop("embedding", None)

        return data

    def _payload_to_memory(self, payload: dict[str, Any], memory_class: type[MemoryT]) -> MemoryT:
        """Convert Qdrant payload back to memory model.

        Args:
            payload: Qdrant payload dictionary.
            memory_class: Target memory class to instantiate.

        Returns:
            Memory model instance.
        """
        # Remove indexed fields not in model
        payload.pop("confidence_value", None)
        return memory_class.model_validate(payload)

    # =========================================================================
    # Store Operations
    # =========================================================================

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

    async def store_inhibitory(self, fact: InhibitoryFact) -> str:
        """Store an inhibitory fact.

        Args:
            fact: InhibitoryFact to store.

        Returns:
            The fact ID.
        """
        return await self._store_memory(fact, "inhibitory")

    async def _store_memory(
        self,
        memory: Episode | Fact | SemanticMemory | ProceduralMemory | InhibitoryFact,
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

    # =========================================================================
    # Search Operations
    # =========================================================================

    async def search_episodes(
        self,
        query_vector: Sequence[float],
        user_id: str,
        org_id: str | None = None,
        limit: int = 10,
        min_importance: float | None = None,
    ) -> list[Episode]:
        """Search for similar episodes.

        Args:
            query_vector: Query embedding vector.
            user_id: User ID for isolation.
            org_id: Optional org ID filter.
            limit: Maximum results to return.
            min_importance: Minimum importance threshold.

        Returns:
            List of matching Episodes sorted by similarity.
        """
        filters = self._build_filters(user_id, org_id)
        if min_importance is not None:
            filters.append(
                models.FieldCondition(
                    key="importance",
                    range=models.Range(gte=min_importance),
                )
            )

        results = await self._search(
            collection="episode",
            query_vector=query_vector,
            filters=filters,
            limit=limit,
        )

        return [
            self._payload_to_memory(r.payload, Episode) for r in results if r.payload is not None
        ]

    async def search_facts(
        self,
        query_vector: Sequence[float],
        user_id: str,
        org_id: str | None = None,
        limit: int = 10,
        min_confidence: float | None = None,
        category: str | None = None,
    ) -> list[Fact]:
        """Search for similar facts.

        Args:
            query_vector: Query embedding vector.
            user_id: User ID for isolation.
            org_id: Optional org ID filter.
            limit: Maximum results to return.
            min_confidence: Minimum confidence threshold.
            category: Filter by fact category.

        Returns:
            List of matching Facts sorted by similarity.
        """
        filters = self._build_filters(user_id, org_id, min_confidence)
        if category is not None:
            filters.append(
                models.FieldCondition(
                    key="category",
                    match=models.MatchValue(value=category),
                )
            )

        results = await self._search(
            collection="fact",
            query_vector=query_vector,
            filters=filters,
            limit=limit,
        )

        return [self._payload_to_memory(r.payload, Fact) for r in results if r.payload is not None]

    async def search_semantic(
        self,
        query_vector: Sequence[float],
        user_id: str,
        org_id: str | None = None,
        limit: int = 10,
        min_confidence: float | None = None,
    ) -> list[SemanticMemory]:
        """Search for similar semantic memories.

        Args:
            query_vector: Query embedding vector.
            user_id: User ID for isolation.
            org_id: Optional org ID filter.
            limit: Maximum results to return.
            min_confidence: Minimum confidence threshold.

        Returns:
            List of matching SemanticMemory sorted by similarity.
        """
        filters = self._build_filters(user_id, org_id, min_confidence)

        results = await self._search(
            collection="semantic",
            query_vector=query_vector,
            filters=filters,
            limit=limit,
        )

        return [
            self._payload_to_memory(r.payload, SemanticMemory)
            for r in results
            if r.payload is not None
        ]

    async def search_procedural(
        self,
        query_vector: Sequence[float],
        user_id: str,
        org_id: str | None = None,
        limit: int = 10,
        min_confidence: float | None = None,
    ) -> list[ProceduralMemory]:
        """Search for similar procedural memories.

        Args:
            query_vector: Query embedding vector.
            user_id: User ID for isolation.
            org_id: Optional org ID filter.
            limit: Maximum results to return.
            min_confidence: Minimum confidence threshold.

        Returns:
            List of matching ProceduralMemory sorted by similarity.
        """
        filters = self._build_filters(user_id, org_id, min_confidence)

        results = await self._search(
            collection="procedural",
            query_vector=query_vector,
            filters=filters,
            limit=limit,
        )

        return [
            self._payload_to_memory(r.payload, ProceduralMemory)
            for r in results
            if r.payload is not None
        ]

    async def search_inhibitory(
        self,
        query_vector: Sequence[float],
        user_id: str,
        org_id: str | None = None,
        limit: int = 10,
    ) -> list[InhibitoryFact]:
        """Search for similar inhibitory facts.

        Args:
            query_vector: Query embedding vector.
            user_id: User ID for isolation.
            org_id: Optional org ID filter.
            limit: Maximum results to return.

        Returns:
            List of matching InhibitoryFact sorted by similarity.
        """
        filters = self._build_filters(user_id, org_id)

        results = await self._search(
            collection="inhibitory",
            query_vector=query_vector,
            filters=filters,
            limit=limit,
        )

        return [
            self._payload_to_memory(r.payload, InhibitoryFact)
            for r in results
            if r.payload is not None
        ]

    def _build_filters(
        self,
        user_id: str,
        org_id: str | None = None,
        min_confidence: float | None = None,
    ) -> list[models.FieldCondition]:
        """Build Qdrant filter conditions.

        Args:
            user_id: Required user ID filter.
            org_id: Optional org ID filter.
            min_confidence: Optional confidence threshold.

        Returns:
            List of filter conditions.
        """
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

        if min_confidence is not None:
            filters.append(
                models.FieldCondition(
                    key="confidence_value",
                    range=models.Range(gte=min_confidence),
                )
            )

        return filters

    async def _search(
        self,
        collection: str,
        query_vector: Sequence[float],
        filters: list[models.FieldCondition],
        limit: int,
    ) -> list[models.ScoredPoint]:
        """Execute a vector search with filters.

        Args:
            collection: Collection type key.
            query_vector: Query embedding.
            filters: Filter conditions.
            limit: Max results.

        Returns:
            List of scored points from Qdrant.
        """
        collection_name = self._collection_name(collection)

        results = await self.client.query_points(
            collection_name=collection_name,
            query=list(query_vector),
            query_filter=models.Filter(must=filters) if filters else None,
            limit=limit,
            with_payload=True,
        )
        return list(results.points)

    # =========================================================================
    # Get/Delete Operations
    # =========================================================================

    async def get_episode(self, episode_id: str, user_id: str) -> Episode | None:
        """Get an episode by ID.

        Args:
            episode_id: Episode identifier.
            user_id: User ID for verification.

        Returns:
            Episode if found and owned by user, None otherwise.
        """
        return await self._get_by_id(episode_id, user_id, "episode", Episode)

    async def get_fact(self, fact_id: str, user_id: str) -> Fact | None:
        """Get a fact by ID."""
        return await self._get_by_id(fact_id, user_id, "fact", Fact)

    async def get_semantic(self, memory_id: str, user_id: str) -> SemanticMemory | None:
        """Get a semantic memory by ID."""
        return await self._get_by_id(memory_id, user_id, "semantic", SemanticMemory)

    async def get_procedural(self, memory_id: str, user_id: str) -> ProceduralMemory | None:
        """Get a procedural memory by ID."""
        return await self._get_by_id(memory_id, user_id, "procedural", ProceduralMemory)

    async def get_inhibitory(self, fact_id: str, user_id: str) -> InhibitoryFact | None:
        """Get an inhibitory fact by ID."""
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

        # Search by memory ID in payload
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

        return self._payload_to_memory(points[0].payload, memory_class)

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

        # Delete by filter (ensures user ownership)
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

        return result.status == models.UpdateStatus.COMPLETED

    # =========================================================================
    # Audit Operations
    # =========================================================================

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

        entries = [
            self._payload_to_memory(r.payload, AuditEntry) for r in results if r.payload is not None
        ]

        # Sort by timestamp descending
        entries.sort(key=lambda e: e.timestamp, reverse=True)

        return entries
