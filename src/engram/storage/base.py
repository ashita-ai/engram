"""Base storage class and helpers.

Contains initialization, collection management, and shared utilities.
"""

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
    NegationFact,
    ProceduralMemory,
    SemanticMemory,
)

if TYPE_CHECKING:
    pass

# Type variable for memory models
MemoryT = TypeVar(
    "MemoryT",
    Episode,
    Fact,
    SemanticMemory,
    ProceduralMemory,
    NegationFact,
    AuditEntry,
)

# Collection names by memory type (keys match API memory_types values)
COLLECTION_NAMES = {
    "episodic": "episodic",
    "factual": "factual",
    "semantic": "semantic",
    "procedural": "procedural",
    "negation": "negation",
    "audit": "audit",
}

# Default embedding dimension (text-embedding-3-small)
DEFAULT_EMBEDDING_DIM = 1536


class StorageBase:
    """Base class for Engram storage with initialization and helpers.

    Provides:
    - Client initialization and lifecycle management
    - Collection creation and indexing
    - Key building and point ID conversion
    - Payload serialization/deserialization
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
        """Initialize the storage client and ensure collections exist."""
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

    async def __aenter__(self) -> StorageBase:
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()

    def _collection_name(self, memory_type: str) -> str:
        """Get full collection name with prefix."""
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
        """
        if org_id:
            return f"{org_id}/{user_id}/{memory_id}"
        return f"personal/{user_id}/{memory_id}"

    @staticmethod
    def _key_to_point_id(key: str) -> str:
        """Convert a storage key to a valid Qdrant point ID.

        Qdrant requires point IDs to be UUIDs or unsigned integers.
        We hash the key to create a deterministic UUID-format string.
        """
        h = hashlib.sha256(key.encode()).hexdigest()[:32]
        return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"

    async def _ensure_collections(self) -> None:
        """Ensure all required collections exist with proper schemas."""
        for memory_type in COLLECTION_NAMES:
            collection_name = self._collection_name(memory_type)

            collections = await self.client.get_collections()
            existing = [c.name for c in collections.collections]

            if collection_name not in existing:
                await self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=self._embedding_dim,
                        distance=models.Distance.COSINE,
                    ),
                )
                await self._create_indexes(collection_name)

    async def _create_indexes(self, collection_name: str) -> None:
        """Create payload indexes for efficient filtering."""
        await self.client.create_payload_index(
            collection_name=collection_name,
            field_name="user_id",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        await self.client.create_payload_index(
            collection_name=collection_name,
            field_name="org_id",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        if "audit" not in collection_name:
            await self.client.create_payload_index(
                collection_name=collection_name,
                field_name="confidence_value",
                field_schema=models.PayloadSchemaType.FLOAT,
            )

    def _memory_to_payload(self, memory: BaseModel) -> dict[str, Any]:
        """Convert a memory model to Qdrant payload."""
        data = memory.model_dump(mode="json")

        if "confidence" in data and isinstance(data["confidence"], dict):
            data["confidence_value"] = data["confidence"].get("value", 0.0)

        data.pop("embedding", None)
        return data

    def _payload_to_memory(self, payload: dict[str, Any], memory_class: type[MemoryT]) -> MemoryT:
        """Convert Qdrant payload back to memory model."""
        payload.pop("confidence_value", None)
        return memory_class.model_validate(payload)
