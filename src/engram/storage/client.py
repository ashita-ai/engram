"""Qdrant storage client for Engram memory system.

This module provides the main EngramStorage class that combines
all storage operations through mixins.

Example:
    ```python
    from engram.storage import EngramStorage

    async with EngramStorage() as storage:
        await storage.store_episode(episode)
        results = await storage.search_episodes(query_vector, user_id="user_123")
    ```
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .audit import AuditMixin
from .base import COLLECTION_NAMES, DEFAULT_EMBEDDING_DIM, StorageBase
from .crud import CRUDMixin
from .history import HistoryMixin
from .search import SearchMixin
from .store import StoreMixin

logger = logging.getLogger(__name__)


class MemoryStats(BaseModel):
    """Statistics about stored memories."""

    model_config = ConfigDict(extra="forbid")

    episodes: int = Field(default=0, ge=0, description="Number of episode memories")
    structured: int = Field(default=0, ge=0, description="Number of structured memories")
    semantic: int = Field(default=0, ge=0, description="Number of semantic memories")
    procedural: int = Field(default=0, ge=0, description="Number of procedural memories")
    pending_consolidation: int = Field(
        default=0, ge=0, description="Episodes awaiting consolidation"
    )
    structured_avg_confidence: float | None = Field(
        default=None, description="Average confidence of structured memories"
    )
    structured_min_confidence: float | None = Field(
        default=None, description="Minimum confidence of structured memories"
    )
    structured_max_confidence: float | None = Field(
        default=None, description="Maximum confidence of structured memories"
    )
    semantic_avg_confidence: float | None = Field(
        default=None, description="Average confidence of semantic memories"
    )


class EngramStorage(StoreMixin, SearchMixin, CRUDMixin, AuditMixin, HistoryMixin, StorageBase):
    """Async Qdrant storage client for Engram memories.

    Handles collection management, multi-tenancy isolation, and CRUD operations
    for all memory types. Uses async Qdrant client for non-blocking I/O.

    This class combines functionality from multiple mixins:
    - StoreMixin: store_episode, store_fact, store_semantic, etc.
    - SearchMixin: search_episodes, search_facts, search_semantic, etc.
    - CRUDMixin: get_episode, delete_episode, get_fact, delete_fact, etc.
    - AuditMixin: log_audit, get_audit_log
    - HistoryMixin: log_history, get_memory_history, get_user_history

    Attributes:
        client: Async Qdrant client instance.

    Example:
        ```python
        storage = EngramStorage()
        await storage.initialize()

        # Store an episode
        episode = Episode(content="Hello", role="user", user_id="user_123", embedding=[...])
        await storage.store_episode(episode)

        # Search for memories
        results = await storage.search_episodes(
            query_vector=[0.1, 0.2, ...],
            user_id="user_123",
            limit=10,
        )
        ```
    """

    async def __aenter__(self) -> EngramStorage:
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()

    async def get_memory_stats(
        self,
        user_id: str,
        org_id: str | None = None,
    ) -> MemoryStats:
        """Get statistics about stored memories.

        Args:
            user_id: User ID to get stats for.
            org_id: Optional org ID filter.

        Returns:
            MemoryStats with counts and confidence stats.
        """
        from qdrant_client import models

        # Build filter for user/org
        filter_conditions = [
            models.FieldCondition(
                key="user_id",
                match=models.MatchValue(value=user_id),
            )
        ]
        if org_id:
            filter_conditions.append(
                models.FieldCondition(
                    key="org_id",
                    match=models.MatchValue(value=org_id),
                )
            )

        query_filter = models.Filter(must=filter_conditions)

        # Count memories in each collection
        prefix = self._get_collection_prefix()

        async def count_collection(name: str) -> int:
            """Count points in a collection with filter."""
            try:
                result = await self.client.count(
                    collection_name=f"{prefix}_{name}",
                    count_filter=query_filter,
                )
                return int(result.count)
            except Exception:
                logger.warning("Failed to count collection %s_%s", prefix, name, exc_info=True)
                return 0

        # Count all memory types
        episodes_count = await count_collection("episodic")
        structured_count = await count_collection("structured")
        semantic_count = await count_collection("semantic")
        procedural_count = await count_collection("procedural")

        # Count pending consolidation (unconsolidated episodes)
        unconsolidated_filter = models.Filter(
            must=[
                *filter_conditions,
                models.FieldCondition(
                    key="consolidated",
                    match=models.MatchValue(value=False),
                ),
            ]
        )
        try:
            pending_result = await self.client.count(
                collection_name=f"{prefix}_episodic",
                count_filter=unconsolidated_filter,
            )
            pending_count = pending_result.count
        except Exception:
            logger.warning("Failed to count pending consolidation", exc_info=True)
            pending_count = 0

        # Get confidence stats for structured memories
        structured_avg = None
        structured_min = None
        structured_max = None
        if structured_count > 0:
            try:
                structured_result = await self.client.scroll(
                    collection_name=f"{prefix}_structured",
                    scroll_filter=query_filter,
                    limit=1000,
                    with_payload=["confidence"],
                )
                confidences = self._extract_confidences_from_points(structured_result[0])
                if confidences:
                    structured_avg = sum(confidences) / len(confidences)
                    structured_min = min(confidences)
                    structured_max = max(confidences)
            except Exception:
                logger.warning("Failed to compute structured confidence stats", exc_info=True)

        # Get confidence stats for semantic memories
        semantic_avg = None
        if semantic_count > 0:
            try:
                semantic_result = await self.client.scroll(
                    collection_name=f"{prefix}_semantic",
                    scroll_filter=query_filter,
                    limit=1000,
                    with_payload=["confidence"],
                )
                confidences = self._extract_confidences_from_points(semantic_result[0])
                if confidences:
                    semantic_avg = sum(confidences) / len(confidences)
            except Exception:
                logger.warning("Failed to compute semantic confidence stats", exc_info=True)

        return MemoryStats(
            episodes=episodes_count,
            structured=structured_count,
            semantic=semantic_count,
            procedural=procedural_count,
            pending_consolidation=pending_count,
            structured_avg_confidence=structured_avg,
            structured_min_confidence=structured_min,
            structured_max_confidence=structured_max,
            semantic_avg_confidence=semantic_avg,
        )

    @staticmethod
    def _extract_confidences_from_points(points: list[Any]) -> list[float]:
        """Extract confidence values from Qdrant scroll result points.

        Handles both dict-style ({"value": 0.9}) and scalar confidence values.

        Args:
            points: List of Qdrant point results with payloads.

        Returns:
            List of confidence float values.
        """
        confidences: list[float] = []
        for point in points:
            if point.payload and "confidence" in point.payload:
                conf = point.payload["confidence"]
                if isinstance(conf, dict) and "value" in conf:
                    confidences.append(conf["value"])
                elif isinstance(conf, int | float):
                    confidences.append(conf)
        return confidences

    def _get_collection_prefix(self) -> str:
        """Get the collection name prefix from settings."""
        from engram.config import settings

        return settings.collection_prefix


__all__ = [
    "EngramStorage",
    "MemoryStats",
    "COLLECTION_NAMES",
    "DEFAULT_EMBEDDING_DIM",
]
