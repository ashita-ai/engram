"""Webhook storage operations for Engram.

Provides methods to store, retrieve, and manage webhooks and delivery logs.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from qdrant_client import models

if TYPE_CHECKING:
    from engram.models import EventType, WebhookConfig, WebhookDelivery


class WebhookMixin:
    """Mixin providing webhook operations for EngramStorage.

    This mixin expects the following attributes/methods from the base class:
    - _collection_name(memory_type) -> str
    - _build_key(memory_id, user_id, org_id) -> str
    - _key_to_point_id(key) -> str
    - _payload_to_memory(payload, memory_class) -> MemoryT
    - _embedding_dim: int
    - client: AsyncQdrantClient
    """

    _collection_name: Any
    _build_key: Any
    _key_to_point_id: Any
    _payload_to_memory: Any
    _embedding_dim: int
    client: Any

    async def store_webhook(self, webhook: WebhookConfig) -> str:
        """Store a webhook configuration.

        Args:
            webhook: WebhookConfig to store.

        Returns:
            The webhook ID.
        """
        collection = self._collection_name("webhooks")
        key = self._build_key(webhook.id, webhook.user_id, webhook.org_id)
        payload = webhook.model_dump(mode="json")

        # Convert HttpUrl to string for storage
        payload["url"] = str(webhook.url)

        # Use zero vector (no semantic search needed)
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

        return webhook.id

    async def get_webhook(
        self,
        webhook_id: str,
        user_id: str,
        org_id: str | None = None,
    ) -> WebhookConfig | None:
        """Get a webhook by ID.

        Args:
            webhook_id: ID of the webhook.
            user_id: User ID for multi-tenancy.
            org_id: Optional org filter.

        Returns:
            WebhookConfig or None if not found.
        """
        from engram.models import WebhookConfig

        collection = self._collection_name("webhooks")
        key = self._build_key(webhook_id, user_id, org_id)

        results = await self.client.retrieve(
            collection_name=collection,
            ids=[self._key_to_point_id(key)],
            with_payload=True,
        )

        if not results:
            return None

        payload = results[0].payload
        if payload is None:
            return None

        webhook: WebhookConfig = self._payload_to_memory(payload, WebhookConfig)
        return webhook

    async def list_webhooks(
        self,
        user_id: str,
        org_id: str | None = None,
        enabled_only: bool = False,
        limit: int = 100,
    ) -> list[WebhookConfig]:
        """List webhooks for a user.

        Args:
            user_id: User to list webhooks for.
            org_id: Optional org filter.
            enabled_only: If True, only return enabled webhooks.
            limit: Maximum webhooks to return.

        Returns:
            List of WebhookConfig.
        """
        from engram.models import WebhookConfig

        collection = self._collection_name("webhooks")

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

        if enabled_only:
            filters.append(
                models.FieldCondition(
                    key="enabled",
                    match=models.MatchValue(value=True),
                )
            )

        results, _ = await self.client.scroll(
            collection_name=collection,
            scroll_filter=models.Filter(must=filters),
            limit=limit,
            with_payload=True,
        )

        return [
            self._payload_to_memory(r.payload, WebhookConfig)
            for r in results
            if r.payload is not None
        ]

    async def get_webhooks_for_event(
        self,
        event_type: EventType,
        user_id: str,
        org_id: str | None = None,
    ) -> list[WebhookConfig]:
        """Get all enabled webhooks that subscribe to an event type.

        Args:
            event_type: The event type to filter for.
            user_id: User ID for multi-tenancy.
            org_id: Optional org filter.

        Returns:
            List of WebhookConfig that subscribe to this event.
        """
        webhooks = await self.list_webhooks(
            user_id=user_id,
            org_id=org_id,
            enabled_only=True,
        )

        return [wh for wh in webhooks if wh.subscribes_to(event_type)]

    async def update_webhook(
        self,
        webhook_id: str,
        user_id: str,
        org_id: str | None = None,
        **updates: Any,
    ) -> WebhookConfig | None:
        """Update a webhook configuration.

        Args:
            webhook_id: ID of the webhook to update.
            user_id: User ID for multi-tenancy.
            org_id: Optional org filter.
            **updates: Fields to update.

        Returns:
            Updated WebhookConfig or None if not found.
        """
        from datetime import UTC, datetime

        webhook = await self.get_webhook(webhook_id, user_id, org_id)
        if webhook is None:
            return None

        # Update fields
        for key, value in updates.items():
            if hasattr(webhook, key):
                setattr(webhook, key, value)

        webhook.updated_at = datetime.now(UTC)

        await self.store_webhook(webhook)
        return webhook

    async def delete_webhook(
        self,
        webhook_id: str,
        user_id: str,
        org_id: str | None = None,
    ) -> bool:
        """Delete a webhook configuration.

        Args:
            webhook_id: ID of the webhook to delete.
            user_id: User ID for multi-tenancy.
            org_id: Optional org filter.

        Returns:
            True if deleted, False if not found.
        """
        collection = self._collection_name("webhooks")
        key = self._build_key(webhook_id, user_id, org_id)

        result = await self.client.delete(
            collection_name=collection,
            points_selector=models.PointIdsList(
                points=[self._key_to_point_id(key)],
            ),
        )

        return result is not None

    async def log_delivery(self, delivery: WebhookDelivery) -> str:
        """Log a webhook delivery attempt.

        Args:
            delivery: WebhookDelivery to log.

        Returns:
            The delivery ID.
        """
        collection = self._collection_name("webhook_deliveries")
        key = self._build_key(delivery.id, delivery.user_id, delivery.org_id)
        payload = delivery.model_dump(mode="json")

        # Use zero vector (no semantic search needed)
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

        return delivery.id

    async def update_delivery(self, delivery: WebhookDelivery) -> str:
        """Update an existing delivery record.

        Args:
            delivery: WebhookDelivery with updated fields.

        Returns:
            The delivery ID.
        """
        return await self.log_delivery(delivery)

    async def get_delivery_logs(
        self,
        webhook_id: str,
        user_id: str,
        org_id: str | None = None,
        since: datetime | None = None,
        limit: int = 100,
    ) -> list[WebhookDelivery]:
        """Get delivery logs for a webhook.

        Args:
            webhook_id: ID of the webhook.
            user_id: User ID for multi-tenancy.
            org_id: Optional org filter.
            since: Optional timestamp to filter entries after.
            limit: Maximum entries to return.

        Returns:
            List of WebhookDelivery sorted by timestamp (newest first).
        """
        from engram.models import WebhookDelivery

        collection = self._collection_name("webhook_deliveries")

        filters: list[models.FieldCondition] = [
            models.FieldCondition(
                key="webhook_id",
                match=models.MatchValue(value=webhook_id),
            ),
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

        if since is not None:
            filters.append(
                models.FieldCondition(
                    key="created_at",
                    range=models.Range(gte=since.isoformat()),
                )
            )

        results, _ = await self.client.scroll(
            collection_name=collection,
            scroll_filter=models.Filter(must=filters),
            limit=limit,
            with_payload=True,
        )

        deliveries: list[WebhookDelivery] = [
            self._payload_to_memory(r.payload, WebhookDelivery)
            for r in results
            if r.payload is not None
        ]

        # Sort by created_at descending (newest first)
        deliveries.sort(key=lambda d: d.created_at, reverse=True)

        return deliveries

    async def get_pending_deliveries(
        self,
        user_id: str | None = None,
        limit: int = 100,
    ) -> list[WebhookDelivery]:
        """Get pending or retrying delivery attempts.

        Used by retry workers to find deliveries that need to be retried.

        Args:
            user_id: Optional user filter.
            limit: Maximum entries to return.

        Returns:
            List of WebhookDelivery with status pending or retrying.
        """
        from engram.models import WebhookDelivery

        collection = self._collection_name("webhook_deliveries")

        status_filter = models.Filter(
            should=[
                models.FieldCondition(
                    key="status",
                    match=models.MatchValue(value="pending"),
                ),
                models.FieldCondition(
                    key="status",
                    match=models.MatchValue(value="retrying"),
                ),
            ]
        )

        if user_id is not None:
            full_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="user_id",
                        match=models.MatchValue(value=user_id),
                    ),
                    status_filter,
                ]
            )
        else:
            full_filter = status_filter

        results, _ = await self.client.scroll(
            collection_name=collection,
            scroll_filter=full_filter,
            limit=limit,
            with_payload=True,
        )

        deliveries: list[WebhookDelivery] = [
            self._payload_to_memory(r.payload, WebhookDelivery)
            for r in results
            if r.payload is not None
        ]

        # Sort by next_retry_at for processing order
        deliveries.sort(key=lambda d: d.next_retry_at or d.created_at)

        return deliveries
