"""Webhook models for server-to-server event notifications.

Provides webhook registration, event payloads, and delivery tracking
for reliable async notifications of Engram operations.
"""

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl

from .base import generate_id

# Event types that can trigger webhooks
EventType = Literal[
    "encode_complete",
    "consolidation_started",
    "consolidation_complete",
    "decay_complete",
    "memory_created",
    "memory_updated",
    "memory_archived",
    "memory_deleted",
]

# All available event types for subscription
ALL_EVENT_TYPES: list[EventType] = [
    "encode_complete",
    "consolidation_started",
    "consolidation_complete",
    "decay_complete",
    "memory_created",
    "memory_updated",
    "memory_archived",
    "memory_deleted",
]

# Delivery status
DeliveryStatus = Literal["pending", "success", "failed", "retrying"]


class WebhookConfig(BaseModel):
    """Configuration for a registered webhook.

    Attributes:
        id: Unique identifier for this webhook.
        user_id: User who owns this webhook.
        org_id: Organization (optional).
        url: HTTPS endpoint to receive webhook events.
        secret: Shared secret for HMAC-SHA256 signature verification.
        events: List of event types this webhook subscribes to.
        enabled: Whether this webhook is active.
        created_at: When the webhook was registered.
        updated_at: When the webhook was last modified.
        description: Optional human-readable description.
        max_retries: Maximum delivery attempts (default 5).
        retry_delay_seconds: Initial retry delay (exponential backoff).
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: generate_id("whk"))
    user_id: str = Field(description="User who owns this webhook")
    org_id: str | None = Field(default=None, description="Organization (optional)")
    url: HttpUrl = Field(description="HTTPS endpoint to receive events")
    secret: str = Field(description="Shared secret for HMAC-SHA256 signatures")
    events: list[EventType] = Field(
        default_factory=lambda: list(ALL_EVENT_TYPES),
        description="Event types to subscribe to",
    )
    enabled: bool = Field(default=True, description="Whether webhook is active")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When the webhook was registered",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When the webhook was last modified",
    )
    description: str | None = Field(default=None, description="Human-readable description")
    max_retries: int = Field(default=5, ge=0, le=10, description="Maximum delivery attempts")
    retry_delay_seconds: int = Field(
        default=1, ge=1, le=60, description="Initial retry delay (doubles each attempt)"
    )

    def subscribes_to(self, event_type: EventType) -> bool:
        """Check if this webhook subscribes to the given event type."""
        return self.enabled and event_type in self.events


class WebhookEvent(BaseModel):
    """Event payload sent to webhook endpoints.

    Attributes:
        id: Unique identifier for this event.
        event: Event type (encode_complete, consolidation_complete, etc.).
        timestamp: When the event occurred.
        user_id: User who triggered the event.
        org_id: Organization (optional).
        session_id: Session context (optional).
        data: Event-specific payload data.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: generate_id("evt"))
    event: EventType = Field(description="Event type")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When the event occurred",
    )
    user_id: str = Field(description="User who triggered the event")
    org_id: str | None = Field(default=None, description="Organization (optional)")
    session_id: str | None = Field(default=None, description="Session context (optional)")
    data: dict[str, Any] = Field(default_factory=dict, description="Event-specific payload")

    @classmethod
    def for_encode_complete(
        cls,
        user_id: str,
        episode_id: str,
        facts_count: int,
        duration_ms: int,
        org_id: str | None = None,
        session_id: str | None = None,
    ) -> "WebhookEvent":
        """Create event for encode completion."""
        return cls(
            event="encode_complete",
            user_id=user_id,
            org_id=org_id,
            session_id=session_id,
            data={
                "episode_id": episode_id,
                "facts_count": facts_count,
                "duration_ms": duration_ms,
            },
        )

    @classmethod
    def for_consolidation_started(
        cls,
        user_id: str,
        episode_count: int,
        org_id: str | None = None,
    ) -> "WebhookEvent":
        """Create event for consolidation start."""
        return cls(
            event="consolidation_started",
            user_id=user_id,
            org_id=org_id,
            data={
                "episode_count": episode_count,
            },
        )

    @classmethod
    def for_consolidation_complete(
        cls,
        user_id: str,
        facts_extracted: int,
        links_created: int,
        duration_ms: int,
        org_id: str | None = None,
    ) -> "WebhookEvent":
        """Create event for consolidation completion."""
        return cls(
            event="consolidation_complete",
            user_id=user_id,
            org_id=org_id,
            data={
                "facts_extracted": facts_extracted,
                "links_created": links_created,
                "duration_ms": duration_ms,
            },
        )

    @classmethod
    def for_decay_complete(
        cls,
        user_id: str,
        memories_updated: int,
        memories_archived: int,
        duration_ms: int,
        org_id: str | None = None,
    ) -> "WebhookEvent":
        """Create event for decay completion."""
        return cls(
            event="decay_complete",
            user_id=user_id,
            org_id=org_id,
            data={
                "memories_updated": memories_updated,
                "memories_archived": memories_archived,
                "duration_ms": duration_ms,
            },
        )

    @classmethod
    def for_memory_created(
        cls,
        user_id: str,
        memory_id: str,
        memory_type: str,
        org_id: str | None = None,
    ) -> "WebhookEvent":
        """Create event for memory creation."""
        return cls(
            event="memory_created",
            user_id=user_id,
            org_id=org_id,
            data={
                "memory_id": memory_id,
                "memory_type": memory_type,
            },
        )

    @classmethod
    def for_memory_updated(
        cls,
        user_id: str,
        memory_id: str,
        memory_type: str,
        fields_changed: list[str],
        org_id: str | None = None,
    ) -> "WebhookEvent":
        """Create event for memory update."""
        return cls(
            event="memory_updated",
            user_id=user_id,
            org_id=org_id,
            data={
                "memory_id": memory_id,
                "memory_type": memory_type,
                "fields_changed": fields_changed,
            },
        )

    @classmethod
    def for_memory_archived(
        cls,
        user_id: str,
        memory_id: str,
        memory_type: str,
        reason: str,
        org_id: str | None = None,
    ) -> "WebhookEvent":
        """Create event for memory archival."""
        return cls(
            event="memory_archived",
            user_id=user_id,
            org_id=org_id,
            data={
                "memory_id": memory_id,
                "memory_type": memory_type,
                "reason": reason,
            },
        )

    @classmethod
    def for_memory_deleted(
        cls,
        user_id: str,
        memory_id: str,
        memory_type: str,
        org_id: str | None = None,
    ) -> "WebhookEvent":
        """Create event for memory deletion."""
        return cls(
            event="memory_deleted",
            user_id=user_id,
            org_id=org_id,
            data={
                "memory_id": memory_id,
                "memory_type": memory_type,
            },
        )


class WebhookDelivery(BaseModel):
    """Record of a webhook delivery attempt.

    Attributes:
        id: Unique identifier for this delivery attempt.
        webhook_id: ID of the webhook configuration.
        event_id: ID of the event being delivered.
        user_id: User who owns the webhook.
        org_id: Organization (optional).
        status: Delivery status (pending, success, failed, retrying).
        attempt: Current attempt number (1-indexed).
        created_at: When this delivery attempt started.
        completed_at: When this delivery attempt finished.
        response_code: HTTP response status code (if received).
        response_body: HTTP response body (truncated, for debugging).
        error: Error message if delivery failed.
        next_retry_at: When the next retry attempt will occur.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: generate_id("dlv"))
    webhook_id: str = Field(description="ID of the webhook configuration")
    event_id: str = Field(description="ID of the event being delivered")
    user_id: str = Field(description="User who owns the webhook")
    org_id: str | None = Field(default=None, description="Organization (optional)")
    status: DeliveryStatus = Field(default="pending", description="Delivery status")
    attempt: int = Field(default=1, ge=1, description="Current attempt number")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When this delivery started",
    )
    completed_at: datetime | None = Field(
        default=None,
        description="When this delivery finished",
    )
    response_code: int | None = Field(default=None, description="HTTP response status code")
    response_body: str | None = Field(
        default=None,
        description="HTTP response body (truncated to 1000 chars)",
    )
    error: str | None = Field(default=None, description="Error message if failed")
    next_retry_at: datetime | None = Field(
        default=None,
        description="When next retry will occur",
    )

    def mark_success(
        self, response_code: int, response_body: str | None = None
    ) -> "WebhookDelivery":
        """Mark delivery as successful."""
        self.status = "success"
        self.completed_at = datetime.now(UTC)
        self.response_code = response_code
        if response_body:
            self.response_body = response_body[:1000]  # Truncate
        return self

    def mark_failed(
        self,
        error: str,
        response_code: int | None = None,
        response_body: str | None = None,
    ) -> "WebhookDelivery":
        """Mark delivery as failed (no more retries)."""
        self.status = "failed"
        self.completed_at = datetime.now(UTC)
        self.error = error
        self.response_code = response_code
        if response_body:
            self.response_body = response_body[:1000]
        return self

    def mark_retrying(
        self,
        next_retry_at: datetime,
        error: str,
        response_code: int | None = None,
    ) -> "WebhookDelivery":
        """Mark delivery for retry."""
        self.status = "retrying"
        self.error = error
        self.response_code = response_code
        self.next_retry_at = next_retry_at
        return self


__all__ = [
    "ALL_EVENT_TYPES",
    "DeliveryStatus",
    "EventType",
    "WebhookConfig",
    "WebhookDelivery",
    "WebhookEvent",
]
