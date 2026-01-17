"""Unit tests for Engram webhook system."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from engram.models import (
    ALL_EVENT_TYPES,
    WebhookConfig,
    WebhookDelivery,
    WebhookEvent,
)
from engram.webhooks import compute_signature, verify_signature


class TestWebhookConfig:
    """Tests for WebhookConfig model."""

    def test_generates_unique_ids(self):
        """Each WebhookConfig should have a unique ID."""
        webhooks = [
            WebhookConfig(
                user_id="user_1",
                url="https://example.com/webhook",
                secret="test_secret_16chars",
            )
            for _ in range(10)
        ]
        ids = [w.id for w in webhooks]
        assert len(ids) == len(set(ids))

    def test_id_prefix(self):
        """WebhookConfig ID should start with whk_."""
        webhook = WebhookConfig(
            user_id="user_1",
            url="https://example.com/webhook",
            secret="test_secret_16chars",
        )
        assert webhook.id.startswith("whk_")

    def test_defaults_to_all_events(self):
        """WebhookConfig should subscribe to all events by default."""
        webhook = WebhookConfig(
            user_id="user_1",
            url="https://example.com/webhook",
            secret="test_secret_16chars",
        )
        assert set(webhook.events) == set(ALL_EVENT_TYPES)

    def test_enabled_by_default(self):
        """WebhookConfig should be enabled by default."""
        webhook = WebhookConfig(
            user_id="user_1",
            url="https://example.com/webhook",
            secret="test_secret_16chars",
        )
        assert webhook.enabled is True

    def test_subscribes_to_event(self):
        """subscribes_to should check event subscription."""
        webhook = WebhookConfig(
            user_id="user_1",
            url="https://example.com/webhook",
            secret="test_secret_16chars",
            events=["encode_complete", "consolidation_complete"],
        )
        assert webhook.subscribes_to("encode_complete") is True
        assert webhook.subscribes_to("consolidation_complete") is True
        assert webhook.subscribes_to("decay_complete") is False

    def test_disabled_webhook_does_not_subscribe(self):
        """Disabled webhook should not subscribe to any events."""
        webhook = WebhookConfig(
            user_id="user_1",
            url="https://example.com/webhook",
            secret="test_secret_16chars",
            enabled=False,
        )
        assert webhook.subscribes_to("encode_complete") is False

    def test_extra_fields_forbidden(self):
        """WebhookConfig should reject unknown fields."""
        with pytest.raises(ValidationError):
            WebhookConfig(
                user_id="user_1",
                url="https://example.com/webhook",
                secret="test_secret_16chars",
                unknown_field="value",
            )


class TestWebhookEvent:
    """Tests for WebhookEvent model."""

    def test_generates_unique_ids(self):
        """Each WebhookEvent should have a unique ID."""
        events = [
            WebhookEvent(
                event="encode_complete",
                user_id="user_1",
            )
            for _ in range(10)
        ]
        ids = [e.id for e in events]
        assert len(ids) == len(set(ids))

    def test_id_prefix(self):
        """WebhookEvent ID should start with evt_."""
        event = WebhookEvent(
            event="encode_complete",
            user_id="user_1",
        )
        assert event.id.startswith("evt_")

    def test_for_encode_complete(self):
        """for_encode_complete factory should create correct event."""
        event = WebhookEvent.for_encode_complete(
            user_id="user_1",
            episode_id="ep_123",
            facts_count=5,
            duration_ms=150,
            org_id="org_456",
        )
        assert event.event == "encode_complete"
        assert event.user_id == "user_1"
        assert event.org_id == "org_456"
        assert event.data["episode_id"] == "ep_123"
        assert event.data["facts_count"] == 5
        assert event.data["duration_ms"] == 150

    def test_for_consolidation_complete(self):
        """for_consolidation_complete factory should create correct event."""
        event = WebhookEvent.for_consolidation_complete(
            user_id="user_1",
            facts_extracted=10,
            links_created=5,
            duration_ms=2500,
        )
        assert event.event == "consolidation_complete"
        assert event.data["facts_extracted"] == 10
        assert event.data["links_created"] == 5

    def test_for_memory_created(self):
        """for_memory_created factory should create correct event."""
        event = WebhookEvent.for_memory_created(
            user_id="user_1",
            memory_id="sem_123",
            memory_type="semantic",
        )
        assert event.event == "memory_created"
        assert event.data["memory_id"] == "sem_123"
        assert event.data["memory_type"] == "semantic"

    def test_for_memory_deleted(self):
        """for_memory_deleted factory should create correct event."""
        event = WebhookEvent.for_memory_deleted(
            user_id="user_1",
            memory_id="sem_123",
            memory_type="semantic",
        )
        assert event.event == "memory_deleted"
        assert event.data["memory_id"] == "sem_123"


class TestWebhookDelivery:
    """Tests for WebhookDelivery model."""

    def test_generates_unique_ids(self):
        """Each WebhookDelivery should have a unique ID."""
        deliveries = [
            WebhookDelivery(
                webhook_id="whk_123",
                event_id="evt_456",
                user_id="user_1",
            )
            for _ in range(10)
        ]
        ids = [d.id for d in deliveries]
        assert len(ids) == len(set(ids))

    def test_id_prefix(self):
        """WebhookDelivery ID should start with dlv_."""
        delivery = WebhookDelivery(
            webhook_id="whk_123",
            event_id="evt_456",
            user_id="user_1",
        )
        assert delivery.id.startswith("dlv_")

    def test_default_status_is_pending(self):
        """WebhookDelivery should default to pending status."""
        delivery = WebhookDelivery(
            webhook_id="whk_123",
            event_id="evt_456",
            user_id="user_1",
        )
        assert delivery.status == "pending"
        assert delivery.attempt == 1

    def test_mark_success(self):
        """mark_success should update delivery status."""
        delivery = WebhookDelivery(
            webhook_id="whk_123",
            event_id="evt_456",
            user_id="user_1",
        )
        delivery.mark_success(response_code=200, response_body="OK")
        assert delivery.status == "success"
        assert delivery.response_code == 200
        assert delivery.response_body == "OK"
        assert delivery.completed_at is not None

    def test_mark_failed(self):
        """mark_failed should update delivery status."""
        delivery = WebhookDelivery(
            webhook_id="whk_123",
            event_id="evt_456",
            user_id="user_1",
        )
        delivery.mark_failed(
            error="Connection refused",
            response_code=500,
        )
        assert delivery.status == "failed"
        assert delivery.error == "Connection refused"
        assert delivery.response_code == 500
        assert delivery.completed_at is not None

    def test_mark_retrying(self):
        """mark_retrying should schedule next retry."""
        delivery = WebhookDelivery(
            webhook_id="whk_123",
            event_id="evt_456",
            user_id="user_1",
        )
        next_retry = datetime.now(UTC)
        delivery.mark_retrying(
            next_retry_at=next_retry,
            error="Timeout",
            response_code=504,
        )
        assert delivery.status == "retrying"
        assert delivery.next_retry_at == next_retry
        assert delivery.error == "Timeout"

    def test_truncates_long_response_body(self):
        """mark_success should truncate response body to 1000 chars."""
        delivery = WebhookDelivery(
            webhook_id="whk_123",
            event_id="evt_456",
            user_id="user_1",
        )
        long_body = "x" * 2000
        delivery.mark_success(response_code=200, response_body=long_body)
        assert len(delivery.response_body) == 1000


class TestSignature:
    """Tests for HMAC signature functions."""

    def test_compute_signature(self):
        """compute_signature should return sha256 prefixed signature."""
        payload = '{"event": "test"}'
        secret = "test_secret"
        sig = compute_signature(payload, secret)
        assert sig.startswith("sha256=")
        assert len(sig) == 71  # "sha256=" + 64 hex chars

    def test_verify_signature_valid(self):
        """verify_signature should return True for valid signature."""
        payload = '{"event": "test"}'
        secret = "test_secret"
        sig = compute_signature(payload, secret)
        assert verify_signature(payload, secret, sig) is True

    def test_verify_signature_invalid(self):
        """verify_signature should return False for invalid signature."""
        payload = '{"event": "test"}'
        secret = "test_secret"
        wrong_sig = "sha256=000000"
        assert verify_signature(payload, secret, wrong_sig) is False

    def test_verify_signature_wrong_secret(self):
        """verify_signature should fail with wrong secret."""
        payload = '{"event": "test"}'
        sig = compute_signature(payload, "secret1")
        assert verify_signature(payload, "secret2", sig) is False

    def test_signature_deterministic(self):
        """Same payload and secret should produce same signature."""
        payload = '{"event": "test"}'
        secret = "test_secret"
        sig1 = compute_signature(payload, secret)
        sig2 = compute_signature(payload, secret)
        assert sig1 == sig2


class TestEventTypes:
    """Tests for event type validation."""

    def test_all_event_types_list(self):
        """ALL_EVENT_TYPES should contain expected events."""
        expected = {
            "encode_complete",
            "consolidation_started",
            "consolidation_complete",
            "decay_complete",
            "memory_created",
            "memory_updated",
            "memory_archived",
            "memory_deleted",
        }
        assert set(ALL_EVENT_TYPES) == expected

    def test_invalid_event_type_rejected(self):
        """WebhookConfig should reject invalid event types."""
        with pytest.raises(ValidationError):
            WebhookConfig(
                user_id="user_1",
                url="https://example.com/webhook",
                secret="test_secret_16chars",
                events=["invalid_event"],
            )


class TestWebhookConfigWithOrg:
    """Tests for WebhookConfig with organization isolation."""

    def test_create_with_org_id(self):
        """WebhookConfig should accept org_id."""
        webhook = WebhookConfig(
            user_id="user_1",
            org_id="org_456",
            url="https://example.com/webhook",
            secret="test_secret_16chars",
        )
        assert webhook.org_id == "org_456"

    def test_default_org_id_is_none(self):
        """org_id should default to None."""
        webhook = WebhookConfig(
            user_id="user_1",
            url="https://example.com/webhook",
            secret="test_secret_16chars",
        )
        assert webhook.org_id is None
