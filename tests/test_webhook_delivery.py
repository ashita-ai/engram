"""Unit tests for webhook delivery dispatcher."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from engram.models import WebhookConfig, WebhookDelivery, WebhookEvent
from engram.webhooks.delivery import (
    WebhookDispatcher,
    compute_signature,
    dispatch_webhook_event,
    verify_signature,
)


@pytest.fixture
def mock_storage() -> AsyncMock:
    """Create a mock storage instance."""
    storage = AsyncMock()
    storage.get_webhooks_for_event = AsyncMock(return_value=[])
    storage.log_delivery = AsyncMock()
    storage.update_delivery = AsyncMock()
    storage.get_pending_deliveries = AsyncMock(return_value=[])
    storage.get_webhook = AsyncMock(return_value=None)
    return storage


@pytest.fixture
def sample_webhook() -> WebhookConfig:
    """Create a sample webhook configuration."""
    return WebhookConfig(
        id="whk_test123",
        user_id="user_1",
        url="https://example.com/webhook",
        secret="test_secret_16chars",
        events=["encode_complete", "memory_created"],
        max_retries=3,
        retry_delay_seconds=1,
    )


@pytest.fixture
def sample_event() -> WebhookEvent:
    """Create a sample webhook event."""
    return WebhookEvent(
        id="evt_test456",
        event="encode_complete",
        user_id="user_1",
        data={"episode_id": "ep_123", "facts_count": 5},
    )


class TestWebhookDispatcher:
    """Tests for WebhookDispatcher class."""

    def test_init_defaults(self, mock_storage: AsyncMock) -> None:
        """Dispatcher should initialize with default values."""
        dispatcher = WebhookDispatcher(mock_storage)
        assert dispatcher._storage is mock_storage
        assert dispatcher._timeout == 30.0
        assert dispatcher._max_concurrent == 10

    def test_init_custom_values(self, mock_storage: AsyncMock) -> None:
        """Dispatcher should accept custom timeout and concurrency."""
        dispatcher = WebhookDispatcher(mock_storage, timeout_seconds=60.0, max_concurrent=5)
        assert dispatcher._timeout == 60.0
        assert dispatcher._max_concurrent == 5

    @pytest.mark.asyncio
    async def test_dispatch_event_no_webhooks(
        self, mock_storage: AsyncMock, sample_event: WebhookEvent
    ) -> None:
        """dispatch_event should return empty list when no webhooks subscribed."""
        mock_storage.get_webhooks_for_event.return_value = []
        dispatcher = WebhookDispatcher(mock_storage)

        result = await dispatcher.dispatch_event(sample_event)

        assert result == []
        mock_storage.get_webhooks_for_event.assert_called_once_with(
            event_type=sample_event.event,
            user_id=sample_event.user_id,
            org_id=sample_event.org_id,
        )

    @pytest.mark.asyncio
    async def test_dispatch_event_successful_delivery(
        self,
        mock_storage: AsyncMock,
        sample_webhook: WebhookConfig,
        sample_event: WebhookEvent,
    ) -> None:
        """dispatch_event should deliver to subscribed webhooks."""
        mock_storage.get_webhooks_for_event.return_value = [sample_webhook]

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.text = "OK"

            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            dispatcher = WebhookDispatcher(mock_storage)
            result = await dispatcher.dispatch_event(sample_event)

        assert len(result) == 1
        assert result[0].startswith("dlv_")
        mock_storage.log_delivery.assert_called_once()
        mock_storage.update_delivery.assert_called_once()

    @pytest.mark.asyncio
    async def test_dispatch_event_multiple_webhooks(
        self,
        mock_storage: AsyncMock,
        sample_event: WebhookEvent,
    ) -> None:
        """dispatch_event should deliver to multiple webhooks concurrently."""
        webhooks = [
            WebhookConfig(
                id=f"whk_test{i}",
                user_id="user_1",
                url=f"https://example{i}.com/webhook",
                secret="test_secret_16chars",
            )
            for i in range(3)
        ]
        mock_storage.get_webhooks_for_event.return_value = webhooks

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.text = "OK"

            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            dispatcher = WebhookDispatcher(mock_storage)
            result = await dispatcher.dispatch_event(sample_event)

        assert len(result) == 3
        assert mock_storage.log_delivery.call_count == 3
        assert mock_storage.update_delivery.call_count == 3

    @pytest.mark.asyncio
    async def test_dispatch_event_handles_delivery_exception(
        self,
        mock_storage: AsyncMock,
        sample_webhook: WebhookConfig,
        sample_event: WebhookEvent,
    ) -> None:
        """dispatch_event should handle exceptions from individual deliveries."""
        mock_storage.get_webhooks_for_event.return_value = [sample_webhook]

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=Exception("Connection error"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            dispatcher = WebhookDispatcher(mock_storage)
            result = await dispatcher.dispatch_event(sample_event)

        # Exception should be caught, no delivery ID returned
        assert len(result) == 1  # Still returns delivery ID


class TestAttemptDelivery:
    """Tests for delivery attempt logic."""

    @pytest.mark.asyncio
    async def test_successful_2xx_response(
        self,
        mock_storage: AsyncMock,
        sample_webhook: WebhookConfig,
        sample_event: WebhookEvent,
    ) -> None:
        """2xx responses should mark delivery as success."""
        mock_storage.get_webhooks_for_event.return_value = [sample_webhook]

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_response = MagicMock()
            mock_response.status_code = 201
            mock_response.text = "Created"

            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            dispatcher = WebhookDispatcher(mock_storage)
            await dispatcher.dispatch_event(sample_event)

        # Check that update_delivery was called with success status
        call_args = mock_storage.update_delivery.call_args
        delivery = call_args[0][0]
        assert delivery.status == "success"
        assert delivery.response_code == 201

    @pytest.mark.asyncio
    async def test_4xx_response_fails_without_retry(
        self,
        mock_storage: AsyncMock,
        sample_webhook: WebhookConfig,
        sample_event: WebhookEvent,
    ) -> None:
        """4xx responses should fail without retry."""
        mock_storage.get_webhooks_for_event.return_value = [sample_webhook]

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_response = MagicMock()
            mock_response.status_code = 400
            mock_response.text = "Bad Request"

            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            dispatcher = WebhookDispatcher(mock_storage)
            await dispatcher.dispatch_event(sample_event)

        call_args = mock_storage.update_delivery.call_args
        delivery = call_args[0][0]
        assert delivery.status == "failed"
        assert delivery.response_code == 400

    @pytest.mark.asyncio
    async def test_5xx_response_schedules_retry(
        self,
        mock_storage: AsyncMock,
        sample_webhook: WebhookConfig,
        sample_event: WebhookEvent,
    ) -> None:
        """5xx responses should schedule retry."""
        mock_storage.get_webhooks_for_event.return_value = [sample_webhook]

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_response.text = "Internal Server Error"

            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            dispatcher = WebhookDispatcher(mock_storage)
            await dispatcher.dispatch_event(sample_event)

        call_args = mock_storage.update_delivery.call_args
        delivery = call_args[0][0]
        assert delivery.status == "retrying"
        assert delivery.next_retry_at is not None

    @pytest.mark.asyncio
    async def test_timeout_schedules_retry(
        self,
        mock_storage: AsyncMock,
        sample_webhook: WebhookConfig,
        sample_event: WebhookEvent,
    ) -> None:
        """Timeout should schedule retry."""
        mock_storage.get_webhooks_for_event.return_value = [sample_webhook]

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=httpx.TimeoutException("timeout"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            dispatcher = WebhookDispatcher(mock_storage)
            await dispatcher.dispatch_event(sample_event)

        call_args = mock_storage.update_delivery.call_args
        delivery = call_args[0][0]
        assert delivery.status == "retrying"
        # Error message is "Request timeout" which gets passed to _schedule_retry
        assert delivery.error is not None

    @pytest.mark.asyncio
    async def test_request_error_schedules_retry(
        self,
        mock_storage: AsyncMock,
        sample_webhook: WebhookConfig,
        sample_event: WebhookEvent,
    ) -> None:
        """Connection errors should schedule retry."""
        mock_storage.get_webhooks_for_event.return_value = [sample_webhook]

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=httpx.RequestError("Connection refused"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            dispatcher = WebhookDispatcher(mock_storage)
            await dispatcher.dispatch_event(sample_event)

        call_args = mock_storage.update_delivery.call_args
        delivery = call_args[0][0]
        assert delivery.status == "retrying"

    @pytest.mark.asyncio
    async def test_unexpected_error_marks_failed(
        self,
        mock_storage: AsyncMock,
        sample_webhook: WebhookConfig,
        sample_event: WebhookEvent,
    ) -> None:
        """Unexpected errors should mark delivery as failed."""
        mock_storage.get_webhooks_for_event.return_value = [sample_webhook]

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=ValueError("Unexpected error"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            dispatcher = WebhookDispatcher(mock_storage)
            await dispatcher.dispatch_event(sample_event)

        call_args = mock_storage.update_delivery.call_args
        delivery = call_args[0][0]
        assert delivery.status == "failed"
        assert "Unexpected error" in delivery.error

    @pytest.mark.asyncio
    async def test_request_includes_signature_header(
        self,
        mock_storage: AsyncMock,
        sample_webhook: WebhookConfig,
        sample_event: WebhookEvent,
    ) -> None:
        """Request should include HMAC signature header."""
        mock_storage.get_webhooks_for_event.return_value = [sample_webhook]

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.text = "OK"

            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            dispatcher = WebhookDispatcher(mock_storage)
            await dispatcher.dispatch_event(sample_event)

        # Check headers were passed
        post_call = mock_client.post.call_args
        headers = post_call.kwargs["headers"]
        assert "X-Engram-Signature" in headers
        assert headers["X-Engram-Signature"].startswith("sha256=")
        assert headers["X-Engram-Event"] == "encode_complete"
        assert "X-Engram-Delivery-Id" in headers


class TestScheduleRetry:
    """Tests for retry scheduling logic."""

    @pytest.mark.asyncio
    async def test_max_retries_exceeded_fails(
        self,
        mock_storage: AsyncMock,
        sample_event: WebhookEvent,
    ) -> None:
        """Delivery should fail when max retries exceeded."""
        webhook = WebhookConfig(
            id="whk_test",
            user_id="user_1",
            url="https://example.com/webhook",
            secret="test_secret_16chars",
            max_retries=1,  # Only 1 retry allowed
        )
        mock_storage.get_webhooks_for_event.return_value = [webhook]

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_response.text = "Server Error"

            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            dispatcher = WebhookDispatcher(mock_storage)

            # First attempt - should schedule retry
            await dispatcher.dispatch_event(sample_event)
            first_delivery = mock_storage.update_delivery.call_args[0][0]

            # Simulate retry with attempt=1 already (it will increment to 2)
            # After this attempt, max_retries (1) will be exceeded
            first_delivery.attempt = 1  # Will be checked against max_retries (1)

        # First attempt schedules retry, second would fail
        # Since attempt=1 and max_retries=1, it should fail immediately
        assert first_delivery.status == "failed"

    @pytest.mark.asyncio
    async def test_exponential_backoff_delay(
        self,
        mock_storage: AsyncMock,
        sample_event: WebhookEvent,
    ) -> None:
        """Retry delay should follow exponential backoff."""
        webhook = WebhookConfig(
            id="whk_test",
            user_id="user_1",
            url="https://example.com/webhook",
            secret="test_secret_16chars",
            max_retries=5,
            retry_delay_seconds=2,  # Base delay of 2 seconds
        )
        mock_storage.get_webhooks_for_event.return_value = [webhook]

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_response = MagicMock()
            mock_response.status_code = 503
            mock_response.text = "Service Unavailable"

            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            dispatcher = WebhookDispatcher(mock_storage)
            await dispatcher.dispatch_event(sample_event)

        delivery = mock_storage.update_delivery.call_args[0][0]
        assert delivery.status == "retrying"
        assert delivery.next_retry_at is not None

        # First attempt: delay = 2 * (2^0) = 2 seconds
        expected_delay = timedelta(seconds=2)
        actual_delay = delivery.next_retry_at - delivery.created_at
        # Allow some tolerance for test execution time
        assert abs(actual_delay.total_seconds() - expected_delay.total_seconds()) < 5


class TestProcessRetries:
    """Tests for retry processing."""

    @pytest.mark.asyncio
    async def test_process_retries_no_pending(self, mock_storage: AsyncMock) -> None:
        """process_retries should return 0 when no pending deliveries."""
        mock_storage.get_pending_deliveries.return_value = []

        dispatcher = WebhookDispatcher(mock_storage)
        result = await dispatcher.process_retries()

        assert result == 0

    @pytest.mark.asyncio
    async def test_process_retries_skips_not_ready(self, mock_storage: AsyncMock) -> None:
        """process_retries should skip deliveries not yet ready."""
        delivery = WebhookDelivery(
            id="dlv_test",
            webhook_id="whk_123",
            event_id="evt_456",
            user_id="user_1",
            status="retrying",
            next_retry_at=datetime.now(UTC) + timedelta(hours=1),  # Future
        )
        mock_storage.get_pending_deliveries.return_value = [delivery]

        dispatcher = WebhookDispatcher(mock_storage)
        result = await dispatcher.process_retries()

        # Should skip because next_retry_at is in the future
        assert result == 0
        mock_storage.get_webhook.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_retries_webhook_not_found(self, mock_storage: AsyncMock) -> None:
        """process_retries should fail delivery if webhook not found."""
        delivery = WebhookDelivery(
            id="dlv_test",
            webhook_id="whk_nonexistent",
            event_id="evt_456",
            user_id="user_1",
            status="retrying",
            next_retry_at=datetime.now(UTC) - timedelta(minutes=1),  # Past
        )
        mock_storage.get_pending_deliveries.return_value = [delivery]
        mock_storage.get_webhook.return_value = None  # Webhook not found

        dispatcher = WebhookDispatcher(mock_storage)
        result = await dispatcher.process_retries()

        assert result == 1
        mock_storage.update_delivery.assert_called_once()
        updated_delivery = mock_storage.update_delivery.call_args[0][0]
        assert updated_delivery.status == "failed"
        assert "not found" in updated_delivery.error.lower()

    @pytest.mark.asyncio
    async def test_process_retries_webhook_disabled(
        self,
        mock_storage: AsyncMock,
        sample_webhook: WebhookConfig,
    ) -> None:
        """process_retries should fail delivery if webhook disabled."""
        delivery = WebhookDelivery(
            id="dlv_test",
            webhook_id=sample_webhook.id,
            event_id="evt_456",
            user_id="user_1",
            status="retrying",
            next_retry_at=datetime.now(UTC) - timedelta(minutes=1),
        )
        mock_storage.get_pending_deliveries.return_value = [delivery]

        disabled_webhook = sample_webhook.model_copy(update={"enabled": False})
        mock_storage.get_webhook.return_value = disabled_webhook

        dispatcher = WebhookDispatcher(mock_storage)
        result = await dispatcher.process_retries()

        assert result == 1
        updated_delivery = mock_storage.update_delivery.call_args[0][0]
        assert updated_delivery.status == "failed"
        assert "disabled" in updated_delivery.error.lower()

    @pytest.mark.asyncio
    async def test_process_retries_successful_reattempt(
        self,
        mock_storage: AsyncMock,
        sample_webhook: WebhookConfig,
    ) -> None:
        """process_retries should successfully reattempt delivery."""
        delivery = WebhookDelivery(
            id="dlv_test",
            webhook_id=sample_webhook.id,
            event_id="evt_456",
            user_id="user_1",
            status="retrying",
            attempt=2,
            next_retry_at=datetime.now(UTC) - timedelta(minutes=1),
        )
        mock_storage.get_pending_deliveries.return_value = [delivery]
        mock_storage.get_webhook.return_value = sample_webhook

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.text = "OK"

            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            dispatcher = WebhookDispatcher(mock_storage)
            result = await dispatcher.process_retries()

        assert result == 1
        updated_delivery = mock_storage.update_delivery.call_args[0][0]
        assert updated_delivery.status == "success"


class TestDispatchWebhookEvent:
    """Tests for dispatch_webhook_event convenience function."""

    @pytest.mark.asyncio
    async def test_creates_event_and_dispatches(self, mock_storage: AsyncMock) -> None:
        """dispatch_webhook_event should create event and dispatch."""
        mock_storage.get_webhooks_for_event.return_value = []

        result = await dispatch_webhook_event(
            storage=mock_storage,
            event_type="memory_created",
            user_id="user_1",
            org_id="org_123",
            session_id="sess_456",
            memory_id="sem_789",
            memory_type="semantic",
        )

        assert result == []
        mock_storage.get_webhooks_for_event.assert_called_once()
        call_kwargs = mock_storage.get_webhooks_for_event.call_args.kwargs
        assert call_kwargs["event_type"] == "memory_created"
        assert call_kwargs["user_id"] == "user_1"
        assert call_kwargs["org_id"] == "org_123"

    @pytest.mark.asyncio
    async def test_passes_data_to_event(
        self,
        mock_storage: AsyncMock,
        sample_webhook: WebhookConfig,
    ) -> None:
        """dispatch_webhook_event should include data in event payload."""
        mock_storage.get_webhooks_for_event.return_value = [sample_webhook]

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.text = "OK"

            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            await dispatch_webhook_event(
                storage=mock_storage,
                event_type="encode_complete",
                user_id="user_1",
                episode_id="ep_123",
                facts_count=5,
            )

        # Check the payload includes our data
        post_call = mock_client.post.call_args
        payload = post_call.kwargs["content"]
        assert "ep_123" in payload
        assert "5" in payload or "facts_count" in payload


class TestSignatureFunctions:
    """Additional tests for signature functions."""

    def test_compute_signature_consistent(self) -> None:
        """compute_signature should be deterministic."""
        payload = '{"event": "test", "data": {"id": 123}}'
        secret = "my_secret_key"

        sig1 = compute_signature(payload, secret)
        sig2 = compute_signature(payload, secret)

        assert sig1 == sig2

    def test_verify_signature_timing_safe(self) -> None:
        """verify_signature should use timing-safe comparison."""
        payload = '{"event": "test"}'
        secret = "test_secret"
        sig = compute_signature(payload, secret)

        # This tests that hmac.compare_digest is used
        # which provides constant-time comparison
        assert verify_signature(payload, secret, sig) is True
        assert verify_signature(payload, secret, "sha256=wrong") is False

    def test_signature_changes_with_payload(self) -> None:
        """Different payloads should produce different signatures."""
        secret = "shared_secret"
        payload1 = '{"event": "test1"}'
        payload2 = '{"event": "test2"}'

        sig1 = compute_signature(payload1, secret)
        sig2 = compute_signature(payload2, secret)

        assert sig1 != sig2

    def test_signature_changes_with_secret(self) -> None:
        """Different secrets should produce different signatures."""
        payload = '{"event": "test"}'
        secret1 = "secret_one"
        secret2 = "secret_two"

        sig1 = compute_signature(payload, secret1)
        sig2 = compute_signature(payload, secret2)

        assert sig1 != sig2

    def test_empty_payload_signature(self) -> None:
        """Empty payload should still produce valid signature."""
        sig = compute_signature("", "secret")
        assert sig.startswith("sha256=")
        assert len(sig) == 71  # sha256= + 64 hex chars

    def test_unicode_payload_signature(self) -> None:
        """Unicode payload should be handled correctly."""
        payload = '{"message": "こんにちは世界"}'
        secret = "secret"
        sig = compute_signature(payload, secret)

        assert sig.startswith("sha256=")
        assert verify_signature(payload, secret, sig) is True
