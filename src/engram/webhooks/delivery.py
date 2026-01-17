"""Webhook delivery with HMAC signatures and exponential backoff retry.

Implements reliable webhook delivery following industry best practices:
- HMAC-SHA256 signatures for request verification
- Exponential backoff for transient failures
- Delivery logging for debugging and audit
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from engram.models import EventType, WebhookConfig, WebhookDelivery, WebhookEvent
    from engram.storage import EngramStorage

logger = logging.getLogger(__name__)


def compute_signature(payload: str, secret: str) -> str:
    """Compute HMAC-SHA256 signature for a webhook payload.

    Args:
        payload: JSON string payload to sign.
        secret: Shared secret for HMAC.

    Returns:
        Signature in format "sha256=<hex_digest>".
    """
    signature = hmac.new(
        key=secret.encode("utf-8"),
        msg=payload.encode("utf-8"),
        digestmod=hashlib.sha256,
    ).hexdigest()
    return f"sha256={signature}"


def verify_signature(payload: str, secret: str, signature: str) -> bool:
    """Verify HMAC-SHA256 signature for a webhook payload.

    Args:
        payload: JSON string payload that was signed.
        secret: Shared secret for HMAC.
        signature: Signature to verify (format: "sha256=<hex_digest>").

    Returns:
        True if signature is valid, False otherwise.
    """
    expected = compute_signature(payload, secret)
    return hmac.compare_digest(expected, signature)


class WebhookDispatcher:
    """Dispatches webhook events to registered endpoints.

    Handles:
    - Finding webhooks subscribed to an event type
    - Signing payloads with HMAC-SHA256
    - Delivering with exponential backoff retry
    - Logging delivery attempts for debugging

    Example:
        ```python
        dispatcher = WebhookDispatcher(storage)

        # Dispatch an event to all subscribed webhooks
        await dispatcher.dispatch_event(event)

        # Process pending retries
        await dispatcher.process_retries()
        ```
    """

    def __init__(
        self,
        storage: EngramStorage,
        timeout_seconds: float = 30.0,
        max_concurrent: int = 10,
    ) -> None:
        """Initialize the webhook dispatcher.

        Args:
            storage: EngramStorage instance for webhook/delivery data.
            timeout_seconds: HTTP request timeout.
            max_concurrent: Maximum concurrent deliveries.
        """
        self._storage = storage
        self._timeout = timeout_seconds
        self._max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def dispatch_event(self, event: WebhookEvent) -> list[str]:
        """Dispatch an event to all subscribed webhooks.

        Finds all webhooks subscribed to the event type and delivers
        the event to each one concurrently.

        Args:
            event: WebhookEvent to dispatch.

        Returns:
            List of delivery IDs created.
        """
        webhooks = await self._storage.get_webhooks_for_event(
            event_type=event.event,
            user_id=event.user_id,
            org_id=event.org_id,
        )

        if not webhooks:
            logger.debug(
                "No webhooks subscribed to event %s for user %s", event.event, event.user_id
            )
            return []

        delivery_ids: list[str] = []
        tasks = []

        for webhook in webhooks:
            task = self._deliver_to_webhook(webhook, event)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                logger.error("Webhook delivery failed: %s", result)
            elif isinstance(result, str):
                delivery_ids.append(result)

        return delivery_ids

    async def _deliver_to_webhook(
        self,
        webhook: WebhookConfig,
        event: WebhookEvent,
    ) -> str:
        """Deliver an event to a single webhook.

        Creates a delivery record and attempts delivery. On failure,
        schedules retry if within retry limits.

        Args:
            webhook: Webhook configuration.
            event: Event to deliver.

        Returns:
            Delivery ID.
        """
        from engram.models import WebhookDelivery

        # Create delivery record
        delivery = WebhookDelivery(
            webhook_id=webhook.id,
            event_id=event.id,
            user_id=webhook.user_id,
            org_id=webhook.org_id,
            status="pending",
            attempt=1,
        )

        await self._storage.log_delivery(delivery)

        # Attempt delivery
        async with self._semaphore:
            await self._attempt_delivery(webhook, event, delivery)

        return delivery.id

    async def _attempt_delivery(
        self,
        webhook: WebhookConfig,
        event: WebhookEvent,
        delivery: WebhookDelivery,
    ) -> None:
        """Attempt to deliver a webhook event.

        Args:
            webhook: Webhook configuration.
            event: Event to deliver.
            delivery: Delivery record to update.
        """
        payload = event.model_dump_json()
        signature = compute_signature(payload, webhook.secret)

        headers = {
            "Content-Type": "application/json",
            "X-Engram-Signature": signature,
            "X-Engram-Event": event.event,
            "X-Engram-Delivery-Id": delivery.id,
        }

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    str(webhook.url),
                    content=payload,
                    headers=headers,
                )

            if response.status_code >= 200 and response.status_code < 300:
                delivery.mark_success(
                    response_code=response.status_code,
                    response_body=response.text[:1000] if response.text else None,
                )
                logger.info(
                    "Webhook delivered: %s to %s (status %d)",
                    event.event,
                    webhook.url,
                    response.status_code,
                )
            elif response.status_code >= 500:
                # Server error - retry
                await self._schedule_retry(webhook, event, delivery, response)
            else:
                # Client error (4xx) - don't retry
                delivery.mark_failed(
                    error=f"HTTP {response.status_code}: {response.text[:200]}",
                    response_code=response.status_code,
                    response_body=response.text[:1000] if response.text else None,
                )
                logger.warning(
                    "Webhook rejected: %s to %s (status %d)",
                    event.event,
                    webhook.url,
                    response.status_code,
                )

        except httpx.TimeoutException:
            await self._schedule_retry(webhook, event, delivery, error="Request timeout")
        except httpx.RequestError as e:
            await self._schedule_retry(webhook, event, delivery, error=str(e))
        except Exception as e:
            delivery.mark_failed(error=f"Unexpected error: {e}")
            logger.exception("Webhook delivery error: %s", e)

        await self._storage.update_delivery(delivery)

    async def _schedule_retry(
        self,
        webhook: WebhookConfig,
        event: WebhookEvent,
        delivery: WebhookDelivery,
        response: httpx.Response | None = None,
        error: str | None = None,
    ) -> None:
        """Schedule a retry for a failed delivery.

        Uses exponential backoff: 1s, 2s, 4s, 8s, 16s, etc.

        Args:
            webhook: Webhook configuration.
            event: Event being delivered.
            delivery: Delivery record to update.
            response: HTTP response if received.
            error: Error message if no response.
        """
        if delivery.attempt >= webhook.max_retries:
            error_msg = error or f"HTTP {response.status_code}" if response else "Unknown error"
            delivery.mark_failed(
                error=f"Max retries exceeded: {error_msg}",
                response_code=response.status_code if response else None,
                response_body=response.text[:1000] if response and response.text else None,
            )
            logger.warning(
                "Webhook max retries exceeded: %s to %s after %d attempts",
                event.event,
                webhook.url,
                delivery.attempt,
            )
            return

        # Calculate exponential backoff delay
        delay_seconds = webhook.retry_delay_seconds * (2 ** (delivery.attempt - 1))
        next_retry = datetime.now(UTC) + timedelta(seconds=delay_seconds)

        error_msg = error or f"HTTP {response.status_code}" if response else "Unknown error"
        delivery.mark_retrying(
            next_retry_at=next_retry,
            error=error_msg,
            response_code=response.status_code if response else None,
        )
        delivery.attempt += 1

        logger.info(
            "Webhook scheduled for retry: %s to %s (attempt %d at %s)",
            event.event,
            webhook.url,
            delivery.attempt,
            next_retry.isoformat(),
        )

    async def process_retries(self, user_id: str | None = None) -> int:
        """Process pending webhook retries.

        Finds deliveries marked for retry where next_retry_at has passed,
        and attempts to deliver them.

        Args:
            user_id: Optional user filter.

        Returns:
            Number of retries processed.
        """
        deliveries = await self._storage.get_pending_deliveries(
            user_id=user_id,
            limit=100,
        )

        now = datetime.now(UTC)
        processed = 0

        for delivery in deliveries:
            # Skip if not ready for retry
            if delivery.next_retry_at and delivery.next_retry_at > now:
                continue

            webhook = await self._storage.get_webhook(
                webhook_id=delivery.webhook_id,
                user_id=delivery.user_id,
                org_id=delivery.org_id,
            )

            if webhook is None or not webhook.enabled:
                delivery.mark_failed(error="Webhook not found or disabled")
                await self._storage.update_delivery(delivery)
                processed += 1
                continue

            # Reconstruct event from delivery
            from engram.models import WebhookEvent

            # We need to fetch the original event data
            # For retries, we'll create a minimal event for the retry attempt
            # Note: Using encode_complete as a placeholder event type for retry
            event = WebhookEvent(
                id=delivery.event_id,
                event="encode_complete",  # Placeholder for retry delivery
                user_id=delivery.user_id,
                org_id=delivery.org_id,
                data={"retry_attempt": delivery.attempt},
            )

            async with self._semaphore:
                await self._attempt_delivery(webhook, event, delivery)

            processed += 1

        return processed


async def dispatch_webhook_event(
    storage: EngramStorage,
    event_type: EventType,
    user_id: str,
    org_id: str | None = None,
    session_id: str | None = None,
    **data: object,
) -> list[str]:
    """Convenience function to create and dispatch a webhook event.

    Args:
        storage: EngramStorage instance.
        event_type: Type of event.
        user_id: User who triggered the event.
        org_id: Optional organization.
        session_id: Optional session context.
        **data: Event-specific payload data.

    Returns:
        List of delivery IDs created.
    """
    from engram.models import WebhookEvent

    event = WebhookEvent(
        event=event_type,
        user_id=user_id,
        org_id=org_id,
        session_id=session_id,
        data=dict(data),
    )

    dispatcher = WebhookDispatcher(storage)
    return await dispatcher.dispatch_event(event)
