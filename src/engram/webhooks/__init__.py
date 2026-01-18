"""Webhook delivery system for Engram.

Provides HMAC-signed webhook delivery with exponential backoff retry.

Example:
    ```python
    from engram.webhooks import WebhookDispatcher, dispatch_webhook_event

    # Using dispatcher directly
    dispatcher = WebhookDispatcher(storage)
    await dispatcher.dispatch_event(event)

    # Using convenience function
    await dispatch_webhook_event(
        storage,
        event_type="encode_complete",
        user_id="user_123",
        episode_id="ep_456",
        facts_count=3,
    )
    ```
"""

from .delivery import (
    WebhookDispatcher,
    compute_signature,
    dispatch_webhook_event,
    verify_signature,
)

__all__ = [
    "WebhookDispatcher",
    "compute_signature",
    "dispatch_webhook_event",
    "verify_signature",
]
