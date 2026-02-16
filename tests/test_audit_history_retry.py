"""Tests for audit and history retry behavior (SPEC-001).

Verifies that log_audit, log_history, get_audit_log, get_memory_history,
and get_user_history all retry on transient Qdrant failures via @qdrant_retry.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from qdrant_client.http.exceptions import UnexpectedResponse

from engram.models import AuditEntry, HistoryEntry
from engram.storage.retry import _is_retryable_qdrant_error


class TestAuditRetry:
    """Tests for audit write retry behavior."""

    @pytest.mark.asyncio
    async def test_log_audit_retries_on_transient_failure(self) -> None:
        """log_audit should retry on ConnectError and succeed on second attempt."""
        from engram.storage.audit import AuditMixin

        mixin = AuditMixin()
        mixin._collection_name = MagicMock(return_value="engram_audit")
        mixin._build_key = MagicMock(return_value="key_123")
        mixin._key_to_point_id = MagicMock(return_value="point_123")
        mixin._embedding_dim = 384

        # Fail on first call, succeed on second
        mock_client = AsyncMock()
        mock_client.upsert = AsyncMock(side_effect=[httpx.ConnectError("Connection refused"), None])
        mixin.client = mock_client

        entry = AuditEntry(
            event="encode",
            user_id="test_user",
            details={"episode_id": "ep_123", "facts_count": 3},
        )

        result = await mixin.log_audit(entry)

        assert result == entry.id
        assert mock_client.upsert.call_count == 2

    @pytest.mark.asyncio
    async def test_log_audit_raises_after_max_retries(self) -> None:
        """log_audit should raise after exhausting retry attempts."""
        from engram.storage.audit import AuditMixin

        mixin = AuditMixin()
        mixin._collection_name = MagicMock(return_value="engram_audit")
        mixin._build_key = MagicMock(return_value="key_123")
        mixin._key_to_point_id = MagicMock(return_value="point_123")
        mixin._embedding_dim = 384

        mock_client = AsyncMock()
        mock_client.upsert = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
        mixin.client = mock_client

        entry = AuditEntry(
            event="encode",
            user_id="test_user",
            details={"episode_id": "ep_123", "facts_count": 3},
        )

        with pytest.raises(httpx.ConnectError):
            await mixin.log_audit(entry)

        # 3 attempts (initial + 2 retries)
        assert mock_client.upsert.call_count == 3

    @pytest.mark.asyncio
    async def test_log_audit_does_not_retry_client_error(self) -> None:
        """log_audit should not retry on 4xx client errors."""
        from engram.storage.audit import AuditMixin

        mixin = AuditMixin()
        mixin._collection_name = MagicMock(return_value="engram_audit")
        mixin._build_key = MagicMock(return_value="key_123")
        mixin._key_to_point_id = MagicMock(return_value="point_123")
        mixin._embedding_dim = 384

        client_error = UnexpectedResponse(
            status_code=400,
            reason_phrase="Bad Request",
            content=b"bad request",
            headers=httpx.Headers({}),
        )

        mock_client = AsyncMock()
        mock_client.upsert = AsyncMock(side_effect=client_error)
        mixin.client = mock_client

        entry = AuditEntry(
            event="encode",
            user_id="test_user",
            details={"episode_id": "ep_123", "facts_count": 3},
        )

        with pytest.raises(UnexpectedResponse):
            await mixin.log_audit(entry)

        # Only 1 attempt — not retried
        assert mock_client.upsert.call_count == 1

    @pytest.mark.asyncio
    async def test_log_audit_retries_on_server_error(self) -> None:
        """log_audit should retry on 5xx server errors."""
        from engram.storage.audit import AuditMixin

        mixin = AuditMixin()
        mixin._collection_name = MagicMock(return_value="engram_audit")
        mixin._build_key = MagicMock(return_value="key_123")
        mixin._key_to_point_id = MagicMock(return_value="point_123")
        mixin._embedding_dim = 384

        server_error = UnexpectedResponse(
            status_code=503,
            reason_phrase="Service Unavailable",
            content=b"overloaded",
            headers=httpx.Headers({}),
        )

        mock_client = AsyncMock()
        mock_client.upsert = AsyncMock(side_effect=[server_error, None])
        mixin.client = mock_client

        entry = AuditEntry(
            event="encode",
            user_id="test_user",
            details={"episode_id": "ep_123", "facts_count": 3},
        )

        result = await mixin.log_audit(entry)

        assert result == entry.id
        assert mock_client.upsert.call_count == 2


class TestAuditReadRetry:
    """Tests for audit read retry behavior."""

    @pytest.mark.asyncio
    async def test_get_audit_log_retries_on_transient_failure(self) -> None:
        """get_audit_log should retry on ConnectError."""
        from engram.storage.audit import AuditMixin

        mixin = AuditMixin()
        mixin._collection_name = MagicMock(return_value="engram_audit")
        mixin._payload_to_memory = MagicMock(
            return_value=AuditEntry(
                event="encode",
                user_id="test_user",
                details={"episode_id": "ep_123", "facts_count": 1},
            )
        )

        mock_point = MagicMock()
        mock_point.payload = {"event": "encode", "user_id": "test_user"}

        mock_client = AsyncMock()
        mock_client.scroll = AsyncMock(
            side_effect=[
                httpx.ConnectError("Connection refused"),
                ([mock_point], None),
            ]
        )
        mixin.client = mock_client

        result = await mixin.get_audit_log("test_user")

        assert len(result) == 1
        assert mock_client.scroll.call_count == 2


class TestHistoryRetry:
    """Tests for history write retry behavior."""

    @pytest.mark.asyncio
    async def test_log_history_retries_on_transient_failure(self) -> None:
        """log_history should retry on ConnectError and succeed on second attempt."""
        from engram.storage.history import HistoryMixin

        mixin = HistoryMixin()
        mixin._collection_name = MagicMock(return_value="engram_history")
        mixin._build_key = MagicMock(return_value="key_123")
        mixin._key_to_point_id = MagicMock(return_value="point_123")
        mixin._embedding_dim = 384

        mock_client = AsyncMock()
        mock_client.upsert = AsyncMock(side_effect=[httpx.ConnectError("Connection refused"), None])
        mixin.client = mock_client

        entry = HistoryEntry(
            memory_id="sem_123",
            memory_type="semantic",
            user_id="test_user",
            change_type="created",
            trigger="consolidation",
        )

        result = await mixin.log_history(entry)

        assert result == entry.id
        assert mock_client.upsert.call_count == 2

    @pytest.mark.asyncio
    async def test_log_history_raises_after_max_retries(self) -> None:
        """log_history should raise after exhausting retry attempts."""
        from engram.storage.history import HistoryMixin

        mixin = HistoryMixin()
        mixin._collection_name = MagicMock(return_value="engram_history")
        mixin._build_key = MagicMock(return_value="key_123")
        mixin._key_to_point_id = MagicMock(return_value="point_123")
        mixin._embedding_dim = 384

        mock_client = AsyncMock()
        mock_client.upsert = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
        mixin.client = mock_client

        entry = HistoryEntry(
            memory_id="sem_123",
            memory_type="semantic",
            user_id="test_user",
            change_type="created",
            trigger="consolidation",
        )

        with pytest.raises(httpx.ConnectError):
            await mixin.log_history(entry)

        assert mock_client.upsert.call_count == 3

    @pytest.mark.asyncio
    async def test_log_history_retries_on_timeout(self) -> None:
        """log_history should retry on ReadTimeout."""
        from engram.storage.history import HistoryMixin

        mixin = HistoryMixin()
        mixin._collection_name = MagicMock(return_value="engram_history")
        mixin._build_key = MagicMock(return_value="key_123")
        mixin._key_to_point_id = MagicMock(return_value="point_123")
        mixin._embedding_dim = 384

        mock_client = AsyncMock()
        mock_client.upsert = AsyncMock(side_effect=[httpx.ReadTimeout("Read timed out"), None])
        mixin.client = mock_client

        entry = HistoryEntry(
            memory_id="sem_123",
            memory_type="semantic",
            user_id="test_user",
            change_type="updated",
            trigger="decay",
        )

        result = await mixin.log_history(entry)

        assert result == entry.id
        assert mock_client.upsert.call_count == 2


class TestHistoryReadRetry:
    """Tests for history read retry behavior."""

    @pytest.mark.asyncio
    async def test_get_memory_history_retries_on_transient_failure(self) -> None:
        """get_memory_history should retry on ConnectError."""
        from engram.storage.history import HistoryMixin

        mixin = HistoryMixin()
        mixin._collection_name = MagicMock(return_value="engram_history")
        mixin._payload_to_memory = MagicMock(
            return_value=HistoryEntry(
                memory_id="sem_123",
                memory_type="semantic",
                user_id="test_user",
                change_type="created",
                trigger="consolidation",
            )
        )

        mock_point = MagicMock()
        mock_point.payload = {"memory_id": "sem_123"}

        mock_client = AsyncMock()
        mock_client.scroll = AsyncMock(
            side_effect=[
                httpx.ConnectError("Connection refused"),
                ([mock_point], None),
            ]
        )
        mixin.client = mock_client

        result = await mixin.get_memory_history("sem_123", "test_user")

        assert len(result) == 1
        assert mock_client.scroll.call_count == 2

    @pytest.mark.asyncio
    async def test_get_user_history_retries_on_transient_failure(self) -> None:
        """get_user_history should retry on ConnectError."""
        from engram.storage.history import HistoryMixin

        mixin = HistoryMixin()
        mixin._collection_name = MagicMock(return_value="engram_history")
        mixin._payload_to_memory = MagicMock(
            return_value=HistoryEntry(
                memory_id="sem_123",
                memory_type="semantic",
                user_id="test_user",
                change_type="created",
                trigger="consolidation",
            )
        )

        mock_point = MagicMock()
        mock_point.payload = {"memory_id": "sem_123"}

        mock_client = AsyncMock()
        mock_client.scroll = AsyncMock(
            side_effect=[
                httpx.ConnectError("Connection refused"),
                ([mock_point], None),
            ]
        )
        mixin.client = mock_client

        result = await mixin.get_user_history("test_user")

        assert len(result) == 1
        assert mock_client.scroll.call_count == 2


class TestRetryPredicateConsistency:
    """Verify the retry predicate handles all expected error types."""

    def test_connect_error_is_retryable(self) -> None:
        """ConnectError should be retried."""
        assert _is_retryable_qdrant_error(httpx.ConnectError("refused")) is True

    def test_timeout_is_retryable(self) -> None:
        """ReadTimeout should be retried."""
        assert _is_retryable_qdrant_error(httpx.ReadTimeout("timeout")) is True

    def test_500_is_retryable(self) -> None:
        """500 Internal Server Error should be retried."""
        exc = UnexpectedResponse(
            status_code=500,
            reason_phrase="Internal Server Error",
            content=b"",
            headers=httpx.Headers({}),
        )
        assert _is_retryable_qdrant_error(exc) is True

    def test_503_is_retryable(self) -> None:
        """503 Service Unavailable should be retried."""
        exc = UnexpectedResponse(
            status_code=503,
            reason_phrase="Service Unavailable",
            content=b"",
            headers=httpx.Headers({}),
        )
        assert _is_retryable_qdrant_error(exc) is True

    def test_400_is_not_retryable(self) -> None:
        """400 Bad Request should NOT be retried."""
        exc = UnexpectedResponse(
            status_code=400,
            reason_phrase="Bad Request",
            content=b"",
            headers=httpx.Headers({}),
        )
        assert _is_retryable_qdrant_error(exc) is False

    def test_404_is_not_retryable(self) -> None:
        """404 Not Found should NOT be retried."""
        exc = UnexpectedResponse(
            status_code=404,
            reason_phrase="Not Found",
            content=b"",
            headers=httpx.Headers({}),
        )
        assert _is_retryable_qdrant_error(exc) is False

    def test_value_error_is_not_retryable(self) -> None:
        """ValueError should NOT be retried."""
        assert _is_retryable_qdrant_error(ValueError("bad input")) is False
