"""Tests for episode immutability enforcement (SPEC-002).

Verifies that update_episode() silently strips all immutable fields
(content, role, timestamp, session_id) and logs a warning when
the incoming value differs from the stored value.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from engram.models import Episode


class TestEpisodeImmutability:
    """Tests for immutable field stripping in update_episode."""

    def _make_mixin(
        self,
        stored_payload: dict[str, object],
    ) -> tuple[object, AsyncMock]:
        """Create a CRUDMixin with mocked client returning the given payload."""
        from engram.storage.crud import CRUDMixin

        mixin = CRUDMixin()
        mixin._collection_name = MagicMock(return_value="engram_episodic")
        mixin._memory_to_payload = MagicMock(side_effect=lambda ep: ep.model_dump(mode="json"))

        mock_point = MagicMock()
        mock_point.id = "point_abc"
        mock_point.payload = stored_payload

        mock_client = AsyncMock()
        mock_client.scroll = AsyncMock(return_value=([mock_point], None))
        mock_client.set_payload = AsyncMock()
        mixin.client = mock_client

        return mixin, mock_client

    @pytest.mark.asyncio
    async def test_update_episode_strips_timestamp(self) -> None:
        """Timestamp must not be overwritten via update_episode."""
        original_ts = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
        episode = Episode(
            content="hello",
            role="user",
            user_id="test_user",
            timestamp=datetime(2026, 6, 15, 0, 0, 0, tzinfo=UTC),
            session_id="sess_orig",
        )

        stored_payload = episode.model_dump(mode="json")
        stored_payload["timestamp"] = original_ts.isoformat()

        mixin, mock_client = self._make_mixin(stored_payload)

        await mixin.update_episode(episode)

        # Verify set_payload was called and the payload does NOT contain timestamp
        call_args = mock_client.set_payload.call_args
        sent_payload = call_args.kwargs.get("payload") or call_args[1].get("payload")
        assert "timestamp" not in sent_payload

    @pytest.mark.asyncio
    async def test_update_episode_strips_session_id(self) -> None:
        """session_id must not be overwritten via update_episode."""
        episode = Episode(
            content="hello",
            role="user",
            user_id="test_user",
            session_id="sess_new",
        )

        stored_payload = episode.model_dump(mode="json")
        stored_payload["session_id"] = "sess_original"

        mixin, mock_client = self._make_mixin(stored_payload)

        await mixin.update_episode(episode)

        call_args = mock_client.set_payload.call_args
        sent_payload = call_args.kwargs.get("payload") or call_args[1].get("payload")
        assert "session_id" not in sent_payload

    @pytest.mark.asyncio
    async def test_update_episode_strips_content(self) -> None:
        """content must not be overwritten (pre-existing behavior)."""
        episode = Episode(
            content="modified content",
            role="user",
            user_id="test_user",
        )

        stored_payload = episode.model_dump(mode="json")
        stored_payload["content"] = "original content"

        mixin, mock_client = self._make_mixin(stored_payload)

        await mixin.update_episode(episode)

        call_args = mock_client.set_payload.call_args
        sent_payload = call_args.kwargs.get("payload") or call_args[1].get("payload")
        assert "content" not in sent_payload

    @pytest.mark.asyncio
    async def test_update_episode_strips_role(self) -> None:
        """role must not be overwritten (pre-existing behavior)."""
        episode = Episode(
            content="hello",
            role="assistant",
            user_id="test_user",
        )

        stored_payload = episode.model_dump(mode="json")
        stored_payload["role"] = "user"

        mixin, mock_client = self._make_mixin(stored_payload)

        await mixin.update_episode(episode)

        call_args = mock_client.set_payload.call_args
        sent_payload = call_args.kwargs.get("payload") or call_args[1].get("payload")
        assert "role" not in sent_payload

    @pytest.mark.asyncio
    async def test_update_episode_allows_mutable_fields(self) -> None:
        """Mutable fields like consolidated and importance must still update."""
        episode = Episode(
            content="hello",
            role="user",
            user_id="test_user",
            consolidated=True,
            importance=0.9,
        )

        stored_payload = episode.model_dump(mode="json")
        stored_payload["consolidated"] = False
        stored_payload["importance"] = 0.5

        mixin, mock_client = self._make_mixin(stored_payload)

        result = await mixin.update_episode(episode)
        assert result is True

        call_args = mock_client.set_payload.call_args
        sent_payload = call_args.kwargs.get("payload") or call_args[1].get("payload")
        assert sent_payload["consolidated"] is True
        assert sent_payload["importance"] == 0.9

    @pytest.mark.asyncio
    async def test_update_episode_logs_warning_on_timestamp_mutation(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A warning should be logged when timestamp mutation is attempted."""
        original_ts = "2025-01-01T12:00:00+00:00"
        episode = Episode(
            content="hello",
            role="user",
            user_id="test_user",
            timestamp=datetime(2026, 6, 15, 0, 0, 0, tzinfo=UTC),
        )

        stored_payload = episode.model_dump(mode="json")
        stored_payload["timestamp"] = original_ts

        mixin, _ = self._make_mixin(stored_payload)

        with caplog.at_level(logging.WARNING, logger="engram.storage.crud"):
            await mixin.update_episode(episode)

        assert any("immutable episode field 'timestamp'" in rec.message for rec in caplog.records)

    @pytest.mark.asyncio
    async def test_update_episode_logs_warning_on_session_id_mutation(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A warning should be logged when session_id mutation is attempted."""
        episode = Episode(
            content="hello",
            role="user",
            user_id="test_user",
            session_id="sess_new",
        )

        stored_payload = episode.model_dump(mode="json")
        stored_payload["session_id"] = "sess_original"

        mixin, _ = self._make_mixin(stored_payload)

        with caplog.at_level(logging.WARNING, logger="engram.storage.crud"):
            await mixin.update_episode(episode)

        assert any("immutable episode field 'session_id'" in rec.message for rec in caplog.records)

    @pytest.mark.asyncio
    async def test_update_episode_no_warning_when_values_match(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """No warning when the incoming immutable value matches the stored value."""
        episode = Episode(
            content="hello",
            role="user",
            user_id="test_user",
            session_id="sess_same",
        )

        stored_payload = episode.model_dump(mode="json")
        # stored matches incoming — no mutation
        assert stored_payload["session_id"] == "sess_same"

        mixin, _ = self._make_mixin(stored_payload)

        with caplog.at_level(logging.WARNING, logger="engram.storage.crud"):
            await mixin.update_episode(episode)

        immutability_warnings = [
            rec for rec in caplog.records if "immutable episode field" in rec.message
        ]
        assert len(immutability_warnings) == 0

    @pytest.mark.asyncio
    async def test_update_episode_returns_false_when_not_found(self) -> None:
        """update_episode returns False if the episode doesn't exist."""
        from engram.storage.crud import CRUDMixin

        mixin = CRUDMixin()
        mixin._collection_name = MagicMock(return_value="engram_episodic")

        mock_client = AsyncMock()
        mock_client.scroll = AsyncMock(return_value=([], None))
        mixin.client = mock_client

        episode = Episode(
            content="hello",
            role="user",
            user_id="test_user",
        )

        result = await mixin.update_episode(episode)
        assert result is False
