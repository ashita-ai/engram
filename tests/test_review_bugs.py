"""Regression tests for bugs found during code review.

Each test directly covers a specific bug to prevent regressions.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from qdrant_client.http.exceptions import UnexpectedResponse

from engram.config import Settings
from engram.models import SemanticMemory
from engram.models.base import ConfidenceScore, ExtractionMethod
from engram.storage.retry import _is_retryable_qdrant_error
from engram.storage.transaction import TransactionContext
from engram.workflows.decay import run_decay


class TestCollectionPrefixBug:
    """Bug #1: _get_collection_prefix() read from global settings instead of self._prefix.

    When a custom prefix was passed to EngramStorage(), get_memory_stats()
    would query collections using the global default prefix, returning
    wrong data silently.
    """

    def test_get_collection_prefix_uses_instance_prefix(self) -> None:
        """_get_collection_prefix should return self._prefix, not global settings."""
        from engram.storage.base import StorageBase

        storage = StorageBase(prefix="custom_prefix")
        assert storage._prefix == "custom_prefix"

    def test_custom_prefix_propagates_to_collection_names(self) -> None:
        """Collection names should use the instance prefix."""
        from engram.storage.base import StorageBase

        storage = StorageBase(prefix="my_app")
        assert storage._collection_name("episodic") == "my_app_episodic"
        assert storage._collection_name("semantic") == "my_app_semantic"

    def test_get_collection_prefix_matches_instance(self) -> None:
        """_get_collection_prefix should match the instance _prefix."""
        from engram.storage import EngramStorage

        storage = EngramStorage(prefix="test_prefix")
        assert storage._get_collection_prefix() == "test_prefix"


class TestRetryPredicateBug:
    """Bug #2: qdrant_retry caught all UnexpectedResponse including 4xx.

    Client errors (400, 404, 409) are not transient and should not be
    retried. Only 5xx server errors and network errors should trigger retries.
    """

    def test_connect_error_is_retryable(self) -> None:
        """Network connection errors should be retried."""
        exc = httpx.ConnectError("Connection refused")
        assert _is_retryable_qdrant_error(exc) is True

    def test_timeout_is_retryable(self) -> None:
        """Timeout errors should be retried."""
        exc = httpx.ReadTimeout("Read timed out")
        assert _is_retryable_qdrant_error(exc) is True

    def test_5xx_server_error_is_retryable(self) -> None:
        """5xx server errors should be retried."""
        exc = UnexpectedResponse(
            status_code=500,
            reason_phrase="Internal Server Error",
            content=b"server error",
            headers=httpx.Headers(),
        )
        assert _is_retryable_qdrant_error(exc) is True

    def test_502_bad_gateway_is_retryable(self) -> None:
        """502 Bad Gateway should be retried."""
        exc = UnexpectedResponse(
            status_code=502,
            reason_phrase="Bad Gateway",
            content=b"bad gateway",
            headers=httpx.Headers(),
        )
        assert _is_retryable_qdrant_error(exc) is True

    def test_503_service_unavailable_is_retryable(self) -> None:
        """503 Service Unavailable should be retried."""
        exc = UnexpectedResponse(
            status_code=503,
            reason_phrase="Service Unavailable",
            content=b"unavailable",
            headers=httpx.Headers(),
        )
        assert _is_retryable_qdrant_error(exc) is True

    def test_400_bad_request_not_retryable(self) -> None:
        """400 Bad Request should NOT be retried."""
        exc = UnexpectedResponse(
            status_code=400,
            reason_phrase="Bad Request",
            content=b"bad request",
            headers=httpx.Headers(),
        )
        assert _is_retryable_qdrant_error(exc) is False

    def test_404_not_found_not_retryable(self) -> None:
        """404 Not Found should NOT be retried."""
        exc = UnexpectedResponse(
            status_code=404,
            reason_phrase="Not Found",
            content=b"not found",
            headers=httpx.Headers(),
        )
        assert _is_retryable_qdrant_error(exc) is False

    def test_409_conflict_not_retryable(self) -> None:
        """409 Conflict should NOT be retried."""
        exc = UnexpectedResponse(
            status_code=409,
            reason_phrase="Conflict",
            content=b"conflict",
            headers=httpx.Headers(),
        )
        assert _is_retryable_qdrant_error(exc) is False

    def test_422_unprocessable_not_retryable(self) -> None:
        """422 Unprocessable Entity should NOT be retried."""
        exc = UnexpectedResponse(
            status_code=422,
            reason_phrase="Unprocessable Entity",
            content=b"validation error",
            headers=httpx.Headers(),
        )
        assert _is_retryable_qdrant_error(exc) is False

    def test_none_status_code_not_retryable(self) -> None:
        """UnexpectedResponse with None status_code should NOT be retried."""
        exc = UnexpectedResponse(
            status_code=None,
            reason_phrase="Unknown",
            content=b"unknown",
            headers=httpx.Headers(),
        )
        assert _is_retryable_qdrant_error(exc) is False

    def test_unrelated_exception_not_retryable(self) -> None:
        """Non-Qdrant exceptions should NOT be retried."""
        assert _is_retryable_qdrant_error(ValueError("test")) is False
        assert _is_retryable_qdrant_error(RuntimeError("test")) is False


class TestTransactionRollbackBug:
    """Bug #3: Transaction __aexit__ only rolled back on exception.

    Clean exit without commit() left operations persisted but tracked,
    silently leaking uncommitted data.
    """

    @pytest.mark.asyncio
    async def test_clean_exit_without_commit_rolls_back(self) -> None:
        """Exiting transaction without commit() should rollback."""
        storage = MagicMock()
        storage.store_episode = AsyncMock(return_value="ep_123")
        storage.delete_episode = AsyncMock(return_value={"deleted": True})

        episode = MagicMock()
        episode.id = "ep_test"
        episode.user_id = "user_1"

        async with TransactionContext(storage=storage) as txn:
            await txn.store_episode(episode)
            # Deliberately NOT calling txn.commit()

        # Should have rolled back the stored episode
        storage.delete_episode.assert_called_once()

    @pytest.mark.asyncio
    async def test_committed_transaction_not_rolled_back(self) -> None:
        """Committed transaction should NOT be rolled back on clean exit."""
        storage = MagicMock()
        storage.store_episode = AsyncMock(return_value="ep_123")
        storage.delete_episode = AsyncMock(return_value={"deleted": True})

        episode = MagicMock()
        episode.id = "ep_test"
        episode.user_id = "user_1"

        async with TransactionContext(storage=storage) as txn:
            await txn.store_episode(episode)
            txn.commit()

        # Should NOT have rolled back
        storage.delete_episode.assert_not_called()

    @pytest.mark.asyncio
    async def test_exception_still_rolls_back(self) -> None:
        """Exception during transaction should still trigger rollback."""
        storage = MagicMock()
        storage.store_episode = AsyncMock(return_value="ep_123")
        storage.delete_episode = AsyncMock(return_value={"deleted": True})

        episode = MagicMock()
        episode.id = "ep_test"
        episode.user_id = "user_1"

        with pytest.raises(ValueError, match="test error"):
            async with TransactionContext(storage=storage) as txn:
                await txn.store_episode(episode)
                raise ValueError("test error")

        storage.delete_episode.assert_called_once()

    @pytest.mark.asyncio
    async def test_empty_uncommitted_transaction_no_rollback(self) -> None:
        """Empty transaction without commit should not attempt rollback."""
        storage = MagicMock()

        async with TransactionContext(storage=storage):
            pass  # No operations, no commit

        # No rollback should occur since no operations were tracked
        storage.delete_episode.assert_not_called()


class TestDecayUnarchivalBug:
    """Bug #6: Unarchival not persisted when confidence delta was small.

    A memory archived by low-access with above-threshold confidence
    but tiny confidence delta stayed incorrectly archived because the
    unarchival logic was nested inside the "confidence changed
    significantly" branch.
    """

    @pytest.fixture
    def settings(self) -> Settings:
        """Settings with known thresholds."""
        return Settings(
            openai_api_key="sk-test",
            decay_archive_threshold=0.4,
            decay_delete_threshold=0.2,
        )

    def _create_archived_above_threshold_memory(self) -> SemanticMemory:
        """Create a memory that is archived but has above-threshold confidence.

        This simulates a memory that was archived by low-access archival
        but whose confidence is still above the archive threshold.
        """
        memory = SemanticMemory(
            content="Important fact",
            user_id="test_user",
            embedding=[0.1] * 384,
            archived=True,  # Archived by low-access
        )
        # Set confidence factors that produce ~0.575 after recompute (above 0.4 threshold)
        # The delta after recompute will be small since value is already close
        memory.confidence = ConfidenceScore(
            value=0.575,  # Pre-set close to what recompute will produce
            extraction_method=ExtractionMethod.INFERRED,
            extraction_base=0.6,
            supporting_episodes=1,
            verified=False,
            last_confirmed=datetime.now(UTC),
            contradictions=0,
        )
        # Ensure it's not eligible for low-access archival
        memory.derived_at = datetime.now(UTC)
        memory.retrieval_count = 10
        return memory

    @pytest.mark.asyncio
    async def test_archived_memory_with_good_confidence_gets_unarchived(
        self, settings: Settings
    ) -> None:
        """Memory archived by low-access should be unarchived when confidence is above threshold."""
        memory = self._create_archived_above_threshold_memory()

        mock_storage = AsyncMock()
        mock_storage.list_semantic_memories = AsyncMock(return_value=[memory])
        mock_storage.update_semantic_memory = AsyncMock()

        result = await run_decay(
            storage=mock_storage,
            settings=settings,
            user_id="test_user",
            org_id="test_org",
            run_promotion=False,
        )

        # Memory should be unarchived
        mock_storage.update_semantic_memory.assert_called_once()
        updated_memory = mock_storage.update_semantic_memory.call_args[0][0]
        assert updated_memory.archived is False
        assert result.memories_updated == 1
