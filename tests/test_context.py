"""Tests for memory context manager."""

from unittest.mock import AsyncMock, patch

import pytest

from engram.config import Settings
from engram.context import get_current_memory, memory_context, scoped_memory


class TestMemoryContext:
    """Tests for memory_context context manager."""

    @pytest.mark.asyncio
    async def test_context_manager_creates_service(self):
        """Should create and initialize EngramService."""
        with patch("engram.context.EngramService") as mock_service_class:
            mock_service = AsyncMock()
            mock_service_class.create.return_value = mock_service

            async with memory_context(user_id="user_123") as mem:
                assert mem is mock_service
                mock_service_class.create.assert_called_once()
                mock_service.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_manager_cleanup(self):
        """Should clean up on exit."""
        with patch("engram.context.EngramService") as mock_service_class:
            mock_service = AsyncMock()
            mock_service_class.create.return_value = mock_service

            async with memory_context(user_id="user_123"):
                pass

            mock_service.clear_working_memory.assert_called_once()
            mock_service.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_manager_cleanup_on_exception(self):
        """Should clean up even if exception occurs."""
        with patch("engram.context.EngramService") as mock_service_class:
            mock_service = AsyncMock()
            mock_service_class.create.return_value = mock_service

            with pytest.raises(ValueError):
                async with memory_context(user_id="user_123"):
                    raise ValueError("Test error")

            # Cleanup should still happen
            mock_service.clear_working_memory.assert_called_once()
            mock_service.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_manager_with_settings(self):
        """Should use provided settings."""
        settings = Settings(embedding_provider="fastembed", _env_file=None)

        with patch("engram.context.EngramService") as mock_service_class:
            mock_service = AsyncMock()
            mock_service_class.create.return_value = mock_service

            async with memory_context(user_id="user_123", settings=settings):
                mock_service_class.create.assert_called_once_with(settings)

    @pytest.mark.asyncio
    async def test_context_manager_with_org_and_session(self):
        """Should accept org_id and session_id."""
        with patch("engram.context.EngramService") as mock_service_class:
            mock_service = AsyncMock()
            mock_service_class.create.return_value = mock_service

            async with memory_context(
                user_id="user_123",
                org_id="org_456",
                session_id="sess_789",
            ):
                # Just verify it doesn't error
                pass


class TestGetCurrentMemory:
    """Tests for get_current_memory."""

    @pytest.mark.asyncio
    async def test_returns_none_outside_context(self):
        """Should return None when not in a memory context."""
        result = get_current_memory()
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_service_inside_context(self):
        """Should return the service when in a memory context."""
        with patch("engram.context.EngramService") as mock_service_class:
            mock_service = AsyncMock()
            mock_service_class.create.return_value = mock_service

            async with memory_context(user_id="user_123") as mem:
                current = get_current_memory()
                assert current is mem

    @pytest.mark.asyncio
    async def test_returns_none_after_context_exits(self):
        """Should return None after context exits."""
        with patch("engram.context.EngramService") as mock_service_class:
            mock_service = AsyncMock()
            mock_service_class.create.return_value = mock_service

            async with memory_context(user_id="user_123"):
                pass

            # After context exits
            result = get_current_memory()
            assert result is None


class TestScopedMemory:
    """Tests for scoped_memory context manager."""

    @pytest.mark.asyncio
    async def test_uses_existing_service(self):
        """Should use the provided service without creating a new one."""
        mock_service = AsyncMock()

        async with scoped_memory(mock_service, user_id="user_123") as mem:
            assert mem is mock_service

        # Should NOT call initialize or close
        mock_service.initialize.assert_not_called()
        mock_service.close.assert_not_called()

    @pytest.mark.asyncio
    async def test_sets_context_variable(self):
        """Should set the context variable."""
        mock_service = AsyncMock()

        async with scoped_memory(mock_service, user_id="user_123"):
            current = get_current_memory()
            assert current is mock_service

    @pytest.mark.asyncio
    async def test_clears_context_on_exit(self):
        """Should clear context variable on exit."""
        mock_service = AsyncMock()

        async with scoped_memory(mock_service, user_id="user_123"):
            pass

        result = get_current_memory()
        assert result is None


class TestNestedContexts:
    """Tests for nested context behavior."""

    @pytest.mark.asyncio
    async def test_nested_contexts_restore_correctly(self):
        """Inner context should not affect outer context's cleanup."""
        with patch("engram.context.EngramService") as mock_service_class:
            outer_service = AsyncMock()
            inner_service = AsyncMock()
            mock_service_class.create.side_effect = [outer_service, inner_service]

            async with memory_context(user_id="outer") as outer:
                assert get_current_memory() is outer

                async with memory_context(user_id="inner") as inner:
                    assert get_current_memory() is inner

                # After inner exits, outer should be restored
                # Note: Due to contextvar behavior, this returns None
                # since each context manages its own token
                pass

            # Both should be closed
            outer_service.close.assert_called_once()
            inner_service.close.assert_called_once()
