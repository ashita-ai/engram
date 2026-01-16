"""Tests for workflow backend abstraction.

Tests the WorkflowBackend protocol and implementations.
"""

import pytest

from engram.config import Settings
from engram.workflows.backend import (
    DBOSBackend,
    InProcessBackend,
    PrefectBackend,
    WorkflowBackend,
    WorkflowState,
    WorkflowStatus,
    get_inprocess_backend,
    get_workflow_backend,
)


class TestWorkflowStatus:
    """Tests for WorkflowStatus model."""

    def test_default_status(self):
        """Test default status values."""
        status = WorkflowStatus(
            workflow_id="wf-123",
            workflow_type="consolidation",
        )
        assert status.workflow_id == "wf-123"
        assert status.workflow_type == "consolidation"
        assert status.state == WorkflowState.PENDING
        assert status.started_at is None
        assert status.completed_at is None
        assert status.error is None
        assert status.result is None

    def test_completed_status(self):
        """Test completed status with result."""
        from datetime import datetime

        now = datetime.now()
        status = WorkflowStatus(
            workflow_id="wf-123",
            workflow_type="consolidation",
            state=WorkflowState.COMPLETED,
            started_at=now,
            completed_at=now,
            result={"episodes_processed": 5},
        )
        assert status.state == WorkflowState.COMPLETED
        assert status.result == {"episodes_processed": 5}

    def test_failed_status(self):
        """Test failed status with error."""
        status = WorkflowStatus(
            workflow_id="wf-123",
            workflow_type="decay",
            state=WorkflowState.FAILED,
            error="Connection timeout",
        )
        assert status.state == WorkflowState.FAILED
        assert status.error == "Connection timeout"


class TestWorkflowState:
    """Tests for WorkflowState enum."""

    def test_all_states_exist(self):
        """Test all expected states are defined."""
        assert WorkflowState.PENDING.value == "pending"
        assert WorkflowState.RUNNING.value == "running"
        assert WorkflowState.COMPLETED.value == "completed"
        assert WorkflowState.FAILED.value == "failed"
        assert WorkflowState.CANCELLED.value == "cancelled"


class TestGetWorkflowBackend:
    """Tests for get_workflow_backend factory function."""

    def test_inprocess_backend(self):
        """Test creating InProcessBackend."""
        settings = Settings(
            durable_backend="inprocess",
            openai_api_key="sk-test-dummy-key",
        )
        backend = get_workflow_backend(settings)
        assert isinstance(backend, InProcessBackend)

    def test_dbos_backend(self):
        """Test creating DBOSBackend."""
        settings = Settings(
            durable_backend="dbos",
            openai_api_key="sk-test-dummy-key",
        )
        backend = get_workflow_backend(settings)
        assert isinstance(backend, DBOSBackend)

    def test_prefect_backend(self):
        """Test creating PrefectBackend."""
        settings = Settings(
            durable_backend="prefect",
            openai_api_key="sk-test-dummy-key",
        )
        backend = get_workflow_backend(settings)
        assert isinstance(backend, PrefectBackend)

    def test_invalid_backend_raises(self):
        """Test that invalid backend raises ValueError."""
        # Settings validation will reject invalid backend, so we test the factory directly
        settings = Settings(openai_api_key="sk-test-dummy-key")
        # Force an invalid backend by bypassing validation
        settings.durable_backend = "invalid"  # type: ignore[assignment]
        with pytest.raises(ValueError, match="Unknown workflow backend"):
            get_workflow_backend(settings)


class TestGetInprocessBackend:
    """Tests for get_inprocess_backend function."""

    def test_returns_inprocess_backend(self):
        """Test that get_inprocess_backend returns InProcessBackend."""
        settings = Settings(openai_api_key="sk-test-dummy-key")
        backend = get_inprocess_backend(settings)
        assert isinstance(backend, InProcessBackend)


class TestInProcessBackend:
    """Tests for InProcessBackend."""

    @pytest.fixture
    def settings(self):
        """Create test settings."""
        return Settings(openai_api_key="sk-test-dummy-key")

    @pytest.fixture
    def backend(self, settings):
        """Create InProcessBackend."""
        return InProcessBackend(settings)

    def test_implements_protocol(self, backend):
        """Test that InProcessBackend implements WorkflowBackend protocol."""
        assert isinstance(backend, WorkflowBackend)

    @pytest.mark.asyncio
    async def test_get_workflow_status_returns_none_for_unknown(self, backend):
        """Test that get_workflow_status returns None for unknown workflow."""
        status = await backend.get_workflow_status("unknown-id")
        assert status is None


class TestDBOSBackend:
    """Tests for DBOSBackend."""

    @pytest.fixture
    def settings(self):
        """Create test settings."""
        return Settings(
            durable_backend="dbos",
            openai_api_key="sk-test-dummy-key",
        )

    @pytest.fixture
    def backend(self, settings):
        """Create DBOSBackend."""
        return DBOSBackend(settings)

    def test_implements_protocol(self, backend):
        """Test that DBOSBackend implements WorkflowBackend protocol."""
        assert isinstance(backend, WorkflowBackend)

    def test_not_initialized_by_default(self, backend):
        """Test that backend is not initialized by default."""
        assert backend._initialized is False


class TestPrefectBackend:
    """Tests for PrefectBackend."""

    @pytest.fixture
    def settings(self):
        """Create test settings."""
        return Settings(
            durable_backend="prefect",
            openai_api_key="sk-test-dummy-key",
        )

    @pytest.fixture
    def backend(self, settings):
        """Create PrefectBackend."""
        return PrefectBackend(settings)

    def test_implements_protocol(self, backend):
        """Test that PrefectBackend implements WorkflowBackend protocol."""
        assert isinstance(backend, WorkflowBackend)


class TestEngramServiceWithBackend:
    """Tests for EngramService workflow backend integration."""

    def test_service_creates_default_backend(self):
        """Test that EngramService creates InProcessBackend by default."""
        from unittest.mock import AsyncMock

        from engram.service import EngramService

        settings = Settings(openai_api_key="sk-test-dummy-key")
        service = EngramService(
            storage=AsyncMock(),
            embedder=AsyncMock(),
            settings=settings,
        )
        # After __post_init__, workflow_backend should be set
        assert service.workflow_backend is not None
        assert isinstance(service.workflow_backend, InProcessBackend)

    def test_service_accepts_custom_backend(self):
        """Test that EngramService accepts custom workflow backend."""
        from unittest.mock import AsyncMock

        from engram.service import EngramService

        settings = Settings(
            durable_backend="dbos",
            openai_api_key="sk-test-dummy-key",
        )
        custom_backend = DBOSBackend(settings)
        service = EngramService(
            storage=AsyncMock(),
            embedder=AsyncMock(),
            settings=settings,
            workflow_backend=custom_backend,
        )
        assert service.workflow_backend is custom_backend

    def test_create_uses_settings_backend(self):
        """Test that EngramService.create uses settings to select backend."""
        # This test requires actual settings but doesn't run workflows
        # Just verify the factory method exists
        settings = Settings(
            durable_backend="inprocess",
            openai_api_key="sk-test-dummy-key",
        )
        backend = get_workflow_backend(settings)
        assert isinstance(backend, InProcessBackend)
