"""Workflow backend abstraction for swappable execution engines.

This module provides a clean abstraction layer for workflow execution,
supporting multiple backends:

- **InProcess** (default): Direct async function calls, no durability
- **DBOS**: SQLite/PostgreSQL durability, automatic recovery on crash
- **Prefect**: Cloud-native workflow orchestration with flows

Example:
    ```python
    from engram.config import Settings
    from engram.workflows.backend import get_workflow_backend

    settings = Settings(durable_backend="dbos")
    backend = get_workflow_backend(settings)

    # Run consolidation
    result = await backend.run_consolidation(
        storage=storage,
        embedder=embedder,
        user_id="user123",
    )
    ```
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from engram.config import Settings
    from engram.embeddings import Embedder
    from engram.models import Episode
    from engram.storage import EngramStorage

    from .consolidation import ConsolidationResult
    from .decay import DecayResult
    from .promotion import SynthesisResult
    from .structure import StructureResult

logger = logging.getLogger(__name__)


class WorkflowState(str, Enum):
    """State of a workflow execution."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkflowStatus(BaseModel):
    """Status of a workflow execution.

    Attributes:
        workflow_id: Unique identifier for this workflow run.
        workflow_type: Type of workflow (consolidation, decay, structure, promotion).
        state: Current state of the workflow.
        started_at: When the workflow started.
        completed_at: When the workflow completed (if finished).
        error: Error message if failed.
        result: Workflow result (type varies by workflow_type).
    """

    model_config = ConfigDict(extra="forbid")

    workflow_id: str = Field(description="Unique workflow run identifier")
    workflow_type: str = Field(description="Type of workflow")
    state: WorkflowState = Field(default=WorkflowState.PENDING)
    started_at: datetime | None = Field(default=None)
    completed_at: datetime | None = Field(default=None)
    error: str | None = Field(default=None)
    result: dict[str, object] | None = Field(default=None)


@runtime_checkable
class WorkflowBackend(Protocol):
    """Protocol for workflow execution backends.

    All backends must implement these methods for running workflows.
    The protocol allows for both synchronous (in-process) and
    asynchronous (distributed) execution models.
    """

    @abstractmethod
    async def run_consolidation(
        self,
        storage: EngramStorage,
        embedder: Embedder,
        user_id: str,
        org_id: str | None = None,
        consolidation_passes: int = 1,
        similarity_threshold: float = 0.7,
    ) -> ConsolidationResult:
        """Run the consolidation workflow.

        Consolidates unsummarized episodes into semantic memories.

        Args:
            storage: EngramStorage instance.
            embedder: Embedder for vector operations.
            user_id: User ID for multi-tenancy.
            org_id: Optional organization ID.
            consolidation_passes: Number of passes for consolidation.
            similarity_threshold: Threshold for linking similar memories.

        Returns:
            ConsolidationResult with processing statistics.
        """
        ...

    @abstractmethod
    async def run_decay(
        self,
        storage: EngramStorage,
        settings: Settings,
        user_id: str,
        org_id: str | None = None,
        embedder: Embedder | None = None,
        run_promotion: bool = True,
    ) -> DecayResult:
        """Run the decay workflow.

        Applies time-based decay to memory confidence scores.

        Args:
            storage: EngramStorage instance.
            settings: Engram settings with decay configuration.
            user_id: User ID for multi-tenancy.
            org_id: Optional organization ID.
            embedder: Optional embedder for promotion.
            run_promotion: Whether to run promotion after decay.

        Returns:
            DecayResult with processing statistics.
        """
        ...

    @abstractmethod
    async def run_structure(
        self,
        episode: Episode,
        storage: EngramStorage,
        embedder: Embedder,
        model: str | None = None,
        skip_if_structured: bool = True,
    ) -> StructureResult | None:
        """Run the structure workflow for a single episode.

        Extracts structured information from an episode.

        Args:
            episode: The episode to structure.
            storage: EngramStorage instance.
            embedder: Embedder for vector operations.
            model: Optional LLM model override.
            skip_if_structured: Skip if already structured.

        Returns:
            StructureResult or None if skipped.
        """
        ...

    @abstractmethod
    async def run_structure_batch(
        self,
        storage: EngramStorage,
        embedder: Embedder,
        user_id: str,
        org_id: str | None = None,
        limit: int | None = None,
        model: str | None = None,
    ) -> list[StructureResult]:
        """Run the structure workflow for all unstructured episodes.

        Args:
            storage: EngramStorage instance.
            embedder: Embedder for vector operations.
            user_id: User ID to process.
            org_id: Optional org ID filter.
            limit: Maximum episodes to process.
            model: Optional LLM model override.

        Returns:
            List of StructureResult for each processed episode.
        """
        ...

    @abstractmethod
    async def run_promotion(
        self,
        storage: EngramStorage,
        embedder: Embedder,
        user_id: str,
        org_id: str | None = None,
    ) -> SynthesisResult:
        """Run the promotion workflow.

        Synthesizes semantic memories into a procedural memory.

        Args:
            storage: EngramStorage instance.
            embedder: Embedder for vector operations.
            user_id: User ID for multi-tenancy.
            org_id: Optional organization ID.

        Returns:
            SynthesisResult with processing statistics.
        """
        ...

    @abstractmethod
    async def get_workflow_status(self, workflow_id: str) -> WorkflowStatus | None:
        """Get the status of a workflow execution.

        Args:
            workflow_id: The workflow run identifier.

        Returns:
            WorkflowStatus or None if not found.
        """
        ...


class InProcessBackend:
    """In-process workflow execution (no durability).

    This is the default backend that runs workflows as direct async
    function calls. No external infrastructure is needed, but there
    is no durability - if the process crashes, the workflow is lost.

    Suitable for:
    - Development and testing
    - Simple use cases without durability requirements
    - Low-volume production where occasional reruns are acceptable
    """

    def __init__(self, settings: Settings) -> None:
        """Initialize the in-process backend.

        Args:
            settings: Engram configuration settings.
        """
        self.settings = settings
        self._workflow_history: dict[str, WorkflowStatus] = {}

    async def run_consolidation(
        self,
        storage: EngramStorage,
        embedder: Embedder,
        user_id: str,
        org_id: str | None = None,
        consolidation_passes: int = 1,
        similarity_threshold: float = 0.7,
    ) -> ConsolidationResult:
        """Run consolidation as a direct async call."""
        from .consolidation import run_consolidation

        return await run_consolidation(
            storage=storage,
            embedder=embedder,
            user_id=user_id,
            org_id=org_id,
            consolidation_passes=consolidation_passes,
            similarity_threshold=similarity_threshold,
        )

    async def run_decay(
        self,
        storage: EngramStorage,
        settings: Settings,
        user_id: str,
        org_id: str | None = None,
        embedder: Embedder | None = None,
        run_promotion: bool = True,
    ) -> DecayResult:
        """Run decay as a direct async call."""
        from .decay import run_decay

        return await run_decay(
            storage=storage,
            settings=settings,
            user_id=user_id,
            org_id=org_id,
            embedder=embedder,
            run_promotion=run_promotion,
        )

    async def run_structure(
        self,
        episode: Episode,
        storage: EngramStorage,
        embedder: Embedder,
        model: str | None = None,
        skip_if_structured: bool = True,
    ) -> StructureResult | None:
        """Run structure as a direct async call."""
        from .structure import run_structure

        return await run_structure(
            episode=episode,
            storage=storage,
            embedder=embedder,
            model=model,
            skip_if_structured=skip_if_structured,
        )

    async def run_structure_batch(
        self,
        storage: EngramStorage,
        embedder: Embedder,
        user_id: str,
        org_id: str | None = None,
        limit: int | None = None,
        model: str | None = None,
    ) -> list[StructureResult]:
        """Run batch structure as a direct async call."""
        from .structure import run_structure_batch

        return await run_structure_batch(
            storage=storage,
            embedder=embedder,
            user_id=user_id,
            org_id=org_id,
            limit=limit,
            model=model,
        )

    async def run_promotion(
        self,
        storage: EngramStorage,
        embedder: Embedder,
        user_id: str,
        org_id: str | None = None,
    ) -> SynthesisResult:
        """Run promotion as a direct async call."""
        from .promotion import run_synthesis

        return await run_synthesis(
            storage=storage,
            embedder=embedder,
            user_id=user_id,
            org_id=org_id,
        )

    async def get_workflow_status(self, workflow_id: str) -> WorkflowStatus | None:
        """Get workflow status from in-memory history."""
        return self._workflow_history.get(workflow_id)


class DBOSBackend:
    """DBOS-based workflow execution with SQLite durability.

    Uses DBOS for durable execution with automatic retries and
    exactly-once semantics. Requires no external infrastructure
    (uses SQLite by default).

    Suitable for:
    - Production use with durability requirements
    - Single-node deployments
    - Moderate volume workloads
    """

    def __init__(self, settings: Settings) -> None:
        """Initialize the DBOS backend.

        Args:
            settings: Engram configuration settings.
        """
        self.settings = settings
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Ensure DBOS is initialized before running workflows."""
        if self._initialized:
            return

        from dbos import DBOS, DBOSConfig

        db_url = self.settings.database_url or "sqlite:///engram_dbos.sqlite"
        config: DBOSConfig = {
            "name": "engram",
            "system_database_url": db_url,
        }
        DBOS(config=config)
        DBOS.launch()
        self._initialized = True
        logger.info("DBOS backend initialized")

    async def run_consolidation(
        self,
        storage: EngramStorage,
        embedder: Embedder,
        user_id: str,
        org_id: str | None = None,
        consolidation_passes: int = 1,
        similarity_threshold: float = 0.7,
    ) -> ConsolidationResult:
        """Run consolidation with DBOS durability."""
        self._ensure_initialized()

        from dbos import DBOS

        from .consolidation import run_consolidation

        @DBOS.workflow()
        async def durable_consolidation() -> ConsolidationResult:
            return await run_consolidation(
                storage=storage,
                embedder=embedder,
                user_id=user_id,
                org_id=org_id,
                consolidation_passes=consolidation_passes,
                similarity_threshold=similarity_threshold,
            )

        return await durable_consolidation()

    async def run_decay(
        self,
        storage: EngramStorage,
        settings: Settings,
        user_id: str,
        org_id: str | None = None,
        embedder: Embedder | None = None,
        run_promotion: bool = True,
    ) -> DecayResult:
        """Run decay with DBOS durability."""
        self._ensure_initialized()

        from dbos import DBOS

        from .decay import run_decay

        @DBOS.workflow()
        async def durable_decay() -> DecayResult:
            return await run_decay(
                storage=storage,
                settings=settings,
                user_id=user_id,
                org_id=org_id,
                embedder=embedder,
                run_promotion=run_promotion,
            )

        return await durable_decay()

    async def run_structure(
        self,
        episode: Episode,
        storage: EngramStorage,
        embedder: Embedder,
        model: str | None = None,
        skip_if_structured: bool = True,
    ) -> StructureResult | None:
        """Run structure with DBOS durability."""
        self._ensure_initialized()

        from dbos import DBOS

        from .structure import run_structure

        @DBOS.workflow()
        async def durable_structure() -> StructureResult | None:
            return await run_structure(
                episode=episode,
                storage=storage,
                embedder=embedder,
                model=model,
                skip_if_structured=skip_if_structured,
            )

        return await durable_structure()

    async def run_structure_batch(
        self,
        storage: EngramStorage,
        embedder: Embedder,
        user_id: str,
        org_id: str | None = None,
        limit: int | None = None,
        model: str | None = None,
    ) -> list[StructureResult]:
        """Run batch structure with DBOS durability."""
        self._ensure_initialized()

        from dbos import DBOS

        from .structure import run_structure_batch

        @DBOS.workflow()
        async def durable_structure_batch() -> list[StructureResult]:
            return await run_structure_batch(
                storage=storage,
                embedder=embedder,
                user_id=user_id,
                org_id=org_id,
                limit=limit,
                model=model,
            )

        return await durable_structure_batch()

    async def run_promotion(
        self,
        storage: EngramStorage,
        embedder: Embedder,
        user_id: str,
        org_id: str | None = None,
    ) -> SynthesisResult:
        """Run promotion with DBOS durability."""
        self._ensure_initialized()

        from dbos import DBOS

        from .promotion import run_synthesis

        @DBOS.workflow()
        async def durable_promotion() -> SynthesisResult:
            return await run_synthesis(
                storage=storage,
                embedder=embedder,
                user_id=user_id,
                org_id=org_id,
            )

        return await durable_promotion()

    async def get_workflow_status(self, workflow_id: str) -> WorkflowStatus | None:
        """Get workflow status from DBOS."""
        self._ensure_initialized()

        from typing import Any

        from dbos import DBOS

        try:
            status: Any = DBOS.get_workflow_status(workflow_id)
            if status is None:
                return None

            state_map: dict[str, WorkflowState] = {
                "PENDING": WorkflowState.PENDING,
                "RUNNING": WorkflowState.RUNNING,
                "SUCCESS": WorkflowState.COMPLETED,
                "ERROR": WorkflowState.FAILED,
                "CANCELLED": WorkflowState.CANCELLED,
            }

            # DBOS returns a dict-like object with workflow metadata
            return WorkflowStatus(
                workflow_id=workflow_id,
                workflow_type=status.get("name", "unknown"),
                state=state_map.get(status.get("status", "PENDING"), WorkflowState.PENDING),
                started_at=status.get("started_at"),
                completed_at=status.get("completed_at"),
                error=status.get("error"),
            )
        except Exception as e:
            logger.warning(f"Failed to get DBOS workflow status: {e}")
            return None


class PrefectBackend:
    """Prefect-based workflow execution.

    Uses Prefect for cloud-native workflow orchestration.
    Can work with Prefect Cloud or self-hosted Prefect server.

    Suitable for:
    - Cloud-native deployments
    - Teams already using Prefect
    - Complex scheduling requirements
    """

    def __init__(self, settings: Settings) -> None:
        """Initialize the Prefect backend.

        Args:
            settings: Engram configuration settings.
        """
        self.settings = settings
        if settings.prefect_api_url:
            import os

            os.environ["PREFECT_API_URL"] = settings.prefect_api_url

    async def run_consolidation(
        self,
        storage: EngramStorage,
        embedder: Embedder,
        user_id: str,
        org_id: str | None = None,
        consolidation_passes: int = 1,
        similarity_threshold: float = 0.7,
    ) -> ConsolidationResult:
        """Run consolidation as a Prefect flow."""
        from prefect import flow

        from .consolidation import run_consolidation

        @flow(name="engram-consolidation")
        async def consolidation_flow() -> ConsolidationResult:
            return await run_consolidation(
                storage=storage,
                embedder=embedder,
                user_id=user_id,
                org_id=org_id,
                consolidation_passes=consolidation_passes,
                similarity_threshold=similarity_threshold,
            )

        return await consolidation_flow()

    async def run_decay(
        self,
        storage: EngramStorage,
        settings: Settings,
        user_id: str,
        org_id: str | None = None,
        embedder: Embedder | None = None,
        run_promotion: bool = True,
    ) -> DecayResult:
        """Run decay as a Prefect flow."""
        from prefect import flow

        from .decay import run_decay

        @flow(name="engram-decay")
        async def decay_flow() -> DecayResult:
            return await run_decay(
                storage=storage,
                settings=settings,
                user_id=user_id,
                org_id=org_id,
                embedder=embedder,
                run_promotion=run_promotion,
            )

        return await decay_flow()

    async def run_structure(
        self,
        episode: Episode,
        storage: EngramStorage,
        embedder: Embedder,
        model: str | None = None,
        skip_if_structured: bool = True,
    ) -> StructureResult | None:
        """Run structure as a Prefect flow."""
        from prefect import flow

        from .structure import run_structure

        @flow(name="engram-structure")
        async def structure_flow() -> StructureResult | None:
            return await run_structure(
                episode=episode,
                storage=storage,
                embedder=embedder,
                model=model,
                skip_if_structured=skip_if_structured,
            )

        return await structure_flow()

    async def run_structure_batch(
        self,
        storage: EngramStorage,
        embedder: Embedder,
        user_id: str,
        org_id: str | None = None,
        limit: int | None = None,
        model: str | None = None,
    ) -> list[StructureResult]:
        """Run batch structure as a Prefect flow."""
        from prefect import flow

        from .structure import run_structure_batch

        @flow(name="engram-structure-batch")
        async def structure_batch_flow() -> list[StructureResult]:
            return await run_structure_batch(
                storage=storage,
                embedder=embedder,
                user_id=user_id,
                org_id=org_id,
                limit=limit,
                model=model,
            )

        return await structure_batch_flow()

    async def run_promotion(
        self,
        storage: EngramStorage,
        embedder: Embedder,
        user_id: str,
        org_id: str | None = None,
    ) -> SynthesisResult:
        """Run promotion as a Prefect flow."""
        from prefect import flow

        from .promotion import run_synthesis

        @flow(name="engram-promotion")
        async def promotion_flow() -> SynthesisResult:
            return await run_synthesis(
                storage=storage,
                embedder=embedder,
                user_id=user_id,
                org_id=org_id,
            )

        return await promotion_flow()

    async def get_workflow_status(self, workflow_id: str) -> WorkflowStatus | None:
        """Get workflow status from Prefect."""
        try:
            from uuid import UUID

            from prefect.client.orchestration import get_client

            async with get_client() as client:
                flow_run = await client.read_flow_run(UUID(workflow_id))

                state_map = {
                    "PENDING": WorkflowState.PENDING,
                    "RUNNING": WorkflowState.RUNNING,
                    "COMPLETED": WorkflowState.COMPLETED,
                    "FAILED": WorkflowState.FAILED,
                    "CANCELLED": WorkflowState.CANCELLED,
                }

                return WorkflowStatus(
                    workflow_id=str(flow_run.id),
                    workflow_type=flow_run.name or "unknown",
                    state=state_map.get(
                        flow_run.state.type.value if flow_run.state else "PENDING",
                        WorkflowState.PENDING,
                    ),
                    started_at=flow_run.start_time,
                    completed_at=flow_run.end_time,
                )
        except Exception as e:
            logger.warning(f"Failed to get Prefect workflow status: {e}")
            return None


def get_workflow_backend(settings: Settings) -> WorkflowBackend:
    """Get the appropriate workflow backend based on settings.

    Args:
        settings: Engram configuration settings.

    Returns:
        WorkflowBackend instance for the configured backend.

    Raises:
        ValueError: If the configured backend is unknown.

    Example:
        ```python
        from engram.config import Settings
        from engram.workflows.backend import get_workflow_backend

        # Uses ENGRAM_DURABLE_BACKEND environment variable
        settings = Settings()
        backend = get_workflow_backend(settings)

        # Or explicitly configure
        settings = Settings(durable_backend="dbos")
        backend = get_workflow_backend(settings)
        ```
    """
    backend_type = settings.durable_backend.lower()

    if backend_type == "inprocess":
        logger.info("Using in-process workflow backend (no durability)")
        return InProcessBackend(settings)
    elif backend_type == "dbos":
        logger.info("Using DBOS workflow backend")
        return DBOSBackend(settings)
    elif backend_type == "prefect":
        logger.info("Using Prefect workflow backend")
        return PrefectBackend(settings)
    else:
        raise ValueError(
            f"Unknown workflow backend: {backend_type}. " "Use 'inprocess', 'dbos', or 'prefect'."
        )


def get_inprocess_backend(settings: Settings) -> InProcessBackend:
    """Get an in-process backend (no durability).

    This is useful for testing or when durability is not required.

    Args:
        settings: Engram configuration settings.

    Returns:
        InProcessBackend instance.
    """
    return InProcessBackend(settings)


__all__ = [
    "WorkflowBackend",
    "WorkflowState",
    "WorkflowStatus",
    "InProcessBackend",
    "DBOSBackend",
    "PrefectBackend",
    "get_workflow_backend",
    "get_inprocess_backend",
]
