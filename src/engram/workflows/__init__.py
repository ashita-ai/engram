"""Durable workflow execution for Engram.

This module provides durable workflows for background processing tasks
like consolidation and decay. Supports multiple backends:

- **DBOS** (default): SQLite/PostgreSQL-based durability
- **Prefect**: Cloud-native workflow orchestration

Example:
    ```python
    from engram.workflows import DurableAgentFactory
    from engram.config import Settings

    settings = Settings()
    factory = DurableAgentFactory(settings)

    # Get durable agents
    consolidation = factory.get_consolidation_agent()
    decay = factory.get_decay_agent()

    # Use them
    result = await consolidation.run(episodes_text)
    ```

Configuration:
    Set ENGRAM_DURABLE_BACKEND environment variable:
    - "dbos" (default): Local SQLite or PostgreSQL durability
    - "prefect": Requires Prefect server or Prefect Cloud
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from pydantic_ai import Agent

from engram.config import Settings

from .backend import (
    DBOSBackend,
    InProcessBackend,
    PrefectBackend,
    WorkflowBackend,
    WorkflowState,
    WorkflowStatus,
    get_inprocess_backend,
    get_workflow_backend,
)
from .consolidation import (
    ConsolidationResult,
    ExtractedFact,
    IdentifiedLink,
    LLMExtractionResult,
    run_consolidation,
    run_consolidation_from_structured,
)
from .decay import DecayResult, run_decay
from .promotion import SynthesisResult, run_synthesis
from .structure import (
    LLMExtractionOutput,
    StructureResult,
    run_structure,
    run_structure_batch,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Type alias for supported backends
DurableBackend = Literal["inprocess", "dbos", "prefect"]


@dataclass
class DurableAgentFactory:
    """Factory for creating durable workflow agents.

    This factory creates Pydantic AI agents wrapped with the appropriate
    durable execution wrapper based on the configured backend.

    Attributes:
        settings: Engram configuration settings.
        _consolidation_agent: Cached consolidation agent.
        _decay_agent: Cached decay agent.
        _initialized: Whether the backend has been initialized.
    """

    settings: Settings
    _consolidation_agent: Any = field(default=None, init=False, repr=False)
    _decay_agent: Any = field(default=None, init=False, repr=False)
    _initialized: bool = field(default=False, init=False, repr=False)

    @property
    def backend(self) -> DurableBackend:
        """Get the configured durable execution backend."""
        backend = self.settings.durable_backend.lower()
        if backend not in ("dbos", "prefect", "inprocess"):
            raise ValueError(
                f"Unknown durable backend: {backend}. Use 'dbos', 'prefect', or 'inprocess'."
            )
        return backend  # type: ignore[return-value]

    def _create_base_consolidation_agent(self) -> Agent[None, LLMExtractionResult]:
        """Create the base consolidation agent (without durable wrapper)."""
        return Agent(
            self.settings.consolidation_model,
            output_type=LLMExtractionResult,
            name="engram_consolidation",
            instructions="""You are analyzing conversation episodes to extract lasting knowledge.

Extract facts that are:
- Personally identifying (names, emails, preferences)
- Temporally stable (unlikely to change soon)
- Explicitly stated (not inferred)

Identify relationships between concepts.
Flag any contradictions with previously known facts.

Be conservative. When uncertain, don't extract.""",
        )

    def _create_base_decay_agent(self) -> Agent[None, DecayResult]:
        """Create the base decay agent (without durable wrapper)."""
        return Agent(
            self.settings.consolidation_model,  # Use same model for decay
            output_type=DecayResult,
            name="engram_decay",
            instructions="""Evaluate memory relevance and confidence decay.

For each memory, determine:
- Whether it should be retained (still relevant)
- Whether it should be archived (low confidence but historically important)
- Whether it should be deleted (outdated or superseded)

Apply conservative decay - when uncertain, retain.""",
        )

    def _wrap_agent_dbos(self, agent: Agent[None, Any]) -> Any:
        """Wrap agent with DBOS durable execution."""
        from pydantic_ai.durable_exec.dbos import DBOSAgent

        return DBOSAgent(agent)

    def _wrap_agent_prefect(self, agent: Agent[None, Any]) -> Any:
        """Wrap agent with Prefect durable execution."""
        from pydantic_ai.durable_exec.prefect import PrefectAgent

        return PrefectAgent(agent)

    def _wrap_agent(self, agent: Agent[None, Any]) -> Any:
        """Wrap agent with the configured durable execution backend."""
        if self.backend == "dbos":
            return self._wrap_agent_dbos(agent)
        elif self.backend == "prefect":
            return self._wrap_agent_prefect(agent)
        elif self.backend == "inprocess":
            return agent  # No wrapping for in-process
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def initialize(self) -> None:
        """Initialize the durable execution backend.

        For DBOS, this configures and launches DBOS.
        For Prefect, this is a no-op (external setup required).

        Must be called AFTER creating agents but BEFORE using them (for DBOS).
        """
        if self._initialized:
            return

        if self.backend == "dbos":
            from dbos import DBOS, DBOSConfig

            # Use configured database_url or default to SQLite
            db_url = self.settings.database_url or "sqlite:///engram_dbos.sqlite"

            dbos_config: DBOSConfig = {
                "name": "engram",
                "system_database_url": db_url,
            }
            DBOS(config=dbos_config)

            # Create and wrap agents BEFORE launch (required by DBOS)
            self._consolidation_agent = self._wrap_agent(self._create_base_consolidation_agent())
            self._decay_agent = self._wrap_agent(self._create_base_decay_agent())

            DBOS.launch()
            logger.info("DBOS backend initialized")

        elif self.backend == "prefect":
            # Prefect agents can be created anytime
            self._consolidation_agent = self._wrap_agent(self._create_base_consolidation_agent())
            self._decay_agent = self._wrap_agent(self._create_base_decay_agent())
            logger.info("Prefect backend initialized (server must be configured separately)")

        elif self.backend == "inprocess":
            # In-process agents don't need special initialization
            self._consolidation_agent = self._wrap_agent(self._create_base_consolidation_agent())
            self._decay_agent = self._wrap_agent(self._create_base_decay_agent())
            logger.info("In-process backend initialized (no durability)")

        self._initialized = True

    def get_consolidation_agent(self) -> Any:
        """Get the durable consolidation agent.

        Returns:
            DBOSAgent, TemporalAgent, or PrefectAgent wrapping the consolidation agent.

        Raises:
            RuntimeError: If factory not initialized.
        """
        if not self._initialized:
            raise RuntimeError("Factory not initialized. Call initialize() first.")
        return self._consolidation_agent

    def get_decay_agent(self) -> Any:
        """Get the durable decay agent.

        Returns:
            DBOSAgent, TemporalAgent, or PrefectAgent wrapping the decay agent.

        Raises:
            RuntimeError: If factory not initialized.
        """
        if not self._initialized:
            raise RuntimeError("Factory not initialized. Call initialize() first.")
        return self._decay_agent


# Global factory instance for simple usage
_factory: DurableAgentFactory | None = None


def init_workflows(settings: Settings | None = None) -> DurableAgentFactory:
    """Initialize durable workflows with the given settings.

    Args:
        settings: Engram settings. Uses defaults if None.

    Returns:
        The initialized DurableAgentFactory.
    """
    global _factory

    if _factory is not None:
        return _factory

    if settings is None:
        settings = Settings()

    _factory = DurableAgentFactory(settings)
    _factory.initialize()
    return _factory


def shutdown_workflows() -> None:
    """Shutdown the global workflow factory."""
    global _factory
    _factory = None


def get_consolidation_agent() -> Any:
    """Get the global consolidation agent.

    Raises:
        RuntimeError: If init_workflows() not called.
    """
    if _factory is None:
        raise RuntimeError("Workflows not initialized. Call init_workflows() first.")
    return _factory.get_consolidation_agent()


def get_decay_agent() -> Any:
    """Get the global decay agent.

    Raises:
        RuntimeError: If init_workflows() not called.
    """
    if _factory is None:
        raise RuntimeError("Workflows not initialized. Call init_workflows() first.")
    return _factory.get_decay_agent()


__all__ = [
    # Workflow backend abstraction
    "WorkflowBackend",
    "WorkflowState",
    "WorkflowStatus",
    "InProcessBackend",
    "DBOSBackend",
    "PrefectBackend",
    "get_workflow_backend",
    "get_inprocess_backend",
    # Agent factory (for Pydantic AI agent wrapping)
    "DurableAgentFactory",
    "DurableBackend",
    "init_workflows",
    "shutdown_workflows",
    "get_consolidation_agent",
    "get_decay_agent",
    # Workflow results
    "ConsolidationResult",
    "DecayResult",
    "SynthesisResult",
    "ExtractedFact",
    "IdentifiedLink",
    "LLMExtractionResult",
    "LLMExtractionOutput",
    "StructureResult",
    # Workflow functions (for direct use)
    "run_consolidation",
    "run_consolidation_from_structured",
    "run_decay",
    "run_synthesis",
    "run_structure",
    "run_structure_batch",
]
