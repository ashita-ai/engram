"""Temporal reasoning models for state change tracking.

Handles temporal statements like "I no longer use MongoDB" by
preserving the implication that the past state was different.

Example:
    "I no longer use MongoDB" implies:
    - Previous state: user used MongoDB
    - Current state: user does not use MongoDB
    - Change type: stopped
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from engram.models.base import generate_id

# Types of state changes
ChangeType = Literal[
    "started",  # User began doing something
    "stopped",  # User stopped doing something
    "changed",  # User switched from X to Y
    "upgraded",  # User moved to newer version
    "downgraded",  # User moved to older version
    "resumed",  # User started again after stopping
]


class StateChange(BaseModel):
    """A detected state change over time.

    Represents a transition from one state to another, capturing
    both the previous and current states for timeline reconstruction.

    Attributes:
        id: Unique identifier.
        entity: The subject of the change (e.g., "MongoDB", "React").
        previous_state: What was true before (may be inferred).
        current_state: What is true now.
        change_type: Type of transition.
        detected_at: When this change was detected.
        source_memory_id: Memory that triggered this detection.
        confidence: Confidence in this state change.
        trigger_text: Original text that indicated the change.
        inferred_previous: Whether previous state was inferred (not explicit).
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: generate_id("stc"))
    entity: str = Field(description="Subject of the change")
    previous_state: str = Field(description="Previous state (may be inferred)")
    current_state: str = Field(description="Current state")
    change_type: ChangeType = Field(description="Type of change")
    detected_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
    )
    source_memory_id: str = Field(description="Source memory ID")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        default=0.8,
        description="Confidence in this state change",
    )
    trigger_text: str = Field(description="Text that triggered detection")
    inferred_previous: bool = Field(
        default=True,
        description="Whether previous state was inferred",
    )
    user_id: str = Field(description="User ID")
    org_id: str | None = Field(default=None, description="Optional organization")


class TemporalQuery(BaseModel):
    """A query about state at a point in time.

    Used to ask questions like "What tools did the user use before 2024?"

    Attributes:
        entity: Entity to query about (optional, None = all).
        as_of: Point in time to query.
        include_changes: Include state changes in response.
    """

    model_config = ConfigDict(extra="forbid")

    entity: str | None = Field(default=None, description="Entity to query")
    as_of: datetime | None = Field(default=None, description="Point in time")
    include_changes: bool = Field(
        default=True,
        description="Include change history",
    )


class TemporalState(BaseModel):
    """State of an entity at a point in time.

    Result of a temporal query showing what was true at a given time.

    Attributes:
        entity: The entity this state is about.
        state: The state value.
        valid_from: When this state became valid.
        valid_until: When this state ended (None if still current).
        change_that_started: State change that started this period.
        change_that_ended: State change that ended this period.
    """

    model_config = ConfigDict(extra="forbid")

    entity: str = Field(description="Entity this state is about")
    state: str = Field(description="The state value")
    valid_from: datetime | None = Field(
        default=None,
        description="When state became valid",
    )
    valid_until: datetime | None = Field(
        default=None,
        description="When state ended (None if current)",
    )
    change_that_started: str | None = Field(
        default=None,
        description="ID of change that started this state",
    )
    change_that_ended: str | None = Field(
        default=None,
        description="ID of change that ended this state",
    )

    @property
    def is_current(self) -> bool:
        """Check if this state is still valid."""
        return self.valid_until is None


class Timeline(BaseModel):
    """Timeline of states for an entity.

    Attributes:
        entity: The entity this timeline is for.
        states: Ordered list of states over time.
        changes: State changes that define transitions.
    """

    model_config = ConfigDict(extra="forbid")

    entity: str = Field(description="Entity this timeline is for")
    states: list[TemporalState] = Field(default_factory=list)
    changes: list[StateChange] = Field(default_factory=list)

    def current_state(self) -> TemporalState | None:
        """Get the current state (valid_until is None)."""
        for state in self.states:
            if state.is_current:
                return state
        return None

    def state_at(self, when: datetime) -> TemporalState | None:
        """Get state at a specific point in time."""
        for state in self.states:
            start = state.valid_from or datetime.min.replace(tzinfo=UTC)
            end = state.valid_until or datetime.max.replace(tzinfo=UTC)
            if start <= when <= end:
                return state
        return None


class TemporalExtractionResult(BaseModel):
    """Result of temporal state extraction.

    Attributes:
        state_changes: Detected state changes.
        reasoning: LLM reasoning for extractions.
    """

    model_config = ConfigDict(extra="forbid")

    state_changes: list[StateChange] = Field(default_factory=list)
    reasoning: str = Field(default="")
