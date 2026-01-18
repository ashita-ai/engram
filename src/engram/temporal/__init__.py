"""Temporal reasoning for state change tracking.

Detects and tracks temporal state changes in memories, preserving
the implication that past states were different from current states.

Example:
    ```python
    from engram.temporal import detect_state_changes, StateChange

    # Detect state changes
    changes = await detect_state_changes(
        text="I no longer use MongoDB, I switched to PostgreSQL",
        memory_id="mem_123",
        user_id="user_456",
    )

    for change in changes:
        print(f"{change.entity}: {change.previous_state} → {change.current_state}")
    ```
"""

from .detection import (
    TEMPORAL_PATTERNS,
    detect_state_changes,
    detect_state_changes_llm,
    detect_state_changes_regex,
)
from .models import (
    ChangeType,
    StateChange,
    TemporalExtractionResult,
    TemporalQuery,
    TemporalState,
    Timeline,
)

__all__ = [
    # Models
    "ChangeType",
    "StateChange",
    "TemporalExtractionResult",
    "TemporalQuery",
    "TemporalState",
    "Timeline",
    # Functions
    "detect_state_changes",
    "detect_state_changes_llm",
    "detect_state_changes_regex",
    # Constants
    "TEMPORAL_PATTERNS",
]
