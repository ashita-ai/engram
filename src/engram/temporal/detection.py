"""Temporal state change detection using patterns and LLM.

Detects temporal language patterns like "I no longer use X" and
extracts the implied state changes.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .models import ChangeType, StateChange, TemporalExtractionResult

logger = logging.getLogger(__name__)


# Regex patterns for temporal language
# Format: (pattern, change_type, captures: [entity_group, ...])
# Entity pattern: word optionally followed by .word or -word (e.g., Vue.js, VS-Code)
# Keeps it simple to avoid capturing trailing prepositions like "for", "but", "after"
_ENTITY = r"(\w+(?:[\.\-]\w+)?)"

TEMPORAL_PATTERNS: list[tuple[str, ChangeType, list[int]]] = [
    # Stopped patterns
    (rf"I no longer (?:use|work with|have) {_ENTITY}", "stopped", [1]),
    (rf"I stopped using {_ENTITY}", "stopped", [1]),
    (rf"I quit using {_ENTITY}", "stopped", [1]),
    (rf"I gave up on {_ENTITY}", "stopped", [1]),
    (rf"I don't use {_ENTITY} anymore", "stopped", [1]),
    (rf"I'm no longer (?:using|working with) {_ENTITY}", "stopped", [1]),
    # Started patterns
    (rf"I (?:now|recently) (?:use|started using) {_ENTITY}", "started", [1]),
    (rf"I've started (?:using|working with) {_ENTITY}", "started", [1]),
    (rf"I just (?:started|began) (?:using|with) {_ENTITY}", "started", [1]),
    (rf"I picked up {_ENTITY}", "started", [1]),
    # Changed/switched patterns
    (rf"I switched from {_ENTITY} to {_ENTITY}", "changed", [1, 2]),
    (rf"I moved from {_ENTITY} to {_ENTITY}", "changed", [1, 2]),
    (rf"I migrated from {_ENTITY} to {_ENTITY}", "changed", [1, 2]),
    (rf"I replaced {_ENTITY} with {_ENTITY}", "changed", [1, 2]),
    # Used to patterns (implies stopped)
    (rf"I used to (?:use|work with|have) {_ENTITY}", "stopped", [1]),
    (rf"I previously used {_ENTITY}", "stopped", [1]),
    (rf"I was using {_ENTITY} before", "stopped", [1]),
    # Upgraded/downgraded patterns
    (rf"I upgraded (?:to|from \w+ to) {_ENTITY}", "upgraded", [1]),
    (rf"I downgraded (?:to|from \w+ to) {_ENTITY}", "downgraded", [1]),
    # Resumed patterns
    (rf"I'm back to (?:using|working with) {_ENTITY}", "resumed", [1]),
    (rf"I started using {_ENTITY} again", "resumed", [1]),
]


def detect_state_changes_regex(
    text: str,
    memory_id: str,
    user_id: str,
    org_id: str | None = None,
) -> list[StateChange]:
    """Detect state changes using regex patterns.

    Fast pattern-based detection for common temporal phrases.

    Args:
        text: Text to analyze.
        memory_id: Source memory ID.
        user_id: User ID.
        org_id: Optional organization ID.

    Returns:
        List of detected state changes.
    """
    changes: list[StateChange] = []

    for pattern, change_type, capture_groups in TEMPORAL_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            try:
                if change_type == "changed" and len(capture_groups) >= 2:
                    # Switch pattern: from X to Y
                    from_entity = match.group(capture_groups[0]).strip()
                    to_entity = match.group(capture_groups[1]).strip()

                    # Create two changes: stopped using X, started using Y
                    changes.append(
                        StateChange(
                            entity=from_entity,
                            previous_state=f"user uses {from_entity}",
                            current_state=f"user does not use {from_entity}",
                            change_type="stopped",
                            source_memory_id=memory_id,
                            trigger_text=match.group(0),
                            inferred_previous=False,
                            user_id=user_id,
                            org_id=org_id,
                            confidence=0.9,  # Explicit switch
                        )
                    )
                    changes.append(
                        StateChange(
                            entity=to_entity,
                            previous_state=f"user does not use {to_entity}",
                            current_state=f"user uses {to_entity}",
                            change_type="started",
                            source_memory_id=memory_id,
                            trigger_text=match.group(0),
                            inferred_previous=True,
                            user_id=user_id,
                            org_id=org_id,
                            confidence=0.9,
                        )
                    )
                else:
                    # Single entity pattern
                    entity = match.group(capture_groups[0]).strip()

                    if change_type in ("stopped",):
                        previous = f"user uses {entity}"
                        current = f"user does not use {entity}"
                    elif change_type in ("started", "resumed"):
                        previous = f"user does not use {entity}"
                        current = f"user uses {entity}"
                    elif change_type == "upgraded":
                        previous = "user uses older version"
                        current = f"user uses {entity}"
                    elif change_type == "downgraded":
                        previous = "user uses newer version"
                        current = f"user uses {entity}"
                    else:
                        previous = "unknown"
                        current = f"user uses {entity}"

                    changes.append(
                        StateChange(
                            entity=entity,
                            previous_state=previous,
                            current_state=current,
                            change_type=change_type,
                            source_memory_id=memory_id,
                            trigger_text=match.group(0),
                            inferred_previous=True,
                            user_id=user_id,
                            org_id=org_id,
                            confidence=0.85,
                        )
                    )
            except Exception as e:
                logger.warning("Failed to extract state change: %s", e)

    return changes


class LLMStateChanges(BaseModel):
    """LLM output for state change detection."""

    model_config = ConfigDict(extra="forbid")

    changes: list[dict[str, Any]] = Field(default_factory=list)
    reasoning: str = Field(default="")


DETECTION_PROMPT = """You are analyzing text for temporal state changes.

A state change occurs when something that was true becomes false, or vice versa.

Common patterns:
- "I no longer use X" → stopped using X (implies previously used X)
- "I switched from X to Y" → stopped X, started Y
- "I used to use X" → stopped using X
- "I now use X" → started using X
- "I upgraded to X" → upgraded to X

For each state change, extract:
1. entity: The subject of the change (tool, technology, practice, etc.)
2. previous_state: What was true before (may be inferred)
3. current_state: What is true now
4. change_type: One of "started", "stopped", "changed", "upgraded", "downgraded", "resumed"
5. confidence: How confident you are (0.0-1.0)
6. inferred_previous: Whether the previous state was inferred (not explicitly stated)

IMPORTANT:
- Be conservative - only extract clear state changes
- Preferences are NOT state changes unless explicitly temporal
- "I like X" is NOT a state change
- "I now prefer X" IS a state change
"""


async def detect_state_changes_llm(
    text: str,
    memory_id: str,
    user_id: str,
    org_id: str | None = None,
) -> TemporalExtractionResult:
    """Detect state changes using LLM analysis.

    More thorough than regex, catches complex temporal language.

    Args:
        text: Text to analyze.
        memory_id: Source memory ID.
        user_id: User ID.
        org_id: Optional organization ID.

    Returns:
        TemporalExtractionResult with detected changes.
    """
    from pydantic_ai import Agent

    from engram.config import settings
    from engram.workflows.llm_utils import run_agent_with_retry

    agent: Agent[None, LLMStateChanges] = Agent(
        settings.llm_model,
        output_type=LLMStateChanges,
        instructions=DETECTION_PROMPT,
    )

    try:
        llm_output = await run_agent_with_retry(
            agent, f"Analyze for temporal state changes:\n\n{text}"
        )

        changes: list[StateChange] = []
        for change_data in llm_output.changes:
            try:
                change_type = change_data.get("change_type", "changed")
                if change_type not in (
                    "started",
                    "stopped",
                    "changed",
                    "upgraded",
                    "downgraded",
                    "resumed",
                ):
                    change_type = "changed"

                changes.append(
                    StateChange(
                        entity=change_data.get("entity", "unknown"),
                        previous_state=change_data.get("previous_state", "unknown"),
                        current_state=change_data.get("current_state", "unknown"),
                        change_type=change_type,
                        source_memory_id=memory_id,
                        trigger_text=text[:100],
                        inferred_previous=change_data.get("inferred_previous", True),
                        user_id=user_id,
                        org_id=org_id,
                        confidence=change_data.get("confidence", 0.7),
                    )
                )
            except Exception as e:
                logger.warning("Failed to parse state change: %s", e)

        return TemporalExtractionResult(
            state_changes=changes,
            reasoning=llm_output.reasoning,
        )

    except Exception as e:
        logger.exception("LLM state change detection failed: %s", e)
        return TemporalExtractionResult(reasoning=f"Detection failed: {e}")


async def detect_state_changes(
    text: str,
    memory_id: str,
    user_id: str,
    org_id: str | None = None,
    use_llm: bool = False,
) -> list[StateChange]:
    """Detect state changes from text.

    Combines regex patterns with optional LLM analysis.

    Args:
        text: Text to analyze.
        memory_id: Source memory ID.
        user_id: User ID.
        org_id: Optional organization ID.
        use_llm: Whether to use LLM for additional detection.

    Returns:
        List of detected state changes.
    """
    # Always run regex patterns (fast)
    changes = detect_state_changes_regex(text, memory_id, user_id, org_id)

    # Optionally use LLM for more thorough detection
    if use_llm:
        llm_result = await detect_state_changes_llm(text, memory_id, user_id, org_id)

        # Merge LLM results, avoiding duplicates
        existing_entities = {(c.entity.lower(), c.change_type) for c in changes}
        for llm_change in llm_result.state_changes:
            key = (llm_change.entity.lower(), llm_change.change_type)
            if key not in existing_entities:
                changes.append(llm_change)
                existing_entities.add(key)

    logger.info(
        "Detected %d state changes in memory %s (use_llm=%s)",
        len(changes),
        memory_id,
        use_llm,
    )

    return changes
