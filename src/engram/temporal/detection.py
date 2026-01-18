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
TEMPORAL_PATTERNS: list[tuple[str, ChangeType, list[int]]] = [
    # Stopped patterns
    (r"I no longer (?:use|work with|have) (\w+(?:\s+\w+)?)", "stopped", [1]),
    (r"I stopped using (\w+(?:\s+\w+)?)", "stopped", [1]),
    (r"I quit using (\w+(?:\s+\w+)?)", "stopped", [1]),
    (r"I gave up on (\w+(?:\s+\w+)?)", "stopped", [1]),
    (r"I don't use (\w+(?:\s+\w+)?) anymore", "stopped", [1]),
    (r"I'm no longer (?:using|working with) (\w+(?:\s+\w+)?)", "stopped", [1]),
    # Started patterns
    (r"I (?:now|recently) (?:use|started using) (\w+(?:\s+\w+)?)", "started", [1]),
    (r"I've started (?:using|working with) (\w+(?:\s+\w+)?)", "started", [1]),
    (r"I just (?:started|began) (?:using|with) (\w+(?:\s+\w+)?)", "started", [1]),
    (r"I picked up (\w+(?:\s+\w+)?)", "started", [1]),
    # Changed/switched patterns
    (r"I switched from (\w+(?:\s+\w+)?) to (\w+(?:\s+\w+)?)", "changed", [1, 2]),
    (r"I moved from (\w+(?:\s+\w+)?) to (\w+(?:\s+\w+)?)", "changed", [1, 2]),
    (r"I migrated from (\w+(?:\s+\w+)?) to (\w+(?:\s+\w+)?)", "changed", [1, 2]),
    (r"I replaced (\w+(?:\s+\w+)?) with (\w+(?:\s+\w+)?)", "changed", [1, 2]),
    # Used to patterns (implies stopped)
    (r"I used to (?:use|work with|have) (\w+(?:\s+\w+)?)", "stopped", [1]),
    (r"I previously used (\w+(?:\s+\w+)?)", "stopped", [1]),
    (r"I was using (\w+(?:\s+\w+)?) before", "stopped", [1]),
    # Upgraded/downgraded patterns
    (r"I upgraded (?:to|from \w+ to) (\w+(?:\s+\w+)?)", "upgraded", [1]),
    (r"I downgraded (?:to|from \w+ to) (\w+(?:\s+\w+)?)", "downgraded", [1]),
    # Resumed patterns
    (r"I'm back to (?:using|working with) (\w+(?:\s+\w+)?)", "resumed", [1]),
    (r"I started using (\w+(?:\s+\w+)?) again", "resumed", [1]),
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

    agent: Agent[None, LLMStateChanges] = Agent(
        settings.llm_model,
        output_type=LLMStateChanges,
        instructions=DETECTION_PROMPT,
    )

    try:
        result = await agent.run(f"Analyze for temporal state changes:\n\n{text}")
        llm_output = result.output

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
