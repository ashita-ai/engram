"""Decay workflow for updating memory confidence over time.

This workflow runs periodically to:
1. Find memories that haven't been accessed recently
2. Apply exponential decay to their confidence scores
3. Archive or delete memories below threshold

The workflow is durable - it survives crashes and can be retried on failure.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from engram.config import Settings
    from engram.storage import EngramStorage

logger = logging.getLogger(__name__)


class DecayResult(BaseModel):
    """Result of a decay workflow run.

    Attributes:
        memories_updated: Number of memories with updated confidence.
        memories_archived: Number of memories moved to archive.
        memories_deleted: Number of memories permanently deleted.
    """

    model_config = ConfigDict(extra="forbid")

    memories_updated: int = Field(ge=0)
    memories_archived: int = Field(ge=0)
    memories_deleted: int = Field(ge=0)


async def run_decay(
    storage: EngramStorage,
    settings: Settings,
    user_id: str,
    org_id: str | None = None,
) -> DecayResult:
    """Run the decay workflow.

    This workflow:
    1. Fetches all semantic memories for the user
    2. Recomputes confidence using time-based decay
    3. Archives memories below archive threshold
    4. Deletes memories below delete threshold

    Args:
        storage: EngramStorage instance.
        settings: Engram settings with decay configuration.
        user_id: User ID for multi-tenancy.
        org_id: Optional organization ID.

    Returns:
        DecayResult with processing statistics.
    """
    # 1. Fetch all semantic memories (including archived for potential deletion)
    memories = await storage.list_semantic_memories(
        user_id=user_id,
        org_id=org_id,
        include_archived=True,
    )

    if not memories:
        logger.info("No semantic memories found for decay processing")
        return DecayResult(
            memories_updated=0,
            memories_archived=0,
            memories_deleted=0,
        )

    logger.info(f"Processing {len(memories)} memories for decay")

    updated = 0
    archived = 0
    deleted = 0

    for memory in memories:
        # 2. Recompute confidence with decay
        old_confidence = memory.confidence.value
        memory.confidence.recompute_with_weights(settings.confidence_weights)
        new_confidence = memory.confidence.value

        # 3. Check thresholds
        if new_confidence < settings.decay_delete_threshold:
            # Delete memories below delete threshold
            await storage.delete_semantic(memory.id, user_id)
            deleted += 1
            logger.debug(
                f"Deleted memory {memory.id}: confidence {old_confidence:.3f} -> {new_confidence:.3f}"
            )
        elif new_confidence < settings.decay_archive_threshold:
            # Archive memories below archive threshold
            if not memory.archived:
                memory.archived = True
                await storage.update_semantic_memory(memory)
                archived += 1
                logger.debug(
                    f"Archived memory {memory.id}: confidence {old_confidence:.3f} -> {new_confidence:.3f}"
                )
            else:
                # Already archived, just update confidence
                await storage.update_semantic_memory(memory)
                updated += 1
        else:
            # Update confidence if it changed significantly
            if abs(old_confidence - new_confidence) > 0.001:
                # Unarchive if confidence recovered
                if memory.archived:
                    memory.archived = False
                    logger.debug(
                        f"Unarchived memory {memory.id}: confidence recovered to {new_confidence:.3f}"
                    )
                await storage.update_semantic_memory(memory)
                updated += 1

    logger.info(f"Decay complete: {updated} updated, {archived} archived, {deleted} deleted")

    return DecayResult(
        memories_updated=updated,
        memories_archived=archived,
        memories_deleted=deleted,
    )


__all__ = [
    "DecayResult",
    "run_decay",
]
