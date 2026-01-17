"""Decay workflow for updating memory confidence over time.

This workflow runs periodically to:
1. Find memories that haven't been accessed recently
2. Apply exponential decay to their confidence scores
3. Archive or delete memories below threshold
4. Run promotion to elevate behavioral patterns to procedural memory

The workflow is durable - it survives crashes and can be retried on failure.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from engram.config import Settings
    from engram.embeddings import Embedder
    from engram.storage import EngramStorage

logger = logging.getLogger(__name__)

# Default thresholds for access-based archival
LOW_ACCESS_THRESHOLD = 2  # Archive if accessed fewer than this many times
LOW_ACCESS_AGE_DAYS = 90  # Only consider memories older than this


class DecayResult(BaseModel):
    """Result of a decay workflow run.

    Attributes:
        memories_updated: Number of memories with updated confidence.
        memories_archived: Number of memories moved to archive.
        memories_deleted: Number of memories permanently deleted.
        low_access_archived: Number of memories archived due to low access.
        procedural_promoted: Number of memories promoted to procedural.
    """

    model_config = ConfigDict(extra="forbid")

    memories_updated: int = Field(ge=0)
    memories_archived: int = Field(ge=0)
    memories_deleted: int = Field(ge=0)
    low_access_archived: int = Field(ge=0, default=0)
    procedural_promoted: int = Field(ge=0, default=0)


async def run_decay(
    storage: EngramStorage,
    settings: Settings,
    user_id: str,
    org_id: str | None = None,
    embedder: Embedder | None = None,
    run_promotion: bool = True,
) -> DecayResult:
    """Run the decay workflow.

    This workflow:
    1. Fetches all semantic memories for the user
    2. Recomputes confidence using time-based decay
    3. Archives memories below archive threshold (confidence-based)
    4. Archives memories with low access count (access-based)
    5. Deletes memories below delete threshold
    6. Runs promotion to elevate behavioral patterns to procedural memory

    Args:
        storage: EngramStorage instance.
        settings: Engram settings with decay configuration.
        user_id: User ID for multi-tenancy.
        org_id: Optional organization ID.
        embedder: Optional embedder for promotion workflow. Required if run_promotion=True.
        run_promotion: Whether to run promotion after decay. Defaults to True.

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
            low_access_archived=0,
            procedural_promoted=0,
        )

    logger.info(f"Processing {len(memories)} memories for decay")

    updated = 0
    archived = 0
    deleted = 0
    low_access_archived = 0

    # Calculate cutoff date for low-access archival
    low_access_cutoff = datetime.now(UTC) - timedelta(days=LOW_ACCESS_AGE_DAYS)

    for memory in memories:
        # 2. Check for low-access archival (A-MEM style)
        # Archive old memories that are rarely accessed
        if (
            not memory.archived
            and memory.derived_at < low_access_cutoff
            and memory.retrieval_count < LOW_ACCESS_THRESHOLD
        ):
            memory.archived = True
            await storage.update_semantic_memory(memory)
            low_access_archived += 1
            logger.debug(
                f"Low-access archived {memory.id}: {memory.retrieval_count} accesses, "
                f"age {(datetime.now(UTC) - memory.derived_at).days} days"
            )
            continue  # Skip further processing for this memory

        # 3. Recompute confidence with decay
        old_confidence = memory.confidence.value
        memory.confidence.recompute_with_weights(settings.confidence_weights)
        new_confidence = memory.confidence.value

        # 4. Check thresholds
        if new_confidence < settings.decay_delete_threshold:
            # Delete memories below delete threshold
            await storage.delete_semantic(memory.id, user_id)
            deleted += 1
            logger.debug(
                f"Deleted memory {memory.id}: "
                f"confidence {old_confidence:.3f} -> {new_confidence:.3f}"
            )
        elif new_confidence < settings.decay_archive_threshold:
            # Archive memories below archive threshold
            if not memory.archived:
                memory.archived = True
                await storage.update_semantic_memory(memory)
                archived += 1
                logger.debug(
                    f"Archived memory {memory.id}: "
                    f"confidence {old_confidence:.3f} -> {new_confidence:.3f}"
                )
            else:
                # Already archived, just update confidence
                await storage.update_semantic_memory(memory)
                updated += 1
        elif abs(old_confidence - new_confidence) > 0.001:
            # Update confidence if it changed significantly
            # Unarchive if confidence recovered
            if memory.archived:
                memory.archived = False
                logger.debug(
                    f"Unarchived memory {memory.id}: confidence recovered to {new_confidence:.3f}"
                )
            await storage.update_semantic_memory(memory)
            updated += 1

    logger.info(
        f"Decay complete: {updated} updated, {archived} archived, "
        f"{low_access_archived} low-access archived, {deleted} deleted"
    )

    # 5. Run synthesis workflow if requested
    procedural_promoted = 0
    if run_promotion and embedder is not None:
        from engram.workflows.promotion import run_synthesis

        synthesis_result = await run_synthesis(
            storage=storage,
            embedder=embedder,
            user_id=user_id,
            org_id=org_id,
        )
        procedural_promoted = (
            1 if synthesis_result.procedural_created or synthesis_result.procedural_updated else 0
        )
        logger.info(
            f"Synthesis complete: {procedural_promoted} procedural memories created/updated"
        )
    elif run_promotion and embedder is None:
        logger.warning("Skipping synthesis: embedder not provided")

    return DecayResult(
        memories_updated=updated,
        memories_archived=archived,
        memories_deleted=deleted,
        low_access_archived=low_access_archived,
        procedural_promoted=procedural_promoted,
    )


__all__ = [
    "DecayResult",
    "run_decay",
]
