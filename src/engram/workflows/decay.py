"""Decay workflow for updating memory confidence over time.

This workflow runs periodically to:
1. Find memories that haven't been accessed recently
2. Apply exponential decay to their confidence scores
3. Archive or delete memories below threshold

The workflow is durable - it survives crashes and can be retried on failure.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


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


# Workflow implementation will be added in issue #21
# The workflow will use:
# - @DBOS.workflow() decorator for durability
# - Exponential decay formula from config
# - EngramStorage for updating memories
