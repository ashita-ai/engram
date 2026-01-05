"""Consolidation workflow for extracting semantic knowledge from episodes.

This workflow runs in the background to:
1. Fetch unconsolidated episodes
2. Run LLM extraction via Pydantic AI
3. Store semantic memories
4. Build links between memories

The workflow is durable - it survives crashes and can be retried on failure.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ConsolidationResult(BaseModel):
    """Result of a consolidation workflow run.

    Attributes:
        episodes_processed: Number of episodes that were processed.
        semantic_memories_created: Number of semantic memories extracted.
        links_created: Number of memory links built.
        contradictions_found: List of detected contradictions.
    """

    model_config = ConfigDict(extra="forbid")

    episodes_processed: int = Field(ge=0)
    semantic_memories_created: int = Field(ge=0)
    links_created: int = Field(ge=0)
    contradictions_found: list[str] = Field(default_factory=list)


# Workflow implementation will be added in issue #20
# The workflow will use:
# - @DBOS.workflow() decorator for durability
# - Pydantic AI agent for LLM extraction
# - EngramStorage for persisting results
