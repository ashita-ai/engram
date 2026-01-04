"""SemanticMemory model - LLM-inferred knowledge."""

from datetime import UTC, datetime

from pydantic import Field

from .base import ConfidenceScore, MemoryBase, generate_id


class SemanticMemory(MemoryBase):
    """Semantic memory inferred by LLM from conversations.

    Semantic memories are extracted during background consolidation.
    They represent inferred knowledge, preferences, and context that
    isn't explicitly stated but can be derived from conversations.

    Selectivity scoring tracks how well-established a memory is:
    - 0.0: Newly created, broad associations
    - 1.0: Highly selective, well-consolidated

    Attributes:
        content: The inferred semantic content.
        source_episode_ids: Episodes this was derived from.
        related_ids: Links to related memories (for multi-hop reasoning).
        event_at: When the underlying facts were true.
        derived_at: When we inferred this memory.
        confidence: Composite confidence score.
        selectivity_score: How well-consolidated (0=broad, 1=selective).
        consolidation_passes: How many times this has been refined.
    """

    id: str = Field(default_factory=lambda: generate_id("sem"))
    content: str = Field(description="The inferred semantic content")
    source_episode_ids: list[str] = Field(
        default_factory=list,
        description="IDs of Episodes this was derived from",
    )
    related_ids: list[str] = Field(
        default_factory=list,
        description="IDs of related memories for multi-hop reasoning",
    )
    event_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When the underlying facts were true",
    )
    derived_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When we inferred this memory",
    )
    confidence: ConfidenceScore = Field(
        default_factory=lambda: ConfidenceScore.for_inferred(0.6),
        description="Composite confidence score",
    )
    selectivity_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Consolidation selectivity (0=broad, 1=selective)",
    )
    consolidation_passes: int = Field(
        default=1,
        ge=1,
        description="Number of consolidation passes",
    )

    def add_link(self, memory_id: str) -> None:
        """Add a link to a related memory."""
        if memory_id not in self.related_ids:
            self.related_ids.append(memory_id)

    def increase_selectivity(self, delta: float = 0.1) -> None:
        """Increase selectivity score (survived consolidation)."""
        self.selectivity_score = min(1.0, self.selectivity_score + delta)
        self.consolidation_passes += 1

    def decrease_selectivity(self, delta: float = 0.1) -> None:
        """Decrease selectivity score (pruned during consolidation)."""
        self.selectivity_score = max(0.0, self.selectivity_score - delta)

    def __str__(self) -> str:
        """String representation showing content preview."""
        content_preview = (
            self.content[:50] + "..." if len(self.content) > 50 else self.content
        )
        return f"SemanticMemory({content_preview!r})"
