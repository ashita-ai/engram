"""ProceduralMemory model - behavioral patterns and preferences."""

from datetime import UTC, datetime

from pydantic import Field

from .base import ConfidenceScore, MemoryBase, generate_id


class ProceduralMemory(MemoryBase):
    """Procedural memory for behavioral patterns and preferences.

    Procedural memories capture how a user prefers to interact,
    what patterns they follow, and what behaviors to replicate.
    They're promoted from repeated patterns in episodic/semantic memory.

    Examples:
    - "User prefers concise responses"
    - "User asks for code examples in Python"
    - "User likes explanations before code"

    Attributes:
        content: Description of the behavioral pattern.
        trigger_context: When this pattern applies.
        source_episode_ids: Episodes where this pattern was observed.
        related_ids: Links to related memories.
        confidence: Composite confidence score.
        consolidation_strength: How well-established (0=new, 1=strong).
        consolidation_passes: How many times this has been refined.
        retrieval_count: How often this has been retrieved.
        last_accessed: When this memory was last retrieved.
    """

    id: str = Field(default_factory=lambda: generate_id("proc"))
    content: str = Field(description="Description of the behavioral pattern")
    trigger_context: str = Field(
        default="",
        description="Context description for when this applies",
    )
    source_episode_ids: list[str] = Field(
        default_factory=list,
        description="Episodes where this pattern was observed",
    )
    source_semantic_ids: list[str] = Field(
        default_factory=list,
        description="Semantic memories this was synthesized from",
    )
    related_ids: list[str] = Field(
        default_factory=list,
        description="Links to related memories",
    )
    derived_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When this pattern was identified",
    )
    confidence: ConfidenceScore = Field(
        default_factory=lambda: ConfidenceScore.for_inferred(0.6),
        description="Composite confidence score",
    )
    consolidation_strength: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="How well-established via repeated consolidation (0=new, 1=strong)",
    )
    consolidation_passes: int = Field(
        default=0,
        ge=0,
        description="Number of consolidation passes",
    )
    retrieval_count: int = Field(
        default=0,
        ge=0,
        description="Number of times this memory has been retrieved",
    )
    last_accessed: datetime | None = Field(
        default=None,
        description="When this memory was last retrieved",
    )

    def strengthen(self, delta: float = 0.1) -> None:
        """Strengthen memory through consolidation involvement.

        Based on Testing Effect research: memories repeatedly involved
        in retrieval/consolidation become stronger and more stable.
        See: Roediger & Karpicke (2006), PMC5912918.
        """
        self.consolidation_strength = min(1.0, self.consolidation_strength + delta)
        self.consolidation_passes += 1

    def weaken(self, delta: float = 0.1) -> None:
        """Weaken memory (pruned or contradicted during consolidation)."""
        self.consolidation_strength = max(0.0, self.consolidation_strength - delta)

    def record_access(self) -> None:
        """Record that this memory was accessed (activation tracking).

        Increments retrieval_count and updates last_accessed timestamp.
        Called by storage layer on search hits.
        """
        self.retrieval_count += 1
        self.last_accessed = datetime.now(UTC)

    def reinforce(self) -> None:
        """Alias for record_access() for backwards compatibility."""
        self.record_access()

    def add_link(self, memory_id: str) -> None:
        """Add a link to a related memory."""
        if memory_id not in self.related_ids:
            self.related_ids.append(memory_id)

    def __str__(self) -> str:
        """String representation showing pattern content."""
        content_preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"ProceduralMemory({content_preview!r})"
