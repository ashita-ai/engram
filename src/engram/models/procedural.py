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
        access_count: How often this has been used (reinforcement).
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
    access_count: int = Field(
        default=0,
        ge=0,
        description="Usage count for reinforcement learning",
    )
    last_accessed: datetime | None = Field(
        default=None,
        description="When this memory was last retrieved",
    )

    def reinforce(self) -> None:
        """Increment access count (pattern was used successfully)."""
        self.access_count += 1
        self.last_accessed = datetime.now(UTC)

    def record_access(self) -> None:
        """Record that this memory was accessed (activation tracking).

        Alias for reinforce() for API consistency with SemanticMemory.
        """
        self.reinforce()

    def add_link(self, memory_id: str) -> None:
        """Add a link to a related memory."""
        if memory_id not in self.related_ids:
            self.related_ids.append(memory_id)

    def __str__(self) -> str:
        """String representation showing pattern content."""
        content_preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"ProceduralMemory({content_preview!r})"
