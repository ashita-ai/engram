"""Episode model - immutable ground truth storage."""

from datetime import UTC, datetime

from pydantic import Field

from .base import MemoryBase, generate_id


class Episode(MemoryBase):
    """Immutable episodic memory - raw interaction storage.

    Episodes are the ground truth. They store verbatim user/assistant
    interactions and are never modified after creation. All derived
    memories (Facts, Semantic, Procedural) trace back to Episodes.

    Attributes:
        content: The verbatim text content of the interaction.
        role: Who produced this content (user, assistant, system).
        timestamp: When this interaction occurred.
        session_id: Optional session grouping for conversations.
        importance: How important is this episode (0.0-1.0).
    """

    id: str = Field(default_factory=lambda: generate_id("ep"))
    content: str = Field(description="Verbatim content of the interaction")
    role: str = Field(description="Message role: user, assistant, or system")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When this interaction occurred",
    )
    session_id: str | None = Field(
        default=None,
        description="Session ID for conversation grouping",
    )
    importance: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Importance score for prioritization",
    )
    consolidated: bool = Field(
        default=False,
        description="Whether facts/negations have been extracted from this episode",
    )
    summarized: bool = Field(
        default=False,
        description="Whether this episode has been included in a semantic summary",
    )
    summarized_into: str | None = Field(
        default=None,
        description="ID of the semantic memory that summarizes this episode",
    )

    def __str__(self) -> str:
        """String representation showing role and truncated content."""
        content_preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"Episode({self.role}: {content_preview!r})"
