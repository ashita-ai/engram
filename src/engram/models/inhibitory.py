"""InhibitoryFact model - tracking what is NOT true."""

from datetime import UTC, datetime

from pydantic import Field

from .base import ConfidenceScore, MemoryBase, generate_id


class InhibitoryFact(MemoryBase):
    """Inhibitory memory - tracking what is explicitly NOT true.

    Inhibitory facts prevent false matches by recording negations.
    When a user corrects a misunderstanding or explicitly negates
    something, we store that negation to filter future retrievals.

    Inspired by the role of CCK+ interneurons in memory selectivity
    (Tomé et al., Nature Neuroscience 2024).

    Examples:
    - "User does NOT use MongoDB"
    - "User's email is NOT jane@example.com"
    - "User does NOT prefer verbose responses"

    Attributes:
        content: The negation statement.
        negates_pattern: Pattern/keyword this inhibits in retrieval.
        source_episode_ids: Episodes where negation was stated.
        derived_at: When we identified this negation.
        confidence: Composite confidence score.
    """

    id: str = Field(default_factory=lambda: generate_id("inh"))
    content: str = Field(description="The negation statement")
    negates_pattern: str = Field(
        description="Pattern or keyword this inhibits in retrieval"
    )
    source_episode_ids: list[str] = Field(
        default_factory=list,
        description="Episodes where this negation was stated",
    )
    derived_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When this negation was identified",
    )
    confidence: ConfidenceScore = Field(
        default_factory=lambda: ConfidenceScore.for_inferred(0.7),
        description="Composite confidence score",
    )

    def __str__(self) -> str:
        """String representation showing negation content."""
        return f"InhibitoryFact({self.content})"
