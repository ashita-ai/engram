"""Fact model - deterministically extracted factual information."""

from datetime import UTC, datetime

from pydantic import Field

from .base import ConfidenceScore, ExtractionMethod, MemoryBase, generate_id


class Fact(MemoryBase):
    """Factual memory extracted via deterministic pattern matching.

    Facts are extracted from Episodes using pattern matchers (regex, validators)
    rather than LLM inference. This makes them highly reliable with confidence
    scores of 0.9 (extracted) vs 0.6 (inferred).

    Categories include: email, phone, url, date, name, etc.

    Attributes:
        content: The extracted fact (e.g., "email=john@example.com").
        category: Type of fact (email, phone, url, date, etc.).
        source_episode_id: The Episode this was extracted from.
        event_at: When the fact was stated (from source Episode).
        derived_at: When we extracted this fact.
        confidence: Composite confidence score with auditability.
    """

    id: str = Field(default_factory=lambda: generate_id("fact"))
    content: str = Field(description="The extracted fact content")
    category: str = Field(description="Fact category: email, phone, url, date, etc.")
    source_episode_id: str = Field(
        description="ID of the Episode this was extracted from"
    )
    event_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When the fact was stated (from source)",
    )
    derived_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When we extracted this fact",
    )
    confidence: ConfidenceScore = Field(
        default_factory=ConfidenceScore.for_extracted,
        description="Composite confidence score",
    )

    @classmethod
    def from_extraction(
        cls,
        content: str,
        category: str,
        source_episode_id: str,
        user_id: str,
        org_id: str | None = None,
        event_at: datetime | None = None,
        embedding: list[float] | None = None,
    ) -> "Fact":
        """Create a Fact from a pattern extraction result."""
        return cls(
            content=content,
            category=category,
            source_episode_id=source_episode_id,
            user_id=user_id,
            org_id=org_id,
            event_at=event_at or datetime.now(UTC),
            confidence=ConfidenceScore(
                value=0.9,
                extraction_method=ExtractionMethod.EXTRACTED,
                extraction_base=0.9,
            ),
            embedding=embedding,
        )

    def __str__(self) -> str:
        """String representation showing category and content."""
        return f"Fact({self.category}: {self.content})"
