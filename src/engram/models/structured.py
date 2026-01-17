"""StructuredMemory model - per-episode structured extraction.

StructuredMemory represents the single-episode intelligence layer:
- Always created for every Episode (fast mode by default)
- Contains deterministic (regex) extracts always
- Optionally enriched with LLM-extracted entities (rich mode)
- Immutable once created
- Bridges the gap between raw Episodes and cross-episode Semantic memories

Modes:
- fast: Regex extraction only (emails, phones, URLs) - no LLM, immediate
- rich: Regex + LLM extraction (dates, people, orgs, prefs, negations)
"""

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .base import ConfidenceScore, ExtractionMethod, MemoryBase, generate_id

# Structured memory modes
StructuredMode = Literal["fast", "rich"]


class ResolvedDate(BaseModel):
    """A date extracted and resolved from natural language.

    Attributes:
        raw: The original text (e.g., "next Tuesday").
        resolved: The resolved date string (e.g., "2026-01-20").
        context: What this date refers to (e.g., "meeting", "deadline").
    """

    model_config = ConfigDict(extra="forbid")

    raw: str = Field(description="Original text (e.g., 'next Tuesday')")
    resolved: str = Field(description="Resolved date (e.g., '2026-01-20')")
    context: str = Field(default="", description="What this date refers to")


class Person(BaseModel):
    """A person mentioned in the episode.

    Attributes:
        name: The person's name.
        role: Their role or relationship (e.g., "manager", "colleague").
        context: Additional context about this person.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Person's name")
    role: str | None = Field(default=None, description="Role or relationship")
    context: str = Field(default="", description="Additional context")


class Preference(BaseModel):
    """A user preference extracted from the episode.

    Attributes:
        topic: What the preference is about (e.g., "database", "language").
        value: The preferred value (e.g., "PostgreSQL", "Python").
        sentiment: Whether positive, negative, or neutral.
    """

    model_config = ConfigDict(extra="forbid")

    topic: str = Field(description="Topic of preference")
    value: str = Field(description="The preferred value")
    sentiment: str = Field(
        default="positive",
        description="Sentiment: positive, negative, or neutral",
    )


class Negation(BaseModel):
    """An explicit negation or correction from the episode.

    Negations track what is explicitly NOT true, enabling
    filtering of outdated or contradicted information.

    Attributes:
        content: The negation statement (e.g., "does not use MongoDB").
        pattern: Keyword pattern to filter (e.g., "MongoDB").
        context: Why this negation exists (e.g., "switched to PostgreSQL").
    """

    model_config = ConfigDict(extra="forbid")

    content: str = Field(description="The negation statement")
    pattern: str = Field(description="Keyword pattern to filter in retrieval")
    context: str = Field(default="", description="Context for the negation")


class StructuredMemory(MemoryBase):
    """Per-episode structured extraction.

    StructuredMemory is created by processing a single Episode with
    both deterministic (regex) and LLM-based extraction. It represents
    the "understanding" layer - what we know from THIS specific episode.

    Key characteristics:
    - One StructuredMemory per Episode (1:1 relationship)
    - Contains both high-confidence deterministic data and LLM-extracted entities
    - Immutable once created (if extraction is wrong, create a new one)
    - Serves as input to Semantic consolidation (cross-episode synthesis)

    Hierarchy:
        Episodic (raw, verbatim)
            -> Structured (per-episode intelligence, THIS MODEL)
                -> Semantic (cross-episode synthesis)
                    -> Procedural (behavioral profile)

    Attributes:
        source_episode_id: The Episode this was extracted from.

        # Deterministic extraction (0.9 confidence) - from regex extractors
        emails: Email addresses found.
        phones: Phone numbers found.
        urls: URLs found.

        # LLM extraction (0.8 confidence) - from Pydantic AI
        dates: Resolved dates with context.
        people: People mentioned with roles.
        organizations: Organizations mentioned.
        locations: Locations mentioned.
        preferences: User preferences identified.
        negations: Explicit negations/corrections.

        # Summary
        summary: 1-2 sentence summary of the episode.
        keywords: Key terms for retrieval.

        # Metadata
        derived_at: When this extraction was performed.
        confidence: Composite confidence score.
        structured_at: Timestamp marking completion (immutability marker).
    """

    id: str = Field(default_factory=lambda: generate_id("struct"))
    source_episode_id: str = Field(description="ID of the Episode this was extracted from")

    # Provenance tracking
    derivation_method: str = Field(
        default="fast:regex",
        description="How this was extracted (e.g., 'fast:regex', 'rich:llm:gpt-4o-mini')",
    )

    # Deterministic extraction (regex-based, 0.9 confidence)
    emails: list[str] = Field(default_factory=list, description="Email addresses (regex)")
    phones: list[str] = Field(default_factory=list, description="Phone numbers (regex)")
    urls: list[str] = Field(default_factory=list, description="URLs (regex)")

    # LLM extraction (0.8 confidence)
    dates: list[ResolvedDate] = Field(
        default_factory=list,
        description="Resolved dates with context",
    )
    people: list[Person] = Field(
        default_factory=list,
        description="People mentioned with roles",
    )
    organizations: list[str] = Field(
        default_factory=list,
        description="Organizations mentioned",
    )
    locations: list[str] = Field(
        default_factory=list,
        description="Locations mentioned",
    )
    preferences: list[Preference] = Field(
        default_factory=list,
        description="User preferences identified",
    )
    negations: list[Negation] = Field(
        default_factory=list,
        description="Explicit negations/corrections",
    )

    # Summary for retrieval
    summary: str = Field(
        default="",
        description="1-2 sentence summary of the episode content",
    )
    keywords: list[str] = Field(
        default_factory=list,
        description="Key terms for retrieval",
    )

    # Metadata
    derived_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When this extraction was performed",
    )
    confidence: ConfidenceScore = Field(
        default_factory=lambda: ConfidenceScore(
            value=0.8,
            extraction_method=ExtractionMethod.INFERRED,
            extraction_base=0.8,
        ),
        description="Composite confidence score (weighted by extraction methods)",
    )

    # Mode tracking
    mode: StructuredMode = Field(
        default="fast",
        description="Extraction mode: 'fast' (regex only) or 'rich' (regex + LLM)",
    )
    enriched: bool = Field(
        default=False,
        description="Whether LLM enrichment has been applied",
    )

    # Consolidation tracking
    consolidated: bool = Field(
        default=False,
        description="Whether this has been consolidated into SemanticMemory",
    )
    consolidated_into: str | None = Field(
        default=None,
        description="ID of the SemanticMemory this was consolidated into",
    )

    # Retrieval tracking (for Testing Effect)
    retrieval_count: int = Field(
        default=0,
        ge=0,
        description="Number of times this memory has been retrieved",
    )
    last_accessed: datetime | None = Field(
        default=None,
        description="When this memory was last retrieved",
    )

    def record_access(self) -> None:
        """Record that this memory was accessed (activation tracking).

        Increments retrieval_count and updates last_accessed timestamp.
        Called by storage layer on search hits.
        """
        self.retrieval_count += 1
        self.last_accessed = datetime.now(UTC)

    @classmethod
    def from_episode_fast(
        cls,
        source_episode_id: str,
        user_id: str,
        org_id: str | None = None,
        emails: list[str] | None = None,
        phones: list[str] | None = None,
        urls: list[str] | None = None,
        embedding: list[float] | None = None,
    ) -> "StructuredMemory":
        """Create a fast-mode StructuredMemory (regex extracts only).

        Fast mode extracts only deterministic data using regex:
        - Emails, phones, URLs

        No LLM calls are made. This is the default for encode().

        Args:
            source_episode_id: The Episode this was extracted from.
            user_id: User who owns this memory.
            org_id: Optional organization.
            emails: Regex-extracted emails.
            phones: Regex-extracted phones.
            urls: Regex-extracted URLs.
            embedding: Vector embedding.

        Returns:
            StructuredMemory instance in fast mode.
        """
        deterministic_count = len(emails or []) + len(phones or []) + len(urls or [])

        # Fast mode: high confidence for regex extracts, baseline otherwise
        confidence_value = 0.9 if deterministic_count > 0 else 0.7

        return cls(
            source_episode_id=source_episode_id,
            user_id=user_id,
            org_id=org_id,
            emails=emails or [],
            phones=phones or [],
            urls=urls or [],
            embedding=embedding,
            mode="fast",
            enriched=False,
            confidence=ConfidenceScore(
                value=confidence_value,
                extraction_method=ExtractionMethod.EXTRACTED,
                extraction_base=0.9,
            ),
        )

    @classmethod
    def from_episode(
        cls,
        source_episode_id: str,
        user_id: str,
        org_id: str | None = None,
        # Deterministic extracts
        emails: list[str] | None = None,
        phones: list[str] | None = None,
        urls: list[str] | None = None,
        # LLM extracts
        dates: list[ResolvedDate] | None = None,
        people: list[Person] | None = None,
        organizations: list[str] | None = None,
        locations: list[str] | None = None,
        preferences: list[Preference] | None = None,
        negations: list[Negation] | None = None,
        # Summary
        summary: str = "",
        keywords: list[str] | None = None,
        embedding: list[float] | None = None,
        # Provenance
        derivation_method: str | None = None,
    ) -> "StructuredMemory":
        """Create a rich-mode StructuredMemory (regex + LLM extracts).

        Rich mode includes both deterministic and LLM-extracted data:
        - Deterministic: emails, phones, URLs (regex)
        - LLM: dates, people, orgs, locations, preferences, negations

        Computes composite confidence based on what was extracted:
        - If only deterministic data: 0.9 confidence
        - If only LLM data: 0.8 confidence
        - If both: weighted average (~0.85)

        Args:
            source_episode_id: The Episode this was extracted from.
            user_id: User who owns this memory.
            org_id: Optional organization.
            emails: Regex-extracted emails.
            phones: Regex-extracted phones.
            urls: Regex-extracted URLs.
            dates: LLM-extracted dates.
            people: LLM-extracted people.
            organizations: LLM-extracted organizations.
            locations: LLM-extracted locations.
            preferences: LLM-extracted preferences.
            negations: LLM-extracted negations.
            summary: Episode summary.
            keywords: Key terms.
            embedding: Vector embedding.
            derivation_method: How this was extracted (e.g., 'rich:llm:gpt-4o-mini').

        Returns:
            StructuredMemory instance in rich mode.
        """
        # Count deterministic vs LLM extracts for confidence weighting
        deterministic_count = len(emails or []) + len(phones or []) + len(urls or [])
        llm_count = (
            len(dates or [])
            + len(people or [])
            + len(organizations or [])
            + len(locations or [])
            + len(preferences or [])
            + len(negations or [])
        )

        # Compute weighted confidence
        if deterministic_count + llm_count == 0:
            # No extracts, but we have a summary - LLM confidence
            confidence_value = 0.8
        elif deterministic_count > 0 and llm_count == 0:
            # Only deterministic - high confidence
            confidence_value = 0.9
        elif deterministic_count == 0 and llm_count > 0:
            # Only LLM - medium-high confidence
            confidence_value = 0.8
        else:
            # Both - weighted average
            total = deterministic_count + llm_count
            confidence_value = (0.9 * deterministic_count + 0.8 * llm_count) / total

        return cls(
            source_episode_id=source_episode_id,
            user_id=user_id,
            org_id=org_id,
            emails=emails or [],
            phones=phones or [],
            urls=urls or [],
            dates=dates or [],
            people=people or [],
            organizations=organizations or [],
            locations=locations or [],
            preferences=preferences or [],
            negations=negations or [],
            summary=summary,
            keywords=keywords or [],
            embedding=embedding,
            mode="rich",
            enriched=True,
            derivation_method=derivation_method or "rich:llm:unknown",
            confidence=ConfidenceScore(
                value=confidence_value,
                extraction_method=ExtractionMethod.INFERRED,
                extraction_base=0.8,
            ),
        )

    def has_negations(self) -> bool:
        """Check if this structured memory contains any negations."""
        return len(self.negations) > 0

    def get_negation_patterns(self) -> list[str]:
        """Get all negation patterns for filtering."""
        return [n.pattern for n in self.negations]

    def to_embedding_text(self) -> str:
        """Generate text representation for embedding.

        Combines summary, keywords, entities, and preferences into
        a single text block suitable for embedding.
        """
        parts = []

        if self.summary:
            parts.append(self.summary)

        if self.keywords:
            parts.append(f"Keywords: {', '.join(self.keywords)}")

        # Entities
        entities: list[str] = []
        if self.people:
            entities.extend(f"{p.name} ({p.role})" if p.role else p.name for p in self.people)
        if self.organizations:
            entities.extend(self.organizations)
        if self.locations:
            entities.extend(self.locations)
        if entities:
            parts.append(f"Entities: {', '.join(entities)}")

        # Preferences
        if self.preferences:
            pref_strs = [f"{p.topic}: {p.value} ({p.sentiment})" for p in self.preferences]
            parts.append(f"Preferences: {', '.join(pref_strs)}")

        # Negations
        if self.negations:
            neg_strs = [n.content for n in self.negations]
            parts.append(f"Negations: {', '.join(neg_strs)}")

        # Deterministic data
        if self.emails:
            parts.append(f"Emails: {', '.join(self.emails)}")
        if self.phones:
            parts.append(f"Phones: {', '.join(self.phones)}")

        return " | ".join(parts) if parts else ""

    def __str__(self) -> str:
        """String representation showing summary and extract counts."""
        summary_preview = self.summary[:40] + "..." if len(self.summary) > 40 else self.summary
        extract_count = (
            len(self.emails)
            + len(self.phones)
            + len(self.urls)
            + len(self.dates)
            + len(self.people)
            + len(self.preferences)
            + len(self.negations)
        )
        return f"StructuredMemory({summary_preview!r}, {extract_count} extracts)"
