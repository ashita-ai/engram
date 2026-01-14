"""Structure workflow for per-episode LLM extraction.

This workflow creates StructuredMemory from Episodes:
- Runs immediately or deferred after encode()
- Uses deterministic extractors (regex) for emails, phones, URLs
- Uses Pydantic AI for dates, people, orgs, preferences, negations
- Creates an immutable StructuredMemory (one per Episode)

The structure workflow bridges raw Episodes and cross-episode Semantic:
    Episode (raw) -> StructuredMemory (per-episode intelligence) -> SemanticMemory (synthesis)
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from engram.embeddings import Embedder
    from engram.models import Episode
    from engram.storage import EngramStorage

logger = logging.getLogger(__name__)


class ExtractedDate(BaseModel):
    """A date extracted by the LLM."""

    model_config = ConfigDict(extra="forbid")

    raw: str = Field(description="Original text (e.g., 'next Tuesday')")
    resolved: str = Field(description="Resolved date in YYYY-MM-DD format")
    context: str = Field(default="", description="What this date refers to")


class ExtractedPerson(BaseModel):
    """A person extracted by the LLM."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Person's name")
    role: str | None = Field(default=None, description="Role or relationship to the user")
    context: str = Field(default="", description="Additional context")


class ExtractedPreference(BaseModel):
    """A preference extracted by the LLM."""

    model_config = ConfigDict(extra="forbid")

    topic: str = Field(description="Topic of preference (e.g., 'database', 'language')")
    value: str = Field(description="The preferred value (e.g., 'PostgreSQL')")
    sentiment: str = Field(default="positive", description="positive, negative, or neutral")


class ExtractedNegation(BaseModel):
    """A negation or correction extracted by the LLM."""

    model_config = ConfigDict(extra="forbid")

    content: str = Field(description="The negation statement")
    pattern: str = Field(description="Keyword pattern to filter in retrieval")
    context: str = Field(default="", description="Context for the negation")


class LLMExtractionOutput(BaseModel):
    """Structured output from the LLM extraction agent."""

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(
        default="",
        description="1-2 sentence summary of the episode content",
    )
    keywords: list[str] = Field(
        default_factory=list,
        description="Key terms for retrieval (5-10 words)",
    )
    dates: list[ExtractedDate] = Field(
        default_factory=list,
        description="Dates mentioned with resolution and context",
    )
    people: list[ExtractedPerson] = Field(
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
    preferences: list[ExtractedPreference] = Field(
        default_factory=list,
        description="User preferences identified",
    )
    negations: list[ExtractedNegation] = Field(
        default_factory=list,
        description="Explicit negations or corrections",
    )


class StructureResult(BaseModel):
    """Result of a structure workflow run.

    Attributes:
        episode_id: ID of the Episode that was processed.
        structured_memory_id: ID of the created StructuredMemory.
        extracts_count: Total number of entities extracted.
        deterministic_count: Number of regex extractions (emails, phones, URLs).
        llm_count: Number of LLM extractions.
        processing_time_ms: Time taken to process.
    """

    model_config = ConfigDict(extra="forbid")

    episode_id: str
    structured_memory_id: str
    extracts_count: int = Field(ge=0)
    deterministic_count: int = Field(ge=0)
    llm_count: int = Field(ge=0)
    processing_time_ms: float = Field(ge=0.0)


def _run_deterministic_extractors(episode: Episode) -> tuple[list[str], list[str], list[str]]:
    """Run regex-based extractors for emails, phones, URLs.

    Args:
        episode: Episode to extract from.

    Returns:
        Tuple of (emails, phones, urls) lists.
    """
    from engram.extraction import EmailExtractor, PhoneExtractor, URLExtractor

    emails: list[str] = []
    phones: list[str] = []
    urls: list[str] = []

    # Email extraction - returns Facts with .content field
    email_extractor = EmailExtractor()
    for fact in email_extractor.extract(episode):
        emails.append(fact.content)

    # Phone extraction
    phone_extractor = PhoneExtractor()
    for fact in phone_extractor.extract(episode):
        phones.append(fact.content)

    # URL extraction
    url_extractor = URLExtractor()
    for fact in url_extractor.extract(episode):
        urls.append(fact.content)

    return emails, phones, urls


async def _run_llm_extraction(
    content: str,
    role: str,
    model: str | None = None,
) -> LLMExtractionOutput:
    """Run LLM extraction using Pydantic AI.

    Args:
        content: Episode content to extract from.
        role: Role of the message (user, assistant, system).
        model: Optional model override.

    Returns:
        LLMExtractionOutput with extracted entities.
    """
    from pydantic_ai import Agent

    from engram.config import settings

    model_spec = model or settings.consolidation_model

    extraction_prompt = f"""You are extracting structured information from a conversation message.

The message is from a {role}. Extract:

1. SUMMARY: A 1-2 sentence summary of the key information
2. KEYWORDS: 5-10 key terms for retrieval
3. DATES: Any dates mentioned (resolve relative dates like "next Tuesday" to actual dates)
4. PEOPLE: People mentioned with their roles/relationships
5. ORGANIZATIONS: Companies, teams, or groups mentioned
6. LOCATIONS: Places mentioned
7. PREFERENCES: User preferences (what they like/dislike/prefer)
8. NEGATIONS: Explicit corrections or "not" statements that should filter other information

For negations, look for:
- "I don't like X" → pattern="X"
- "We switched from X to Y" → pattern="X" (X is no longer used)
- "I'm not interested in X" → pattern="X"
- "That's wrong, I actually..." → pattern for the wrong info

Be conservative - only extract information that is clearly stated.
Today's date is {datetime.now(UTC).strftime('%Y-%m-%d')} for resolving relative dates."""

    agent: Agent[None, LLMExtractionOutput] = Agent(
        model_spec,
        output_type=LLMExtractionOutput,
        instructions=extraction_prompt,
    )

    result = await agent.run(content)
    return result.output


async def run_structure(
    episode: Episode,
    storage: EngramStorage,
    embedder: Embedder,
    model: str | None = None,
    skip_if_structured: bool = True,
) -> StructureResult | None:
    """Run the structure workflow on a single Episode.

    Creates a StructuredMemory containing both deterministic (regex)
    and LLM-extracted entities. The StructuredMemory is stored and
    linked to the Episode.

    Args:
        episode: The Episode to process.
        storage: EngramStorage instance.
        embedder: Embedder for generating vectors.
        model: Optional model override for LLM.
        skip_if_structured: Skip if already structured (default True).

    Returns:
        StructureResult with processing statistics, or None if skipped.
    """
    import time

    from engram.models import (
        Negation,
        Person,
        Preference,
        QuickExtracts,
        ResolvedDate,
        StructuredMemory,
    )

    start_time = time.monotonic()

    # Check if already structured
    if skip_if_structured and episode.structured:
        logger.debug(f"Episode {episode.id} already structured, skipping")
        return None

    # 1. Run deterministic extractors
    emails, phones, urls = _run_deterministic_extractors(episode)
    deterministic_count = len(emails) + len(phones) + len(urls)

    logger.debug(
        f"Deterministic extraction: {len(emails)} emails, {len(phones)} phones, {len(urls)} urls"
    )

    # 2. Update Episode's quick_extracts for fast access
    quick_extracts = QuickExtracts(emails=emails, phones=phones, urls=urls)

    # 3. Run LLM extraction
    llm_output = await _run_llm_extraction(episode.content, episode.role, model)

    # Convert LLM output to model types
    dates = [
        ResolvedDate(raw=d.raw, resolved=d.resolved, context=d.context) for d in llm_output.dates
    ]
    people = [Person(name=p.name, role=p.role, context=p.context) for p in llm_output.people]
    preferences = [
        Preference(topic=p.topic, value=p.value, sentiment=p.sentiment)
        for p in llm_output.preferences
    ]
    negations = [
        Negation(content=n.content, pattern=n.pattern, context=n.context)
        for n in llm_output.negations
    ]

    llm_count = (
        len(dates)
        + len(people)
        + len(llm_output.organizations)
        + len(llm_output.locations)
        + len(preferences)
        + len(negations)
    )

    logger.debug(
        f"LLM extraction: {len(dates)} dates, {len(people)} people, "
        f"{len(llm_output.organizations)} orgs, {len(llm_output.locations)} locations, "
        f"{len(preferences)} preferences, {len(negations)} negations"
    )

    # 4. Create StructuredMemory
    structured = StructuredMemory.from_episode(
        source_episode_id=episode.id,
        user_id=episode.user_id,
        org_id=episode.org_id,
        # Deterministic
        emails=emails,
        phones=phones,
        urls=urls,
        # LLM
        dates=dates,
        people=people,
        organizations=llm_output.organizations,
        locations=llm_output.locations,
        preferences=preferences,
        negations=negations,
        # Summary
        summary=llm_output.summary,
        keywords=llm_output.keywords,
    )

    # 5. Generate embedding from structured content
    embedding_text = structured.to_embedding_text()
    if embedding_text:
        structured.embedding = await embedder.embed(embedding_text)
    else:
        # Fallback to original content
        structured.embedding = episode.embedding

    # 6. Store StructuredMemory
    await storage.store_structured(structured)

    # 7. Update Episode with structured flag and quick_extracts
    episode.structured = True
    episode.structured_into = structured.id
    episode.quick_extracts = quick_extracts

    # Update episode in storage
    await storage.update_episode(episode)

    elapsed_ms = (time.monotonic() - start_time) * 1000

    logger.info(
        f"Structured episode {episode.id} -> {structured.id} "
        f"({deterministic_count} regex, {llm_count} LLM, {elapsed_ms:.1f}ms)"
    )

    return StructureResult(
        episode_id=episode.id,
        structured_memory_id=structured.id,
        extracts_count=deterministic_count + llm_count,
        deterministic_count=deterministic_count,
        llm_count=llm_count,
        processing_time_ms=elapsed_ms,
    )


async def run_structure_batch(
    storage: EngramStorage,
    embedder: Embedder,
    user_id: str,
    org_id: str | None = None,
    limit: int | None = None,
    model: str | None = None,
) -> list[StructureResult]:
    """Run structure workflow on all unstructured episodes.

    Args:
        storage: EngramStorage instance.
        embedder: Embedder for generating vectors.
        user_id: User ID to process.
        org_id: Optional org ID filter.
        limit: Maximum episodes to process.
        model: Optional model override for LLM.

    Returns:
        List of StructureResult for each processed episode.
    """
    # Get unstructured episodes
    episodes = await storage.get_unstructured_episodes(
        user_id=user_id,
        org_id=org_id,
        limit=limit,
    )

    if not episodes:
        logger.info("No unstructured episodes found")
        return []

    logger.info(f"Processing {len(episodes)} unstructured episodes")

    results: list[StructureResult] = []
    for episode in episodes:
        result = await run_structure(
            episode=episode,
            storage=storage,
            embedder=embedder,
            model=model,
            skip_if_structured=True,
        )
        if result:
            results.append(result)

    logger.info(f"Structured {len(results)} episodes")
    return results


__all__ = [
    "ExtractedDate",
    "ExtractedNegation",
    "ExtractedPerson",
    "ExtractedPreference",
    "LLMExtractionOutput",
    "StructureResult",
    "run_structure",
    "run_structure_batch",
]
