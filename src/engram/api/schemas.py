"""Pydantic schemas for API request/response models."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from engram.models import Staleness

# Valid memory types for the memory_types parameter
MemoryType = Literal["episodic", "structured", "semantic", "procedural", "working"]
ALL_MEMORY_TYPES: set[str] = {
    "episodic",
    "structured",
    "semantic",
    "procedural",
    "working",
}


class EncodeRequest(BaseModel):
    """Request body for encoding a memory.

    Attributes:
        content: The text content to encode.
        role: Role of the speaker (user, assistant, system).
        user_id: User ID for multi-tenancy isolation.
        org_id: Optional organization ID.
        session_id: Optional session ID for grouping.
        importance: Importance score (0.0-1.0).
        enrich: LLM enrichment mode (False=regex only, True=sync LLM, "background"=async).
    """

    model_config = ConfigDict(extra="forbid")

    content: str = Field(min_length=1, description="Text content to encode")
    role: Literal["user", "assistant", "system"] = Field(
        default="user", description="Role of the speaker"
    )
    user_id: str = Field(min_length=1, description="User ID for isolation")
    org_id: str | None = Field(default=None, description="Optional org ID")
    session_id: str | None = Field(default=None, description="Optional session ID")
    importance: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Importance score (auto-calculated if not provided)",
    )
    enrich: bool | Literal["background"] = Field(
        default=False,
        description="LLM enrichment: False=regex only, True=sync LLM, 'background'=async",
    )


class StructuredResponse(BaseModel):
    """Response model for a structured memory extraction.

    Attributes:
        id: Unique structured memory ID.
        source_episode_id: ID of the source episode.
        mode: Extraction mode ("fast" or "rich").
        enriched: Whether LLM enrichment was applied.
        emails: Extracted email addresses.
        phones: Extracted phone numbers.
        urls: Extracted URLs.
        confidence: Confidence score.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    source_episode_id: str
    mode: str
    enriched: bool
    emails: list[str] = Field(default_factory=list)
    phones: list[str] = Field(default_factory=list)
    urls: list[str] = Field(default_factory=list)
    confidence: float


class EpisodeResponse(BaseModel):
    """Response model for an episode.

    Attributes:
        id: Unique episode ID.
        content: The episode content.
        role: Role of the speaker.
        user_id: User ID.
        org_id: Optional org ID.
        session_id: Optional session ID.
        importance: Importance score.
        created_at: ISO timestamp of creation.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    content: str
    role: str
    user_id: str
    org_id: str | None = None
    session_id: str | None = None
    importance: float
    created_at: str


class EncodeResponse(BaseModel):
    """Response body for encode operation.

    Attributes:
        episode: The stored episode.
        structured: The structured memory with extractions.
        extract_count: Total number of extracts (emails + phones + urls).
    """

    model_config = ConfigDict(extra="forbid")

    episode: EpisodeResponse
    structured: StructuredResponse
    extract_count: int = Field(description="Total number of extracts")


class RecallRequest(BaseModel):
    """Request body for recalling memories.

    Attributes:
        query: Natural language query.
        user_id: User ID for multi-tenancy isolation.
        org_id: Optional organization ID filter.
        limit: Maximum results to return.
        min_confidence: Minimum confidence for facts.
        min_selectivity: Minimum selectivity for semantic memories (0.0-1.0).
        memory_types: List of memory types to search. None means all types.
        include_sources: Whether to include source episodes in results.
        follow_links: Enable multi-hop reasoning via related_ids.
        max_hops: Maximum link traversal depth when follow_links=True.
        freshness: Freshness mode for results.
        as_of: Optional bi-temporal filter (only memories derived before this time).
    """

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, description="Natural language query")
    user_id: str = Field(min_length=1, description="User ID for isolation")
    org_id: str | None = Field(default=None, description="Optional org ID filter")
    limit: int = Field(default=10, ge=1, le=100, description="Maximum results")
    min_confidence: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Minimum confidence threshold"
    )
    min_selectivity: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Minimum selectivity for semantic memories"
    )
    memory_types: list[MemoryType] | None = Field(
        default=None,
        description="Memory types to search. None means all.",
    )
    include_sources: bool = Field(default=False, description="Include source episodes in results")
    follow_links: bool = Field(default=False, description="Enable multi-hop reasoning")
    max_hops: int = Field(default=2, ge=1, le=5, description="Maximum link traversal depth")
    freshness: Literal["best_effort", "fresh_only"] = Field(
        default="best_effort",
        description="Freshness mode: best_effort returns all, fresh_only only consolidated",
    )
    as_of: datetime | None = Field(
        default=None,
        description="Bi-temporal query: only return memories derived before this time",
    )
    include_system_prompts: bool = Field(
        default=False,
        description="Include system prompt episodes in results (default False)",
    )


class SourceEpisodeSummary(BaseModel):
    """Lightweight summary of a source episode.

    Used when include_sources=True in recall requests.

    Attributes:
        id: Episode ID.
        content: Episode content.
        role: Role of the speaker.
        timestamp: ISO timestamp.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    content: str
    role: str
    timestamp: str


class RecallResultResponse(BaseModel):
    """Response model for a single recalled memory.

    Attributes:
        memory_type: Type of memory (episodic, structured, semantic, procedural).
        content: The memory content.
        score: Similarity score (0.0-1.0).
        confidence: Confidence score for structured/semantic memories.
        memory_id: Unique memory ID.
        source_episode_id: Source episode ID (single source).
        source_episode_ids: Source episode IDs for memories with multiple sources.
        source_episodes: Source episode details (when include_sources=True).
        related_ids: IDs of related memories (for multi-hop).
        hop_distance: Distance from original query result (0=direct, 1=1-hop, etc.).
        staleness: Freshness state (fresh, consolidating, stale).
        consolidated_at: When this memory was last consolidated.
        metadata: Additional memory-specific metadata.
    """

    model_config = ConfigDict(extra="forbid")

    memory_type: str
    content: str
    score: float
    confidence: float | None = None
    memory_id: str
    source_episode_id: str | None = None
    source_episode_ids: list[str] = Field(default_factory=list)
    source_episodes: list[SourceEpisodeSummary] = Field(default_factory=list)
    related_ids: list[str] = Field(default_factory=list)
    hop_distance: int = Field(default=0, ge=0, description="Distance from original query result")
    staleness: Staleness = Field(default=Staleness.FRESH, description="Freshness state")
    consolidated_at: str | None = Field(default=None, description="When last consolidated")
    metadata: dict[str, Any] = Field(default_factory=dict)


class RecallResponse(BaseModel):
    """Response body for recall operation.

    Attributes:
        query: The original query.
        results: List of recalled memories.
        count: Number of results returned.
    """

    model_config = ConfigDict(extra="forbid")

    query: str
    results: list[RecallResultResponse]
    count: int


class HealthResponse(BaseModel):
    """Response for health check endpoint.

    Attributes:
        status: Service status (healthy, degraded, unhealthy).
        version: API version.
        storage_connected: Whether storage is connected.
    """

    model_config = ConfigDict(extra="forbid")

    status: Literal["healthy", "degraded", "unhealthy"]
    version: str
    storage_connected: bool


class MemoryCounts(BaseModel):
    """Counts of memories by type.

    Attributes:
        episodes: Number of episode memories.
        structured: Number of structured memories.
        semantic: Number of semantic memories (from consolidation).
        procedural: Number of procedural memories.
    """

    model_config = ConfigDict(extra="forbid")

    episodes: int = Field(ge=0)
    structured: int = Field(ge=0)
    semantic: int = Field(ge=0)
    procedural: int = Field(ge=0)


class ConfidenceStats(BaseModel):
    """Confidence statistics for memories.

    Attributes:
        structured_avg: Average confidence of structured memories.
        structured_min: Minimum confidence of structured memories.
        structured_max: Maximum confidence of structured memories.
        semantic_avg: Average confidence of semantic memories.
    """

    model_config = ConfigDict(extra="forbid")

    structured_avg: float | None = Field(default=None, description="Average structured confidence")
    structured_min: float | None = Field(default=None, description="Minimum structured confidence")
    structured_max: float | None = Field(default=None, description="Maximum structured confidence")
    semantic_avg: float | None = Field(default=None, description="Average semantic confidence")


class MemoryStatsResponse(BaseModel):
    """Response for memory statistics endpoint.

    Provides visibility into what memories are stored, their types,
    and confidence levels.

    Attributes:
        user_id: User ID for the stats.
        org_id: Optional org ID filter.
        counts: Memory counts by type.
        confidence: Confidence statistics.
        pending_consolidation: Number of episodes awaiting consolidation.
    """

    model_config = ConfigDict(extra="forbid")

    user_id: str
    org_id: str | None = None
    counts: MemoryCounts
    confidence: ConfidenceStats
    pending_consolidation: int = Field(ge=0, description="Episodes awaiting LLM consolidation")


class WorkingMemoryResponse(BaseModel):
    """Response for working memory endpoint.

    Working memory contains episodes from the current session.
    It's volatile (in-memory only) and cleared when the session ends.

    Attributes:
        episodes: List of episodes in working memory.
        count: Number of episodes in working memory.
    """

    model_config = ConfigDict(extra="forbid")

    episodes: list[EpisodeResponse]
    count: int = Field(ge=0, description="Number of episodes in working memory")


class SourcesResponse(BaseModel):
    """Response for get_sources endpoint.

    Returns the source episodes that a derived memory (structured, semantic,
    procedural) was extracted from.

    Attributes:
        memory_id: The ID of the derived memory.
        memory_type: Type of memory (structured, semantic, procedural).
        sources: Source episodes in chronological order.
        count: Number of source episodes.
    """

    model_config = ConfigDict(extra="forbid")

    memory_id: str
    memory_type: str
    sources: list[EpisodeResponse]
    count: int = Field(ge=0, description="Number of source episodes")


class SourceEpisodeDetail(BaseModel):
    """Detail about a source episode in verification result.

    Attributes:
        id: Episode ID.
        content: Episode content.
        role: Role of the speaker.
        timestamp: ISO timestamp.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    content: str
    role: str
    timestamp: str


class VerificationResponse(BaseModel):
    """Response for verify endpoint.

    Provides full traceability from a derived memory back to
    its source episodes with an explanation of derivation.

    Attributes:
        memory_id: ID of the verified memory.
        memory_type: Type of memory (structured, semantic, procedural).
        content: The memory content.
        verified: True if sources found and traceable.
        source_episodes: Source episode details.
        extraction_method: How memory was extracted.
        confidence: Current confidence score.
        explanation: Human-readable derivation trace.
    """

    model_config = ConfigDict(extra="forbid")

    memory_id: str = Field(description="ID of the verified memory")
    memory_type: str = Field(description="Type: structured, semantic, procedural")
    content: str = Field(description="The memory content")
    verified: bool = Field(description="True if sources found and traceable")
    source_episodes: list[SourceEpisodeDetail] = Field(
        default_factory=list, description="Source episode details"
    )
    extraction_method: str = Field(description="How memory was extracted")
    confidence: float = Field(ge=0.0, le=1.0, description="Current confidence score")
    explanation: str = Field(description="Human-readable derivation trace")


class BulkDeleteResponse(BaseModel):
    """Response for bulk delete (GDPR erasure) endpoint.

    Provides counts of deleted memories by type for confirmation.

    Attributes:
        user_id: User whose memories were deleted.
        org_id: Optional org filter used.
        deleted_counts: Count of deleted memories by type.
        total_deleted: Total number of memories deleted.
    """

    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(description="User whose memories were deleted")
    org_id: str | None = Field(default=None, description="Org filter used (if any)")
    deleted_counts: dict[str, int] = Field(description="Counts by memory type")
    total_deleted: int = Field(ge=0, description="Total memories deleted")
