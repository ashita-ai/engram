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


class GetMemoryResponse(BaseModel):
    """Response for retrieving a specific memory by ID.

    Returns the memory content and metadata regardless of memory type.
    The response format adapts based on the memory type.

    Attributes:
        id: Memory ID.
        memory_type: Type of memory (episodic, structured, semantic, procedural).
        content: The memory content (varies by type).
        user_id: User ID.
        org_id: Optional org ID.
        confidence: Confidence score (None for episodic).
        source_episode_ids: Source episode IDs (for derived memories).
        created_at: ISO timestamp of creation.
        metadata: Additional type-specific metadata.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    memory_type: str
    content: str
    user_id: str
    org_id: str | None = None
    confidence: float | None = None
    source_episode_ids: list[str] = Field(default_factory=list)
    created_at: str
    metadata: dict[str, object] = Field(default_factory=dict)


# Valid sort fields for memory listing
SortField = Literal["created_at", "confidence"]
SortOrder = Literal["asc", "desc"]


class MemoryListItem(BaseModel):
    """A single memory item in a list response.

    Attributes:
        id: Memory ID.
        memory_type: Type of memory (episodic, structured, semantic, procedural).
        content: The memory content (preview, may be truncated).
        user_id: User ID.
        org_id: Optional org ID.
        session_id: Session ID (for episodic memories).
        confidence: Confidence score (None for episodic).
        created_at: ISO timestamp of creation.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    memory_type: str
    content: str
    user_id: str
    org_id: str | None = None
    session_id: str | None = None
    confidence: float | None = None
    created_at: str


class MemoryListResponse(BaseModel):
    """Response for listing memories with metadata filters.

    Attributes:
        memories: List of memory items.
        total: Total number of matching memories (before pagination).
        limit: Maximum items returned per page.
        offset: Number of items skipped.
        has_more: Whether there are more items available.
    """

    model_config = ConfigDict(extra="forbid")

    memories: list[MemoryListItem] = Field(default_factory=list)
    total: int = Field(ge=0, description="Total matching memories")
    limit: int = Field(ge=1, description="Page size")
    offset: int = Field(ge=0, description="Items skipped")
    has_more: bool = Field(description="Whether more items are available")


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


# ============================================================================
# Batch Encode Schemas
# ============================================================================


class BatchEncodeItem(BaseModel):
    """A single item in a batch encode request.

    Attributes:
        content: The text content to encode.
        role: Role of the speaker (user, assistant, system).
        importance: Importance score (0.0-1.0).
    """

    model_config = ConfigDict(extra="forbid")

    content: str = Field(min_length=1, description="Text content to encode")
    role: Literal["user", "assistant", "system"] = Field(
        default="user", description="Role of the speaker"
    )
    importance: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Importance score (auto-calculated if not provided)",
    )


class BatchEncodeRequest(BaseModel):
    """Request body for batch encoding multiple memories.

    Attributes:
        user_id: User ID for multi-tenancy isolation.
        org_id: Optional organization ID.
        session_id: Optional session ID for grouping.
        items: Array of encode items.
        enrich: LLM enrichment mode for all items.
        continue_on_error: If True, continue processing even if some items fail.
    """

    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(min_length=1, description="User ID for isolation")
    org_id: str | None = Field(default=None, description="Optional org ID")
    session_id: str | None = Field(default=None, description="Optional session ID")
    items: list[BatchEncodeItem] = Field(
        min_length=1,
        description="Array of items to encode",
    )
    enrich: bool | Literal["background"] = Field(
        default=False,
        description="LLM enrichment: False=regex only, True=sync LLM, 'background'=async",
    )
    continue_on_error: bool = Field(
        default=True,
        description="Continue processing even if some items fail",
    )


class BatchEncodeItemResult(BaseModel):
    """Result for a single item in batch encode.

    Attributes:
        index: Index of this item in the original request.
        success: Whether encoding succeeded.
        episode: The stored episode (if successful).
        structured: The structured memory with extractions (if successful).
        extract_count: Total number of extracts (if successful).
        error: Error message (if failed).
    """

    model_config = ConfigDict(extra="forbid")

    index: int = Field(ge=0, description="Index in the original request")
    success: bool = Field(description="Whether encoding succeeded")
    episode: EpisodeResponse | None = Field(default=None, description="Stored episode")
    structured: StructuredResponse | None = Field(default=None, description="Structured memory")
    extract_count: int | None = Field(default=None, description="Number of extracts")
    error: str | None = Field(default=None, description="Error message if failed")


class BatchEncodeResponse(BaseModel):
    """Response body for batch encode operation.

    Attributes:
        total: Total number of items in the request.
        succeeded: Number of items that succeeded.
        failed: Number of items that failed.
        results: Individual results for each item.
    """

    model_config = ConfigDict(extra="forbid")

    total: int = Field(ge=0, description="Total items in request")
    succeeded: int = Field(ge=0, description="Number of successful encodes")
    failed: int = Field(ge=0, description="Number of failed encodes")
    results: list[BatchEncodeItemResult] = Field(description="Individual results")


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
    diversity: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Diversity parameter for MMR reranking (0.0-1.0). Higher values return more diverse results.",
    )
    expand_query: bool = Field(
        default=False,
        description="Expand query with LLM-generated related terms for better recall",
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


class ProvenanceEventResponse(BaseModel):
    """Response model for a single provenance event.

    Attributes:
        timestamp: When the event occurred.
        event_type: Type of derivation event.
        description: Human-readable description.
        memory_id: ID of the memory involved.
        metadata: Additional event data.
    """

    model_config = ConfigDict(extra="forbid")

    timestamp: str = Field(description="When the event occurred (ISO format)")
    event_type: str = Field(description="Type of event")
    description: str = Field(description="Human-readable description")
    memory_id: str | None = Field(default=None, description="ID of memory involved")
    metadata: dict[str, object] = Field(default_factory=dict, description="Additional data")


class IntermediateMemoryResponse(BaseModel):
    """Response model for an intermediate memory in provenance chain.

    Attributes:
        id: Memory ID.
        type: Memory type (structured, semantic).
        summary_or_content: Summary or content preview.
        derivation_method: How this memory was derived.
        derived_at: When derived.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Memory ID")
    type: str = Field(description="Memory type")
    summary_or_content: str = Field(description="Summary or content preview")
    derivation_method: str = Field(description="How this was derived")
    derived_at: str = Field(description="When derived (ISO format)")


class ProvenanceResponse(BaseModel):
    """Response for provenance endpoint.

    Provides complete derivation chain from source episodes through
    intermediate memories to the final derived memory.

    Attributes:
        memory_id: ID of the traced memory.
        memory_type: Type of memory (structured, semantic, procedural).
        derivation_method: How this memory was derived.
        derivation_reasoning: LLM's explanation (if applicable).
        derived_at: When derivation occurred.
        source_episodes: Source episode details.
        intermediate_memories: Intermediate derivations.
        timeline: Chronological derivation events.
    """

    model_config = ConfigDict(extra="forbid")

    memory_id: str = Field(description="ID of the traced memory")
    memory_type: str = Field(description="Type: structured, semantic, procedural")
    derivation_method: str = Field(description="How this memory was derived")
    derivation_reasoning: str | None = Field(
        default=None, description="LLM's explanation (if applicable)"
    )
    derived_at: str | None = Field(default=None, description="When derived (ISO format)")
    source_episodes: list[SourceEpisodeDetail] = Field(
        default_factory=list, description="Source episode details"
    )
    intermediate_memories: list[IntermediateMemoryResponse] = Field(
        default_factory=list, description="Intermediate derivations"
    )
    timeline: list[ProvenanceEventResponse] = Field(
        default_factory=list, description="Chronological derivation events"
    )


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


# ============================================================================
# Workflow Trigger Schemas
# ============================================================================


class WorkflowTriggerRequest(BaseModel):
    """Base request for triggering a workflow.

    Attributes:
        user_id: User ID for multi-tenancy isolation.
        org_id: Organization/project ID for isolation.
        async_execution: If True, run in background and return workflow ID for polling.
    """

    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(min_length=1, description="User ID for isolation")
    org_id: str = Field(
        min_length=1,
        description="Organization/project ID. Required to scope workflows to a single project.",
    )
    async_execution: bool = Field(
        default=False,
        description="If True, run in background and return workflow ID for status polling",
    )


class ConsolidateRequest(WorkflowTriggerRequest):
    """Request for triggering consolidation workflow.

    Attributes:
        consolidation_passes: Number of LLM passes for iterative refinement.
        similarity_threshold: Threshold for semantic similarity matching.
    """

    consolidation_passes: int = Field(
        default=1,
        ge=1,
        le=5,
        description="Number of LLM passes for refinement",
    )
    similarity_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Similarity threshold for memory linking",
    )


class DecayRequest(WorkflowTriggerRequest):
    """Request for triggering decay workflow.

    Attributes:
        run_promotion: Whether to run promotion after decay.
    """

    run_promotion: bool = Field(
        default=True,
        description="Run promotion workflow after decay completes",
    )


class PromoteRequest(WorkflowTriggerRequest):
    """Request for triggering promotion/synthesis workflow."""

    pass


class StructureRequest(BaseModel):
    """Request for triggering structure workflow on a specific episode.

    Attributes:
        episode_id: ID of the episode to structure.
        user_id: User ID for multi-tenancy isolation.
        model: Optional model override for LLM extraction.
        skip_if_structured: Skip if already has StructuredMemory.
    """

    model_config = ConfigDict(extra="forbid")

    episode_id: str = Field(min_length=1, description="Episode ID to structure")
    user_id: str = Field(min_length=1, description="User ID for isolation")
    model: str | None = Field(default=None, description="Optional model override")
    skip_if_structured: bool = Field(
        default=True,
        description="Skip if episode already has StructuredMemory",
    )


class StructureBatchRequest(WorkflowTriggerRequest):
    """Request for batch structure workflow.

    Attributes:
        limit: Maximum episodes to process.
        model: Optional model override for LLM extraction.
    """

    limit: int | None = Field(
        default=None,
        ge=1,
        description="Maximum episodes to process (None = all unstructured)",
    )
    model: str | None = Field(default=None, description="Optional model override")


class WorkflowStatusResponse(BaseModel):
    """Response for workflow status queries.

    Attributes:
        workflow_id: Unique workflow execution ID.
        workflow_type: Type of workflow (consolidate, decay, promote, structure).
        state: Current state (pending, running, completed, failed, cancelled).
        started_at: When the workflow started.
        completed_at: When the workflow completed (if finished).
        error: Error message if failed.
        result: Workflow result data if completed.
    """

    model_config = ConfigDict(extra="forbid")

    workflow_id: str = Field(description="Unique workflow execution ID")
    workflow_type: str = Field(description="Type of workflow")
    state: Literal["pending", "running", "completed", "failed", "cancelled"] = Field(
        description="Current workflow state"
    )
    started_at: str | None = Field(default=None, description="ISO timestamp of start")
    completed_at: str | None = Field(default=None, description="ISO timestamp of completion")
    error: str | None = Field(default=None, description="Error message if failed")
    result: dict[str, Any] | None = Field(default=None, description="Workflow result if completed")


class ConsolidateResponse(BaseModel):
    """Response for consolidation workflow.

    Attributes:
        workflow_id: Workflow execution ID (for async tracking).
        episodes_processed: Number of episodes consolidated.
        semantic_memories_created: Number of semantic memories created.
        links_created: Number of memory links created.
        compression_ratio: Compression ratio achieved.
    """

    model_config = ConfigDict(extra="forbid")

    workflow_id: str | None = Field(default=None, description="Workflow ID for async tracking")
    episodes_processed: int = Field(ge=0, description="Episodes consolidated")
    semantic_memories_created: int = Field(ge=0, description="Semantic memories created")
    links_created: int = Field(ge=0, description="Memory links created")
    compression_ratio: float = Field(ge=0.0, description="Compression ratio")


class DecayResponse(BaseModel):
    """Response for decay workflow.

    Attributes:
        workflow_id: Workflow execution ID (for async tracking).
        memories_updated: Memories with updated confidence.
        memories_archived: Memories moved to archive.
        memories_deleted: Memories permanently deleted.
        procedural_promoted: Memories promoted to procedural.
    """

    model_config = ConfigDict(extra="forbid")

    workflow_id: str | None = Field(default=None, description="Workflow ID for async tracking")
    memories_updated: int = Field(ge=0, description="Memories with updated confidence")
    memories_archived: int = Field(ge=0, description="Memories archived")
    memories_deleted: int = Field(ge=0, description="Memories deleted")
    procedural_promoted: int = Field(ge=0, description="Memories promoted to procedural")


class PromoteResponse(BaseModel):
    """Response for promotion/synthesis workflow.

    Attributes:
        workflow_id: Workflow execution ID (for async tracking).
        semantics_analyzed: Number of semantic memories analyzed.
        procedural_created: Whether a new procedural was created.
        procedural_updated: Whether an existing procedural was updated.
        procedural_id: ID of the created/updated procedural memory.
    """

    model_config = ConfigDict(extra="forbid")

    workflow_id: str | None = Field(default=None, description="Workflow ID for async tracking")
    semantics_analyzed: int = Field(ge=0, description="Semantic memories analyzed")
    procedural_created: bool = Field(description="Whether new procedural was created")
    procedural_updated: bool = Field(description="Whether existing procedural was updated")
    procedural_id: str | None = Field(default=None, description="Procedural memory ID")


class StructureResponse(BaseModel):
    """Response for structure workflow.

    Attributes:
        episode_id: Episode that was structured.
        structured_memory_id: ID of the created StructuredMemory.
        extracts_count: Total entities extracted.
        deterministic_count: Regex extractions (emails, phones, URLs).
        llm_count: LLM extractions.
        processing_time_ms: Processing time in milliseconds.
        skipped: Whether processing was skipped (already structured).
    """

    model_config = ConfigDict(extra="forbid")

    episode_id: str = Field(description="Episode that was structured")
    structured_memory_id: str | None = Field(
        default=None, description="Created StructuredMemory ID"
    )
    extracts_count: int = Field(ge=0, description="Total entities extracted")
    deterministic_count: int = Field(ge=0, description="Regex extractions")
    llm_count: int = Field(ge=0, description="LLM extractions")
    processing_time_ms: int = Field(ge=0, description="Processing time in ms")
    skipped: bool = Field(default=False, description="Whether processing was skipped")


class StructureBatchResponse(BaseModel):
    """Response for batch structure workflow.

    Attributes:
        workflow_id: Workflow execution ID (for async tracking).
        episodes_processed: Number of episodes processed.
        total_extracts: Total entities extracted across all episodes.
        results: Individual results per episode.
    """

    model_config = ConfigDict(extra="forbid")

    workflow_id: str | None = Field(default=None, description="Workflow ID for async tracking")
    episodes_processed: int = Field(ge=0, description="Episodes processed")
    total_extracts: int = Field(ge=0, description="Total entities extracted")
    results: list[StructureResponse] = Field(
        default_factory=list, description="Individual episode results"
    )


# ============================================================================
# Memory Link Schemas
# ============================================================================

# Valid link types
LinkType = Literal["related", "supersedes", "contradicts"]


class CreateLinkRequest(BaseModel):
    """Request to create a link between memories.

    Attributes:
        target_id: ID of the memory to link to.
        link_type: Type of link (related, supersedes, contradicts).
        user_id: User ID for multi-tenancy isolation.
    """

    model_config = ConfigDict(extra="forbid")

    target_id: str = Field(min_length=1, description="ID of the memory to link to")
    link_type: LinkType = Field(
        default="related",
        description="Type of link: related (bidirectional), supersedes (directional), contradicts (bidirectional)",
    )
    user_id: str = Field(min_length=1, description="User ID for isolation")


class LinkDetail(BaseModel):
    """Details about a memory link.

    Attributes:
        target_id: ID of the linked memory.
        link_type: Type of the link.
        created_at: When the link was created (if available).
    """

    model_config = ConfigDict(extra="forbid")

    target_id: str = Field(description="ID of the linked memory")
    link_type: str = Field(description="Type of link")


class CreateLinkResponse(BaseModel):
    """Response for link creation.

    Attributes:
        source_id: ID of the source memory.
        target_id: ID of the target memory.
        link_type: Type of link created.
        bidirectional: Whether a reverse link was also created.
    """

    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(description="ID of the source memory")
    target_id: str = Field(description="ID of the target memory")
    link_type: str = Field(description="Type of link")
    bidirectional: bool = Field(description="Whether reverse link was created")


class LinksListResponse(BaseModel):
    """Response for listing memory links.

    Attributes:
        memory_id: ID of the memory.
        links: List of links from this memory.
        count: Number of links.
    """

    model_config = ConfigDict(extra="forbid")

    memory_id: str = Field(description="ID of the memory")
    links: list[LinkDetail] = Field(default_factory=list, description="Links from this memory")
    count: int = Field(ge=0, description="Number of links")


class DeleteLinkResponse(BaseModel):
    """Response for link deletion.

    Attributes:
        source_id: ID of the source memory.
        target_id: ID of the removed link target.
        removed: Whether the link was removed.
        reverse_removed: Whether the reverse link was also removed.
    """

    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(description="ID of the source memory")
    target_id: str = Field(description="ID of the removed link target")
    removed: bool = Field(description="Whether the link was removed")
    reverse_removed: bool = Field(description="Whether reverse link was also removed")


# ============================================================================
# Memory Update Schemas
# ============================================================================


class UpdateMemoryRequest(BaseModel):
    """Request to update a memory.

    All fields are optional - only provided fields are updated.

    Attributes:
        content: New content (triggers re-embedding).
        confidence: New confidence value (0.0-1.0).
        tags: New tags list (semantic/procedural only).
        keywords: New keywords list (semantic only).
        context: New context description (semantic only).
        user_id: User ID for multi-tenancy isolation.
        org_id: Optional organization ID for further isolation.
    """

    model_config = ConfigDict(extra="forbid")

    content: str | None = Field(default=None, description="New content (triggers re-embedding)")
    confidence: float | None = Field(
        default=None, ge=0.0, le=1.0, description="New confidence value"
    )
    tags: list[str] | None = Field(default=None, description="New tags list")
    keywords: list[str] | None = Field(default=None, description="New keywords list")
    context: str | None = Field(default=None, description="New context description")
    user_id: str = Field(min_length=1, description="User ID for isolation")
    org_id: str | None = Field(default=None, description="Optional organization ID")


class UpdateChange(BaseModel):
    """Record of a single field change.

    Attributes:
        field: Name of the changed field.
        old_value: Previous value (as string for audit).
        new_value: New value (as string for audit).
    """

    model_config = ConfigDict(extra="forbid")

    field: str = Field(description="Name of the changed field")
    old_value: str = Field(description="Previous value")
    new_value: str = Field(description="New value")


class UpdateMemoryResponse(BaseModel):
    """Response for memory update operation.

    Attributes:
        memory_id: ID of the updated memory.
        memory_type: Type of memory (structured, semantic, procedural).
        updated: Whether any changes were made.
        re_embedded: Whether content was re-embedded.
        changes: List of field changes for audit trail.
    """

    model_config = ConfigDict(extra="forbid")

    memory_id: str = Field(description="ID of the updated memory")
    memory_type: str = Field(description="Type of memory")
    updated: bool = Field(description="Whether any changes were made")
    re_embedded: bool = Field(default=False, description="Whether content was re-embedded")
    changes: list[UpdateChange] = Field(default_factory=list, description="List of field changes")


# ============================================================================
# Conflict Detection Schemas
# ============================================================================


class DetectConflictsRequest(BaseModel):
    """Request body for detecting conflicts.

    Attributes:
        user_id: User ID for multi-tenancy.
        org_id: Optional organization ID.
        memory_type: Memory type to check (semantic or structured).
        similarity_threshold: Minimum similarity to consider as potential conflict.
    """

    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(min_length=1, description="User ID for isolation")
    org_id: str | None = Field(default=None, description="Optional org ID")
    memory_type: Literal["semantic", "structured"] = Field(
        default="semantic", description="Memory type to check"
    )
    similarity_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum similarity to consider as potential conflict",
    )


class ConflictResponse(BaseModel):
    """Response for a detected conflict.

    Attributes:
        id: Conflict ID.
        memory_a_id: ID of the first memory.
        memory_a_content: Content of the first memory.
        memory_b_id: ID of the second memory.
        memory_b_content: Content of the second memory.
        conflict_type: Type of conflict (direct, implicit, temporal).
        confidence: Confidence that this is a true conflict.
        explanation: Explanation of the conflict.
        resolution: How the conflict was resolved.
        detected_at: When the conflict was detected.
        resolved_at: When the conflict was resolved.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Conflict ID")
    memory_a_id: str = Field(description="ID of the first memory")
    memory_a_content: str = Field(description="Content of the first memory")
    memory_b_id: str = Field(description="ID of the second memory")
    memory_b_content: str = Field(description="Content of the second memory")
    conflict_type: str = Field(description="Type: direct, implicit, temporal")
    confidence: float = Field(description="Confidence in conflict detection")
    explanation: str = Field(description="Explanation of the conflict")
    resolution: str | None = Field(default=None, description="Resolution if resolved")
    detected_at: datetime = Field(description="When detected")
    resolved_at: datetime | None = Field(default=None, description="When resolved")


class ConflictListResponse(BaseModel):
    """Response for listing conflicts.

    Attributes:
        conflicts: List of detected conflicts.
        count: Total number of conflicts.
    """

    model_config = ConfigDict(extra="forbid")

    conflicts: list[ConflictResponse] = Field(default_factory=list)
    count: int = Field(ge=0, description="Total conflicts")


class DetectConflictsResponse(BaseModel):
    """Response for conflict detection workflow.

    Attributes:
        conflicts_found: Number of new conflicts detected.
        conflicts: List of detected conflicts.
    """

    model_config = ConfigDict(extra="forbid")

    conflicts_found: int = Field(ge=0, description="Number of new conflicts")
    conflicts: list[ConflictResponse] = Field(default_factory=list)


class ResolveConflictRequest(BaseModel):
    """Request to resolve a conflict.

    Attributes:
        resolution: Resolution type.
    """

    model_config = ConfigDict(extra="forbid")

    resolution: Literal["newer_wins", "flag_for_review", "lower_confidence", "create_negation"] = (
        Field(description="Resolution type")
    )


# History schemas
ChangeType = Literal["created", "updated", "strengthened", "weakened", "archived", "deleted"]
TriggerType = Literal[
    "encode", "consolidation", "decay", "promotion", "manual", "retrieval", "system"
]


class HistoryEntryResponse(BaseModel):
    """Response model for a single history entry.

    Attributes:
        id: Unique identifier for this history entry.
        memory_id: ID of the memory that changed.
        memory_type: Type of memory (structured, semantic, procedural).
        timestamp: When the change occurred.
        change_type: Type of change.
        trigger: What caused the change.
        before: Previous state (null for create).
        after: New state (null for delete).
        diff: What specifically changed.
        reason: Human-readable explanation.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="History entry ID")
    memory_id: str = Field(description="ID of the memory that changed")
    memory_type: str = Field(description="Type: structured, semantic, procedural")
    timestamp: datetime = Field(description="When the change occurred")
    change_type: ChangeType = Field(description="Type of change")
    trigger: TriggerType = Field(description="What caused the change")
    before: dict[str, Any] | None = Field(default=None, description="Previous state")
    after: dict[str, Any] | None = Field(default=None, description="New state")
    diff: dict[str, Any] = Field(default_factory=dict, description="What changed")
    reason: str | None = Field(default=None, description="Explanation")


class HistoryListResponse(BaseModel):
    """Response for history queries.

    Attributes:
        entries: List of history entries.
        count: Total number of entries returned.
    """

    model_config = ConfigDict(extra="forbid")

    entries: list[HistoryEntryResponse] = Field(default_factory=list)
    count: int = Field(ge=0, description="Number of entries")


# ============================================================================
# Webhook Schemas
# ============================================================================

EventType = Literal[
    "encode_complete",
    "consolidation_started",
    "consolidation_complete",
    "decay_complete",
    "memory_created",
    "memory_updated",
    "memory_archived",
    "memory_deleted",
]

DeliveryStatus = Literal["pending", "success", "failed", "retrying"]


class CreateWebhookRequest(BaseModel):
    """Request to register a new webhook.

    Attributes:
        url: HTTPS endpoint to receive webhook events.
        secret: Shared secret for HMAC-SHA256 signature verification.
        events: List of event types to subscribe to. Defaults to all events.
        description: Optional human-readable description.
        user_id: User ID for multi-tenancy isolation.
        org_id: Optional organization ID.
    """

    model_config = ConfigDict(extra="forbid")

    url: str = Field(min_length=1, description="HTTPS endpoint URL")
    secret: str = Field(min_length=16, description="Shared secret (min 16 chars)")
    events: list[EventType] | None = Field(
        default=None,
        description="Event types to subscribe to (default: all)",
    )
    description: str | None = Field(default=None, description="Human-readable description")
    user_id: str = Field(min_length=1, description="User ID for isolation")
    org_id: str | None = Field(default=None, description="Optional org ID")


class UpdateWebhookRequest(BaseModel):
    """Request to update a webhook.

    All fields are optional - only provided fields are updated.

    Attributes:
        url: New HTTPS endpoint URL.
        secret: New shared secret.
        events: New list of event types to subscribe to.
        enabled: Whether webhook is active.
        description: New description.
        user_id: User ID for multi-tenancy isolation.
        org_id: Optional organization ID.
    """

    model_config = ConfigDict(extra="forbid")

    url: str | None = Field(default=None, description="New HTTPS endpoint URL")
    secret: str | None = Field(default=None, min_length=16, description="New shared secret")
    events: list[EventType] | None = Field(default=None, description="New event types")
    enabled: bool | None = Field(default=None, description="Enable/disable webhook")
    description: str | None = Field(default=None, description="New description")
    user_id: str = Field(min_length=1, description="User ID for isolation")
    org_id: str | None = Field(default=None, description="Optional org ID")


class WebhookResponse(BaseModel):
    """Response model for a webhook configuration.

    Attributes:
        id: Unique webhook ID.
        url: HTTPS endpoint URL.
        events: List of subscribed event types.
        enabled: Whether webhook is active.
        description: Human-readable description.
        created_at: When the webhook was registered.
        updated_at: When the webhook was last modified.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Webhook ID")
    url: str = Field(description="HTTPS endpoint URL")
    events: list[str] = Field(description="Subscribed event types")
    enabled: bool = Field(description="Whether webhook is active")
    description: str | None = Field(default=None, description="Description")
    created_at: datetime = Field(description="When registered")
    updated_at: datetime = Field(description="When last modified")


class WebhookListResponse(BaseModel):
    """Response for listing webhooks.

    Attributes:
        webhooks: List of webhook configurations.
        count: Total number of webhooks.
    """

    model_config = ConfigDict(extra="forbid")

    webhooks: list[WebhookResponse] = Field(default_factory=list)
    count: int = Field(ge=0, description="Number of webhooks")


class WebhookDeliveryResponse(BaseModel):
    """Response model for a webhook delivery attempt.

    Attributes:
        id: Unique delivery ID.
        webhook_id: ID of the webhook configuration.
        event_id: ID of the event delivered.
        status: Delivery status.
        attempt: Attempt number.
        created_at: When delivery started.
        completed_at: When delivery finished.
        response_code: HTTP response status code.
        error: Error message if failed.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Delivery ID")
    webhook_id: str = Field(description="Webhook configuration ID")
    event_id: str = Field(description="Event ID")
    status: DeliveryStatus = Field(description="Delivery status")
    attempt: int = Field(ge=1, description="Attempt number")
    created_at: datetime = Field(description="When delivery started")
    completed_at: datetime | None = Field(default=None, description="When delivery finished")
    response_code: int | None = Field(default=None, description="HTTP response code")
    error: str | None = Field(default=None, description="Error message")


class WebhookDeliveryListResponse(BaseModel):
    """Response for listing webhook delivery logs.

    Attributes:
        deliveries: List of delivery records.
        count: Total number of deliveries.
    """

    model_config = ConfigDict(extra="forbid")

    deliveries: list[WebhookDeliveryResponse] = Field(default_factory=list)
    count: int = Field(ge=0, description="Number of deliveries")


# ============================================================================
# Session Management Schemas
# ============================================================================


class SessionSummary(BaseModel):
    """Summary of a session for listing.

    Attributes:
        session_id: Session identifier.
        episode_count: Number of episodes in this session.
        first_episode_at: Timestamp of the first episode.
        last_episode_at: Timestamp of the most recent episode.
    """

    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(description="Session identifier")
    episode_count: int = Field(ge=0, description="Number of episodes in session")
    first_episode_at: str = Field(description="Timestamp of first episode (ISO format)")
    last_episode_at: str = Field(description="Timestamp of last episode (ISO format)")


class SessionListResponse(BaseModel):
    """Response for listing sessions.

    Attributes:
        sessions: List of session summaries.
        count: Total number of sessions.
    """

    model_config = ConfigDict(extra="forbid")

    sessions: list[SessionSummary] = Field(default_factory=list)
    count: int = Field(ge=0, description="Total number of sessions")


class SessionDetailResponse(BaseModel):
    """Response for session details.

    Attributes:
        session_id: Session identifier.
        user_id: User ID.
        org_id: Optional org ID.
        episodes: List of episodes in this session.
        episode_count: Number of episodes.
        first_episode_at: Timestamp of the first episode.
        last_episode_at: Timestamp of the most recent episode.
    """

    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(description="Session identifier")
    user_id: str = Field(description="User ID")
    org_id: str | None = Field(default=None, description="Org ID")
    episodes: list[EpisodeResponse] = Field(default_factory=list)
    episode_count: int = Field(ge=0, description="Number of episodes")
    first_episode_at: str | None = Field(
        default=None, description="Timestamp of first episode (ISO format)"
    )
    last_episode_at: str | None = Field(
        default=None, description="Timestamp of last episode (ISO format)"
    )


class SessionDeleteResponse(BaseModel):
    """Response for session deletion.

    Attributes:
        session_id: Session that was deleted.
        deleted: Whether deletion succeeded.
        episodes_deleted: Number of episodes deleted (if cascade).
        structured_deleted: Number of structured memories deleted (if cascade).
        semantic_deleted: Number of semantic memories deleted (if cascade).
        semantic_updated: Number of semantic memories updated (if soft cascade).
        procedural_deleted: Number of procedural memories deleted (if cascade).
        procedural_updated: Number of procedural memories updated (if soft cascade).
    """

    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(description="Session that was deleted")
    deleted: bool = Field(description="Whether deletion succeeded")
    episodes_deleted: int = Field(ge=0, description="Episodes deleted")
    structured_deleted: int = Field(ge=0, default=0, description="Structured memories deleted")
    semantic_deleted: int = Field(ge=0, default=0, description="Semantic memories deleted")
    semantic_updated: int = Field(ge=0, default=0, description="Semantic memories updated")
    procedural_deleted: int = Field(ge=0, default=0, description="Procedural memories deleted")
    procedural_updated: int = Field(ge=0, default=0, description="Procedural memories updated")
