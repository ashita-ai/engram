"""Pydantic schemas for API request/response models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class EncodeRequest(BaseModel):
    """Request body for encoding a memory.

    Attributes:
        content: The text content to encode.
        role: Role of the speaker (user, assistant, system).
        user_id: User ID for multi-tenancy isolation.
        org_id: Optional organization ID.
        session_id: Optional session ID for grouping.
        importance: Importance score (0.0-1.0).
        run_extraction: Whether to run fact extraction.
    """

    model_config = ConfigDict(extra="forbid")

    content: str = Field(min_length=1, description="Text content to encode")
    role: Literal["user", "assistant", "system"] = Field(
        default="user", description="Role of the speaker"
    )
    user_id: str = Field(min_length=1, description="User ID for isolation")
    org_id: str | None = Field(default=None, description="Optional org ID")
    session_id: str | None = Field(default=None, description="Optional session ID")
    importance: float = Field(default=0.5, ge=0.0, le=1.0, description="Importance score")
    run_extraction: bool = Field(default=True, description="Whether to run fact extraction")


class FactResponse(BaseModel):
    """Response model for an extracted fact.

    Attributes:
        id: Unique fact ID.
        content: The extracted fact content.
        category: Fact category (email, phone, date, etc.).
        confidence: Confidence score.
        source_episode_id: ID of the source episode.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    content: str
    category: str
    confidence: float
    source_episode_id: str


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
        facts: List of extracted facts.
        fact_count: Number of facts extracted.
    """

    model_config = ConfigDict(extra="forbid")

    episode: EpisodeResponse
    facts: list[FactResponse] = Field(default_factory=list)
    fact_count: int = Field(description="Number of facts extracted")


class RecallRequest(BaseModel):
    """Request body for recalling memories.

    Attributes:
        query: Natural language query.
        user_id: User ID for multi-tenancy isolation.
        org_id: Optional organization ID filter.
        limit: Maximum results to return.
        min_confidence: Minimum confidence for facts.
        include_episodes: Whether to search episodes.
        include_facts: Whether to search facts.
    """

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, description="Natural language query")
    user_id: str = Field(min_length=1, description="User ID for isolation")
    org_id: str | None = Field(default=None, description="Optional org ID filter")
    limit: int = Field(default=10, ge=1, le=100, description="Maximum results")
    min_confidence: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Minimum confidence threshold"
    )
    include_episodes: bool = Field(default=True, description="Search episodes")
    include_facts: bool = Field(default=True, description="Search facts")


class RecallResultResponse(BaseModel):
    """Response model for a single recalled memory.

    Attributes:
        memory_type: Type of memory (episode, fact, semantic, etc.).
        content: The memory content.
        score: Similarity score (0.0-1.0).
        confidence: Confidence score for facts/semantic memories.
        memory_id: Unique memory ID.
        source_episode_id: Source episode ID for facts.
        metadata: Additional memory-specific metadata.
    """

    model_config = ConfigDict(extra="forbid")

    memory_type: str
    content: str
    score: float
    confidence: float | None = None
    memory_id: str
    source_episode_id: str | None = None
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
