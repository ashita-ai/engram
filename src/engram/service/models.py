"""Service layer models for Engram.

Contains Pydantic models for encode/recall operations:
- EncodeResult: Result of encoding a memory
- RecallResult: A single recalled memory with score
- VerificationResult: Result of verifying a memory
- SourceEpisodeSummary: Lightweight episode summary
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from engram.models import Episode, Staleness, StructuredMemory


class EncodeResult(BaseModel):
    """Result of encoding a memory.

    Attributes:
        episode: The stored episode (immutable ground truth).
        structured: The structured memory (per-episode intelligence).
    """

    model_config = ConfigDict(extra="forbid")

    episode: Episode
    structured: StructuredMemory


class SourceEpisodeSummary(BaseModel):
    """Lightweight summary of a source episode."""

    model_config = ConfigDict(extra="forbid")

    id: str
    content: str
    role: str
    timestamp: str


class RecallResult(BaseModel):
    """A single recalled memory with similarity score.

    Attributes:
        memory_type: Type of memory (episodic, structured, semantic, procedural).
        content: The memory content.
        score: Similarity score (0.0-1.0).
        confidence: Confidence score for structured/semantic memories.
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
    score: float = Field(ge=0.0, le=1.0)
    confidence: float | None = None
    memory_id: str
    source_episode_id: str | None = None
    source_episode_ids: list[str] = Field(default_factory=list)
    source_episodes: list[SourceEpisodeSummary] = Field(default_factory=list)
    related_ids: list[str] = Field(default_factory=list)
    hop_distance: int = Field(default=0, ge=0)
    staleness: Staleness = Field(default=Staleness.FRESH)
    consolidated_at: str | None = Field(default=None)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("score", mode="before")
    @classmethod
    def clamp_score(cls, v: float) -> float:
        """Clamp score to [0, 1] to handle floating point precision errors."""
        return max(0.0, min(1.0, v))


class VerificationResult(BaseModel):
    """Result of verifying a memory against its sources.

    Provides full traceability from a derived memory back to
    the source episode(s) it was extracted from.

    Attributes:
        memory_id: ID of the verified memory.
        memory_type: Type of memory (structured, semantic, procedural).
        content: The memory content.
        verified: Whether sources were found and content matches.
        source_episodes: Source episode contents.
        extraction_method: How the memory was extracted.
        confidence: Current confidence score.
        explanation: Human-readable derivation trace.
    """

    model_config = ConfigDict(extra="forbid")

    memory_id: str = Field(description="ID of the verified memory")
    memory_type: str = Field(description="Type: structured, semantic, procedural")
    content: str = Field(description="The memory content")
    verified: bool = Field(description="True if sources found and traceable")
    source_episodes: list[dict[str, Any]] = Field(
        default_factory=list, description="Source episode details"
    )
    extraction_method: str = Field(description="How memory was extracted")
    confidence: float = Field(ge=0.0, le=1.0, description="Current confidence score")
    explanation: str = Field(description="Human-readable derivation trace")


__all__ = [
    "EncodeResult",
    "RecallResult",
    "SourceEpisodeSummary",
    "VerificationResult",
]
