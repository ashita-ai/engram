"""Proactive contradiction detection between memories.

Detects conflicts between memories using semantic similarity and LLM analysis.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import Agent

from engram.models.base import OperationStatus

if TYPE_CHECKING:
    from engram.embeddings import Embedder
    from engram.models import SemanticMemory, StructuredMemory

logger = logging.getLogger(__name__)


class ConflictDetection(BaseModel):
    """A detected conflict between two memories.

    Attributes:
        id: Unique identifier for this conflict.
        memory_a_id: ID of the first memory.
        memory_a_content: Content of the first memory.
        memory_b_id: ID of the second memory.
        memory_b_content: Content of the second memory.
        conflict_type: Type of conflict (direct, implicit, temporal).
        confidence: Confidence that this is a true conflict (0.0-1.0).
        explanation: LLM explanation of why this is a conflict.
        resolution: How the conflict was resolved (if resolved).
        detected_at: When the conflict was detected.
        resolved_at: When the conflict was resolved (if resolved).
        user_id: User ID for multi-tenancy.
        org_id: Optional organization ID.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(
        default_factory=lambda: f"conflict_{datetime.now(UTC).strftime('%Y%m%d%H%M%S%f')}"
    )
    memory_a_id: str = Field(description="ID of the first memory")
    memory_a_content: str = Field(description="Content of the first memory")
    memory_b_id: str = Field(description="ID of the second memory")
    memory_b_content: str = Field(description="Content of the second memory")
    conflict_type: str = Field(description="Type of conflict: direct, implicit, or temporal")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence that this is a true conflict")
    explanation: str = Field(description="Explanation of the conflict")
    resolution: str | None = Field(
        default=None,
        description="How the conflict was resolved",
    )
    detected_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    resolved_at: datetime | None = Field(default=None)
    user_id: str = Field(description="User ID for multi-tenancy")
    org_id: str | None = Field(default=None, description="Optional organization ID")


class ConflictAnalysis(BaseModel):
    """LLM output for conflict analysis between two memories.

    IMPORTANT: Always check `status` before trusting results. When `status`
    is FAILED, the `is_conflict=False` result is a fallback, not a finding.
    """

    model_config = ConfigDict(extra="forbid")

    status: OperationStatus = Field(
        default=OperationStatus.SUCCESS,
        description="Operation status - check this before trusting results",
    )
    is_conflict: bool = Field(description="Whether these memories conflict")
    conflict_type: str = Field(
        default="none",
        description="Type: direct (explicit contradiction), implicit (logical inconsistency), temporal (time-based), none",
    )
    confidence: float = Field(
        ge=0.0, le=1.0, default=0.0, description="Confidence in conflict detection"
    )
    explanation: str = Field(default="", description="Explanation of why they conflict or don't")
    suggested_resolution: str = Field(
        default="",
        description="Suggested resolution: newer_wins, flag_for_review, lower_confidence, create_negation",
    )
    error_message: str | None = Field(
        default=None,
        description="Error details when status is FAILED",
    )


class BatchConflictAnalysis(BaseModel):
    """LLM output for batch conflict analysis."""

    model_config = ConfigDict(extra="forbid")

    conflicts: list[ConflictAnalysis] = Field(
        default_factory=list, description="List of detected conflicts"
    )
    pair_indices: list[tuple[int, int]] = Field(
        default_factory=list, description="Indices of conflicting memory pairs"
    )


_conflict_agent: Agent[None, ConflictAnalysis] | None = None


def get_conflict_agent(model: str = "openai:gpt-4o-mini") -> Agent[None, ConflictAnalysis]:
    """Get or create the conflict detection agent.

    Args:
        model: LLM model to use for conflict detection.

    Returns:
        The conflict detection agent.
    """
    global _conflict_agent
    if _conflict_agent is None:
        _conflict_agent = Agent(
            model,
            output_type=ConflictAnalysis,
            system_prompt="""You are a conflict detection assistant for a memory system.

Given two memories, determine if they contradict each other.

Types of conflicts:
- DIRECT: Explicit contradiction (e.g., "likes coffee" vs "hates coffee")
- IMPLICIT: Logical inconsistency (e.g., "vegetarian" vs "favorite steak restaurant")
- TEMPORAL: Time-based conflict (e.g., "works at Company A" vs "quit Company A last month")

Guidelines:
- Focus on factual contradictions, not differences in detail
- Consider context: same topic but different aspects is not a conflict
- High confidence (>0.8) only for clear, explicit contradictions
- Medium confidence (0.5-0.8) for implicit or uncertain conflicts
- Low confidence (<0.5) for potential but unclear conflicts

For resolution suggestions:
- newer_wins: If one memory is clearly more recent and supersedes the other
- flag_for_review: If resolution requires human judgment
- lower_confidence: If both might be valid in different contexts
- create_negation: If one memory explicitly negates the other""",
        )
    return _conflict_agent


async def analyze_pair(
    memory_a_content: str,
    memory_b_content: str,
    model: str = "openai:gpt-4o-mini",
) -> ConflictAnalysis:
    """Analyze a pair of memories for conflicts.

    Args:
        memory_a_content: Content of the first memory.
        memory_b_content: Content of the second memory.
        model: LLM model to use.

    Returns:
        ConflictAnalysis with conflict detection results.
    """
    agent = get_conflict_agent(model)
    try:
        from engram.workflows.llm_utils import run_agent_with_retry

        prompt = f"""Analyze these two memories for conflicts:

MEMORY A:
{memory_a_content}

MEMORY B:
{memory_b_content}

Determine if they contradict each other and explain why."""

        return await run_agent_with_retry(agent, prompt)
    except Exception as e:
        logger.warning(f"Conflict analysis failed: {e}")
        return ConflictAnalysis(
            status=OperationStatus.FAILED,
            is_conflict=False,
            conflict_type="none",
            confidence=0.0,
            explanation="Analysis failed due to LLM error",
            error_message=str(e),
        )


async def detect_contradictions(
    new_memories: list[SemanticMemory],
    existing_memories: list[SemanticMemory],
    embedder: Embedder,
    user_id: str,
    org_id: str | None = None,
    similarity_threshold: float = 0.5,
    model: str = "openai:gpt-4o-mini",
) -> list[ConflictDetection]:
    """Detect contradictions between new and existing memories.

    Uses a two-stage approach:
    1. Find semantically similar pairs (potential conflicts)
    2. Use LLM to analyze if they actually conflict

    Args:
        new_memories: Newly created memories to check.
        existing_memories: Existing memories to check against.
        embedder: Embedder for computing similarity.
        user_id: User ID for multi-tenancy.
        org_id: Optional organization ID.
        similarity_threshold: Minimum similarity to consider as potential conflict.
        model: LLM model for conflict analysis.

    Returns:
        List of detected conflicts.
    """
    from engram.service.helpers import cosine_similarity

    if not new_memories or not existing_memories:
        return []

    conflicts: list[ConflictDetection] = []

    for new_mem in new_memories:
        new_embedding = new_mem.embedding
        if not new_embedding:
            new_embedding = await embedder.embed(new_mem.content)

        for existing_mem in existing_memories:
            # Skip self-comparison
            if new_mem.id == existing_mem.id:
                continue

            existing_embedding = existing_mem.embedding
            if not existing_embedding:
                existing_embedding = await embedder.embed(existing_mem.content)

            # Calculate semantic similarity
            similarity = cosine_similarity(new_embedding, existing_embedding)

            # Only check pairs that are semantically similar (potential conflicts)
            if similarity >= similarity_threshold:
                logger.debug(
                    f"Checking potential conflict: {new_mem.id} vs {existing_mem.id} "
                    f"(similarity: {similarity:.2f})"
                )

                # Use LLM to analyze for actual conflict
                analysis = await analyze_pair(
                    new_mem.content,
                    existing_mem.content,
                    model=model,
                )

                # Skip failed analyses - don't treat LLM errors as "no conflict"
                if analysis.status == OperationStatus.FAILED:
                    logger.warning(
                        f"Skipping conflict check for {new_mem.id} vs {existing_mem.id}: "
                        f"LLM analysis failed - {analysis.error_message}"
                    )
                    continue

                if analysis.is_conflict and analysis.confidence >= 0.5:
                    conflict = ConflictDetection(
                        memory_a_id=new_mem.id,
                        memory_a_content=new_mem.content,
                        memory_b_id=existing_mem.id,
                        memory_b_content=existing_mem.content,
                        conflict_type=analysis.conflict_type,
                        confidence=analysis.confidence,
                        explanation=analysis.explanation,
                        user_id=user_id,
                        org_id=org_id,
                    )
                    conflicts.append(conflict)
                    logger.info(
                        f"Detected conflict between {new_mem.id} and {existing_mem.id}: "
                        f"{analysis.conflict_type} ({analysis.confidence:.2f})"
                    )

    return conflicts


async def detect_contradictions_in_structured(
    new_memories: list[StructuredMemory],
    existing_memories: list[StructuredMemory],
    embedder: Embedder,
    user_id: str,
    org_id: str | None = None,
    similarity_threshold: float = 0.5,
    model: str = "openai:gpt-4o-mini",
) -> list[ConflictDetection]:
    """Detect contradictions between StructuredMemory instances.

    Uses summary content for comparison.

    Args:
        new_memories: Newly created StructuredMemories.
        existing_memories: Existing StructuredMemories.
        embedder: Embedder for computing similarity.
        user_id: User ID for multi-tenancy.
        org_id: Optional organization ID.
        similarity_threshold: Minimum similarity to consider.
        model: LLM model for analysis.

    Returns:
        List of detected conflicts.
    """
    from engram.service.helpers import cosine_similarity

    if not new_memories or not existing_memories:
        return []

    conflicts: list[ConflictDetection] = []

    for new_mem in new_memories:
        new_embedding = new_mem.embedding
        if not new_embedding:
            new_embedding = await embedder.embed(new_mem.summary)

        for existing_mem in existing_memories:
            if new_mem.id == existing_mem.id:
                continue

            existing_embedding = existing_mem.embedding
            if not existing_embedding:
                existing_embedding = await embedder.embed(existing_mem.summary)

            similarity = cosine_similarity(new_embedding, existing_embedding)

            if similarity >= similarity_threshold:
                analysis = await analyze_pair(
                    new_mem.summary,
                    existing_mem.summary,
                    model=model,
                )

                # Skip failed analyses - don't treat LLM errors as "no conflict"
                if analysis.status == OperationStatus.FAILED:
                    logger.warning(
                        f"Skipping conflict check for {new_mem.id} vs {existing_mem.id}: "
                        f"LLM analysis failed - {analysis.error_message}"
                    )
                    continue

                if analysis.is_conflict and analysis.confidence >= 0.5:
                    conflict = ConflictDetection(
                        memory_a_id=new_mem.id,
                        memory_a_content=new_mem.summary,
                        memory_b_id=existing_mem.id,
                        memory_b_content=existing_mem.summary,
                        conflict_type=analysis.conflict_type,
                        confidence=analysis.confidence,
                        explanation=analysis.explanation,
                        user_id=user_id,
                        org_id=org_id,
                    )
                    conflicts.append(conflict)

    return conflicts


__all__ = [
    "BatchConflictAnalysis",
    "ConflictAnalysis",
    "ConflictDetection",
    "analyze_pair",
    "detect_contradictions",
    "detect_contradictions_in_structured",
    "get_conflict_agent",
]
