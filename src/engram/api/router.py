"""FastAPI router for Engram API endpoints."""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from engram.service import EngramService

from .schemas import (
    ConfidenceStats,
    EncodeRequest,
    EncodeResponse,
    EpisodeResponse,
    FactResponse,
    HealthResponse,
    MemoryCounts,
    MemoryStatsResponse,
    RecallRequest,
    RecallResponse,
    RecallResultResponse,
    SourceEpisodeDetail,
    SourceEpisodeSummary,
    SourcesResponse,
    VerificationResponse,
    WorkingMemoryResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Service instance (set by app lifespan)
_service: EngramService | None = None


def set_service(service: EngramService) -> None:
    """Set the global service instance."""
    global _service
    _service = service


async def get_service() -> EngramService:
    """Dependency to get the EngramService instance."""
    if _service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not initialized",
        )
    return _service


ServiceDep = Annotated[EngramService, Depends(get_service)]


@router.get("/health", response_model=HealthResponse, tags=["system"])
async def health_check() -> HealthResponse:
    """Check service health.

    Returns the current health status of the Engram service,
    including storage connectivity.
    """
    storage_connected = _service is not None

    if storage_connected:
        return HealthResponse(
            status="healthy",
            version="0.1.0",
            storage_connected=True,
        )
    else:
        return HealthResponse(
            status="unhealthy",
            version="0.1.0",
            storage_connected=False,
        )


@router.post(
    "/encode",
    response_model=EncodeResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["memory"],
)
async def encode(
    request: EncodeRequest,
    service: ServiceDep,
) -> EncodeResponse:
    """Encode content as an episode and extract facts.

    This endpoint stores the provided content as an episode in the
    memory system and optionally runs fact extraction to identify
    structured data like emails, phone numbers, dates, etc.

    Args:
        request: Encode request with content and options.
        service: Injected EngramService.

    Returns:
        The stored episode and extracted facts.

    Raises:
        HTTPException: If encoding fails.
    """
    try:
        result = await service.encode(
            content=request.content,
            role=request.role,
            user_id=request.user_id,
            org_id=request.org_id,
            session_id=request.session_id,
            importance=request.importance,
            run_extraction=request.run_extraction,
        )

        # Convert to response models
        episode_response = EpisodeResponse(
            id=result.episode.id,
            content=result.episode.content,
            role=result.episode.role,
            user_id=result.episode.user_id,
            org_id=result.episode.org_id,
            session_id=result.episode.session_id,
            importance=result.episode.importance,
            created_at=result.episode.timestamp.isoformat(),
        )

        fact_responses = [
            FactResponse(
                id=fact.id,
                content=fact.content,
                category=fact.category,
                confidence=fact.confidence.value,
                source_episode_id=fact.source_episode_id,
            )
            for fact in result.facts
        ]

        return EncodeResponse(
            episode=episode_response,
            facts=fact_responses,
            fact_count=len(fact_responses),
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    except Exception as e:
        logger.exception("Failed to encode memory")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred while encoding the memory",
        ) from e


@router.post("/recall", response_model=RecallResponse, tags=["memory"])
async def recall(
    request: RecallRequest,
    service: ServiceDep,
) -> RecallResponse:
    """Recall memories by semantic similarity.

    This endpoint searches the memory system for content similar
    to the provided query. It can search across episodes and facts,
    returning unified results sorted by similarity score.

    Args:
        request: Recall request with query and options.
        service: Injected EngramService.

    Returns:
        List of recalled memories with similarity scores.

    Raises:
        HTTPException: If recall fails.
    """
    try:
        # Use recall_at for bi-temporal queries, recall for standard queries
        if request.as_of is not None:
            results = await service.recall_at(
                query=request.query,
                as_of=request.as_of,
                user_id=request.user_id,
                org_id=request.org_id,
                limit=request.limit,
                min_confidence=request.min_confidence,
                include_episodes=request.include_episodes,
                include_facts=request.include_facts,
            )
        else:
            results = await service.recall(
                query=request.query,
                user_id=request.user_id,
                org_id=request.org_id,
                limit=request.limit,
                min_confidence=request.min_confidence,
                min_selectivity=request.min_selectivity,
                include_episodes=request.include_episodes,
                include_facts=request.include_facts,
                include_semantic=request.include_semantic,
                include_working=request.include_working,
                include_sources=request.include_sources,
                follow_links=request.follow_links,
                max_hops=request.max_hops,
                freshness=request.freshness,
            )

        result_responses = [
            RecallResultResponse(
                memory_type=r.memory_type,
                content=r.content,
                score=r.score,
                confidence=r.confidence,
                memory_id=r.memory_id,
                source_episode_id=r.source_episode_id,
                source_episodes=[
                    SourceEpisodeSummary(
                        id=s.id,
                        content=s.content,
                        role=s.role,
                        timestamp=s.timestamp,
                    )
                    for s in r.source_episodes
                ],
                related_ids=r.related_ids,
                hop_distance=r.hop_distance,
                staleness=r.staleness,
                consolidated_at=r.consolidated_at,
                metadata=r.metadata,
            )
            for r in results
        ]

        return RecallResponse(
            query=request.query,
            results=result_responses,
            count=len(result_responses),
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    except Exception as e:
        logger.exception("Failed to recall memories")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred while recalling memories",
        ) from e


@router.get("/memories/stats", response_model=MemoryStatsResponse, tags=["memory"])
async def get_memory_stats(
    user_id: str,
    service: ServiceDep,
    org_id: str | None = None,
) -> MemoryStatsResponse:
    """Get statistics about stored memories.

    Returns counts of memories by type, confidence statistics,
    and the number of episodes pending consolidation.

    Args:
        user_id: User ID to get stats for.
        service: Injected EngramService.
        org_id: Optional organization ID filter.

    Returns:
        Memory statistics including counts and confidence.
    """
    try:
        stats = await service.storage.get_memory_stats(
            user_id=user_id,
            org_id=org_id,
        )

        return MemoryStatsResponse(
            user_id=user_id,
            org_id=org_id,
            counts=MemoryCounts(
                episodes=stats.episodes,
                facts=stats.facts,
                semantic=stats.semantic,
                procedural=stats.procedural,
                inhibitory=stats.inhibitory,
            ),
            confidence=ConfidenceStats(
                facts_avg=stats.facts_avg_confidence,
                facts_min=stats.facts_min_confidence,
                facts_max=stats.facts_max_confidence,
                semantic_avg=stats.semantic_avg_confidence,
            ),
            pending_consolidation=stats.pending_consolidation,
        )

    except Exception as e:
        logger.exception("Failed to get memory stats")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred while retrieving memory statistics",
        ) from e


@router.get("/memories/working", response_model=WorkingMemoryResponse, tags=["memory"])
async def get_working_memory(
    service: ServiceDep,
) -> WorkingMemoryResponse:
    """Get current session's working memory.

    Working memory contains episodes from the current session.
    It's volatile (in-memory only) and cleared when the session ends.

    Returns:
        List of episodes in working memory.
    """
    episodes = service.get_working_memory()

    episode_responses = [
        EpisodeResponse(
            id=ep.id,
            content=ep.content,
            role=ep.role,
            user_id=ep.user_id,
            org_id=ep.org_id,
            session_id=ep.session_id,
            importance=ep.importance,
            created_at=ep.timestamp.isoformat(),
        )
        for ep in episodes
    ]

    return WorkingMemoryResponse(
        episodes=episode_responses,
        count=len(episode_responses),
    )


@router.delete(
    "/memories/working",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["memory"],
)
async def clear_working_memory(
    service: ServiceDep,
) -> None:
    """Clear working memory.

    Removes all episodes from the current session's working memory.
    This is typically called at the end of a session.
    """
    service.clear_working_memory()


@router.get("/memories/{memory_id}/sources", response_model=SourcesResponse, tags=["memory"])
async def get_sources(
    memory_id: str,
    user_id: str,
    service: ServiceDep,
) -> SourcesResponse:
    """Get source episodes for a derived memory.

    Traces a derived memory (fact, semantic, procedural, or inhibitory)
    back to the source episode(s) it was extracted from. This enables
    provenance tracking and allows users to verify the origin of any
    derived knowledge.

    Args:
        memory_id: ID of the derived memory (must start with fact_, sem_, proc_, or inh_).
        user_id: User ID for multi-tenancy isolation.
        service: Injected EngramService.

    Returns:
        Source episodes in chronological order.

    Raises:
        HTTPException: 400 if memory_id format is invalid.
        HTTPException: 404 if memory not found.
    """
    # Determine memory type from ID prefix
    if memory_id.startswith("fact_"):
        memory_type = "fact"
    elif memory_id.startswith("sem_"):
        memory_type = "semantic"
    elif memory_id.startswith("proc_"):
        memory_type = "procedural"
    elif memory_id.startswith("inh_"):
        memory_type = "inhibitory"
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid memory ID format: {memory_id}. "
            "Expected prefix: fact_, sem_, proc_, or inh_",
        )

    try:
        episodes = await service.get_sources(memory_id, user_id)

        episode_responses = [
            EpisodeResponse(
                id=ep.id,
                content=ep.content,
                role=ep.role,
                user_id=ep.user_id,
                org_id=ep.org_id,
                session_id=ep.session_id,
                importance=ep.importance,
                created_at=ep.timestamp.isoformat(),
            )
            for ep in episodes
        ]

        return SourcesResponse(
            memory_id=memory_id,
            memory_type=memory_type,
            sources=episode_responses,
            count=len(episode_responses),
        )

    except KeyError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        ) from e
    except Exception as e:
        logger.exception("Failed to get sources for memory %s", memory_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred while retrieving memory sources",
        ) from e


@router.get("/memories/{memory_id}/verify", response_model=VerificationResponse, tags=["memory"])
async def verify_memory(
    memory_id: str,
    user_id: str,
    service: ServiceDep,
) -> VerificationResponse:
    """Verify a memory against its source episodes.

    Traces a derived memory (fact, semantic, procedural, or inhibitory)
    back to its source episode(s) and provides a human-readable explanation
    of how it was derived. This is core to Engram's "memory you can trust"
    value proposition.

    Args:
        memory_id: ID of the derived memory (must start with fact_, sem_, proc_, or inh_).
        user_id: User ID for multi-tenancy isolation.
        service: Injected EngramService.

    Returns:
        Verification result with source traceability and explanation.

    Raises:
        HTTPException: 400 if memory_id format is invalid.
        HTTPException: 404 if memory not found.

    Example:
        GET /memories/fact_abc123/verify?user_id=u1

        Response:
        {
            "memory_id": "fact_abc123",
            "memory_type": "fact",
            "content": "email=john@example.com",
            "verified": true,
            "source_episodes": [
                {"id": "ep_xyz", "content": "My email is john@example.com", ...}
            ],
            "extraction_method": "extracted",
            "confidence": 0.9,
            "explanation": "Pattern-matched email from source episode(s). ..."
        }
    """
    # Validate memory ID format
    valid_prefixes = ("fact_", "sem_", "proc_", "inh_")
    if not memory_id.startswith(valid_prefixes):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid memory ID format: {memory_id}. "
            "Expected prefix: fact_, sem_, proc_, or inh_",
        )

    try:
        result = await service.verify(memory_id, user_id)

        # Convert source episodes to response format
        source_episode_responses = [
            SourceEpisodeDetail(
                id=ep["id"],
                content=ep["content"],
                role=ep["role"],
                timestamp=ep["timestamp"],
            )
            for ep in result.source_episodes
        ]

        return VerificationResponse(
            memory_id=result.memory_id,
            memory_type=result.memory_type,
            content=result.content,
            verified=result.verified,
            source_episodes=source_episode_responses,
            extraction_method=result.extraction_method,
            confidence=result.confidence,
            explanation=result.explanation,
        )

    except KeyError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        ) from e
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    except Exception as e:
        logger.exception("Failed to verify memory %s", memory_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred while verifying the memory",
        ) from e
