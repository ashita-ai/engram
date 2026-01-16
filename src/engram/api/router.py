"""FastAPI router for Engram API endpoints."""

from __future__ import annotations

import logging
import time
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from engram.models import AuditEntry
from engram.service import EngramService

from .schemas import (
    BulkDeleteResponse,
    ConfidenceStats,
    ConsolidateRequest,
    ConsolidateResponse,
    DecayRequest,
    DecayResponse,
    EncodeRequest,
    EncodeResponse,
    EpisodeResponse,
    HealthResponse,
    MemoryCounts,
    MemoryStatsResponse,
    PromoteRequest,
    PromoteResponse,
    RecallRequest,
    RecallResponse,
    RecallResultResponse,
    SourceEpisodeDetail,
    SourceEpisodeSummary,
    SourcesResponse,
    StructureBatchRequest,
    StructureBatchResponse,
    StructuredResponse,
    StructureRequest,
    StructureResponse,
    VerificationResponse,
    WorkflowStatusResponse,
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
            enrich=request.enrich,
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

        structured_response = StructuredResponse(
            id=result.structured.id,
            source_episode_id=result.structured.source_episode_id,
            mode=result.structured.mode,
            enriched=result.structured.enriched,
            emails=result.structured.emails,
            phones=result.structured.phones,
            urls=result.structured.urls,
            confidence=result.structured.confidence.value,
        )

        extract_count = (
            len(result.structured.emails)
            + len(result.structured.phones)
            + len(result.structured.urls)
        )

        return EncodeResponse(
            episode=episode_response,
            structured=structured_response,
            extract_count=extract_count,
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
        # Convert Literal list to str list for service compatibility
        memory_types: list[str] | None = (
            list(request.memory_types) if request.memory_types is not None else None
        )

        # Use recall_at for bi-temporal queries, recall for standard queries
        if request.as_of is not None:
            results = await service.recall_at(
                query=request.query,
                as_of=request.as_of,
                user_id=request.user_id,
                org_id=request.org_id,
                limit=request.limit,
                min_confidence=request.min_confidence,
                memory_types=memory_types,
            )
        else:
            results = await service.recall(
                query=request.query,
                user_id=request.user_id,
                org_id=request.org_id,
                limit=request.limit,
                min_confidence=request.min_confidence,
                min_selectivity=request.min_selectivity,
                memory_types=memory_types,
                include_sources=request.include_sources,
                follow_links=request.follow_links,
                max_hops=request.max_hops,
                freshness=request.freshness,
                include_system_prompts=request.include_system_prompts,
                diversity=request.diversity,
            )

        result_responses = [
            RecallResultResponse(
                memory_type=r.memory_type,
                content=r.content,
                score=r.score,
                confidence=r.confidence,
                memory_id=r.memory_id,
                source_episode_id=r.source_episode_id,
                source_episode_ids=r.source_episode_ids,
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
                structured=stats.structured,
                semantic=stats.semantic,
                procedural=stats.procedural,
            ),
            confidence=ConfidenceStats(
                structured_avg=stats.structured_avg_confidence,
                structured_min=stats.structured_min_confidence,
                structured_max=stats.structured_max_confidence,
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

    Traces a derived memory (structured, semantic, or procedural)
    back to the source episode(s) it was extracted from. This enables
    provenance tracking and allows users to verify the origin of any
    derived knowledge.

    Args:
        memory_id: ID of the derived memory (must start with struct_, sem_, or proc_).
        user_id: User ID for multi-tenancy isolation.
        service: Injected EngramService.

    Returns:
        Source episodes in chronological order.

    Raises:
        HTTPException: 400 if memory_id format is invalid.
        HTTPException: 404 if memory not found.
    """
    # Determine memory type from ID prefix
    if memory_id.startswith("struct_"):
        memory_type = "structured"
    elif memory_id.startswith("sem_"):
        memory_type = "semantic"
    elif memory_id.startswith("proc_"):
        memory_type = "procedural"
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid memory ID format: {memory_id}. "
            "Expected prefix: struct_, sem_, or proc_",
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

    Traces a derived memory (structured, semantic, or procedural)
    back to its source episode(s) and provides a human-readable explanation
    of how it was derived. This is core to Engram's "memory you can trust"
    value proposition.

    Args:
        memory_id: ID of the derived memory (must start with struct_, sem_, or proc_).
        user_id: User ID for multi-tenancy isolation.
        service: Injected EngramService.

    Returns:
        Verification result with source traceability and explanation.

    Raises:
        HTTPException: 400 if memory_id format is invalid.
        HTTPException: 404 if memory not found.

    Example:
        GET /memories/struct_abc123/verify?user_id=u1

        Response:
        {
            "memory_id": "struct_abc123",
            "memory_type": "structured",
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
    valid_prefixes = ("struct_", "sem_", "proc_")
    if not memory_id.startswith(valid_prefixes):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid memory ID format: {memory_id}. "
            "Expected prefix: struct_, sem_, or proc_",
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


@router.delete(
    "/memories/{memory_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["memory"],
)
async def delete_memory(
    memory_id: str,
    user_id: str,
    service: ServiceDep,
    org_id: str | None = None,
) -> None:
    """Delete a specific memory by ID.

    Deletes a memory from the appropriate collection based on its ID prefix.
    Supported memory types:
    - ep_*: Episodic memories (ground truth)
    - struct_*: Structured memories (extracted facts)
    - sem_*: Semantic memories (consolidated knowledge)
    - proc_*: Procedural memories (behavioral patterns)

    Args:
        memory_id: ID of the memory to delete.
        user_id: User ID for multi-tenancy isolation.
        service: Injected EngramService.
        org_id: Optional organization ID.

    Raises:
        HTTPException: 400 if memory_id format is invalid.
        HTTPException: 404 if memory not found.
    """
    start_time = time.time()

    # Determine memory type from ID prefix and call appropriate delete method
    if memory_id.startswith("ep_"):
        memory_type = "episodic"
    elif memory_id.startswith("struct_"):
        memory_type = "structured"
    elif memory_id.startswith("sem_"):
        memory_type = "semantic"
    elif memory_id.startswith("proc_"):
        memory_type = "procedural"
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid memory ID format: {memory_id}. "
            "Expected prefix: ep_, struct_, sem_, or proc_",
        )

    try:
        # Call appropriate delete method based on memory type
        if memory_type == "episodic":
            deleted = await service.storage.delete_episode(memory_id, user_id)
        elif memory_type == "structured":
            deleted = await service.storage.delete_structured(memory_id, user_id)
        elif memory_type == "semantic":
            deleted = await service.storage.delete_semantic(memory_id, user_id)
        else:  # procedural
            deleted = await service.storage.delete_procedural(memory_id, user_id)

        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Memory not found: {memory_id}",
            )

        # Log audit entry for deletion
        duration_ms = int((time.time() - start_time) * 1000)
        audit_entry = AuditEntry.for_delete(
            user_id=user_id,
            memory_id=memory_id,
            memory_type=memory_type,
            org_id=org_id,
            duration_ms=duration_ms,
        )
        await service.storage.log_audit(audit_entry)

        logger.info("Deleted %s memory %s for user %s", memory_type, memory_id, user_id)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to delete memory %s", memory_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred while deleting the memory",
        ) from e


@router.delete(
    "/users/{user_id}/memories",
    response_model=BulkDeleteResponse,
    tags=["memory"],
)
async def delete_all_user_memories(
    user_id: str,
    service: ServiceDep,
    org_id: str | None = None,
) -> BulkDeleteResponse:
    """Delete all memories for a user (GDPR right to erasure).

    This endpoint permanently deletes all memories across all collections
    for the specified user. This operation is irreversible and is intended
    for GDPR compliance (right to erasure / right to be forgotten).

    Args:
        user_id: User ID whose memories should be deleted.
        service: Injected EngramService.
        org_id: Optional organization ID filter.

    Returns:
        Counts of deleted memories by type.

    Raises:
        HTTPException: If deletion fails.
    """
    start_time = time.time()

    try:
        deleted_counts = await service.storage.delete_all_user_memories(
            user_id=user_id,
            org_id=org_id,
        )

        total_deleted = sum(deleted_counts.values())

        # Log audit entry for bulk deletion
        duration_ms = int((time.time() - start_time) * 1000)
        audit_entry = AuditEntry.for_bulk_delete(
            user_id=user_id,
            deleted_counts=deleted_counts,
            org_id=org_id,
            duration_ms=duration_ms,
        )
        await service.storage.log_audit(audit_entry)

        logger.info(
            "Bulk deleted %d memories for user %s: %s",
            total_deleted,
            user_id,
            deleted_counts,
        )

        return BulkDeleteResponse(
            user_id=user_id,
            org_id=org_id,
            deleted_counts=deleted_counts,
            total_deleted=total_deleted,
        )

    except Exception as e:
        logger.exception("Failed to bulk delete memories for user %s", user_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred while deleting user memories",
        ) from e


# ============================================================================
# Workflow Trigger Endpoints
# ============================================================================


@router.post(
    "/workflows/consolidate",
    response_model=ConsolidateResponse,
    tags=["workflows"],
)
async def trigger_consolidation(
    request: ConsolidateRequest,
    service: ServiceDep,
) -> ConsolidateResponse:
    """Trigger memory consolidation workflow.

    Consolidates episodic memories into semantic memories by:
    1. Grouping related episodes
    2. Extracting lasting knowledge via LLM
    3. Creating semantic memories with links

    Args:
        request: Consolidation request with user filters and options.
        service: Injected EngramService.

    Returns:
        Consolidation results including counts and compression ratio.

    Raises:
        HTTPException: If consolidation fails.
    """
    try:
        assert service.workflow_backend is not None, "workflow_backend not initialized"
        result = await service.workflow_backend.run_consolidation(
            storage=service.storage,
            embedder=service.embedder,
            user_id=request.user_id,
            org_id=request.org_id,
            consolidation_passes=request.consolidation_passes,
            similarity_threshold=request.similarity_threshold,
        )

        return ConsolidateResponse(
            episodes_processed=result.episodes_processed,
            semantic_memories_created=result.semantic_memories_created,
            links_created=result.links_created,
            compression_ratio=result.compression_ratio,
        )

    except Exception as e:
        logger.exception("Failed to run consolidation for user %s", request.user_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred during consolidation",
        ) from e


@router.post(
    "/workflows/decay",
    response_model=DecayResponse,
    tags=["workflows"],
)
async def trigger_decay(
    request: DecayRequest,
    service: ServiceDep,
) -> DecayResponse:
    """Trigger memory decay workflow.

    Applies time-based confidence decay to memories:
    1. Reduces confidence based on age and access patterns
    2. Archives memories below threshold
    3. Deletes memories below delete threshold
    4. Optionally runs promotion for behavioral patterns

    Args:
        request: Decay request with user filters and options.
        service: Injected EngramService.

    Returns:
        Decay results including update/archive/delete counts.

    Raises:
        HTTPException: If decay fails.
    """
    try:
        assert service.workflow_backend is not None, "workflow_backend not initialized"
        result = await service.workflow_backend.run_decay(
            storage=service.storage,
            settings=service.settings,
            user_id=request.user_id,
            org_id=request.org_id,
            embedder=service.embedder,
            run_promotion=request.run_promotion,
        )

        return DecayResponse(
            memories_updated=result.memories_updated,
            memories_archived=result.memories_archived,
            memories_deleted=result.memories_deleted,
            procedural_promoted=result.procedural_promoted,
        )

    except Exception as e:
        logger.exception("Failed to run decay for user %s", request.user_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred during decay",
        ) from e


@router.post(
    "/workflows/promote",
    response_model=PromoteResponse,
    tags=["workflows"],
)
async def trigger_promotion(
    request: PromoteRequest,
    service: ServiceDep,
) -> PromoteResponse:
    """Trigger promotion/synthesis workflow.

    Promotes semantic memories to procedural memories:
    1. Analyzes semantic memories for behavioral patterns
    2. Synthesizes procedural knowledge
    3. Creates/updates procedural memory for user

    Args:
        request: Promotion request with user filters.
        service: Injected EngramService.

    Returns:
        Promotion results including procedural memory created/updated.

    Raises:
        HTTPException: If promotion fails.
    """
    try:
        assert service.workflow_backend is not None, "workflow_backend not initialized"
        result = await service.workflow_backend.run_promotion(
            storage=service.storage,
            embedder=service.embedder,
            user_id=request.user_id,
            org_id=request.org_id,
        )

        return PromoteResponse(
            semantics_analyzed=result.semantics_analyzed,
            procedural_created=result.procedural_created,
            procedural_updated=result.procedural_updated,
            procedural_id=result.procedural_id,
        )

    except Exception as e:
        logger.exception("Failed to run promotion for user %s", request.user_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred during promotion",
        ) from e


@router.post(
    "/workflows/structure",
    response_model=StructureResponse,
    tags=["workflows"],
)
async def trigger_structure(
    request: StructureRequest,
    service: ServiceDep,
) -> StructureResponse:
    """Trigger structure workflow for a specific episode.

    Extracts structured data from an episode via LLM:
    1. Runs deterministic extraction (emails, phones, URLs)
    2. Runs LLM extraction (dates, people, preferences, negations)
    3. Creates StructuredMemory linked to episode

    Args:
        request: Structure request with episode ID.
        service: Injected EngramService.

    Returns:
        Structure results including extraction counts.

    Raises:
        HTTPException: 404 if episode not found.
        HTTPException: If structure fails.
    """
    try:
        # Get the episode
        episode = await service.storage.get_episode(request.episode_id, request.user_id)
        if episode is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Episode not found: {request.episode_id}",
            )

        assert service.workflow_backend is not None, "workflow_backend not initialized"
        result = await service.workflow_backend.run_structure(
            episode=episode,
            storage=service.storage,
            embedder=service.embedder,
            model=request.model,
            skip_if_structured=request.skip_if_structured,
        )

        if result is None:
            # Episode was skipped (already structured)
            return StructureResponse(
                episode_id=request.episode_id,
                structured_memory_id=None,
                extracts_count=0,
                deterministic_count=0,
                llm_count=0,
                processing_time_ms=0,
                skipped=True,
            )

        return StructureResponse(
            episode_id=result.episode_id,
            structured_memory_id=result.structured_memory_id,
            extracts_count=result.extracts_count,
            deterministic_count=result.deterministic_count,
            llm_count=result.llm_count,
            processing_time_ms=int(result.processing_time_ms),
            skipped=False,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to run structure for episode %s", request.episode_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred during structure",
        ) from e


@router.post(
    "/workflows/structure/batch",
    response_model=StructureBatchResponse,
    tags=["workflows"],
)
async def trigger_structure_batch(
    request: StructureBatchRequest,
    service: ServiceDep,
) -> StructureBatchResponse:
    """Trigger batch structure workflow.

    Processes multiple unstructured episodes at once:
    1. Finds episodes without StructuredMemory
    2. Runs LLM extraction on each
    3. Returns aggregated results

    Args:
        request: Batch structure request with user filters and limit.
        service: Injected EngramService.

    Returns:
        Batch results including total extracts and per-episode details.

    Raises:
        HTTPException: If batch structure fails.
    """
    try:
        assert service.workflow_backend is not None, "workflow_backend not initialized"
        results = await service.workflow_backend.run_structure_batch(
            storage=service.storage,
            embedder=service.embedder,
            user_id=request.user_id,
            org_id=request.org_id,
            limit=request.limit,
            model=request.model,
        )

        response_results = [
            StructureResponse(
                episode_id=r.episode_id,
                structured_memory_id=r.structured_memory_id,
                extracts_count=r.extracts_count,
                deterministic_count=r.deterministic_count,
                llm_count=r.llm_count,
                processing_time_ms=int(r.processing_time_ms),
                skipped=False,
            )
            for r in results
        ]

        total_extracts = sum(r.extracts_count for r in results)

        return StructureBatchResponse(
            episodes_processed=len(results),
            total_extracts=total_extracts,
            results=response_results,
        )

    except Exception as e:
        logger.exception("Failed to run batch structure for user %s", request.user_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred during batch structure",
        ) from e


@router.get(
    "/workflows/{workflow_id}/status",
    response_model=WorkflowStatusResponse,
    tags=["workflows"],
)
async def get_workflow_status(
    workflow_id: str,
    service: ServiceDep,
) -> WorkflowStatusResponse:
    """Get status of a workflow execution.

    Polls the status of an async workflow execution.
    Only available when async_execution=True was used.

    Args:
        workflow_id: The workflow execution ID.
        service: Injected EngramService.

    Returns:
        Current workflow status including state and result if completed.

    Raises:
        HTTPException: 404 if workflow not found.
    """
    try:
        assert service.workflow_backend is not None, "workflow_backend not initialized"
        status_result = await service.workflow_backend.get_workflow_status(workflow_id)

        if status_result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workflow not found: {workflow_id}",
            )

        return WorkflowStatusResponse(
            workflow_id=status_result.workflow_id,
            workflow_type=status_result.workflow_type,
            state=status_result.state.value,
            started_at=status_result.started_at.isoformat() if status_result.started_at else None,
            completed_at=(
                status_result.completed_at.isoformat() if status_result.completed_at else None
            ),
            error=status_result.error,
            result=status_result.result,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to get workflow status for %s", workflow_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred while getting workflow status",
        ) from e
