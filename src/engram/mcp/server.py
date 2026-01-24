"""MCP server implementation for Engram.

This module provides an MCP (Model Context Protocol) server that exposes
Engram's memory capabilities as tools that can be used by Claude Code
and other MCP-compatible clients.

The server uses STDIO transport for communication, which is the standard
for local MCP servers integrated with Claude Code.

Tools provided:
- engram_encode: Store a memory and extract structured data
- engram_recall: Search memories by semantic similarity
- engram_verify: Verify a memory's provenance
- engram_stats: Get memory statistics for a user
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from mcp.server.fastmcp import FastMCP

from engram.config import Settings
from engram.service import EngramService

# Configure logging to stderr (stdout is reserved for MCP protocol)
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def create_server() -> FastMCP:
    """Create and configure the MCP server with Engram tools.

    Returns:
        Configured FastMCP server instance.
    """
    mcp = FastMCP("engram")

    # Shared service instance (lazy initialization)
    _service: EngramService | None = None
    _service_lock = asyncio.Lock()

    async def get_service() -> EngramService:
        """Get or create the shared EngramService instance."""
        nonlocal _service
        async with _service_lock:
            if _service is None:
                settings = Settings()
                _service = EngramService.create(settings)
                await _service.initialize()
            return _service

    @mcp.tool()
    async def engram_encode(
        content: str,
        user_id: str,
        role: str = "user",
        session_id: str | None = None,
        enrich: bool = False,
    ) -> str:
        """Store a memory with automatic content extraction.

        Encodes content into Engram's memory system, extracting structured
        data like emails, phone numbers, URLs, and (optionally with enrich=True)
        LLM-extracted entities like people, preferences, and negations.

        Args:
            content: The text content to store as a memory.
            user_id: User identifier for multi-tenancy isolation.
            role: Role of the content source (user, assistant, system).
            session_id: Optional session identifier for grouping memories.
            enrich: If True, use LLM extraction for richer data (slower but more complete).

        Returns:
            JSON string with episode ID and extracted data summary.

        Example:
            >>> engram_encode("My email is john@example.com", user_id="user_123")
            '{"episode_id": "ep_...", "emails": ["john@example.com"], ...}'
        """
        service = await get_service()
        result = await service.encode(
            content=content,
            role=role,
            user_id=user_id,
            session_id=session_id,
            enrich=enrich,
        )

        # Build response with key information
        response: dict[str, Any] = {
            "episode_id": result.episode.id,
            "structured_id": result.structured.id,
            "emails": result.structured.emails,
            "phones": result.structured.phones,
            "urls": result.structured.urls,
            "enriched": result.structured.enriched,
        }

        # Include LLM-extracted data if enriched
        if result.structured.enriched:
            response["people"] = [p.model_dump() for p in result.structured.people]
            response["preferences"] = [p.model_dump() for p in result.structured.preferences]
            response["negations"] = [n.model_dump() for n in result.structured.negations]
            if result.structured.summary:
                response["summary"] = result.structured.summary

        return json.dumps(response, indent=2)

    @mcp.tool()
    async def engram_recall(
        query: str,
        user_id: str,
        limit: int = 10,
        min_confidence: float | None = None,
        memory_types: str | None = None,
        include_sources: bool = False,
    ) -> str:
        """Search memories by semantic similarity.

        Retrieves memories that are semantically similar to the query,
        searching across different memory types (episodic, structured,
        semantic, procedural).

        Args:
            query: Natural language query to search for.
            user_id: User identifier for multi-tenancy isolation.
            limit: Maximum number of results to return (default: 10).
            min_confidence: Minimum confidence threshold (0.0-1.0) for filtering.
            memory_types: Comma-separated list of types to search (e.g., "episodic,semantic").
                        Valid types: episodic, structured, semantic, procedural, working.
                        If not specified, searches all types.
            include_sources: If True, include source episode details for derived memories.

        Returns:
            JSON string with list of matching memories and their scores.

        Example:
            >>> engram_recall("email address", user_id="user_123")
            '[{"memory_type": "structured", "content": "...", "score": 0.89, ...}]'
        """
        service = await get_service()

        # Parse memory_types if provided
        types_list: list[str] | None = None
        if memory_types:
            types_list = [t.strip() for t in memory_types.split(",")]

        results = await service.recall(
            query=query,
            user_id=user_id,
            limit=limit,
            min_confidence=min_confidence,
            memory_types=types_list,
            include_sources=include_sources,
        )

        # Convert results to serializable format
        response: list[dict[str, Any]] = []
        for r in results:
            item: dict[str, Any] = {
                "memory_type": r.memory_type,
                "memory_id": r.memory_id,
                "content": r.content,
                "score": round(r.score, 4),
            }
            if r.confidence is not None:
                item["confidence"] = round(r.confidence, 4)
            if r.source_episode_id:
                item["source_episode_id"] = r.source_episode_id
            if r.source_episode_ids:
                item["source_episode_ids"] = r.source_episode_ids
            if r.source_episodes:
                item["source_episodes"] = [
                    {"id": s.id, "content": s.content, "role": s.role, "timestamp": s.timestamp}
                    for s in r.source_episodes
                ]
            if r.related_ids:
                item["related_ids"] = r.related_ids

            response.append(item)

        return json.dumps(response, indent=2)

    @mcp.tool()
    async def engram_verify(
        memory_id: str,
        user_id: str,
    ) -> str:
        """Verify a memory's provenance back to source episodes.

        Traces a derived memory (structured, semantic, or procedural) back
        to its source episode(s) and provides an explanation of how it was
        derived, including confidence breakdown.

        Args:
            memory_id: ID of the memory to verify (must start with struct_, sem_, or proc_).
            user_id: User identifier for multi-tenancy isolation.

        Returns:
            JSON string with verification result including source episodes and explanation.

        Example:
            >>> engram_verify("struct_abc123", user_id="user_123")
            '{"verified": true, "explanation": "Pattern-matched from source...", ...}'
        """
        service = await get_service()

        try:
            result = await service.verify(memory_id=memory_id, user_id=user_id)

            response: dict[str, Any] = {
                "memory_id": result.memory_id,
                "memory_type": result.memory_type,
                "content": result.content,
                "verified": result.verified,
                "extraction_method": result.extraction_method,
                "confidence": round(result.confidence, 4),
                "explanation": result.explanation,
                "source_episodes": result.source_episodes,
            }

            return json.dumps(response, indent=2)

        except Exception as e:
            return json.dumps({"error": str(e), "memory_id": memory_id}, indent=2)

    @mcp.tool()
    async def engram_stats(
        user_id: str,
    ) -> str:
        """Get memory statistics for a user.

        Returns counts and summaries of the user's memories across all types,
        useful for understanding the state of the memory system.

        Args:
            user_id: User identifier to get statistics for.

        Returns:
            JSON string with memory counts by type and confidence statistics.

        Example:
            >>> engram_stats(user_id="user_123")
            '{"episodic_count": 42, "structured_count": 42, "semantic_count": 5, ...}'
        """
        service = await get_service()

        # Get comprehensive stats from storage
        stats = await service.storage.get_memory_stats(user_id)

        # Build response with key information
        response: dict[str, Any] = {
            "user_id": user_id,
            "episodic_count": stats.episodes,
            "structured_count": stats.structured,
            "semantic_count": stats.semantic,
            "procedural_count": stats.procedural,
            "total_memories": stats.episodes + stats.structured + stats.semantic + stats.procedural,
            "pending_consolidation": stats.pending_consolidation,
        }

        # Add confidence statistics if available
        if stats.structured_avg_confidence is not None:
            response["structured_avg_confidence"] = round(stats.structured_avg_confidence, 4)
        if stats.structured_min_confidence is not None:
            response["structured_min_confidence"] = round(stats.structured_min_confidence, 4)
        if stats.structured_max_confidence is not None:
            response["structured_max_confidence"] = round(stats.structured_max_confidence, 4)
        if stats.semantic_avg_confidence is not None:
            response["semantic_avg_confidence"] = round(stats.semantic_avg_confidence, 4)

        return json.dumps(response, indent=2)

    return mcp


def main() -> None:
    """Run the MCP server with STDIO transport."""
    mcp = create_server()
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
