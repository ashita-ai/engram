"""MCP server implementation for Engram.

This module provides an MCP (Model Context Protocol) server that exposes
Engram's memory capabilities as tools that can be used by Claude Code
and other MCP-compatible clients.

The server uses STDIO transport for communication, which is the standard
for local MCP servers integrated with Claude Code.

Tools provided (10):

Core Operations:
- engram_encode: Store a memory and extract structured data
- engram_recall: Search memories with full retrieval semantics (multi-hop, diversity, reranking)
- engram_recall_at: Bi-temporal query - memories as they existed at a point in time
- engram_search: Filter/list memories by metadata without semantic search
- engram_verify: Verify a memory's provenance
- engram_stats: Get memory statistics for a user

Memory Management:
- engram_get: Get a specific memory by ID
- engram_delete: Delete a memory by ID

Workflows:
- engram_consolidate: Consolidate episodes into semantic memories
- engram_promote: Promote semantic memories to procedural

For admin/debug tools (linking, batch import, sessions, history), see engram-admin server.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from mcp.server.fastmcp import FastMCP

from engram.config import Settings
from engram.service import EngramService

from .context import get_default_user, get_project_context

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
        user_id: str | None = None,
        role: str = "user",
        session_id: str | None = None,
        enrich: bool = False,
    ) -> str:
        """Store content as a memory. Use this for every user message you want remembered.

        WHEN TO USE:
        - Store user statements: preferences, facts, decisions, requests
        - Store assistant responses that contain commitments or information given
        - Store any content the user might want to recall later

        ENRICH OPTION:
        - enrich=False (default): Fast regex extraction of emails, phones, URLs.
          Use for routine logging where speed matters.
        - enrich=True: LLM extraction of people, preferences, negations, summaries.
          Use for important content with nuanced meaning (e.g., "I stopped liking X").

        ROLES:
        - "user": Content from the human (most common)
        - "assistant": Content from the AI (store promises, answers given)
        - "system": Metadata or context information

        SESSIONS:
        Include session_id to group memories by conversation. This enables:
        - engram_search by session_id to review a conversation
        - Session-aware reranking in engram_recall

        AUTO-DETECTION:
        - user_id: Auto-detected from ENGRAM_USER env, git user.name, or system username
        - org_id: Auto-detected from git remote, repo name, or directory name

        Args:
            content: The text to store (conversation turn, note, fact).
            user_id: User identifier. Auto-detected if not provided.
            role: "user", "assistant", or "system".
            session_id: Group memories by conversation/session.
            enrich: True for LLM extraction (slower, richer); False for regex only.

        Returns:
            JSON with episode_id, structured_id, and extracted data.
        """
        # Auto-detect user_id if not provided
        resolved_user_id = user_id or get_default_user()
        if not resolved_user_id:
            return json.dumps(
                {"error": "user_id required. Set ENGRAM_USER env var or pass explicitly."},
                indent=2,
            )

        # Auto-detect org_id from project context
        org_id = get_project_context()

        service = await get_service()
        result = await service.encode(
            content=content,
            role=role,
            user_id=resolved_user_id,
            org_id=org_id,
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
            "duration_ms": result.duration_ms,
            "timing": result.timing,
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
        user_id: str | None = None,
        limit: int = 10,
        min_confidence: float | None = None,
        memory_types: str | None = None,
        include_sources: bool = False,
        session_id: str | None = None,
        follow_links: bool = False,
        max_hops: int = 2,
        diversity: float = 0.0,
        apply_negation_filter: bool = True,
        freshness: str = "best_effort",
        expand_query: bool = False,
        rerank: bool = True,
    ) -> str:
        """Search memories by semantic similarity. This is the primary retrieval tool.

        Searches all memory types by default: episodic (raw conversations), structured
        (extracted entities), semantic (synthesized facts), procedural (behavioral patterns),
        and working (current session).

        WHEN TO USE WHICH OPTIONS:
        - Default (no options): Simple semantic search across all memory types
        - follow_links=True: When you need context beyond the direct match
          (e.g., "what do I know about this person AND their connections?")
        - diversity=0.3: When results seem redundant or you need varied perspectives
        - memory_types="semantic,procedural": When you want only synthesized knowledge,
          not raw conversation history
        - include_sources=True: When you need to verify or cite the original source
        - expand_query=True: When the query is vague and you want broader recall

        TOOL SELECTION GUIDE:
        - Use engram_recall for: "What do I know about X?", semantic questions
        - Use engram_search for: "List all memories from session Y", metadata filters
        - Use engram_recall_at for: "What did I know about X last week?", time travel
        - Use engram_get for: Fetching a specific memory by ID

        AUTO-DETECTION:
        - user_id: Auto-detected from ENGRAM_USER env, git user.name, or system username
        - org_id: Auto-detected from git remote, repo name, or directory name

        Args:
            query: Natural language query to search for.
            user_id: User identifier. Auto-detected if not provided.
            limit: Maximum results (default 10). Use 3-5 for focused answers, 20+ for exploration.
            min_confidence: Filter by confidence (0.0-1.0). Use 0.7+ for high-certainty answers.
            memory_types: Comma-separated types: episodic, structured, semantic, procedural, working.
                        Omit to search all. Use "semantic,procedural" for synthesized knowledge only.
            include_sources: Include source episode IDs for derived memories. Enable for citations.
            session_id: Current session ID. Provides relevance boost to current conversation.
            follow_links: Enable multi-hop reasoning via A-MEM links. Use when you need
                        connected context (e.g., person -> their preferences -> related people).
            max_hops: Link traversal depth (1-3). 1=direct links, 2=friends-of-friends.
            diversity: MMR diversity (0.0-1.0). 0.3 recommended when results seem repetitive.
            apply_negation_filter: Filter out negated memories (e.g., "user STOPPED liking X").
            freshness: "best_effort" (default) returns all; "fresh_only" for fully consolidated.
            expand_query: LLM-expand vague queries. Slower but better recall for broad questions.
            rerank: Context-aware reranking (default True). Disable only for raw similarity scores.

        Returns:
            JSON array of memories with: memory_type, memory_id, content, score, confidence.
        """
        # Auto-detect user_id if not provided
        resolved_user_id = user_id or get_default_user()
        if not resolved_user_id:
            return json.dumps(
                {"error": "user_id required. Set ENGRAM_USER env var or pass explicitly."},
                indent=2,
            )

        # Auto-detect org_id from project context
        org_id = get_project_context()

        service = await get_service()

        # Parse memory_types if provided
        types_list: list[str] | None = None
        if memory_types:
            types_list = [t.strip() for t in memory_types.split(",")]

        results = await service.recall(
            query=query,
            user_id=resolved_user_id,
            org_id=org_id,
            limit=limit,
            min_confidence=min_confidence,
            memory_types=types_list,
            include_sources=include_sources,
            session_id=session_id,
            follow_links=follow_links,
            max_hops=max_hops,
            diversity=diversity,
            apply_negation_filter=apply_negation_filter,
            freshness=freshness,
            expand_query=expand_query,
            rerank=rerank,
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
            if r.hop_distance is not None and r.hop_distance > 0:
                item["hop_distance"] = r.hop_distance
            if r.staleness:
                item["staleness"] = (
                    r.staleness.value if hasattr(r.staleness, "value") else str(r.staleness)
                )

            response.append(item)

        return json.dumps(response, indent=2)

    @mcp.tool()
    async def engram_verify(
        memory_id: str,
        user_id: str | None = None,
    ) -> str:
        """Verify a memory's provenance back to source episodes.

        Traces a derived memory (structured, semantic, or procedural) back
        to its source episode(s) and provides an explanation of how it was
        derived, including confidence breakdown.

        AUTO-DETECTION:
        - user_id: Auto-detected from ENGRAM_USER env, git user.name, or system username
        - org_id: Auto-detected from git remote, repo name, or directory name

        Args:
            memory_id: ID of the memory to verify (must start with struct_, sem_, or proc_).
            user_id: User identifier. Auto-detected if not provided.

        Returns:
            JSON string with verification result including source episodes and explanation.
        """
        # Auto-detect user_id if not provided
        resolved_user_id = user_id or get_default_user()
        if not resolved_user_id:
            return json.dumps(
                {"error": "user_id required. Set ENGRAM_USER env var or pass explicitly."},
                indent=2,
            )

        service = await get_service()

        try:
            result = await service.verify(memory_id=memory_id, user_id=resolved_user_id)

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
        user_id: str | None = None,
    ) -> str:
        """Get memory statistics for a user.

        Returns counts and summaries of the user's memories across all types,
        useful for understanding the state of the memory system.

        AUTO-DETECTION:
        - user_id: Auto-detected from ENGRAM_USER env, git user.name, or system username
        - org_id: Auto-detected from git remote, repo name, or directory name

        Args:
            user_id: User identifier. Auto-detected if not provided.

        Returns:
            JSON string with memory counts by type and confidence statistics.
        """
        # Auto-detect user_id if not provided
        resolved_user_id = user_id or get_default_user()
        if not resolved_user_id:
            return json.dumps(
                {"error": "user_id required. Set ENGRAM_USER env var or pass explicitly."},
                indent=2,
            )

        # Auto-detect org_id from project context
        org_id = get_project_context()

        service = await get_service()

        # Get comprehensive stats from storage
        stats = await service.storage.get_memory_stats(resolved_user_id, org_id=org_id)

        # Build response with key information
        response: dict[str, Any] = {
            "user_id": resolved_user_id,
            "org_id": org_id,
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

    @mcp.tool()
    async def engram_delete(
        memory_id: str,
        user_id: str | None = None,
        cascade: bool = False,
    ) -> str:
        """Delete a memory by ID.

        Removes a memory from storage. For episodes, can optionally cascade
        to delete derived structured memories.

        AUTO-DETECTION:
        - user_id: Auto-detected from ENGRAM_USER env, git user.name, or system username

        Args:
            memory_id: ID of the memory to delete (ep_, struct_, sem_, or proc_ prefix).
            user_id: User identifier. Auto-detected if not provided.
            cascade: If True and deleting an episode, also delete its derived structured memory.

        Returns:
            JSON string with deletion result.
        """
        # Auto-detect user_id if not provided
        resolved_user_id = user_id or get_default_user()
        if not resolved_user_id:
            return json.dumps(
                {"error": "user_id required. Set ENGRAM_USER env var or pass explicitly."},
                indent=2,
            )

        service = await get_service()
        cascade_deleted: list[str] = []

        try:
            deleted: bool = False

            if memory_id.startswith("ep_"):
                # Get structured memory for cascade before deleting episode
                if cascade:
                    struct = await service.storage.get_structured_for_episode(
                        memory_id, resolved_user_id
                    )
                    if struct:
                        await service.storage.delete_structured(struct.id, resolved_user_id)
                        cascade_deleted.append(struct.id)
                # delete_episode returns dict with "deleted" key
                result = await service.storage.delete_episode(memory_id, resolved_user_id)
                deleted = bool(result.get("deleted", False))
            elif memory_id.startswith("struct_"):
                deleted = await service.storage.delete_structured(memory_id, resolved_user_id)
            elif memory_id.startswith("sem_"):
                deleted = await service.storage.delete_semantic(memory_id, resolved_user_id)
            elif memory_id.startswith("proc_"):
                deleted = await service.storage.delete_procedural(memory_id, resolved_user_id)
            else:
                return json.dumps(
                    {
                        "error": f"Unknown memory type prefix in ID: {memory_id}",
                        "memory_id": memory_id,
                    },
                    indent=2,
                )

            response: dict[str, Any] = {
                "deleted": deleted,
                "memory_id": memory_id,
            }
            if cascade_deleted:
                response["cascade_deleted"] = cascade_deleted

            return json.dumps(response, indent=2)

        except Exception as e:
            return json.dumps({"error": str(e), "memory_id": memory_id}, indent=2)

    @mcp.tool()
    async def engram_get(
        memory_id: str,
        user_id: str | None = None,
    ) -> str:
        """Get a specific memory by ID.

        Retrieves full details of a memory including all metadata.

        AUTO-DETECTION:
        - user_id: Auto-detected from ENGRAM_USER env, git user.name, or system username

        Args:
            memory_id: ID of the memory (ep_, struct_, sem_, or proc_ prefix).
            user_id: User identifier. Auto-detected if not provided.

        Returns:
            JSON string with memory details or error if not found.
        """
        # Auto-detect user_id if not provided
        resolved_user_id = user_id or get_default_user()
        if not resolved_user_id:
            return json.dumps(
                {"error": "user_id required. Set ENGRAM_USER env var or pass explicitly."},
                indent=2,
            )

        service = await get_service()

        try:
            if memory_id.startswith("ep_"):
                episode = await service.storage.get_episode(memory_id, resolved_user_id)
                if episode is None:
                    return json.dumps(
                        {"error": "Episode not found", "memory_id": memory_id}, indent=2
                    )
                return json.dumps(
                    {
                        "id": episode.id,
                        "memory_type": "episodic",
                        "content": episode.content,
                        "role": episode.role,
                        "timestamp": episode.timestamp.isoformat() if episode.timestamp else None,
                        "session_id": episode.session_id,
                        "importance": episode.importance,
                        "summarized": episode.summarized,
                    },
                    indent=2,
                )

            elif memory_id.startswith("struct_"):
                structured = await service.storage.get_structured(memory_id, resolved_user_id)
                if structured is None:
                    return json.dumps(
                        {"error": "StructuredMemory not found", "memory_id": memory_id}, indent=2
                    )
                return json.dumps(
                    {
                        "id": structured.id,
                        "memory_type": "structured",
                        "source_episode_id": structured.source_episode_id,
                        "emails": structured.emails,
                        "phones": structured.phones,
                        "urls": structured.urls,
                        "people": [p.model_dump() for p in structured.people],
                        "preferences": [p.model_dump() for p in structured.preferences],
                        "negations": [n.model_dump() for n in structured.negations],
                        "summary": structured.summary,
                        "enriched": structured.enriched,
                        "confidence": structured.confidence.value,
                        "derived_at": structured.derived_at.isoformat()
                        if structured.derived_at
                        else None,
                    },
                    indent=2,
                )

            elif memory_id.startswith("sem_"):
                semantic = await service.storage.get_semantic(memory_id, resolved_user_id)
                if semantic is None:
                    return json.dumps(
                        {"error": "SemanticMemory not found", "memory_id": memory_id}, indent=2
                    )
                return json.dumps(
                    {
                        "id": semantic.id,
                        "memory_type": "semantic",
                        "content": semantic.content,
                        "source_episode_ids": semantic.source_episode_ids,
                        "related_ids": semantic.related_ids,
                        "confidence": semantic.confidence.value,
                        "consolidation_strength": semantic.consolidation_strength,
                        "consolidation_passes": semantic.consolidation_passes,
                        "derived_at": semantic.derived_at.isoformat()
                        if semantic.derived_at
                        else None,
                        "last_accessed": semantic.last_accessed.isoformat()
                        if semantic.last_accessed
                        else None,
                    },
                    indent=2,
                )

            elif memory_id.startswith("proc_"):
                procedural = await service.storage.get_procedural(memory_id, resolved_user_id)
                if procedural is None:
                    return json.dumps(
                        {"error": "ProceduralMemory not found", "memory_id": memory_id}, indent=2
                    )
                return json.dumps(
                    {
                        "id": procedural.id,
                        "memory_type": "procedural",
                        "content": procedural.content,
                        "source_episode_ids": procedural.source_episode_ids,
                        "source_semantic_ids": procedural.source_semantic_ids,
                        "confidence": procedural.confidence.value,
                        "consolidation_strength": procedural.consolidation_strength,
                        "derived_at": procedural.derived_at.isoformat()
                        if procedural.derived_at
                        else None,
                    },
                    indent=2,
                )

            else:
                return json.dumps(
                    {
                        "error": f"Unknown memory type prefix in ID: {memory_id}",
                        "memory_id": memory_id,
                    },
                    indent=2,
                )

        except Exception as e:
            return json.dumps({"error": str(e), "memory_id": memory_id}, indent=2)

    @mcp.tool()
    async def engram_consolidate(
        user_id: str | None = None,
    ) -> str:
        """Consolidate episodes into semantic memories.

        Implements the memory consolidation pipeline: takes all unsummarized
        episodes for a user and creates a semantic memory that captures the
        key facts and relationships. This is the core learning loop.

        Based on Complementary Learning Systems theory: episodic (hippocampus)
        to semantic (neocortex) transfer with compression.

        AUTO-DETECTION:
        - user_id: Auto-detected from ENGRAM_USER env, git user.name, or system username
        - org_id: Auto-detected from git remote, repo name, or directory name

        Args:
            user_id: User identifier. Auto-detected if not provided.

        Returns:
            JSON string with consolidation statistics.
        """
        # Auto-detect user_id if not provided
        resolved_user_id = user_id or get_default_user()
        if not resolved_user_id:
            return json.dumps(
                {"error": "user_id required. Set ENGRAM_USER env var or pass explicitly."},
                indent=2,
            )

        # Auto-detect org_id from project context
        org_id = get_project_context()

        service = await get_service()

        try:
            result = await service.consolidate(user_id=resolved_user_id, org_id=org_id)

            return json.dumps(
                {
                    "success": True,
                    "episodes_processed": result.episodes_processed,
                    "semantic_memories_created": result.semantic_memories_created,
                    "links_created": result.links_created,
                    "compression_ratio": round(result.compression_ratio, 2)
                    if result.compression_ratio
                    else None,
                },
                indent=2,
            )

        except Exception as e:
            return json.dumps({"error": str(e), "success": False}, indent=2)

    @mcp.tool()
    async def engram_promote(
        user_id: str | None = None,
    ) -> str:
        """Promote semantic memories to procedural memory.

        Creates or updates a procedural memory that captures behavioral patterns,
        preferences, and communication style from all semantic memories.

        Design: ONE procedural memory per user (replaces existing).

        AUTO-DETECTION:
        - user_id: Auto-detected from ENGRAM_USER env, git user.name, or system username
        - org_id: Auto-detected from git remote, repo name, or directory name

        Args:
            user_id: User identifier. Auto-detected if not provided.

        Returns:
            JSON string with promotion statistics.
        """
        # Auto-detect user_id if not provided
        resolved_user_id = user_id or get_default_user()
        if not resolved_user_id:
            return json.dumps(
                {"error": "user_id required. Set ENGRAM_USER env var or pass explicitly."},
                indent=2,
            )

        # Auto-detect org_id from project context
        org_id = get_project_context()

        service = await get_service()

        try:
            result = await service.create_procedural(user_id=resolved_user_id, org_id=org_id)

            return json.dumps(
                {
                    "success": True,
                    "procedural_id": result.procedural_id,
                    "procedural_created": result.procedural_created,
                    "procedural_updated": result.procedural_updated,
                    "semantics_analyzed": result.semantics_analyzed,
                },
                indent=2,
            )

        except Exception as e:
            return json.dumps({"error": str(e), "success": False}, indent=2)

    @mcp.tool()
    async def engram_search(
        user_id: str | None = None,
        memory_types: str | None = None,
        created_after: str | None = None,
        created_before: str | None = None,
        min_confidence: float | None = None,
        session_id: str | None = None,
        sort_by: str = "created_at",
        sort_order: str = "desc",
        limit: int = 50,
        offset: int = 0,
    ) -> str:
        """Browse/filter/list memories by metadata. Use instead of engram_recall when you
        don't have a semantic query - just want to list or filter memories.

        This tool combines listing and filtering. Use it for:
        - Listing all memories of a type
        - Filtering by time range, session, or confidence
        - Paginated browsing

        WHEN TO USE THIS vs engram_recall:
        - engram_search: "Show memories from last week", "List high-confidence facts",
          "What memories are in session X?", "List all semantic memories"
        - engram_recall: "What does the user prefer for breakfast?" (semantic question)

        COMMON PATTERNS:
        - List all semantic: memory_types="semantic"
        - Recent activity: created_after="2024-01-01", sort_by="created_at", sort_order="desc"
        - High-confidence facts: memory_types="semantic", min_confidence=0.8
        - Session review: session_id="session_xyz" to see all memories from a conversation
        - Paginated browse: limit=20, offset=0, then offset=20 for next page

        AUTO-DETECTION:
        - user_id: Auto-detected from ENGRAM_USER env, git user.name, or system username
        - org_id: Auto-detected from git remote, repo name, or directory name

        Args:
            user_id: User identifier. Auto-detected if not provided.
            memory_types: Comma-separated: episodic, structured, semantic, procedural.
                         Omit to list all types.
            created_after: ISO datetime (e.g., "2024-01-15T00:00:00Z").
            created_before: ISO datetime for upper bound.
            min_confidence: Filter by confidence (0.0-1.0). Use 0.7+ for reliable facts.
            session_id: Filter to a specific conversation session.
            sort_by: "created_at" (default) or "confidence".
            sort_order: "desc" (newest first) or "asc".
            limit: Results per page (default 50).
            offset: Skip N results for pagination.

        Returns:
            JSON with total count and memories array.
        """
        from datetime import datetime

        # Auto-detect user_id if not provided
        resolved_user_id = user_id or get_default_user()
        if not resolved_user_id:
            return json.dumps(
                {"error": "user_id required. Set ENGRAM_USER env var or pass explicitly."},
                indent=2,
            )

        # Auto-detect org_id from project context
        org_id = get_project_context()

        service = await get_service()

        try:
            # Parse memory_types
            types_list: list[str] | None = None
            if memory_types:
                types_list = [t.strip() for t in memory_types.split(",")]

            # Parse datetime filters
            after_dt: datetime | None = None
            before_dt: datetime | None = None
            if created_after:
                after_dt = datetime.fromisoformat(created_after.replace("Z", "+00:00"))
            if created_before:
                before_dt = datetime.fromisoformat(created_before.replace("Z", "+00:00"))

            results, total = await service.storage.search_memories(
                user_id=resolved_user_id,
                org_id=org_id,
                memory_types=types_list,
                created_after=after_dt,
                created_before=before_dt,
                min_confidence=min_confidence,
                session_id=session_id,
                sort_by=sort_by,
                sort_order=sort_order,
                limit=limit,
                offset=offset,
            )

            return json.dumps(
                {
                    "total": total,
                    "limit": limit,
                    "offset": offset,
                    "memories": results,
                },
                indent=2,
            )

        except Exception as e:
            return json.dumps({"error": str(e)}, indent=2)

    @mcp.tool()
    async def engram_recall_at(
        query: str,
        as_of: str,
        user_id: str | None = None,
        limit: int = 10,
        min_confidence: float | None = None,
        memory_types: str | None = None,
    ) -> str:
        """Recall memories as they existed at a specific point in time.

        Bi-temporal query that only returns memories derived before the
        specified timestamp. Useful for understanding what the system
        "knew" at a given moment.

        AUTO-DETECTION:
        - user_id: Auto-detected from ENGRAM_USER env, git user.name, or system username
        - org_id: Auto-detected from git remote, repo name, or directory name

        Args:
            query: Natural language query to search for.
            as_of: ISO datetime - only memories derived before this time.
            user_id: User identifier. Auto-detected if not provided.
            limit: Maximum results (default 10).
            min_confidence: Minimum confidence threshold.
            memory_types: Comma-separated types (currently supports episodic,structured).

        Returns:
            JSON string with matching memories from that point in time.
        """
        from datetime import datetime

        # Auto-detect user_id if not provided
        resolved_user_id = user_id or get_default_user()
        if not resolved_user_id:
            return json.dumps(
                {"error": "user_id required. Set ENGRAM_USER env var or pass explicitly."},
                indent=2,
            )

        # Auto-detect org_id from project context
        org_id = get_project_context()

        service = await get_service()

        try:
            # Parse as_of datetime
            as_of_dt = datetime.fromisoformat(as_of.replace("Z", "+00:00"))

            # Parse memory_types
            types_list: list[str] | None = None
            if memory_types:
                types_list = [t.strip() for t in memory_types.split(",")]

            results = await service.recall_at(
                query=query,
                as_of=as_of_dt,
                user_id=resolved_user_id,
                org_id=org_id,
                limit=limit,
                min_confidence=min_confidence,
                memory_types=types_list,
            )

            # Convert to JSON
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
                if r.metadata:
                    item["metadata"] = r.metadata
                response.append(item)

            return json.dumps(response, indent=2)

        except Exception as e:
            return json.dumps({"error": str(e)}, indent=2)

    return mcp


def main() -> None:
    """Run the MCP server with STDIO transport."""
    mcp = create_server()
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
