"""MCP server for Engram.

Provides Model Context Protocol (MCP) tools for Claude Code and other
MCP-compatible clients to interact with Engram's memory system.

Example usage with Claude Code:

    Add to your Claude Code MCP settings:
    ```json
    {
      "mcpServers": {
        "engram": {
          "command": "uv",
          "args": ["run", "--extra", "mcp", "python", "-m", "engram.mcp"]
        }
      }
    }
    ```

Available tools (10):
- engram_encode: Store a memory with content extraction
- engram_recall: Search memories by semantic similarity
- engram_recall_at: Query memories as of a point in time
- engram_search: Filter/list memories by metadata
- engram_verify: Verify memory provenance back to sources
- engram_stats: Get memory statistics for a user
- engram_get: Get a specific memory by ID
- engram_delete: Delete a memory by ID
- engram_consolidate: Consolidate episodes into semantic memories
- engram_promote: Promote semantic memories to procedural
"""

from .server import create_server, main

__all__ = ["create_server", "main"]
