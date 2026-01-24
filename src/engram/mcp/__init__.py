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

Available tools:
- engram_encode: Store a memory with content extraction
- engram_recall: Search memories by semantic similarity
- engram_verify: Verify memory provenance back to sources
- engram_stats: Get memory statistics for a user
"""

from .server import create_server, main

__all__ = ["create_server", "main"]
