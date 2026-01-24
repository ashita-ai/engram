"""Entry point for running Engram MCP server as a module.

Usage:
    python -m engram.mcp

Or with uv:
    uv run --extra mcp python -m engram.mcp
"""

from .server import main

if __name__ == "__main__":
    main()
