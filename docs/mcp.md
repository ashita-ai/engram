# MCP Server Integration

Engram provides an MCP (Model Context Protocol) server that allows Claude Code and other MCP-compatible clients to use Engram as a memory system directly from within their environment.

## Overview

The MCP server exposes four tools:

| Tool | Description |
|------|-------------|
| `engram_encode` | Store a memory with automatic content extraction |
| `engram_recall` | Search memories by semantic similarity |
| `engram_verify` | Verify memory provenance back to sources |
| `engram_stats` | Get memory statistics for a user |

## Setup

### Prerequisites

1. **Qdrant** must be running (default: `http://localhost:6333`)

   ```bash
   docker run -p 6333:6333 qdrant/qdrant
   ```

2. Install Engram with the MCP extra:

   ```bash
   cd engram
   uv sync --extra mcp
   ```

### Claude Code Configuration

Add Engram to your Claude Code MCP settings (`~/.claude/settings.json`):

```json
{
  "mcpServers": {
    "engram": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/engram", "--extra", "mcp", "python", "-m", "engram.mcp"]
    }
  }
}
```

Replace `/path/to/engram` with the actual path to your Engram installation.

### Environment Variables

The MCP server uses the same environment variables as the main Engram service:

| Variable | Default | Description |
|----------|---------|-------------|
| `ENGRAM_QDRANT_URL` | `http://localhost:6333` | Qdrant connection URL |
| `ENGRAM_EMBEDDING_PROVIDER` | `fastembed` | Embedding backend (`fastembed` or `openai`) |
| `OPENAI_API_KEY` | - | Required if using `openai` embedding provider |

## Usage Examples

Once configured, the tools are available in Claude Code:

### Storing Memories

```
Use engram_encode to store "My email is john@example.com and I prefer Python" for user_id "demo"
```

This will:
1. Store the content as an immutable episode
2. Extract structured data (emails, phones, URLs via regex)
3. Return the episode ID and extracted data

### Searching Memories

```
Use engram_recall to search for "programming language preferences" for user_id "demo"
```

This returns semantically similar memories ranked by relevance score.

### Verifying Provenance

```
Use engram_verify to verify memory "struct_abc123" for user_id "demo"
```

This traces the memory back to its source episode and explains how it was derived.

### Getting Statistics

```
Use engram_stats for user_id "demo"
```

Returns counts of all memory types and enrichment status.

## Tool Reference

### engram_encode

Store a memory with automatic content extraction.

**Parameters:**
- `content` (required): The text content to store
- `user_id` (required): User identifier for isolation
- `role` (optional): Source role - "user", "assistant", or "system" (default: "user")
- `session_id` (optional): Session identifier for grouping
- `enrich` (optional): If true, use LLM extraction for richer data (default: false)

**Returns:** JSON with episode_id, structured_id, and extracted data

### engram_recall

Search memories by semantic similarity.

**Parameters:**
- `query` (required): Natural language search query
- `user_id` (required): User identifier for isolation
- `limit` (optional): Maximum results (default: 10)
- `min_confidence` (optional): Minimum confidence threshold (0.0-1.0)
- `memory_types` (optional): Comma-separated list of types to search (episodic, structured, semantic, procedural, working)
- `include_sources` (optional): Include source episode details (default: false)

**Returns:** JSON array of matching memories with scores

### engram_verify

Verify a memory's provenance.

**Parameters:**
- `memory_id` (required): Memory ID (must start with struct_, sem_, or proc_)
- `user_id` (required): User identifier for isolation

**Returns:** JSON with verification result, source episodes, and explanation

### engram_stats

Get memory statistics for a user.

**Parameters:**
- `user_id` (required): User identifier

**Returns:** JSON with counts of all memory types

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  Claude Code                     │
│    (or other MCP-compatible client)             │
└────────────────────┬────────────────────────────┘
                     │ STDIO (JSON-RPC)
                     ▼
┌─────────────────────────────────────────────────┐
│              Engram MCP Server                   │
│  ┌─────────────┐  ┌─────────────┐              │
│  │engram_encode│  │engram_recall│  ...          │
│  └──────┬──────┘  └──────┬──────┘              │
│         │                │                       │
│         ▼                ▼                       │
│  ┌──────────────────────────────────────────┐  │
│  │            EngramService                  │  │
│  │  (encode, recall, verify operations)     │  │
│  └────────────────────┬─────────────────────┘  │
└───────────────────────┼─────────────────────────┘
                        │
                        ▼
              ┌─────────────────┐
              │     Qdrant      │
              │ (Vector Store)  │
              └─────────────────┘
```

## Running Manually

For testing or debugging, you can run the MCP server directly:

```bash
# Run with uv
uv run --extra mcp python -m engram.mcp

# The server communicates via STDIO (JSON-RPC)
# It will wait for MCP protocol messages on stdin
```

## Troubleshooting

### "Connection refused" to Qdrant

Ensure Qdrant is running:

```bash
docker run -p 6333:6333 qdrant/qdrant
```

### "mcp module not found"

Install with the MCP extra:

```bash
uv sync --extra mcp
```

### Tool calls failing silently

Check the Engram logs. The MCP server logs to stderr (stdout is reserved for MCP protocol):

```bash
ENGRAM_LOG_LEVEL=DEBUG uv run --extra mcp python -m engram.mcp
```

### Memory not persisting

Ensure you're using consistent `user_id` values across encode and recall operations.
