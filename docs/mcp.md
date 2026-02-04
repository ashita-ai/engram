# MCP Server

Engram provides an MCP (Model Context Protocol) server that allows Claude Code and other MCP-compatible clients to use Engram as a memory system directly from within their environment.

## Overview

The MCP server exposes 10 tools:

| Tool | Description |
|------|-------------|
| `engram_encode` | Store a memory with automatic content extraction |
| `engram_recall` | Search memories by semantic similarity |
| `engram_verify` | Verify memory provenance back to sources |
| `engram_stats` | Get memory statistics for a user |
| `engram_delete` | Delete a specific memory |
| `engram_get` | Get a specific memory by ID |
| `engram_consolidate` | Consolidate episodes into semantic memories |
| `engram_promote` | Promote semantic memories to procedural |
| `engram_search` | Advanced search with filters |
| `engram_recall_at` | Time-travel recall (memories as of a point in time) |

## Setup

### Required: OpenAI API Key

Engram uses OpenAI for embeddings and LLM operations. Set **either** of these environment variables — engram accepts both:

- `OPENAI_API_KEY` (standard OpenAI convention)
- `ENGRAM_OPENAI_API_KEY` (engram-prefixed)

If both are set, `ENGRAM_OPENAI_API_KEY` takes precedence.

**Docker gotcha:** Docker's `-e VAR_NAME` syntax (without `=value`) inherits the variable from the **parent process**, not from your shell profile. When Claude Code launches the MCP server, the parent process is Claude Code's Node runtime — which may not have your shell's env vars. To be safe, pass both names so whichever is available gets picked up:

```json
"-e", "OPENAI_API_KEY",
"-e", "ENGRAM_OPENAI_API_KEY"
```

### Docker with DBOS (Recommended)

DBOS provides durable workflow execution with automatic recovery. Consolidation and other workflows survive restarts and failures.

```bash
git clone https://github.com/ashita-ai/engram.git
cd engram

# Start infrastructure (Qdrant + PostgreSQL)
docker compose -f docker-compose.full.yml up -d

# Build the MCP image
docker build -t engram-mcp .
```

Add to your MCP config in `~/.claude.json` (under `mcpServers` or `projects.<path>.mcpServers`):

```json
{
  "engram": {
    "type": "stdio",
    "command": "docker",
    "args": [
      "run", "-i", "--rm",
      "-e", "ENGRAM_QDRANT_URL=http://host.docker.internal:6333",
      "-e", "ENGRAM_USER=your-username",
      "-e", "ENGRAM_EMBEDDING_PROVIDER=openai",
      "-e", "OPENAI_API_KEY",
      "-e", "ENGRAM_OPENAI_API_KEY",
      "-e", "ENGRAM_DURABLE_BACKEND=dbos",
      "-e", "ENGRAM_DATABASE_URL=postgresql://engram:engram@host.docker.internal:5432/engram_dbos",
      "engram-mcp"
    ]
  }
}
```

Both `-e OPENAI_API_KEY` and `-e ENGRAM_OPENAI_API_KEY` are passed so whichever is set in Claude Code's environment gets forwarded to the container. You don't need to hardcode the key value.

**Why DBOS?**
- Workflows (consolidation, decay) are durable and survive restarts
- Automatic recovery of interrupted operations
- Full observability via DBOS Conductor

### Docker Minimal (No Durability)

For quick testing without workflow durability:

```bash
docker compose up -d  # Starts Qdrant only
docker build -t engram-mcp .
```

```json
{
  "engram": {
    "type": "stdio",
    "command": "docker",
    "args": [
      "run", "-i", "--rm",
      "-e", "ENGRAM_QDRANT_URL=http://host.docker.internal:6333",
      "-e", "ENGRAM_EMBEDDING_PROVIDER=openai",
      "-e", "OPENAI_API_KEY",
      "-e", "ENGRAM_OPENAI_API_KEY",
      "engram-mcp"
    ]
  }
}
```

### Local Installation (Alternative)

1. Start infrastructure:

   ```bash
   # Qdrant (required)
   docker run -d -p 6333:6333 -v qdrant_data:/qdrant/storage qdrant/qdrant

   # PostgreSQL for DBOS (recommended)
   docker run -d -p 5432:5432 \
     -e POSTGRES_USER=engram \
     -e POSTGRES_PASSWORD=engram \
     -e POSTGRES_DB=engram_dbos \
     postgres:15-alpine
   ```

2. Install Engram with the MCP extra:

   ```bash
   cd engram
   uv sync --extra mcp
   ```

3. Add to your MCP config in `~/.claude.json`:

```json
{
  "engram": {
    "command": "uv",
    "args": ["run", "--directory", "/path/to/engram", "python", "-m", "engram.mcp"],
    "env": {
      "OPENAI_API_KEY": "sk-proj-YOUR_KEY_HERE",
      "ENGRAM_DURABLE_BACKEND": "dbos",
      "ENGRAM_DATABASE_URL": "postgresql://engram:engram@localhost:5432/engram_dbos"
    }
  }
}
```

Replace `/path/to/engram` with your actual path.

For local (non-Docker) setup, the `env` block sets environment variables directly in the subprocess, so either variable name works.

## Tools

| Tool | Description |
|------|-------------|
| `engram_encode` | Store a memory with automatic extraction |
| `engram_recall` | Search memories semantically |
| `engram_recall_at` | Query memories as of a point in time |
| `engram_search` | Filter/list memories by metadata |
| `engram_verify` | Trace memory back to source |
| `engram_stats` | Get memory statistics |
| `engram_get` | Get specific memory by ID |
| `engram_delete` | Delete a memory |
| `engram_consolidate` | Episodes → semantic memories |
| `engram_promote` | Semantic → procedural |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ENGRAM_QDRANT_URL` | `http://localhost:6333` | Qdrant server URL |
| `ENGRAM_EMBEDDING_PROVIDER` | `fastembed` | `fastembed` or `openai` |
| `ENGRAM_OPENAI_API_KEY` | - | OpenAI API key (or use `OPENAI_API_KEY` — either works) |
| `ENGRAM_USER` | auto | Default user_id (auto-detected from git/system) |

## Usage

Once configured, tools are available in Claude Code:

```
# Store a memory
Use engram_encode to store "My email is john@example.com" for user_id "demo"

# Search memories
Use engram_recall to search for "email address" for user_id "demo"

# Get statistics
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

### engram_delete

Delete a specific memory.

**Parameters:**
- `memory_id` (required): Memory ID to delete
- `user_id` (required): User identifier for isolation

**Returns:** JSON with deletion confirmation

### engram_get

Get a specific memory by ID.

**Parameters:**
- `memory_id` (required): Memory ID to retrieve
- `user_id` (required): User identifier for isolation

**Returns:** JSON with memory details

### engram_consolidate

Consolidate episodes into semantic memories.

**Parameters:**
- `user_id` (required): User identifier
- `force` (optional): Force consolidation even if threshold not met

**Returns:** JSON with consolidation results

### engram_promote

Promote semantic memories to procedural memories.

**Parameters:**
- `user_id` (required): User identifier
- `force` (optional): Force promotion even if threshold not met

**Returns:** JSON with promotion results

### engram_search

Advanced search with filters.

**Parameters:**
- `query` (required): Search query
- `user_id` (required): User identifier
- `memory_types` (optional): Comma-separated list of types
- `session_id` (optional): Filter by session
- `min_confidence` (optional): Minimum confidence threshold
- `limit` (optional): Maximum results

**Returns:** JSON array of matching memories

### engram_recall_at

Time-travel recall - search memories as they existed at a specific point in time.

**Parameters:**
- `query` (required): Search query
- `user_id` (required): User identifier
- `as_of` (required): ISO timestamp (e.g., "2024-01-15T10:30:00Z")
- `limit` (optional): Maximum results

**Returns:** JSON array of memories that existed at the specified time

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
# Docker (recommended)
docker compose up -d
docker exec -it engram-mcp python -m engram.mcp

# Local
uv run python -m engram.mcp

# The server communicates via STDIO (JSON-RPC)
# It will wait for MCP protocol messages on stdin
```

## Troubleshooting

### "The api_key client option must be set" error

This error means Engram can't find your OpenAI API key. Engram accepts either `OPENAI_API_KEY` or `ENGRAM_OPENAI_API_KEY`.

**Docker users:** Pass both env var names (without values) so Docker inherits whichever is set:

```json
"-e", "OPENAI_API_KEY",
"-e", "ENGRAM_OPENAI_API_KEY"
```

Docker's `-e VAR_NAME` syntax (without `=value`) inherits from the **parent process**. When Claude Code launches the MCP server, the parent is Claude Code's Node runtime. Make sure the key is set in either your shell environment (before launching Claude Code) or in the MCP server's `env` block in `~/.claude.json`.

### "Connection refused" to Qdrant

If using Docker:

```bash
docker compose up -d
docker compose logs qdrant  # Check if healthy
```

If running locally:

```bash
docker run -d -p 6333:6333 -v qdrant_data:/qdrant/storage qdrant/qdrant
```

### "mcp module not found" (local installation)

Install with the MCP extra:

```bash
uv sync --extra mcp
```

### Tool calls failing

```bash
# Docker
docker compose logs engram-mcp

# Local
ENGRAM_LOG_LEVEL=DEBUG uv run python -m engram.mcp
```

### Memory not persisting

1. Ensure you're using consistent `user_id` values across encode and recall operations
2. For Docker: data is stored in the `qdrant_data` volume (persists across restarts)
3. For local Qdrant: use `-v qdrant_data:/qdrant/storage` to persist data

## Migrating Existing Data

If you have an existing Qdrant instance with engram data and want to switch to the new Docker Compose setup:

### Option 1: Copy volume data (recommended)

```bash
# 1. Find your existing volume
docker volume ls | grep qdrant

# 2. Stop old container
docker stop <old-qdrant-container>

# 3. Copy data to new volume
docker run --rm \
  -v <old-volume-name>:/from \
  -v engram_qdrant_data:/to \
  alpine cp -a /from/. /to/

# 4. Start new compose
cd /path/to/engram
docker compose up -d
```

### Option 2: Use existing volume directly

Edit `docker-compose.yml` to reference your existing volume:

```yaml
volumes:
  qdrant_data:
    external: true
    name: your_existing_volume_name
```

### Option 3: Export/Import via Qdrant API

```bash
# Export from old instance
curl http://localhost:6333/collections/engram_episodic/points/scroll \
  -H "Content-Type: application/json" \
  -d '{"limit": 10000, "with_payload": true, "with_vectors": true}' > episodic.json

# Import to new instance (after docker compose up -d)
# Use Qdrant's upsert API to restore
```
