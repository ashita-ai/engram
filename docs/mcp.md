# MCP Server

Engram provides an MCP server for Claude Code integration.

## Setup

```bash
# 1. Start Qdrant
docker compose up -d

# 2. Install dependencies
uv sync --extra mcp

# 3. Add to Claude Code
claude mcp add engram -- uv run --directory /path/to/engram python -m engram.mcp

# 4. Restart Claude Code
```

Replace `/path/to/engram` with your actual path.

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
| `ENGRAM_OPENAI_API_KEY` | - | Required if using openai embeddings |
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

## Troubleshooting

**"Connection refused"**
```bash
docker compose up -d  # Ensure Qdrant is running
curl http://localhost:6333/health  # Should return {"status":"ok"}
```

**"mcp module not found"**
```bash
uv sync --extra mcp  # Install MCP dependencies
```

**Tool calls failing**
```bash
ENGRAM_LOG_LEVEL=DEBUG uv run python -m engram.mcp  # Run manually with debug
```
