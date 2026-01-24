<p align="center">
  <img src="assets/engram.jpg" alt="Engram Logo">
</p>

# Engram

**Memory you can trust.**

A memory system for AI applications that preserves ground truth, tracks confidence, and prevents hallucinations.

## The Problem

AI memory systems have an accuracy crisis. Recent benchmarks show:

> "All systems achieve answer accuracies below 56%, with hallucination rate and omission rate remaining high."
>
> — [HaluMem: Hallucinations in LLM Memory Systems](https://arxiv.org/html/2511.03506)

The fundamental issue: once source data is lost, errors cannot be corrected.

## The Solution

Engram preserves ground truth and tracks confidence:

1. **Store first, derive later** — Raw conversations stored verbatim. LLM extraction happens in background where errors can be caught.
2. **Track confidence** — Every memory carries a composite score: extraction method + corroboration + recency + verification.
3. **Verify on retrieval** — Applications filter by confidence. High-stakes queries use only trusted memories.
4. **Enable recovery** — Derived memories trace to sources. Errors can be corrected by re-deriving.

## Quick Start

### Installation

```bash
# Clone and install
git clone https://github.com/ashita-ai/engram.git
cd engram
uv sync --extra dev

# Start Qdrant (vector database)
docker run -p 6333:6333 qdrant/qdrant
```

### Python SDK

```python
from engram.service import EngramService

async with EngramService.create() as engram:
    # Store interaction (immediate, preserves ground truth)
    result = await engram.encode(
        content="My email is john@example.com",
        role="user",
        user_id="user_123",
    )
    print(f"Episode: {result.episode.id}")
    print(f"Emails extracted: {result.structured.emails}")  # ["john@example.com"]

    # Retrieve with confidence filtering
    memories = await engram.recall(
        query="What's the user's email?",
        user_id="user_123",
        min_confidence=0.7,
    )

    # Verify any memory back to source
    verified = await engram.verify(memories[0].memory_id, user_id="user_123")
    print(verified.explanation)
```

### REST API

```bash
# Start the server
uv run uvicorn engram.api.app:app --port 8000

# Encode a memory
curl -X POST http://localhost:8000/api/v1/encode \
  -H "Content-Type: application/json" \
  -d '{"content": "My email is john@example.com", "role": "user", "user_id": "user_123"}'

# Recall memories
curl -X POST http://localhost:8000/api/v1/recall \
  -H "Content-Type: application/json" \
  -d '{"query": "email", "user_id": "user_123"}'

# Batch encode (bulk import)
curl -X POST http://localhost:8000/api/v1/encode/batch \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_123",
    "items": [
      {"content": "Message 1", "role": "user"},
      {"content": "Response 1", "role": "assistant"}
    ]
  }'
```

## Memory Types

| Type | Confidence | Purpose |
|------|------------|---------|
| **Working** | N/A | Current session context (in-memory, volatile) |
| **Episodic** | Highest | Ground truth, verbatim storage, immutable |
| **Structured** | High | Per-episode extraction (emails, phones, URLs, negations) |
| **Semantic** | Variable | Cross-episode knowledge synthesis (LLM-derived) |
| **Procedural** | Variable | Behavioral patterns and preferences |

### Memory Flow

```
Episode (raw, immutable)
    │
    ├──→ Structured (per-episode: emails, phones, negations)
    │
    └──→ Semantic (LLM consolidation: N episodes → 1 summary)
              │
              └──→ Procedural (behavioral synthesis)
```

## REST API Reference

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/encode` | Store a memory and extract facts |
| `POST` | `/encode/batch` | Bulk import multiple memories |
| `POST` | `/recall` | Semantic search across all memory types |
| `GET` | `/memories/{id}` | Get a specific memory by ID |
| `GET` | `/memories` | List memories with filters |
| `DELETE` | `/memories/{id}` | Delete a memory (with cascade options) |
| `PATCH` | `/memories/{id}` | Update memory content/metadata |
| `GET` | `/memories/{id}/sources` | Trace memory to source episodes |
| `GET` | `/memories/{id}/verify` | Verify memory with explanation |
| `GET` | `/memories/{id}/provenance` | Full derivation chain |

### Workflow Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/workflows/consolidate` | Episodes → Semantic memories |
| `POST` | `/workflows/decay` | Apply confidence decay |
| `POST` | `/workflows/promote` | Semantic → Procedural synthesis |
| `POST` | `/workflows/structure` | LLM extraction for single episode |
| `POST` | `/workflows/structure/batch` | LLM extraction for multiple episodes |

### Additional Features

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/memories/{id}/links` | Create memory links |
| `GET` | `/memories/{id}/links` | List memory links |
| `DELETE` | `/memories/{id}/links/{target}` | Remove a link |
| `POST` | `/conflicts/detect` | Detect contradictions |
| `GET` | `/conflicts` | List detected conflicts |
| `POST` | `/webhooks` | Register event webhooks |
| `GET` | `/memories/{id}/history` | Memory change history |
| `DELETE` | `/users/{user_id}/memories` | GDPR erasure |

### Session Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/sessions` | List sessions for a user |
| `GET` | `/sessions/{session_id}` | Get session details with episodes |
| `DELETE` | `/sessions/{session_id}` | Delete session (with cascade options) |

## Confidence Scoring

Confidence is a composite score:

| Factor | Weight | Description |
|--------|--------|-------------|
| Extraction method | 50% | verbatim=1.0, regex=0.9, LLM=0.6 |
| Corroboration | 25% | Number of supporting sources |
| Recency | 15% | How recently confirmed |
| Verification | 10% | Format validation passed |

Every score is auditable: *"0.73 because: extracted (0.9 base), 3 sources, confirmed 2 months ago."*

## Configuration

Environment variables (prefix: `ENGRAM_`):

| Variable | Default | Description |
|----------|---------|-------------|
| `ENV` | `development` | Environment: `development`, `production`, or `test` |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant connection |
| `EMBEDDING_PROVIDER` | `fastembed` | Embedding backend |
| `AUTH_ENABLED` | auto | Enable Bearer token auth (auto: true in production) |
| `AUTH_SECRET_KEY` | - | Secret key for auth (required in production) |
| `RATE_LIMIT_ENABLED` | `false` | Enable rate limiting |
| `BATCH_ENCODE_MAX_ITEMS` | `100` | Max batch size |

**Security Notes:**
- In production (`ENGRAM_ENV=production`), auth is enabled by default
- Using the default secret key in production will raise an error
- Generate a secret key: `python -c "import secrets; print(secrets.token_hex(32))"`

See [docs/development.md](docs/development.md) for full configuration reference.

## Development

```bash
# Run tests
uv run pytest tests/ -v --no-cov

# Code quality
uv run ruff check src/engram/
uv run mypy src/engram/

# Pre-commit hooks
uv run pre-commit install
uv run pre-commit run --all-files
```

## Claude Code Integration

Engram provides an MCP server for direct integration with Claude Code:

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

This exposes four tools: `engram_encode`, `engram_recall`, `engram_verify`, and `engram_stats`.

See [docs/mcp.md](docs/mcp.md) for full setup instructions.

## Documentation

- [Architecture](docs/architecture.md) — Memory types, data flow, storage design
- [Development Guide](docs/development.md) — Setup, configuration, workflow
- [API Reference](docs/api.md) — Detailed endpoint documentation
- [MCP Integration](docs/mcp.md) — Claude Code and MCP client setup

## Status

Beta. Core functionality complete with comprehensive test coverage (800+ tests).

## License

MIT
