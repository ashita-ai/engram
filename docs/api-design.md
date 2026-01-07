# API Design

Integration patterns and API decisions for Engram.

> **Status**: Pre-alpha. This document describes target API design, not current implementation.

## Design Principles

1. **SDK-first** — Python SDK is the primary interface; REST API is the escape hatch
2. **Explicit over magic** — Users understand exactly what's happening
3. **Framework-agnostic** — No coupling to LangChain, Pydantic AI, etc.
4. **Convenience without commitment** — Helpers that don't lock you in

## SDK Strategy

### Why SDK-First

| Approach | Adoption | Friction |
|----------|----------|----------|
| API-first | Universal | Boilerplate, auth, error handling |
| **SDK-first** | Python/JS devs | Copy-paste works immediately |
| Framework-first | Deep integration | Coupled to framework's fate |

Python dominates AI/ML. `pip install engram` feels familiar (OpenAI pattern).

### Repo Structure

Start monorepo, split later if needed:

```
engram/
├── src/engram/          # Core server
├── sdks/
│   ├── python/          # engram-python
│   └── typescript/      # engram-js (future)
├── examples/
│   ├── fastapi/
│   ├── flask/
│   └── pydantic-ai/
└── docs/
```

Split to separate repos when >5 contributors or independent release cycles needed.

## Multi-Tenancy Model

### Hierarchy

```
Organization (billing, admin)
  └── User (memory isolation)
       └── Session (conversation context)
```

### Why This Structure

- **B2B ready** — enterprises need org-level controls
- **Natural billing** — charge per org, not per user
- **Clear isolation** — user A never sees user B's memories
- **Conversation grouping** — sessions enable "conversations"

### API

```python
memory = Engram(
    org_id="acme_corp",      # Optional, defaults to personal
    user_id="user_123",      # Required
    session_id="chat_abc",   # Optional, for conversation grouping
)
```

Single-user mode: just omit `org_id` and `session_id`.

### Storage

Prefixed keys in Qdrant:
```
{org_id}/{user_id}/{session_id}/{memory_type}/{id}
```

## Core API

### Explicit Pattern (Primary)

The core API is explicit and works everywhere:

```python
from engram import Engram

# Initialize
memory = Engram(
    user_id="user_123",
    qdrant_url="http://localhost:6333",  # Or use ENGRAM_QDRANT_URL env var
)

# Store interaction (immediate, preserves ground truth)
await memory.encode(
    content="My email is john@example.com",
    role="user",
    session_id="chat_abc",  # Optional
)

# Retrieve with confidence filtering
memories = await memory.recall(
    query="What's the user's email?",
    memory_types=["factual", "semantic"],
    min_confidence=0.7,
    limit=10,
)

# Access results
for mem in memories:
    print(mem.content)
    print(mem.confidence.value)      # 0.0-1.0
    print(mem.confidence.explain())  # "0.85: extracted, 3 sources, confirmed yesterday"

# Background operations (run periodically or on-demand)
await memory.consolidate()  # Extract semantics (link building planned)
await memory.decay()        # Update scores, archive stale memories
```

### Context Manager (Convenience)

Framework-agnostic convenience using Python's contextvars:

```python
from engram import memory_context

async def handle_chat(user_id: str, message: str):
    async with memory_context(user_id=user_id) as mem:
        # Encode user message
        await mem.encode(content=message, role="user")

        # Recall relevant context
        context = await mem.recall(query=message)

        # Generate response (your LLM call)
        response = await generate_response(message, context)

        # Encode assistant response
        await mem.encode(content=response, role="assistant")

        return response
        # Auto-commits on exit, handles cleanup
```

### Why Not Decorators?

We provide the context manager primitive. Users can build decorators if they want:

```python
# User's code (not maintained by Engram)
from functools import wraps
from engram import memory_context

def remember(user_id: str):
    def decorator(fn):
        @wraps(fn)
        async def wrapper(*args, **kwargs):
            async with memory_context(user_id=user_id):
                return await fn(*args, **kwargs)
        return wrapper
    return decorator

# Usage
@remember(user_id="user_123")
async def chat(message: str):
    ...
```

**Why we don't ship decorators:**
- Framework-specific decorators require N integrations to maintain
- Framework updates become our breaking changes
- Context manager gives users the tools to build what they need
- If a pattern wins in the community, we can adopt it later

## Async Patterns

### Layer 1: Write-Through with Background Enrichment

`encode()` returns immediately after synchronous work:

```
encode() called
    │
    ├──► Episode stored (sync, fast) ✓
    ├──► Factual extraction (sync, regex) ✓
    │
    └──► Background job queued:
         ├── Semantic extraction (async, LLM)
         ├── Link building (async)
         └── Consolidation (async)
```

```python
# Returns immediately after sync work
await memory.encode(content="I use PostgreSQL daily")

# Facts extracted synchronously (emails, dates, etc.)
# Semantic meaning extracted in background
```

### Layer 2: Freshness Hints (Builds on Layer 1)

Recall results include staleness metadata:

```python
memories = await memory.recall(
    query="What databases does the user know?",
    freshness="best_effort",  # Return what we have NOW
)

for mem in memories:
    print(mem.content)
    print(mem.staleness)  # "fresh" | "consolidating" | "stale"
    print(mem.consolidated_at)
```

### Layer 3: Consolidation Streaming (Builds on Layer 2)

Observe background processing in real-time:

```python
# For UIs that want to show "thinking..." states
async for status in memory.consolidation_stream():
    print(status.event)
    # "extracted_fact: email=john@example.com"
    # "linked: PostgreSQL → relational_databases"
    # "consolidation_complete"
```

### Webhooks (Alternative to Streaming)

For server-to-server integrations:

```python
memory = Engram(
    user_id="user_123",
    webhook_url="https://your-app.com/engram/webhook",
)

# Your webhook receives:
# POST /engram/webhook
# {
#   "event": "consolidation_complete",
#   "user_id": "user_123",
#   "session_id": "chat_abc",
#   "facts_extracted": 3,
#   "links_created": 2
# }
```

## REST API

The SDK wraps a REST API. Direct API access for other languages:

### Endpoints

```
POST   /v1/encode              Store an interaction
POST   /v1/recall              Retrieve memories
POST   /v1/consolidate         Trigger consolidation
POST   /v1/decay               Trigger decay pass

GET    /v1/memories/{id}       Get specific memory
DELETE /v1/memories/{id}       Delete specific memory

GET    /v1/episodes            List episodes
GET    /v1/facts               List facts
GET    /v1/health              Health check
```

### Authentication

```bash
curl -X POST https://api.engram.dev/v1/encode \
  -H "Authorization: Bearer eng_xxxxx" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_123",
    "content": "My email is john@example.com",
    "role": "user"
  }'
```

### Response Format

```json
{
  "success": true,
  "data": {
    "episode_id": "ep_abc123",
    "facts_extracted": [
      {
        "id": "fact_xyz",
        "content": "email=john@example.com",
        "confidence": {
          "value": 0.92,
          "extraction_method": "extracted",
          "supporting_episodes": 1
        }
      }
    ],
    "consolidation_status": "queued"
  }
}
```

## Configuration

### Environment Variables

```bash
# Required
ENGRAM_QDRANT_URL=http://localhost:6333

# Optional
ENGRAM_API_KEY=eng_xxxxx              # For managed service
ENGRAM_EMBEDDING_MODEL=text-embedding-3-small
ENGRAM_CONSOLIDATION_MODEL=gpt-4o-mini
ENGRAM_WEBHOOK_URL=https://...

# Confidence tuning
ENGRAM_CONFIDENCE_EXTRACTION_WEIGHT=0.50
ENGRAM_CONFIDENCE_CORROBORATION_WEIGHT=0.25
ENGRAM_CONFIDENCE_RECENCY_WEIGHT=0.15
ENGRAM_CONFIDENCE_VERIFICATION_WEIGHT=0.10
ENGRAM_CONFIDENCE_DECAY_HALF_LIFE_DAYS=365
```

### Programmatic Configuration

```python
from engram import Engram, ConfidenceWeights

memory = Engram(
    user_id="user_123",
    qdrant_url="http://localhost:6333",
    embedding_model="text-embedding-3-small",
    consolidation_model="gpt-4o-mini",
    confidence_weights=ConfidenceWeights(
        extraction=0.6,
        corroboration=0.2,
        recency=0.1,
        verification=0.1,
        decay_half_life_days=180,
    ),
)
```

## Deployment Models

### Phase 1: Self-Hosted (Docker Compose)

```bash
# Clone and run
git clone https://github.com/ashita-ai/engram
cd engram
cp .env.example .env
# Add your LLM API keys to .env

docker compose up -d
```

Users run their own instance. Lower pressure, no SLA, community finds bugs.

### Phase 2: Managed Service

```python
# Just add API key, no infrastructure
memory = Engram(
    api_key="eng_xxxxx",
    user_id="user_123",
)
```

For users who validated on self-hosted and want managed infrastructure.

## Framework Examples

We provide **examples**, not maintained integrations:

### FastAPI Example

```python
# examples/fastapi/main.py
from fastapi import FastAPI, Depends
from engram import Engram, memory_context

app = FastAPI()

def get_memory(user_id: str):
    return Engram(user_id=user_id)

@app.post("/chat")
async def chat(message: str, user_id: str):
    async with memory_context(user_id=user_id) as mem:
        await mem.encode(content=message, role="user")
        context = await mem.recall(query=message)

        response = await generate_response(message, context)

        await mem.encode(content=response, role="assistant")
        return {"response": response}
```

### Pydantic AI Example

```python
# examples/pydantic-ai/main.py
from pydantic_ai import Agent
from engram import Engram

memory = Engram(user_id="user_123")

agent = Agent(
    "openai:gpt-4o",
    system_prompt="You are a helpful assistant with memory.",
)

@agent.tool
async def remember(content: str) -> str:
    """Store something in memory."""
    await memory.encode(content=content, role="user")
    return "Remembered."

@agent.tool
async def recall(query: str) -> str:
    """Recall relevant memories."""
    memories = await memory.recall(query=query, limit=5)
    return "\n".join(m.content for m in memories)
```

## Versioning

### API Versioning

URL-based versioning:
```
/v1/encode
/v2/encode  (future)
```

### SDK Versioning

Semantic versioning:
```
engram==0.1.0  # Pre-release
engram==1.0.0  # Stable API
engram==1.1.0  # Backwards-compatible additions
engram==2.0.0  # Breaking changes
```

## Error Handling

### SDK Errors

```python
from engram import Engram, EngramError, RateLimitError, ValidationError

try:
    await memory.encode(content="...")
except ValidationError as e:
    print(f"Invalid input: {e.field} - {e.message}")
except RateLimitError as e:
    print(f"Rate limited. Retry after {e.retry_after} seconds")
except EngramError as e:
    print(f"Engram error: {e}")
```

### API Errors

```json
{
  "success": false,
  "error": {
    "code": "validation_error",
    "message": "content is required",
    "field": "content"
  }
}
```

## Rate Limiting

### Self-Hosted

No rate limits by default. Configure via environment:

```bash
ENGRAM_RATE_LIMIT_ENCODE=100/minute
ENGRAM_RATE_LIMIT_RECALL=200/minute
```

### Managed Service

Tier-based limits:

| Tier | Encode | Recall | Consolidation |
|------|--------|--------|---------------|
| Free | 100/min | 200/min | 10/hour |
| Pro | 1000/min | 2000/min | 100/hour |
| Enterprise | Custom | Custom | Custom |
