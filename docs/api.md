# REST API Reference

Engram exposes a REST API for memory operations. All endpoints are prefixed with `/api/v1`.

## Authentication

Currently no authentication is required. Multi-tenancy is enforced via `user_id` and optional `org_id` parameters.

---

## Memory Endpoints

### POST /encode

Store content as an episode and extract structured data.

**Request Body:**
```json
{
  "content": "My email is user@example.com",
  "role": "user",
  "user_id": "user_123",
  "org_id": "org_456",
  "session_id": "session_abc",
  "importance": 0.8,
  "enrich": false
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `content` | string | Yes | - | Text content to encode |
| `role` | string | No | `"user"` | Speaker role: `user`, `assistant`, `system` |
| `user_id` | string | Yes | - | User ID for multi-tenancy |
| `org_id` | string | No | `null` | Optional organization ID |
| `session_id` | string | No | `null` | Optional session grouping |
| `importance` | float | No | auto | Importance score 0.0-1.0 (auto-calculated if omitted) |
| `enrich` | bool/string | No | `false` | LLM enrichment: `false`=regex only, `true`=sync LLM, `"background"`=async |

**Response (201 Created):**
```json
{
  "episode": {
    "id": "ep_abc123",
    "content": "My email is user@example.com",
    "role": "user",
    "user_id": "user_123",
    "org_id": "org_456",
    "session_id": "session_abc",
    "importance": 0.6,
    "created_at": "2025-01-15T10:30:00Z"
  },
  "structured": {
    "id": "struct_xyz789",
    "source_episode_id": "ep_abc123",
    "mode": "fast",
    "enriched": false,
    "emails": ["user@example.com"],
    "phones": [],
    "urls": [],
    "confidence": 0.9
  },
  "extract_count": 1
}
```

---

### POST /recall

Search memories by semantic similarity.

**Request Body:**
```json
{
  "query": "What is the user's contact information?",
  "user_id": "user_123",
  "org_id": "org_456",
  "limit": 10,
  "min_confidence": 0.7,
  "min_selectivity": 0.0,
  "memory_types": ["episodic", "structured", "semantic"],
  "include_sources": true,
  "follow_links": true,
  "max_hops": 2,
  "freshness": "best_effort",
  "as_of": "2025-01-01T00:00:00Z",
  "expand_query": false,
  "diversity": 0.0
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `query` | string | Yes | - | Natural language query |
| `user_id` | string | Yes | - | User ID for isolation |
| `org_id` | string | No | `null` | Optional org filter |
| `limit` | int | No | `10` | Max results (1-100) |
| `min_confidence` | float | No | `null` | Minimum confidence threshold |
| `min_selectivity` | float | No | `0.0` | Minimum selectivity for semantic memories |
| `memory_types` | array | No | all | Types to search: `episodic`, `structured`, `semantic`, `procedural`, `working` |
| `include_sources` | bool | No | `false` | Include source episode details |
| `follow_links` | bool | No | `false` | Enable multi-hop reasoning via related_ids |
| `max_hops` | int | No | `2` | Max link traversal depth (1-5) |
| `freshness` | string | No | `"best_effort"` | `best_effort` or `fresh_only` |
| `as_of` | datetime | No | `null` | Bi-temporal query: only memories derived before this time |
| `expand_query` | bool | No | `false` | Use LLM to expand query with related terms |
| `diversity` | float | No | `0.0` | Result diversity via MMR reranking (0.0-1.0) |

**Response (200 OK):**
```json
{
  "query": "What is the user's contact information?",
  "results": [
    {
      "memory_type": "structured",
      "content": "email: user@example.com",
      "score": 0.85,
      "confidence": 0.9,
      "memory_id": "struct_xyz789",
      "source_episode_id": "ep_abc123",
      "source_episode_ids": ["ep_abc123"],
      "source_episodes": [
        {
          "id": "ep_abc123",
          "content": "My email is user@example.com",
          "role": "user",
          "timestamp": "2025-01-15T10:30:00Z"
        }
      ],
      "related_ids": [],
      "hop_distance": 0,
      "staleness": "fresh",
      "consolidated_at": null,
      "metadata": {}
    }
  ],
  "count": 1
}
```

---

### GET /memories/stats

Get memory statistics for a user.

**Query Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `user_id` | string | Yes | User ID |
| `org_id` | string | No | Optional org filter |

**Response (200 OK):**
```json
{
  "user_id": "user_123",
  "org_id": null,
  "counts": {
    "episodes": 150,
    "structured": 145,
    "semantic": 12,
    "procedural": 1
  },
  "confidence": {
    "structured_avg": 0.87,
    "structured_min": 0.6,
    "structured_max": 0.95,
    "semantic_avg": 0.72
  },
  "pending_consolidation": 23
}
```

---

### GET /memories/working

Get current session's working memory (volatile, in-memory only).

**Response (200 OK):**
```json
{
  "episodes": [
    {
      "id": "ep_abc123",
      "content": "Hello",
      "role": "user",
      "user_id": "user_123",
      "importance": 0.5,
      "created_at": "2025-01-15T10:30:00Z"
    }
  ],
  "count": 1
}
```

---

### DELETE /memories/working

Clear working memory for current session.

**Response:** 204 No Content

---

### GET /memories/{memory_id}/sources

Get source episodes for a derived memory.

**Path Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `memory_id` | string | Memory ID (must start with `struct_`, `sem_`, or `proc_`) |

**Query Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `user_id` | string | Yes | User ID for verification |

**Response (200 OK):**
```json
{
  "memory_id": "sem_xyz789",
  "memory_type": "semantic",
  "sources": [
    {
      "id": "ep_abc123",
      "content": "My email is user@example.com",
      "role": "user",
      "user_id": "user_123",
      "importance": 0.6,
      "created_at": "2025-01-15T10:30:00Z"
    }
  ],
  "count": 1
}
```

---

### GET /memories/{memory_id}/verify

Verify a memory against its source episodes with explanation.

**Path Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `memory_id` | string | Memory ID (must start with `struct_`, `sem_`, or `proc_`) |

**Query Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `user_id` | string | Yes | User ID for verification |

**Response (200 OK):**
```json
{
  "memory_id": "struct_xyz789",
  "memory_type": "structured",
  "content": "email: user@example.com",
  "verified": true,
  "source_episodes": [
    {
      "id": "ep_abc123",
      "content": "My email is user@example.com",
      "role": "user",
      "timestamp": "2025-01-15T10:30:00Z"
    }
  ],
  "extraction_method": "extracted",
  "confidence": 0.9,
  "explanation": "Pattern-matched email from source episode(s). High confidence due to deterministic extraction."
}
```

---

### DELETE /memories/{memory_id}

Delete a specific memory by ID.

**Path Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `memory_id` | string | Memory ID (prefix determines type: `ep_`, `struct_`, `sem_`, `proc_`) |

**Query Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `user_id` | string | Yes | User ID for verification |
| `org_id` | string | No | Optional org filter |

**Response:** 204 No Content

**Errors:**
- 400: Invalid memory ID format
- 404: Memory not found

---

### DELETE /users/{user_id}/memories

Delete all memories for a user (GDPR right to erasure).

**Path Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `user_id` | string | User whose memories to delete |

**Query Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `org_id` | string | No | Optional org filter |

**Response (200 OK):**
```json
{
  "user_id": "user_123",
  "org_id": null,
  "deleted_counts": {
    "episodic": 150,
    "structured": 145,
    "semantic": 12,
    "procedural": 1
  },
  "total_deleted": 308
}
```

---

## Workflow Endpoints

Manually trigger background workflows for consolidation, decay, and memory processing.

### POST /workflows/consolidate

Trigger memory consolidation workflow. Consolidates episodic memories into semantic memories.

**Request Body:**
```json
{
  "user_id": "user_123",
  "org_id": "org_456",
  "consolidation_passes": 1,
  "similarity_threshold": 0.7,
  "async_execution": false
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `user_id` | string | Yes | - | User ID for isolation |
| `org_id` | string | No | `null` | Optional org filter |
| `consolidation_passes` | int | No | `1` | Number of LLM passes for refinement (1-5) |
| `similarity_threshold` | float | No | `0.7` | Threshold for memory linking (0.0-1.0) |
| `async_execution` | bool | No | `false` | Run in background, return workflow ID |

**Response (200 OK):**
```json
{
  "workflow_id": null,
  "episodes_processed": 25,
  "semantic_memories_created": 3,
  "links_created": 8,
  "compression_ratio": 8.33
}
```

**What it does:**
1. Fetches unconsolidated StructuredMemories for the user
2. Groups related memories by semantic similarity
3. Uses LLM to synthesize N structured memories into 1 semantic memory
4. Creates bidirectional links between related memories
5. Marks structured memories as consolidated

---

### POST /workflows/decay

Trigger memory decay workflow. Applies time-based confidence decay.

**Request Body:**
```json
{
  "user_id": "user_123",
  "org_id": "org_456",
  "run_promotion": true,
  "async_execution": false
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `user_id` | string | Yes | - | User ID for isolation |
| `org_id` | string | No | `null` | Optional org filter |
| `run_promotion` | bool | No | `true` | Run promotion workflow after decay |
| `async_execution` | bool | No | `false` | Run in background |

**Response (200 OK):**
```json
{
  "workflow_id": null,
  "memories_updated": 45,
  "memories_archived": 3,
  "memories_deleted": 1,
  "procedural_promoted": 2
}
```

**What it does:**
1. Calculates new confidence scores based on time since last access
2. Updates confidence for all memories
3. Archives memories below archive threshold (default 0.1)
4. Deletes memories below delete threshold (default 0.01)
5. Optionally runs promotion to update procedural memory

---

### POST /workflows/promote

Trigger promotion/synthesis workflow. Promotes semantic memories to procedural.

**Request Body:**
```json
{
  "user_id": "user_123",
  "org_id": "org_456",
  "async_execution": false
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `user_id` | string | Yes | - | User ID for isolation |
| `org_id` | string | No | `null` | Optional org filter |
| `async_execution` | bool | No | `false` | Run in background |

**Response (200 OK):**
```json
{
  "workflow_id": null,
  "semantics_analyzed": 12,
  "procedural_created": true,
  "procedural_updated": false,
  "procedural_id": "proc_abc123"
}
```

**What it does:**
1. Fetches all semantic memories for the user
2. Uses LLM to synthesize behavioral patterns
3. Creates or updates ONE procedural memory per user
4. Links procedural to source semantic memories

---

### POST /workflows/structure

Trigger structure workflow for a single episode. Extracts structured data via LLM.

**Request Body:**
```json
{
  "episode_id": "ep_abc123",
  "user_id": "user_123",
  "model": "openai:gpt-4o-mini",
  "skip_if_structured": true
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `episode_id` | string | Yes | - | Episode ID to structure |
| `user_id` | string | Yes | - | User ID for verification |
| `model` | string | No | default | Optional model override |
| `skip_if_structured` | bool | No | `true` | Skip if already has StructuredMemory |

**Response (200 OK):**
```json
{
  "episode_id": "ep_abc123",
  "structured_memory_id": "struct_xyz789",
  "extracts_count": 5,
  "deterministic_count": 2,
  "llm_count": 3,
  "processing_time_ms": 1250,
  "skipped": false
}
```

**What it does:**
1. Runs deterministic extraction (emails, phones, URLs) - 0.9 confidence
2. Runs LLM extraction (dates, people, preferences, negations) - 0.8 confidence
3. Creates StructuredMemory linked to episode
4. Returns extraction counts and timing

---

### POST /workflows/structure/batch

Trigger batch structure workflow for multiple episodes.

**Request Body:**
```json
{
  "user_id": "user_123",
  "org_id": "org_456",
  "limit": 50,
  "model": "openai:gpt-4o-mini",
  "async_execution": false
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `user_id` | string | Yes | - | User ID for isolation |
| `org_id` | string | No | `null` | Optional org filter |
| `limit` | int | No | `null` | Max episodes to process (null = all unstructured) |
| `model` | string | No | default | Optional model override |
| `async_execution` | bool | No | `false` | Run in background |

**Response (200 OK):**
```json
{
  "workflow_id": null,
  "episodes_processed": 15,
  "total_extracts": 42,
  "results": [
    {
      "episode_id": "ep_001",
      "structured_memory_id": "struct_001",
      "extracts_count": 3,
      "deterministic_count": 1,
      "llm_count": 2,
      "processing_time_ms": 850,
      "skipped": false
    }
  ]
}
```

---

### GET /workflows/{workflow_id}/status

Get status of an async workflow execution.

**Path Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `workflow_id` | string | Workflow execution ID |

**Response (200 OK):**
```json
{
  "workflow_id": "wf_abc123",
  "workflow_type": "consolidation",
  "state": "completed",
  "started_at": "2025-01-15T10:30:00Z",
  "completed_at": "2025-01-15T10:35:00Z",
  "error": null,
  "result": {
    "episodes_processed": 25,
    "semantic_memories_created": 3
  }
}
```

| State | Description |
|-------|-------------|
| `pending` | Workflow queued but not started |
| `running` | Workflow currently executing |
| `completed` | Workflow finished successfully |
| `failed` | Workflow encountered an error |
| `cancelled` | Workflow was cancelled |

---

## System Endpoints

### GET /health

Check service health status.

**Response (200 OK):**
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "storage_connected": true
}
```

| Status | Description |
|--------|-------------|
| `healthy` | All systems operational |
| `degraded` | Some features unavailable |
| `unhealthy` | Service not ready |

---

## Error Responses

All endpoints return standard error responses:

**400 Bad Request:**
```json
{
  "detail": "Invalid memory ID format: xyz. Expected prefix: ep_, struct_, sem_, or proc_"
}
```

**404 Not Found:**
```json
{
  "detail": "Memory not found: ep_nonexistent"
}
```

**500 Internal Server Error:**
```json
{
  "detail": "An internal error occurred while processing the request"
}
```

**503 Service Unavailable:**
```json
{
  "detail": "Service not initialized"
}
```
