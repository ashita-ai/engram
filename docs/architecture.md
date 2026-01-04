# Architecture

Engram is a memory system for LLM applications built on Pydantic AI with durable execution.

## Stack

```
┌─────────────────────────────────────────────────────────────┐
│                      Application                             │
├─────────────────────────────────────────────────────────────┤
│                      Engram API                              │
│         encode() / recall() / consolidate() / decay()        │
├─────────────────────────────────────────────────────────────┤
│                     Pydantic AI                              │
│              Agents with structured outputs                   │
├─────────────────────────────────────────────────────────────┤
│               Durable Execution (pick one)                   │
│         ┌──────────┬──────────┬──────────┐                   │
│         │ Temporal │  DBOS    │  Prefect │                   │
│         └──────────┴──────────┴──────────┘                   │
├─────────────────────────────────────────────────────────────┤
│                       Qdrant                                 │
│              Vector storage for all memory types             │
└─────────────────────────────────────────────────────────────┘
```

## Durable Execution Options

Engram's `consolidate()` operation runs LLM extraction in the background. For production, wrap agents with durable execution:

### Option 1: DBOS (Recommended for simplicity)

Lightweight, in-process. Just needs PostgreSQL or SQLite.

```python
from pydantic_ai import Agent
from pydantic_ai.ext.dbos import DBOSAgent
from pydantic import BaseModel

class ConsolidationResult(BaseModel):
    facts: list[Fact]
    confidence_scores: dict[str, float]

consolidation_agent = Agent(
    "openai:gpt-4",
    result_type=ConsolidationResult,
)

# Wrap for durability - survives crashes, resumes from checkpoint
durable_agent = DBOSAgent(consolidation_agent)

async def consolidate(episodes: list[Episode]):
    result = await durable_agent.run(
        f"Extract facts from: {format_episodes(episodes)}"
    )
    await store_facts(result.data.facts)
```

**Pros**: No extra infrastructure, automatic checkpointing, in-process
**Cons**: Single-node only

### Option 2: Temporal (Enterprise)

Full workflow orchestration. Requires Temporal server.

```python
from pydantic_ai import Agent
from pydantic_ai.ext.temporal import TemporalAgent

consolidation_agent = Agent(
    "openai:gpt-4",
    result_type=ConsolidationResult,
)

durable_agent = TemporalAgent(consolidation_agent)

# Same API - Temporal handles durability transparently
async def consolidate(episodes: list[Episode]):
    result = await durable_agent.run(
        f"Extract facts from: {format_episodes(episodes)}"
    )
    await store_facts(result.data.facts)
```

**Pros**: Enterprise-grade, distributed, full visibility
**Cons**: Requires Temporal server/cloud

### Option 3: Prefect (Orchestration + UI)

Full orchestration with dashboard. Cloud or self-hosted.

```python
from pydantic_ai import Agent
from pydantic_ai.ext.prefect import PrefectAgent

consolidation_agent = Agent(
    "openai:gpt-4",
    result_type=ConsolidationResult,
)

durable_agent = PrefectAgent(consolidation_agent)

async def consolidate(episodes: list[Episode]):
    result = await durable_agent.run(
        f"Extract facts from: {format_episodes(episodes)}"
    )
    await store_facts(result.data.facts)
```

**Pros**: Great UI, cloud platform, scheduling built-in
**Cons**: More setup than DBOS

### Comparison

| Feature | DBOS | Temporal | Prefect |
|---------|------|----------|---------|
| Infrastructure | PostgreSQL/SQLite | Temporal server | Prefect server/cloud |
| Complexity | Low | Medium | Medium |
| Distributed | No | Yes | Yes |
| UI/Dashboard | No | Yes | Yes |
| Best for | MVP, simple deployments | Enterprise, complex workflows | Teams wanting visibility |

## Design Principles

### 1. Ground Truth Preservation

Raw interactions stored as immutable episodic memories. All derived knowledge traces back to source. Errors are correctable.

### 2. Confidence Tracking

Every memory carries provenance:

| Source Type | Confidence | Method |
|-------------|------------|--------|
| `verbatim` | 100% | Direct quote |
| `extracted` | High | Pattern matching (deterministic) |
| `inferred` | Variable | LLM-derived |

### 3. Deferred Processing

Expensive LLM work is batched and deferred:

| Operation | When | Cost |
|-----------|------|------|
| Episode storage | Immediate | Low (embed + store) |
| Factual extraction | Immediate | Low (pattern matching) |
| Semantic inference | Background | Medium (LLM, durable) |
| Consolidation | Scheduled | Medium (LLM, batched) |
| Decay | Scheduled | Low (math) |

## Memory Types

| Type | Mutability | Decay | Confidence | Storage |
|------|------------|-------|------------|---------|
| **Episodic** | Immutable | Fast | High (verbatim) | Qdrant |
| **Factual** | Immutable | Slow | High (extracted) | Qdrant |
| **Semantic** | Mutable | Slow | Variable | Qdrant |
| **Procedural** | Mutable | Very slow | Variable | Qdrant |
| **Working** | Volatile | N/A | N/A | In-memory |
| **Scratchpad** | Mutable | N/A | N/A | Qdrant |

## Data Flow

```
Interaction
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                         ENCODE                               │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│   │ Store       │  │ Pattern     │  │ Score               │ │
│   │ Episode     │  │ Extract     │  │ Importance          │ │
│   │ (verbatim)  │  │ Facts       │  │                     │ │
│   └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘ │
└──────────┼────────────────┼────────────────────┼────────────┘
           │                │                    │
           ▼                ▼                    ▼
      [Episodic]       [Factual]           [importance]
           │                                    │
           └──────────────┬─────────────────────┘
                          │
           ┌──────────────┴──────────────┐
           │  BACKGROUND (Durable)        │
           │  ┌──────────┐ ┌──────────┐  │
           │  │Consolidate│ │  Decay   │  │
           │  │(Pydantic  │ │          │  │
           │  │ AI Agent) │ │          │  │
           │  └─────┬─────┘ └─────┬────┘  │
           └────────┼─────────────┼───────┘
                    │             │
                    ▼             ▼
               [Semantic]    [updated
               [Procedural]   scores]
```

## Storage: Qdrant

All memory types stored in Qdrant with type-specific collections:

```
qdrant/
├── episodic/
│   ├── vectors (embeddings)
│   └── payload (content, timestamp, session_id, importance)
├── factual/
│   ├── vectors (embeddings)
│   └── payload (content, category, source_episode_id, confidence)
├── semantic/
│   ├── vectors (embeddings)
│   └── payload (content, source_episode_ids, confidence)
└── procedural/
    ├── vectors (embeddings)
    └── payload (content, trigger_context, confidence)
```

### Indexing

- **Episodic**: HNSW with time-decay weighting
- **Factual/Semantic**: Standard HNSW
- **Procedural**: HNSW with context filtering

## API

```python
from engram import MemoryStore

memory = MemoryStore(
    qdrant_url="...",
    user_id="user_123",
    durable_backend="dbos"  # or "temporal" or "prefect"
)

# Store (immediate, preserves ground truth)
await memory.encode(interaction, extract_facts=True)

# Retrieve (with confidence filtering)
memories = await memory.recall(
    query="What databases does the user work with?",
    memory_types=["factual", "semantic"],
    min_confidence=0.7,
    include_sources=True
)

# Verify a fact against its source
verified = await memory.verify(fact_id)

# Background operations (run as durable workflows)
await memory.consolidate()
await memory.decay()
```

## Background Operations

### Consolidation

Runs on schedule. Extracts semantic patterns from episodes using Pydantic AI:

```python
from pydantic import BaseModel
from pydantic_ai import Agent

class ExtractedFacts(BaseModel):
    facts: list[str]
    confidence: float
    reasoning: str

consolidation_agent = Agent(
    "openai:gpt-4",
    result_type=ExtractedFacts,
    system_prompt="""
    Extract semantic facts from these conversation episodes.
    Only extract facts with high confidence.
    Explain your reasoning.
    """
)

async def consolidate():
    episodes = await get_unconsolidated_episodes()
    result = await consolidation_agent.run(format_episodes(episodes))

    for fact in result.data.facts:
        await store_semantic(
            content=fact,
            source_episode_ids=[e.id for e in episodes],
            confidence=result.data.confidence
        )
```

### Decay

Updates importance scores based on time and access:

```python
async def decay():
    for memory in await get_all_memories():
        time_factor = exp(-time_since_access / decay_constant)
        access_factor = 1.0 + (0.1 * access_count)
        importance_factor = memory.importance

        new_score = base_score * time_factor * access_factor * importance_factor

        if new_score < deletion_threshold:
            await archive(memory)
        else:
            await update_score(memory, new_score)
```

Decay constants by type:
- Episodic: Fast (days)
- Factual: Slow (months)
- Semantic: Slow (months)
- Procedural: Very slow (years)
