# Architecture

Engram is a memory system for LLM applications built on Pydantic AI with durable execution.

> **Status**: Pre-alpha. This document describes the target architecture. Code examples show proposed APIs, not current implementations. Durable execution integrations (DBOS, Temporal, Prefect) are planned but not yet built.

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
| `verbatim` | Highest | Direct quote, immutable source |
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
| **Episodic** | Immutable | Fast | Highest (verbatim) | Qdrant |
| **Factual** | Immutable | Slow | High (extracted) | Qdrant |
| **Semantic** | Mutable | Slow | Variable | Qdrant |
| **Procedural** | Mutable | Very slow | Variable | Qdrant |
| **Working** | Volatile | N/A | N/A | In-memory |

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

## Planned Enhancements

Features informed by recent research (2024-2025):

### Dynamic Memory Linking

Inspired by A-MEM (NeurIPS 2025), memories form explicit links to related memories:

```python
class SemanticMemory(BaseModel):
    content: str
    source_episode_ids: list[str]
    confidence: float
    related_ids: list[str] = []  # Links to related memories
```

During consolidation, identify related memories and populate links. Enables multi-hop reasoning: "User uses PostgreSQL" → "PostgreSQL is relational" → "User prefers relational DBs".

### Bi-Temporal Timestamps

Add explicit `derived_at` to track when facts were extracted:

```python
class Fact(BaseModel):
    content: str
    source_episode_id: str
    event_at: datetime      # When the fact was true (from episode)
    derived_at: datetime    # When we extracted it
    confidence: float
```

Enables "what did we know on date X?" queries for debugging and audit.

### Formalized Buffer Promotion

The Cognitive Workspace paper shows 58.6% memory reuse with explicit promotion policies. Our memory types already form a hierarchy:

```
Working (volatile) → Episodic (fast decay) → Semantic (slow decay) → Procedural (very slow)
```

Promotion triggers:
- High-importance episode → immediate semantic extraction
- Repeated pattern → procedural memory
- Low access + time → archive or delete

### Selectivity Through Pruning

Inspired by dynamic engram research ([Tomé et al., Nature Neuroscience 2024](https://www.nature.com/articles/s41593-023-01551-w)), memories start broad and become selective during consolidation:

```python
class SemanticMemory(BaseModel):
    content: str
    source_episode_ids: list[str]
    confidence: float
    selectivity_score: float = 0.0  # 0 = broad, 1 = highly selective
    consolidation_passes: int = 0
```

During consolidation:
1. Initial extraction captures broad associations (low selectivity)
2. Subsequent passes prune weak/contradicted associations
3. Selectivity score increases as memory stabilizes
4. High-selectivity memories are more reliable for retrieval

This mirrors biological findings: engram neurons transition from unselective → selective over ~12 hours, with only 10-40% overlap between encoding and recall populations.

### Inhibitory Facts (Negative Knowledge)

Also from dynamic engram research: inhibitory plasticity is critical for memory selectivity. CCK+ interneurons actively suppress irrelevant associations.

```python
class InhibitoryFact(BaseModel):
    """Track what is explicitly NOT true to prevent false matches."""
    content: str                    # "User does NOT use MongoDB"
    negates_pattern: str            # Pattern this inhibits
    source_episode_ids: list[str]
    confidence: float
```

Use cases:
- User corrects a misunderstanding → create inhibitory fact
- Contradictory information detected → suppress weaker association
- Explicit negation in conversation → "I don't use Windows"

During retrieval, inhibitory facts filter out false positives before ranking