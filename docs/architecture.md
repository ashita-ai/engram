# Architecture

Engram is a memory system for LLM applications built on Pydantic AI with durable execution.

> **Status**: Pre-alpha. This document describes the target architecture. Code examples show proposed APIs, not current implementations.

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
│               Durable Execution                              │
│         ┌──────────────┬──────────────┐                      │
│         │ DBOS (local) │   Temporal   │                      │
│         └──────────────┴──────────────┘                      │
├─────────────────────────────────────────────────────────────┤
│                       Qdrant                                 │
│              Vector storage for all memory types             │
└─────────────────────────────────────────────────────────────┘
```

## Design Principles

### 1. Ground Truth Preservation

Raw interactions stored as immutable episodic memories. All derived knowledge traces back to source. Errors are correctable.

### 2. Confidence Tracking

Every derived memory carries a composite confidence score with full auditability:

```python
class ConfidenceScore(BaseModel):
    value: float                      # Final score 0.0-1.0
    extraction_method: str            # verbatim/extracted/inferred
    extraction_base: float            # Base score from method
    supporting_episodes: int          # Corroboration count
    last_confirmed: datetime          # When last seen/confirmed
    contradictions: int               # Conflicting evidence count
```

**Hybrid formula (configurable weights with sane defaults):**

```python
class ConfidenceWeights(BaseModel):
    extraction: float = 0.50          # How it was extracted
    corroboration: float = 0.25       # How many sources support it
    recency: float = 0.15             # How recently confirmed
    verification: float = 0.10        # Format/validity checks
    decay_half_life_days: int = 365   # How fast unconfirmed facts decay

# Default weights sum to 1.0
confidence = (
    extraction_base * weights.extraction +
    corroboration_score * weights.corroboration +
    recency_score * weights.recency +
    verification_score * weights.verification
)
```

Configure via environment or constructor:
```python
memory = MemoryStore(
    confidence_weights=ConfidenceWeights(
        extraction=0.6,      # Trust extraction method more
        corroboration=0.2,   # Less emphasis on multiple sources
        recency=0.1,
        verification=0.1,
    )
)
```

**Base scores by extraction method:**

| Method | Base Score | Rationale |
|--------|------------|-----------|
| `verbatim` | 1.0 | Exact quote, immutable |
| `extracted` | 0.9 | Deterministic pattern match |
| `inferred` | 0.6 | LLM-derived, uncertain |

**Corroboration boost:** Same fact from 5 episodes scores higher than 1 episode.

**Confidence decay (configurable half-life):**
```python
# Default: ~63% retained after 1 year, configurable per use case
decay_half_life_days: int = 365  # Can be set in ConfidenceWeights
confidence *= exp(-days_since_confirmed / decay_half_life_days)
```

**Why auditable:** Every confidence score can be explained — "0.73 because: extracted (0.9 base), 3 supporting episodes, last confirmed 2 months ago, no contradictions."

### 3. Bi-Temporal Awareness

Every derived fact tracks two timestamps:
- `event_at`: When the fact was true (from the source episode)
- `derived_at`: When we extracted it

Enables "what did we know on date X?" queries for debugging and audit.

### 4. Dynamic Memory Linking

Memories form explicit links to related memories via `related_ids`, enabling multi-hop reasoning:
- "User uses PostgreSQL" → "PostgreSQL is relational" → "User prefers relational DBs"

**Implementation**:
- `related_ids` field on SemanticMemory and ProceduralMemory stores links
- During consolidation, the LLM identifies relationships between memories
- Links are stored bidirectionally and persisted to storage
- `follow_links` parameter in recall() traverses links for multi-hop reasoning
- `_find_matching_memory` uses exact, normalized, and substring matching to resolve link content

Inspired by A-MEM research showing 2x improvement on multi-hop reasoning benchmarks.

### 5. Selectivity Through Consolidation

Memories start broad and become selective through repeated consolidation passes. Initial extraction captures many associations; subsequent passes prune weak or contradicted ones.

The `selectivity_score` on SemanticMemory is directly inspired by dynamic engram research ([Tomé et al., Nature Neuroscience 2024](https://www.nature.com/articles/s41593-023-01551-w)): biological engrams transition from unselective → selective over ~12 hours via inhibitory plasticity. Engram models this with a score that increases as memories survive consolidation passes.

### 6. Negation Tracking

Track what is explicitly NOT true to prevent false matches. When a user corrects a misunderstanding or explicitly negates something, we store that negation as a `NegationFact`.

This is an **engineering construct** for storing semantic negations (e.g., "User does NOT use MongoDB"), not an implementation of neural inhibition mechanisms.

### 7. Deferred Processing

Expensive LLM work is batched and deferred:

| Operation | When | Cost |
|-----------|------|------|
| Episode storage | Immediate | Low (embed + store) |
| Factual extraction | Immediate | Low (pattern matching) |
| Semantic inference | Background | Medium (LLM, durable) |
| Consolidation | Scheduled | Medium (LLM, batched) |
| Decay | Scheduled | Low (math) |

### 8. Buffer Promotion

Memory types form a hierarchy with explicit promotion/demotion:

```
Working (volatile) → Episodic (fast decay) → Semantic (slow decay) → Procedural (very slow)
```

**Promotion implementation** (`run_promotion` workflow):
- Analyzes semantic memories for behavioral patterns (keywords: "prefers", "always", "tends to", etc.)
- Promotes when: selectivity_score >= 0.5, consolidation_passes >= 2, confidence >= 0.7
- Creates ProceduralMemory with trigger_context extracted from content
- Links procedural back to source semantic memory
- Deduplicates to prevent creating duplicate procedural memories

Demotion triggers:
- Low access + time → archive via decay workflow
- Very low confidence → delete via decay workflow

## Memory Types

| Type | Mutability | Decay | Confidence | Purpose |
|------|------------|-------|------------|---------|
| **Episodic** | Immutable | Fast | Highest | Raw interactions, ground truth |
| **Factual** | Immutable | Slow | High | Pattern-extracted facts (emails, dates) |
| **Semantic** | Mutable | Slow | Variable | LLM-inferred knowledge |
| **Procedural** | Mutable | Very slow | Variable | Behavioral patterns, preferences |
| **Negation** | Mutable | Slow | Variable | What is NOT true (negations) |
| **Working** | Volatile | N/A | N/A | Current context (in-memory) |

## Data Models

### Episode (Immutable Ground Truth)

```python
class Episode(BaseModel):
    id: str
    content: str
    timestamp: datetime
    session_id: str
    user_id: str
    importance: float                    # 0.0 - 1.0
    embedding: list[float]
```

### Fact (Deterministically Extracted)

```python
class Fact(BaseModel):
    id: str
    content: str
    category: str                        # email, phone, date, name, etc.
    source_episode_id: str
    event_at: datetime                   # When fact was true
    derived_at: datetime                 # When we extracted it
    confidence: ConfidenceScore          # Composite score with auditability
    embedding: list[float]
```

### SemanticMemory (LLM-Inferred)

```python
class SemanticMemory(BaseModel):
    id: str
    content: str
    source_episode_ids: list[str]        # Provenance
    related_ids: list[str]               # Links to related memories
    event_at: datetime
    derived_at: datetime
    confidence: ConfidenceScore          # Composite score with auditability
    selectivity_score: float             # 0 = broad, 1 = highly selective
    consolidation_passes: int            # How many times refined
    embedding: list[float]
```

### ProceduralMemory (Behavioral Patterns)

```python
class ProceduralMemory(BaseModel):
    id: str
    content: str                         # "User prefers concise responses"
    trigger_context: str                 # When this applies
    source_episode_ids: list[str]
    related_ids: list[str]
    confidence: ConfidenceScore          # Composite score with auditability
    access_count: int                    # Reinforcement through use
    embedding: list[float]
```

### NegationFact (Negative Knowledge)

```python
class NegationFact(BaseModel):
    id: str
    content: str                         # "User does NOT use MongoDB"
    negates_pattern: str                 # Pattern this negates
    source_episode_ids: list[str]
    derived_at: datetime
    confidence: ConfidenceScore          # Composite score with auditability
    embedding: list[float]
```

## Data Flow

```
Interaction
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                         ENCODE                               │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│   │ Store       │  │ Pattern     │  │ Detect              │ │
│   │ Episode     │  │ Extract     │  │ Negations           │ │
│   │ (verbatim)  │  │ Facts       │  │ (negations)         │ │
│   └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘ │
└──────────┼────────────────┼────────────────────┼────────────┘
           │                │                    │
           ▼                ▼                    ▼
      [Episodic]       [Factual]          [Inhibitory]
           │
           └──────────────┬─────────────────────┘
                          │
           ┌──────────────┴──────────────┐
           │  BACKGROUND (Durable)        │
           │  ┌──────────┐ ┌──────────┐  │
           │  │Consolidate│ │  Decay   │  │
           │  │+ Link     │ │          │  │
           │  │+ Prune    │ │          │  │
           │  └─────┬─────┘ └─────┬────┘  │
           └────────┼─────────────┼───────┘
                    │             │
                    ▼             ▼
               [Semantic]    [updated scores]
               [Procedural]  [promotions]
               [Links]       [archival]
```

## Storage: Qdrant

All persistent memory types stored in Qdrant with type-specific collections (Working memory is in-memory only):

```
engram_episodic     → vectors + payload (content, timestamp, session_id, importance)
engram_factual      → vectors + payload (content, category, source_episode_id, event_at, derived_at, confidence)
engram_semantic     → vectors + payload (content, source_episode_ids, related_ids, confidence, selectivity_score)
engram_procedural   → vectors + payload (content, trigger_context, access_count, confidence)
engram_negation     → vectors + payload (content, negates_pattern, source_episode_ids, confidence)
```

### Indexing

- **Episodic**: HNSW with time-decay weighting
- **Factual/Semantic/Negation**: Standard HNSW
- **Procedural**: HNSW with context filtering

## API

```python
from engram import MemoryStore

memory = MemoryStore(
    qdrant_url="http://localhost:6333",
    user_id="user_123",
    durable_backend="dbos"  # or "temporal"
)

# Store (immediate, preserves ground truth)
await memory.encode(interaction, extract_facts=True)

# Retrieve (with confidence filtering, negation filtering)
memories = await memory.recall(
    query="What databases does the user work with?",
    memory_types=["factual", "semantic"],
    min_confidence=0.7,
    min_selectivity=0.5,           # Only well-consolidated memories
    include_sources=True,
    follow_links=True,             # Multi-hop reasoning
)

# Verify a fact against its source
verified = await memory.verify(fact_id)

# Temporal queries
memories = await memory.recall_at(
    query="What did we know about the user?",
    as_of=datetime(2024, 6, 1),    # Point-in-time query
)

# Background operations (run as durable workflows)
await memory.consolidate()         # Extract, link, prune
await memory.decay()               # Update scores, archive, promote
```

## Background Operations

### Consolidation

Runs on schedule. Extracts semantic patterns, builds links, prunes weak associations:

```python
from pydantic import BaseModel
from pydantic_ai import Agent

class ConsolidationResult(BaseModel):
    facts: list[str]
    links: list[tuple[str, str]]       # (memory_id, related_id)
    pruned: list[str]                  # IDs to reduce selectivity
    confidence: float
    reasoning: str

consolidation_agent = Agent(
    "openai:gpt-4o-mini",
    result_type=ConsolidationResult,
    system_prompt="""
    Extract semantic facts from these conversation episodes.
    Identify relationships between memories.
    Flag weak or contradicted associations for pruning.
    Only extract facts with high confidence.
    Explain your reasoning.
    """
)

async def consolidate():
    episodes = await get_unconsolidated_episodes()
    existing = await get_related_memories(episodes)

    result = await consolidation_agent.run(
        format_context(episodes, existing)
    )

    # Store new semantic memories
    for fact in result.data.facts:
        await store_semantic(
            content=fact,
            source_episode_ids=[e.id for e in episodes],
            confidence=result.data.confidence,
            selectivity_score=0.0,     # Starts broad
            consolidation_passes=1,
        )

    # Build links
    for memory_id, related_id in result.data.links:
        await add_link(memory_id, related_id)

    # Increase selectivity for surviving memories
    for memory_id in result.data.pruned:
        await update_selectivity(memory_id, delta=-0.1)

    for memory in existing:
        if memory.id not in result.data.pruned:
            await update_selectivity(memory.id, delta=+0.1)
            await increment_consolidation_passes(memory.id)
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
        elif should_promote(memory):
            await promote(memory)      # Episodic → Semantic, etc.
        else:
            await update_score(memory, new_score)
```

Decay constants by type:
- Episodic: Fast (days)
- Factual: Slow (months)
- Semantic: Slow (months)
- Procedural: Very slow (years)
- Negation: Slow (months)

## Durable Execution

### DBOS (Local Development)

```python
from dbos import DBOS

@DBOS.workflow()
async def consolidate_workflow(episode_ids: list[str]):
    episodes = await get_episodes(episode_ids)
    result = await consolidation_agent.run(format_episodes(episodes))
    await store_results(result.data)
```

### Temporal (Production)

```python
from temporalio import workflow, activity

@activity.defn
async def extract_and_link(episode_ids: list[str]) -> ConsolidationResult:
    episodes = await get_episodes(episode_ids)
    result = await consolidation_agent.run(format_episodes(episodes))
    return result.data

@workflow.defn
class ConsolidateWorkflow:
    @workflow.run
    async def run(self, episode_ids: list[str]):
        result = await workflow.execute_activity(
            extract_and_link,
            episode_ids,
            start_to_close_timeout=timedelta(minutes=5),
        )
        await workflow.execute_activity(
            store_results,
            result,
            start_to_close_timeout=timedelta(seconds=30),
        )
```

## Research Foundations

This architecture is informed by recent research (2024-2025):

| Feature | Research | Key Finding |
|---------|----------|-------------|
| Dynamic linking | [A-MEM](https://arxiv.org/abs/2502.12110) | 2x improvement on multi-hop reasoning |
| Buffer promotion | [Cognitive Workspace](https://arxiv.org/abs/2508.13171) | 58.6% memory reuse vs 0% for naive RAG |
| Selectivity scoring | [Tomé et al.](https://www.nature.com/articles/s41593-023-01551-w) | Engrams transition unselective → selective via inhibitory plasticity |
| Ground truth | [HaluMem](https://arxiv.org/html/2511.03506) | <56% accuracy without source preservation |
