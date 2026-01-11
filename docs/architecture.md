# Architecture

Engram is a memory system for LLM applications built on Pydantic AI with durable execution.

> **Status**: Pre-alpha. Core APIs (`encode`, `recall`, `verify`, `recall_at`) are implemented. Background workflows (`consolidate`, `decay`, `promote`) are functional.

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
│    ┌──────────────┬──────────────┬──────────────┐            │
│    │ DBOS (local) │   Temporal   │   Prefect    │            │
│    └──────────────┴──────────────┴──────────────┘            │
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
- `derived_at`: When it was extracted

Enables "what was known on date X?" queries for debugging and audit.

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

### 5. Consolidation Strength (Testing Effect)

Memories that are repeatedly involved in consolidation become stronger and more stable. This is based on the **Testing Effect** research ([Roediger & Karpicke 2006](https://pmc.ncbi.nlm.nih.gov/articles/PMC5912918/), [Karpicke & Roediger 2008](https://www.sciencedirect.com/science/article/abs/pii/S1364661310002081)):

> "Repeated remembering strengthens memories much more so than repeated learning."

The `consolidation_strength` field on SemanticMemory tracks how well-established a memory is:
- 0.0: Newly created, not yet reinforced
- 1.0: Highly consolidated, repeatedly reinforced

**Implementation**: During consolidation, `strengthen()` is called when existing memories:
1. Get linked to new memories via semantic similarity
2. Receive LLM-identified links to new memories
3. Undergo evolution (tag/keyword/context updates)

Each call increases `consolidation_strength` by 0.1 and increments `consolidation_passes`.

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

### 8. Hierarchical Memory Compression

Memory types form a compression hierarchy:

```
Working (volatile) → Episodic (immutable ground truth)
                         │
                         ├─→ Factual (pattern-extracted)
                         ├─→ Negation (contradiction tracking)
                         │
                         └─→ Semantic (N episodes → 1 summary via LLM)
                                   │
                                   └─→ Procedural (all semantics → 1 behavioral profile)
```

**Consolidation** (`run_consolidation` workflow):
- Fetches all unsummarized episodes for a user
- Uses LLM to create ONE semantic summary from N episodes (map-reduce for large batches)
- Marks episodes as `summarized=True` and links them via `summarized_into`
- Bidirectional traceability: `episode.summarized_into` ↔ `semantic.source_episode_ids`

**Procedural Synthesis** (`run_synthesis` workflow):
- Fetches ALL semantic memories for a user
- Uses LLM to synthesize ONE behavioral profile per user
- Replaces existing procedural memory (doesn't accumulate)
- Links to source semantics via `source_semantic_ids`

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
    summarized: bool                     # Included in semantic summary?
    summarized_into: str | None          # ID of semantic memory (bidirectional link)
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
    consolidation_strength: float        # 0 = new, 1 = well-established
    consolidation_passes: int            # How many times refined
    embedding: list[float]
```

### ProceduralMemory (Behavioral Patterns)

```python
class ProceduralMemory(BaseModel):
    id: str
    content: str                         # Behavioral profile description
    trigger_context: str                 # When this applies
    source_episode_ids: list[str]        # All episodes (via semantics)
    source_semantic_ids: list[str]       # Semantics synthesized from
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
      [Episodic]       [Factual]          [Negation]
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
engram_semantic     → vectors + payload (content, source_episode_ids, related_ids, confidence, consolidation_strength)
engram_procedural   → vectors + payload (content, trigger_context, access_count, confidence)
engram_negation     → vectors + payload (content, negates_pattern, source_episode_ids, confidence)
```

### Indexing

- **Episodic**: HNSW with time-decay weighting
- **Factual/Semantic/Negation**: Standard HNSW
- **Procedural**: HNSW with context filtering

## Semantic Layer

**Everything is embedded.** Every memory type carries a vector embedding, enabling semantic similarity search across all content. This is the core of how Engram handles natural language queries.

### How It Works

```
User: "My email is user@example.com"
                    │
                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  1. EMBED: Convert text to vector                                        │
│     content → embedding model → [0.12, -0.34, 0.56, ...]                 │
└──────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  2. STORE: Save with embedding                                           │
│     Episode { content, embedding } → Qdrant                              │
└──────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  3. EXTRACT: Pattern match + embed extracted facts                       │
│     "user@example.com" → embedding → Fact { content, embedding }         │
└──────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  4. CONSOLIDATE: LLM inference + embed semantic memories                 │
│     "User's email is user@example.com" → SemanticMemory { embedding }    │
└──────────────────────────────────────────────────────────────────────────┘
```

### Semantic Search in Practice

All recall operations use vector similarity search:

```python
# Query gets embedded, then searches across all memory types
query = "contact information"
query_vector = embed(query)  # [0.23, -0.45, 0.67, ...]

# Searches:
# - Episodes: "my email is user@example.com" → similarity 0.82
# - Facts: "user@example.com" → similarity 0.78
# - Semantic: "User's primary contact is email" → similarity 0.85
```

### Why Pattern Extraction + Embeddings?

| Approach | Speed | Confidence | Semantic |
|----------|-------|------------|----------|
| **Pattern extraction only** | Fast | High (0.9) | Limited to exact matches |
| **Embedding only** | Fast | N/A | Full semantic similarity |
| **Pattern + Embedding** | Fast | High (0.9) | Both exact AND semantic |

Pattern extractors (emails, phones, dates) provide **high-confidence structured data** quickly. The embedding layer makes that same data **semantically discoverable**.

Example: "user@example.com" is:
- Pattern-extracted with 0.9 confidence (deterministic)
- Embedded so "contact info" query finds it (semantic)

### Semantic Intelligence Features

| Feature | What It Does | Implementation |
|---------|--------------|----------------|
| **Semantic recall** | Natural language queries work | All queries embedded, cosine similarity |
| **Semantic negation filtering** | "I don't use MongoDB" filters related content | Negation patterns embedded, similarity threshold |
| **Semantic fact deduplication** | Prevents storing duplicate facts | New facts compared to existing via embedding |
| **Multi-hop reasoning** | Follow related_ids to connected memories | Links discovered via semantic similarity |

## API

```python
from engram.service import EngramService

# Initialize with async context manager
async with EngramService.create() as engram:
    # Store (immediate, preserves ground truth)
    result = await engram.encode(
        content="My email is john@example.com",
        role="user",
        user_id="user_123",
    )

    # Retrieve (with confidence filtering)
    memories = await engram.recall(
        query="What databases does the user work with?",
        user_id="user_123",
        memory_types=["factual", "semantic"],
        min_confidence=0.7,
        follow_links=True,             # Multi-hop reasoning
    )

    # Verify a fact against its source
    verified = await engram.verify(memory_id, user_id="user_123")

    # Temporal queries
    memories = await engram.recall_at(
        query="What was known about the user?",
        user_id="user_123",
        as_of=datetime(2024, 6, 1),    # Point-in-time query
    )

    # Background operations (run as durable workflows)
    await engram.consolidate(user_id="user_123")  # Extract, link, strengthen
```

## Background Operations

### Consolidation (Hierarchical Summarization)

Implemented via `run_consolidation()`. Compresses N episodes into 1 semantic summary:

```python
from pydantic import BaseModel
from pydantic_ai import Agent

class SummaryOutput(BaseModel):
    summary: str                       # 3-6 sentence summary
    keywords: list[str]                # Key terms for retrieval
    context: str                       # Domain/topic context

# Map-reduce for large episode batches
async def consolidate(user_id: str):
    # 1. Fetch ALL unsummarized episodes
    episodes = await get_unsummarized_episodes(user_id)

    # 2. Map phase: chunk and summarize each chunk
    chunks = chunk_episodes(episodes, max_per_chunk=20)
    chunk_summaries = [await summarize_chunk(chunk) for chunk in chunks]

    # 3. Reduce phase: combine chunk summaries
    if len(chunk_summaries) == 1:
        final_summary = chunk_summaries[0]
    else:
        final_summary = await reduce_summaries(chunk_summaries)

    # 4. Create semantic memory with bidirectional links
    memory = SemanticMemory(
        content=final_summary.summary,
        source_episode_ids=[ep.id for ep in episodes],
        keywords=final_summary.keywords,
    )
    await store_semantic(memory)

    # 5. Mark episodes as summarized (bidirectional link)
    await mark_episodes_summarized(
        episode_ids=[ep.id for ep in episodes],
        semantic_id=memory.id,
    )

    # 6. Build links to existing similar memories
    existing = await find_similar_memories(memory.embedding)
    for similar in existing:
        memory.add_link(similar.id)
        similar.add_link(memory.id)
        similar.strengthen(delta=0.1)  # Testing Effect
```

### Procedural Synthesis

Implemented via `run_synthesis()`. Synthesizes all semantics into one behavioral profile:

```python
async def create_procedural(user_id: str):
    # 1. Fetch ALL semantic memories
    semantics = await list_semantic_memories(user_id)

    # 2. LLM synthesis into behavioral profile
    profile = await synthesize_behavioral_profile(semantics)

    # 3. Create/replace ONE procedural per user
    procedural = ProceduralMemory(
        content=profile.content,
        source_semantic_ids=[s.id for s in semantics],
        source_episode_ids=deduplicated_episode_ids,
    )

    # Replace existing if present
    existing = await list_procedural_memories(user_id)
    if existing:
        await update_procedural(procedural)
    else:
        await store_procedural(procedural)
```

### Decay

Implemented via `run_decay()`. Updates importance scores based on time and access:

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
| Consolidation strength | [Roediger & Karpicke 2006](https://pmc.ncbi.nlm.nih.gov/articles/PMC5912918/) | Repeated retrieval strengthens memories (Testing Effect) |
| Ground truth | [HaluMem](https://arxiv.org/html/2511.03506) | <56% accuracy without source preservation |
| RIF suppression | [Anderson et al. (1994)](https://pubmed.ncbi.nlm.nih.gov/7931095/) | Retrieval actively suppresses similar non-retrieved items (implemented) |
