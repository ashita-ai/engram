# Architecture

Engram is a memory system for LLM applications built on Pydantic AI with durable execution.

> **Status**: Beta. Core APIs, REST endpoints, and workflows fully implemented with 800+ tests.

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
│    ┌──────────────────────┬──────────────────────┐           │
│    │ DBOS (SQLite/PgSQL)  │   Prefect (Flows)    │           │
│    └──────────────────────┴──────────────────────┘           │
├─────────────────────────────────────────────────────────────┤
│                       Qdrant                                 │
│              Vector storage for all memory types             │
└─────────────────────────────────────────────────────────────┘
```

## Design Principles

### 1. Ground Truth Preservation

Raw interactions stored as immutable episodic memories. All derived knowledge traces back to source. Errors are correctable.

### 2. Confidence Tracking

Every derived memory carries a confidence score appropriate to its type:

```
┌────────────────┬──────────────────────────────────────────────────┐
│ Memory Type    │ Confidence Method                                │
├────────────────┼──────────────────────────────────────────────────┤
│ Episodic       │ 1.0 always (verbatim, immutable ground truth)    │
│ Structured     │ LLM assesses during extraction (single call)     │
│ Semantic       │ LLM assesses during synthesis (single call)      │
│ Procedural     │ Bayesian updating with accumulating evidence     │
└────────────────┴──────────────────────────────────────────────────┘
```

**Why this architecture?**
- Episodic memories ARE the ground truth — no confidence needed beyond 1.0
- Structured/Semantic use LLMs — LLMs should assess their own certainty
- Procedural memories accumulate evidence — perfect for Bayesian updating

#### LLM-Assessed Confidence (Structured/Semantic)

The LLM returns confidence **alongside** the extraction/synthesis in a single call:

```python
class LLMExtractionOutput(BaseModel):
    # Extracted data
    summary: str
    keywords: list[str]
    people: list[Person]
    preferences: list[Preference]
    negations: list[Negation]

    # Confidence assessment (same call)
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_reasoning: str
```

This is efficient (no extra API call) and contextually aware — the LLM assesses confidence while it has full context of what it just extracted.

**Scoring guide for LLM:**
- 0.9-1.0: Explicitly stated, no hedging, clear and unambiguous
- 0.7-0.9: Clearly implied or stated with minor hedging
- 0.5-0.7: Reasonably inferred but not directly stated
- 0.3-0.5: Speculative, significant hedging, or ambiguous
- 0.0-0.3: Contradicted, heavily hedged, or likely wrong

#### Bayesian Confidence (Procedural)

Procedural memories track behavioral patterns observed over time. We use a Beta-Bernoulli model:

```python
from engram.confidence import BayesianConfidence

# Start with a prior belief
bc = BayesianConfidence.from_prior("weak")  # 50% initial

# Update with observations
bc.update(observed=True)   # Saw the behavior
bc.update(observed=True)   # Saw it again
bc.update(observed=False)  # Didn't see it

# Check confidence
print(bc.confidence)  # ~0.67
print(bc.credible_interval_95)  # (0.35, 0.90)
```

Features:
- **Priors**: uninformative, weak, optimistic, pessimistic
- **Batch updates**: `bc.update_batch(confirmations=5, contradictions=2)`
- **Decay**: `bc.decay(factor=0.95)` for time-based uncertainty
- **Credible intervals**: Statistical bounds on confidence

#### Composite Confidence Score

All confidence information is wrapped in `ConfidenceScore` for auditability:

```python
class ConfidenceScore(BaseModel):
    value: float                      # Final score 0.0-1.0
    extraction_method: str            # verbatim/extracted/inferred
    extraction_base: float            # Base score from method
    supporting_episodes: int          # Corroboration count
    last_confirmed: datetime          # When last seen/confirmed
    contradictions: int               # Conflicting evidence count
    llm_reasoning: str | None         # LLM's explanation (if applicable)
```

**Composite formula (applied during recompute):**

```python
confidence = (
    extraction_base * 0.50 +           # How it was extracted
    corroboration_score * 0.25 +       # How many sources support it
    recency_score * 0.15 +             # How recently confirmed
    verification_score * 0.10          # Format/validity checks
)
```

**Why auditable:** Every confidence score can be explained via `.explain()` — "0.73: inferred, 3 sources, confirmed 2 months ago, LLM: Clearly stated preference..."

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

Track what is explicitly NOT true to prevent false matches. When a user corrects a misunderstanding or explicitly negates something, we store that negation in `StructuredMemory.negations`.

This is an **engineering construct** for storing semantic negations (e.g., "User does NOT use MongoDB"), not an implementation of neural inhibition mechanisms.

### 7. Deferred Processing

Expensive LLM work is batched and deferred:

| Operation | When | Cost | Durable |
|-----------|------|------|---------|
| Episode storage | Immediate | Low (embed + store) | N/A |
| Pattern extraction | Immediate | Low (regex: emails, phones, URLs) | N/A |
| Structure (LLM) | Immediate or Background | Medium (LLM enrichment) | Yes (DBOS/Prefect) |
| Consolidation | Scheduled | Medium (LLM, batched) | Yes (DBOS/Prefect) |
| Decay | Scheduled | Low (math) | Yes (DBOS/Prefect) |

### 8. Hierarchical Memory Compression

Memory types form a compression hierarchy:

```
Working (volatile) → Episodic (immutable ground truth)
                         │
                         └─→ StructuredMemory (per-episode LLM extraction, immutable)
                                   │
                                   └─→ Semantic (N structured → 1 summary via LLM)
                                             │
                                             └─→ Procedural (all semantics → 1 behavioral profile)
```

**Structure** (`run_structure` workflow):
- Called immediately after `encode()` or deferred to batch processing
- Uses regex extractors for emails, phones, URLs (0.9 confidence)
- Uses LLM for dates, people, preferences, negations (0.8 confidence)
- Creates ONE StructuredMemory per Episode (immutable)
- Stores in Episode.quick_extracts for immediate access

**Consolidation** (`run_consolidation_from_structured` workflow):
- Fetches all unconsolidated StructuredMemories for a user
- Synthesizes pre-extracted summaries/entities into ONE SemanticMemory
- Marks StructuredMemories as `consolidated=True`
- Bidirectional traceability: `structured.source_episode_id` ↔ `semantic.source_episode_ids`

**Procedural Synthesis** (`run_synthesis` workflow):
- Fetches ALL semantic memories for a user
- Uses LLM to synthesize ONE behavioral profile per user
- Replaces existing procedural memory (doesn't accumulate)
- Links to source semantics via `source_semantic_ids`

Demotion triggers:
- Low access + time → archive via decay workflow
- Very low confidence → delete via decay workflow


### 9. Automatic Importance Detection

Episode importance determines how aggressively memories are processed and retained. Rather than requiring callers to manually assign importance, Engram calculates it automatically during `encode()` using fast heuristics that add no latency to the hot path.

**How importance is calculated:**

```python
# Base score
importance = 0.5

# Extracted facts indicate concrete information (+0.05 per fact, max +0.15)
importance += min(0.15, len(facts) * 0.05)

# Negations are corrections - critical for accuracy (+0.1 per negation, max +0.2)
importance += min(0.2, len(negations) * 0.1)

# Importance keywords signal user intent (+0.05 per keyword, max +0.1)
importance += min(0.1, keyword_matches * 0.05)

# Role adjustments
if role == "user":
    importance += 0.05   # User messages are primary information
if role == "system":
    importance -= 0.1    # System prompts less relevant for recall

# Final score clamped to [0.0, 1.0]
```

**Importance keywords** (checked case-insensitively):
`remember`, `important`, `don't forget`, `always`, `never`, `critical`, `key`, `must`, `essential`, `priority`, `urgent`, `note that`, `keep in mind`, `fyi`, `heads up`

**Why automatic detection matters:**
- **No caller burden**: Applications don't need to determine importance
- **Extraction-based signals**: Facts and negations extracted from content boost importance automatically
- **High-importance consolidation**: Episodes with `importance >= high_importance_threshold` (default 0.8) trigger immediate consolidation, ensuring critical information is processed promptly
- **Zero added latency**: All heuristics use fast pattern matching during the existing extraction phase

**Example importance scores:**

| Content | Role | Facts | Negations | Importance |
|---------|------|-------|-----------|------------|
| "hi there" | user | 0 | 0 | 0.55 |
| "my email is user@example.com" | user | 1 | 0 | 0.60 |
| "remember, I prefer PostgreSQL" | user | 0 | 0 | 0.60 |
| "actually I don't use MongoDB" | user | 0 | 1 | 0.65 |
| "important: my phone is 555-1234, email is x@y.com" | user | 2 | 0 | 0.70 |

Callers can still override with an explicit `importance` parameter when they have additional context.

## Memory Types

| Type | Mutability | Decay | Confidence | Purpose |
|------|------------|-------|------------|---------|
| **Episodic** | Immutable | Fast | Highest | Raw interactions, ground truth |
| **Structured** | Immutable | Slow | High (0.8-0.9) | Per-episode LLM extraction |
| **Semantic** | Mutable | Slow | Variable | Cross-episode LLM synthesis |
| **Procedural** | Mutable | Very slow | Variable | Behavioral patterns, preferences |
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
    quick_extracts: QuickExtracts | None # Immediate regex results
    structured: bool                     # Has StructuredMemory been created?
    structured_into: str | None          # ID of StructuredMemory
```

### StructuredMemory (Per-Episode LLM Extraction)

NEW in v0.x. Bridges Episode and Semantic with per-episode intelligence.

```python
class StructuredMemory(BaseModel):
    id: str
    source_episode_id: str               # One-to-one with Episode

    # Deterministic extraction (0.9 confidence)
    emails: list[str]
    phones: list[str]
    urls: list[str]

    # LLM extraction (0.8 confidence)
    dates: list[ResolvedDate]            # {"raw": "next Tuesday", "resolved": "2025-01-14"}
    people: list[Person]                 # {"name": "John", "role": "manager"}
    organizations: list[str]
    locations: list[str]
    preferences: list[Preference]        # {"topic": "database", "value": "PostgreSQL"}
    negations: list[Negation]            # {"content": "doesn't use MongoDB", "pattern": "MongoDB"}

    summary: str                         # 1-2 sentence episode summary
    keywords: list[str]                  # Key terms for retrieval
    derived_at: datetime
    confidence: ConfidenceScore          # 0.8-0.9 based on extraction method
    embedding: list[float]

    # Consolidation tracking
    consolidated: bool                   # Included in semantic synthesis?
    consolidated_into: str | None        # ID of SemanticMemory
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

## Data Flow

```
Interaction
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                         ENCODE                               │
│   ┌─────────────────┐  ┌─────────────────────────────────┐  │
│   │ Store Episode   │  │ Quick Extract (regex)           │  │
│   │ (verbatim)      │  │ emails, phones, URLs            │  │
│   └────────┬────────┘  └───────────────┬─────────────────┘  │
└────────────┼───────────────────────────┼────────────────────┘
             │                           │
             ▼                           ▼
        [Episodic]              [Episode.quick_extracts]
             │
             │
┌────────────┴────────────────────────────────────────────────┐
│                    STRUCTURE (deferred)                      │
│   ┌─────────────────────────────────────────────────────┐   │
│   │  LLM Extraction per Episode:                         │   │
│   │  dates, people, organizations, preferences, negations│   │
│   │  + summary, keywords                                 │   │
│   └──────────────────────────┬──────────────────────────┘   │
└──────────────────────────────┼──────────────────────────────┘
                               │
                               ▼
                       [StructuredMemory]
                               │
┌──────────────────────────────┴──────────────────────────────┐
│                  CONSOLIDATE (background)                    │
│   ┌─────────────────────────────────────────────────────┐   │
│   │  Synthesize N StructuredMemories → 1 SemanticMemory  │   │
│   │  + Link to similar existing memories                 │   │
│   │  + Update consolidation_strength                     │   │
│   └──────────────────────────┬──────────────────────────┘   │
└──────────────────────────────┼──────────────────────────────┘
                               │
                               ▼
                          [Semantic]
                               │
                               └───→ [Procedural] (user behavioral profile)
```

## Storage: Qdrant

All persistent memory types stored in Qdrant with type-specific collections (Working memory is in-memory only):

```
engram_episodic     → vectors + payload (content, timestamp, session_id, importance, quick_extracts)
engram_structured   → vectors + payload (summary, keywords, entities, negations, source_episode_id, confidence)
engram_semantic     → vectors + payload (content, source_episode_ids, related_ids, confidence, consolidation_strength)
engram_procedural   → vectors + payload (content, trigger_context, access_count, confidence)
engram_audit        → vectors + payload (audit trail)
```

### Indexing

- **Episodic**: HNSW with time-decay weighting
- **Structured/Semantic**: Standard HNSW
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
│  3. EXTRACT: Pattern match + store in StructuredMemory                   │
│     "user@example.com" → StructuredMemory.emails                         │
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
# - Structured: per-episode extraction → similarity 0.78
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
        memory_types=["structured", "semantic"],
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
- Structured: Slow (months)
- Semantic: Slow (months)
- Procedural: Very slow (years)

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

### Prefect (Flow Orchestration)

```python
from prefect import flow

@flow(name="engram-consolidation")
async def consolidation_flow(user_id: str, org_id: str | None = None):
    episodes = await get_episodes(user_id, org_id)
    result = await consolidation_agent.run(format_episodes(episodes))
    await store_results(result.data)
    return result
```


## Research Foundations

This architecture is informed by recent research (2024-2025):

| Feature | Research | Key Finding |
|---------|----------|-------------|
| Dynamic linking | [A-MEM](https://arxiv.org/abs/2502.12110) | 2x improvement on multi-hop reasoning |
| Buffer promotion | [Cognitive Workspace](https://arxiv.org/abs/2508.13171) | 58.6% memory reuse vs 0% for naive RAG |
| Consolidation strength | [Roediger & Karpicke 2006](https://pmc.ncbi.nlm.nih.gov/articles/PMC5912918/) | Repeated retrieval strengthens memories (Testing Effect) |
| Ground truth | [HaluMem](https://arxiv.org/html/2511.03506) | <56% accuracy without source preservation |
