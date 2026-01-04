# Architecture

Engram is a memory system for LLM applications built on cognitive science principles rather than ad-hoc engineering.

## Design Principles

### 1. Ground Truth Preservation

Raw interactions are stored as immutable episodic memories. All derived knowledge (facts, inferences, preferences) maintains a reference back to source episodes. If extraction or consolidation introduces errors, the original data is always available for re-derivation.

### 2. Separation of Certainty

Not all memories are equal. Engram distinguishes between:

- **Verbatim**: Direct quotes, explicit statements
- **Extracted**: Pattern-matched facts with high confidence
- **Inferred**: LLM-synthesized knowledge with inherent uncertainty
- **Consolidated**: Patterns derived from multiple episodes

Each memory carries its source type and confidence score. Retrieval can filter by certainty threshold.

### 3. Deferred Processing

Expensive LLM operations are batched and deferred, not triggered per-message:

| Operation | When | Cost |
|-----------|------|------|
| Episode storage | Immediate | Low (embed + store) |
| Factual extraction | Immediate | Low (pattern matching) |
| Semantic inference | Background | Medium (LLM) |
| Consolidation | Scheduled | Medium (LLM, batched) |
| Decay | Scheduled | Low (math) |

This avoids the token explosion and latency problems of per-message LLM processing.

### 4. Explicit Memory Types

Different memory types have different characteristics and require different handling:

| Type | Mutability | Decay Rate | Confidence | Index Strategy |
|------|------------|------------|------------|----------------|
| Episodic | Immutable | Fast | High (verbatim) | Time + relevance |
| Factual | Immutable | Slow | High (extracted) | Pure relevance |
| Semantic | Mutable | Slow | Variable | Pure relevance |
| Procedural | Mutable | Very slow | Variable | Context-filtered |

## Memory Types

### Episodic Memory

Raw interactions exactly as they occurred.

```python
EpisodicMemory(
    content="User: What's the best way to handle async in Python?",
    timestamp=datetime(2024, 1, 15, 14, 30),
    session_id="sess_abc123",
    source="verbatim",
    confidence=1.0
)
```

Properties:
- Never modified after creation
- Source of truth for all derived memories
- Decays relatively quickly unless accessed
- Indexed by time and semantic similarity

### Factual Memory

Explicit facts extracted through deterministic pattern matching.

```python
FactualMemory(
    content="email: john@example.com",
    category="contact",
    source_episode_id="ep_xyz789",
    extraction_method="pattern",  # not LLM
    confidence=0.95
)
```

Properties:
- Extracted via regex, NER, or structured parsing
- No LLM interpretation — deterministic
- High confidence, low risk of corruption
- Slow decay, high retrieval priority

Examples of factual extraction:
- "My name is [X]" → `name: X`
- "I work at [X]" → `employer: X`
- "My email is [X]" → `email: X`
- "I'm using Python [X]" → `tool: Python, version: X`

### Semantic Memory

Inferred knowledge synthesized from episodes.

```python
SemanticMemory(
    content="User is experienced with async Python patterns",
    source_episode_ids=["ep_001", "ep_042", "ep_089"],
    source="inferred",
    confidence=0.7,
    last_consolidated=datetime(2024, 1, 20)
)
```

Properties:
- Derived through LLM inference
- Tagged as uncertain — confidence < 1.0
- Can be rebuilt from source episodes if corrupted
- Updated through consolidation

### Procedural Memory

Learned behavioral patterns and preferences.

```python
ProceduralMemory(
    content="User prefers concise code examples over lengthy explanations",
    trigger_context="code_assistance",
    source_episode_ids=["ep_012", "ep_034", "ep_056"],
    confidence=0.8
)
```

Properties:
- Activated by context matching, not direct query
- Inferred from patterns across episodes
- Influences response style, not content
- Very slow decay

### Working Memory

Current conversation context. Volatile, not persisted.

```python
WorkingMemory(
    items=["current_task: debug auth flow", "file: src/auth.py", "error: token expired"],
    capacity=7,  # Miller's Law
    attention_weights=[0.9, 0.7, 0.95]
)
```

Properties:
- Limited capacity (~7 items)
- Determines what gets encoded to long-term storage
- Cleared between sessions
- High-attention items more likely to be encoded

### Scratchpad (Agent State)

Lossless execution context for agent workflows.

```python
Scratchpad(
    task_id="task_abc",
    state={
        "current_file": "/src/auth.py",
        "line_number": 142,
        "pending_actions": ["run_tests", "commit"],
        "variables": {"token": "eyJ..."}
    },
    checkpoints=[...]
)
```

Properties:
- Exact, lossless storage — no summarization
- Temporal ordering preserved
- Supports checkpoint/restore
- Separate from cognitive memory types

## Data Flow

```
Interaction
    │
    ▼
┌─────────────────────────────────────────────────────┐
│                    ENCODE                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │  Embed +    │  │   Pattern   │  │  Importance │ │
│  │   Store     │  │   Extract   │  │   Scoring   │ │
│  │  Episode    │  │   Facts     │  │             │ │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘ │
└─────────┼────────────────┼────────────────┼─────────┘
          │                │                │
          ▼                ▼                ▼
     [Episodic]       [Factual]      [importance]
          │                              score
          │                                │
          └──────────────┬─────────────────┘
                         │
          ┌──────────────┴──────────────┐
          │      BACKGROUND JOBS         │
          │  ┌──────────┐ ┌──────────┐  │
          │  │Consolidate│ │  Decay   │  │
          │  │(scheduled)│ │(scheduled)│  │
          │  └─────┬─────┘ └─────┬────┘  │
          └────────┼─────────────┼───────┘
                   │             │
                   ▼             ▼
              [Semantic]    [updated
              [Procedural]   scores]
```

## Storage Architecture

All memory types are stored in Qdrant with type-specific collections and indexing strategies.

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
│   └── payload (content, source_episode_ids, confidence, last_consolidated)
└── procedural/
    ├── vectors (embeddings)
    └── payload (content, trigger_context, confidence)
```

### Indexing Strategies

**Episodic**: HNSW with time-decay weighting. Recent memories score higher.

**Factual**: Standard HNSW. Pure semantic relevance.

**Semantic**: Standard HNSW. Pure semantic relevance.

**Procedural**: HNSW with context filtering. Only retrieve when trigger context matches.

## Retrieval

```python
memories = await memory.recall(
    query="What databases does the user work with?",
    memory_types=["factual", "semantic"],  # skip episodic noise
    min_confidence=0.7,                     # only high-confidence
    limit=10
)
```

Retrieval parameters:
- `memory_types`: Which stores to search
- `min_confidence`: Filter by certainty
- `recency_weight`: Balance relevance vs recency (for episodic)
- `include_sources`: Return source episode IDs for verification

## Background Operations

### Consolidation

Runs on schedule (e.g., hourly, daily). Analyzes accumulated episodes to extract semantic patterns.

```python
# Pseudocode
episodes = fetch_unconsolidated_episodes(since=last_consolidation)
clusters = cluster_by_topic(episodes)
for cluster in clusters:
    if len(cluster) >= consolidation_threshold:
        semantic_fact = llm_extract_pattern(cluster)
        store_semantic(semantic_fact, source_episodes=cluster)
        mark_consolidated(cluster)
```

### Decay

Runs on schedule. Updates importance scores based on time and access patterns.

```python
# Pseudocode
for memory in all_memories:
    time_factor = exp(-time_since_creation / decay_constant)
    access_factor = 1.0 + (0.1 * access_count)  # retrieval strengthens
    importance_factor = memory.importance  # high-importance decays slower

    new_score = base_score * time_factor * access_factor * importance_factor

    if new_score < deletion_threshold:
        delete(memory)
    else:
        update_score(memory, new_score)
```

Decay constants vary by memory type:
- Episodic: Fast decay (days)
- Factual: Slow decay (months)
- Semantic: Slow decay (months)
- Procedural: Very slow decay (years)
