<p align="center">
  <img src="assets/engram.jpg" alt="Engram Logo">
</p>

# Engram

**Memory you can trust.**

A memory system for AI applications that preserves ground truth, tracks confidence, and prevents hallucinations.

## The Problem

AI memory systems have an accuracy crisis. Recent benchmarks show:

> "All systems achieve answer accuracies below 56%, with hallucination rate and omission rate remaining high... Systems suffer omission rates above 50%."
>
> — [HaluMem: Hallucinations in LLM Memory Systems](https://arxiv.org/html/2511.03506)

Why? Most systems use LLM extraction on every message. This compounds errors:

| Approach | What Goes Wrong |
|----------|----------------|
| **Summarize conversations** | Loses details. Can't recall specifics. |
| **LLM extraction on write** | Extraction errors become permanent. Hallucinations propagate. |
| **Store in vector DB only** | No structure. Retrieves irrelevant noise. |

The fundamental issue: once source data is lost, errors cannot be corrected.

## The Solution

Engram preserves ground truth and tracks confidence:

1. **Store first, derive later** — Raw conversations stored verbatim. LLM extraction happens in background where errors can be caught.

2. **Track confidence** — Every fact carries a composite confidence score: extraction method + corroboration + recency + verification. Fully auditable.

3. **Verify on retrieval** — Applications filter by confidence. High-stakes queries use only trusted facts.

4. **Enable recovery** — Derived facts trace to sources. Errors can be corrected by re-deriving.

## How Trust Works

```
User: "My email is john@example.com"
    ↓
Episodic Memory (immutable, verbatim)
    ↓
Factual Memory: email=john@example.com
├── Source: Episode #1234
├── Extraction: pattern match (deterministic)
└── Confidence: 0.9

Later query: "What's the user's email?"
    ↓
Retrieval with min_confidence=0.9
    ↓
Returns: john@example.com (verified against source)
```

## Memory Types

```mermaid
flowchart LR
    subgraph INPUT [" "]
        direction TB
        WM([🧠 WORKING])
    end

    subgraph CORE [" "]
        direction TB
        EP([📝 EPISODIC])
        FACT{{📊 FACTUAL}}
        SEM([💡 SEMANTIC])
        PROC([⚙️ PROCEDURAL])
        NEG([🚫 NEGATION])
    end

    subgraph STORE [" "]
        direction TB
        QD[(🗄️ QDRANT)]
    end

    WM ==>|encode| EP
    EP -->|extract| FACT
    EP -->|negate| NEG
    EP -->|consolidate| SEM
    EP -->|pattern| PROC

    EP -.->|store| QD
    FACT -.->|store| QD
    SEM -.->|store| QD
    PROC -.->|store| QD
    NEG -.->|store| QD

    style WM fill:#fbbf24,stroke:#b45309,stroke-width:3px,color:#1c1917
    style EP fill:#a78bfa,stroke:#7c3aed,stroke-width:3px,color:#1c1917
    style FACT fill:#60a5fa,stroke:#2563eb,stroke-width:3px,color:#1c1917
    style SEM fill:#34d399,stroke:#059669,stroke-width:3px,color:#1c1917
    style PROC fill:#f472b6,stroke:#db2777,stroke-width:3px,color:#1c1917
    style NEG fill:#f87171,stroke:#dc2626,stroke-width:3px,color:#1c1917
    style QD fill:#fb923c,stroke:#c2410c,stroke-width:3px,color:#1c1917

    style INPUT fill:transparent,stroke:#fbbf24,stroke-width:2px,stroke-dasharray:5
    style CORE fill:transparent,stroke:#a78bfa,stroke-width:2px,stroke-dasharray:5
    style STORE fill:transparent,stroke:#fb923c,stroke-width:2px,stroke-dasharray:5
```

| Type | Confidence | Source | Use Case |
|------|------------|--------|----------|
| **Working** | N/A | Current context | Active conversation (in-memory, volatile) |
| **Episodic** | Highest | Verbatim storage | Ground truth, audit trail |
| **Factual** | High | Pattern extraction | Emails, dates, names |
| **Semantic** | Variable | LLM inference | Preferences, context |
| **Procedural** | Variable | LLM inference | Behavioral preferences |
| **Negation** | Variable | Negation detection | What is NOT true |

## Preventing Hallucinations

### 1. Deterministic Extraction First

Pattern matching before LLMs — no hallucination possible:

```python
# High confidence, reproducible
EMAIL_PATTERN = r'\b[\w.-]+@[\w.-]+\.\w+\b'
facts = extract_patterns(message, [EMAIL_PATTERN])
```

### 2. Defer LLM Work

Batch in background where errors can be caught:

```python
# Critical path: store ground truth + extract patterns
result = await engram.encode(content, role="user", user_id="u1")  # Fast, no LLM
# Facts extracted immediately via regex (emails, phones, dates)

# Background: derive semantics with oversight (coming soon)
# await engram.consolidate()  # LLM extraction, batched
```

### 3. Confidence-Gated Retrieval

Applications choose their trust level:

```python
# High-stakes: only verified facts
trusted = await engram.recall(query, user_id="u1", min_confidence=0.9)

# Exploratory: include inferences
all_relevant = await engram.recall(query, user_id="u1", min_confidence=0.5)
```

### 4. Source Verification

Trace any fact back to its source:

```python
# Debug: why does the system believe this?
result = await engram.verify("fact_abc123", user_id="u1")
print(result.explanation)
# → "Pattern-matched email from source episode(s). Source: ep_xyz (2024-01-15 10:30). Confidence: 0.90"

# Get raw source episodes
episodes = await engram.get_sources("fact_abc123", user_id="u1")
# → Returns original conversation where email was mentioned
```

## Usage

```python
from engram.service import EngramService

# Initialize with async context manager
async with EngramService.create() as engram:
    # Store interaction (immediate, preserves ground truth)
    result = await engram.encode(
        content="My email is john@example.com",
        role="user",
        user_id="user_123",
    )
    print(f"Stored episode {result.episode.id}")
    print(f"Extracted {len(result.facts)} facts")

    # Retrieve with confidence filtering
    memories = await engram.recall(
        query="What's the user's email?",
        user_id="user_123",
        memory_types=["factual", "semantic"],
        min_confidence=0.7,
    )

    # Verify a specific fact
    if memories:
        verified = await engram.verify(memories[0].memory_id, user_id="user_123")
        print(verified.explanation)
```

> **Coming Soon**: `consolidate()` and `decay()` background operations ([#22](https://github.com/ashita-ai/engram/issues/22))

## Design Principles

### Ground Truth is Sacred

Every derived memory points back to source episodes. If extraction makes a mistake, re-derive from the original.

### Confidence is Composite and Auditable

Confidence isn't a single number — it's a composite score you can explain:

| Factor | Weight | Example |
|--------|--------|---------|
| Extraction method | 50% | Pattern-matched = 0.9, LLM-inferred = 0.6 |
| Corroboration | 25% | 5 supporting episodes > 1 episode |
| Recency | 15% | Confirmed yesterday > confirmed last year |
| Verification | 10% | Email format valid, date in range |

Every score is auditable: *"0.73 because: extracted (0.9 base), 3 sources, last confirmed 2 months ago."*

### Memories Get Smarter Over Time

Engram isn't just storage—it's a system that learns:

- **Frequently-used memories strengthen** — Every time a memory participates in consolidation or gets linked to new information, it gets stronger. Based on the [Testing Effect](https://www.sciencedirect.com/topics/psychology/testing-effect), one of the most robust findings in memory research.

- **Related memories find each other** — When you say "prefer Python for scripts," that automatically links to "using Python at work." No manual organization needed.

- **Irrelevant stuff fades away** — Memories decay over time. Unimportant memories fade; important ones persist. This keeps the store relevant.

- **Retrieval suppresses competitors** — Based on [Anderson et al. (1994)](https://pubmed.ncbi.nlm.nih.gov/7931095/), when memories are retrieved, similar non-retrieved memories are suppressed. Enable with `rif_enabled=True` on recall to naturally prune redundant memories.

### Fast Path Stays Fast

| Operation | When | Cost |
|-----------|------|------|
| Store episode | Every message | Low (embed + store) |
| Extract facts | Every message | Low (regex, no LLM) |
| Infer semantics | Background | Medium (LLM, batched) |

## Documentation

- [Architecture](docs/architecture.md) — Memory types, data flow, storage
- [Development Guide](docs/development.md) — Setup, configuration, workflow
- [Research Foundations](docs/research/overview.md) — Theoretical basis and the accuracy problem
- [Competitive Analysis](docs/research/competitive.md) — How Engram compares to alternatives

## Status

Pre-alpha. Architecture and design phase.

## License

MIT
