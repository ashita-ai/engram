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

2. **Track confidence** — Every fact carries provenance: `verbatim` (100%), `extracted` (high), `inferred` (variable).

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
└── Confidence: 1.0

Later query: "What's the user's email?"
    ↓
Retrieval with min_confidence=0.9
    ↓
Returns: john@example.com (verified against source)
```

## Memory Types

```mermaid
graph TB
    subgraph ENGRAM [" "]
        direction TB

        EP["<b>EPISODIC</b><br><i>verbatim</i>"]
        FACT["<b>FACTUAL</b><br><i>extracted</i>"]
        SEM["<b>SEMANTIC</b><br><i>inferred</i>"]
        WM["<b>WORKING</b><br><i>current context</i>"]
        PROC["<b>PROCEDURAL</b><br><i>preferences</i>"]

        WM -->|encode| EP
        EP -->|pattern match| FACT
        EP -->|consolidate| SEM
    end

    QD[("<b>QDRANT</b>")]

    EP -.-> QD
    FACT -.-> QD
    SEM -.-> QD
    PROC -.-> QD

    classDef episodic fill:#c4b5fd,stroke:#7c3aed,stroke-width:2px,color:#1e1b4b
    classDef factual fill:#93c5fd,stroke:#2563eb,stroke-width:2px,color:#1e3a5f
    classDef semantic fill:#6ee7b7,stroke:#059669,stroke-width:2px,color:#064e3b
    classDef working fill:#fcd34d,stroke:#d97706,stroke-width:2px,color:#78350f
    classDef procedural fill:#f9a8d4,stroke:#db2777,stroke-width:2px,color:#500724
    classDef storage fill:#fdba74,stroke:#ea580c,stroke-width:3px,color:#431407

    class EP episodic
    class FACT factual
    class SEM semantic
    class WM working
    class PROC procedural
    class QD storage
```

| Type | Confidence | Source | Use Case |
|------|------------|--------|----------|
| **Episodic** | 100% | Verbatim storage | Ground truth, audit trail |
| **Factual** | High | Pattern extraction | Emails, dates, names |
| **Semantic** | Variable | LLM inference | Preferences, context |
| **Procedural** | High | Pattern detection | Behavioral preferences |

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
# Critical path: store ground truth only
await memory.encode(interaction)  # Fast, no LLM

# Background: derive with oversight
await memory.consolidate()  # LLM extraction, batched
```

### 3. Confidence-Gated Retrieval

Applications choose their trust level:

```python
# High-stakes: only verified facts
trusted = await memory.recall(query, min_confidence=0.9)

# Exploratory: include inferences
all_relevant = await memory.recall(query, min_confidence=0.5)
```

### 4. Source Verification

Trace any fact back to its source:

```python
# Debug: why does the system believe this?
fact = await memory.get_fact("user_email")
episodes = await memory.get_sources(fact.source_ids)
# → Shows original conversation where email was mentioned
```

## Usage

```python
from engram import MemoryStore

memory = MemoryStore(qdrant_url="...", user_id="user_123")

# Store interaction (immediate, preserves ground truth)
await memory.encode(interaction, extract_facts=True)

# Retrieve with confidence filtering
memories = await memory.recall(
    query="What databases does the user work with?",
    memory_types=["factual", "semantic"],
    min_confidence=0.7
)

# Verify a specific fact
verified = await memory.verify(fact_id)

# Background: consolidate and decay
await memory.consolidate()
await memory.decay()
```

## Design Principles

### Ground Truth is Sacred

Every derived memory points back to source episodes. If extraction makes a mistake, re-derive from the original.

### Confidence is Explicit

| Source Type | Confidence | Method |
|-------------|------------|--------|
| `verbatim` | 100% | Direct quote |
| `extracted` | High | Pattern-matched, deterministic |
| `inferred` | Variable | LLM-derived |

### Forgetting is a Feature

Memories decay over time. Unimportant memories fade; important ones persist. This keeps the store relevant.

### Fast Path Stays Fast

| Operation | When | Cost |
|-----------|------|------|
| Store episode | Every message | Low (embed + store) |
| Extract facts | Every message | Low (regex, no LLM) |
| Infer semantics | Background | Medium (LLM, batched) |

## Documentation

- [Accuracy & Hallucination Prevention](research/accuracy.md) — Why ground truth matters
- [Architecture](docs/architecture.md) — Memory types, data flow, storage
- [Research Foundations](research/overview.md) — Theoretical basis

## Status

Pre-alpha. Architecture and design phase.

## License

MIT
