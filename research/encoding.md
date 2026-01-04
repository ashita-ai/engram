# Encoding Depth

Why Engram extracts meaning, not just stores text.

## The Argument

Storing raw text is shallow. You can search it, but you can't reason about it effectively. Extracting meaning—facts, relationships, preferences—creates richer representations that support better retrieval.

The levels of processing framework provides the theoretical basis.

## Levels of Processing (Craik & Lockhart, 1972)

Memory strength depends on processing depth during encoding:

| Level | Processing Type | Example | Retention |
|-------|-----------------|---------|-----------|
| **Shallow** | Structural features | "Is it uppercase?" | Poor |
| **Intermediate** | Phonemic features | "Does it rhyme?" | Moderate |
| **Deep** | Semantic meaning | "What does it mean?" | Strong |

> "Depth of processing determines retention, not time spent or repetition alone."
> — [Craik & Lockhart, 1972](http://wixtedlab.ucsd.edu/publications/Psych%20218/Craik_Lockhart_1972.pdf)

## The Evidence

Craik and Tulving (1975) tested this directly:

**Experiment**: Present words with questions at three levels:
- Structural: "Is it in capital letters?"
- Phonemic: "Does it rhyme with 'weight'?"
- Semantic: "Would it fit in 'The girl placed the _____ on the table'?"

**Result**: Semantic processing produced significantly better recall and recognition.

## Why Deep Processing Works

Deep processing creates:
- **More retrieval cues** — Connections to existing knowledge
- **Richer representations** — Meaning, not just surface features
- **Elaboration** — Links to related concepts
- **Distinctiveness** — Unique encoding improves discrimination

## Engram Application

The `encode()` operation processes interactions deeply:

```python
async def encode(interaction):
    # Store verbatim (ground truth)
    episode = await store_episode(interaction)

    # Extract facts (deep processing - deterministic)
    facts = extract_facts(interaction.content)  # Pattern matching
    for fact in facts:
        await store_factual(fact, source=episode.id, confidence="high")

    # Later: infer semantics (deep processing - LLM)
    # Deferred to consolidate() for batching
```

Types of deep processing:

| Processing | Method | Depth | When |
|------------|--------|-------|------|
| **Fact extraction** | Pattern matching | Deep | Immediate |
| **Relationship extraction** | Pattern/LLM | Deep | Immediate/Deferred |
| **Preference inference** | LLM | Deep | Deferred (consolidate) |
| **Semantic clustering** | Embedding | Deep | Background |

**Why this matters**:
- **Better retrieval** — Query by meaning, not just keywords
- **Richer context** — Extracted facts inform responses
- **Structured knowledge** — Facts are queryable, not just searchable

## Contrast: Shallow vs Deep Storage

| Shallow (raw text) | Deep (extracted meaning) |
|-------------------|-------------------------|
| "My email is john@example.com" | `email: john@example.com` |
| Search: substring match | Query: `get_fact("email")` |
| No structure | Typed, structured |
| Retrieval: fuzzy | Retrieval: exact |

## References

- [Craik & Lockhart (1972). Levels of processing: A framework for memory research](http://wixtedlab.ucsd.edu/publications/Psych%20218/Craik_Lockhart_1972.pdf)
- [Craik & Tulving (1975). Depth of processing and retention](https://www.simplypsychology.org/levelsofprocessing.html)
- [Wikipedia: Levels of processing model](https://en.wikipedia.org/wiki/Levels_of_processing_model)
