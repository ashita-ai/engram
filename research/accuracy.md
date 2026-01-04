# The Accuracy Problem

Why AI memory systems need ground truth preservation.

## The Problem

Most AI memory systems use LLM-based extraction on every message. This creates compounding problems:

1. **Extraction errors** — LLMs make mistakes when extracting facts
2. **Information loss** — Summarization discards details
3. **Error propagation** — Corrupted memories distort future decisions
4. **No recovery** — Once source data is lost, errors are permanent

## Empirical Evidence

### HaluMem Benchmark (2024)

The HaluMem benchmark evaluated memory-augmented LLM systems on factual accuracy:

> "All systems achieve answer accuracies below 56%, with both hallucination rate and omission rate remaining high... Systems suffer omission rates above 50%, primarily stemming from insufficient coverage in memory extraction."
>
> — [HaluMem: Evaluating Hallucinations in Memory Systems](https://arxiv.org/html/2511.03506)

Key findings:
- **<56% accuracy** on factual questions
- **>50% omission rate** — systems fail to extract important facts
- **Hallucination** — systems generate false memories
- Root cause: LLM extraction during write is unreliable

### Memory as Reconstruction

Cognitive science shows that human memory is reconstructive, not reproductive:

> "Memories are not simply recordings of the past but are constructed in the present... Both episodic and semantic memory are constructive."
>
> — [Constructing Memories, Cognitive Science 2025](https://onlinelibrary.wiley.com/doi/10.1111/cogs.70113)

> "Remembering is not the re-excitation of innumerable fixed, lifeless and fragmentary traces. It is an imaginative reconstruction."
>
> — [Bartlett, 1932](https://en.wikipedia.org/wiki/Reconstructive_memory)

This is how human memory works—but it's a problem for AI systems that need reliability.

### LLM Extraction Errors

Research on LLM information extraction shows consistent error patterns:

> "Current LLMs are able to achieve good performance on some tasks but still have problems on... complex knowledge extraction, multi-hop reasoning."
>
> — [Survey on Information Extraction with LLMs](https://arxiv.org/abs/2312.17617)

Error types in LLM extraction:
- **Omission** — Missing important facts
- **Hallucination** — Generating facts that weren't present
- **Conflation** — Merging distinct facts incorrectly
- **Misattribution** — Assigning facts to wrong entities

## The Solution: Ground Truth Preservation

### Principle

Store raw data first. Derive later. Never lose the original.

```
Episode (immutable) → Derived Facts → Application
       ↑                    ↓
       └────── recovery ────┘
```

### Implementation

1. **Episodic memories are immutable** — Raw interactions stored verbatim
2. **Derived facts trace to source** — Every extraction links to its episode
3. **Confidence is explicit** — Know what's certain vs. inferred
4. **Errors are correctable** — Re-extract from ground truth if needed

### Benefits

| Without Ground Truth | With Ground Truth |
|---------------------|-------------------|
| Extraction error → permanent corruption | Extraction error → re-derive from source |
| "Why does the AI think X?" → unknown | "Why does the AI think X?" → traceable |
| Errors compound over time | Errors are isolated and correctable |
| Single extraction attempt | Multiple attempts possible |

## Confidence Tracking

Ground truth preservation enables composite confidence scoring:

| Factor | Weight | Description |
|--------|--------|-------------|
| Extraction method | 50% | verbatim=1.0, extracted=0.9, inferred=0.6 |
| Corroboration | 25% | Multiple sources increase score |
| Recency | 15% | Recently confirmed facts score higher |
| Verification | 10% | Format checks passed (valid email, etc.) |

Every score is auditable — you can explain *why* confidence is 0.73.

Applications can filter by confidence:
- High-stakes decisions: `min_confidence=0.9`
- Exploratory queries: `min_confidence=0.5`

**Confidence decay:** Facts not re-confirmed decay over time (~63% after 1 year), keeping the store current.

## Deterministic vs. LLM Extraction

Engram uses deterministic extraction where possible:

| Extraction Type | Method | Confidence | Example |
|----------------|--------|------------|---------|
| **Pattern** | Regex/rules | High | Email: `\S+@\S+\.\w+` |
| **Structured** | JSON parsing | High | `{"email": "..."}` |
| **Semantic** | LLM inference | Variable | "User seems experienced" |

Deterministic extraction:
- Is reproducible (same input → same output)
- Has known failure modes
- Doesn't hallucinate
- Is auditable

LLM extraction:
- Handles ambiguous/complex cases
- Can infer implicit information
- May hallucinate
- Should be deferred and batched

## Detecting Hallucinations

Engram enables hallucination detection through source verification:

### Verification Strategies

| Strategy | How It Works | When to Use |
|----------|--------------|-------------|
| **Source check** | Trace derived fact to episode, verify content | High-stakes facts |
| **Confidence threshold** | Only use facts above confidence threshold | All retrieval |
| **Multi-source** | Require multiple episodes supporting a fact | Important decisions |
| **Recency check** | Weight recent episodes higher | Time-sensitive facts |

### Implementation

```python
async def verify_fact(fact):
    """Verify a derived fact against its source episodes."""
    # Get source episodes
    episodes = await get_episodes(fact.source_ids)

    # Re-extract from source
    re_extracted = await extract_fact(episodes, fact.key)

    # Compare
    if re_extracted != fact.value:
        return VerificationResult(
            status="mismatch",
            original=fact.value,
            re_extracted=re_extracted,
            confidence=0.0
        )

    return VerificationResult(
        status="verified",
        value=fact.value,
        confidence=fact.confidence
    )
```

### Consistency Checks

Detect contradictions in the memory store:

```python
async def check_consistency(memory_store, user_id):
    """Find contradicting facts in the memory store."""
    facts = await get_all_facts(user_id)

    contradictions = []
    for fact_a, fact_b in combinations(facts, 2):
        if fact_a.key == fact_b.key and fact_a.value != fact_b.value:
            contradictions.append((fact_a, fact_b))

    return contradictions
```

## Preventing Hallucinations

Prevention is better than detection:

### 1. Deterministic Extraction First

Use pattern matching before LLMs:

```python
# High confidence - no hallucination possible
EMAIL_PATTERN = r'\b[\w.-]+@[\w.-]+\.\w+\b'
PHONE_PATTERN = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'

def extract_deterministic(text):
    facts = []
    for email in re.findall(EMAIL_PATTERN, text):
        facts.append(Fact(key="email", value=email, confidence=1.0))
    return facts
```

### 2. Defer LLM Extraction

Batch LLM work in background where errors can be caught:

```python
# Don't do this on critical path
# fact = await llm_extract(message)  # Can hallucinate

# Do this instead
await store_episode(message)  # Ground truth preserved
# Later, in background:
await consolidate()  # LLM extraction with review
```

### 3. Multi-Episode Confirmation

Require multiple sources before creating semantic facts:

```python
async def create_semantic_fact(pattern, episodes):
    """Only create fact if supported by multiple episodes."""
    supporting = [ep for ep in episodes if pattern in ep.content]

    if len(supporting) < MIN_SUPPORTING_EPISODES:
        return None  # Not enough evidence

    return SemanticFact(
        content=pattern,
        confidence=len(supporting) / len(episodes),
        sources=[ep.id for ep in supporting]
    )
```

### 4. Confidence-Gated Retrieval

Filter by confidence at retrieval time:

```python
# Application can choose trust level
trusted_facts = await memory.recall(
    query=query,
    min_confidence=0.9,  # Only high-confidence facts
    source_types=["verbatim", "extracted"]  # No inferred
)
```

## Data Provenance

Ground truth preservation provides complete provenance:

```
Semantic Memory: "User prefers Python"
├── Derived from Episode #1234 (2024-01-15)
├── Extraction method: LLM inference
├── Confidence: 0.85
└── Supporting episodes: [#1234, #1267, #1301]
```

This enables:
- **Debugging** — Why does the system believe X?
- **Correction** — Update the derivation, not the source
- **Audit** — Complete trail for compliance
- **Recovery** — Re-derive if extraction improves

## References

### LLM Memory Accuracy
- [HaluMem: Evaluating Hallucinations in Memory Systems](https://arxiv.org/html/2511.03506)
- [Survey on Information Extraction with LLMs](https://arxiv.org/abs/2312.17617)

### Reconstructive Memory
- [Constructing Memories, Episodic and Semantic](https://onlinelibrary.wiley.com/doi/10.1111/cogs.70113)
- [Wikipedia: Reconstructive memory](https://en.wikipedia.org/wiki/Reconstructive_memory)
- Bartlett, F.C. (1932). Remembering: A Study in Experimental and Social Psychology.
- Loftus, E.F. (1979). Eyewitness Testimony.

### Data Provenance
- [W3C PROV: Provenance Data Model](https://www.w3.org/TR/prov-dm/)
- [Data Lineage: What, Why, and How](https://www.alation.com/blog/what-is-data-lineage/)
