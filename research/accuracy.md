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

Ground truth preservation enables explicit confidence:

| Source Type | Confidence | Method |
|-------------|------------|--------|
| `verbatim` | 100% | Direct quote from episode |
| `extracted` | High | Pattern-matched, deterministic |
| `inferred` | Variable | LLM-derived, uncertain |

Applications can filter by confidence:
- High-stakes decisions: `min_confidence=0.9`
- Exploratory queries: `min_confidence=0.5`

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
