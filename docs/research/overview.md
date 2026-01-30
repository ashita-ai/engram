# Research Foundations

This document explains *why* Engram makes its specific design choices, with citations to supporting research.

---

## The Problem: AI Memory Accuracy

Most AI memory systems use LLM-based extraction on every message. This creates compounding problems:

1. **Extraction errors** — LLMs make mistakes when extracting facts
2. **Information loss** — Summarization discards details
3. **Error propagation** — Corrupted memories distort future decisions
4. **No recovery** — Once source data is lost, errors are permanent

### Empirical Evidence

The HaluMem benchmark (2024) evaluated memory-augmented LLM systems on factual accuracy:

> "All systems achieve answer accuracies below 56%, with both hallucination rate and omission rate remaining high... Systems suffer omission rates above 50%, primarily stemming from insufficient coverage in memory extraction."
> — [HaluMem: Evaluating Hallucinations in Memory Systems](https://arxiv.org/html/2511.03506)

Error types in LLM extraction:
- **Omission** — Missing important facts
- **Hallucination** — Generating facts that weren't present
- **Conflation** — Merging distinct facts incorrectly
- **Misattribution** — Assigning facts to wrong entities

### The Solution

Store raw data first. Derive later. Never lose the original.

```
Episode (immutable) → Derived Facts → Application
       ↑                    ↓
       └────── recovery ────┘
```

---

## Design Decisions and Their Research Basis

### 1. Multiple Memory Types

**Design**: Engram maintains five distinct memory types (working, episodic, structured, semantic, procedural) rather than a single undifferentiated store.

**Research basis**:

The cognitive science literature strongly supports distinct memory systems:

> "The distinction between episodic and semantic memory has been one of the most influential in cognitive psychology... supported by neuroimaging studies showing different patterns of brain activation."
> — [Tulving, 1972; Squire, 2004](https://pmc.ncbi.nlm.nih.gov/articles/PMC2952732/)

> "The medial temporal lobe memory system is needed for the acquisition of episodic and semantic memory... Different brain systems support declarative versus nondeclarative memory."
> — [Squire & Zola, 1996](https://pubmed.ncbi.nlm.nih.gov/8905164/)

Key evidence:
- Patients with hippocampal damage lose episodic memory while retaining semantic facts (H.M., [Scoville & Milner, 1957](https://pubmed.ncbi.nlm.nih.gov/13432090/))
- Procedural memory is neurally distinct from declarative memory ([Milner, 1962](https://www.sciencedirect.com/topics/neuroscience/implicit-memory))
- Working memory has limited capacity and different dynamics than long-term storage ([Baddeley, 1974](https://en.wikipedia.org/wiki/Baddeley%27s_model_of_working_memory))

---

### 2. Ground Truth Preservation

**Design**: Raw episodes are stored verbatim and immutably. All derived knowledge (facts, semantics) traces back to source episodes.

**Research basis**:

Memory research shows that information is lost during consolidation and retrieval:

> "Memories are not simply recordings of the past but are constructed in the present... Both episodic and semantic memory are constructive."
> — [Constructing Memories, Cognitive Science 2025](https://onlinelibrary.wiley.com/doi/10.1111/cogs.70113)

> "Memory is reconstructive rather than reproductive. Each act of remembering involves reconstruction that can introduce distortion."
> — [Bartlett, 1932; Loftus, 1979](https://en.wikipedia.org/wiki/Reconstructive_memory)

This is a problem for AI systems. Recent benchmarks show alarming error rates:

> "All systems achieve answer accuracies below 56%, with both hallucination rate and omission rate remaining high... Systems suffer omission rates above 50%, primarily stemming from insufficient coverage in memory extraction."
> — [HaluMem: Hallucinations in LLM Memory, 2024](https://arxiv.org/html/2511.03506)

**Engineering implication**: If derived memories can be wrong, the original is needed to recover. Engram preserves ground truth so extraction errors are correctable.

---

### 3. Confidence Tracking

**Design**: Every memory carries a composite confidence score combining multiple signals:
- **Extraction method** (50%): `verbatim` (1.0), `extracted` (0.9), `inferred` (0.6)
- **Corroboration** (25%): Multiple sources increase confidence
- **Recency** (15%): Recently confirmed facts score higher
- **Verification** (10%): Format checks (valid email, reasonable date)

Scores are fully auditable — every confidence value can be explained.

**Research basis**:

Metacognitive research shows that humans track confidence in their memories:

> "Feeling of knowing (FOK) judgments are predictions about future memory performance... People can accurately assess whether they will be able to recognize information they cannot currently recall."
> — [Hart, 1965; Metcalfe, 1986](https://pubmed.ncbi.nlm.nih.gov/14296044/)

> "Metamemory—knowledge about one's own memory capabilities and strategies—is a critical component of memory function."
> — [Nelson & Narens, 1990](https://link.springer.com/chapter/10.1007/978-1-4612-3202-8_7)

Signal detection theory provides a formal framework:

> "Memory decisions involve both the strength of the memory trace and a criterion for responding. Confidence reflects distance from the criterion."
> — [Wixted & Mickes, 2010](https://pubmed.ncbi.nlm.nih.gov/20192556/)

**Engineering implication**: Retrieval systems should expose confidence so applications can filter by reliability. Composite scoring with corroboration and recency factors provides richer signal than extraction method alone.

---

### 4. Intelligent Forgetting

**Design**: Memories decay over time based on importance and access patterns. Unimportant memories fade; important ones persist.

**Research basis**:

The Ebbinghaus forgetting curve (1885) established that memory decays exponentially:

> "Retention = e^(-t/S), where t is time and S is memory strength. Memory drops ~56% within the first hour, ~66% within 24 hours."
> — [Ebbinghaus, 1885](https://pmc.ncbi.nlm.nih.gov/articles/PMC4492928/)

Modern research shows forgetting is adaptive, not a failure:

> "Forgetting is not simply a failure of memory but may reflect an active process that helps optimize memory function by reducing interference."
> — [Anderson & Hulbert, 2021](https://www.annualreviews.org/doi/10.1146/annurev-psych-072720-094140)

> "Memory encoding can be framed as compression under resource constraints. Semantic memory encodes broad regularities; episodic memory retains specifics for key experiences."
> — [Adaptive Compression Framework, Nature Reviews Psychology 2025](https://www.nature.com/articles/s44159-025-00458-6)

Selective strengthening occurs naturally through retrieval:

> "Each successful retrieval strengthens the memory trace. The spacing effect shows distributed practice outperforms massed practice."
> — [Pimsleur, 1967; Wozniak, 1985](https://supermemo.guru/wiki/History_of_the_optimization_of_repetition_spacing)

**Engineering implication**: Unbounded memory growth causes relevance problems. Forgetting keeps the store focused on what matters.

---

### 5. Deferred Consolidation

**Design**: Expensive LLM work (semantic inference, pattern detection) happens in background batches, not on the critical path.

**Research basis**:

Memory consolidation in the brain is a slow, offline process:

> "Systems consolidation occurs over days to years. Memories gradually shift from hippocampus-dependent to neocortex-dependent."
> — [McClelland et al., 1995](https://pubmed.ncbi.nlm.nih.gov/7624455/)

> "Sleep plays a critical role in memory consolidation. Slow-wave sleep supports declarative memory consolidation."
> — [Walker & Stickgold, 2006](https://www.nature.com/articles/nrn1739)

> "REM sleep cueing impairs recognition memory for cued memories. Memory benefit was driven by the product of SWS and REM sleep cueing, not independent contributions."
> — [Yuksel et al., 2025](https://www.nature.com/articles/s42003-025-07868-5)

**Note**: Sleep research is cited for context, not because Engram models sleep stages. Deferred consolidation is an engineering pattern (batch processing is cheaper and allows error correction), not a neuroscience simulation. The sleep literature shows that consolidation is naturally an offline process, which validates separating fast encoding from slow consolidation.

Recent neuroscience shows multiple consolidation pathways:

> "Long-term memory can form independently of short-term memory... We now have evidence of at least two distinct pathways to memory formation."
> — [Max Planck Florida, December 2024](https://www.news-medical.net/news/20241206/Researchers-discover-new-pathway-to-forming-long-term-memories-in-the-brain.aspx)

**Engineering implication**: Real-time LLM extraction is expensive and error-prone. Batched offline processing is cheaper and allows error correction.

---

### 6. Two-Phase Semantic Processing

**Design**: Engram separates encoding into two phases:
1. `encode()` — Stores verbatim + extracts deterministic facts (immediate, no LLM)
2. `consolidate()` — Extracts relationships, preferences, patterns (background, LLM)

**Research basis**:

The levels of processing framework shows deeper encoding produces stronger memories:

> "Memory strength depends on how deeply information is processed during encoding. Semantic processing (meaning) produces better retention than structural processing (surface features)."
> — [Craik & Lockhart, 1972](http://wixtedlab.ucsd.edu/publications/Psych%20218/Craik_Lockhart_1972.pdf)

> "Elaboration—connecting new information to existing knowledge—enhances depth and produces more retrieval cues."
> — [Craik & Tulving, 1975](https://www.simplypsychology.org/levelsofprocessing.html)

**Engineering implication**: Deep processing improves retrieval, but LLM extraction is slow and error-prone. Engram defers LLM work to background consolidation where errors can be caught and corrected.

---

## Known Limitations

Engram uses cognitive science as design inspiration, not strict implementation.

### What Engram Doesn't Model

| Research Concept | What the Paper Says | What Engram Implements |
|-----------------|---------------------|------------------------|
| **Surprise-based encoding** (Nagy et al. 2025) | Episodic memory acts as "life raft" for surprising experiences that don't compress well | `_calculate_surprise()` boosts importance of novel content (configurable via `surprise_scoring_enabled`, `surprise_weight`) |
| **SWS×REM interaction** (Yuksel et al. 2025) | Memory benefit comes from product of SWS and REM, not independent contributions | Simple time-based decay; no sleep-stage modeling |
| **REM as forgetting facilitator** (Yuksel et al. 2025) | REM cueing may actively facilitate forgetting of non-cued memories | Time-based decay; no sleep-phase forgetting modeling |
| **Compression-as-learning** (Nagy et al. 2025) | Semantic learning IS learning to compress; measures encoding efficiency | Extract facts but don't model compression ratios |

### Theoretical Limitations

| Theory | Known Issues |
|--------|--------------|
| Atkinson-Shiffrin | Superseded by Baddeley's Working Memory Model; STM is not unitary |
| Episodic/Semantic | Better understood as continuum than discrete categories ([McKoon et al., 1986](https://pubmed.ncbi.nlm.nih.gov/2939185/)) |
| Consolidation | Standard theory actively debated; Multiple Trace Theory challenges it ([Nadel & Moscovitch, 1997](https://pmc.ncbi.nlm.nih.gov/articles/PMC9720899/)) |
| Ebbinghaus decay | Oversimplification; interference and context matter; may follow power law |

**Why use them anyway**: The core insights are valuable for engineering even if the details are debated:
- Different information needs different handling
- Ground truth preservation prevents error propagation
- Some forgetting is necessary for relevance
- Offline processing is cheaper and more reliable

---

## Further Reading

- [Competitive Analysis](competitive.md) — How Engram compares to alternatives

---

## Key References

### Foundational Cognitive Science
- Atkinson & Shiffrin (1968). Human memory: A proposed system and its control processes.
- Tulving (1972). Episodic and semantic memory.
- Craik & Lockhart (1972). Levels of processing: A framework for memory research.
- Ebbinghaus (1885). Memory: A contribution to experimental psychology.
- Squire & Zola (1996). Structure and function of declarative and nondeclarative memory systems.

### Recent Research (2024-2025)
- [HaluMem: Hallucinations in LLM Memory Systems](https://arxiv.org/html/2511.03506)
- [Constructive Memory](https://onlinelibrary.wiley.com/doi/10.1111/cogs.70113) — Episodic and semantic memory are constructive
- [Adaptive Compression Framework](https://www.nature.com/articles/s44159-025-00458-6) (Nagy et al. 2025) — Memory as compression under constraints. Semantic memory encodes regularities; episodic memory preserves surprising experiences. *Engram implements surprise-based importance scoring via `_calculate_surprise()` to prioritize novel content.*
- [Sleep and Memory Consolidation](https://www.nature.com/articles/s42003-025-07868-5) (Yuksel et al. 2025) — Memory benefit from SWS×REM product interaction; REM cueing alone impairs memory. *Cited for context; Engram does not model sleep stages.*
- [Dual Pathways to LTM](https://www.news-medical.net/news/20241206/Researchers-discover-new-pathway-to-forming-long-term-memories-in-the-brain.aspx) — LTM can form independently of STM
- [Molecular Memory Timers](https://www.sciencedaily.com/releases/2025/11/251130050712.htm) — Importance gating in memory persistence
- [Testing Effect / Retrieval Practice](https://pmc.ncbi.nlm.nih.gov/articles/PMC5912918/) — Roediger & Karpicke 2006: "Repeated remembering strengthens memories much more so than repeated learning." Considered [one of the most robust phenomena in memory research](https://www.sciencedirect.com/topics/psychology/testing-effect). Engram's `consolidation_strength` field is based on this finding.
