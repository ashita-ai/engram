# Memory Systems

Why Engram uses multiple memory types.

## The Argument

Not all information is the same. An email address is different from an inferred preference. A verbatim quote is different from a summarized understanding. Treating all memory uniformly loses important distinctions.

The cognitive science literature supports this with evidence for distinct memory systems.

## Foundational Theory: Atkinson-Shiffrin (1968)

Richard Atkinson and Richard Shiffrin proposed that human memory consists of distinct stores:

1. **Sensory Register** — Brief storage of raw input (milliseconds)
2. **Short-Term Store** — Limited capacity (~7±2 items), temporary (seconds to minutes)
3. **Long-Term Store** — Large capacity, persistent

Each store has different characteristics: capacity limits, decay rates, access patterns.

## Evidence for Distinct Systems

**Neuropsychological dissociations**:
- Patient H.M. ([Scoville & Milner, 1957](https://pubmed.ncbi.nlm.nih.gov/13432090/)): Hippocampal damage destroyed ability to form new episodic memories while preserving existing semantic knowledge
- Patients with semantic dementia: Progressive loss of semantic memory while episodic memory remains intact

**Serial position effect**:
- First items in a list → long-term memory (primacy)
- Last items → short-term memory (recency)
- Different decay characteristics

**Implicit vs explicit**:
- H.M. could learn motor skills (mirror drawing) despite no memory of practice
- Procedural memory is neurally distinct from declarative ([Milner, 1962](https://www.sciencedirect.com/topics/neuroscience/implicit-memory))

## Squire's Taxonomy

Larry Squire refined the distinction:

```
Long-Term Memory
├── Declarative (explicit)
│   ├── Episodic (events, experiences)
│   └── Semantic (facts, knowledge)
└── Non-declarative (implicit)
    └── Procedural (skills, habits)
```

> "The medial temporal lobe memory system is needed for declarative memory... Different brain systems support declarative versus nondeclarative memory."
> — [Squire & Zola, 1996](https://pubmed.ncbi.nlm.nih.gov/8905164/)

## Engram Application

Engram implements six memory types based on this research:

| Engram Type | Cognitive Basis | Key Characteristics |
|-------------|-----------------|---------------------|
| **Working** | Short-term store | Limited capacity, volatile, current context |
| **Episodic** | Episodic LTM | Verbatim events, timestamped, decays |
| **Factual** | Semantic LTM | High-confidence extracted facts |
| **Semantic** | Semantic LTM | Inferred knowledge, variable confidence |
| **Procedural** | Procedural memory | Behavioral patterns, slow to change |
| **Scratchpad** | Working memory extension | Agent execution state |

**Why this matters for AI**:
- **Working** has size limits → must summarize or offload
- **Episodic** preserves ground truth → errors can be corrected
- **Factual vs Semantic** distinguishes confidence levels → trustworthy retrieval
- **Procedural** captures implicit preferences → personalization

## Engineering Extensions

Two types extend beyond cognitive science:

**Factual memory**: Cognitive science treats all facts as semantic. Engram subdivides based on extraction method:
- `factual` = pattern-extracted, deterministic, high confidence
- `semantic` = LLM-inferred, variable confidence

This addresses the practical problem of LLM extraction errors.

**Scratchpad**: Agent execution state (file paths, task progress) requires lossless storage. No cognitive analog—pure engineering.

## References

- [Squire & Zola (1996). Structure and function of declarative and nondeclarative memory systems](https://pubmed.ncbi.nlm.nih.gov/8905164/)
- [Tulving (1972). Episodic and semantic memory](https://pmc.ncbi.nlm.nih.gov/articles/PMC2952732/)
- [Scoville & Milner (1957). Loss of recent memory after bilateral hippocampal lesions](https://pubmed.ncbi.nlm.nih.gov/13432090/)
- [Baddeley (1974). Working Memory Model](https://en.wikipedia.org/wiki/Baddeley%27s_model_of_working_memory)
- [Atkinson & Shiffrin (1968). Human memory: A proposed system](https://app.nova.edu/toolbox/instructionalproducts/edd8124/articles/1968-Atkinson_and_Shiffrin.pdf)
