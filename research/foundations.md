# Theoretical Foundations

Engram is inspired by cognitive science research. This page summarizes the key theories. We use these as design inspiration, not strict implementation.

## Memory Systems

**Source**: Tulving (1972), Squire & Zola (1996)

The brain has distinct memory systems with different characteristics:

| System | Type | Characteristics |
|--------|------|-----------------|
| **Working memory** | Short-term | Limited capacity (~7 items), volatile |
| **Episodic** | Declarative | Events with context, decays over time |
| **Semantic** | Declarative | Facts without context, stable |
| **Procedural** | Non-declarative | Skills/habits, implicit, very stable |

**Engram application**: Six memory types with different storage and retrieval strategies.

**Key references**:
- [Squire & Zola (1996)](https://pubmed.ncbi.nlm.nih.gov/8905164/)
- [Tulving (1972)](https://pmc.ncbi.nlm.nih.gov/articles/PMC2952732/)

---

## Forgetting Curves

**Source**: Ebbinghaus (1885)

Memory decays exponentially: `R = e^(-t/S)`

- ~56% forgotten within first hour
- ~66% forgotten within 24 hours
- Curve asymptotes — some memories persist indefinitely
- Access reinforces memory (spacing effect)

**Engram application**: `decay()` operation with importance-weighted forgetting.

**Key references**:
- [Murre & Dros (2015)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4492928/)
- [Anderson & Hulbert (2021)](https://www.annualreviews.org/doi/10.1146/annurev-psych-072720-094140)

---

## Memory Consolidation

**Source**: McClelland et al. (1995), Walker & Stickgold (2006)

Memories consolidate over time:
- Episodic → semantic transformation
- Occurs during sleep/offline periods
- Slow process (days to years)

**Engram application**: `consolidate()` runs in background, not on critical path.

**Key references**:
- [McClelland et al. (1995)](https://pubmed.ncbi.nlm.nih.gov/7624455/)
- [Walker & Stickgold (2006)](https://www.nature.com/articles/nrn1739)

---

## Levels of Processing

**Source**: Craik & Lockhart (1972)

Deeper processing produces stronger memories:
- Shallow: surface features
- Deep: semantic meaning

**Engram application**: `encode()` extracts meaning, not just stores text.

**Key reference**: [Craik & Lockhart (1972)](http://wixtedlab.ucsd.edu/publications/Psych%20218/Craik_Lockhart_1972.pdf)

---

## Retrieval Strengthening

**Source**: Roediger & Karpicke (2006)

Each retrieval strengthens memory. Testing > passive review.

**Engram application**: `recall()` updates memory strength.

**Key reference**: [Roediger & Karpicke (2006)](https://pubmed.ncbi.nlm.nih.gov/16507066/)

---

## Known Limitations

These theories have limitations and ongoing debates:

| Theory | Known Issues |
|--------|--------------|
| Multi-store model | Superseded by Baddeley's Working Memory Model |
| Episodic/semantic distinction | Better understood as continuum |
| Consolidation | Standard theory actively debated |
| Ebbinghaus curves | May follow power law, not exponential |

We use these as engineering inspiration despite the debates. The core insights remain useful:
- Different information needs different handling
- Some forgetting is necessary
- Offline processing is cheaper and more reliable
