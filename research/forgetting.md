# Forgetting Curves

Why Engram implements intelligent forgetting.

## The Argument

Unbounded memory growth causes problems:
- Retrieval gets slower
- Irrelevant memories create noise
- Storage costs increase indefinitely
- Old, stale information pollutes results

Forgetting is not a bug—it's a feature. The research supports this.

## Ebbinghaus Forgetting Curve (1885)

Hermann Ebbinghaus conducted the first quantitative studies of memory. He memorized over 2,000 nonsense syllables and measured retention at intervals.

Memory decay follows an exponential curve:

```
R = e^(-t/S)
```

Where:
- R = retention (0 to 1)
- t = time since learning
- S = memory strength

Key findings:
- ~56% forgotten within the first hour
- ~66% forgotten within 24 hours
- ~75% forgotten within a week
- Curve asymptotes—some memories persist indefinitely

> "A direct replication of Ebbinghaus' classic forgetting curve experiment confirms the exponential nature of forgetting."
> — [Murre & Dros, 2015 (PLOS ONE)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4492928/)

## Forgetting as Adaptive

Modern research shows forgetting optimizes memory function:

> "Forgetting is not simply a failure of memory but may reflect an active process that helps optimize memory function by reducing interference."
> — [Anderson & Hulbert, 2021](https://www.annualreviews.org/doi/10.1146/annurev-psych-072720-094140)

> "Memory encoding can be framed as compression under resource constraints. Semantic memory encodes regularities; episodic memory retains specifics for key experiences."
> — [Adaptive Compression Framework, Nature Reviews Psychology 2025](https://www.nature.com/articles/s44159-025-00458-6)

Selective forgetting serves multiple functions:
- **Reduces interference** — Old memories don't compete with new ones
- **Prioritizes important information** — High-salience memories resist decay
- **Compresses knowledge** — Episodes abstract into semantic understanding

## Factors Affecting Decay

| Factor | Effect on Decay |
|--------|-----------------|
| **Importance** | High importance → slower decay |
| **Meaningfulness** | Meaningful content → slower decay |
| **Repetition** | Each access flattens the curve |
| **Emotional salience** | Emotional memories → slower decay |
| **Sleep** | Consolidation during sleep preserves memories |

## Engram Application

The `decay()` operation implements importance-weighted forgetting:

```python
# Pseudocode
for memory in episodic_memories:
    time_elapsed = now - memory.last_accessed
    decay_rate = base_rate / memory.importance

    memory.strength *= exp(-time_elapsed / decay_rate)

    if memory.strength < threshold:
        archive_or_delete(memory)
```

Key design decisions:
- **Importance protects against decay** — High-importance memories persist
- **Access reinforces memory** — Retrieved memories get stronger (spaced repetition)
- **Different rates per type** — Episodic decays faster than semantic
- **Threshold-based cleanup** — Memories below threshold are archived

**Why this matters**:
- Keeps memory store relevant and fast
- Naturally surfaces frequently-accessed information
- Reduces storage costs over time
- Prevents irrelevant old memories from polluting retrieval

## References

- [Murre & Dros (2015). Replication and Analysis of Ebbinghaus' Forgetting Curve](https://pmc.ncbi.nlm.nih.gov/articles/PMC4492928/)
- [Anderson & Hulbert (2021). Active Forgetting](https://www.annualreviews.org/doi/10.1146/annurev-psych-072720-094140)
- [Nature Reviews Psychology (2025). Adaptive Compression Framework](https://www.nature.com/articles/s44159-025-00458-6)
- [SuperMemo: Exponential nature of forgetting](https://supermemo.guru/wiki/Exponential_nature_of_forgetting)
