# Retrieval and Strengthening

Why accessing memories makes them persist.

## The Argument

Memories that are used should persist. Memories that are never accessed should fade. This creates a natural relevance filter—the memory store adapts to actual usage patterns.

The spaced repetition literature provides the foundation.

## The Testing Effect

Actively retrieving information strengthens memory more than passive review:

> "Each successful retrieval strengthens the memory trace."
> — [Roediger & Karpicke, 2006](https://pubmed.ncbi.nlm.nih.gov/16507066/)

This is counterintuitive: testing feels harder than re-reading, but produces better retention.

## Spaced Repetition

Optimal retention occurs when reviews are spaced at increasing intervals:

**Pimsleur (1967)** — Graduated-interval recall:
```
5 sec → 25 sec → 2 min → 10 min → 1 hr → 5 hr → 1 day → 5 days → ...
```

**Wozniak (1985)** — SuperMemo algorithm:
- Track individual item difficulty
- Optimize intervals per item
- 200%+ retention improvement over massed practice

> "Difficulty of retrieval correlates with learning benefit—desirable difficulty."
> — [Bjork, 1994](https://www.researchgate.net/publication/281322665_Memory_and_Metamemory_Considerations_in_the_Training_of_Human_Beings)

## Why Retrieval Strengthens Memory

Theories:

1. **Elaborative retrieval** — Recalling activates related concepts, creating more connections
2. **Retrieval-specific encoding** — The act of retrieval creates a new memory trace
3. **Desirable difficulty** — Effortful retrieval signals importance

## Engram Application

The `recall()` operation updates memory strength:

```python
async def recall(query, memory_types, min_confidence):
    memories = await search(query, memory_types, min_confidence)

    for memory in memories:
        # Retrieval strengthens the memory
        memory.access_count += 1
        memory.last_accessed = now()
        memory.strength = reinforce(memory.strength, memory.access_count)
        await update(memory)

    return memories
```

Strength update formula:
```python
def reinforce(current_strength, access_count):
    # Each access provides diminishing but real reinforcement
    boost = base_boost * (1 / log(access_count + 1))
    return min(current_strength + boost, max_strength)
```

**How this creates relevance filtering**:
1. User queries about topic X
2. Memories about X are retrieved and strengthened
3. Memories about unqueried topics decay
4. Over time, memory store reflects actual usage patterns

## Interaction with Decay

Retrieval and decay work together:

| Memory State | Decay Rate | Access Effect |
|--------------|------------|---------------|
| Never accessed | Fast decay | — |
| Occasionally accessed | Slow decay | Strength boost |
| Frequently accessed | Minimal decay | Approaches max strength |

This creates three tiers:
- **Active** — Frequently used, strong, fast retrieval
- **Accessible** — Occasionally used, moderate strength
- **Fading** — Unused, decaying toward threshold

## References

- [Roediger & Karpicke (2006). The power of testing memory](https://pubmed.ncbi.nlm.nih.gov/16507066/)
- [Bjork (1994). Memory and metamemory considerations](https://www.researchgate.net/publication/281322665_Memory_and_Metamemory_Considerations_in_the_Training_of_Human_Beings)
- [Pimsleur (1967). A memory schedule](https://en.wikipedia.org/wiki/Spaced_repetition)
- [SuperMemo: History of spaced repetition](https://supermemo.guru/wiki/History_of_the_optimization_of_repetition_spacing)
- [Duolingo Research: A Trainable Spaced Repetition Model](https://research.duolingo.com/papers/settles.acl16.pdf)
