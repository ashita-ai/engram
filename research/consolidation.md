# Consolidation

Why Engram defers expensive processing to background jobs.

## The Argument

Running LLM inference on every message is:
- Expensive (API costs)
- Slow (latency on critical path)
- Error-prone (extraction happens before full context is available)

The brain doesn't consolidate memories in real-time either. Consolidation is a slow, offline process. This provides a biological precedent for deferred processing.

## Systems Consolidation

Memory consolidation is the process by which newly formed memories become stable and integrated. Two levels:

1. **Synaptic consolidation** — Hours; cellular-level stabilization
2. **Systems consolidation** — Days to years; reorganization across brain regions

> "During systems consolidation, memories gradually shift from hippocampus-dependent to neocortex-dependent."
> — [McClelland et al., 1995](https://pubmed.ncbi.nlm.nih.gov/7624455/)

## The Transformation

Episodic memories (events with contextual details) consolidate into semantic memories (decontextualized facts):

```
Episode: "On Tuesday at 3pm, user asked about PostgreSQL B-tree indexes"
    ↓ consolidation
Semantic: "User works with PostgreSQL databases"
```

The transformation:
- **Extracts patterns** across multiple episodes
- **Loses contextual details** (time, place, exact wording)
- **Creates stable knowledge** that resists forgetting

## Sleep and Consolidation

Consolidation occurs primarily during sleep:

> "Sleep plays a critical role in memory consolidation. Slow-wave sleep supports declarative memory consolidation."
> — [Walker & Stickgold, 2006](https://www.nature.com/articles/nrn1739)

> "Both slow-wave and REM sleep contribute to emotional memory consolidation, with different sleep stages serving different functions."
> — [Nature Communications Biology, 2025](https://www.nature.com/articles/s42003-025-07868-5)

Why offline processing makes sense:
- No time pressure
- Full context available
- Can process multiple episodes together
- Errors can be detected and corrected

## Recent Findings: Multiple Pathways

New research complicates the traditional model:

> "Long-term memory can form independently of short-term memory... We now have evidence of at least two distinct pathways to memory formation."
> — [Max Planck Florida, December 2024](https://www.news-medical.net/news/20241206/Researchers-discover-new-pathway-to-forming-long-term-memories-in-the-brain.aspx)

Implication: Some memories may bypass working memory entirely. This could inform future Engram designs.

## Engram Application

The `consolidate()` operation mirrors biological consolidation:

```python
# Runs as background job, not on critical path
async def consolidate():
    # Gather recent episodes
    episodes = await get_episodes_since(last_consolidation)

    # Batch LLM inference (cheaper, more accurate with context)
    inferences = await batch_infer(episodes)

    # Create semantic memories with source links
    for inference in inferences:
        await store_semantic(
            content=inference.content,
            confidence=inference.confidence,
            sources=inference.episode_ids  # Provenance
        )
```

Key design decisions:
- **Deferred** — Not on the request path
- **Batched** — Multiple episodes processed together
- **Traced** — Semantic memories link to source episodes
- **Recoverable** — Can re-consolidate if extraction improves

**Why this matters**:
- Fast path stays fast (no LLM latency)
- Better accuracy (more context for inference)
- Lower cost (batched API calls)
- Error recovery (ground truth preserved)

## References

- [McClelland et al. (1995). Complementary learning systems in the brain](https://pubmed.ncbi.nlm.nih.gov/7624455/)
- [Walker & Stickgold (2006). Sleep, memory, and plasticity](https://www.nature.com/articles/nrn1739)
- [Nature Communications Biology (2025). Sleep and emotional memory](https://www.nature.com/articles/s42003-025-07868-5)
- [Max Planck Florida (2024). Dual pathways to long-term memory](https://www.news-medical.net/news/20241206/Researchers-discover-new-pathway-to-forming-long-term-memories-in-the-brain.aspx)
- [Tulving (1972). Episodic and semantic memory](https://pmc.ncbi.nlm.nih.gov/articles/PMC2952732/)
