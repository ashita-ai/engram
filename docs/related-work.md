# Related Work

This document surveys LLM memory systems and recent cognitive science research relevant to Engram's design.

## LLM Memory Architectures (2024-2025)

Three main architectural patterns have emerged for LLM memory:

| Approach | Core Idea |
|----------|-----------|
| **Tiered memory** | OS-like memory hierarchy (RAM/disk metaphor) |
| **Knowledge graphs** | Temporal knowledge graphs with bi-temporal tracking |
| **Agentic memory** | Zettelkasten-inspired self-organizing networks |

### Key Research Papers

| Paper | Date | Key Contribution |
|-------|------|------------------|
| Human-inspired Episodic Memory for Infinite Context LLMs | Jul 2024 | Episodic memory for long context |
| Titans: Learning to Memorize at Test Time | Dec 2024 | Test-time memory learning |
| From RAG to Memory: Non-Parametric Continual Learning | Feb 2025 | RAG → true memory distinction |
| A-MEM: Agentic Memory for LLM Agents | Feb 2025 | Zettelkasten-inspired organization |
| Zep: Temporal Knowledge Graph Architecture | Jan 2025 | Bi-temporal knowledge graphs |

### Bi-Temporal Models

Some systems track two timestamps for every fact:
- **Event time (T)**: When the fact was actually true
- **Ingestion time (T')**: When we learned the fact

This enables temporal queries like "What did we know as of last Tuesday?" and retroactive corrections.

### Self-Organizing Memory (A-MEM)

The A-MEM paper introduces a Zettelkasten-inspired approach where memories self-organize through dynamic linking. Instead of fixed memory types, the agent decides how to organize memories based on content.

**Open question for Engram**: Could self-organization replace fixed memory types? What's lost if memories aren't typed?

---

## Cognitive Science Updates (2024-2025)

### Dual Pathways to Long-Term Memory (December 2024)

Max Planck Florida researchers discovered that long-term memory can form independently of short-term memory:

> "The prevailing theory suggested a single pathway, where short-term memories were consolidated into long-term memories. However, we now have strong evidence of at least two distinct pathways to memory formation."

**Implication**: The standard Atkinson-Shiffrin STM→LTM transfer model may be incomplete.

Source: [Researchers discover new pathway to forming long-term memories](https://www.news-medical.net/news/20241206/Researchers-discover-new-pathway-to-forming-long-term-memories-in-the-brain.aspx)

### Molecular Memory Timers (November 2024)

Research published in Nature shows memory is orchestrated by molecular "timers" across brain regions:

- Multiple brain regions gradually reorganize memories into more enduring forms
- The thalamus acts as gatekeeper — decides which memories should persist
- Gates assess salience and promote durability

**Implication**: Importance scoring and selective consolidation have biological basis.

Source: [Why some memories last a lifetime while others fade fast](https://www.sciencedaily.com/releases/2025/11/251130050712.htm)

### Adaptive Compression Framework (Nature Reviews Psychology, 2025)

Memory encoding framed as compression under resource constraints:

- Semantic memory encodes broad regularities
- Episodic memory retains specific information for key experiences
- Surprising events trigger learning updates to the semantic model

**Implication**: Supports approach of storing episodes, extracting patterns, forgetting noise.

Source: [Adaptive compression as a unifying framework](https://www.nature.com/articles/s44159-025-00458-6)

### Constructive Memory (Cognitive Science, 2025)

Both episodic AND semantic memory are constructive:

- Traditional view of semantic memory storing "invariant knowledge structures" is likely false
- Episodic and semantic memory share causal mechanisms
- Domain-general system for representing and navigating relations

**Implication**: Our clean factual/semantic separation is more engineering than biology.

Source: [Constructing Memories, Episodic and Semantic](https://onlinelibrary.wiley.com/doi/10.1111/cogs.70113)

### Sleep and Emotional Memory (2025)

Both slow-wave and REM sleep contribute to emotional memory consolidation:

- Different sleep stages serve different consolidation functions
- Emotional salience affects consolidation pathway

Source: [Both slow wave and rapid eye movement sleep contribute to emotional memory consolidation](https://www.nature.com/articles/s42003-025-07868-5)

---

## Research Questions for Engram

### Self-Organization vs Fixed Types

The A-MEM approach raises questions:
- Could self-organization replace our fixed six types?
- Is the Zettelkasten metaphor better than the cognitive science metaphor?
- What's lost if memories aren't typed?

### Bi-Temporal Model

Should episodic memories track both event time and ingestion time? This would enable:
- "What did we know as of last Tuesday?"
- Retroactive corrections
- Temporal reasoning

### Dual Memory Pathways

New neuroscience suggests LTM can form independently of STM. Does this invalidate our working → episodic → semantic flow? Should some memories go directly to semantic?

### Memory as Compression

The adaptive compression framework aligns well with Engram's approach but adds nuance:
- Semantic = regularities (what's normal)
- Episodic = exceptions (what's surprising/important)

Should we prioritize surprising/important episodes? Is our decay model aligned with this view?

---

## Sources

### LLM Memory Systems
- [A-MEM Paper (arXiv)](https://arxiv.org/abs/2502.12110)
- [Zep Paper (arXiv)](https://arxiv.org/abs/2501.13956)
- [Graphiti GitHub](https://github.com/getzep/graphiti)
- [Agent Memory Paper List](https://github.com/Shichun-Liu/Agent-Memory-Paper-List)

### Cognitive Science
- [Dual memory pathways](https://www.news-medical.net/news/20241206/Researchers-discover-new-pathway-to-forming-long-term-memories-in-the-brain.aspx)
- [Molecular memory timers](https://www.sciencedaily.com/releases/2025/11/251130050712.htm)
- [Adaptive compression framework](https://www.nature.com/articles/s44159-025-00458-6)
- [Constructive memory](https://onlinelibrary.wiley.com/doi/10.1111/cogs.70113)
