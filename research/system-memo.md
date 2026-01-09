# Engram: An Intelligent Memory System

**Memo** | January 2025

---

## The Problem

Current AI memory systems suffer from a fundamental accuracy problem. The HaluMem benchmark (2024) shows all systems achieve below 56% accuracy, with omission rates above 50%. The root cause: these systems discard raw interactions after LLM extraction, making errors permanent and unrecoverable.

## Our Approach

Engram treats memory as a **layered system** where raw truth flows upward through increasingly refined representations, each layer serving a distinct purpose and backed by cognitive science research.

### The Memory Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│  PROCEDURAL    Behavioral patterns ("User prefers concise")     │
│                Promoted from semantic via Testing Effect        │
├─────────────────────────────────────────────────────────────────┤
│  SEMANTIC      Inferred knowledge, linked via A-MEM             │
│                Strengthened through repeated consolidation      │
├─────────────────────────────────────────────────────────────────┤
│  FACTUAL       Pattern-extracted facts (emails, dates, phones)  │
│                Deterministic extraction, no LLM hallucination   │
├─────────────────────────────────────────────────────────────────┤
│  EPISODIC      Immutable ground truth (verbatim interactions)   │
│                Recovery path when derived memories are wrong    │
└─────────────────────────────────────────────────────────────────┘
```

**Key insight**: Information flows upward, but ground truth is never discarded. When a semantic memory seems wrong, we can trace it back to source episodes and re-extract.

### How Components Work Together

**1. Encode (Immediate)**
Raw interaction → Episodic memory (immutable) → Pattern extraction → Factual memories

No LLM on the write path. Deterministic patterns (regex for emails, validators for dates) extract high-confidence facts immediately. LLM work is deferred.

**2. Consolidate (Background)**
Episodes batch → LLM extraction → Semantic memories → Dynamic linking → Strength updates

During consolidation, the LLM extracts semantic knowledge. New memories link to existing ones via embedding similarity (A-MEM research). Existing memories that participate in consolidation get **strengthened** (Testing Effect: "repeated remembering strengthens memories more than repeated learning").

**3. Recall (Query Time)**
Query → Multi-type retrieval → Confidence filtering → Link traversal → RIF suppression

Retrieval pulls from multiple memory types, filters by confidence, and follows links for multi-hop reasoning. Optionally, RIF (Retrieval-Induced Forgetting) suppresses near-miss competitors, naturally pruning redundant memories.

**4. Decay (Periodic)**
Time passes → Unaccessed memories decay → Low-confidence pruned → Important memories persist

Ebbinghaus-inspired forgetting keeps the store relevant. Memories decay based on time and access patterns. High-importance, frequently-accessed memories persist; stale, unused ones fade.

### Why This Is Intelligent

| Behavior | Mechanism | Research Basis |
|----------|-----------|----------------|
| Memories strengthen with use | `consolidation_strength` increases during consolidation | Testing Effect (Roediger & Karpicke 2006) |
| Related memories link together | Dynamic linking via semantic similarity | A-MEM (NeurIPS 2025) |
| Competitors get suppressed | RIF decays near-miss confidence | Anderson, Bjork & Bjork (1994) |
| Unimportant memories fade | Time-based decay with access weighting | Ebbinghaus (1885) |
| Patterns promote to procedures | High-strength semantic → procedural | Cognitive Workspace (2024) |
| Errors are recoverable | Episodic ground truth preserved | Memory reconstruction research |

The system exhibits **emergent intelligence**: memories that matter naturally persist and strengthen, while noise fades away. This isn't programmed behavior—it emerges from the interaction of consolidation, retrieval, and decay.

### What Makes Us Different

| Capability | Engram | Competitors |
|------------|--------|-------------|
| Ground truth preservation | Yes (episodic immutable) | Mem0: No, Zep: Yes, Letta: Partial |
| Composite confidence scoring | Yes (method + corroboration + recency) | None |
| Strength through consolidation | Yes (Testing Effect) | None track this |
| Retrieval-induced forgetting | Yes (opt-in) | None |
| Principled decay | Yes (Ebbinghaus-inspired) | None |

## Conclusion

Engram is not just a memory store—it's a system where memories **compete, strengthen, link, and fade** based on principles from cognitive science. The result is a memory layer that naturally surfaces what matters and forgets what doesn't, while maintaining the ground truth needed to recover from errors.

This is memory that learns.

---

*References: HaluMem (arxiv.org/html/2511.03506), Testing Effect (PMC5912918), A-MEM (arxiv.org/abs/2502.12110), RIF (PMID 7931095), Cognitive Workspace (arxiv.org/abs/2508.13171)*
