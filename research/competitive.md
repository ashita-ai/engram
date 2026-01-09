# Competitive Analysis

Technical comparison of AI memory systems. Based on published papers and documentation, not marketing claims.

## Summary

| System | Ground Truth | Confidence | Forgetting | Bi-Temporal | Dynamic Linking | Selectivity | RIF |
|--------|--------------|------------|------------|-------------|-----------------|-------------|-----|
| **Engram** | Yes | Yes | Yes | Yes | Yes | Planned | Yes |
| **Mem0** | No | No | No | No | No | No | No |
| **Zep/Graphiti** | Yes | No | No | Yes | Partial | No | No |
| **Letta/MemGPT** | Partial | No | No | No | No | No | No |

## Mem0

**Architecture**: Two-phase LLM extraction on every write.

```
Message → LLM Extract → LLM Categorize (ADD/UPDATE/DELETE/NOOP) → Graph Store
```

**What it does well**:
- Compresses conversation history (26K → 7K tokens claimed)
- Simple API
- Active development

**Limitations**:
- **No ground truth preservation** — Original messages discarded after extraction
- **LLM extraction on write** — Errors become permanent
- **No confidence tracking** — All memories treated as equally reliable
- **No forgetting** — Unbounded growth

**Source**: [Mem0 Paper](https://arxiv.org/abs/2504.19413), [Documentation](https://docs.mem0.ai)

---

## Zep/Graphiti

**Architecture**: Bi-temporal knowledge graph with three subgraphs.

```
Episode Subgraph (raw) ──┬──→ Semantic Entity Subgraph (extracted)
                         └──→ Community Subgraph (aggregated)
```

**What it does well**:
- **Preserves ground truth** in Episode Subgraph
- **Bi-temporal tracking** — `valid_at` and `invalid_at` on edges
- Graph enables relationship queries
- Active open-source development

**Limitations**:
- **No confidence tracking** — Extracted facts lack provenance metadata
- **No principled forgetting** — Graph grows indefinitely
- **Complex schema** — Three interconnected subgraphs
- **LLM extraction on write** — Same error propagation risk as Mem0

**Source**: [Graphiti Paper](https://arxiv.org/abs/2501.00227), [Documentation](https://docs.getzep.com)

---

## Letta/MemGPT

**Architecture**: OS-inspired memory hierarchy.

```
Main Context (working) ←→ Archival Memory (long-term)
         ↓
   Recall Memory (retrieval)
```

**What it does well**:
- Agent-native design
- Self-editing memory
- Long-running conversation support

**Limitations**:
- **Unclear ground truth preservation** — Documentation doesn't specify
- **No confidence tracking**
- **No principled forgetting**
- Focus is on agent architecture, not memory accuracy

**Source**: [MemGPT Paper](https://arxiv.org/abs/2310.08560), [Documentation](https://docs.letta.com)

---

## Engram's Differentiators

### 1. Ground Truth Preservation

Like Zep, we preserve raw interactions. Unlike Zep:
- Explicit immutability guarantees
- Clear separation between episodic (immutable) and derived (mutable)
- Recovery path when extraction fails

### 2. Composite Confidence Scoring

Unique to Engram — confidence is a composite score, not a single tier:

| Factor | Weight | What it measures |
|--------|--------|------------------|
| Extraction method | 50% | verbatim (1.0), extracted (0.9), inferred (0.6) |
| Corroboration | 25% | How many episodes support this fact |
| Recency | 15% | When was this last confirmed |
| Verification | 10% | Format/validity checks passed |

**Key differentiator:** Scores are fully auditable. You can explain *why* confidence is 0.73, not just that it is.

Applications filter by confidence. High-stakes queries use only trusted facts. Facts decay if not re-confirmed.

### 3. Deterministic Extraction First

Most systems use LLM extraction on every write. Engram:
1. Pattern match first (emails, dates, etc.) — no hallucination possible
2. Defer LLM work to background consolidation
3. Batch for efficiency and review

### 4. Principled Forgetting

Inspired by Ebbinghaus forgetting curves:
- Memories decay based on time + access + importance
- Unimportant memories fade; important ones persist
- Keeps the store relevant, prevents unbounded growth

---

## Research Innovations (2024-2025)

Novel approaches from recent literature:

### Bi-Temporal Data Model

**What**: Track both event time (when fact was true) and ingestion time (when we learned it).

**Who has it**: Zep/Graphiti

**Value**: Enables "what did we know on date X?" queries. Useful for debugging and audit.

**Engram**: Implements bi-temporal tracking with explicit `event_at` and `derived_at` timestamps on all derived memories.

### Dynamic Memory Linking (A-MEM)

**What**: Memories form explicit links to related memories, enabling multi-hop reasoning.

**Source**: [A-MEM Paper](https://arxiv.org/abs/2502.12110) (NeurIPS 2025)

**Results**: 2x improvement on multi-hop reasoning benchmarks.

**Engram**: Implements dynamic linking via `related_ids` field on semantic and procedural memories. During consolidation, the LLM identifies relationships which are stored bidirectionally. Link traversal via `follow_links` parameter enables multi-hop reasoning.

### Hierarchical Cognitive Buffers

**What**: Multiple buffer levels with active promotion/demotion policies.

**Source**: [Cognitive Workspace](https://arxiv.org/abs/2508.13171)

**Results**: 58.6% memory reuse vs 0% for naive RAG.

**Engram**: Implements buffer promotion hierarchy (Working → Episodic → Semantic → Procedural). The `run_promotion` workflow promotes well-consolidated semantic memories with behavioral patterns to procedural memory based on selectivity score, consolidation passes, and confidence.

### Dynamic Engrams (Selectivity Through Consolidation)

**What**: Memory engrams are not static—neurons continuously "drop out of" and "drop into" engrams during consolidation. Engrams transition from unselective → selective over ~12 hours.

**Source**: [Tomé et al., Nature Neuroscience 2024](https://www.nature.com/articles/s41593-023-01551-w)

**Key findings**:
- Only 10-40% overlap between neurons activated during learning vs recall
- Inhibitory plasticity (CCK+ interneurons) is critical for selectivity
- Training stimulus reactivation during consolidation required for selectivity to emerge

**Engram**: The `selectivity_score` (0.0-1.0) field on SemanticMemory is directly inspired by this research. The field exists with `increase_selectivity()` and `decrease_selectivity()` methods, but consolidation does not yet call them. Future work will update this score during consolidation passes to model how engrams become more selective over time.

Note: `NegationFact` (which stores semantic negations like "User does NOT use MongoDB") is a separate engineering construct, not an implementation of neural inhibition.

### Retrieval-Induced Forgetting (RIF)

**What**: Retrieving a subset of items causes active suppression of related non-retrieved items. This is an inhibitory process, not just competition from strengthened items.

**Source**: [Anderson, Bjork & Bjork (1994)](https://pubmed.ncbi.nlm.nih.gov/7931095/) — "Remembering can cause forgetting: Retrieval dynamics in long-term memory." *Journal of Experimental Psychology: Learning, Memory, and Cognition*, 20(5), 1063-1087.

**Key findings**:
- Suppression is strongest for high-similarity items (not dissimilar ones)
- The effect is inhibitory (active suppression), not just competition from strengthening
- Suppression endures 20+ minutes in human experiments

**Engram**: Implements RIF as opt-in via `rif_enabled=True` on recall. After retrieval, memories that scored above `rif_threshold` but weren't returned get confidence decay (`rif_decay`, default 0.1). Episodic memories are exempt (immutable ground truth). Confidence floors at 0.1 to prevent total forgetting.

---

## What We Don't Do (Intentionally)

| Approach | Why We Avoid It |
|----------|-----------------|
| LLM extraction on write | Errors become permanent |
| Token compression focus | We optimize for accuracy, not size |
| Complex graph schemas | Over-engineering for most use cases |
| Self-editing memory | Risks ground truth corruption |

---

## References

- [HaluMem: Hallucinations in LLM Memory](https://arxiv.org/html/2511.03506) — Benchmark showing <56% accuracy
- [Mem0 Paper](https://arxiv.org/abs/2504.19413)
- [Graphiti Paper](https://arxiv.org/abs/2501.00227)
- [MemGPT Paper](https://arxiv.org/abs/2310.08560)
- [A-MEM Paper](https://arxiv.org/abs/2502.12110)
- [Cognitive Workspace](https://arxiv.org/abs/2508.13171)
- [Dynamic and Selective Engrams](https://www.nature.com/articles/s41593-023-01551-w) — Tomé et al., Nature Neuroscience 2024
- [Retrieval-Induced Forgetting](https://pubmed.ncbi.nlm.nih.gov/7931095/) — Anderson, Bjork & Bjork, 1994
