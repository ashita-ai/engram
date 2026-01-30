# Competitive Analysis

Technical comparison of AI memory systems. Based on published papers and documentation, not marketing claims.

## Summary

| System | Ground Truth | Confidence | Forgetting | Bi-Temporal | Dynamic Linking | Strength Tracking |
|--------|--------------|------------|------------|-------------|-----------------|-------------------|
| **Engram** | Yes | Yes | Yes | Partial | Yes | Yes |
| **Mem0** | No | No | Partial | No | Yes (Mem0ᵍ) | No |
| **Zep/Graphiti** | Yes | No | No | Yes | Partial | No |
| **LangChain** | No | No | No | No | Partial | No |
| **LlamaIndex** | No | No | No | No | Partial | No |
| **Letta/MemGPT** | Partial | No | No | No | No | No |

**Notes**:
- **Strength Tracking**: Whether memories get *stronger* through repeated consolidation (Testing Effect). No system except Engram tracks consolidation_strength or passes.
- **Engram Bi-Temporal**: Only SemanticMemory has full bi-temporal support (`event_at` + `derived_at`). StructuredMemory and ProceduralMemory track only `derived_at`.
- **Mem0 Forgetting**: Mem0 has lifecycle policies for time-based expiration, but not principled decay based on access/importance.
- **Mem0 Dynamic Linking**: Mem0ᵍ (graph variant) stores entities as nodes and relations as edges.

## Mem0

**Architecture**: Two-phase LLM extraction on every write, with optional graph memory (Mem0ᵍ).

```
Message → LLM Extract → LLM Categorize (ADD/UPDATE/DELETE/NOOP) → Store
                                                                    ↓
                                                    Vector Store OR Graph Store (Mem0ᵍ)
```

**Mem0ᵍ (Graph Variant)**:
- Entity Extractor identifies entities as nodes
- Relations Generator infers labeled edges
- Conflict Detector flags overlapping/contradictory elements
- Update Resolver decides: add, merge, invalidate, or skip

**What it does well**:
- Compresses conversation history (26K → 7K tokens claimed)
- Simple API
- Active development ($24M raised Oct 2025)
- Graph memory (Mem0ᵍ) enables relationship queries
- Lifecycle policies for time-based expiration
- 26% improvement over OpenAI baseline on LLM-as-Judge metric

**Limitations**:
- **No ground truth preservation** — Original messages discarded after extraction
- **LLM extraction on write** — Errors become permanent
- **No confidence tracking** — All memories treated as equally reliable
- **Forgetting is time-based only** — No access/importance-based decay
- **No consolidation strength** — Memories don't get stronger through repeated use

**Source**: [Mem0 Paper](https://arxiv.org/abs/2504.19413), [Documentation](https://docs.mem0.ai), [Graph Memory Docs](https://docs.mem0.ai/open-source/features/graph-memory)

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

**Source**: [Zep Paper](https://arxiv.org/abs/2501.13956), [Documentation](https://help.getzep.com)

---

## LangChain Memory

**Architecture**: Multiple memory module types as framework components.

```
Memory Types: ConversationBufferMemory, ConversationSummaryBufferMemory,
             ConversationEntityMemory, VectorStoreRetrieverMemory
```

**What it does well**:
- Multiple memory types for different use cases
- Integrates with full LangChain ecosystem
- Enterprise components (Redis, PostgreSQL, MongoDB via LangGraph)
- Context-aware systems across sessions

**Limitations**:
- **No ground truth preservation** — Memory is for context, not accuracy
- **No confidence tracking** — No distinction between certain and uncertain
- **No principled forgetting** — Manual cleanup required
- **No consolidation** — Memory types don't form a hierarchy

**Source**: [Documentation](https://docs.langchain.com/)

---

## LlamaIndex

**Architecture**: RAG-native document indexing with memory modules.

```
Documents → Indexing → Query Engine / Chat Engine → Response + Context
```

**What it does well**:
- RAG-optimized retrieval pipelines
- Hybrid search (vector + keyword)
- Multi-agent memory support
- Extensive integrations with vector databases

**Limitations**:
- **No ground truth preservation** — Focus on retrieval, not memory accuracy
- **No confidence tracking** — Standard RAG approach
- **Stateless agents** — External integration needed for persistence
- **No consolidation hierarchy** — Flat memory structure

**Source**: [Documentation](https://developers.llamaindex.ai/)

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

## Engram

**Architecture**: Ground truth preservation with deferred LLM processing.

```
Message → Embed → Store Episode (immutable)
              ↓
        Pattern Extract → StructuredMemory (emails, phones, URLs - no LLM)
              │
              └──────────────────────────────────────────────┐
                                                             ↓
                                            [BACKGROUND - Durable Workflow]
                                                             ↓
                                            LLM Consolidation (batched)
                                                    ↓
                                    ┌───────────────┼───────────────┐
                                    ↓               ↓               ↓
                            SemanticMemory    Links/Relations   Negations
                                    ↓               ↓               ↓
                            Strengthen existing memories (Testing Effect)
```

**Write path characteristics**:
- **Immediate**: Episode stored verbatim + pattern extraction + embedding
- **Deferred**: LLM consolidation runs in background (durable workflow)
- **Everything embedded**: All memory types semantically searchable

**What it does well**:
- Ground truth preserved (episodes immutable)
- Pattern extraction before LLM (no hallucination on structured data)
- Composite confidence scores (auditable)
- Principled forgetting (access + importance + time)
- Consolidation strength (Testing Effect)
- Semantic search across all memory types

**Source**: [Architecture](../architecture.md), [GitHub](https://github.com/ashita-ai/engram)

---

## Engram's Differentiators

### 1. Ground Truth Preservation

Like Zep, Engram preserves raw interactions. Unlike Zep:
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

**Key differentiator:** Scores are fully auditable. The system can explain *why* confidence is 0.73, not just that it is.

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

**What**: Track both event time (when fact was true) and ingestion time (when it was learned).

**Who has it**: Zep/Graphiti

**Value**: Enables "what was known on date X?" queries. Useful for debugging and audit.

**Engram**: Partial bi-temporal support. SemanticMemory has both `event_at` (when facts were true) and `derived_at` (when inferred). StructuredMemory and ProceduralMemory track only `derived_at`. Full bi-temporal queries are supported via `recall_at` endpoint for "what was known at time X" scenarios.

### Dynamic Memory Linking (A-MEM)

**What**: Memories form explicit links to related memories, enabling multi-hop reasoning.

**Source**: [A-MEM Paper](https://arxiv.org/abs/2502.12110) (NeurIPS 2025)

**Results**: 2x improvement on multi-hop reasoning benchmarks.

**Engram**: Implements dynamic linking via `related_ids` field on semantic and procedural memories. During consolidation, the LLM identifies relationships which are stored bidirectionally. Link traversal via `follow_links` parameter enables multi-hop reasoning.

### Hierarchical Cognitive Buffers

**What**: Multiple buffer levels with active promotion/demotion policies.

**Source**: [Cognitive Workspace](https://arxiv.org/abs/2508.13171)

**Results**: 58.6% memory reuse vs 0% for naive RAG.

**Engram**: Implements buffer promotion hierarchy (Working → Episodic → Semantic → Procedural). The `run_promotion` workflow promotes well-consolidated semantic memories with behavioral patterns to procedural memory based on consolidation strength, consolidation passes, and confidence.

### Consolidation Strength (Testing Effect)

**What**: Memories that are repeatedly involved in retrieval/consolidation become stronger and more stable.

**Source**: [Roediger & Karpicke 2006](https://pubmed.ncbi.nlm.nih.gov/26151629/), [Karpicke & Roediger 2008](https://www.sciencedirect.com/science/article/abs/pii/S1364661310002081)

**Key findings**:
- "Repeated remembering strengthens memories much more so than repeated learning"
- Retrieval acts as a rapid consolidation event
- Memories involved in retrieval become stronger and more stable

**Engram**: The `consolidation_strength` (0.0-1.0) field on SemanticMemory tracks how well-established a memory is. During consolidation, `strengthen()` is called when existing memories get linked to new memories (via semantic similarity or LLM identification) or receive evolution updates. Each call increases consolidation_strength by 0.1 and increments `consolidation_passes`.

Note: Negations (stored in `StructuredMemory.negations`, e.g., "User does NOT use MongoDB") are a separate engineering construct.

---

## What Engram Doesn't Do (Intentionally)

| Approach | Why Engram Avoids It |
|----------|-----------------|
| LLM extraction on write | Errors become permanent |
| Token compression focus | Engram optimizes for accuracy, not size |
| Complex graph schemas | Over-engineering for most use cases |
| Self-editing memory | Risks ground truth corruption |

---

## Known Gaps

Features documented elsewhere but not yet implemented:

| Gap | Impact | Status |
|-----|--------|--------|
| **Cascade deletion** | Deleting episodes orphans derived memories with stale `source_episode_ids` | [#137](https://github.com/ashita-ai/engram/issues/137) |
| **Full bi-temporal on all types** | Only SemanticMemory has `event_at`; others have only `derived_at` | Future work |
| **Bayesian confidence** | Current confidence is static weighted sum, not learned | [#136](https://github.com/ashita-ai/engram/issues/136) |

---

## References

- [HaluMem: Evaluating Hallucinations in Memory Systems](https://arxiv.org/abs/2511.03506) — Benchmark showing <70% accuracy
- [Mem0 Paper](https://arxiv.org/abs/2504.19413)
- [Zep Paper](https://arxiv.org/abs/2501.13956)
- [MemGPT Paper](https://arxiv.org/abs/2310.08560)
- [A-MEM Paper](https://arxiv.org/abs/2502.12110)
- [Cognitive Workspace](https://arxiv.org/abs/2508.13171)
- [Adaptive Compression Framework](https://www.nature.com/articles/s44159-025-00458-6) — Nagy et al. 2025 (episodic/semantic distinction)
- [Sleep and Memory](https://www.nature.com/articles/s42003-025-07868-5) — Yuksel et al. 2025 (SWS×REM interaction)
- [Testing Effect](https://pubmed.ncbi.nlm.nih.gov/26151629/) — Roediger & Karpicke, 2006
- [Retrieval-Induced Forgetting](https://pubmed.ncbi.nlm.nih.gov/7931095/) — Anderson, Bjork & Bjork, 1994
