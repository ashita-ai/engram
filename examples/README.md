# Engram Examples

Comprehensive examples demonstrating all of Engram's features.

## Directory Structure

```
examples/
├── local/              # No external dependencies
│   ├── extraction.py   # Pattern extraction + negation detection
│   ├── confidence.py   # Confidence scoring system
│   └── memory_types.py # All 6 memory types explained
└── external/           # Requires Qdrant + API keys
    ├── quickstart.py   # Core encode/recall/verify workflow
    ├── advanced.py     # RIF, multi-hop, negation filtering
    └── consolidation.py # LLM consolidation + linking
```

## Quick Start

### Local Examples (No Setup Required)

Run without any external dependencies:

```bash
# Pattern extraction (8 extractors + negation detection)
uv run python examples/local/extraction.py

# Confidence scoring system
uv run python examples/local/confidence.py

# All 6 memory types explained
uv run python examples/local/memory_types.py
```

### External Examples (Requires Qdrant + API Key)

1. Start Qdrant:
```bash
docker run -p 6333:6333 qdrant/qdrant
```

2. Set up `.env` (copy from `.env.example`):
```bash
ENGRAM_OPENAI_API_KEY=sk-...
ENGRAM_EMBEDDING_PROVIDER=openai
ENGRAM_EMBEDDING_MODEL=text-embedding-3-small
ENGRAM_QDRANT_URL=http://localhost:6333
```

3. Run examples:
```bash
# Core workflow: encode, recall, verify
uv run python examples/external/quickstart.py

# Advanced: RIF, multi-hop, negation filtering
uv run python examples/external/advanced.py

# LLM consolidation + semantic memories
uv run python examples/external/consolidation.py
```

---

## Local Examples

### Pattern Extraction
**`local/extraction.py`**

Demonstrates:
- All 8 pattern extractors (email, phone, URL, date, quantity, language, name, ID)
- Negation detection ("I don't use X", "not interested in Y")
- Why pattern extraction before LLM matters (no hallucination possible)

### Confidence Scoring
**`local/confidence.py`**

Explains:
- Three extraction methods: VERBATIM (100%), EXTRACTED (90%), INFERRED (60%)
- Weighted formula: method (50%) + corroboration (25%) + recency (15%) + verification (10%)
- Decay over time without reconfirmation
- Human-readable explanations for every score

### Memory Types
**`local/memory_types.py`**

Covers all 6 memory types:
1. **Working** - Current session context (volatile)
2. **Episodic** - Raw interactions (immutable ground truth)
3. **Factual** - Pattern-extracted facts (high confidence)
4. **Semantic** - LLM-inferred knowledge (variable confidence)
5. **Procedural** - Behavioral patterns (how to do things)
6. **Negation** - What is NOT true (prevents contradictions)

---

## External Examples

### Quickstart
**`external/quickstart.py`**

Core workflow demonstrating:
- `encode()` - Store episodes, extract facts
- `recall()` - Semantic search with filtering
- `verify()` - Trace any memory to its source
- `min_confidence` - Confidence-gated retrieval
- `include_sources` - Get source episodes in results
- `memory_types` - Filter by memory type
- Working memory management

### Advanced Features
**`external/advanced.py`**

Advanced recall features:
- **All 6 memory types** - Query across episodic, factual, semantic, procedural, negation, working
- **Negation filtering** - Automatically exclude contradicted information
- **Multi-hop reasoning** - `follow_links=True` traverses related_ids
- **RIF (Retrieval-Induced Forgetting)** - Suppress competing memories
- **Freshness filtering** - Only return consolidated memories
- **Selectivity filtering** - Filter semantic memories by context-specificity

### Consolidation
**`external/consolidation.py`**

LLM-powered semantic extraction:
- Store raw episodes (ground truth)
- Run LLM consolidation (GPT-4o-mini)
- Create semantic memories with confidence
- Link related memories (multi-hop)
- Consolidation strength (Testing Effect)
- Incremental consolidation
- Verify derived memories back to sources

---

## Feature Coverage

| Feature | Local Example | External Example |
|---------|---------------|------------------|
| Pattern extraction | `extraction.py` | `quickstart.py` |
| Negation detection | `extraction.py` | `advanced.py` |
| Confidence scoring | `confidence.py` | `quickstart.py` |
| All 6 memory types | `memory_types.py` | `advanced.py` |
| encode/recall | - | `quickstart.py` |
| verify() | - | `quickstart.py` |
| min_confidence | - | `quickstart.py` |
| include_sources | - | `quickstart.py` |
| Multi-hop (follow_links) | - | `advanced.py` |
| RIF suppression | - | `advanced.py` |
| Negation filtering | - | `advanced.py` |
| Freshness filtering | - | `advanced.py` |
| Selectivity filtering | - | `advanced.py` |
| LLM consolidation | - | `consolidation.py` |
| Memory linking | - | `consolidation.py` |
| Consolidation strength | - | `consolidation.py` |

---

## Requirements Summary

| Example | Qdrant | OpenAI API |
|---------|--------|------------|
| local/extraction.py | - | - |
| local/confidence.py | - | - |
| local/memory_types.py | - | - |
| external/quickstart.py | ✅ | ✅ |
| external/advanced.py | ✅ | ✅ |
| external/consolidation.py | ✅ | ✅ |
