# Engram Examples

Comprehensive examples demonstrating all of Engram's features.

## Directory Structure

```
examples/
├── local/              # No external dependencies
│   ├── extraction.py   # Pattern extraction (email, phone, URL)
│   ├── confidence.py   # Confidence scoring system
│   └── memory_types.py # All 5 memory types explained
└── external/           # Requires Qdrant + API keys
    ├── quickstart.py   # Core encode/recall/verify workflow
    ├── structured.py   # StructuredMemory + LLM enrichment
    ├── advanced.py     # Multi-hop, negation filtering
    └── consolidation.py # LLM consolidation + linking
```

## Quick Start

### Local Examples (No Setup Required)

Run without any external dependencies:

```bash
# Pattern extraction (3 extractors: email, phone, URL)
uv run python examples/local/extraction.py

# Confidence scoring system
uv run python examples/local/confidence.py

# All 4 memory types explained
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

# StructuredMemory + LLM enrichment
uv run python examples/external/structured.py

# Advanced: multi-hop, negation filtering
uv run python examples/external/advanced.py

# LLM consolidation + semantic memories
uv run python examples/external/consolidation.py
```

---

## Local Examples

### Pattern Extraction
**`local/extraction.py`**

Demonstrates:
- 3 pattern extractors (email, phone, URL)
- Why pattern extraction before LLM matters (no hallucination possible)
- How extraction feeds into StructuredMemory

### Confidence Scoring
**`local/confidence.py`**

Explains:
- Three extraction methods: VERBATIM (100%), EXTRACTED (90%), INFERRED (60%)
- Weighted formula: method (50%) + corroboration (25%) + recency (15%) + verification (10%)
- Decay over time without reconfirmation
- Human-readable explanations for every score

### Memory Types
**`local/memory_types.py`**

Covers all 4 persistent memory types:
1. **Episodic** - Raw interactions (immutable ground truth)
2. **Structured** - Per-episode extraction (emails, phones, URLs, negations)
3. **Semantic** - LLM-inferred knowledge (variable confidence)
4. **Procedural** - Behavioral patterns (how to do things)

Plus Working memory (volatile, in-session only).

---

## External Examples

### Quickstart
**`external/quickstart.py`**

Core workflow demonstrating:
- `encode()` - Store episodes, extract structured data
- `recall()` - Semantic search with filtering
- `verify()` - Trace any memory to its source
- `min_confidence` - Confidence-gated retrieval
- `include_sources` - Get source episodes in results
- `memory_types` - Filter by memory type
- Working memory management

### Structured Memory
**`external/structured.py`**

StructuredMemory extraction modes:
- `enrich=False` (default) - Fast regex-only extraction (emails, phones, URLs)
- `enrich=True` - Sync LLM extraction (dates, people, preferences, negations)
- `enrich="background"` - Queue for background processing
- Negation detection (requires LLM enrichment) and filtering

### Advanced Features
**`external/advanced.py`**

Advanced recall features:
- **All 4 memory types** - Query across episodic, structured, semantic, procedural
- **Negation filtering** - Automatically exclude contradicted information
- **Multi-hop reasoning** - `follow_links=True` traverses related_ids
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
| Negation detection | - | `structured.py` (requires enrich=True) |
| Confidence scoring | `confidence.py` | `quickstart.py` |
| All 4 memory types | `memory_types.py` | `advanced.py` |
| encode/recall | - | `quickstart.py` |
| verify() | - | `quickstart.py` |
| min_confidence | - | `quickstart.py` |
| include_sources | - | `quickstart.py` |
| StructuredMemory modes | - | `structured.py` |
| Multi-hop (follow_links) | - | `advanced.py` |
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
| external/structured.py | ✅ | ✅ |
| external/advanced.py | ✅ | ✅ |
| external/consolidation.py | ✅ | ✅ |
