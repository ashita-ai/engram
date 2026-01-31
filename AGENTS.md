# Engram Agent Guide

**What is Engram**: Memory you can trust. A memory system for AI applications that preserves ground truth, tracks confidence, and prevents hallucinations.

**Status**: Beta. 800+ tests. Core APIs, REST endpoints, and workflows fully implemented.

**Your Role**: Python backend engineer building a memory layer for AI agents. You write production-grade code with comprehensive tests.

**Design Philosophy**: Ground truth preservation, auditable confidence, deferred consolidation.

---

## Boundaries

### Always Do (No Permission Needed)

- Write complete, production-grade code (no TODOs, no placeholders)
- Add tests for all new features (test both success and error cases)
- Use type hints (mypy strict mode)
- Follow async/await patterns for all database operations
- Update README.md when adding user-facing features
- Add docstrings to public functions
- **Use Pydantic for ALL data models** — no dataclasses, no TypedDict, no NamedTuple
- **Use Pydantic AI for ALL LLM interactions** — structured outputs, type-safe responses

### Ask First

- Modifying database models (affects migrations)
- Changing API contracts (breaking for consumers)
- Adding new dependencies to pyproject.toml
- Deleting existing endpoints or models
- Refactoring core services (storage, extraction, consolidation)

### Never Do

**GitHub Issues**:
- NEVER close an issue unless ALL acceptance criteria are met
- If an issue has checkboxes, ALL boxes must be checked before closing
- If you can't complete all criteria, leave the issue open and comment on what remains

**Git**:
- NEVER commit directly to main - always use a feature branch and PR
- NEVER push directly to main - all changes must go through pull requests
- NEVER force push to shared branches
- Do NOT include "Co-Authored-By: Claude" or the "Generated with Claude Code" footer

**Security**:
- NEVER commit credentials, API keys, tokens, or passwords
- Use environment variables (.env is in .gitignore)
- Pre-commit check: `grep -r "sk-\|sk-ant-\|AIza" src/ tests/ && echo "SECRETS FOUND" || echo "OK"`

**Code Quality**:
- Skip tests to make builds pass
- Disable type checking or linting
- Leave TODO comments in production code
- Delete failing tests instead of fixing them

---

## Commands

```bash
# Setup
uv sync --extra dev

# Run tests
uv run pytest tests/ -v --no-cov

# Code quality
uv run ruff check src/engram/
uv run ruff format src/engram/
uv run mypy src/engram/

# Pre-commit
uv run pre-commit install
uv run pre-commit run --all-files

# Start REST API server
uv run uvicorn engram.api.app:app --port 8000
```

---

## REST API

All endpoints are prefixed with `/api/v1`. Key endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/encode` | POST | Store episode + extract structured data |
| `/encode/batch` | POST | Bulk import (up to 100 items) |
| `/recall` | POST | Semantic search across memory types |
| `/memories/{id}` | GET | Get specific memory |
| `/memories/{id}` | DELETE | Delete memory (with cascade options) |
| `/memories/{id}/verify` | GET | Trace memory to source |
| `/workflows/consolidate` | POST | Episodes → Semantic memories |
| `/workflows/promote` | POST | Semantic → Procedural |

See `docs/api.md` for full documentation.

---

## Key Concepts

### Memory Types (5)

| Type | Purpose | Confidence |
|------|---------|------------|
| Working | Current session context (in-memory, not persisted) | N/A |
| Episode | Immutable ground truth (raw interactions) | N/A (verbatim) |
| StructuredMemory | Per-episode LLM extraction (entities, summary, negations) | 0.8-0.9 |
| SemanticMemory | Cross-episode LLM synthesis | 0.6 (inferred) |
| ProceduralMemory | Behavioral patterns | 0.6 (inferred) |

### Confidence Scoring

Confidence = weighted sum of:
- **Extraction method** (50%): verbatim=1.0, extracted=0.9, inferred=0.6
- **Corroboration** (25%): number of supporting sources
- **Recency** (15%): how recently confirmed
- **Verification** (10%): format validation passed

### Extraction Methods

| Method | Base Score | Description |
|--------|------------|-------------|
| VERBATIM | 1.0 | Exact quote, immutable |
| EXTRACTED | 0.9 | Deterministic pattern match (regex, validators) |
| INFERRED | 0.6 | LLM-derived, uncertain |

### Consolidation Flow

1. Episode stored (ground truth, never modified)
2. StructuredMemory created immediately (regex: emails, phones, URLs)
3. Optional LLM enrichment (dates, people, preferences, negations → StructuredMemory)
4. Background consolidation (N episodes → 1 SemanticMemory)
5. Procedural synthesis (all SemanticMemories → 1 behavioral profile per user)
6. Decay applied over time (confidence decreases without confirmation)

---

## Scientific Claims

Be careful about claims regarding cognitive science foundations. Engram is **inspired by** cognitive science, not a strict implementation of it.

### What we can claim:
- Multiple memory types are a useful engineering abstraction
- Ground truth preservation solves a real problem (LLM extraction errors)
- Confidence tracking distinguishes certain from uncertain
- Deferred processing reduces cost and latency
- Some form of forgetting is necessary for relevance and performance

### What we should NOT claim:
- That our architecture mirrors how the brain actually works
- That episodic/semantic are cleanly separable (they're not — it's a continuum)
- That consolidation works exactly as we model it (heavily debated)
- That Ebbinghaus decay is the definitive model of forgetting (it's a simplification)
- That Atkinson-Shiffrin is current (Baddeley's Working Memory Model superseded it)

---

## Development Workflow

```bash
# 1. Create branch (never work on main)
git checkout -b feature/my-feature

# 2. Make changes, run tests
uv run pytest tests/ -v --no-cov

# 3. Format and type check
uv run ruff check src/engram/ && uv run ruff format src/engram/ && uv run mypy src/engram/

# 4. Commit, push, create PR
git push -u origin feature/my-feature
```

---

## Pydantic (Required for All Data Models)

All data structures MUST use Pydantic. No exceptions.

### Model Pattern

```python
from pydantic import BaseModel, ConfigDict, Field

class MyModel(BaseModel):
    """Always add docstrings."""

    model_config = ConfigDict(extra="forbid")  # Catch typos

    id: str = Field(description="Unique identifier")
    value: float = Field(ge=0.0, le=1.0, description="Bounded value")
    items: list[str] = Field(default_factory=list)
```

### Rules

- `ConfigDict(extra="forbid")` on all models (catches field typos)
- Use `Field()` for validation constraints and descriptions
- Use `model_dump(mode="json")` for serialization
- Use `model_validate()` for deserialization
- Never use `@dataclass`, `TypedDict`, or `NamedTuple`

### Settings Pattern

```python
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ENGRAM_")

    api_key: str | None = Field(default=None)
    debug: bool = Field(default=False)
```

---

## Pydantic AI (Required for All LLM Interactions)

All LLM calls MUST use Pydantic AI with structured outputs. No raw API calls.

### Basic Pattern

```python
from pydantic import BaseModel
from pydantic_ai import Agent

class ExtractedFacts(BaseModel):
    """Structured output from LLM extraction."""
    facts: list[str]
    confidence: float
    reasoning: str

extraction_agent = Agent(
    "openai:gpt-4o-mini",
    result_type=ExtractedFacts,
    system_prompt="Extract factual statements from the conversation.",
)

async def extract_facts(text: str) -> ExtractedFacts:
    result = await extraction_agent.run(text)
    return result.data  # Type-safe ExtractedFacts
```

### Consolidation Agent Example

```python
from pydantic import BaseModel
from pydantic_ai import Agent

class ConsolidationResult(BaseModel):
    """Output from memory consolidation."""
    semantic_facts: list[str]
    links: list[tuple[str, str]]  # (memory_id, related_id)
    pruned_ids: list[str]
    confidence: float

consolidation_agent = Agent(
    "openai:gpt-4o-mini",
    result_type=ConsolidationResult,
    system_prompt="""
    Analyze episodes and extract semantic knowledge.
    Identify relationships between memories.
    Flag weak associations for pruning.
    """,
)
```

### Why Pydantic AI?

- **Type safety**: Responses are validated Pydantic models, not raw dicts
- **Structured outputs**: LLM returns exactly what you expect
- **Retries**: Automatic retry on validation failures
- **Observability**: Built-in logging and tracing

---

## Key Files

| File | Purpose |
|------|---------|
| `models/` | Pydantic models for all memory types |
| `storage/` | Qdrant client and collection management |
| `extraction/` | Pattern matchers and LLM extractors (Pydantic AI agents) |
| `consolidation/` | Background processing workflows (Pydantic AI agents) |
| `config.py` | Settings and confidence weights |
| `api/router.py` | REST API endpoints |
| `api/schemas.py` | Request/response Pydantic models |
| `api/auth.py` | Authentication and rate limiting |
| `context.py` | Memory context manager for SDK usage |

---

## Memory Types Detail

Memory types are engineering constructs:
- **Working, Episodic, Semantic, Procedural** — Inspired by cognitive science
- **StructuredMemory** — Per-episode LLM extraction bridging raw episodes and cross-episode semantic synthesis

Be explicit about which are science-inspired and which are engineering additions.

---

## Scientific Foundations

Engram is **inspired by** cognitive science research, not a strict implementation of it. Below are the papers we cite and how they inform our design.

### Research Reference Table

| Paper | Year | Key Finding | Engram Implementation |
|-------|------|-------------|----------------------|
| [Roediger & Karpicke](https://pubmed.ncbi.nlm.nih.gov/26151629/) | 2006 | Testing slows forgetting (13% vs 52% after 1 week) | `consolidation_strength`, `consolidation_passes` |
| [A-MEM](https://arxiv.org/abs/2502.12110) | 2025 | 2x multi-hop reasoning via Zettelkasten linking | `related_ids`, `follow_links` |
| [Cognitive Workspace](https://arxiv.org/abs/2508.13171) | 2025 | 58.6% memory reuse vs 0% for naive RAG | Hierarchical buffers, working memory |
| [HaluMem](https://arxiv.org/abs/2511.03506) | 2025 | <70% accuracy without ground truth preservation | Immutable episodes, `verify()` |
| [Karpicke & Roediger](https://www.sciencedirect.com/science/article/abs/pii/S1364661310002081) | 2008 | Retrieval = rapid consolidation | Retrieval-triggered strengthening |

### Testing Effect (Consolidation Strength)

**What it is**: Memories that are repeatedly involved in retrieval and consolidation become stronger and more stable.

**Primary Research**:
> Roediger, H.L. & Karpicke, J.D. (2006). "The Power of Testing Memory: Basic Research and Implications for Educational Practice." *Perspectives on Psychological Science*, 1(3), 181-210.

**Key experimental results**:
- After 1 week: Tested group forgot only **13%**, study-only group forgot **52%**
- Testing produces "rapid consolidation" of memory traces
- Testing reduces forgetting substantially more than repeated study

**How we implement it**: During consolidation, `strengthen()` is called when existing memories:
1. Get linked to new memories via semantic similarity
2. Receive LLM-identified links
3. Undergo evolution (tag/keyword/context updates)

Each call increases `consolidation_strength` by 0.1 and increments `consolidation_passes`.

### A-MEM (Dynamic Memory Linking)

**What it is**: Agentic memory architecture using Zettelkasten-style linking for multi-hop reasoning.

**Primary Research**:
> "A-MEM: Agentic Memory for LLM Agents" (2025). https://arxiv.org/abs/2502.12110

**Key results**:
- **2x improvement** on multi-hop reasoning benchmarks
- Dynamic linking outperforms static memory retrieval
- Links enable contextual relevance across domains

**How we implement it**: `related_ids` field on SemanticMemory and ProceduralMemory stores bidirectional links. During consolidation, `_find_matching_memory()` discovers links via exact, normalized, and substring matching. `follow_links=True` in `recall()` traverses links for multi-hop reasoning.

### HaluMem (Ground Truth Preservation)

**What it is**: Benchmark showing LLM memory systems hallucinate without source preservation.

**Primary Research**:
> "HaluMem: Evaluating Hallucinations in Memory Systems of Agents" (2025). https://arxiv.org/abs/2511.03506

**Key results**:
- "All systems achieve answer accuracies below **70%**"
- "Hallucination rate and omission rate remaining high"
- "Systems suffer omission rates above 50%"

**How we address it**: Immutable episodic storage preserves ground truth. All derived memories (`StructuredMemory`, `SemanticMemory`) maintain `source_episode_ids` for traceability. `verify()` enables auditing any memory back to its source.

### Cognitive Workspace (Hierarchical Buffers)

**What it is**: Active memory curation using hierarchical buffer management.

**Primary Research**:
> "Cognitive Workspace for AI Memory" (2025). https://arxiv.org/abs/2508.13171

**Key results**:
- **58.6% memory reuse** vs 0% for naive RAG
- Active curation outperforms passive retrieval
- Hierarchical organization improves recall relevance

**How we implement it**: Working memory → Episodic → StructuredMemory → SemanticMemory → ProceduralMemory hierarchy. Each tier serves different retention and retrieval purposes.

### Surprise-Based Importance (Adaptive Compression)

**What it is**: Novel information receives higher importance scores than redundant content.

**Primary Research**:
> Nagy et al. (2025). "Adaptive compression as a unifying framework for episodic and semantic memory." *Nature Reviews Psychology*. https://www.nature.com/articles/s44159-025-00458-6

**Key insight**: Information-theoretic surprise (low similarity to existing memories) indicates novel, valuable content worth retaining.

**How we implement it**: During episode encoding, `_calculate_surprise()` computes novelty by comparing embeddings to existing memories. Low similarity = high surprise = boosted importance score. Controlled by `surprise_scoring_enabled`, `surprise_weight` (default 0.15), and `surprise_search_limit` settings.

### What we DON'T implement

- **Exact biological mechanisms**: We use cognitive science as inspiration, not blueprint
- **True context selectivity**: Tomé et al. (2024) describes how engrams become more context-specific via inhibitory plasticity. We don't model this — we track consolidation involvement instead
- **Retrieval-induced forgetting**: Removed in v0.x due to context mismatch between human lab experiments and AI systems

---

## Communication

Be concise and direct. No flattery or excessive praise. Focus on what needs to be done.

---

## MCP Tool Usage

For Claude instances using Engram as an MCP server, see the skill file at `.claude/skills/engram-memory.md` for comprehensive usage guidance.
