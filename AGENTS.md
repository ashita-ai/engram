# Engram Agent Guide

**What is Engram**: Memory you can trust. A memory system for AI applications that preserves ground truth, tracks confidence, and prevents hallucinations.

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
```

---

## Key Concepts

### Memory Types (6 Total)

| Type | Purpose | Confidence |
|------|---------|------------|
| Working | Current session context (in-memory, not persisted) | N/A |
| Episode | Immutable ground truth (raw interactions) | N/A (verbatim) |
| Fact | Pattern-extracted facts (emails, phones, dates) | 0.9 (extracted) |
| SemanticMemory | LLM-inferred knowledge | 0.6 (inferred) |
| ProceduralMemory | Behavioral patterns | 0.6 (inferred) |
| NegationFact | What is NOT true (negations) | 0.7 (inferred) |

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
2. Pattern extraction runs (emails, phones, dates → Facts)
3. Background consolidation (LLM → SemanticMemory, ProceduralMemory)
4. Decay applied over time (confidence decreases without confirmation)

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

---

## Memory Types Detail

The six memory types are engineering constructs:
- **Working, Episodic, Semantic, Procedural** — Inspired by cognitive science
- **Factual** — Engineering subdivision (verbatim vs inferred) not from cognitive science
- **Negation** — Engineering construct for storing semantic negations (what is NOT true)

Be explicit about which are science-inspired and which are engineering additions.

---

## Scientific Foundations

### Retrieval-Induced Forgetting (RIF)

**What it is**: Retrieving a subset of items causes forgetting of related non-retrieved items. This is an active inhibitory process, not just competition from strengthened items.

**Primary Research**:
> Anderson, M.C., Bjork, R.A., & Bjork, E.L. (1994). "Remembering can cause forgetting: Retrieval dynamics in long-term memory." *Journal of Experimental Psychology: Learning, Memory, and Cognition*, 20(5), 1063-1087. https://pubmed.ncbi.nlm.nih.gov/7931095/

**Key findings from the paper**:
1. Retrieving some items from a category suppresses related non-retrieved items
2. Suppression is strongest for high-similarity items (not dissimilar ones)
3. The effect is inhibitory (active suppression), not just competition from strengthening
4. Suppression endures 20+ minutes in human experiments

**How we implement it** (opt-in via `rif_enabled=True`):
- After recall, we identify memories that scored above `rif_threshold` but weren't returned
- These "competitors" get confidence decay (`rif_decay`, default 0.1)
- Episodic memories are exempt (immutable ground truth)
- Confidence floors at 0.1 to prevent total forgetting

```python
# RIF in action
results = await recall(
    query="user preferences",
    user_id="u1",
    limit=5,
    rif_enabled=True,     # Enable suppression
    rif_threshold=0.5,    # Min similarity to be a "competitor"
    rif_decay=0.1,        # Confidence decay amount
)
# Behind the scenes: similar memories that scored 0.5+ but weren't
# in top 5 get confidence reduced by 0.1
```

**What we DON'T claim**: Our confidence decay is an engineering approximation of inhibitory suppression. The original research studied human memory with specific experimental paradigms (category-cued recall). We adapt the core principle for AI memory systems.

### Context Selectivity (Implemented)

**What it is**: Memories should respond specifically to their original context, not overgeneralize.

**Research**: Tomé et al. "Dynamic and selective engrams emerge with memory consolidation" Nature Neuroscience (2024). Describes how engrams become more context-specific over time.

**Status**: Implemented. During consolidation, `increase_selectivity()` is called when existing memories:
1. Get linked to new memories via semantic similarity
2. Receive LLM-identified links
3. Undergo evolution (tag/keyword/context updates)

Each call increases `selectivity_score` by 0.1 and increments `consolidation_passes`. This is distinct from RIF - it tracks memory precision, not inter-memory competition.

### What we DON'T implement

- **Tomé et al. inhibitory plasticity**: The paper describes neurons competing WITHIN a single engram. We don't model individual neurons.
- **Exact biological mechanisms**: We use cognitive science as inspiration, not blueprint.

---

## Communication

Be concise and direct. No flattery or excessive praise. Focus on what needs to be done.
