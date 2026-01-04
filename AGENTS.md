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

### Memory Types

| Type | Purpose | Confidence |
|------|---------|------------|
| Episode | Immutable ground truth (raw interactions) | N/A (verbatim) |
| Fact | Pattern-extracted facts (emails, phones, dates) | 0.9 (extracted) |
| SemanticMemory | LLM-inferred knowledge | 0.6 (inferred) |
| ProceduralMemory | Behavioral patterns | 0.6 (inferred) |
| InhibitoryFact | What is NOT true (negations) | 0.7 (inferred) |
| Working | Current session context (in-memory) | N/A |

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

## Key Files

| File | Purpose |
|------|---------|
| `models/` | Pydantic models for all memory types |
| `storage/` | Qdrant client and collection management |
| `extraction/` | Pattern matchers and LLM extractors |
| `consolidation/` | Background processing workflows |
| `config.py` | Settings and confidence weights |

---

## Memory Types Detail

The six memory types are engineering constructs:
- **Working, Episodic, Semantic, Procedural** — Inspired by cognitive science
- **Factual** — Engineering subdivision (verbatim vs inferred) not from cognitive science
- **Inhibitory** — Inspired by CCK+ interneurons in memory selectivity (Tomé et al.)

Be explicit about which are science-inspired and which are engineering additions.

---

## Communication

Be concise and direct. No flattery or excessive praise. Focus on what needs to be done.
