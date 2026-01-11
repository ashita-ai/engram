# Consolidation Redesign: Hierarchical Compression

## Vision

Transform consolidation from "fact extraction" to "hierarchical compression" aligned with cognitive science research on memory consolidation.

```
EPISODIC (raw, immutable, automatic)
    │
    ├──→ FACTS/NEGATIONS (pattern-extracted, high confidence)
    │
    └──→ SEMANTIC (LLM summary of N episodes → 1 memory)
              │
              └──→ PROCEDURAL (LLM synthesis of all semantics → behavioral patterns)
```

## Current vs Target

| Aspect | Current | Target |
|--------|---------|--------|
| Semantic creation | Extract facts (1 ep → N facts) | Summarize (N eps → 1 summary) |
| Compression | None (expansion) | Real (8:1 ratio) |
| Procedural | Auto-promote behavioral keywords | Explicit synthesis of all semantics |
| Episode tracking | `consolidated` flag | `summarized` flag |

## Research Basis

- **Complementary Learning Systems** (McClelland et al., 1995): Hippocampus → neocortex transfer with compression
- **Sleep consolidation**: Episodic details lost, gist retained
- **Cognitive Workspace** (arxiv.org/abs/2508.13171): Hierarchical buffers with selective consolidation

---

## Phase 1: Model Updates

### 1.1 Episode Model
**File:** `src/engram/models/episode.py`

```python
class Episode(MemoryBase):
    # Existing fields...

    # Rename or clarify
    consolidated: bool = False  # Keep for backwards compat
    summarized: bool = False    # New: included in a semantic summary
    summarized_into: str | None = None  # ID of semantic memory that contains this
```

**Decision needed:** Keep both `consolidated` and `summarized`, or rename `consolidated` to `summarized`?

Recommendation: Keep `consolidated` for fact/negation extraction, add `summarized` for semantic summaries. An episode can be consolidated (facts extracted) but not yet summarized.

### 1.2 Semantic Memory Model
**File:** `src/engram/models/semantic.py`

Current model already has:
- `source_episode_ids: list[str]` - episodes this was derived from
- `content: str` - the extracted content

Change: Content becomes a SUMMARY, not individual facts. The model structure works, but usage changes.

### 1.3 Procedural Memory Model
**File:** `src/engram/models/procedural.py`

Add:
```python
class ProceduralMemory(MemoryBase):
    # Existing fields...

    source_semantic_ids: list[str] = Field(
        default_factory=list,
        description="Semantic memories this was synthesized from"
    )
```

---

## Phase 2: Consolidation Workflow Rewrite

### 2.1 New Consolidation Logic
**File:** `src/engram/workflows/consolidation.py`

**Current flow:**
1. Get unconsolidated episodes
2. LLM extracts individual facts
3. Store each fact as separate semantic memory
4. Mark episodes consolidated

**New flow:**
1. Get unsummarized episodes (up to batch_size, e.g., 10)
2. LLM creates ONE summary of those episodes
3. Store summary as ONE semantic memory
4. Mark episodes as summarized, link to semantic ID
5. Repeat until all episodes summarized

### 2.2 New LLM Prompt

```python
SUMMARIZATION_PROMPT = """You are summarizing a batch of conversation episodes into a single coherent memory.

Create ONE summary that captures:
- Key facts about the user (name, preferences, context)
- Important decisions or statements made
- Any patterns in behavior or communication style

The summary should be:
- Concise (2-5 sentences)
- Written in third person ("The user...", "They mentioned...")
- Focused on lasting/stable information, not transient details

Do NOT list individual facts. Create a coherent narrative summary.

Episodes to summarize:
{episodes}

Existing context (previous summaries):
{existing_semantics}
"""
```

### 2.3 Updated Consolidation Function

```python
async def run_consolidation(
    storage: EngramStorage,
    embedder: Embedder,
    user_id: str,
    org_id: str | None = None,
    batch_size: int = 10,  # Episodes per summary
) -> ConsolidationResult:
    """Create semantic summaries from unsummarized episodes."""

    # 1. Get unsummarized episodes
    episodes = await storage.get_unsummarized_episodes(
        user_id=user_id,
        limit=batch_size,
    )

    if not episodes:
        return ConsolidationResult(episodes_processed=0, ...)

    # 2. Get existing semantics for context
    existing = await storage.list_semantic_memories(user_id)

    # 3. LLM creates ONE summary
    summary_content = await _create_summary(episodes, existing)

    # 4. Store semantic memory
    episode_ids = [ep.id for ep in episodes]
    semantic = SemanticMemory(
        content=summary_content,
        source_episode_ids=episode_ids,
        user_id=user_id,
        ...
    )
    await storage.store_semantic(semantic)

    # 5. Mark episodes as summarized
    for ep in episodes:
        ep.summarized = True
        ep.summarized_into = semantic.id
        await storage.update_episode(ep)

    return ConsolidationResult(
        episodes_processed=len(episodes),
        semantic_memories_created=1,  # Always 1 per batch
        ...
    )
```

### 2.4 Keep Fact/Negation Extraction Separate

The current extraction of facts (emails, phones) and negations ("I don't use X") should remain as a SEPARATE process from summarization:

- `run_extraction()` - pattern-based fact/negation extraction (runs on encode)
- `run_consolidation()` - LLM summarization (runs periodically or on-demand)

This is already how it works for facts/negations during encode.

---

## Phase 3: Procedural Synthesis

### 3.1 New Procedural Creation
**File:** `src/engram/workflows/promotion.py` → rename to `synthesis.py`?

**Current:** Auto-promotes semantics with behavioral keywords
**New:** Explicit synthesis of ALL semantics into behavioral patterns

### 3.2 New Synthesis Function

```python
async def create_procedural_memory(
    storage: EngramStorage,
    embedder: Embedder,
    user_id: str,
    org_id: str | None = None,
) -> ProceduralMemory:
    """Synthesize all semantic memories into a procedural memory.

    Called explicitly by the user, not automatically.
    Creates cross-session behavioral patterns.
    """

    # 1. Get ALL semantic memories
    semantics = await storage.list_semantic_memories(user_id)

    if not semantics:
        raise ValueError("No semantic memories to synthesize")

    # 2. LLM synthesizes behavioral patterns
    patterns = await _synthesize_patterns(semantics)

    # 3. Create procedural memory
    procedural = ProceduralMemory(
        content=patterns,
        source_semantic_ids=[s.id for s in semantics],
        user_id=user_id,
        ...
    )

    await storage.store_procedural(procedural)
    return procedural
```

### 3.3 Synthesis Prompt

```python
SYNTHESIS_PROMPT = """You are analyzing semantic memory summaries to identify lasting behavioral patterns.

From these summaries, identify:
- Communication preferences (tone, detail level, format)
- Technical preferences (languages, tools, frameworks)
- Work patterns (how they approach problems)
- Personal context (role, goals, constraints)

Create a behavioral profile that can guide future interactions.

Format as a coherent description, not a list.

Semantic summaries to analyze:
{semantics}
"""
```

---

## Phase 4: Service & API Updates

### 4.1 Service Methods
**File:** `src/engram/service.py`

```python
class EngramService:
    async def consolidate(self, user_id: str) -> ConsolidationResult:
        """Create semantic summary from unsummarized episodes."""
        return await run_consolidation(
            storage=self.storage,
            embedder=self.embedder,
            user_id=user_id,
        )

    async def create_procedural(self, user_id: str) -> ProceduralMemory:
        """Synthesize procedural memory from all semantics."""
        return await create_procedural_memory(
            storage=self.storage,
            embedder=self.embedder,
            user_id=user_id,
        )
```

### 4.2 API Endpoints
**File:** `src/engram/api/router.py`

```python
@router.post("/consolidate", response_model=ConsolidationResponse)
async def consolidate(user_id: str, service: ServiceDep):
    """Create semantic summary from unsummarized episodes."""
    result = await service.consolidate(user_id)
    return ConsolidationResponse(...)

@router.post("/procedural", response_model=ProceduralResponse)
async def create_procedural(user_id: str, service: ServiceDep):
    """Synthesize procedural memory from all semantics."""
    result = await service.create_procedural(user_id)
    return ProceduralResponse(...)
```

---

## Phase 5: Storage Updates

### 5.1 New Storage Methods
**File:** `src/engram/storage/base.py`

```python
async def get_unsummarized_episodes(
    self, user_id: str, limit: int = 10
) -> list[Episode]:
    """Get episodes that haven't been included in a semantic summary."""
    # Filter where summarized=False
    ...

async def update_episode(self, episode: Episode) -> None:
    """Update episode (for marking summarized)."""
    ...
```

---

## Phase 6: Testing & Migration

### 6.1 Update Tests
- `tests/test_consolidation.py` - test summarization behavior
- `tests/test_service.py` - test new methods
- New `tests/test_procedural_synthesis.py`

### 6.2 Update Demos
- `examples/external/consolidation.py` - show summarization
- `examples/external/quickstart.py` - show full flow
- New demo showing procedural synthesis

### 6.3 Migration
- Existing semantic memories: keep as-is (they're still valid derived memories)
- Existing episodes: set `summarized=False` for all (they'll be summarized on next consolidation)

---

## Summary of File Changes

| File | Change Type |
|------|-------------|
| `src/engram/models/episode.py` | Add `summarized`, `summarized_into` fields |
| `src/engram/models/procedural.py` | Add `source_semantic_ids` field |
| `src/engram/workflows/consolidation.py` | Major rewrite for summarization |
| `src/engram/workflows/promotion.py` | Rewrite as synthesis.py |
| `src/engram/service.py` | Add `consolidate()`, `create_procedural()` |
| `src/engram/api/router.py` | Add endpoints |
| `src/engram/storage/base.py` | Add `get_unsummarized_episodes()`, `update_episode()` |
| Tests | Update all affected tests |
| Demos | Update to show new behavior |

---

## Decisions (Confirmed)

1. **Batch size**: ALL messages since last summary. If too many for one LLM call, chunk and map-reduce into one memory.
2. **Timing**: Both auto AND manual supported. Auto triggers on session end or configurable threshold.
3. **Procedural**: ONE per user. Updates/replaces existing, doesn't accumulate.
4. **Old semantics**: Include all semantics when creating/updating procedural.
5. **Backwards compatibility**: Existing episodes get `summarized=False`, will be included in next consolidation.

---

## Estimated Effort

| Phase | Complexity | Risk |
|-------|------------|------|
| 1. Model Updates | Low | Low |
| 2. Consolidation Rewrite | High | Medium |
| 3. Procedural Synthesis | Medium | Low |
| 4. Service/API | Low | Low |
| 5. Storage | Medium | Low |
| 6. Testing | Medium | Low |

Total: ~2-3 days of focused work
