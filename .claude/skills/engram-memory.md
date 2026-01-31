---
name: engram-memory
description: Persistent memory system for AI agents. Use when you need to remember user information across sessions, recall past conversations, or track preferences and facts over time.
---

# Engram Memory

Store and recall information across sessions. Ground truth preservation prevents hallucination.

## Quick Reference

| Task | Tool |
|------|------|
| Store user info | `engram_encode` |
| "What do I know about X?" | `engram_recall` |
| List/filter memories | `engram_search` |
| Check a fact's source | `engram_verify` |

## When to Store (engram_encode)

**Always encode:**
- User preferences: "I prefer dark mode"
- Contact info: emails, phones, addresses
- Decisions: "Let's use PostgreSQL"
- Facts about people/projects
- Negations: "I stopped using Twitter"

**Use `enrich=True` for:**
- Preferences and negations (needs LLM to understand meaning)
- Important content worth the extra processing

**Skip encoding:**
- "OK", "Got it", acknowledgments
- Questions without factual content
- Already encoded this session

## When to Recall

**Check memory when:**
- User asks about their preferences
- You need context from past sessions
- Answering questions about people/projects discussed before

**Don't check memory for:**
- General knowledge questions
- Current conversation already has the answer
- Hypothetical or new topics

## Tool Selection

```
Have a semantic question? ("What does user prefer?")
└── engram_recall
    ├── Need connected context? → follow_links=True
    └── Results redundant? → diversity=0.3

Just filtering? ("Show memories from last week")
└── engram_search
    └── Filter by: session_id, created_after, memory_types

Point-in-time? ("What did I know last Tuesday?")
└── engram_recall_at

Specific memory ID?
└── engram_get
```

## Memory Types

| Type | Use For |
|------|---------|
| `episodic` | Exact wording, quotes |
| `structured` | Extracted entities (emails, phones) |
| `semantic` | Synthesized facts, preferences |
| `procedural` | Behavioral patterns |

Default searches all. Use `memory_types="semantic,procedural"` for synthesized knowledge only.

## Confidence

| Score | Meaning |
|-------|---------|
| 0.9+ | Verbatim/extracted - trust fully |
| 0.7-0.9 | LLM extracted - trust with context |
| 0.5-0.7 | Inferred - verify if critical |
| <0.5 | Low confidence - don't rely on |

Use `min_confidence=0.7` when you need reliable facts.

---

## Advanced: Linking Memories

Connect related memories for multi-hop reasoning:

```
1. Find memories: engram_recall("Alice") → sem_123
2. Find related: engram_recall("Alice's preferences") → sem_456
3. Link them: engram_link(sem_123, sem_456, link_type="related")
4. Query with links: engram_recall("Alice", follow_links=True)
   → Returns Alice + her preferences + connected context
```

**Link types:**
- `related`: General association (bidirectional)
- `supersedes`: New info replaces old
- `contradicts`: Conflicting information

## Advanced: Consolidation

Run periodically to synthesize patterns:

```
engram_consolidate(user_id)  # Episodes → Semantic
engram_promote(user_id)      # Semantic → Procedural
```

**When:** After long conversations (10+ episodes), at session end, when user asks about patterns.

## Advanced: Conflict Detection

When you notice conflicting information:

```
engram_detect_conflicts(user_id)  # Find conflicts
engram_link(new_id, old_id, link_type="supersedes")  # Mark resolution
```

## Error Handling

Tools return JSON with `error` field on failure:
```json
{"error": "Memory not found: sem_xyz"}
```

Check for errors before processing results.
