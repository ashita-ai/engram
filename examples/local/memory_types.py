#!/usr/bin/env python3
"""Memory types overview demo.

Demonstrates Engram's four persistent memory types:
1. Episodic - Raw interactions (immutable ground truth)
2. Structured - Per-episode extraction (emails, phones, URLs, negations)
3. Semantic - LLM-inferred knowledge (variable confidence)
4. Procedural - Behavioral patterns (how to do things)

Working memory (volatile, in-session) is also available but not persisted.

No external dependencies required - runs entirely locally.
"""

from engram.models import (
    ConfidenceScore,
    Episode,
    Negation,
    ProceduralMemory,
    SemanticMemory,
    StructuredMemory,
)


def main() -> None:
    print("=" * 70)
    print("Engram Memory Types Demo")
    print("=" * 70)

    # =========================================================================
    # Overview
    # =========================================================================
    print("""
  Engram organizes memory into four persistent types:

  ┌─────────────────────────────────────────────────────────────────────┐
  │  WORKING MEMORY (volatile, in-session only)                        │
  │    └─> Episodes flow through working memory during conversation    │
  └─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │  EPISODIC MEMORY (immutable ground truth)                          │
  │    └─> Raw interactions stored verbatim, never modified            │
  └─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │  STRUCTURED MEMORY (per-episode extraction)                        │
  │    └─> Regex extracts (emails, phones, URLs)                       │
  │    └─> Negations (what is NOT true)                                │
  │    └─> LLM enrichment optional (people, preferences, dates)        │
  └─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │  SEMANTIC MEMORY (cross-episode LLM synthesis)                     │
  │    └─> Consolidates multiple episodes into summarized knowledge    │
  └─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │  PROCEDURAL MEMORY (behavioral patterns)                           │
  │    └─> How to interact with this user, promoted from semantic      │
  └─────────────────────────────────────────────────────────────────────┘
    """)

    # =========================================================================
    # 1. Working Memory
    # =========================================================================
    print("\n1. WORKING MEMORY")
    print("-" * 70)
    print("""
  Purpose: Current session context
  Persistence: In-memory only (volatile)
  Confidence: N/A

  Working memory holds episodes during an active conversation.
  It's cleared when the session ends.

  ```python
  # Get current session's context
  working = engram.get_working_memory()

  # Clear at session end
  engram.clear_working_memory()
  ```
    """)

    # =========================================================================
    # 2. Episodic Memory
    # =========================================================================
    print("\n2. EPISODIC MEMORY")
    print("-" * 70)

    episode = Episode(
        content="My email is alice@example.com and I work at Acme Corp.",
        role="user",
        user_id="demo_user",
        org_id="demo_org",
        session_id="demo_session",
        importance=0.7,
    )

    print("  Example Episode:")
    print(f"    ID: {episode.id}")
    print(f'    Content: "{episode.content}"')
    print(f"    Role: {episode.role}")
    print(f"    Timestamp: {episode.timestamp}")
    print(f"    Importance: {episode.importance}")
    print(f"    Summarized: {episode.summarized}")

    print("""
  Key Properties:
  - IMMUTABLE: Never modified after creation
  - Ground truth: All derived memories trace back here
  - Recovery: If extraction fails, re-derive from episodes
    """)

    # =========================================================================
    # 3. Structured Memory
    # =========================================================================
    print("\n3. STRUCTURED MEMORY")
    print("-" * 70)

    structured = StructuredMemory.from_episode_fast(
        source_episode_id=episode.id,
        user_id="demo_user",
        org_id="demo_org",
        emails=["alice@example.com"],
    )

    print("  Example StructuredMemory (fast mode):")
    print(f"    ID: {structured.id}")
    print(f"    Source Episode: {structured.source_episode_id}")
    print(f"    Mode: {structured.mode}")
    print(f"    Enriched: {structured.enriched}")
    print(f"    Emails: {structured.emails}")
    print(f"    Confidence: {structured.confidence.value:.0%}")

    # Rich mode example with negations
    structured_rich = StructuredMemory.from_episode(
        source_episode_id=episode.id,
        user_id="demo_user",
        org_id="demo_org",
        emails=["alice@example.com"],
        negations=[
            Negation(
                content="does not use MongoDB",
                pattern="MongoDB",
                context="switched to PostgreSQL",
            )
        ],
    )

    print("\n  Example StructuredMemory (rich mode with negation):")
    print(f"    ID: {structured_rich.id}")
    print(f"    Mode: {structured_rich.mode}")
    print(f"    Enriched: {structured_rich.enriched}")
    print(f"    Negations: {[n.content for n in structured_rich.negations]}")
    print(f"    Confidence: {structured_rich.confidence.value:.0%}")

    print("""
  Key Properties:
  - One per Episode (1:1 relationship)
  - Fast mode: regex-only extraction (emails, phones, URLs)
  - Rich mode: regex + LLM extraction (dates, people, preferences, negations)
  - Negations stored here to filter contradicted information
  - Categories: email, phone, url, date, person, preference, negation
    """)

    # =========================================================================
    # 4. Semantic Memory
    # =========================================================================
    print("\n4. SEMANTIC MEMORY")
    print("-" * 70)

    semantic = SemanticMemory(
        content="User prefers Python for backend development and uses FastAPI",
        source_episode_ids=[episode.id],
        user_id="demo_user",
        org_id="demo_org",
        confidence=ConfidenceScore.for_inferred(supporting_episodes=2),
        related_ids=["sem_abc123", "proc_xyz789"],
        consolidation_strength=0.3,
        consolidation_passes=2,
    )

    print("  Example SemanticMemory:")
    print(f"    ID: {semantic.id}")
    print(f'    Content: "{semantic.content}"')
    print(f"    Sources: {len(semantic.source_episode_ids)} episode(s)")
    print(f"    Confidence: {semantic.confidence.value:.0%}")
    print(f"    Related IDs: {semantic.related_ids}")
    print(f"    Consolidation Strength: {semantic.consolidation_strength}")
    print(f"    Consolidation Passes: {semantic.consolidation_passes}")

    print("""
  Key Properties:
  - Created by LLM consolidation (background)
  - Cross-episode synthesis (N episodes -> 1 semantic)
  - Variable confidence (60% base for inferred)
  - Links to related memories (multi-hop reasoning)
  - Strengthens with repeated consolidation (Testing Effect)
    """)

    # =========================================================================
    # 5. Procedural Memory
    # =========================================================================
    print("\n5. PROCEDURAL MEMORY")
    print("-" * 70)

    procedural = ProceduralMemory(
        content="When user asks about code style, refer to their preference for "
        "type hints, docstrings, and 88-char line limits",
        source_episode_ids=[episode.id],
        user_id="demo_user",
        org_id="demo_org",
        confidence=ConfidenceScore.for_inferred(supporting_episodes=3),
        trigger_context="code style, formatting, linting",
        access_count=5,
    )

    print("  Example ProceduralMemory:")
    print(f"    ID: {procedural.id}")
    print(f'    Content: "{procedural.content[:60]}..."')
    print(f'    Trigger Context: "{procedural.trigger_context}"')
    print(f"    Access Count: {procedural.access_count}")
    print(f"    Confidence: {procedural.confidence.value:.0%}")

    print("""
  Key Properties:
  - HOW to do things (behavioral patterns)
  - Synthesized from all semantic memories for a user
  - One per user (holistic profile)
  - Trigger context specifies when to activate
  - Access count tracks usage
    """)

    # =========================================================================
    # Summary Table
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
  | Type       | Persistence | Confidence | Created By       | Purpose              |
  |------------|-------------|------------|------------------|----------------------|
  | Working    | Volatile    | N/A        | encode()         | Session context      |
  | Episodic   | Permanent   | N/A        | encode()         | Ground truth         |
  | Structured | Permanent   | 90%        | encode()         | Per-episode extracts |
  | Semantic   | Permanent   | 60%        | consolidate()    | Knowledge synthesis  |
  | Procedural | Permanent   | 60%        | create_procedural() | Behavioral profile|

  Hierarchy: Episode -> Structured -> Semantic -> Procedural
    """)

    print("=" * 70)
    print("Demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
