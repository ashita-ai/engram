#!/usr/bin/env python3
"""Memory types overview demo.

Demonstrates Engram's six memory types:
1. Working - Current session context (volatile)
2. Episodic - Raw interactions (immutable ground truth)
3. Factual - Pattern-extracted facts (high confidence)
4. Semantic - LLM-inferred knowledge (variable confidence)
5. Procedural - Behavioral patterns (how to do things)
6. Negation - What is NOT true (prevents contradictions)

No external dependencies required - runs entirely locally.
"""

from engram.models import (
    ConfidenceScore,
    Episode,
    Fact,
    NegationFact,
    ProceduralMemory,
    SemanticMemory,
)


def main() -> None:
    print("=" * 70)
    print("Engram Memory Types Demo")
    print("=" * 70)

    # =========================================================================
    # Overview
    # =========================================================================
    print("""
  Engram organizes memory into six types, inspired by cognitive science:

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
              ┌─────────────────────┼─────────────────────┐
              ▼                     ▼                     ▼
  ┌───────────────────┐ ┌───────────────────┐ ┌───────────────────┐
  │  FACTUAL          │ │  SEMANTIC         │ │  NEGATION         │
  │  (extracted)      │ │  (inferred)       │ │  (what's NOT true)│
  │  emails, phones   │ │  preferences      │ │  contradictions   │
  │  dates, names     │ │  knowledge        │ │  corrections      │
  └───────────────────┘ └───────────────────┘ └───────────────────┘
                                    │
                                    ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │  PROCEDURAL MEMORY (behavioral patterns)                           │
  │    └─> How to do things, promoted from semantic memories           │
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
        importance=0.7,
    )

    print("  Example Episode:")
    print(f"    ID: {episode.id}")
    print(f'    Content: "{episode.content}"')
    print(f"    Role: {episode.role}")
    print(f"    Timestamp: {episode.timestamp}")
    print(f"    Importance: {episode.importance}")
    print(f"    Consolidated: {episode.consolidated}")

    print("""
  Key Properties:
  - IMMUTABLE: Never modified after creation
  - Ground truth: All derived memories trace back here
  - Recovery: If extraction fails, re-derive from episodes
    """)

    # =========================================================================
    # 3. Factual Memory
    # =========================================================================
    print("\n3. FACTUAL MEMORY")
    print("-" * 70)

    fact = Fact(
        content="alice@example.com",
        category="email",
        source_episode_id=episode.id,
        user_id="demo_user",
        confidence=ConfidenceScore.for_extracted(),
    )

    print("  Example Fact:")
    print(f"    ID: {fact.id}")
    print(f'    Content: "{fact.content}"')
    print(f"    Category: {fact.category}")
    print(f"    Source: {fact.source_episode_id}")
    print(f"    Confidence: {fact.confidence.value:.0%}")
    print(f"    Derived At: {fact.derived_at}")

    print("""
  Key Properties:
  - Extracted via pattern matching (deterministic)
  - High confidence (90% base)
  - No hallucination possible
  - Categories: email, phone, url, date, quantity, language, name, id
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
  - Promoted from well-consolidated semantic memories
  - Trigger context specifies when to activate
  - Access count tracks usage
    """)

    # =========================================================================
    # 6. Negation Memory
    # =========================================================================
    print("\n6. NEGATION MEMORY")
    print("-" * 70)

    negation = NegationFact(
        content="User does NOT use MongoDB - they explicitly stated they switched to PostgreSQL",
        negates_pattern="MongoDB",
        source_episode_ids=[episode.id],
        user_id="demo_user",
        confidence=ConfidenceScore.for_inferred(confidence=0.7),  # Negations use 0.7
    )

    print("  Example NegationFact:")
    print(f"    ID: {negation.id}")
    print(f'    Content: "{negation.content}"')
    print(f'    Negates Pattern: "{negation.negates_pattern}"')
    print(f"    Confidence: {negation.confidence.value:.0%}")

    print("""
  Key Properties:
  - Stores what is NOT true
  - Prevents returning contradicted information
  - Detected from phrases like "I don't", "not interested", "no longer"
  - Filters out matching memories during recall
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
  | Factual    | Permanent   | 90%        | Pattern matching | Structured data      |
  | Semantic   | Permanent   | 60%        | LLM consolidation| Knowledge/preferences|
  | Procedural | Permanent   | 60%        | Promotion        | Behavioral patterns  |
  | Negation   | Permanent   | 70%        | Negation detector| What's NOT true      |
    """)

    print("=" * 70)
    print("Demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
