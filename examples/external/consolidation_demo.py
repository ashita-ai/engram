#!/usr/bin/env python3
"""Consolidation workflow demo - LLM-based semantic memory extraction.

This example demonstrates the consolidation workflow:
1. Store raw episodes (conversation turns)
2. Run LLM consolidation to extract semantic knowledge
3. View the extracted semantic memories

This is the main feature that requires an external LLM API call.

Prerequisites:
    - Qdrant running locally (docker run -p 6333:6333 qdrant/qdrant)
    - OpenAI API key: ENGRAM_OPENAI_API_KEY=sk-...
    - FastEmbed for embeddings (no key needed)

Usage:
    ENGRAM_OPENAI_API_KEY=sk-... python examples/external/consolidation_demo.py
"""

import asyncio
import os
import sys

# Check for API key before importing anything
if not os.environ.get("ENGRAM_OPENAI_API_KEY"):
    print("Error: ENGRAM_OPENAI_API_KEY environment variable required")
    print("\nThis demo requires an OpenAI API key because it uses GPT-4o-mini")
    print("to extract semantic knowledge from conversation episodes.")
    print("\nUsage:")
    print("  ENGRAM_OPENAI_API_KEY=sk-... python examples/external/consolidation_demo.py")
    sys.exit(1)

from engram.service import EngramService
from engram.workflows.consolidation import run_consolidation


async def main() -> None:
    """Run the consolidation demo."""
    print("=" * 60)
    print("Engram Consolidation Demo (LLM-powered)")
    print("=" * 60)

    async with EngramService.create() as engram:
        user_id = "consolidation_demo_user"

        # =====================================================================
        # STEP 1: Store raw episodes (conversation turns)
        # =====================================================================
        print("\n1. Storing conversation episodes...")

        conversations = [
            ("user", "Hi! My name is Sarah and I'm a software engineer at Acme Corp."),
            ("assistant", "Nice to meet you, Sarah! What kind of software do you work on at Acme?"),
            ("user", "I mainly work on backend systems in Python. I love using FastAPI."),
            ("assistant", "FastAPI is great! Do you work with databases too?"),
            ("user", "Yes, we use PostgreSQL for our main database and Redis for caching."),
            ("user", "My work email is sarah.dev@acmecorp.com if you need to reach me."),
            ("assistant", "Got it! I'll remember your email sarah.dev@acmecorp.com."),
            ("user", "I prefer dark mode in all my applications, and I like vim keybindings."),
        ]

        for role, content in conversations:
            await engram.encode(content=content, role=role, user_id=user_id)
            print(f"  + [{role}] {content[:50]}...")

        print(f"\n  Stored {len(conversations)} episodes")

        # =====================================================================
        # STEP 2: Run LLM consolidation
        # =====================================================================
        print("\n2. Running LLM consolidation (GPT-4o-mini)...")
        print("   This extracts semantic knowledge from the raw episodes.")

        result = await run_consolidation(
            storage=engram._storage,
            embedder=engram._embedder,
            user_id=user_id,
        )

        print("\n   Consolidation complete:")
        print(f"   - Episodes processed: {result.episodes_processed}")
        print(f"   - Semantic memories created: {result.semantic_memories_created}")
        print(f"   - Links identified: {result.links_created}")
        if result.contradictions_found:
            print(f"   - Contradictions: {result.contradictions_found}")

        # =====================================================================
        # STEP 3: Query the semantic memories
        # =====================================================================
        print("\n3. Querying consolidated semantic memories...")

        queries = [
            "What is Sarah's job?",
            "What technologies does Sarah use?",
            "What are Sarah's preferences?",
        ]

        for query in queries:
            print(f"\n   Query: '{query}'")
            results = await engram.recall(
                query=query,
                user_id=user_id,
                limit=3,
            )

            for r in results:
                type_badge = f"[{r.memory_type}]"
                confidence = f" ({r.confidence:.0%})" if r.confidence else ""
                print(f"     {type_badge} {r.content[:55]}...{confidence}")

    print("\n" + "=" * 60)
    print("Consolidation demo complete!")
    print("=" * 60)
    print("\nKey takeaways:")
    print("- Episodes store raw conversation turns (ground truth)")
    print("- Consolidation uses LLM to extract semantic knowledge")
    print("- Semantic memories have confidence scores (default 0.6 for inferred)")
    print("- Both episode and semantic memories are searchable")


if __name__ == "__main__":
    asyncio.run(main())
