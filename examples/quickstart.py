#!/usr/bin/env python3
"""Quickstart example for Engram.

This example demonstrates the core encode/recall workflow:
1. Initialize the Engram service
2. Encode some memories (conversations)
3. Recall relevant memories using semantic search

Prerequisites:
    - Qdrant running locally (docker run -p 6333:6333 qdrant/qdrant)
    - OpenAI API key set (OPENAI_API_KEY env var) OR use FastEmbed

Usage:
    # With OpenAI embeddings (default)
    OPENAI_API_KEY=sk-... python examples/quickstart.py

    # With local FastEmbed (no API key needed)
    ENGRAM_EMBEDDING_PROVIDER=fastembed python examples/quickstart.py
"""

import asyncio

from engram.service import EngramService


async def main() -> None:
    """Run the quickstart demo."""
    print("=" * 60)
    print("Engram Quickstart Demo")
    print("=" * 60)

    # Create and initialize the service
    async with EngramService.create() as engram:
        user_id = "demo_user"

        # =====================================================================
        # ENCODE: Store some memories
        # =====================================================================
        print("\n📝 Encoding memories...")

        conversations = [
            ("user", "My email is alice@example.com and I work at TechCorp."),
            ("assistant", "Got it! I'll remember your email alice@example.com."),
            ("user", "I have a meeting tomorrow at 3:30 PM about the Q4 budget."),
            ("user", "Call me at 555-123-4567 if there are any issues."),
            ("user", "The project costs $15,000 and we need it done by January 15th."),
        ]

        for role, content in conversations:
            result = await engram.encode(
                content=content,
                role=role,
                user_id=user_id,
            )
            print(f"  ✓ Stored episode: {content[:50]}...")
            if result.facts:
                for fact in result.facts:
                    print(f"    → Extracted {fact.category}: {fact.content}")

        # =====================================================================
        # RECALL: Search for relevant memories
        # =====================================================================
        print("\n🔍 Recalling memories...")

        queries = [
            "What is Alice's contact information?",
            "When is the meeting?",
            "How much does the project cost?",
        ]

        for query in queries:
            print(f"\n  Query: '{query}'")
            results = await engram.recall(
                query=query,
                user_id=user_id,
                limit=3,
            )

            if results:
                for r in results:
                    confidence_str = f" (confidence: {r.confidence:.0%})" if r.confidence else ""
                    print(f"    [{r.memory_type}] {r.content[:60]}...{confidence_str}")
            else:
                print("    No results found")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
