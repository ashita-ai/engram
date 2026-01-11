#!/usr/bin/env python3
"""Quickstart demo - core encode/recall workflow with verification.

Demonstrates:
- encode(): Store episodes and extract facts automatically
- recall(): Semantic search across memory types
- verify(): Trace any memory back to its source
- Confidence-based filtering

Prerequisites:
    - Qdrant running: docker run -p 6333:6333 qdrant/qdrant
    - API key in .env: ENGRAM_OPENAI_API_KEY=sk-...
"""

import asyncio

from engram.service import EngramService
from engram.storage import EngramStorage


async def cleanup_demo_data(storage: EngramStorage, user_id: str) -> None:
    """Delete all data for the demo user to ensure clean slate."""
    from qdrant_client import models

    collections = ["episodic", "semantic", "factual", "negation", "procedural"]
    for memory_type in collections:
        collection = f"engram_{memory_type}"
        try:
            await storage.client.delete(
                collection_name=collection,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="user_id",
                                match=models.MatchValue(value=user_id),
                            )
                        ]
                    )
                ),
            )
        except Exception:
            pass  # Collection might not exist


async def main() -> None:
    print("=" * 70)
    print("Engram Quickstart Demo")
    print("=" * 70)

    async with EngramService.create() as engram:
        user_id = "quickstart_demo"

        # Clean up any existing data from previous runs
        await cleanup_demo_data(engram.storage, user_id)

        # =====================================================================
        # 1. ENCODE: Store episodes and extract facts
        # =====================================================================
        print("\n1. ENCODING MEMORIES")
        print("-" * 70)
        print("  Episodes are stored verbatim. Facts are auto-extracted.\n")

        messages = [
            ("user", "Hi! I'm Jordan, a data scientist at TechFlow Inc."),
            ("assistant", "Nice to meet you, Jordan! What kind of data science work?"),
            ("user", "I specialize in NLP and work mainly with Python and PyTorch."),
            ("user", "My work email is jordan.ds@techflow.io"),
            ("user", "I prefer dark mode and vim keybindings in all my tools."),
            ("user", "I don't use R anymore - switched completely to Python."),
        ]

        facts_extracted = []
        negations_extracted = []

        for role, content in messages:
            result = await engram.encode(content=content, role=role, user_id=user_id)
            extras = []
            if result.facts:
                facts_extracted.extend(result.facts)
                extras.append("+fact")
            if result.negations:
                negations_extracted.extend(result.negations)
                extras.append("+negation")
            extras_str = f"  [{', '.join(extras)}]" if extras else ""
            print(f"  [{role:9}] {content[:50]}...{extras_str}")

        print("\n  Results:")
        print(f"    Episodes stored: {len(messages)}")
        print(f"    Facts extracted: {len(facts_extracted)} (emails, phones, etc.)")
        print(f'    Negations found: {len(negations_extracted)} ("I don\'t use...")')

        # =====================================================================
        # 2. RECALL: Semantic search
        # =====================================================================
        print("\n\n2. SEMANTIC RECALL")
        print("-" * 70)
        print("  Natural language queries search across all memory types.\n")

        # Query 1: Email - should find the factual memory
        print('  Query: "contact information"')
        results = await engram.recall(query="contact email address", user_id=user_id, limit=3)
        for r in results[:3]:
            conf = f" ({r.confidence:.0%})" if r.confidence else ""
            print(f"    [{r.memory_type:8}] {r.content[:45]}...{conf}")

        # Query 2: Programming - should find Python/PyTorch content
        print('\n  Query: "programming tools and frameworks"')
        results = await engram.recall(
            query="programming languages frameworks tools", user_id=user_id, limit=3
        )
        for r in results[:3]:
            conf = f" ({r.confidence:.0%})" if r.confidence else ""
            print(f"    [{r.memory_type:8}] {r.content[:45]}...{conf}")

        # Query 3: Work - should find TechFlow
        print('\n  Query: "company employer"')
        results = await engram.recall(query="company employer workplace", user_id=user_id, limit=3)
        for r in results[:3]:
            conf = f" ({r.confidence:.0%})" if r.confidence else ""
            print(f"    [{r.memory_type:8}] {r.content[:45]}...{conf}")

        # =====================================================================
        # 3. MEMORY TYPE FILTERING
        # =====================================================================
        print("\n\n3. FILTER BY MEMORY TYPE")
        print("-" * 70)
        print("  Search specific memory types for different use cases.\n")

        # Only episodic - raw conversation
        print("  memory_types=['episodic'] - Raw conversation history:")
        results = await engram.recall(
            query="Jordan", user_id=user_id, memory_types=["episodic"], limit=3
        )
        for r in results[:2]:
            print(f'    "{r.content[:55]}..."')

        # Only factual - extracted facts
        print("\n  memory_types=['factual'] - Extracted structured data:")
        results = await engram.recall(
            query="Jordan", user_id=user_id, memory_types=["factual"], limit=3
        )
        for r in results:
            print(f"    {r.content} (confidence: {r.confidence:.0%})")

        # Only negation - what's NOT true
        print("\n  memory_types=['negation'] - What the user DOESN'T do:")
        results = await engram.recall(
            query="tools", user_id=user_id, memory_types=["negation"], limit=3
        )
        for r in results:
            print(f'    "{r.content}" (confidence: {r.confidence:.0%})')

        # =====================================================================
        # 4. VERIFY: Trace back to source
        # =====================================================================
        print("\n\n4. SOURCE VERIFICATION")
        print("-" * 70)
        print("  Every derived memory traces back to its source episode.\n")

        # Find a fact to verify
        results = await engram.recall(
            query="email", user_id=user_id, memory_types=["factual"], limit=1
        )

        if results:
            fact = results[0]
            print("  Derived Memory:")
            print(f'    Content: "{fact.content}"')
            print(f"    Type: {fact.memory_type}")
            print(f"    Confidence: {fact.confidence:.0%}")

            # Verify it
            verification = await engram.verify(fact.memory_id, user_id=user_id)
            print("\n  Verification Result:")
            print(f"    Verified: {verification.verified}")
            print(f"    Method: {verification.extraction_method}")

            if verification.source_episodes:
                src = verification.source_episodes[0]
                print("\n  Source Episode (ground truth):")
                print(f"    \"{src['content']}\"")
                print(f"    Role: {src['role']}")
                print(f"    Timestamp: {src['timestamp']}")

        # =====================================================================
        # 5. CONFIDENCE FILTERING
        # =====================================================================
        print("\n\n5. CONFIDENCE-GATED RECALL")
        print("-" * 70)
        print("  Filter by confidence to control precision vs recall.\n")

        # Show how confidence filtering works
        all_results = await engram.recall(query="Jordan data", user_id=user_id, limit=10)

        print("  All memories with confidence scores:")
        for r in all_results[:5]:
            conf = f"{r.confidence:.0%}" if r.confidence else "N/A"
            print(f"    [{r.memory_type:8}] conf={conf:>4} | {r.content[:40]}...")

        # Only high-confidence
        high_conf = await engram.recall(
            query="Jordan data", user_id=user_id, min_confidence=0.85, limit=10
        )
        print(f"\n  With min_confidence=0.85: {len(high_conf)} results (facts only)")

        # =====================================================================
        # 6. WORKING MEMORY
        # =====================================================================
        print("\n\n6. WORKING MEMORY")
        print("-" * 70)

        working = engram.get_working_memory()
        print(f"  Current session: {len(working)} episodes in working memory")
        print("  Working memory is:")
        print("    - Volatile (cleared when session ends)")
        print("    - Instant access (no DB round-trip)")
        print("    - Included in recall by default")

    print("\n" + "=" * 70)
    print("Quickstart complete!")
    print("=" * 70)
    print("""
Key Takeaways:
1. encode() stores episodes AND auto-extracts facts/negations
2. recall() searches semantically across all memory types
3. verify() traces any derived memory to its source
4. Confidence scores distinguish certain from uncertain
5. Memory types serve different purposes (raw vs derived)
""")


if __name__ == "__main__":
    asyncio.run(main())
