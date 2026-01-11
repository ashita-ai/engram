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
        # 2. RECALL: Semantic search across memory types
        # =====================================================================
        print("\n\n2. SEMANTIC RECALL")
        print("-" * 70)
        print("  One query searches across all memory types at once.\n")

        # Query that will hit multiple memory types
        print('  Query: "contact information email address"')
        results = await engram.recall(
            query="contact information email address",
            user_id=user_id,
            limit=5,
        )

        # Group by memory type to show the diversity
        by_type: dict[str, list[str]] = {}
        for r in results:
            if r.memory_type not in by_type:
                by_type[r.memory_type] = []
            by_type[r.memory_type].append(r.content[:50])

        for mem_type, contents in by_type.items():
            print(f"  [{mem_type.upper()}]")
            for content in contents[:2]:
                print(f"    → {content}...")
            print()

        # =====================================================================
        # 3. MEMORY TYPE FILTERING
        # =====================================================================
        print("\n\n3. MEMORY TYPES: RAW vs DERIVED")
        print("-" * 70)
        print("  Engram separates ground truth (episodic) from derived knowledge.\n")

        # Episodic = Ground Truth
        print("  EPISODIC (Ground Truth) - Exact user statements:")
        results = await engram.recall(
            query="email", user_id=user_id, memory_types=["episodic"], limit=2
        )
        for r in results:
            print(f'    "{r.content}"')

        # Factual = Pattern-Extracted
        print("\n  FACTUAL (Extracted) - Structured data from patterns:")
        results = await engram.recall(
            query="email", user_id=user_id, memory_types=["factual"], limit=2
        )
        for r in results:
            print(f"    {r.content}  ← extracted with {r.confidence:.0%} confidence")

        # Negation = What's NOT true
        print("\n  NEGATION (Extracted) - What the user explicitly doesn't do:")
        results = await engram.recall(
            query="programming", user_id=user_id, memory_types=["negation"], limit=2
        )
        for r in results:
            print(f'    "{r.content}"  ← use to filter contradicted info')

        print("\n  Key insight: Episodic is immutable. Derived memories trace back to it.")

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
        print("\n\n5. CONFIDENCE-BASED FILTERING")
        print("-" * 70)
        print("  Control precision vs recall with min_confidence threshold.\n")

        # Show how confidence filtering works
        all_results = await engram.recall(query="Jordan", user_id=user_id, limit=8)

        print("  All memories (no confidence filter):")
        for r in all_results[:4]:
            conf = f"{r.confidence:.0%}" if r.confidence else "---"
            print(f"    [{r.memory_type:8}] {conf:>4} conf | {r.content[:38]}...")

        # Only high-confidence derived memories (factual, semantic)
        high_conf = await engram.recall(
            query="Jordan",
            user_id=user_id,
            min_confidence=0.85,
            memory_types=["factual", "semantic"],  # Only derived types have confidence
            limit=5,
        )

        print(f"\n  With min_confidence=0.85 (derived memories only): {len(high_conf)} result(s)")
        for r in high_conf[:3]:
            conf = f"{r.confidence:.0%}" if r.confidence else "---"
            print(f"    [{r.memory_type:8}] {conf} conf | {r.content[:38]}...")
        print("\n  Note: Episodic memories are ground truth - no confidence needed.")
        print("  Derived memories (factual, semantic) track extraction certainty.")

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
