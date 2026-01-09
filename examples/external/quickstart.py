#!/usr/bin/env python3
"""Quickstart demo - core encode/recall workflow with verification.

Demonstrates:
- encode(): Store episodes and extract facts
- recall(): Semantic search with confidence filtering
- verify(): Trace any memory back to its source
- Multi-tenancy isolation

Prerequisites:
    - Qdrant running: docker run -p 6333:6333 qdrant/qdrant
    - API key in .env: ENGRAM_OPENAI_API_KEY=sk-...
"""

import asyncio

from engram.service import EngramService


async def main() -> None:
    print("=" * 70)
    print("Engram Quickstart Demo")
    print("=" * 70)

    async with EngramService.create() as engram:
        user_id = "quickstart_demo"

        # =====================================================================
        # 1. ENCODE: Store episodes and extract facts
        # =====================================================================
        print("\n1. ENCODING MEMORIES")
        print("-" * 70)

        messages = [
            ("user", "Hi! I'm Jordan, a data scientist at TechFlow Inc."),
            ("assistant", "Nice to meet you, Jordan! What kind of data science work do you do?"),
            ("user", "I specialize in NLP and work mainly with Python and PyTorch."),
            ("user", "My work email is jordan.ds@techflow.io"),
            ("user", "I prefer dark mode and vim keybindings in all my tools."),
            ("user", "I don't use R anymore - switched completely to Python."),
        ]

        for role, content in messages:
            result = await engram.encode(content=content, role=role, user_id=user_id)
            facts_info = f", {len(result.facts)} facts" if result.facts else ""
            print(f"  [{role}] {content[:50]}...{facts_info}")

        print(f"\n  Stored {len(messages)} episodes")

        # =====================================================================
        # 2. RECALL: Semantic search
        # =====================================================================
        print("\n\n2. BASIC RECALL")
        print("-" * 70)

        queries = [
            "What is Jordan's email?",
            "What programming languages does Jordan use?",
            "Where does Jordan work?",
        ]

        for query in queries:
            print(f'\n  Query: "{query}"')
            results = await engram.recall(query=query, user_id=user_id, limit=3)
            for r in results:
                conf = f" ({r.confidence:.0%})" if r.confidence else ""
                print(f"    [{r.memory_type}] {r.content[:55]}...{conf}")

        # =====================================================================
        # 3. CONFIDENCE FILTERING
        # =====================================================================
        print("\n\n3. CONFIDENCE-GATED RECALL")
        print("-" * 70)
        print("  Filter results by minimum confidence level:\n")

        query = "programming"

        for min_conf in [0.9, 0.7, 0.5]:
            results = await engram.recall(
                query=query,
                user_id=user_id,
                min_confidence=min_conf,
                limit=5,
            )
            fact_count = sum(1 for r in results if r.memory_type == "factual")
            print(f"  min_confidence={min_conf}: {len(results)} results ({fact_count} facts)")

        # =====================================================================
        # 4. VERIFY: Trace back to source
        # =====================================================================
        print("\n\n4. SOURCE VERIFICATION")
        print("-" * 70)
        print("  Every derived memory traces back to source episodes:\n")

        # Find a fact to verify
        results = await engram.recall(
            query="email",
            user_id=user_id,
            memory_types=["factual"],
            limit=1,
        )

        if results:
            fact = results[0]
            print(f'  Memory: "{fact.content}"')
            print(f"  Type: {fact.memory_type}")
            print(f"  ID: {fact.memory_id}")

            # Verify it
            verification = await engram.verify(fact.memory_id, user_id=user_id)
            print("\n  Verification:")
            print(f"    Verified: {verification.verified}")
            print(f"    Method: {verification.extraction_method}")
            print(f"    Confidence: {verification.confidence:.0%}")
            print(f"    Explanation: {verification.explanation}")

            if verification.source_episodes:
                print("\n  Source Episode:")
                src = verification.source_episodes[0]
                print(f"    \"{src['content']}\"")
                print(f"    Timestamp: {src['timestamp']}")

        # =====================================================================
        # 5. INCLUDE SOURCES
        # =====================================================================
        print("\n\n5. RECALL WITH SOURCES")
        print("-" * 70)
        print("  Include source episodes directly in recall results:\n")

        results = await engram.recall(
            query="Jordan's job",
            user_id=user_id,
            include_sources=True,
            limit=2,
        )

        for r in results:
            print(f"  [{r.memory_type}] {r.content[:50]}...")
            if r.source_episodes:
                for src in r.source_episodes[:1]:
                    print(f'    Source: "{src.content[:40]}..."')

        # =====================================================================
        # 6. MEMORY TYPE FILTERING
        # =====================================================================
        print("\n\n6. FILTER BY MEMORY TYPE")
        print("-" * 70)

        for types in [["episodic"], ["factual"], ["episodic", "factual"]]:
            results = await engram.recall(
                query="Jordan",
                user_id=user_id,
                memory_types=types,
                limit=10,
            )
            type_counts = {}
            for r in results:
                type_counts[r.memory_type] = type_counts.get(r.memory_type, 0) + 1
            print(f"  memory_types={types}: {type_counts}")

        # =====================================================================
        # 7. WORKING MEMORY
        # =====================================================================
        print("\n\n7. WORKING MEMORY")
        print("-" * 70)

        working = engram.get_working_memory()
        print(f"  Current session has {len(working)} episodes in working memory")
        print("  Working memory is volatile - cleared when session ends")

    print("\n" + "=" * 70)
    print("Quickstart complete!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
