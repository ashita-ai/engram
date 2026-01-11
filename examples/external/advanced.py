#!/usr/bin/env python3
"""Advanced features demo - RIF, multi-hop, negation filtering.

Demonstrates Engram's advanced recall features:
- Retrieval-Induced Forgetting (RIF) - suppress competing memories
- Multi-hop reasoning via follow_links
- Negation filtering - exclude contradicted information
- Freshness and selectivity filtering
- All 6 memory types

Prerequisites:
    - Qdrant running: docker run -p 6333:6333 qdrant/qdrant
    - API key in .env: ENGRAM_OPENAI_API_KEY=sk-...
"""

import asyncio
import logging

from engram.service import EngramService
from engram.storage import EngramStorage
from engram.workflows import init_workflows, shutdown_workflows
from engram.workflows.consolidation import run_consolidation

# Suppress INFO logs from DBOS to keep output clean
logging.getLogger("dbos").setLevel(logging.WARNING)


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
    print("Engram Advanced Features Demo")
    print("=" * 70)

    # Initialize durable workflows (DBOS or Temporal based on config)
    try:
        init_workflows()
        print("  [Durable workflows initialized]")
    except Exception as e:
        print(f"  [Durable workflows not available: {e}]")

    async with EngramService.create() as engram:
        user_id = "advanced_demo"

        # Clean up any existing data from previous runs
        await cleanup_demo_data(engram.storage, user_id)

        # =====================================================================
        # Setup: Create a rich memory landscape
        # =====================================================================
        print("\n1. CREATING TEST DATA")
        print("-" * 70)

        conversations = [
            # Basic info
            ("user", "I'm Alex, a backend developer at CloudScale."),
            ("user", "My email is alex@cloudscale.io"),
            ("user", "I've been coding for 10 years, mostly in Go and Python."),
            # Preferences
            ("user", "I prefer PostgreSQL for databases. It's my go-to choice."),
            ("user", "For caching, I always use Redis."),
            ("user", "I use Neovim as my primary editor with a custom config."),
            # Positive statement that will be contradicted by negation
            ("user", "I used to use MongoDB heavily at my previous job."),
            # Negations - what is NOT true (contradicts the positive statement)
            ("user", "I don't use MongoDB anymore - too many scaling issues."),
            ("user", "I'm not interested in blockchain or Web3."),
            ("user", "I never use Windows for development, only Linux."),
            # More context
            ("user", "My favorite framework is FastAPI for Python APIs."),
            ("user", "For Go, I prefer using the standard library over frameworks."),
        ]

        negations_detected = 0
        for role, content in conversations:
            result = await engram.encode(content=content, role=role, user_id=user_id)
            negations_detected += len(result.negations)
        print(f"  Stored {len(conversations)} episodes")
        print(f"  Negations detected during encode: {negations_detected}")

        # Run consolidation to create semantic memories
        print("  Running consolidation...")
        result = await run_consolidation(
            storage=engram.storage,
            embedder=engram.embedder,
            user_id=user_id,
        )
        print(f"  Created {result.semantic_memories_created} semantic memories")
        print(f"  Created {result.links_created} links")

        # =====================================================================
        # 2. ALL 6 MEMORY TYPES
        # =====================================================================
        print("\n\n2. QUERYING ALL MEMORY TYPES")
        print("-" * 70)

        all_types = ["episodic", "factual", "semantic", "procedural", "negation", "working"]

        results = await engram.recall(
            query="Alex developer",
            user_id=user_id,
            memory_types=all_types,
            limit=20,
        )

        type_counts: dict[str, int] = {}
        for r in results:
            type_counts[r.memory_type] = type_counts.get(r.memory_type, 0) + 1

        print('  Query: "Alex developer"')
        print(f"  Results by type: {type_counts}")

        for memory_type in all_types:
            type_results = [r for r in results if r.memory_type == memory_type]
            if type_results:
                print(f"\n  [{memory_type.upper()}]")
                for r in type_results[:2]:
                    conf = f" ({r.confidence:.0%})" if r.confidence else ""
                    print(f"    {r.content[:60]}...{conf}")

        # =====================================================================
        # 3. NEGATION FILTERING
        # =====================================================================
        print("\n\n3. NEGATION FILTERING")
        print("-" * 70)
        print("  Negations automatically filter out contradicted information:\n")

        # Query for MongoDB directly - negation filter should remove it
        print('  Query: "MongoDB database"')

        # Without negation filtering - should include MongoDB episode
        results_unfiltered = await engram.recall(
            query="MongoDB database",
            user_id=user_id,
            apply_negation_filter=False,
            limit=5,
        )

        # With negation filtering - MongoDB should be filtered out
        results_filtered = await engram.recall(
            query="MongoDB database",
            user_id=user_id,
            apply_negation_filter=True,
            limit=5,
        )

        print(f"\n  Without negation filter: {len(results_unfiltered)} results")
        mongo_count_unfiltered = 0
        for r in results_unfiltered[:3]:
            mongo_flag = " [MONGODB - would be filtered]" if "mongo" in r.content.lower() else ""
            if "mongo" in r.content.lower():
                mongo_count_unfiltered += 1
            print(f"    {r.content[:50]}...{mongo_flag}")

        print(f"\n  With negation filter: {len(results_filtered)} results")
        mongo_count_filtered = 0
        for r in results_filtered[:3]:
            mongo_flag = " [MONGODB]" if "mongo" in r.content.lower() else ""
            if "mongo" in r.content.lower():
                mongo_count_filtered += 1
            print(f"    {r.content[:50]}...{mongo_flag}")

        if mongo_count_unfiltered > mongo_count_filtered:
            print(
                f"\n  ✓ Negation filter removed {mongo_count_unfiltered - mongo_count_filtered} MongoDB result(s)"
            )

        # Show stored negations
        print("\n  Stored negations:")
        negation_results = await engram.recall(
            query="don't use",
            user_id=user_id,
            memory_types=["negation"],
            limit=5,
        )
        for r in negation_results:
            print(f"    {r.content}")

        # =====================================================================
        # 4. MULTI-HOP REASONING
        # =====================================================================
        print("\n\n4. MULTI-HOP REASONING")
        print("-" * 70)
        print("  Follow links to discover connected memories:\n")

        # Without link following
        results_no_links = await engram.recall(
            query="programming preferences",
            user_id=user_id,
            follow_links=False,
            limit=3,
        )

        # With link following
        results_with_links = await engram.recall(
            query="programming preferences",
            user_id=user_id,
            follow_links=True,
            max_hops=2,
            limit=10,
        )

        print(f"  Without follow_links: {len(results_no_links)} results")
        print(f"  With follow_links (max_hops=2): {len(results_with_links)} results")

        # Show linked memories
        linked_results = [r for r in results_with_links if r.hop_distance > 0]
        if linked_results:
            print(f"\n  Discovered via links ({len(linked_results)} memories):")
            for r in linked_results[:3]:
                print(f"    [hop={r.hop_distance}] {r.content[:50]}...")

        # =====================================================================
        # 5. RETRIEVAL-INDUCED FORGETTING (RIF)
        # =====================================================================
        print("\n\n5. RETRIEVAL-INDUCED FORGETTING (RIF)")
        print("-" * 70)
        print("  Based on Anderson et al. (1994): retrieving memories")
        print("  suppresses similar non-retrieved memories.\n")

        # First, check current confidence of some memories
        all_results = await engram.recall(
            query="Alex coding",
            user_id=user_id,
            limit=10,
            rif_enabled=False,  # Don't apply RIF yet
        )

        print(f"  Before RIF: {len(all_results)} candidate memories")
        for r in all_results[:5]:
            conf = f"{r.confidence:.0%}" if r.confidence else "N/A"
            print(f"    [{r.memory_type}] conf={conf} | {r.content[:40]}...")

        # Now do a retrieval with RIF enabled
        print("\n  Retrieving top 3 with RIF enabled...")
        retrieved = await engram.recall(
            query="Alex coding",
            user_id=user_id,
            limit=3,
            rif_enabled=True,
            rif_threshold=0.5,
            rif_decay=0.1,
        )

        print(f"\n  Retrieved: {len(retrieved)} memories")
        print("  Competing memories above threshold 0.5 got confidence decay of 0.1")
        print("\n  Note: Episodic memories are exempt (immutable ground truth)")

        # =====================================================================
        # 6. FRESHNESS FILTERING
        # =====================================================================
        print("\n\n6. FRESHNESS FILTERING")
        print("-" * 70)
        print("  Filter by consolidation status:\n")

        # Best effort (default) - returns all
        results_all = await engram.recall(
            query="Alex",
            user_id=user_id,
            freshness="best_effort",
            limit=10,
        )

        # Fresh only - only fully consolidated
        results_fresh = await engram.recall(
            query="Alex",
            user_id=user_id,
            freshness="fresh_only",
            limit=10,
        )

        print(f"  freshness='best_effort': {len(results_all)} results")
        print(f"  freshness='fresh_only': {len(results_fresh)} results")

        # Count staleness
        stale_count = sum(1 for r in results_all if r.staleness.value == "stale")
        fresh_count = sum(1 for r in results_all if r.staleness.value == "fresh")
        print(f"\n  Staleness breakdown: {fresh_count} fresh, {stale_count} stale")

        # =====================================================================
        # 7. SELECTIVITY FILTERING
        # =====================================================================
        print("\n\n7. SELECTIVITY FILTERING")
        print("-" * 70)
        print("  Filter semantic memories by context-specificity:\n")

        for min_sel in [0.0, 0.3, 0.5]:
            results = await engram.recall(
                query="Alex preferences",
                user_id=user_id,
                memory_types=["semantic"],
                min_selectivity=min_sel,
                limit=10,
            )
            print(f"  min_selectivity={min_sel}: {len(results)} semantic memories")

    print("\n" + "=" * 70)
    print("Advanced features demo complete!")
    print("=" * 70)

    # Cleanup durable workflows
    shutdown_workflows()


if __name__ == "__main__":
    asyncio.run(main())
