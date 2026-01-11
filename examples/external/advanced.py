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
        # 2. MEMORY TYPES
        # =====================================================================
        print("\n\n2. MEMORY TYPES")
        print("-" * 70)
        print("  Engram stores 5 persistent memory types:\n")

        # Query persisted types only (skip working - it duplicates episodic)
        persisted_types = ["episodic", "factual", "semantic", "procedural", "negation"]

        results = await engram.recall(
            query="Alex developer",
            user_id=user_id,
            memory_types=persisted_types,
            limit=20,
        )

        type_counts: dict[str, int] = {}
        for r in results:
            type_counts[r.memory_type] = type_counts.get(r.memory_type, 0) + 1

        print(f'  Query: "Alex developer" → {type_counts}')

        for memory_type in persisted_types:
            type_results = [r for r in results if r.memory_type == memory_type]
            if type_results:
                desc = {
                    "episodic": "raw conversations",
                    "factual": "pattern-extracted",
                    "semantic": "LLM-inferred",
                    "procedural": "behavioral patterns",
                    "negation": "what is NOT true",
                }.get(memory_type, "")
                print(f"\n  [{memory_type.upper()}] ({desc})")
                for r in type_results[:2]:
                    conf = f" ({r.confidence:.0%})" if r.confidence else ""
                    print(f"    {r.content[:55]}...{conf}")

        # =====================================================================
        # 3. NEGATION FILTERING
        # =====================================================================
        print("\n\n3. NEGATION FILTERING")
        print("-" * 70)
        print("  Negations track what is NOT true and filter contradicted info.\n")

        # First, show what negations were detected
        print("  Stored negations (detected during encode):")
        negation_results = await engram.recall(
            query="don't use never",
            user_id=user_id,
            memory_types=["negation"],
            limit=5,
        )
        for r in negation_results:
            print(f"    ✗ {r.content}")

        # Query for databases - should show PostgreSQL preference, not MongoDB
        print('\n  Query: "database preferences" (episodic only)')

        # Without negation filtering - includes the MongoDB statement
        results_unfiltered = await engram.recall(
            query="database preferences",
            user_id=user_id,
            memory_types=["episodic"],
            apply_negation_filter=False,
            limit=4,
        )

        # With negation filtering - MongoDB should be filtered
        results_filtered = await engram.recall(
            query="database preferences",
            user_id=user_id,
            memory_types=["episodic"],
            apply_negation_filter=True,
            limit=4,
        )

        # Count MongoDB mentions
        mongo_unfiltered = [r for r in results_unfiltered if "mongo" in r.content.lower()]
        mongo_filtered = [r for r in results_filtered if "mongo" in r.content.lower()]

        print(f"\n  WITHOUT negation filter ({len(results_unfiltered)} results):")
        for r in results_unfiltered[:3]:
            flag = " ← CONTRADICTED" if "mongo" in r.content.lower() else ""
            print(f"    {r.content[:55]}...{flag}")

        print(f"\n  WITH negation filter ({len(results_filtered)} results):")
        for r in results_filtered[:3]:
            print(f"    {r.content[:55]}...")

        removed = len(mongo_unfiltered) - len(mongo_filtered)
        if removed > 0:
            print(f"\n  ✓ Negation filter removed {removed} contradicted MongoDB result(s)")
        print("\n  Use case: Prevent hallucinating that user still uses MongoDB.")

        # =====================================================================
        # 4. MULTI-HOP REASONING
        # =====================================================================
        print("\n\n4. MULTI-HOP REASONING")
        print("-" * 70)
        print("  Follow links to discover connected memories.\n")

        # Without link following
        results_no_links = await engram.recall(
            query="programming preferences",
            user_id=user_id,
            memory_types=["semantic", "factual"],  # Types that have links
            follow_links=False,
            limit=5,
        )

        # With link following
        results_with_links = await engram.recall(
            query="programming preferences",
            user_id=user_id,
            memory_types=["semantic", "factual"],
            follow_links=True,
            max_hops=2,
            limit=10,
        )

        print(f"  Without follow_links: {len(results_no_links)} results")
        for r in results_no_links[:3]:
            links = f" (links to {len(r.related_ids)})" if r.related_ids else ""
            print(f"    {r.content[:50]}...{links}")

        print(f"\n  With follow_links (max_hops=2): {len(results_with_links)} results")

        # Show linked memories discovered via traversal
        linked_results = [r for r in results_with_links if r.hop_distance and r.hop_distance > 0]
        if linked_results:
            print("\n  Discovered via link traversal:")
            for r in linked_results[:3]:
                print(f"    [hop {r.hop_distance}] {r.content[:50]}...")
        else:
            print("\n  (No additional memories discovered via links)")
            print("  Links are created during consolidation when memories are related.")

        # =====================================================================
        # 5. RETRIEVAL-INDUCED FORGETTING (RIF)
        # =====================================================================
        print("\n\n5. RETRIEVAL-INDUCED FORGETTING (RIF)")
        print("-" * 70)
        print("  Based on Anderson et al. (1994): retrieving memories")
        print("  suppresses similar non-retrieved memories.\n")

        # Get semantic/factual memories only (these have confidence to decay)
        before_rif = await engram.recall(
            query="Alex coding",
            user_id=user_id,
            memory_types=["semantic", "factual"],
            limit=10,
            rif_enabled=False,
        )

        # Store confidence before RIF
        conf_before: dict[str, float] = {}
        print("  BEFORE RIF (semantic/factual memories):")
        for r in before_rif[:5]:
            conf_before[r.memory_id] = r.confidence or 0
            print(f"    [{r.confidence:.0%}] {r.content[:50]}...")

        # Apply RIF: retrieve top 2, suppress competitors
        print("\n  Retrieving top 2 with RIF enabled...")
        print("  (Competitors above threshold 0.5 get confidence decay of 0.1)")
        retrieved = await engram.recall(
            query="Alex coding",
            user_id=user_id,
            memory_types=["semantic", "factual"],
            limit=2,
            rif_enabled=True,
            rif_threshold=0.5,
            rif_decay=0.1,
        )

        retrieved_ids = {r.memory_id for r in retrieved}
        print(f"\n  RETRIEVED ({len(retrieved)}):")
        for r in retrieved:
            print(f"    ✓ [{r.confidence:.0%}] {r.content[:50]}...")

        # Check confidence after RIF
        after_rif = await engram.recall(
            query="Alex coding",
            user_id=user_id,
            memory_types=["semantic", "factual"],
            limit=10,
            rif_enabled=False,
        )

        print("\n  SUPPRESSED (competitors not retrieved):")
        suppressed_count = 0
        for r in after_rif:
            if r.memory_id not in retrieved_ids and r.memory_id in conf_before:
                old_conf = conf_before[r.memory_id]
                new_conf = r.confidence or 0
                if new_conf < old_conf:
                    suppressed_count += 1
                    print(f"    ✗ [{old_conf:.0%}→{new_conf:.0%}] {r.content[:45]}...")
        if suppressed_count == 0:
            print("    (No memories suppressed in this run)")

        print("\n  Note: Episodic memories are exempt (immutable ground truth)")

        # =====================================================================
        # 6. FRESHNESS FILTERING
        # =====================================================================
        print("\n\n6. FRESHNESS FILTERING")
        print("-" * 70)
        print("  Filter by consolidation status.\n")
        print("  Episodes are 'stale' until consolidated into semantic memories.")
        print("  Semantic/factual memories are always 'fresh' (derived from consolidation).\n")

        # Add new episodes AFTER consolidation to create stale memories
        print("  Adding new episodes (not yet consolidated)...")
        await engram.encode(
            content="I'm also learning Kubernetes for container orchestration.",
            role="user",
            user_id=user_id,
        )
        await engram.encode(
            content="My preferred cloud provider is AWS.",
            role="user",
            user_id=user_id,
        )

        # Query including episodic to show stale vs fresh
        results_all = await engram.recall(
            query="Alex cloud",
            user_id=user_id,
            memory_types=["episodic", "semantic"],
            freshness="best_effort",
            limit=10,
        )

        fresh_mems = [r for r in results_all if r.staleness.value == "fresh"]
        stale_mems = [r for r in results_all if r.staleness.value == "stale"]

        print(f"\n  Query: 'Alex cloud' → {len(results_all)} total")
        print(f"    Fresh (consolidated): {len(fresh_mems)}")
        print(f"    Stale (unconsolidated): {len(stale_mems)}")

        if stale_mems:
            print("\n  STALE episodes (new, not yet consolidated):")
            for r in stale_mems[:2]:
                print(f"    ⚠ [{r.memory_type}] {r.content[:45]}...")

        if fresh_mems:
            print("\n  FRESH memories (already consolidated):")
            for r in fresh_mems[:2]:
                print(f"    ✓ [{r.memory_type}] {r.content[:45]}...")

        # Show filtering - check if stale memories are excluded
        results_fresh_only = await engram.recall(
            query="Alex cloud",
            user_id=user_id,
            memory_types=["episodic", "semantic"],
            freshness="fresh_only",
            limit=10,
        )

        # Check if the stale memories are in fresh_only results
        stale_ids = {r.memory_id for r in stale_mems}
        stale_in_fresh = [r for r in results_fresh_only if r.memory_id in stale_ids]

        print(f"\n  With freshness='fresh_only': {len(results_fresh_only)} results")
        if len(stale_in_fresh) == 0 and len(stale_mems) > 0:
            print("  ✓ Stale episodes excluded from results")
        else:
            print(f"  Note: {len(stale_in_fresh)} stale memories still in results")

    print("\n" + "=" * 70)
    print("Advanced features demo complete!")
    print("=" * 70)

    # Cleanup durable workflows
    shutdown_workflows()


if __name__ == "__main__":
    asyncio.run(main())
