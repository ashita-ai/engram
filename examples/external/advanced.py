#!/usr/bin/env python3
"""Advanced features demo - multi-hop reasoning, negation filtering.

Demonstrates Engram's advanced recall features:
- Multi-hop reasoning via follow_links
- Negation filtering - exclude contradicted information
- Summarization status filtering
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
        # Demo identifiers
        user_id = "advanced_demo"
        org_id = "demo_org"
        session_id = "advanced_session_001"

        # Clean up any existing data from previous runs
        await cleanup_demo_data(engram.storage, user_id)

        # Verify cleanup worked
        stats_after_cleanup = await engram.storage.get_memory_stats(user_id)
        if stats_after_cleanup.episodes > 0 or stats_after_cleanup.negation > 0:
            print(
                f"  [WARNING: Cleanup incomplete - episodes={stats_after_cleanup.episodes}, negation={stats_after_cleanup.negation}]"
            )
        else:
            print("  [Previous demo data cleaned up]")

        # Clean up any existing data from previous runs
        await cleanup_demo_data(engram.storage, user_id)

        # Verify cleanup worked
        stats_after_cleanup = await engram.storage.get_memory_stats(user_id)
        if stats_after_cleanup.episodes > 0 or stats_after_cleanup.negation > 0:
            print(
                f"  [WARNING: Cleanup incomplete - episodes={stats_after_cleanup.episodes}, negation={stats_after_cleanup.negation}]"
            )
        else:
            print("  [Previous demo data cleaned up]")

        # =====================================================================
        # Setup: Create memories in batches to demonstrate links
        # =====================================================================
        print("\n1. CREATING TEST DATA (in batches for linking)")
        print("-" * 70)

        # BATCH 1: Basic info about Alex
        batch1 = [
            ("user", "I'm Alex, a backend developer at CloudScale."),
            ("user", "My email is alex@cloudscale.io"),
            ("user", "I've been coding for 10 years, mostly in Go and Python."),
            ("user", "I prefer PostgreSQL for databases. It's my go-to choice."),
        ]

        print("  Batch 1: Basic profile...")
        for role, content in batch1:
            await engram.encode(
                content=content, role=role, user_id=user_id, org_id=org_id, session_id=session_id
            )

        # Consolidate batch 1 → creates first semantic memory
        result1 = await run_consolidation(
            storage=engram.storage, embedder=engram.embedder, user_id=user_id, org_id=org_id
        )
        print(f"    → {len(batch1)} episodes → {result1.semantic_memories_created} semantic")

        # BATCH 2: Technical preferences (will link to batch 1)
        batch2 = [
            ("user", "For caching, I always use Redis alongside PostgreSQL."),
            ("user", "I use Neovim as my primary editor with a custom config."),
            ("user", "My favorite framework is FastAPI for Python APIs."),
            ("user", "For Go, I prefer using the standard library over frameworks."),
        ]

        print("  Batch 2: Technical preferences...")
        for role, content in batch2:
            await engram.encode(
                content=content, role=role, user_id=user_id, org_id=org_id, session_id=session_id
            )

        # Consolidate batch 2 → should create links to batch 1's semantic
        result2 = await run_consolidation(
            storage=engram.storage, embedder=engram.embedder, user_id=user_id, org_id=org_id
        )
        print(f"    → {len(batch2)} episodes → {result2.semantic_memories_created} semantic")
        print(f"    → {result2.links_created} links created to existing memories")

        # BATCH 3: Negations and corrections
        batch3 = [
            ("user", "I used to use MongoDB heavily at my previous job."),
            ("user", "I don't use MongoDB anymore - too many scaling issues."),
            ("user", "I'm not interested in blockchain or Web3."),
            ("user", "I never use Windows for development, only Linux."),
        ]

        print("  Batch 3: Corrections and negations...")
        negations_detected = 0
        for role, content in batch3:
            result = await engram.encode(
                content=content, role=role, user_id=user_id, org_id=org_id, session_id=session_id
            )
            negations_detected += len(result.negations)
        print(f"    → {len(batch3)} episodes, {negations_detected} negations detected")

        # Consolidate batch 3
        result3 = await run_consolidation(
            storage=engram.storage, embedder=engram.embedder, user_id=user_id, org_id=org_id
        )
        print(f"    → {result3.semantic_memories_created} semantic, {result3.links_created} links")

        total_episodes = len(batch1) + len(batch2) + len(batch3)
        print(f"\n  TOTAL: {total_episodes} episodes stored")

        # =====================================================================
        # 2. MEMORY TYPES
        # =====================================================================
        print("\n\n2. MEMORY TYPES")
        print("-" * 70)
        print("  Engram stores 5 persistent memory types:\n")

        # Get actual counts from storage
        stats = await engram.storage.get_memory_stats(user_id)
        print(
            f"  Counts: episodes={stats.episodes}, facts={stats.facts}, "
            f"semantic={stats.semantic}, negation={stats.negation}"
        )

        # Show examples of each type
        for memory_type, desc in [
            ("episodic", "raw conversations"),
            ("factual", "pattern-extracted"),
            ("semantic", "LLM-summarized"),
            ("negation", "what is NOT true"),
        ]:
            results = await engram.recall(
                query="Alex developer PostgreSQL",
                user_id=user_id,
                memory_types=[memory_type],
                limit=2,
            )
            if results:
                print(f"\n  [{memory_type.upper()}] ({desc})")
                for r in results[:2]:
                    conf = f" ({r.confidence:.0%})" if r.confidence else ""
                    print(f"    {r.content[:55]}...{conf}")

        # =====================================================================
        # 3. NEGATION FILTERING
        # =====================================================================
        print("\n\n3. NEGATION FILTERING")
        print("-" * 70)
        print("  Negations track what is NOT true and filter contradicted info.\n")

        # Show stored negations
        print("  Stored negations:")
        negation_results = await engram.recall(
            query="don't use never",
            user_id=user_id,
            memory_types=["negation"],
            limit=5,
        )
        for r in negation_results:
            print(f"    ✗ {r.content}")

        # Query for databases - should show PostgreSQL preference, not MongoDB
        print('\n  Query: "database" (episodic only)')

        # Without negation filtering - includes the MongoDB statement
        results_unfiltered = await engram.recall(
            query="database",
            user_id=user_id,
            memory_types=["episodic"],
            apply_negation_filter=False,
            limit=4,
        )

        # With negation filtering - MongoDB should be filtered
        results_filtered = await engram.recall(
            query="database",
            user_id=user_id,
            memory_types=["episodic"],
            apply_negation_filter=True,
            limit=4,
        )

        print(f"\n  WITHOUT negation filter ({len(results_unfiltered)} results):")
        for r in results_unfiltered[:4]:
            flag = " ← CONTRADICTED" if "mongo" in r.content.lower() else ""
            print(f"    {r.content[:55]}...{flag}")

        print(f"\n  WITH negation filter ({len(results_filtered)} results):")
        for r in results_filtered[:4]:
            print(f"    {r.content[:55]}...")

        removed = len(results_unfiltered) - len(results_filtered)
        if removed > 0:
            print(f"\n  ✓ Removed {removed} contradicted result(s) — fewer but accurate")
        print(
            "\n  Behavior: Returns fewer results rather than backfilling with irrelevant content."
        )

        # =====================================================================
        # 4. MULTI-HOP REASONING
        # =====================================================================
        print("\n\n4. MULTI-HOP REASONING")
        print("-" * 70)
        print("  Follow links to discover connected memories.\n")

        # Check what links exist
        semantic_mems = await engram.storage.list_semantic_memories(user_id)
        total_links = sum(len(s.related_ids) for s in semantic_mems)
        print(f"  Semantic memories: {len(semantic_mems)}")
        print(f"  Total links between them: {total_links}")

        if total_links > 0:
            # Show the linked memories
            for sem in semantic_mems:
                if sem.related_ids:
                    print(f"\n  Memory: {sem.content[:50]}...")
                    print(f"    Links to: {sem.related_ids}")

            # Query with link following
            results_no_links = await engram.recall(
                query="PostgreSQL database",
                user_id=user_id,
                memory_types=["semantic"],
                follow_links=False,
                limit=5,
            )

            results_with_links = await engram.recall(
                query="PostgreSQL database",
                user_id=user_id,
                memory_types=["semantic"],
                follow_links=True,
                max_hops=2,
                limit=5,
            )

            print("\n  Query 'PostgreSQL database':")
            print(f"    Without follow_links: {len(results_no_links)} results")
            print(f"    With follow_links: {len(results_with_links)} results")

            # Show hop distances
            linked_results = [r for r in results_with_links if r.hop_distance > 0]
            if linked_results:
                print("\n  Discovered via link traversal:")
                for r in linked_results:
                    print(f"    [hop {r.hop_distance}] {r.content[:45]}...")
        else:
            print("\n  Note: No links created in this run.")
            print(
                "  Links are created when LLM identifies relationships between semantic memories."
            )
            print("  Try running consolidation multiple times to build more connections.")

        # =====================================================================
        # 5. SUMMARIZATION STATUS FILTERING
        # =====================================================================
        print("\n\n5. SUMMARIZATION STATUS FILTERING")
        print("-" * 70)
        print("  Filter by consolidation status.\n")
        print("  Episodes can be 'unsummarized' (not yet included in semantic summaries)")
        print("  or 'summarized' (already compressed into semantic memory).\n")

        # Add new episodes AFTER consolidation to show unsummarized state
        print("  Adding 2 new episodes (not yet summarized)...")
        await engram.encode(
            content="I'm also learning Kubernetes for container orchestration.",
            role="user",
            user_id=user_id,
            org_id=org_id,
            session_id=session_id,
        )
        await engram.encode(
            content="My preferred cloud provider is AWS.",
            role="user",
            user_id=user_id,
            org_id=org_id,
            session_id=session_id,
        )

        # Query including episodic to show summarized vs unsummarized
        results_all = await engram.recall(
            query="cloud containers",
            user_id=user_id,
            memory_types=["episodic"],
            freshness="best_effort",
            limit=10,
        )

        summarized = [r for r in results_all if r.staleness.value == "fresh"]
        unsummarized = [r for r in results_all if r.staleness.value == "stale"]

        print(f"\n  Query: 'cloud containers' → {len(results_all)} episodes")
        print(f"    Summarized: {len(summarized)}")
        print(f"    Unsummarized: {len(unsummarized)}")

        if unsummarized:
            print("\n  UNSUMMARIZED (new ground truth, pending consolidation):")
            for r in unsummarized[:2]:
                print(f"    📝 {r.content[:55]}...")

        # Show filtering
        results_summarized_only = await engram.recall(
            query="cloud containers",
            user_id=user_id,
            memory_types=["episodic"],
            freshness="fresh_only",
            limit=10,
        )

        print(f"\n  Filtering to summarized-only: {len(results_summarized_only)} results")
        if len(results_summarized_only) < len(results_all):
            print("  ✓ Unsummarized episodes excluded")

    print("\n" + "=" * 70)
    print("Advanced features demo complete!")
    print("=" * 70)

    # Cleanup durable workflows
    shutdown_workflows()


if __name__ == "__main__":
    asyncio.run(main())
