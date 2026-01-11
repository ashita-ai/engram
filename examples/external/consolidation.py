#!/usr/bin/env python3
"""Consolidation workflow demo - LLM-powered semantic extraction.

Demonstrates the full consolidation pipeline:
- Episode storage (ground truth)
- LLM consolidation (semantic memory extraction)
- Memory linking (related_ids for multi-hop)
- Consolidation strength (Testing Effect)
- Side-by-side view of raw vs derived memories

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


async def get_memory_counts(storage: EngramStorage, user_id: str) -> dict[str, int]:
    """Get actual counts of memories by type from storage."""
    stats = await storage.get_memory_stats(user_id)
    return {
        "episodic": stats.episodes,
        "factual": stats.facts,
        "semantic": stats.semantic,
        "negation": stats.negation,
    }


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
    print("Engram Consolidation Demo")
    print("=" * 70)

    # Initialize durable workflows (DBOS or Temporal based on config)
    try:
        init_workflows()
        print("  [Durable workflows initialized]")
    except Exception as e:
        print(f"  [Durable workflows not available: {e}]")

    async with EngramService.create() as engram:
        user_id = "consolidation_demo"

        # Clean up any existing data from previous runs
        await cleanup_demo_data(engram.storage, user_id)
        print("  [Previous demo data cleaned up]")

        # =====================================================================
        # 1. STORE EPISODES (Ground Truth)
        # =====================================================================
        print("\n1. STORING RAW EPISODES (Ground Truth)")
        print("-" * 70)
        print("  Every message is stored verbatim as an Episode - immutable.\n")

        conversations = [
            ("user", "Hi! I'm Morgan, a machine learning engineer."),
            ("assistant", "Nice to meet you, Morgan! What areas of ML do you focus on?"),
            ("user", "I specialize in NLP and transformer architectures."),
            ("user", "My email is morgan.ml@airesearch.io"),
            ("user", "I use PyTorch exclusively - tried TensorFlow but prefer PyTorch."),
            ("user", "For experiment tracking, I use Weights & Biases."),
            ("user", "I deploy models using FastAPI and Docker."),
            ("user", "My team uses Hugging Face for pretrained models."),
            ("user", "We're currently working on a RAG system."),
            ("user", "I don't use Keras anymore - too high-level for research."),
        ]

        # Track episode IDs and which ones produced facts/negations
        episode_ids: list[str] = []
        fact_source_episode: str | None = None
        negation_source_episode: str | None = None

        for role, content in conversations:
            result = await engram.encode(content=content, role=role, user_id=user_id)
            episode_ids.append(result.episode.id)
            extras = []
            if result.facts:
                extras.append(f"+fact:{result.facts[0].category}")
                fact_source_episode = result.episode.id
            if result.negations:
                extras.append(f"+negation:{result.negations[0].negates_pattern}")
                negation_source_episode = result.episode.id
            extras_str = f"  {' '.join(extras)}" if extras else ""
            print(f"  [{role:9}] {content[:50]}...{extras_str}")

        # =====================================================================
        # 2. CHECK WHAT'S STORED BEFORE CONSOLIDATION
        # =====================================================================
        print("\n\n2. WHAT'S STORED BEFORE CONSOLIDATION")
        print("-" * 70)

        counts = await get_memory_counts(engram.storage, user_id)
        print(f"  Episodic (ground truth): {counts['episodic']} episodes")
        print(f"    └─ Factual (extracted): {counts['factual']} (email pattern)")
        print(f"    └─ Negation (extracted): {counts['negation']} (Keras pattern)")
        print(f"    └─ Semantic (LLM):       {counts['semantic']} <-- Empty! Need consolidation")

        # Show source episode linkage
        print("\n  SOURCE EPISODE LINKAGE:")
        print("  Each derived memory links back to its source episode:\n")

        # Get the factual memory and show its source
        factual_results = await engram.recall(
            query="email",
            user_id=user_id,
            memory_types=["factual"],
            limit=1,
        )
        if factual_results and fact_source_episode:
            fact = factual_results[0]
            print(f'    Factual: "{fact.content[:40]}..."')
            print(f"      → source_episode_id: {fact.source_episode_id}")
            print(f"      → matches episode #4: {fact.source_episode_id == fact_source_episode}")

        # Get the negation memory and show its source
        negation_results = await engram.recall(
            query="Keras",
            user_id=user_id,
            memory_types=["negation"],
            limit=1,
        )
        if negation_results and negation_source_episode:
            neg = negation_results[0]
            print(f'\n    Negation: "{neg.content[:40]}..."')
            print(f"      → source_episode_ids: {neg.source_episode_ids}")
            print(
                f"      → matches episode #10: {negation_source_episode in neg.source_episode_ids}"
            )

        # =====================================================================
        # 3. RUN LLM CONSOLIDATION
        # =====================================================================
        print("\n\n3. RUNNING LLM CONSOLIDATION")
        print("-" * 70)
        print("  The LLM reads episodes and extracts semantic knowledge.")
        print("  (Conservative extraction - prefers precision over recall)\n")

        result = await run_consolidation(
            storage=engram.storage,
            embedder=engram.embedder,
            user_id=user_id,
        )

        print(f"  Episodes processed:       {result.episodes_processed}")
        print(f"  Semantic memories created:{result.semantic_memories_created}")
        print(f"  Existing strengthened:    {result.memories_strengthened}")
        print(f"  Links created:            {result.links_created}")

        if result.semantic_memories_created < 3:
            print("\n  Note: LLM extraction is intentionally conservative.")
            print("  Pattern-extracted facts (email) are more reliable than LLM inference.")

        # =====================================================================
        # 4. WHAT'S STORED AFTER CONSOLIDATION
        # =====================================================================
        print("\n\n4. WHAT'S STORED AFTER CONSOLIDATION")
        print("-" * 70)

        counts = await get_memory_counts(engram.storage, user_id)
        print(f"  Episodic (ground truth): {counts['episodic']} episodes (unchanged)")
        print(f"    └─ Factual (extracted): {counts['factual']}")
        print(f"    └─ Negation (extracted): {counts['negation']}")
        print(f"    └─ Semantic (LLM):       {counts['semantic']} <-- Created by LLM!")

        # =====================================================================
        # 5. RAW vs DERIVED - Side by Side
        # =====================================================================
        print("\n\n5. RAW EPISODES vs DERIVED SEMANTIC MEMORIES")
        print("-" * 70)
        print("  Semantic memories are extracted from the episode batch.\n")

        # Get all semantic memories
        semantic_results = await engram.recall(
            query="Morgan's background",
            user_id=user_id,
            memory_types=["semantic"],
            limit=10,
        )

        print("  RAW EPISODES (ground truth):")
        for i, (role, content) in enumerate(conversations[:5], 1):
            print(f'    {i}. [{role}] "{content[:55]}..."')
        print(f"    ... and {len(conversations) - 5} more\n")

        print("  SEMANTIC MEMORIES (LLM-extracted):")
        for sem in semantic_results[:5]:
            print(f'    - "{sem.content}" ({sem.confidence:.0%})')
        if len(semantic_results) > 5:
            print(f"    ... and {len(semantic_results) - 5} more\n")
        else:
            print()

        # =====================================================================
        # 6. LINKED MEMORIES
        # =====================================================================
        print("\n6. MEMORY LINKING")
        print("-" * 70)
        print("  Semantic memories are linked to related memories.\n")

        memories_with_links = [r for r in semantic_results if r.related_ids]
        if memories_with_links:
            mem = memories_with_links[0]
            print(f'  Memory: "{mem.content}"')
            print(f"  Links to {len(mem.related_ids)} related memories:")

            # Follow links
            linked_results = await engram.recall(
                query=mem.content,
                user_id=user_id,
                follow_links=True,
                max_hops=2,
                limit=10,
            )
            for lr in linked_results[:3]:
                if lr.hop_distance and lr.hop_distance > 0:
                    print(f"    -> [hop {lr.hop_distance}] {lr.content[:50]}...")
        else:
            print("  No links yet (first consolidation run).")
            print("  Links accumulate over multiple consolidation passes.")

        # =====================================================================
        # 7. ADD MORE DATA AND CONSOLIDATE AGAIN
        # =====================================================================
        print("\n\n7. ADD MORE DATA & CONSOLIDATE AGAIN")
        print("-" * 70)

        new_messages = [
            ("user", "By the way, my full name is Morgan Chen."),
            ("user", "I'm also learning Rust for systems programming."),
        ]

        print("  Adding new episodes:")
        for role, content in new_messages:
            await engram.encode(content=content, role=role, user_id=user_id)
            print(f"    [{role}] {content}")

        result2 = await run_consolidation(
            storage=engram.storage,
            embedder=engram.embedder,
            user_id=user_id,
        )

        print("\n  Second consolidation:")
        print(f"    New episodes processed:    {result2.episodes_processed}")
        print(f"    New semantic memories:     {result2.semantic_memories_created}")
        print(f"    Existing memories strengthened: {result2.memories_strengthened}")

        # =====================================================================
        # 8. CONSOLIDATION STRENGTH (Testing Effect)
        # =====================================================================
        print("\n\n8. CONSOLIDATION STRENGTH")
        print("-" * 70)
        print("  Memories that are repeatedly consolidated become stronger.\n")

        # Get updated semantic memories
        updated_semantics = await engram.recall(
            query="Morgan",
            user_id=user_id,
            memory_types=["semantic"],
            limit=5,
        )

        for r in updated_semantics[:3]:
            strength = r.metadata.get("selectivity", 0)
            print(f'  "{r.content[:45]}..."')
            print(f"    Strength: {strength:.2f} (increases by 0.1 per consolidation)")
            print()

        # =====================================================================
        # 9. FINAL STATE
        # =====================================================================
        print("\n9. FINAL MEMORY STATE")
        print("-" * 70)

        final_counts = await get_memory_counts(engram.storage, user_id)
        print(
            f"  Episodic (ground truth): {final_counts['episodic']} episodes (10 initial + 2 added)"
        )
        print(f"    └─ Factual (extracted): {final_counts['factual']}")
        print(f"    └─ Negation (extracted): {final_counts['negation']}")
        print(f"    └─ Semantic (LLM):       {final_counts['semantic']}")

    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)
    print("""
Key Concepts:
1. Episodes are RAW ground truth (immutable, never modified)
2. Facts are pattern-extracted (emails, dates, etc.) with high confidence
3. Negations detect "I don't use X" patterns
4. Semantic memories are LLM-inferred knowledge with lower confidence
5. Every derived memory traces back to source episodes
6. Consolidation strength tracks how well-established a memory is
    """)

    shutdown_workflows()


if __name__ == "__main__":
    asyncio.run(main())
