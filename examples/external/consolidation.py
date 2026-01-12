#!/usr/bin/env python3
"""Consolidation workflow demo - Hierarchical memory compression.

Demonstrates the full consolidation pipeline:
- Episode storage (ground truth)
- LLM consolidation (N episodes → 1 semantic summary)
- Procedural synthesis (all semantics → 1 behavioral profile)
- Bidirectional traceability (episodes ↔ semantics ↔ procedural)

Prerequisites:
    - Qdrant running: docker run -p 6333:6333 qdrant/qdrant
    - API key in .env: ENGRAM_OPENAI_API_KEY=sk-...
"""

import asyncio
import logging

from engram.service import EngramService
from engram.storage import EngramStorage
from engram.workflows import init_workflows, shutdown_workflows

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
        "procedural": stats.procedural,
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
    print("Engram Hierarchical Consolidation Demo")
    print("=" * 70)

    # Initialize durable workflows (DBOS or Temporal based on config)
    try:
        init_workflows()
        print("  [Durable workflows initialized]")
    except Exception as e:
        print(f"  [Durable workflows not available: {e}]")

    async with EngramService.create() as engram:
        # Demo identifiers
        user_id = "consolidation_demo"
        org_id = "demo_org"
        session_id = "consolidation_session_001"

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

        episode_ids: list[str] = []
        for role, content in conversations:
            result = await engram.encode(
                content=content, role=role, user_id=user_id, org_id=org_id, session_id=session_id
            )
            episode_ids.append(result.episode.id)
            extras = []
            if result.facts:
                extras.append(f"+fact:{result.facts[0].category}")
            if result.negations:
                extras.append("+negation")
            extras_str = f"  {' '.join(extras)}" if extras else ""
            print(f"  [{role:9}] {content[:50]}...{extras_str}")

        # =====================================================================
        # 2. BEFORE CONSOLIDATION
        # =====================================================================
        print("\n\n2. BEFORE CONSOLIDATION")
        print("-" * 70)
        print("  Episodic + factual/negation available immediately.\n")

        counts = await get_memory_counts(engram.storage, user_id)
        print(f"  Episodic (ground truth): {counts['episodic']} episodes")
        print(f"    └─ Factual (extracted): {counts['factual']} (email pattern)")
        print(f"    └─ Negation (extracted): {counts['negation']} (Keras pattern)")
        print(f"    └─ Semantic (LLM):       {counts['semantic']} ← Empty!")
        print(f"    └─ Procedural (LLM):     {counts['procedural']} ← Empty!")

        # =====================================================================
        # 3. RUN LLM CONSOLIDATION (N episodes → 1 summary)
        # =====================================================================
        print("\n\n3. LLM CONSOLIDATION: N Episodes → 1 Summary")
        print("-" * 70)
        print("  Hierarchical compression: summarizes episodes into semantic memory.\n")

        result = await engram.consolidate(user_id=user_id, org_id=org_id)

        print(f"  Episodes processed:        {result.episodes_processed}")
        print(f"  Semantic memories created: {result.semantic_memories_created}")
        print(f"  Compression ratio:         {result.compression_ratio:.1f}:1")
        print(f"  Links created:             {result.links_created}")

        # =====================================================================
        # 4. SOURCE EPISODE LINKAGE
        # =====================================================================
        print("\n\n4. SOURCE EPISODE LINKAGE (Bidirectional)")
        print("-" * 70)
        print("  Every semantic memory links to its source episodes.\n")

        # Get semantic memories directly from storage (more reliable than recall)
        semantic_memories = await engram.storage.list_semantic_memories(user_id)

        if semantic_memories:
            sem = semantic_memories[0]  # Most recent
            print(f"  Semantic Memory: {sem.id}")
            print(f'    Content: "{sem.content[:60]}..."')
            print(f"    source_episode_ids: {sem.source_episode_ids[:3]}...")
            print(f"    ({len(sem.source_episode_ids)} episodes → 1 summary)")

            # Verify one of the source episodes
            print("\n  Verification (episode → semantic):")
            ep = await engram.storage.get_episode(sem.source_episode_ids[0], user_id)
            if ep:
                print(f"    Episode {ep.id[:20]}...")
                print(f"      summarized: {ep.summarized}")
                if ep.summarized_into:
                    print(f"      summarized_into: {ep.summarized_into[:20]}...")
                    print(f"      ✓ Links match: {ep.summarized_into == sem.id}")
                else:
                    print("      summarized_into: None")

        # =====================================================================
        # 5. RAW vs DERIVED
        # =====================================================================
        print("\n\n5. RAW EPISODES vs DERIVED SUMMARY")
        print("-" * 70)

        print("  RAW EPISODES (ground truth):")
        for i, (role, content) in enumerate(conversations[:5], 1):
            print(f'    {i}. [{role}] "{content[:50]}..."')
        print(f"    ... and {len(conversations) - 5} more\n")

        print("  DERIVED SEMANTIC SUMMARY:")
        for sem in semantic_memories:
            print(f'    "{sem.content}"')
            print(f"    (confidence: {sem.confidence.value:.0%})")

        # =====================================================================
        # 6. ADD MORE DATA AND CONSOLIDATE AGAIN
        # =====================================================================
        print("\n\n6. ADD MORE DATA & CONSOLIDATE AGAIN")
        print("-" * 70)

        new_messages = [
            ("user", "By the way, my full name is Morgan Chen."),
            ("user", "I'm also learning Rust for systems programming."),
        ]

        print("  Adding new episodes:")
        for role, content in new_messages:
            await engram.encode(
                content=content, role=role, user_id=user_id, org_id=org_id, session_id=session_id
            )
            print(f"    [{role}] {content}")

        result2 = await engram.consolidate(user_id=user_id, org_id=org_id)

        print("\n  Second consolidation:")
        print(f"    New episodes processed:    {result2.episodes_processed}")
        print(f"    New semantic memories:     {result2.semantic_memories_created}")
        print(f"    Compression ratio:         {result2.compression_ratio:.1f}:1")

        # =====================================================================
        # 7. PROCEDURAL SYNTHESIS (all semantics → 1 behavioral profile)
        # =====================================================================
        print("\n\n7. PROCEDURAL SYNTHESIS: All Semantics → 1 Behavioral Profile")
        print("-" * 70)
        print("  Creates ONE procedural memory per user from all semantics.\n")

        synthesis_result = await engram.create_procedural(user_id=user_id, org_id=org_id)

        print(f"  Semantics analyzed:    {synthesis_result.semantics_analyzed}")
        print(f"  Procedural created:    {synthesis_result.procedural_created}")
        print(f"  Procedural ID:         {synthesis_result.procedural_id}")

        # Show the procedural memory (fetch directly for full details)
        proc_memories = await engram.storage.list_procedural_memories(user_id)

        if proc_memories:
            proc = proc_memories[0]
            print("\n  BEHAVIORAL PROFILE:")
            # Split into lines for readability
            lines = proc.content.split("\n")
            for line in lines[:8]:
                if line.strip():
                    print(f"    {line.strip()}")
            print(f"\n    source_semantic_ids: {proc.source_semantic_ids}")
            print(f"    ({len(proc.source_semantic_ids)} semantics → 1 procedural)")

        # =====================================================================
        # 8. FINAL MEMORY STATE
        # =====================================================================
        print("\n\n8. FINAL MEMORY STATE")
        print("-" * 70)

        final_counts = await get_memory_counts(engram.storage, user_id)
        print(f"  Episodic (ground truth): {final_counts['episodic']} episodes")
        print(f"    └─ Factual (extracted): {final_counts['factual']}")
        print(f"    └─ Negation (extracted): {final_counts['negation']}")
        print(f"    └─ Semantic (LLM):       {final_counts['semantic']} summaries")
        print(f"    └─ Procedural (LLM):     {final_counts['procedural']} behavioral profile")

        print("\n  HIERARCHICAL COMPRESSION:")
        print(
            f"    {final_counts['episodic']} episodes → {final_counts['semantic']} semantic summaries → {final_counts['procedural']} procedural"
        )

    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)
    print("""
Key Concepts:
1. Episodes are RAW ground truth (immutable, never modified)
2. Facts/negations are pattern-extracted with high confidence
3. Consolidation compresses N episodes → 1 semantic summary
4. Procedural synthesis compresses all semantics → 1 behavioral profile
5. Bidirectional links: episode.summarized_into ↔ semantic.source_episode_ids
6. Each layer has source traceability back to ground truth
    """)

    shutdown_workflows()


if __name__ == "__main__":
    asyncio.run(main())
