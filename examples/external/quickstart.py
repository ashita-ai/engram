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
    print("Engram Quickstart Demo")
    print("=" * 70)

    # Initialize durable workflows for consolidation
    try:
        init_workflows()
    except Exception:
        pass  # Workflows may already be initialized

    async with EngramService.create() as engram:
        # Demo identifiers
        user_id = "quickstart_demo"
        org_id = "demo_org"
        session_id = "quickstart_session_001"

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
            # Indirect statements - LLM must INFER, so lower confidence
            ("user", "My team says I'm always the one debugging the tricky regex issues."),
            ("user", "I've been mass liking posts about Rust on Twitter lately."),
        ]

        facts_extracted = []
        negations_extracted = []

        for role, content in messages:
            result = await engram.encode(
                content=content, role=role, user_id=user_id, org_id=org_id, session_id=session_id
            )
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
        # 2. RECALL BEFORE CONSOLIDATION
        # =====================================================================
        print("\n\n2. RECALL BEFORE CONSOLIDATION")
        print("-" * 70)
        print("  Episodic and factual memories are available IMMEDIATELY.\n")

        # Get actual counts from storage to show what's available
        pre_stats = await engram.storage.get_memory_stats(user_id)
        print("  Available now (before LLM consolidation):")
        print(f"    Episodic: {pre_stats.episodes}")
        print(f"    Factual:  {pre_stats.facts}")
        print(f"    Negation: {pre_stats.negation}")
        print(f"    Semantic: {pre_stats.semantic} ← Empty! Requires consolidation")
        print()
        print("  ✓ Episodic (raw) and factual (extracted) available instantly")
        print("  ✗ Semantic memories require consolidation (LLM processing)")

        # =====================================================================
        # 3. RUN CONSOLIDATION
        # =====================================================================
        print("\n\n3. LLM CONSOLIDATION")
        print("-" * 70)
        print("  The LLM reads episodes and extracts semantic knowledge.\n")

        consolidation_result = await run_consolidation(
            storage=engram.storage,
            embedder=engram.embedder,
            user_id=user_id,
            org_id=org_id,
        )
        print(f"  Semantic memories created: {consolidation_result.semantic_memories_created}")
        print(f"  Links created: {consolidation_result.links_created}")

        # Add MORE episodes to enable linking (new memories link to existing)
        print("\n  Adding related episodes to enable linking...")
        more_messages = [
            # These should create memories that link to existing Python/PyTorch memories
            ("user", "I've been using Python for about 5 years now."),
            ("user", "PyTorch is essential for my NLP work at TechFlow."),
        ]
        additional_facts = 0
        for role, content in more_messages:
            result = await engram.encode(
                content=content, role=role, user_id=user_id, org_id=org_id, session_id=session_id
            )
            extras = []
            if result.facts:
                additional_facts += len(result.facts)
                extras.append("+fact")
            extras_str = f"  [{', '.join(extras)}]" if extras else ""
            print(f"    [{role}] {content[:40]}...{extras_str}")
        if additional_facts:
            print(f"    (Extracted {additional_facts} additional fact(s))")

        print("\n  Running second consolidation (links new → existing)...")
        result2 = await run_consolidation(
            storage=engram.storage,
            embedder=engram.embedder,
            user_id=user_id,
            org_id=org_id,
        )
        print(f"  New semantic memories: {result2.semantic_memories_created}")
        print(f"  Links created: {result2.links_created}")

        # Show all memories with first 50 chars
        print("\n  ALL MEMORIES (first 50 chars):")
        all_memories = await engram.recall(query="user", user_id=user_id, limit=20)
        memories_by_type: dict[str, list[str]] = {}
        for m in all_memories:
            if m.memory_type not in memories_by_type:
                memories_by_type[m.memory_type] = []
            memories_by_type[m.memory_type].append(m.content[:50])

        for mem_type in ["episodic", "factual", "negation", "semantic"]:
            if mem_type in memories_by_type:
                print(f"\n    [{mem_type.upper()}]")
                for content in memories_by_type[mem_type][:3]:
                    print(f"      • {content}...")
                if len(memories_by_type[mem_type]) > 3:
                    print(f"      ... and {len(memories_by_type[mem_type]) - 3} more")

        # Show links if any were created
        total_links = consolidation_result.links_created + result2.links_created
        if total_links > 0:
            print(f"\n  LINKED MEMORIES ({total_links} links):")
            semantic_with_links = await engram.recall(
                query="user", user_id=user_id, memory_types=["semantic"], limit=10
            )
            for sem in semantic_with_links:
                if sem.related_ids:
                    print(f"    • {sem.content[:40]}...")
                    print(f"      └─ linked to {len(sem.related_ids)} other memory(s)")

        # =====================================================================
        # 4. RECALL AFTER CONSOLIDATION
        # =====================================================================
        print("\n\n4. RECALL AFTER CONSOLIDATION")
        print("-" * 70)
        print("  Now semantic memories are included in results.\n")

        # Get actual counts from storage after consolidation
        post_stats = await engram.storage.get_memory_stats(user_id)
        print("  Available now (after LLM consolidation):")
        print(f"    Episodic: {post_stats.episodes}")
        print(f"    Factual:  {post_stats.facts}")
        print(f"    Negation: {post_stats.negation}")
        print(f"    Semantic: {post_stats.semantic} ← Created by consolidation!")
        print()
        print("  ✓ Semantic memories now available for recall")

        # =====================================================================
        # 5. MEMORY TYPES WITH CONFIDENCE
        # =====================================================================
        print("\n\n5. MEMORY TYPES WITH CONFIDENCE")
        print("-" * 70)
        print("  Engram separates ground truth from derived knowledge.\n")

        # Get ALL semantic memories to show confidence range
        print("  SEMANTIC MEMORIES (LLM-Inferred):")
        results = await engram.recall(
            query="user",  # Broad query to get all semantic memories
            user_id=user_id,
            memory_types=["semantic"],
            limit=10,
        )
        # Sort by confidence to show range
        results_sorted = sorted(results, key=lambda r: r.confidence or 0, reverse=True)
        for r in results_sorted[:5]:
            print(f"    [{r.confidence:.0%}] {r.content[:55]}...")

        print("\n  Note: LLM is CONSERVATIVE - only extracts certain facts (0.9).")
        print("  Uncertain statements are intentionally NOT extracted to avoid hallucination.")
        print("  Lower confidence (0.6-0.8) would appear for implicit/inferred knowledge.")

        # Negation
        print("\n  NEGATION (what the user does NOT do):")
        results = await engram.recall(
            query="programming", user_id=user_id, memory_types=["negation"], limit=2
        )
        for r in results:
            print(f'    ✗ "{r.content}"')

        # =====================================================================
        # 6. LINKED MEMORIES (Multi-hop)
        # =====================================================================
        print("\n\n6. LINKED MEMORIES")
        print("-" * 70)
        print("  Consolidation creates links between related memories.\n")

        # Check if any semantic memories have links (from consolidation above)
        semantic_mems = await engram.recall(
            query="user", user_id=user_id, memory_types=["semantic"], limit=10
        )
        mems_with_links = [m for m in semantic_mems if m.related_ids]

        if mems_with_links:
            print(f"  {len(mems_with_links)} semantic memories have links:")
            for m in mems_with_links[:3]:
                print(f"    • {m.content[:45]}...")
                print(f"      └─ linked to: {m.related_ids[:2]}")

            # Demonstrate follow_links traversal
            print("\n  Multi-hop traversal with follow_links=True:")
            results_with_hops = await engram.recall(
                query="Jordan", user_id=user_id, follow_links=True, max_hops=2, limit=10
            )
            hopped = [r for r in results_with_hops if r.hop_distance > 0]
            if hopped:
                print(f"    Found {len(hopped)} memories via link traversal:")
                for r in hopped[:2]:
                    print(f"      [hop {r.hop_distance}] {r.content[:45]}...")
            else:
                print("    (No additional memories found via traversal)")
        else:
            print("  No links created in this run.")
            print("  Links are created when the LLM identifies related facts.")
            print("  They typically grow over multiple consolidation passes.")

        # =====================================================================
        # 7. SOURCE VERIFICATION
        # =====================================================================
        print("\n\n7. SOURCE VERIFICATION")
        print("-" * 70)
        print("  Every derived memory traces back to its source episode.\n")

        # Find a semantic memory to verify
        results = await engram.recall(
            query="Jordan", user_id=user_id, memory_types=["semantic"], limit=1
        )

        if results:
            sem = results[0]
            print("  Derived Memory:")
            print(f'    Content: "{sem.content}"')
            print(f"    Type: {sem.memory_type}")
            print(f"    Confidence: {sem.confidence:.0%}")
            print(f"    Source episode IDs: {sem.source_episode_ids[:2]}...")

            # Verify it
            verification = await engram.verify(sem.memory_id, user_id=user_id)
            print("\n  Verification Result:")
            print(f"    Verified: {verification.verified}")
            print(f"    Method: {verification.extraction_method}")

            if verification.source_episodes:
                src = verification.source_episodes[0]
                print("\n  Source Episode (ground truth):")
                print(f"    \"{src['content']}\"")
                print(f"    Role: {src['role']}")

    print("\n" + "=" * 70)
    print("Quickstart complete!")
    print("=" * 70)
    print("""
Key Takeaways:
1. encode() stores episodes AND extracts facts/negations instantly
2. Episodic + factual available BEFORE consolidation
3. Consolidation creates semantic memories via LLM
4. Confidence scores show extraction certainty (0.6-0.9)
5. Links enable multi-hop reasoning across memories
6. Every derived memory traces back to source episodes
""")

    shutdown_workflows()


if __name__ == "__main__":
    asyncio.run(main())
