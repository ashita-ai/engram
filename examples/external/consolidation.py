#!/usr/bin/env python3
"""Consolidation workflow demo - LLM-powered semantic extraction.

Demonstrates the full consolidation pipeline:
- Episode storage (ground truth)
- LLM consolidation (semantic memory extraction)
- Memory linking (related_ids for multi-hop)
- Consolidation strength (Testing Effect)
- Memory evolution and updates

Prerequisites:
    - Qdrant running: docker run -p 6333:6333 qdrant/qdrant
    - API key in .env: ENGRAM_OPENAI_API_KEY=sk-...
"""

import asyncio
import logging

from engram.service import EngramService
from engram.workflows import init_workflows, shutdown_workflows
from engram.workflows.consolidation import run_consolidation

# Suppress INFO logs from DBOS to keep output clean
logging.getLogger("dbos").setLevel(logging.WARNING)


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

        # =====================================================================
        # 1. STORE EPISODES (Ground Truth)
        # =====================================================================
        print("\n1. STORING EPISODES (Ground Truth)")
        print("-" * 70)
        print("  Episodes are stored verbatim - immutable ground truth.\n")

        conversations = [
            # Session 1: Introduction
            ("user", "Hi! I'm Morgan, a machine learning engineer."),
            ("assistant", "Nice to meet you, Morgan! What areas of ML do you focus on?"),
            ("user", "I specialize in NLP and transformer architectures."),
            ("user", "My email is morgan.ml@airesearch.io"),
            # Session 2: Technical preferences
            ("user", "I use PyTorch exclusively - tried TensorFlow but prefer PyTorch."),
            ("user", "For experiment tracking, I use Weights & Biases."),
            ("user", "I deploy models using FastAPI and Docker."),
            # Session 3: More context
            ("user", "My team uses Hugging Face for pretrained models."),
            ("user", "We're currently working on a RAG system."),
            ("user", "I don't use Keras anymore - too high-level for research."),
        ]

        negations_detected = 0
        for role, content in conversations:
            result = await engram.encode(content=content, role=role, user_id=user_id)
            extras = []
            if result.facts:
                extras.append(f"facts: {[f.category for f in result.facts]}")
            if result.negations:
                extras.append(f"negations: {len(result.negations)}")
                negations_detected += len(result.negations)
            extras_str = f" ({', '.join(extras)})" if extras else ""
            print(f"  [{role}] {content[:45]}...{extras_str}")

        print(f"\n  Total: {len(conversations)} episodes stored")
        print(f"  Negations detected during encode: {negations_detected}")

        # =====================================================================
        # 2. PRE-CONSOLIDATION STATE
        # =====================================================================
        print("\n\n2. PRE-CONSOLIDATION STATE")
        print("-" * 70)

        # Query before consolidation
        results = await engram.recall(
            query="Morgan's ML expertise",
            user_id=user_id,
            limit=10,
        )

        type_counts: dict[str, int] = {}
        for r in results:
            type_counts[r.memory_type] = type_counts.get(r.memory_type, 0) + 1

        print(f"  Memory types before consolidation: {type_counts}")
        print("  (Semantic memories created by consolidation)")

        # =====================================================================
        # 3. RUN CONSOLIDATION
        # =====================================================================
        print("\n\n3. RUNNING LLM CONSOLIDATION")
        print("-" * 70)
        print("  Consolidation extracts semantic knowledge from episodes.\n")

        result = await run_consolidation(
            storage=engram.storage,
            embedder=engram.embedder,
            user_id=user_id,
        )

        print(f"  Episodes processed: {result.episodes_processed}")
        print(f"  Semantic memories created: {result.semantic_memories_created}")
        print(f"  Links created: {result.links_created}")
        print(f"  Negations created: {result.negations_created}")
        if result.contradictions_found:
            print(f"  Contradictions found: {result.contradictions_found}")

        # =====================================================================
        # 4. POST-CONSOLIDATION STATE
        # =====================================================================
        print("\n\n4. POST-CONSOLIDATION STATE")
        print("-" * 70)

        results = await engram.recall(
            query="Morgan's ML expertise",
            user_id=user_id,
            limit=15,
        )

        type_counts = {}
        for r in results:
            type_counts[r.memory_type] = type_counts.get(r.memory_type, 0) + 1

        print(f"  Memory types after consolidation: {type_counts}")

        # Show semantic memories
        semantic_results = [r for r in results if r.memory_type == "semantic"]
        if semantic_results:
            print("\n  Semantic memories created:")
            for r in semantic_results[:5]:
                print(f"    [{r.confidence:.0%}] {r.content[:55]}...")
                if r.related_ids:
                    print(f"         Links: {len(r.related_ids)} related memories")

        # =====================================================================
        # 5. MEMORY LINKING
        # =====================================================================
        print("\n\n5. MEMORY LINKING (Multi-hop)")
        print("-" * 70)
        print("  Semantic memories link to related memories.")
        print("  Links are created when consolidation finds related existing memories.\n")

        # Find a semantic memory with links
        memories_with_links = [r for r in semantic_results if r.related_ids]

        if memories_with_links:
            r = memories_with_links[0]
            print(f'  Memory: "{r.content[:50]}..."')
            print(f"  Related IDs: {r.related_ids[:3]}")

            # Follow links using the memory's own content as query
            linked_results = await engram.recall(
                query=r.content,
                user_id=user_id,
                follow_links=True,
                max_hops=2,
                limit=10,
            )

            linked = [lr for lr in linked_results if lr.hop_distance and lr.hop_distance > 0]
            print(f"\n  Following links discovered {len(linked)} additional memories:")
            for lr in linked[:3]:
                print(f"    [hop={lr.hop_distance}] {lr.content[:45]}...")
        else:
            print("  No links created yet (first consolidation run).")
            print("  Links build up over multiple consolidation passes.")
            print("  See advanced.py for a more complete multi-hop demo.")

        # =====================================================================
        # 6. CONSOLIDATION STRENGTH (Testing Effect)
        # =====================================================================
        print("\n\n6. CONSOLIDATION STRENGTH (Testing Effect)")
        print("-" * 70)
        print("  Memories strengthen through repeated consolidation.\n")

        # Show consolidation metadata
        for r in semantic_results[:3]:
            passes = r.metadata.get("consolidation_passes", 0)
            selectivity = r.metadata.get("selectivity", 0)
            print(f'  Memory: "{r.content[:40]}..."')
            print(f"    Consolidation passes: {passes}")
            print(f"    Selectivity score: {selectivity:.2f}")
            print()

        print("  Each consolidation pass increases strength.")
        print("  Stronger memories are prioritized in recall.")

        # =====================================================================
        # 7. RUN CONSOLIDATION AGAIN
        # =====================================================================
        print("\n\n7. INCREMENTAL CONSOLIDATION")
        print("-" * 70)
        print("  Add more episodes and consolidate again:\n")

        new_messages = [
            ("user", "I'm also learning Rust for systems programming."),
            ("user", "We might migrate some Python code to Rust for performance."),
        ]

        for role, content in new_messages:
            await engram.encode(content=content, role=role, user_id=user_id)
            print(f"  [{role}] {content}")

        result2 = await run_consolidation(
            storage=engram.storage,
            embedder=engram.embedder,
            user_id=user_id,
        )

        print("\n  Second consolidation:")
        print(f"    Episodes processed: {result2.episodes_processed}")
        print(f"    New semantic memories: {result2.semantic_memories_created}")
        print(f"    New links: {result2.links_created}")
        print(f"    Memories strengthened: {result2.memories_strengthened}")

        # =====================================================================
        # 8. VERIFY DERIVED MEMORIES
        # =====================================================================
        print("\n\n8. VERIFYING DERIVED MEMORIES")
        print("-" * 70)
        print("  Every semantic memory traces back to source episodes:\n")

        # Get a semantic memory
        if semantic_results:
            sem = semantic_results[0]
            print(f'  Memory: "{sem.content[:50]}..."')
            print(f"  ID: {sem.memory_id}")

            # Verify it
            verification = await engram.verify(sem.memory_id, user_id)
            print("\n  Verification:")
            print(f"    Method: {verification.extraction_method}")
            print(f"    Confidence: {verification.confidence:.0%}")
            print(f"    Sources: {len(verification.source_episodes)} episodes")

            if verification.source_episodes:
                print("\n  Source episodes:")
                for src in verification.source_episodes[:2]:
                    print(f"    \"{src['content'][:45]}...\"")

    print("\n" + "=" * 70)
    print("Consolidation demo complete!")
    print("=" * 70)
    print("""
Key Takeaways:
1. Episodes store raw interactions (immutable ground truth)
2. Consolidation uses LLM to extract semantic knowledge
3. Semantic memories link to related memories (multi-hop)
4. Repeated consolidation strengthens memories (Testing Effect)
5. Every derived memory traces back to source episodes
    """)

    # Cleanup durable workflows
    shutdown_workflows()


if __name__ == "__main__":
    asyncio.run(main())
