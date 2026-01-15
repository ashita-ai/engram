#!/usr/bin/env python3
"""StructuredMemory demo - per-episode extraction with optional LLM enrichment.

Demonstrates:
- enrich=False: Default mode, regex extraction only (fast)
- enrich=True: Sync LLM enrichment (blocks until complete)
- enrich="background": Queue for background processing
- StructuredMemory fields (emails, phones, URLs, negations)

Prerequisites:
    - Qdrant running: docker run -p 6333:6333 qdrant/qdrant
    - API key in .env: ENGRAM_OPENAI_API_KEY=sk-...
"""

import asyncio
import logging

from engram.service import EngramService
from engram.storage import EngramStorage
from engram.workflows import init_workflows, shutdown_workflows

# Suppress INFO logs to keep output clean
logging.getLogger("dbos").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


async def cleanup_demo_data(storage: EngramStorage, user_id: str) -> None:
    """Delete all data for the demo user to ensure clean slate."""
    from qdrant_client import models

    collections = ["episodic", "semantic", "structured", "procedural"]
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
    print("Engram StructuredMemory Demo")
    print("=" * 70)

    init_workflows()
    user_id = "demo_structured"

    async with EngramService.create() as engram:
        await cleanup_demo_data(engram.storage, user_id)
        print("  [Previous demo data cleaned up]\n")

        # ================================================================
        # 1. DEFAULT ENCODE (Regex Only - Fast Mode)
        # ================================================================
        print("1. DEFAULT ENCODE (Regex Only - Fast Mode)")
        print("-" * 70)
        print("  Default mode: regex extraction only, no LLM calls.\n")

        fast_messages = [
            "Hi, I'm Jordan Chen, a data engineer at TechFlow Inc.",
            "My email is jordan.chen@techflow.io",
            "I mainly work with Python and PostgreSQL.",
        ]

        for msg in fast_messages:
            result = await engram.encode(
                content=msg,
                role="user",
                user_id=user_id,
            )
            extract_count = (
                len(result.structured.emails)
                + len(result.structured.phones)
                + len(result.structured.urls)
            )
            extract_info = f"  +{extract_count} extract(s)" if extract_count > 0 else ""
            print(f"  [FAST] {msg[:50]}...{extract_info}")

        print("\n  Result: Episodes stored, email extracted via regex.")
        print("  StructuredMemory created in fast mode (no LLM calls).\n")

        # ================================================================
        # 2. ENRICHED ENCODE (Sync LLM Extraction)
        # ================================================================
        print("2. ENRICHED ENCODE (Sync LLM Extraction)")
        print("-" * 70)
        print("  enrich=True: Blocks until LLM completes.\n")

        sync_messages = [
            "I've been using Python for about 8 years now.",
            "I prefer PostgreSQL over MongoDB for most projects.",
            "My manager is Sarah Kim - she handles the data platform team.",
        ]

        for msg in sync_messages:
            result = await engram.encode(
                content=msg,
                role="user",
                user_id=user_id,
                enrich=True,  # Sync LLM extraction
            )
            enriched = "enriched" if result.structured.enriched else "fast"
            print(f"  [{enriched.upper()}] {msg[:50]}...")

        print("\n  Result: StructuredMemory created with LLM enrichment.")
        print("  LLM extracted: people, preferences, negations, etc.\n")

        # ================================================================
        # 3. BACKGROUND ENRICHMENT (Deferred Processing)
        # ================================================================
        print("3. BACKGROUND ENRICHMENT (Deferred Processing)")
        print("-" * 70)
        print("  enrich='background': Returns immediately, processes later.\n")

        deferred_messages = [
            "We have a team meeting every Tuesday at 2pm.",
            "I don't use Windows anymore - switched to macOS last year.",
            "Our next project deadline is March 15th, 2025.",
            "I'm learning Rust on the side for systems programming.",
        ]

        for msg in deferred_messages:
            result = await engram.encode(
                content=msg,
                role="user",
                user_id=user_id,
                enrich="background",  # Queue for later
            )
            print(f"  [QUEUED] {msg[:50]}...")

        print("\n  Episodes stored immediately (no blocking).")
        print("  LLM enrichment will happen in background.\n")

        # ================================================================
        # 4. MEMORY COUNTS
        # ================================================================
        print("4. MEMORY COUNTS")
        print("-" * 70)

        stats = await engram.storage.get_memory_stats(user_id)
        print(f"  Episodes:   {stats.episodes}")
        print(f"  Structured: {stats.structured}")
        print(f"  Semantic:   {stats.semantic}")
        print(f"  Procedural: {stats.procedural}")

        # ================================================================
        # 5. STRUCTURED MEMORY DETAILS
        # ================================================================
        print("\n5. STRUCTURED MEMORY DETAILS")
        print("-" * 70)
        print("  StructuredMemory contains per-episode extracts:\n")

        # Get structured memories
        structured_results = await engram.recall(
            query="PostgreSQL preference",
            user_id=user_id,
            memory_types=["structured"],
            limit=3,
        )

        if structured_results:
            for struct in structured_results[:3]:
                print(f"  Content: {struct.content[:60]}...")
                print(f"  Mode: {struct.metadata.get('mode', 'unknown')}")
                print(f"  Enriched: {struct.metadata.get('enriched', False)}")
                print(f"  Confidence: {struct.confidence:.0%}")
                if struct.metadata.get("emails"):
                    print(f"  Emails: {struct.metadata['emails']}")
                print()

        # ================================================================
        # 6. NEGATION DETECTION
        # ================================================================
        print("6. NEGATION DETECTION")
        print("-" * 70)
        print("  StructuredMemory captures negations for filtering.\n")

        # Encode a message with explicit negation
        result = await engram.encode(
            content="I don't use MongoDB anymore - switched to PostgreSQL last year.",
            role="user",
            user_id=user_id,
            enrich=True,
        )

        print(f"  Input: '{result.episode.content}'")
        print(f"  Negations detected: {len(result.structured.negations)}")
        for neg in result.structured.negations:
            print(f"    - Pattern: '{neg.pattern}'")
            print(f"      Content: '{neg.content}'")
            print(f"      Context: '{neg.context}'")

        # ================================================================
        # 7. NEGATION FILTERING IN RECALL
        # ================================================================
        print("\n7. NEGATION FILTERING IN RECALL")
        print("-" * 70)
        print("  Negations filter out contradicted information.\n")

        # Search for MongoDB-related content
        mongodb_results = await engram.recall(
            query="MongoDB database",
            user_id=user_id,
            memory_types=["episodic"],
            apply_negation_filter=False,
            limit=5,
        )

        mongodb_filtered = await engram.recall(
            query="MongoDB database",
            user_id=user_id,
            memory_types=["episodic"],
            apply_negation_filter=True,
            limit=5,
        )

        print("  Query: 'MongoDB database'")
        print(f"  Without negation filter: {len(mongodb_results)} results")
        print(f"  With negation filter:    {len(mongodb_filtered)} results")

        if len(mongodb_results) > len(mongodb_filtered):
            print("\n  Filtered out content matching 'I don't use MongoDB'")

    shutdown_workflows()

    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)
    print(
        """
Key Concepts:
1. enrich=False (default): Fast regex-only extraction
2. enrich=True: Sync LLM extraction (blocks until complete)
3. enrich="background": Queue for background processing
4. StructuredMemory: Per-episode extracts (emails, phones, URLs, negations)
5. Negations: Track what is NOT true, filter contradicted info
6. Hierarchy: Episode -> Structured -> Semantic -> Procedural
"""
    )


if __name__ == "__main__":
    asyncio.run(main())
