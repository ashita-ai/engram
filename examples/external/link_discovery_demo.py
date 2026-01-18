#!/usr/bin/env python3
"""Demo script for LLM-driven link discovery.

Tests the link discovery module with real memories and LLM calls.
Requires OPENAI_API_KEY or ANTHROPIC_API_KEY to be set.
"""

import asyncio
import os
import sys

# Add src to path for local development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))


async def main():
    """Run link discovery demo."""
    from engram.linking import (
        LINK_TYPE_DESCRIPTIONS,
        discover_links,
    )

    print("=" * 60)
    print("LLM-Driven Link Discovery Demo")
    print("=" * 60)
    print()

    # Show available link types
    print("Available relationship types:")
    for link_type, description in LINK_TYPE_DESCRIPTIONS.items():
        print(f"  - {link_type}: {description[:50]}...")
    print()

    # Define test memories
    new_memory = {
        "id": "mem_new_001",
        "content": "The user switched from React to Vue.js for their new project because Vue has better TypeScript support and a smaller bundle size.",
    }

    candidate_memories = [
        {
            "id": "mem_old_001",
            "content": "The user prefers React for building web applications and has been using it for 3 years.",
            "keywords": ["react", "web", "frontend"],
            "tags": ["preference", "technology"],
        },
        {
            "id": "mem_old_002",
            "content": "The user started a new e-commerce project in January 2024.",
            "keywords": ["project", "e-commerce"],
            "tags": ["work"],
        },
        {
            "id": "mem_old_003",
            "content": "The user values performance and small bundle sizes in their web applications.",
            "keywords": ["performance", "optimization"],
            "tags": ["preference"],
        },
        {
            "id": "mem_old_004",
            "content": "The user is learning TypeScript to improve code quality.",
            "keywords": ["typescript", "learning"],
            "tags": ["skill"],
        },
    ]

    print("NEW MEMORY:")
    print(f"  {new_memory['content']}")
    print()

    print("CANDIDATE MEMORIES:")
    for i, mem in enumerate(candidate_memories, 1):
        print(f"  {i}. [{mem['id']}] {mem['content'][:60]}...")
    print()

    print("Running LLM link discovery...")
    print("-" * 40)

    try:
        result = await discover_links(
            new_memory_content=new_memory["content"],
            new_memory_id=new_memory["id"],
            candidate_memories=candidate_memories,
            min_confidence=0.5,  # Lower threshold for demo
        )

        print(f"\nDiscovery reasoning: {result.reasoning}")
        print()

        if result.links:
            print(f"DISCOVERED LINKS ({len(result.links)}):")
            for link in result.links:
                print(f"  → {link.target_id}")
                print(f"    Type: {link.link_type}")
                print(f"    Confidence: {link.confidence:.2f}")
                print(f"    Reasoning: {link.reasoning}")
                print(f"    Bidirectional: {link.bidirectional}")
                print()
        else:
            print("No links discovered above confidence threshold.")

        if result.evolutions:
            print(f"SUGGESTED EVOLUTIONS ({len(result.evolutions)}):")
            for evo in result.evolutions:
                print(f"  → {evo.memory_id}")
                print(f"    Field: {evo.field}")
                print(f"    New value: {evo.new_value}")
                print(f"    Reason: {evo.reason}")
                print()
        else:
            print("No evolutions suggested.")

        print("=" * 60)
        print("Demo complete!")

    except Exception as e:
        print(f"ERROR: {e}")
        print()
        print("Make sure you have OPENAI_API_KEY or ANTHROPIC_API_KEY set.")
        raise


if __name__ == "__main__":
    asyncio.run(main())
