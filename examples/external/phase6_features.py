#!/usr/bin/env python3
"""End-to-end demonstration of Phase 6 features.

This example demonstrates:
1. Query Expansion - LLM-powered query enhancement for better recall
2. Diversity Sampling - MMR reranking for varied results
3. Contradiction Detection - Finding conflicts between memories
4. Memory Update - Correcting/updating existing memories

Requirements:
- Qdrant running: docker run -p 6333:6333 qdrant/qdrant
- OpenAI API key set: export OPENAI_API_KEY=your-key
- Install: uv sync --extra dev

Usage:
    uv run python examples/external/phase6_features.py
"""

from __future__ import annotations

import asyncio
import os
import sys

import httpx

BASE_URL = "http://localhost:8000/api/v1"
USER_ID = "phase6_demo_user"


def check_server() -> bool:
    """Check if server is running."""
    try:
        response = httpx.get(f"{BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except httpx.ConnectError:
        return False


async def cleanup_user(client: httpx.AsyncClient) -> None:
    """Delete all memories for the demo user."""
    response = await client.delete(f"{BASE_URL}/users/{USER_ID}/memories")
    if response.status_code == 200:
        data = response.json()
        print(f"Cleaned up {data.get('total_deleted', 0)} existing memories")


async def encode_memories(client: httpx.AsyncClient) -> list[str]:
    """Encode test memories for demonstration."""
    memories = [
        # Related memories about preferences
        ("I love drinking coffee every morning", "user"),
        ("My favorite coffee is Ethiopian single-origin", "user"),
        ("I usually have two cups of coffee before noon", "user"),
        # Contradictory memory
        ("I hate coffee, it makes me jittery", "user"),
        # Different topic - work
        ("I work as a software engineer at a startup", "user"),
        ("My team uses Python and TypeScript", "user"),
        # Another different topic - hobbies
        ("I enjoy hiking on weekends", "user"),
        ("Photography is my main hobby", "user"),
    ]

    episode_ids = []
    print("\n=== ENCODING MEMORIES ===")
    for content, role in memories:
        response = await client.post(
            f"{BASE_URL}/encode",
            json={
                "content": content,
                "role": role,
                "user_id": USER_ID,
                "enrich": True,  # Enable LLM enrichment
            },
        )
        if response.status_code == 201:
            data = response.json()
            episode_ids.append(data["episode"]["id"])
            print(f"  Encoded: {content[:50]}...")
        else:
            print(f"  ERROR encoding: {response.text}")

    return episode_ids


async def demo_query_expansion(client: httpx.AsyncClient) -> None:
    """Demonstrate query expansion feature."""
    print("\n" + "=" * 60)
    print("QUERY EXPANSION DEMO")
    print("=" * 60)
    print("\nQuery expansion uses LLM to add related terms to your query,")
    print("improving recall for semantically related memories.\n")

    query = "caffeine habits"

    # Without expansion
    print(f"Query: '{query}'")
    print("\n--- Without Query Expansion ---")
    response = await client.post(
        f"{BASE_URL}/recall",
        json={
            "query": query,
            "user_id": USER_ID,
            "limit": 5,
            "expand_query": False,
        },
    )
    if response.status_code == 200:
        data = response.json()
        print(f"Results: {data['count']}")
        for r in data["results"]:
            print(f"  [{r['score']:.3f}] {r['content'][:60]}...")
    else:
        print(f"ERROR: {response.text}")

    # With expansion
    print("\n--- With Query Expansion ---")
    print("(LLM expands 'caffeine habits' to include 'coffee', 'morning', etc.)")
    response = await client.post(
        f"{BASE_URL}/recall",
        json={
            "query": query,
            "user_id": USER_ID,
            "limit": 5,
            "expand_query": True,
        },
    )
    if response.status_code == 200:
        data = response.json()
        print(f"Results: {data['count']}")
        for r in data["results"]:
            print(f"  [{r['score']:.3f}] {r['content'][:60]}...")
    else:
        print(f"ERROR: {response.text}")


async def demo_diversity_sampling(client: httpx.AsyncClient) -> None:
    """Demonstrate diversity sampling feature."""
    print("\n" + "=" * 60)
    print("DIVERSITY SAMPLING DEMO")
    print("=" * 60)
    print("\nDiversity sampling uses MMR (Maximal Marginal Relevance) to")
    print("return varied results instead of similar/redundant ones.\n")

    query = "what do I like"

    # Without diversity (may return similar coffee memories)
    print(f"Query: '{query}'")
    print("\n--- Without Diversity (diversity=0.0) ---")
    response = await client.post(
        f"{BASE_URL}/recall",
        json={
            "query": query,
            "user_id": USER_ID,
            "limit": 4,
            "diversity": 0.0,
        },
    )
    if response.status_code == 200:
        data = response.json()
        print(f"Results: {data['count']}")
        for r in data["results"]:
            print(f"  [{r['score']:.3f}] {r['content'][:60]}...")
    else:
        print(f"ERROR: {response.text}")

    # With diversity (should spread across topics)
    print("\n--- With Diversity (diversity=0.5) ---")
    print("(Higher diversity = more varied results from different topics)")
    response = await client.post(
        f"{BASE_URL}/recall",
        json={
            "query": query,
            "user_id": USER_ID,
            "limit": 4,
            "diversity": 0.5,
        },
    )
    if response.status_code == 200:
        data = response.json()
        print(f"Results: {data['count']}")
        for r in data["results"]:
            print(f"  [{r['score']:.3f}] {r['content'][:60]}...")
    else:
        print(f"ERROR: {response.text}")


async def demo_contradiction_detection(client: httpx.AsyncClient) -> None:
    """Demonstrate contradiction detection feature."""
    print("\n" + "=" * 60)
    print("CONTRADICTION DETECTION DEMO")
    print("=" * 60)
    print("\nContradiction detection finds conflicting memories using")
    print("semantic similarity + LLM analysis.\n")

    # First, trigger consolidation to create semantic memories
    print("Triggering consolidation to create semantic memories...")
    response = await client.post(
        f"{BASE_URL}/workflows/consolidate",
        json={"user_id": USER_ID},
    )
    if response.status_code == 200:
        data = response.json()
        print(f"  Created {data.get('semantic_memories_created', 0)} semantic memories")
    else:
        print(f"  Consolidation response: {response.status_code}")

    # Detect conflicts
    print("\nDetecting contradictions in semantic memories...")
    response = await client.post(
        f"{BASE_URL}/conflicts/detect",
        json={
            "user_id": USER_ID,
            "memory_type": "semantic",
            "similarity_threshold": 0.3,  # Lower threshold to find more potential conflicts
        },
    )
    if response.status_code == 200:
        data = response.json()
        print(f"\nConflicts found: {data['conflicts_found']}")
        for conflict in data["conflicts"]:
            print(f"\n  Conflict ID: {conflict['id']}")
            print(f"  Type: {conflict['conflict_type']}")
            print(f"  Confidence: {conflict['confidence']:.2f}")
            print(f"  Memory A: {conflict['memory_a_content'][:50]}...")
            print(f"  Memory B: {conflict['memory_b_content'][:50]}...")
            print(f"  Explanation: {conflict['explanation']}")
    else:
        print(f"ERROR: {response.text}")

    # List all conflicts
    print("\n--- Listing All Conflicts ---")
    response = await client.get(
        f"{BASE_URL}/conflicts",
        params={"user_id": USER_ID, "include_resolved": False},
    )
    if response.status_code == 200:
        data = response.json()
        print(f"Total unresolved conflicts: {data['count']}")

        # Resolve a conflict if found
        if data["conflicts"]:
            conflict_id = data["conflicts"][0]["id"]
            print(f"\n--- Resolving Conflict {conflict_id} ---")
            response = await client.post(
                f"{BASE_URL}/conflicts/{conflict_id}/resolve",
                json={"resolution": "newer_wins"},
            )
            if response.status_code == 200:
                resolved = response.json()
                print(f"  Resolution: {resolved['resolution']}")
                print(f"  Resolved at: {resolved['resolved_at']}")
            else:
                print(f"  ERROR resolving: {response.text}")
    else:
        print(f"ERROR: {response.text}")


async def demo_memory_update(client: httpx.AsyncClient) -> None:
    """Demonstrate memory update feature."""
    print("\n" + "=" * 60)
    print("MEMORY UPDATE DEMO")
    print("=" * 60)
    print("\nMemory update allows correcting/updating existing memories")
    print("(except episodic - those are immutable ground truth).\n")

    # First, get a semantic memory to update
    response = await client.post(
        f"{BASE_URL}/recall",
        json={
            "query": "coffee",
            "user_id": USER_ID,
            "limit": 5,
            "memory_types": ["semantic"],
        },
    )

    semantic_id = None
    if response.status_code == 200:
        data = response.json()
        for r in data["results"]:
            if r["memory_type"] == "semantic":
                semantic_id = r["memory_id"]
                print(f"Found semantic memory: {semantic_id}")
                print(f"  Current content: {r['content'][:80]}...")
                print(f"  Current confidence: {r.get('confidence', 'N/A')}")
                break

    if semantic_id:
        # Update the memory
        print(f"\n--- Updating Memory {semantic_id} ---")
        response = await client.patch(
            f"{BASE_URL}/memories/{semantic_id}",
            json={
                "user_id": USER_ID,
                "confidence": 0.95,
                "tags": ["verified", "coffee-preference"],
            },
        )
        if response.status_code == 200:
            data = response.json()
            print(f"  Updated: {data['updated']}")
            print(f"  Re-embedded: {data['re_embedded']}")
            print("  Changes:")
            for change in data["changes"]:
                print(f"    {change['field']}: {change['old_value']} -> {change['new_value']}")
        else:
            print(f"  ERROR: {response.text}")

        # Try to update episodic (should fail)
        print("\n--- Attempting to Update Episodic Memory (should fail) ---")
        response = await client.patch(
            f"{BASE_URL}/memories/ep_test123",
            json={
                "user_id": USER_ID,
                "content": "modified content",
            },
        )
        if response.status_code == 400:
            print(f"  Correctly rejected: {response.json()['detail'][:60]}...")
        else:
            print(f"  Unexpected response: {response.status_code}")
    else:
        print("No semantic memories found to update. Run consolidation first.")


async def main() -> None:
    """Run all Phase 6 feature demos."""
    print("=" * 60)
    print("PHASE 6 FEATURES - END-TO-END DEMONSTRATION")
    print("=" * 60)

    # Check prerequisites
    if not os.getenv("OPENAI_API_KEY"):
        print("\nERROR: OPENAI_API_KEY not set")
        print("Export your API key: export OPENAI_API_KEY=your-key")
        sys.exit(1)

    if not check_server():
        print("\nERROR: Server not running")
        print("Start the server: uv run uvicorn engram.api.app:app --reload")
        sys.exit(1)

    print("\nServer is running. Starting demos...\n")

    async with httpx.AsyncClient(timeout=60.0) as client:
        # Clean up any existing data
        await cleanup_user(client)

        # Encode test memories
        await encode_memories(client)

        # Run feature demos
        await demo_query_expansion(client)
        await demo_diversity_sampling(client)
        await demo_contradiction_detection(client)
        await demo_memory_update(client)

        # Final cleanup
        print("\n" + "=" * 60)
        print("DEMO COMPLETE")
        print("=" * 60)
        await cleanup_user(client)


if __name__ == "__main__":
    asyncio.run(main())
