#!/usr/bin/env python3
"""REST API client demonstration.

This example shows how to use the Engram REST API with httpx.
First, start the server in another terminal:

    uvicorn engram.api:app --reload

Then run this script:

    python examples/api_client.py

The API provides:
    POST /api/v1/encode  - Store memories with fact extraction
    POST /api/v1/recall  - Semantic similarity search
    GET  /api/v1/health  - Health check
"""

import asyncio

import httpx

BASE_URL = "http://localhost:8000/api/v1"


async def main() -> None:
    """Run the API client demo."""
    print("=" * 60)
    print("Engram REST API Demo")
    print("=" * 60)
    print(f"\nConnecting to {BASE_URL}...")

    async with httpx.AsyncClient(timeout=30.0) as client:
        # =====================================================================
        # Health Check
        # =====================================================================
        print("\n🏥 Checking API health...")
        try:
            resp = await client.get(f"{BASE_URL}/health")
            resp.raise_for_status()
            health = resp.json()
            print(f"  Status: {health['status']}")
            print(f"  Version: {health['version']}")
            print(f"  Storage: {'connected' if health['storage_connected'] else 'disconnected'}")
        except httpx.ConnectError:
            print("\n❌ Could not connect to API server!")
            print("   Start the server with: uvicorn engram.api:app --reload")
            return

        user_id = "api_demo_user"

        # =====================================================================
        # Encode: Store memories
        # =====================================================================
        print("\n📝 Encoding memories via API...")

        memories = [
            {
                "content": "My name is Alex and my email is alex@demo.com",
                "role": "user",
                "user_id": user_id,
            },
            {
                "content": "I have a doctor's appointment on March 15th at 2:30 PM",
                "role": "user",
                "user_id": user_id,
            },
            {
                "content": "Call me at +1-555-987-6543 for urgent matters",
                "role": "user",
                "user_id": user_id,
            },
        ]

        for memory in memories:
            resp = await client.post(f"{BASE_URL}/encode", json=memory)
            resp.raise_for_status()
            result = resp.json()

            print(f"\n  ✓ Encoded: \"{memory['content'][:40]}...\"")
            print(f"    Episode ID: {result['episode']['id']}")
            print(f"    Facts extracted: {result['fact_count']}")
            for fact in result["facts"]:
                print(f"      • {fact['category']}: {fact['content']}")

        # =====================================================================
        # Recall: Search memories
        # =====================================================================
        print("\n🔍 Recalling memories via API...")

        queries = [
            {"query": "contact information", "user_id": user_id, "limit": 5},
            {"query": "upcoming appointments", "user_id": user_id, "limit": 5},
        ]

        for query_req in queries:
            resp = await client.post(f"{BASE_URL}/recall", json=query_req)
            resp.raise_for_status()
            result = resp.json()

            print(f"\n  Query: \"{query_req['query']}\"")
            print(f"  Found: {result['count']} results")
            for r in result["results"]:
                confidence = f" ({r['confidence']:.0%})" if r.get("confidence") else ""
                print(f"    [{r['memory_type']}] {r['content'][:50]}...{confidence}")

    print(f"\n{'=' * 60}")
    print("API demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
