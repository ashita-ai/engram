#!/usr/bin/env python3
"""Multi-tenancy demonstration.

This example shows how Engram isolates data between:
1. Different users (user_id)
2. Different organizations (org_id)

Each user only sees their own memories, even when sharing
the same Qdrant instance.

Prerequisites:
    - Qdrant running locally (docker run -p 6333:6333 qdrant/qdrant)
    - OpenAI API key OR ENGRAM_EMBEDDING_PROVIDER=fastembed

Usage:
    python examples/multi_tenant.py
"""

import asyncio

from engram.service import EngramService


async def main() -> None:
    """Run the multi-tenancy demo."""
    print("=" * 60)
    print("Engram Multi-Tenancy Demo")
    print("=" * 60)

    async with EngramService.create() as engram:
        # =====================================================================
        # Setup: Create memories for different users
        # =====================================================================
        print("\n📝 Creating memories for different users...")

        # Alice's memories
        await engram.encode(
            content="My password hint is 'favorite pet name'",
            role="user",
            user_id="alice",
        )
        await engram.encode(
            content="My email is alice@personal.com",
            role="user",
            user_id="alice",
        )

        # Bob's memories
        await engram.encode(
            content="My password hint is 'first car model'",
            role="user",
            user_id="bob",
        )
        await engram.encode(
            content="My email is bob@personal.com",
            role="user",
            user_id="bob",
        )

        print("  ✓ Created 2 memories for Alice")
        print("  ✓ Created 2 memories for Bob")

        # =====================================================================
        # User Isolation: Each user only sees their own data
        # =====================================================================
        print("\n🔒 Testing user isolation...")

        # Alice's search
        alice_results = await engram.recall(
            query="password hint",
            user_id="alice",
            limit=5,
        )
        print("\n  Alice searches for 'password hint':")
        for r in alice_results:
            print(f"    → {r.content[:50]}...")

        # Bob's search
        bob_results = await engram.recall(
            query="password hint",
            user_id="bob",
            limit=5,
        )
        print("\n  Bob searches for 'password hint':")
        for r in bob_results:
            print(f"    → {r.content[:50]}...")

        # Verify isolation
        alice_contents = [r.content for r in alice_results]
        bob_contents = [r.content for r in bob_results]

        assert "bob" not in str(alice_contents).lower(), "Alice should not see Bob's data!"
        assert "alice" not in str(bob_contents).lower(), "Bob should not see Alice's data!"
        print("\n  ✓ User isolation verified!")

        # =====================================================================
        # Organization Isolation
        # =====================================================================
        print("\n🏢 Testing organization isolation...")

        # TechCorp employee memories
        await engram.encode(
            content="Q4 revenue target: $10M",
            role="user",
            user_id="charlie",
            org_id="techcorp",
        )

        # StartupInc employee memories
        await engram.encode(
            content="Q4 revenue target: $500K",
            role="user",
            user_id="charlie",  # Same user, different org!
            org_id="startupinc",
        )

        print("  ✓ Charlie has memories in both TechCorp and StartupInc")

        # Charlie at TechCorp
        techcorp_results = await engram.recall(
            query="revenue target",
            user_id="charlie",
            org_id="techcorp",
            limit=5,
        )
        print("\n  Charlie@TechCorp searches for 'revenue':")
        for r in techcorp_results:
            print(f"    → {r.content}")

        # Charlie at StartupInc
        startup_results = await engram.recall(
            query="revenue target",
            user_id="charlie",
            org_id="startupinc",
            limit=5,
        )
        print("\n  Charlie@StartupInc searches for 'revenue':")
        for r in startup_results:
            print(f"    → {r.content}")

        # Verify org isolation
        techcorp_content = str([r.content for r in techcorp_results])
        startup_content = str([r.content for r in startup_results])

        assert "$10M" in techcorp_content, "TechCorp should see $10M"
        assert "$500K" in startup_content, "StartupInc should see $500K"
        assert "$500K" not in techcorp_content, "TechCorp should NOT see StartupInc data"
        assert "$10M" not in startup_content, "StartupInc should NOT see TechCorp data"
        print("\n  ✓ Organization isolation verified!")

    print(f"\n{'=' * 60}")
    print("Multi-tenancy demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
