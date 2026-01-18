#!/usr/bin/env python3
"""Demo script for Phase 7 advanced memory features.

Demonstrates:
1. Temporal Reasoning - Detecting state changes from text
2. Entity Resolution - Extracting and clustering entities
3. Confidence Propagation - PageRank-style confidence spreading

Requires OPENAI_API_KEY or ANTHROPIC_API_KEY for LLM features.
"""

import asyncio
import os
import sys

# Add src to path for local development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))


def print_section(title: str) -> None:
    """Print a section header."""
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


async def demo_temporal_reasoning() -> None:
    """Demonstrate temporal state change detection."""
    from engram.temporal import (
        detect_state_changes,
        detect_state_changes_regex,
    )

    print_section("1. TEMPORAL REASONING")
    print("Detecting state changes from natural language...")
    print()

    test_cases = [
        "I no longer use MongoDB for my projects.",
        "I switched from React to Vue.js last month.",
        "I've started using TypeScript for all new code.",
        "I used to work with Java but now I prefer Python.",
        "I upgraded to PostgreSQL 16 yesterday.",
        "I'm back to using Vim after trying VS Code for a year.",
    ]

    for text in test_cases:
        print(f'INPUT: "{text}"')
        changes = detect_state_changes_regex(
            text=text,
            memory_id="demo_mem",
            user_id="demo_user",
        )

        if changes:
            for change in changes:
                print(f"  → Entity: {change.entity}")
                print(f"    Type: {change.change_type}")
                print(f"    Previous: {change.previous_state}")
                print(f"    Current: {change.current_state}")
                print(f"    Confidence: {change.confidence:.2f}")
        else:
            print("  → No state changes detected")
        print()

    # Test async version with LLM
    print("-" * 40)
    print("Testing async detection (regex only, no LLM)...")
    complex_text = (
        "After 5 years with MongoDB, I finally migrated to PostgreSQL and couldn't be happier."
    )
    changes = await detect_state_changes(
        text=complex_text,
        memory_id="demo_mem_async",
        user_id="demo_user",
        use_llm=False,  # Set to True to test LLM
    )
    print(f'INPUT: "{complex_text}"')
    print(f"Detected {len(changes)} state change(s)")
    for change in changes:
        print(f"  → {change.entity}: {change.change_type}")


async def demo_entity_resolution() -> None:
    """Demonstrate entity extraction and resolution."""
    from engram.entities import Entity, EntityMention

    print_section("2. ENTITY RESOLUTION")
    print("Demonstrating entity models and alias management...")
    print()

    # Create an entity
    john = Entity(
        canonical_name="John Smith",
        entity_type="person",
        user_id="demo_user",
    )

    print(f"Created entity: {john.canonical_name}")
    print(f"  Type: {john.entity_type}")
    print(f"  ID: {john.id}")
    print()

    # Add aliases
    aliases = ["John", "J. Smith", "Johnny", "Mr. Smith"]
    for alias in aliases:
        john.add_alias(alias)
    print(f"Added aliases: {john.aliases}")

    # Test matching
    test_names = ["John Smith", "john", "J. SMITH", "Jane Doe", "Johnny"]
    print()
    print("Testing name matching:")
    for name in test_names:
        matches = john.matches_name(name)
        print(f"  '{name}' → {'✓ matches' if matches else '✗ no match'}")

    # Add memory references
    john.add_memory("mem_001")
    john.add_memory("mem_002")
    john.add_memory("mem_001")  # Duplicate, should be ignored
    print()
    print(f"Memory references: {john.memory_ids}")

    # Demonstrate merging
    print()
    print("-" * 40)
    print("Demonstrating entity merging...")

    # Create another entity that's the same person
    john_alt = Entity(
        canonical_name="Johnny Smith",
        entity_type="person",
        user_id="demo_user",
        aliases=["JS"],
        memory_ids=["mem_003", "mem_004"],
        attributes={"department": "Engineering"},
    )

    print(f"Merging '{john_alt.canonical_name}' into '{john.canonical_name}'")
    john.merge_from(john_alt)

    print("After merge:")
    print(f"  Canonical: {john.canonical_name}")
    print(f"  Aliases: {john.aliases}")
    print(f"  Memories: {john.memory_ids}")
    print(f"  Attributes: {john.attributes}")
    print(f"  Merge count: {john.merge_count}")

    # Demonstrate entity mentions
    print()
    print("-" * 40)
    print("Creating entity mentions...")

    mentions = [
        EntityMention(
            text="John",
            entity_type="person",
            memory_id="mem_005",
            context="John mentioned he likes Python",
            confidence=0.9,
        ),
        EntityMention(
            text="Acme Corp",
            entity_type="organization",
            memory_id="mem_005",
            context="works at Acme Corp",
            confidence=0.85,
        ),
    ]

    for mention in mentions:
        print(f"  Mention: '{mention.text}' ({mention.entity_type})")
        print(f"    Context: {mention.context}")
        print(f"    Confidence: {mention.confidence}")


async def demo_confidence_propagation() -> None:
    """Demonstrate PageRank-style confidence propagation."""
    from engram.models import ConfidenceScore, ExtractionMethod, SemanticMemory
    from engram.propagation import (
        PropagationConfig,
        compute_link_strength,
        propagate_confidence,
        propagate_distrust,
    )

    print_section("3. CONFIDENCE PROPAGATION")
    print("Demonstrating PageRank-style confidence spreading...")
    print()

    def make_memory(
        id: str,
        content: str,
        confidence: float,
        extraction_method: ExtractionMethod = ExtractionMethod.INFERRED,
        related_ids: list[str] | None = None,
        link_types: dict[str, str] | None = None,
    ) -> SemanticMemory:
        return SemanticMemory(
            id=id,
            content=content,
            source_episode_ids=["ep_demo"],
            user_id="demo_user",
            embedding=[0.1] * 4,
            confidence=ConfidenceScore(
                value=confidence,
                extraction_method=extraction_method,
                extraction_base=confidence,
            ),
            related_ids=related_ids or [],
            link_types=link_types or {},
        )

    # Create a network of memories
    print("Creating memory network:")
    memories = [
        make_memory(
            "m1",
            "User prefers Python (verbatim from conversation)",
            confidence=0.95,
            extraction_method=ExtractionMethod.VERBATIM,
            related_ids=["m2", "m3"],
            link_types={"m2": "elaborates", "m3": "related"},
        ),
        make_memory(
            "m2",
            "User uses Python for data science",
            confidence=0.7,
            extraction_method=ExtractionMethod.INFERRED,
            related_ids=["m1"],
            link_types={"m1": "related"},
        ),
        make_memory(
            "m3",
            "User knows pandas and numpy",
            confidence=0.6,
            extraction_method=ExtractionMethod.INFERRED,
            related_ids=["m1", "m2"],
            link_types={"m1": "related", "m2": "related"},
        ),
    ]

    for m in memories:
        method = m.confidence.extraction_method.value
        print(f"  [{m.id}] {m.content[:40]}...")
        print(f"       Confidence: {m.confidence.value:.2f} ({method})")
        print(f"       Links to: {m.related_ids}")
    print()

    # Compute link strengths
    print("-" * 40)
    print("Computing link strengths:")
    for source in memories:
        for target_id in source.related_ids:
            target = next((m for m in memories if m.id == target_id), None)
            if target:
                strength = compute_link_strength(source, target)
                link_type = source.link_types.get(target_id, "related")
                print(f"  {source.id} → {target_id}: {strength:.2f} ({link_type})")
    print()

    # Run propagation
    print("-" * 40)
    print("Running confidence propagation...")
    config = PropagationConfig(
        damping_factor=0.85,
        max_iterations=5,
        convergence_threshold=0.001,
    )

    print(f"Config: damping={config.damping_factor}, max_iter={config.max_iterations}")
    print()

    before = {m.id: m.confidence.value for m in memories}
    result = await propagate_confidence(memories, config)

    print("Result:")
    print(f"  Iterations: {result.iterations}")
    print(f"  Converged: {result.converged}")
    print(f"  Memories updated: {result.memories_updated}")
    print(f"  Total boost: {result.total_boost_applied:.4f}")
    print()

    print("Confidence changes:")
    for m in memories:
        old = before[m.id]
        new = m.confidence.value
        delta = new - old
        if delta != 0:
            print(f"  [{m.id}] {old:.3f} → {new:.3f} ({delta:+.3f})")
        else:
            print(f"  [{m.id}] {old:.3f} (unchanged)")

    # Demonstrate distrust propagation
    print()
    print("-" * 40)
    print("Demonstrating distrust propagation...")

    low_conf_memory = make_memory(
        "m_low",
        "Contradicted information",
        confidence=0.25,
        related_ids=["m_target"],
    )
    target_memory = make_memory(
        "m_target",
        "Target memory",
        confidence=0.8,
        related_ids=["m_low"],
    )

    print(f"Source (low confidence): {low_conf_memory.confidence.value:.2f}")
    print(f"Target before: {target_memory.confidence.value:.2f}")

    penalized = await propagate_distrust(low_conf_memory, [target_memory])

    print(f"Target after: {target_memory.confidence.value:.2f}")
    print(f"Memories penalized: {penalized}")


async def main() -> None:
    """Run all Phase 7 demos."""
    print()
    print("╔" + "═" * 58 + "╗")
    print("║" + " PHASE 7: ADVANCED MEMORY FEATURES DEMO ".center(58) + "║")
    print("╚" + "═" * 58 + "╝")

    await demo_temporal_reasoning()
    await demo_entity_resolution()
    await demo_confidence_propagation()

    print()
    print_section("DEMO COMPLETE")
    print("Phase 7 features demonstrated:")
    print("  ✓ Temporal Reasoning - State change detection")
    print("  ✓ Entity Resolution - Extraction and alias management")
    print("  ✓ Confidence Propagation - PageRank-style spreading")
    print()


if __name__ == "__main__":
    asyncio.run(main())
