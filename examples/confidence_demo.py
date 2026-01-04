#!/usr/bin/env python3
"""Confidence scoring demonstration.

This example shows how Engram tracks confidence in memories:

1. VERBATIM (1.0) - Exact quotes, never modified
2. EXTRACTED (0.9) - Deterministic pattern extraction
3. INFERRED (0.6) - LLM-derived, uncertain

Confidence is calculated as:
    confidence = (
        extraction_base × 0.50 +
        corroboration × 0.25 +
        recency × 0.15 +
        verification × 0.10
    )

Usage:
    python examples/confidence_demo.py
"""

from engram.models import Episode, Fact, SemanticMemory
from engram.models.base import ConfidenceScore, ExtractionMethod


def main() -> None:
    """Run the confidence demo."""
    print("=" * 60)
    print("Engram Confidence Scoring Demo")
    print("=" * 60)

    # =========================================================================
    # Extraction Methods
    # =========================================================================
    print("\n📊 Extraction Methods and Base Confidence:")

    methods = [
        (ExtractionMethod.VERBATIM, 1.0, "Exact quote - immutable ground truth"),
        (ExtractionMethod.EXTRACTED, 0.9, "Pattern extraction (regex, validators)"),
        (ExtractionMethod.INFERRED, 0.6, "LLM inference - uncertain"),
    ]

    for method, base, description in methods:
        print(f"\n  {method.value.upper():10} → {base:.0%} base confidence")
        print(f"  {' ' * 13} {description}")

    # =========================================================================
    # Creating Confidence Scores
    # =========================================================================
    print(f"\n{'─' * 60}")
    print("Creating Confidence Scores")
    print("─" * 60)

    # Verbatim confidence (episodes)
    verbatim = ConfidenceScore.for_verbatim()
    print("\n  Verbatim (for episodes):")
    print(f"    Value: {verbatim.value:.0%}")
    print(f"    Method: {verbatim.extraction_method.value}")

    # Extracted confidence (facts from extractors)
    extracted = ConfidenceScore.for_extracted()
    print("\n  Extracted (for facts):")
    print(f"    Value: {extracted.value:.0%}")
    print(f"    Method: {extracted.extraction_method.value}")

    # Inferred confidence (semantic memories from LLM)
    inferred = ConfidenceScore.for_inferred()
    print("\n  Inferred (for semantic memories):")
    print(f"    Value: {inferred.value:.0%}")
    print(f"    Method: {inferred.extraction_method.value}")

    # =========================================================================
    # Memory Types with Confidence
    # =========================================================================
    print(f"\n{'─' * 60}")
    print("Memory Types with Confidence")
    print("─" * 60)

    # Episode - ground truth, no confidence needed
    episode = Episode(
        content="My email is alice@example.com",
        role="user",
        user_id="demo",
    )
    print("\n  Episode (Ground Truth):")
    print(f'    Content: "{episode.content}"')
    print("    Note: Episodes don't have confidence - they ARE the truth")

    # Fact - extracted with high confidence
    fact = Fact(
        content="alice@example.com",
        category="email",
        source_episode_id=episode.id,
        user_id="demo",
        confidence=ConfidenceScore.for_extracted(),
    )
    print("\n  Fact (Extracted):")
    print(f'    Content: "{fact.content}"')
    print(f"    Category: {fact.category}")
    print(f"    Confidence: {fact.confidence.value:.0%}")
    print(f"    Source: {fact.source_episode_id}")

    # Semantic memory - inferred with lower confidence
    semantic = SemanticMemory(
        content="Alice uses example.com for personal email",
        source_episode_ids=[episode.id],
        user_id="demo",
        confidence=ConfidenceScore.for_inferred(),
    )
    print("\n  SemanticMemory (Inferred):")
    print(f'    Content: "{semantic.content}"')
    print(f"    Confidence: {semantic.confidence.value:.0%}")
    print("    ⚠️  Lower confidence - LLM may have made assumptions")

    # =========================================================================
    # Confidence Weights
    # =========================================================================
    print(f"\n{'─' * 60}")
    print("Confidence Formula")
    print("─" * 60)

    print("\n  confidence = weighted sum of:")
    print("    • Extraction method: 50%  (verbatim/extracted/inferred)")
    print("    • Corroboration:     25%  (supporting sources)")
    print("    • Recency:           15%  (how recently confirmed)")
    print("    • Verification:      10%  (format validation)")

    print("\n  Example calculation for extracted email:")
    print("    extraction_base = 0.9 (extracted)")
    print("    corroboration   = 1.0 (first occurrence)")
    print("    recency         = 1.0 (just extracted)")
    print("    verification    = 1.0 (valid email format)")
    print("    ─────────────────────────")
    confidence = 0.9 * 0.5 + 1.0 * 0.25 + 1.0 * 0.15 + 1.0 * 0.10
    print(f"    final confidence = {confidence:.0%}")

    # =========================================================================
    # Decay Over Time
    # =========================================================================
    print(f"\n{'─' * 60}")
    print("Confidence Decay")
    print("─" * 60)

    print("\n  Memories decay without confirmation:")
    print("    • Half-life: 365 days (configurable)")
    print("    • Archive threshold: 10% confidence")
    print("    • Delete threshold: 1% confidence")
    print("\n  Regular confirmation resets recency score!")

    print(f"\n{'=' * 60}")
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
