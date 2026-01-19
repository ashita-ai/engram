#!/usr/bin/env python3
"""Confidence scoring system demo.

Demonstrates Engram's intelligent confidence architecture:

- Episodic:    1.0 always (verbatim, immutable ground truth)
- Structured:  LLM assesses confidence during extraction (single call)
- Semantic:    LLM assesses confidence during synthesis (single call)
- Procedural:  Bayesian updating with accumulating evidence

No external dependencies required - runs entirely locally.
"""

from datetime import UTC, datetime, timedelta

from engram.confidence import BayesianConfidence, bayesian_update, combine_bayesian_confidences
from engram.models import ConfidenceScore, ExtractionMethod


def main() -> None:
    print("=" * 70)
    print("Engram Intelligent Confidence Scoring Demo")
    print("=" * 70)

    # =========================================================================
    # Part 1: Architecture Overview
    # =========================================================================
    print("\n1. CONFIDENCE ARCHITECTURE BY MEMORY TYPE")
    print("-" * 70)
    print("""
  ┌────────────────┬──────────────────────────────────────────────────┐
  │ Memory Type    │ Confidence Method                                │
  ├────────────────┼──────────────────────────────────────────────────┤
  │ Episodic       │ 1.0 always (verbatim, immutable ground truth)    │
  │ Structured     │ LLM assesses during extraction (single call)     │
  │ Semantic       │ LLM assesses during synthesis (single call)      │
  │ Procedural     │ Bayesian updating with accumulating evidence     │
  └────────────────┴──────────────────────────────────────────────────┘

  Why this architecture?
  - Episodic memories are raw, unprocessed - they ARE the ground truth
  - Structured/Semantic use LLMs, so LLMs should assess their own certainty
  - Procedural memories accumulate evidence over time - perfect for Bayesian
    """)

    # =========================================================================
    # Part 2: Extraction Methods
    # =========================================================================
    print("\n2. EXTRACTION METHODS (Base Scores)")
    print("-" * 70)
    print("  Each method has a different base confidence score:\n")

    # Create scores using factory methods
    verbatim = ConfidenceScore.for_verbatim()
    extracted = ConfidenceScore.for_extracted()
    inferred = ConfidenceScore.for_inferred()

    methods = [
        ("VERBATIM", verbatim, "Exact quote, immutable ground truth"),
        ("EXTRACTED", extracted, "Pattern-matched (regex, deterministic)"),
        ("INFERRED", inferred, "LLM-derived, uncertain"),
    ]

    for name, score, desc in methods:
        print(f"  {name}: {score.value:.0%} base")
        print(f"    {desc}")
        print(f"    Method: {score.extraction_method.value}")
        print()

    # =========================================================================
    # Part 3: LLM-Assessed Confidence
    # =========================================================================
    print("\n3. LLM-ASSESSED CONFIDENCE (Structured/Semantic)")
    print("-" * 70)
    print("""
  The LLM returns confidence ALONGSIDE the extraction in a single call.
  This is efficient (no extra API call) and contextually aware.

  Example extraction output from LLM:
  """)

    # Simulate LLM-assessed confidence for different scenarios
    scenarios = [
        (
            0.95,
            "User explicitly stated 'I work at Google'",
            "Direct statement, no hedging",
        ),
        (
            0.75,
            "User prefers Python",
            "Implied from context, minor hedging",
        ),
        (
            0.55,
            "User might be interested in machine learning",
            "Speculative inference from conversation topic",
        ),
        (
            0.30,
            "User dislikes JavaScript",
            "Weak evidence, single offhand comment",
        ),
    ]

    for confidence, content, reasoning in scenarios:
        score = ConfidenceScore.for_inferred(
            confidence=confidence,
            reasoning=reasoning,
        )
        print(f"  Content: {content}")
        print(f"  Confidence: {score.value:.0%}")
        print(f"  LLM Reasoning: {reasoning}")
        print(f"  Explain: {score.explain()}")
        print()

    # =========================================================================
    # Part 4: Bayesian Confidence (Procedural)
    # =========================================================================
    print("\n4. BAYESIAN CONFIDENCE (Procedural Memories)")
    print("-" * 70)
    print("""
  Procedural memories track behavioral patterns observed over time.
  We use a Beta-Bernoulli model for Bayesian updating:

  - Start with a prior belief (e.g., "weak" = 50/50)
  - Update with each observation (confirm or contradict)
  - Confidence converges as evidence accumulates
    """)

    # Demonstrate Bayesian updating
    print("  Example: Tracking if user prefers morning meetings\n")

    bc = BayesianConfidence.from_prior("weak")
    print(f"  Initial (weak prior): {bc.confidence:.0%}")
    print(f"    Strength: {bc.strength}")

    observations = [True, True, False, True, True, True, False, True, True, True]
    for i, observed in enumerate(observations, 1):
        bc.update(observed=observed)
        print(
            f"  After observation {i} ({'confirmed' if observed else 'contradicted'}): "
            f"{bc.confidence:.0%} ({bc.strength})"
        )

    print(f"\n  Final: {bc.explain()}")
    low, high = bc.credible_interval_95
    print(f"  95% Credible Interval: [{low:.0%}, {high:.0%}]")

    # =========================================================================
    # Part 5: Priors and Batch Updates
    # =========================================================================
    print("\n\n5. BAYESIAN PRIORS AND BATCH UPDATES")
    print("-" * 70)

    priors = ["uninformative", "weak", "optimistic", "pessimistic"]
    print("  Different prior beliefs:\n")
    for prior in priors:
        bc = BayesianConfidence.from_prior(prior)
        print(f"  {prior:15}: {bc.confidence:.0%} initial confidence")

    print("\n  Batch update (5 confirmations, 2 contradictions):\n")
    bc = BayesianConfidence.from_prior("weak")
    print(f"  Before: {bc.confidence:.0%}")
    bc.update_batch(confirmations=5, contradictions=2)
    print(f"  After:  {bc.confidence:.0%}")
    print(f"  {bc.explain()}")

    # =========================================================================
    # Part 6: Combining Bayesian Confidences
    # =========================================================================
    print("\n\n6. COMBINING BAYESIAN CONFIDENCES")
    print("-" * 70)
    print("""
  When consolidating multiple procedural observations, we combine
  their Bayesian confidences by pooling evidence:
    """)

    bc1 = BayesianConfidence.from_observations(5, 1)
    bc2 = BayesianConfidence.from_observations(3, 2)

    print(f"  Source 1: {bc1.explain()}")
    print(f"  Source 2: {bc2.explain()}")

    combined = combine_bayesian_confidences([bc1, bc2])
    print(f"\n  Combined: {combined.explain()}")

    # =========================================================================
    # Part 7: Decay Over Time
    # =========================================================================
    print("\n\n7. BAYESIAN DECAY OVER TIME")
    print("-" * 70)
    print("""
  Without new observations, confidence becomes more uncertain:
    """)

    bc = BayesianConfidence.from_observations(10, 2)
    print(f"  Initial: {bc.confidence:.0%} (variance: {bc.variance:.4f})")

    for i in range(3):
        bc.decay(factor=0.8)
        print(f"  After decay {i + 1}: {bc.confidence:.0%} (variance: {bc.variance:.4f})")

    # =========================================================================
    # Part 8: Composite Confidence with Corroboration
    # =========================================================================
    print("\n\n8. COMPOSITE CONFIDENCE FACTORS")
    print("-" * 70)
    print("  Real-world examples combining all factors:\n")

    # High confidence: extracted, multiple sources, verified
    high = ConfidenceScore(
        value=0.9,
        extraction_method=ExtractionMethod.EXTRACTED,
        extraction_base=0.9,
        supporting_episodes=3,
        last_confirmed=datetime.now(UTC) - timedelta(days=1),
        verified=True,
    )
    high.recompute()
    print(f"  HIGH CONFIDENCE: {high.value:.0%}")
    print("    Extracted email, 3 sources, confirmed yesterday, format verified")
    print(f"    {high.explain()}")

    # Medium confidence: inferred with LLM reasoning
    medium = ConfidenceScore.for_inferred(
        confidence=0.72,
        supporting_episodes=2,
        reasoning="User mentioned twice with consistent context",
    )
    medium.recompute()
    print(f"\n  MEDIUM CONFIDENCE: {medium.value:.0%}")
    print("    LLM-inferred preference, 2 sources, with reasoning")
    print(f"    {medium.explain()}")

    # Low confidence: inferred, old, contradiction
    low_score = ConfidenceScore(
        value=0.6,
        extraction_method=ExtractionMethod.INFERRED,
        extraction_base=0.6,
        supporting_episodes=1,
        last_confirmed=datetime.now(UTC) - timedelta(days=180),
        contradictions=1,
    )
    low_score.recompute()
    print(f"\n  LOW CONFIDENCE: {low_score.value:.0%}")
    print("    LLM-inferred, 1 source, 6 months old, 1 contradiction")
    print(f"    {low_score.explain()}")

    # =========================================================================
    # Part 9: Convenience Function
    # =========================================================================
    print("\n\n9. QUICK BAYESIAN UPDATES")
    print("-" * 70)
    print("  For one-off updates without managing state:\n")

    prior = 0.5
    observations = [True, True, True, False]
    result = bayesian_update(prior, observations)
    print(f"  Prior: {prior:.0%}")
    print(f"  Observations: {observations}")
    print(f"  Posterior: {result:.0%}")

    # =========================================================================
    # Part 10: Summary
    # =========================================================================
    print("\n\n10. ARCHITECTURE SUMMARY")
    print("-" * 70)
    print("""
  ┌─────────────────────────────────────────────────────────────────┐
  │                    Confidence by Memory Type                    │
  ├─────────────────────────────────────────────────────────────────┤
  │ EPISODIC                                                        │
  │   Confidence: 1.0 (always)                                      │
  │   Why: Raw, verbatim ground truth. Immutable.                   │
  ├─────────────────────────────────────────────────────────────────┤
  │ STRUCTURED (per-episode LLM extraction)                         │
  │   Confidence: LLM-assessed (0.0-1.0)                            │
  │   Why: LLM extracts entities AND assesses confidence together   │
  │   Performance: Single LLM call for both                         │
  ├─────────────────────────────────────────────────────────────────┤
  │ SEMANTIC (cross-episode LLM synthesis)                          │
  │   Confidence: LLM-assessed (0.0-1.0)                            │
  │   Why: LLM synthesizes AND assesses source agreement            │
  │   Performance: Single LLM call for both                         │
  ├─────────────────────────────────────────────────────────────────┤
  │ PROCEDURAL (behavioral patterns)                                │
  │   Confidence: Bayesian updating (Beta-Bernoulli)                │
  │   Why: Patterns accumulate evidence over time                   │
  │   Features: Priors, batch updates, decay, credible intervals    │
  └─────────────────────────────────────────────────────────────────┘

  All confidence scores are AUDITABLE via .explain()
    """)

    print("=" * 70)
    print("Demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
