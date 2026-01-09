#!/usr/bin/env python3
"""Confidence scoring system demo.

Demonstrates Engram's composite confidence scoring:
- Three extraction methods with different base scores
- Weighted formula: method (50%) + corroboration (25%) + recency (15%) + verification (10%)
- Human-readable explanations for every score

No external dependencies required - runs entirely locally.
"""

from datetime import UTC, datetime, timedelta

from engram.models import ConfidenceScore, ExtractionMethod


def main() -> None:
    print("=" * 70)
    print("Engram Confidence Scoring Demo")
    print("=" * 70)

    # =========================================================================
    # Part 1: Extraction Methods
    # =========================================================================
    print("\n1. EXTRACTION METHODS")
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
    # Part 2: Factory Methods
    # =========================================================================
    print("\n2. CREATING CONFIDENCE SCORES")
    print("-" * 70)

    # Verbatim (highest confidence)
    verbatim = ConfidenceScore.for_verbatim()
    print("  ConfidenceScore.for_verbatim()")
    print(f"    Value: {verbatim.value:.0%}")
    print(f"    Explain: {verbatim.explain()}")

    # Extracted (high confidence)
    extracted = ConfidenceScore.for_extracted()
    print("\n  ConfidenceScore.for_extracted()")
    print(f"    Value: {extracted.value:.0%}")
    print(f"    Explain: {extracted.explain()}")

    # Inferred (moderate confidence)
    inferred = ConfidenceScore.for_inferred()
    print("\n  ConfidenceScore.for_inferred()")
    print(f"    Value: {inferred.value:.0%}")
    print(f"    Explain: {inferred.explain()}")

    # =========================================================================
    # Part 3: Corroboration (Multiple Sources)
    # =========================================================================
    print("\n\n3. CORROBORATION (MULTIPLE SOURCES)")
    print("-" * 70)
    print("  Confidence increases when multiple sources confirm a fact:\n")

    for num_sources in [1, 2, 3, 5, 10]:
        score = ConfidenceScore.for_extracted(supporting_episodes=num_sources)
        # Recompute to apply corroboration weight
        score.recompute()
        print(f"  {num_sources} source(s): {score.value:.0%}")
        print(f"    {score.explain()}")
        print()

    # =========================================================================
    # Part 4: Recency Decay
    # =========================================================================
    print("\n4. RECENCY DECAY")
    print("-" * 70)
    print("  Confidence decreases as memories age:\n")

    now = datetime.now(UTC)
    ages = [
        ("Just now", now),
        ("1 day ago", now - timedelta(days=1)),
        ("1 week ago", now - timedelta(weeks=1)),
        ("1 month ago", now - timedelta(days=30)),
        ("6 months ago", now - timedelta(days=180)),
        ("1 year ago", now - timedelta(days=365)),
    ]

    for label, last_confirmed in ages:
        score = ConfidenceScore(
            value=0.9,
            extraction_method=ExtractionMethod.EXTRACTED,
            extraction_base=0.9,
            last_confirmed=last_confirmed,
        )
        # Recompute to apply recency decay
        score.recompute()
        print(f"  {label}: {score.value:.0%}")

    # =========================================================================
    # Part 5: Combined Factors
    # =========================================================================
    print("\n\n5. COMBINED CONFIDENCE FACTORS")
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

    # Medium confidence: inferred, single source
    medium = ConfidenceScore(
        value=0.6,
        extraction_method=ExtractionMethod.INFERRED,
        extraction_base=0.6,
        supporting_episodes=1,
        last_confirmed=datetime.now(UTC) - timedelta(days=7),
    )
    medium.recompute()
    print(f"\n  MEDIUM CONFIDENCE: {medium.value:.0%}")
    print("    LLM-inferred preference, 1 source, confirmed last week")
    print(f"    {medium.explain()}")

    # Low confidence: inferred, old, contradiction
    low = ConfidenceScore(
        value=0.6,
        extraction_method=ExtractionMethod.INFERRED,
        extraction_base=0.6,
        supporting_episodes=1,
        last_confirmed=datetime.now(UTC) - timedelta(days=180),
        contradictions=1,
    )
    low.recompute()
    print(f"\n  LOW CONFIDENCE: {low.value:.0%}")
    print("    LLM-inferred, 1 source, 6 months old, 1 contradiction")
    print(f"    {low.explain()}")

    # =========================================================================
    # Part 6: Confidence Formula
    # =========================================================================
    print("\n\n6. CONFIDENCE FORMULA")
    print("-" * 70)
    print("""
  Confidence is a weighted composite score:

  | Factor            | Weight | Description                          |
  |-------------------|--------|--------------------------------------|
  | Extraction method | 50%    | verbatim=1.0, extracted=0.9, inferred=0.6 |
  | Corroboration     | 25%    | More sources = higher confidence     |
  | Recency           | 15%    | Decays over time without confirmation |
  | Verification      | 10%    | Format/validity checks passed        |

  Formula:
    confidence = 0.5 * method_score
               + 0.25 * corroboration_score
               + 0.15 * recency_score
               + 0.10 * verification_score

  Additional:
    - Contradictions apply a 10% penalty per contradiction
    - Corroboration uses log scale (10 sources = max boost)
    - Recency uses exponential decay (365-day half-life)

  Every score is AUDITABLE - the system can explain WHY confidence is 0.73.
    """)

    # =========================================================================
    # Part 7: Using Confidence in Queries
    # =========================================================================
    print("\n7. USING CONFIDENCE IN QUERIES")
    print("-" * 70)
    print("""
  Applications filter by confidence based on use case:

  ```python
  # High-stakes: only verified facts
  trusted = await engram.recall(query, user_id="u1", min_confidence=0.9)

  # Exploratory: include inferences
  all_relevant = await engram.recall(query, user_id="u1", min_confidence=0.5)

  # Debug: see everything
  everything = await engram.recall(query, user_id="u1", min_confidence=0.0)
  ```
    """)

    print("=" * 70)
    print("Demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
