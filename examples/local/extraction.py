#!/usr/bin/env python3
"""Pattern extraction and negation detection demo.

Demonstrates Engram's deterministic extraction pipeline:
- 8 pattern extractors (email, phone, URL, date, quantity, language, name, ID)
- Negation detection ("I don't use X", "not interested in Y")

No external dependencies required - runs entirely locally.
"""

from engram.extraction import (
    DateExtractor,
    EmailExtractor,
    IDExtractor,
    LanguageExtractor,
    NameExtractor,
    NegationDetector,
    PhoneExtractor,
    QuantityExtractor,
    URLExtractor,
)
from engram.models import Episode


def main() -> None:
    print("=" * 70)
    print("Engram Pattern Extraction Demo")
    print("=" * 70)

    # =========================================================================
    # Part 1: Individual Extractors
    # =========================================================================
    print("\n1. PATTERN EXTRACTORS")
    print("-" * 70)

    extractors = [
        ("Email", EmailExtractor(), "Contact me at alice@example.com or bob@corp.io"),
        ("Phone", PhoneExtractor(), "Call me at (415) 555-1234 or +1-800-555-0199"),
        ("URL", URLExtractor(), "Check out https://github.com/ashita-ai/engram"),
        ("Date", DateExtractor(), "Meeting on 2024-03-15, deadline is March 20th"),
        ("Quantity", QuantityExtractor(), "The package weighs 5 kg and measures 30 cm"),
        # LanguageExtractor detects SPOKEN language (e.g., "en"), not programming languages
        ("Language (spoken)", LanguageExtractor(), "Bonjour, je parle français"),
        ("Name", NameExtractor(), "I'm Sarah Chen, working with Dr. James Smith"),
        ("ID", IDExtractor(), "ISBN: 978-0-13-468599-1 for the reference book"),
    ]

    for name, extractor, text in extractors:
        print(f"\n  {name}Extractor")
        print(f'  Input: "{text}"')
        episode = Episode(content=text, role="user", user_id="demo", org_id="demo_org")
        facts = extractor.extract(episode)
        if facts:
            for fact in facts:
                print(f"  -> {fact.category}: {fact.content}")
        else:
            print("  -> (no matches)")

    # =========================================================================
    # Part 2: Full Pipeline
    # =========================================================================
    print("\n\n2. FULL EXTRACTION PIPELINE")
    print("-" * 70)

    # Use default_pipeline() which excludes LanguageExtractor
    # (spoken language codes like "en" aren't useful for memory recall)
    from engram.extraction import default_pipeline

    pipeline = default_pipeline()

    complex_message = """
    Hi, I'm Alex Johnson and my email is alex.j@techcorp.com.
    You can reach me at +1-415-555-6543. Our project deadline is 2024-06-01.
    The server has 64 GB of RAM and 500 GB storage. I mainly use Go and Rust.
    More info at https://techcorp.com/projects.
    """

    print(f"  Input: {complex_message.strip()}")
    episode = Episode(content=complex_message, role="user", user_id="demo", org_id="demo_org")
    all_facts = pipeline.run(episode)

    print(f"\n  Extracted {len(all_facts)} facts:")
    for fact in all_facts:
        print(f"    [{fact.category}] {fact.content} (confidence: {fact.confidence.value:.0%})")

    # =========================================================================
    # Part 3: Negation Detection
    # =========================================================================
    print("\n\n3. NEGATION DETECTION")
    print("-" * 70)
    print("  Negations capture what is NOT true - critical for accuracy.")

    detector = NegationDetector()

    negation_examples = [
        "I don't use MongoDB, we switched to PostgreSQL",
        "I'm not interested in blockchain or crypto",
        "We no longer support Python 2",
        "I never use Windows, only macOS and Linux",
        "I don't have a Twitter account",
    ]

    for text in negation_examples:
        print(f'\n  Input: "{text}"')
        episode = Episode(content=text, role="user", user_id="demo", org_id="demo_org")
        negations = detector.detect(episode)
        if negations:
            for neg in negations:
                print(f'  -> Negates: "{neg.negates_pattern}"')
                print(f'     Content: "{neg.content}"')
        else:
            print("  -> (no negations detected)")

    # =========================================================================
    # Part 4: Why This Matters
    # =========================================================================
    print("\n\n4. WHY PATTERN EXTRACTION MATTERS")
    print("-" * 70)
    print("""
  Engram extracts patterns BEFORE using LLMs:

  | Approach          | Confidence | Hallucination Risk |
  |-------------------|------------|-------------------|
  | Pattern extraction | 90%        | Zero              |
  | LLM extraction     | 60%        | Possible          |

  Pattern-matched facts are:
  - Deterministic (same input = same output)
  - Auditable (regex patterns are inspectable)
  - Fast (no API calls needed)
  - Trustworthy (no hallucination possible)

  LLM consolidation runs later in background for semantic inference.
    """)

    print("=" * 70)
    print("Demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
