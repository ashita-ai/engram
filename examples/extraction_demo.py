#!/usr/bin/env python3
"""Extraction pipeline demonstration.

This example shows all 8 extractors in action:
1. EmailExtractor - RFC-compliant email addresses
2. PhoneExtractor - International phone numbers
3. URLExtractor - Web URLs
4. DateExtractor - Natural language dates
5. QuantityExtractor - Physical quantities (5km, 72°F)
6. LanguageExtractor - Language detection
7. NameExtractor - Human names
8. IDExtractor - ISBN, IBAN, SSN, credit cards (masked)

Usage:
    python examples/extraction_demo.py
"""

from engram.extraction import (
    DateExtractor,
    EmailExtractor,
    IDExtractor,
    LanguageExtractor,
    NameExtractor,
    PhoneExtractor,
    QuantityExtractor,
    URLExtractor,
    default_pipeline,
)
from engram.models import Episode


def demonstrate_extractor(
    name: str,
    extractor: EmailExtractor
    | PhoneExtractor
    | URLExtractor
    | DateExtractor
    | QuantityExtractor
    | LanguageExtractor
    | NameExtractor
    | IDExtractor,
    text: str,
) -> None:
    """Run a single extractor and show results."""
    episode = Episode(content=text, role="user", user_id="demo")
    facts = extractor.extract(episode)

    print(f"\n{'─' * 60}")
    print(f"📌 {name}")
    print(f'   Input: "{text}"')
    if facts:
        for fact in facts:
            print(f"   ✓ {fact.category}: {fact.content}")
    else:
        print("   (no matches)")


def main() -> None:
    """Run the extraction demo."""
    print("=" * 60)
    print("Engram Extraction Pipeline Demo")
    print("=" * 60)
    print("\nAll 8 extractors use battle-tested libraries:")
    print("  • email-validator, phonenumbers, validators")
    print("  • dateparser, Pint, langdetect")
    print("  • nameparser, python-stdnum")

    # =========================================================================
    # Individual Extractor Demos
    # =========================================================================

    demonstrate_extractor(
        "EmailExtractor (email-validator)",
        EmailExtractor(),
        "Contact me at alice@example.com or support@company.org",
    )

    demonstrate_extractor(
        "PhoneExtractor (phonenumbers)",
        PhoneExtractor(),
        "Call +1 (555) 123-4567 or reach our UK office at +44 20 7946 0958",
    )

    demonstrate_extractor(
        "URLExtractor (validators)",
        URLExtractor(),
        "Check out https://github.com/ashita-ai/engram and http://localhost:8000/docs",
    )

    demonstrate_extractor(
        "DateExtractor (dateparser)",
        DateExtractor(),
        "The meeting is tomorrow at 3pm. Project deadline: January 15, 2025.",
    )

    demonstrate_extractor(
        "QuantityExtractor (Pint)",
        QuantityExtractor(),
        "The package weighs 2.5 kg and measures 30 cm. Temperature is 72°F.",
    )

    demonstrate_extractor(
        "LanguageExtractor (langdetect)",
        LanguageExtractor(),
        "Bonjour, comment allez-vous aujourd'hui? J'espère que tout va bien.",
    )

    demonstrate_extractor(
        "NameExtractor (nameparser)",
        NameExtractor(),
        "Please contact Dr. Jane Smith or Mr. Robert Johnson III for details.",
    )

    demonstrate_extractor(
        "IDExtractor (python-stdnum)",
        IDExtractor(),
        "ISBN: 978-0-13-468599-1, Credit Card: 4532015112830366, SSN: 123-45-6789",
    )

    # =========================================================================
    # Full Pipeline Demo
    # =========================================================================

    print(f"\n{'=' * 60}")
    print("Full Pipeline Demo")
    print("=" * 60)

    pipeline = default_pipeline()
    print(f"\nPipeline has {len(pipeline.extractors)} extractors")

    complex_text = """
    Hi, I'm Dr. Alice Johnson. You can reach me at alice.johnson@techcorp.com
    or call +1-555-867-5309. Our meeting is scheduled for next Tuesday at 2pm.

    The prototype costs $15,000 and weighs about 3.5 kg. It should be ready
    by March 15, 2025. For payment, my card is 4532015112830366.

    More info at https://techcorp.com/prototype - the specs are in the PDF.

    Merci beaucoup pour votre aide!
    """

    episode = Episode(content=complex_text, role="user", user_id="demo")
    facts = pipeline.run(episode)

    print(f"\nInput text ({len(complex_text)} chars):")
    print(f'  "{complex_text[:100]}..."')
    print(f"\nExtracted {len(facts)} facts:")

    # Group by category for nice output
    by_category: dict[str, list[str]] = {}
    for fact in facts:
        by_category.setdefault(fact.category, []).append(fact.content)

    for category, contents in sorted(by_category.items()):
        print(f"\n  {category}:")
        for content in contents:
            print(f"    • {content}")

    print(f"\n{'=' * 60}")
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
